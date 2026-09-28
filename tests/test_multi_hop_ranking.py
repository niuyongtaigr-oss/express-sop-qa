"""multi_hop 去重与排序 — 正确性回归

修复前有两个会**静默出错**的问题 (都不报错, 只是引用变少 / 顺序变差):
  1. 去重键用 `content[:50]` —— 法规条文前缀高度相似, 不同条款被判为重复丢弃
  2. `all_chunks` 是「第一轮全部 + 第二轮全部 + …」的拼接序, 直接截断会让第一轮的
     低分结果挤掉第二轮的高分结果, 恰好削掉多跳的收益
"""

from app.agents.nodes import _dedupe_and_rank, make_multi_hop_node

from test_graph_nodes import StubLLM, make_settings

# 法规文本的常态: 一条很长的公共前缀, 差异只在句尾/数字上。
# 前 60+ 字完全一致 —— 旧的 `content[:50]` 启发式必然把这两条判成同一条。
_LONG_PREFIX = (
    "第二十八条 经营快递业务的企业应当遵守下列规定，违反本条规定的，"
    "由邮政管理部门责令改正，予以警告或者通报批评，可以并处罚款："
)

CLAUSE_A = {
    "content": _LONG_PREFIX + "情节较轻的，处一万元以下的罚款。",
    "metadata": {"doc_id": "X", "chunk_index": 1, "title": "条例", "tags": "罚款"},
    "similarity": 0.6,
}
CLAUSE_B = {
    "content": _LONG_PREFIX + "情节严重的，处一万元以上三万元以下的罚款。",
    "metadata": {"doc_id": "X", "chunk_index": 9, "title": "条例", "tags": "赔偿"},
    "similarity": 0.8,
}
OTHER_DOC = {
    "content": _LONG_PREFIX + "本条同样适用于经营快递业务的国际货物运输代理企业。",
    "metadata": {"doc_id": "Y", "chunk_index": 9, "title": "办法", "tags": "通用"},
    "similarity": 0.7,
}


# ── 去重键 ───────────────────────────────────────────────

def test_fixture_really_defeats_prefix_heuristic():
    """先证明这组夹具确实能触发旧 bug: 前 50 字一模一样"""
    assert CLAUSE_A["content"][:50] == CLAUSE_B["content"][:50]
    assert CLAUSE_A["content"] != CLAUSE_B["content"]


def test_prefix_identical_clauses_are_both_kept():
    """前 50 字相同但 chunk_index 不同 → 必须都保留 (修复前会丢一条)"""
    ranked = _dedupe_and_rank([CLAUSE_A, CLAUSE_B])
    assert len(ranked) == 2
    assert {c["metadata"]["chunk_index"] for c in ranked} == {1, 9}


def test_same_index_different_doc_are_distinct():
    """chunk_index 相同但 doc_id 不同 → 是两条不同条款 (键必须是二元组)"""
    assert len(_dedupe_and_rank([CLAUSE_B, OTHER_DOC])) == 2


def test_true_duplicate_is_merged_keeping_highest_similarity():
    """真正同一条 chunk 只保留一份, 且保留相似度更高的那次命中"""
    low = {**CLAUSE_B, "similarity": 0.3}
    high = {**CLAUSE_B, "similarity": 0.95}
    ranked = _dedupe_and_rank([low, high])
    assert len(ranked) == 1
    assert ranked[0]["similarity"] == 0.95


def test_missing_chunk_index_falls_back_to_content():
    """元数据缺 chunk_index 时不能崩, 也不能误合并不同内容"""
    a = {"content": "甲", "metadata": {"doc_id": "Z"}, "similarity": 0.5}
    b = {"content": "乙", "metadata": {"doc_id": "Z"}, "similarity": 0.5}
    assert len(_dedupe_and_rank([a, b])) == 2


# ── 排序 ─────────────────────────────────────────────────

def test_rank_is_by_similarity_not_insertion_order():
    """按相似度降序, 而不是按「哪一轮先检索到」"""
    ranked = _dedupe_and_rank([CLAUSE_A, OTHER_DOC, CLAUSE_B])  # 0.6, 0.7, 0.8
    assert [c["similarity"] for c in ranked] == [0.8, 0.7, 0.6]


def test_rank_is_stable_on_ties():
    """同分保持原序 —— 否则重排会在没有新信息时凭空打乱顺序"""
    a = {**CLAUSE_A, "similarity": 0.5}
    b = {**CLAUSE_B, "similarity": 0.5}
    ranked = _dedupe_and_rank([a, b])
    assert [c["metadata"]["chunk_index"] for c in ranked] == [1, 9]


# ── 端到端: 第二轮的高分结果不能被第一轮的低分结果挤掉 ────

class RoundRag:
    """按轮次返回不同 chunk 的桩 RAG"""

    def __init__(self, rounds: list[list[dict]]):
        self._rounds = rounds
        self._i = 0
        self.generated_with: list[dict] | None = None

    def retrieve(self, query, top_k=None, tenant_id="default"):
        idx = min(self._i, len(self._rounds) - 1)
        self._i += 1
        return [dict(c) for c in self._rounds[idx]]

    def generate(self, query, chunks, history=None):
        self.generated_with = chunks
        return "multi answer"


def test_second_round_high_score_survives_truncation():
    """第一轮 3 条低分 + 第二轮 1 条高分, top_k=1 → 高分那条必须进入生成上下文"""
    low = [
        {**CLAUSE_A, "similarity": 0.2,
         "metadata": {**CLAUSE_A["metadata"], "chunk_index": i}}
        for i in range(3)
    ]
    high = [{**OTHER_DOC, "similarity": 0.95}]
    rag = RoundRag([low, high])
    llm = StubLLM(intent="multi_hop")

    node = make_multi_hop_node(
        rag, llm,
        make_settings(top_k=1, multi_hop_max_rounds=2,
                      multi_hop_similarity_threshold=0.6),
    )
    result = node({"question": "罚款多少", "history": None, "tenant_id": "t"})

    assert rag.generated_with is not None
    assert rag.generated_with[0]["similarity"] == 0.95   # 拼接序会把它排到最后
    assert result["sources"][0]["similarity"] == 0.95    # sources 同样按相关度排序
    assert result["rounds"] == 2                         # 确认确实跑了两轮


def test_sources_dedupe_keeps_distinct_clauses():
    """跨轮重复召回同一条时合并, 但不同条款必须都留在 sources 里"""
    round1 = [CLAUSE_A, OTHER_DOC]
    round2 = [{**CLAUSE_A, "similarity": 0.9}]   # 第一轮的 A 在第二轮被更高分命中
    rag = RoundRag([round1, round2])
    llm = StubLLM(intent="multi_hop")

    node = make_multi_hop_node(
        rag, llm,
        make_settings(multi_hop_max_rounds=2, multi_hop_similarity_threshold=0.6),
    )
    result = node({"question": "q", "history": None, "tenant_id": "t"})

    assert len(result["sources"]) == 2           # A 合并成一条, 不是三条
    assert {s["doc_id"] for s in result["sources"]} == {"X", "Y"}
    a = next(s for s in result["sources"] if s["doc_id"] == "X")
    assert a["similarity"] == 0.9                # 留的是高分那次


# ── 命中度阈值: 只有向量余弦能与阈值比较 ──────────────────
# BM25 归一化分是"排名映射": 该模式下的最高分恒为 0.9, 与真实相关度无关。
# 拿它和阈值比永远通过 → 多跳的"信息够了才停"退化成"永远停"。

from app.agents.nodes import _hit_is_confident, _rank_key  # noqa: E402


def _bm25_chunk(sim=0.9):
    """纯 BM25 / BM25 独有命中: similarity 是归一化排名映射"""
    return {**CLAUSE_A, "similarity": sim, "score_kind": "bm25_norm"}


def test_bm25_top_score_never_counts_as_confident():
    """BM25 模式下的最高分恒为 0.9, 不能据此认定"信息够了" """
    settings = make_settings(multi_hop_similarity_threshold=0.6)
    assert _hit_is_confident(_bm25_chunk(0.9), settings) is False


def test_cosine_above_threshold_is_confident():
    settings = make_settings(multi_hop_similarity_threshold=0.6)
    assert _hit_is_confident({**CLAUSE_A, "similarity": 0.81,
                             "score_kind": "cosine"}, settings) is True
    assert _hit_is_confident({**CLAUSE_A, "similarity": 0.42,
                             "score_kind": "cosine"}, settings) is False


def test_bm25_only_top_hit_does_not_stop_multi_hop():
    """端到端: 首轮只命中 BM25 分时, 必须继续跑第二轮, 不能提前停"""
    bm = [_bm25_chunk(0.9)]
    second = [{**OTHER_DOC, "similarity": 0.5, "score_kind": "cosine"}]
    rag = RoundRag([bm, second])
    llm = StubLLM(intent="multi_hop")

    node = make_multi_hop_node(
        rag, llm,
        make_settings(multi_hop_max_rounds=2, multi_hop_similarity_threshold=0.6),
    )
    result = node({"question": "q", "history": None, "tenant_id": "t"})
    assert result["rounds"] == 2          # 0.9 是排名假象, 不应据此停止


# ── 排序分: 重排开启时以 rerank_score 为准 ────────────────

def test_rank_key_prefers_rerank_score():
    """similarity 是重排**之前**的分; 排序若还用它, 重排等于没生效"""
    c = {**CLAUSE_A, "similarity": 0.9,
         "metadata": {**CLAUSE_A["metadata"], "rerank_score": 1.0}}
    assert _rank_key(c) == 1.0


def test_dedupe_orders_by_rerank_score():
    low_sim_high_rerank = {
        **CLAUSE_A, "similarity": 0.2,
        "metadata": {**CLAUSE_A["metadata"], "rerank_score": 9.0},
    }
    high_sim_low_rerank = {
        **CLAUSE_B, "similarity": 0.95,
        "metadata": {**CLAUSE_B["metadata"], "rerank_score": 1.0},
    }
    ranked = _dedupe_and_rank([high_sim_low_rerank, low_sim_high_rerank])
    assert [c["metadata"]["rerank_score"] for c in ranked] == [9.0, 1.0]


def test_rerank_score_zero_is_respected():
    """rerank_score=0 是有效分数 (最低), 不能被当成"没有分数"而退回 similarity"""
    c = {**CLAUSE_A, "similarity": 0.9,
         "metadata": {**CLAUSE_A["metadata"], "rerank_score": 0.0}}
    assert _rank_key(c) == 0.0


# ── 重排开启时只跑一轮 ───────────────────────────────────

def test_rerank_enabled_limits_multi_hop_to_one_round():
    """重排与多跳解决同一个问题, 不叠加 —— 否则是 N 次额外 LLM 调用
    且各轮 rerank_score 跨轮不可比"""
    rag = RoundRag([[{**CLAUSE_A, "similarity": 0.1}]])   # 低分也不会再跑第二轮
    llm = StubLLM(intent="multi_hop")

    node = make_multi_hop_node(
        rag, llm,
        make_settings(rerank_enabled=True, multi_hop_max_rounds=3,
                      multi_hop_similarity_threshold=0.6),
    )
    result = node({"question": "q", "history": None, "tenant_id": "t"})
    assert result["rounds"] == 1
    assert rag._i == 1          # 真的只检索了一次 (不是"轮次字段写 1")
