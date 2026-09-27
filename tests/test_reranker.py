"""重排层测试 (Stub LLM 驱动, 无需 Ollama)

重点覆盖**降级路径**: 重排是"锦上添花"的一步, 它失败时绝不能把检索一起拖垮 ——
线上表现应该是"精度退回粗排水平", 而不是整个查询 500。
"""

import pytest

from app.config import Settings
from app.infrastructure.reranker import (
    LLMReranker,
    NoopReranker,
    RerankResult,
    RerankScore,
    create_reranker,
)
from app.infrastructure.vector_store import RetrievedChunk


def make_chunk(text: str, similarity: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(content=text, metadata={"doc_id": "d"}, distance=0.0,
                          similarity=similarity)


class StubLLM:
    """structured_invoke 返回预设结果或抛错; 记录调用次数与消息内容"""

    def __init__(self, result=None, error: Exception | None = None):
        self._result = result
        self._error = error
        self.calls: list = []

    def structured_invoke(self, schema, messages):
        self.calls.append((schema, messages))
        if self._error is not None:
            raise self._error
        return self._result


def score(*pairs) -> RerankResult:
    return RerankResult(scores=[RerankScore(index=i, relevance=r) for i, r in pairs])


# ── NoopReranker ─────────────────────────────────────────

def test_noop_reranker_truncates_only():
    chunks = [make_chunk(f"c{i}") for i in range(5)]
    out = NoopReranker().rerank("q", chunks, top_k=2)
    assert [c.content for c in out] == ["c0", "c1"]


def test_noop_reranker_handles_fewer_than_top_k():
    out = NoopReranker().rerank("q", [make_chunk("only")], top_k=3)
    assert len(out) == 1


# ── LLMReranker: 正常路径 ────────────────────────────────

def test_reranker_reorders_by_score():
    chunks = [make_chunk("无关的"), make_chunk("部分相关"), make_chunk("正是答案")]
    llm = StubLLM(result=score((0, 0.0), (1, 0.5), (2, 1.0)))
    out = LLMReranker(llm).rerank("q", chunks, top_k=2)
    assert [c.content for c in out] == ["正是答案", "部分相关"]


def test_reranker_attaches_score_to_metadata():
    chunks = [make_chunk("a"), make_chunk("b")]
    llm = StubLLM(result=score((0, 0.1), (1, 0.9)))
    out = LLMReranker(llm).rerank("q", chunks, top_k=1)
    assert out[0].metadata["rerank_score"] == 0.9
    assert out[0].metadata["doc_id"] == "d"   # 原有元数据不丢


def test_reranker_is_stable_on_ties():
    """同分保持粗排原序 —— 否则重排在无信息时反而打乱已有排序"""
    chunks = [make_chunk("a"), make_chunk("b"), make_chunk("c")]
    llm = StubLLM(result=score((0, 0.5), (1, 0.5), (2, 0.5)))
    out = LLMReranker(llm).rerank("q", chunks, top_k=3)
    assert [c.content for c in out] == ["a", "b", "c"]


def test_reranker_unscored_candidates_sink():
    """模型漏打分时按 0 分沉底, 不能因为"没提到"就当作高分"""
    chunks = [make_chunk("a"), make_chunk("b")]
    llm = StubLLM(result=score((1, 0.9)))      # 只给了 b 的分数
    out = LLMReranker(llm).rerank("q", chunks, top_k=2)
    assert [c.content for c in out] == ["b", "a"]


def test_reranker_ignores_out_of_range_index():
    chunks = [make_chunk("a")]
    llm = StubLLM(result=score((0, 0.3), (99, 1.0)))   # 99 越界
    out = LLMReranker(llm).rerank("q", chunks, top_k=1)
    assert [c.content for c in out] == ["a"]


def test_reranker_single_candidate_skips_llm():
    """只有一条候选时没有排序可言, 不该浪费一次 LLM 调用"""
    llm = StubLLM(result=score((0, 1.0)))
    out = LLMReranker(llm).rerank("q", [make_chunk("only")], top_k=3)
    assert len(out) == 1
    assert llm.calls == []


def test_reranker_truncates_candidate_text():
    """候选正文要截断后再送模型, 否则候选一多上下文会爆"""
    long_text = "破" * 5000
    llm = StubLLM(result=score((0, 1.0)))
    LLMReranker(llm, max_chars_per_candidate=100).rerank("q", [make_chunk(long_text),
                                                              make_chunk("x")], top_k=1)
    _schema, messages = llm.calls[0]
    assert long_text not in messages[1].content
    assert "破" * 100 in messages[1].content


def test_reranker_caps_candidate_count():
    chunks = [make_chunk(f"c{i}") for i in range(30)]
    llm = StubLLM(result=score((0, 1.0)))
    LLMReranker(llm, max_candidates=5).rerank("q", chunks, top_k=3)
    _schema, messages = llm.calls[0]
    assert "[4]" in messages[1].content
    assert "[5]" not in messages[1].content      # 第 6 条起不送模型


def test_reranker_uses_listwise_single_call():
    """列表式: N 个候选只调一次 LLM, 不是逐个候选调 N 次"""
    chunks = [make_chunk(f"c{i}") for i in range(8)]
    llm = StubLLM(result=score(*[(i, 0.5) for i in range(8)]))
    LLMReranker(llm).rerank("q", chunks, top_k=3)
    assert len(llm.calls) == 1


# ── LLMReranker: 降级路径 ────────────────────────────────

def test_reranker_falls_back_when_llm_raises():
    """重排调用失败 → 回退粗排顺序, 绝不让检索整体失败"""
    chunks = [make_chunk("a"), make_chunk("b"), make_chunk("c")]
    llm = StubLLM(error=RuntimeError("ollama down"))
    out = LLMReranker(llm).rerank("q", chunks, top_k=2)
    assert [c.content for c in out] == ["a", "b"]


def test_reranker_falls_back_when_no_scores():
    chunks = [make_chunk("a"), make_chunk("b")]
    llm = StubLLM(result=RerankResult(scores=[]))
    out = LLMReranker(llm).rerank("q", chunks, top_k=2)
    assert [c.content for c in out] == ["a", "b"]


def test_reranker_falls_back_when_index_all_invalid():
    chunks = [make_chunk("a"), make_chunk("b")]
    llm = StubLLM(result=score((77, 1.0)))
    out = LLMReranker(llm).rerank("q", chunks, top_k=2)
    assert [c.content for c in out] == ["a", "b"]


# ── 工厂 ─────────────────────────────────────────────────

def _settings(**overrides):
    base = dict(rerank_enabled=False, rerank_candidates=12, rerank_max_chars=300)
    base.update(overrides)
    return Settings(_env_file=None, **base)


def test_factory_returns_noop_when_disabled():
    assert isinstance(create_reranker(_settings(), StubLLM()), NoopReranker)


def test_factory_returns_llm_reranker_when_enabled():
    reranker = create_reranker(_settings(rerank_enabled=True), StubLLM())
    assert isinstance(reranker, LLMReranker)


# ── 与 RagService 的接线 ─────────────────────────────────

class RecordingStore:
    """记录 retrieve 收到的 top_k, 用于验证"重排时先多取候选" """

    def __init__(self, n_chunks: int = 30):
        self.asked_top_k: list[int] = []
        self._chunks = [make_chunk(f"c{i}") for i in range(n_chunks)]

    def count(self) -> int:
        return len(self._chunks)

    def retrieve(self, query, top_k=3, tenant_id="default"):
        self.asked_top_k.append(top_k)
        return self._chunks[:top_k]


def _rag(settings, reranker):
    from app.services.rag_service import RagService

    return RagService(RecordingStore(), None, settings, reranker=reranker)


def test_retrieve_fetches_only_top_k_when_rerank_disabled():
    store_rag = _rag(_settings(top_k=3), NoopReranker())
    store_rag.retrieve("q")
    assert store_rag._store.asked_top_k == [3]


def test_retrieve_fetches_more_candidates_when_rerank_enabled():
    """重排的意义是从更大候选池里挑, 所以要先多取 —— 只取 top_k 再重排没意义"""
    settings = _settings(top_k=3, rerank_enabled=True, rerank_candidates=12)
    llm = StubLLM(result=score(*[(i, 0.5) for i in range(12)]))
    rag = _rag(settings, LLMReranker(llm))
    out = rag.retrieve("q")
    assert rag._store.asked_top_k == [12]     # 粗排取 12
    assert len(out) == 3                      # 精排后截到 top_k


@pytest.mark.parametrize("top_k,expected", [(20, 20), (5, 12)])
def test_retrieve_candidate_count_is_at_least_top_k(top_k, expected):
    """top_k 大于候选数时要取够 —— 否则用户要 20 条却只拿到 12 条"""
    settings = _settings(rerank_enabled=True, rerank_candidates=12)
    llm = StubLLM(result=score(*[(i, 0.5) for i in range(30)]))
    rag = _rag(settings, LLMReranker(llm, max_candidates=30))
    rag.retrieve("q", top_k=top_k)
    assert rag._store.asked_top_k == [expected]
