"""P1.5+P1.6 向量库测试 — 混合检索 (RRF 融合) + 多文档管理

用 FakeEmbedding + 临时目录驱动真实 Chroma PersistentClient, 无需 Ollama。
"""

import pytest

from app.infrastructure.vector_store import ChromaVectorStore, RetrievedChunk


class ConstEmbedding:
    """恒定向量: 向量检索对所有文档一视同仁, 便于验证 BM25 在融合中的决定性作用"""

    def embed(self, texts):
        return [[1.0, 0.5] for _ in texts]


class KeywordEmbedding:
    """按关键词维度的向量: 用于文档管理/检索结构测试"""

    _DIMS = {"破损": 0, "遗失": 1, "理赔": 2, "拦截": 3}

    def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * 5
            v[4] = 1.0  # bias
            for kw, idx in self._DIMS.items():
                if kw in t:
                    v[idx] = 1.0
            out.append(v)
        return out


def _store(tmp_path, mode="hybrid", **kw):
    return ChromaVectorStore(
        persist_dir=str(tmp_path),
        collection_name="test_kb",
        embeddings=kw.pop("embedding", ConstEmbedding()),
        chunk_size=200,
        chunk_overlap=0,
        retrieval_mode=mode,
    )


# 一个文档正文 (足够长, 可分多个 chunk: chunk_size=200, overlap=0; 各段内容不同)
_SOP_BASE = (
    "包裹破损需立即在系统异常登记，单独放置异常件区域并拍照留存证据，"
    "贵重物品破损需2小时内联系客服中心备案。理赔流程：网点提交照片和运单号，"
    "省区审核后总公司理赔部打款。若因网点暴力分拣导致破损，处以每件500元罚款。"
)
_LONG_SOP = "".join(f"{i + 1}. {_SOP_BASE}\n" for i in range(6))

_LOST_BASE = (
    "包裹连续3天未扫描更新将自动标记为疑似遗失。网点收到预警后需24小时内完成"
    "全网找寻。确认遗失的包裹按保价金额赔偿，未保价最高不超过运费3倍。"
)
_LONG_LOST = "".join(f"{i + 1}. {_LOST_BASE}\n" for i in range(6))


# ── 文档管理 (P1.6) ──────────────────────────────────────
def test_add_list_remove_documents(tmp_path):
    store = _store(tmp_path)
    assert store.add_document("sop", "默认文档", _LONG_SOP) >= 2
    n2 = store.add_document("lost", "遗失规范", _LONG_LOST)
    assert n2 >= 2
    docs = store.list_documents()
    assert {d["doc_id"] for d in docs} == {"sop", "lost"}
    by_id = {d["doc_id"]: d for d in docs}
    assert by_id["sop"]["title"] == "默认文档"
    assert by_id["lost"]["chunk_count"] == n2

    removed = store.remove_document("sop")
    assert removed >= 2
    docs = store.list_documents()
    assert [d["doc_id"] for d in docs] == ["lost"]


def test_upsert_replaces_document_chunks(tmp_path):
    store = _store(tmp_path)
    before = store.add_document("sop", "v1", _LONG_SOP)
    assert before >= 2
    # 覆盖: 旧 chunk 删除, 新内容只有 1 个 chunk
    n = store.add_document("sop", "v2", "包裹破损要拍照留存。")
    assert n == 1
    assert store.count() == 1
    docs = store.list_documents()
    assert docs[0]["title"] == "v2"


def test_remove_tenant_only_affects_that_tenant(tmp_path):
    """删某个租户**不能**波及其他租户 —— 集合是共用的"""
    store = _store(tmp_path)
    store.add_document("a", "A", "包裹破损处理规范。", tenant_id="alice")
    store.add_document("b", "B", "包裹遗失处理规范。", tenant_id="bob")

    removed = store.remove_tenant("alice")
    assert removed > 0
    assert store.list_documents("alice") == []
    assert [d["doc_id"] for d in store.list_documents("bob")] == ["b"]
    # BM25 索引也要跟着丢, 否则会检索到已删内容
    assert all("alice" not in c.metadata.get("tenant_id", "")
               for c in store.retrieve("破损", tenant_id="bob"))


def test_remove_tenant_is_noop_for_unknown_tenant(tmp_path):
    store = _store(tmp_path)
    store.add_document("a", "A", "包裹破损处理规范。")
    assert store.remove_tenant("nobody") == 0
    assert store.count() > 0


def test_chunk_metadata_has_doc_fields(tmp_path):
    store = _store(tmp_path)
    store.add_document("sop", "默认", "包裹破损赔偿500元罚款。")
    chunks = store.retrieve("破损", top_k=1)
    assert chunks[0].metadata["doc_id"] == "sop"
    assert chunks[0].metadata["title"] == "默认"
    assert "破损" in chunks[0].metadata["tags"]


# ── 混合检索 (P1.5) ──────────────────────────────────────
def test_hybrid_retrieve_returns_top_k(tmp_path):
    store = _store(tmp_path)
    store.add_document("sop", "默认", _LONG_SOP)
    store.add_document("lost", "遗失", _LONG_LOST)
    chunks = store.retrieve("破损怎么赔偿", top_k=3)
    assert len(chunks) == 3
    assert all(isinstance(c, RetrievedChunk) for c in chunks)
    # similarity 均落在 (0,1]
    assert all(0 < c.similarity <= 1 for c in chunks)


def test_pure_vector_mode(tmp_path):
    store = _store(tmp_path, mode="vector")
    store.add_document("sop", "默认", _LONG_SOP)
    chunks = store.retrieve("破损", top_k=2)
    assert len(chunks) == 2


def test_bm25_decides_when_vector_is_tie(tmp_path):
    """恒定向量 → 向量检索全平局; BM25 命中文档必须排在融合结果第 1"""
    store = _store(tmp_path, embedding=ConstEmbedding())
    store.add_document("sop", "默认", "包裹破损赔偿500元罚款。\n" * 2)
    store.add_document("lost", "遗失", "包裹遗失24小时内完成找寻。\n" * 2)
    chunks = store.retrieve("500元罚款", top_k=1)
    assert chunks[0].metadata["doc_id"] == "sop"


def test_rrf_merge_orders_and_pseudo_sim():
    """RRF 融合逻辑: 双列表共同命中文档居首; BM25 独有命中带伪相似度"""
    # 不触碰 Chroma: 直接构造未初始化的实例, 仅测融合函数
    store = object.__new__(ChromaVectorStore)
    from app.infrastructure.bm25 import BM25Index

    store._bm25 = BM25Index(
        ["A", "B", "C", "D"],
        [{"doc_id": "a"}, {"doc_id": "b"}, {"doc_id": "c"}, {"doc_id": "d"}],
    )
    # 手工构造: vec=[A,C], bm 命中 [C,D] (D 为 BM25 独有)
    vec = [
        RetrievedChunk(content="A", metadata={"doc_id": "a"}),
        RetrievedChunk(content="C", metadata={"doc_id": "c"}, similarity=0.9),
    ]
    merged = store._hybrid_merge(vec, [(2, 5.0), (3, 3.0)], top_k=3)
    # C 同时出现在两个列表 → RRF 分数最高 → 第 1
    assert merged[0].content == "C"
    # D 是 BM25 独有 → 出现在结果中且相似度为伪相似度 (0.4, 0.9]
    d = next(c for c in merged if c.content == "D")
    assert 0.4 < d.similarity <= 0.9
    assert d.metadata["doc_id"] == "d"
