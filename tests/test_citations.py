"""可定位引用 — sources 必须带上 chunk_index (P2)

"引用了哪个文档"与"引用了哪一段"是两件事。原先 sources 只有 doc_id 级别的
出处, 于是一段被误引用的回答在外观上完全正常 —— 用户无法核对, 也无法跳转
到原文对应段落。

同时这里守住一个易漂移点: RagService 与 multi_hop 节点各有一份 chunk→source
的映射, 加字段时漏改一份不会有任何报错。现在两边共用 SourceItem.from_chunk,
测试断言两者产出的字段集合完全一致。
"""

from app.agents.nodes import _to_source
from app.config import Settings
from app.schemas.chat import SourceItem
from app.infrastructure.vector_store import RetrievedChunk
from app.services.rag_service import RagService


def _chunk(index=3, tags="罚款", doc_id="d1", similarity=0.81234):
    return {
        "content": "第二十八条 经营快递业务的企业……",
        "metadata": {"doc_id": doc_id, "title": "快递暂行条例",
                     "tags": tags, "chunk_index": index},
        "distance": 0.2,
        "similarity": similarity,
    }


# ── from_chunk ───────────────────────────────────────────

def test_from_chunk_carries_chunk_index():
    s = SourceItem.from_chunk(_chunk(index=7))
    assert s.chunk_index == 7
    assert s.doc_id == "d1"
    assert s.title == "快递暂行条例"
    assert s.similarity == 0.8123          # 保留 4 位


def test_missing_chunk_index_is_none_not_error():
    """老索引 / 手工灌入的 chunk 可能没有 chunk_index —— 不能因此报错"""
    c = {"content": "x", "metadata": {"doc_id": "d"}, "similarity": 0.5}
    assert SourceItem.from_chunk(c).chunk_index is None


def test_nullish_metadata_fields_do_not_crash():
    """Chroma 的 metadata 允许 None —— None 不能变成 pydantic 校验错误"""
    c = {"content": "x",
         "metadata": {"doc_id": None, "title": None, "tags": None, "chunk_index": None},
         "similarity": 0.5}
    s = SourceItem.from_chunk(c)
    assert (s.doc_id, s.title, s.tags, s.chunk_index) == ("", "", "", None)


def test_no_metadata_at_all():
    assert SourceItem.from_chunk({"content": "x"}).similarity == 0.0


def test_similarity_none_is_zero():
    assert SourceItem.from_chunk(
        {"content": "x", "metadata": {}, "similarity": None}
    ).similarity == 0.0


# ── 两个入口必须一致 (防漂移) ────────────────────────────

def test_both_mappers_agree_on_field_set():
    """RagService._to_sources 与 nodes._to_source 产出的字段集合必须完全相同"""
    c = _chunk()
    from_rag = RagService._to_sources([c])[0]
    from_node = _to_source(c)
    assert set(from_rag) == set(from_node) == set(SourceItem.model_fields)
    assert from_rag == from_node


def test_rag_sources_include_chunk_index_end_to_end():
    """走一遍 retrieve → _to_sources, 确认 chunk_index 真的透出来了

    只测 from_chunk 的话, 映射写对了但调用点没用它, 测试依然全绿。
    """
    class StubStore:
        def count(self):
            return 1

        def retrieve(self, query, top_k=3, tenant_id="default"):
            # 用真实的 RetrievedChunk: 桩自造一个"碰巧字段够用"的对象, 正是
            # 让这类测试在真实接口变化后仍然全绿的原因
            return [RetrievedChunk(content=_chunk(index=41)["content"],
                                   metadata=_chunk(index=41)["metadata"],
                                   distance=0.2, similarity=0.9)]

    class StubLLM:
        def invoke(self, messages):
            return "依据第二十八条……"

    rag = RagService(StubStore(), StubLLM(), Settings(_env_file=None))
    rag._ready = True                      # 跳过 ingest, 本测试只关心映射
    sources = rag.ask("罚款多少")["sources"]

    assert sources[0]["chunk_index"] == 41
    assert sources[0]["doc_id"] == "d1"
