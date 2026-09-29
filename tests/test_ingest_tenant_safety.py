"""语料重建**不得**波及其他租户 (跨租户破坏)

真实缺陷 (由一次对抗性评审实测发现): `RagService.ingest()` 在三种情形下调用
`store.clear_all()` —— 显式 force、清单缺失、分块参数变更 —— 而 `clear_all()`
删的是**整个集合**。集合是多租户共用的, 于是:

  · 任何一个持有效密钥的租户, 打一次 `POST /rag/ingest?force=true`,
    就能把**所有租户**上传的文档一起删掉;
  · 更糟的是**启动路径**: 只要 `data/chroma/index_manifest.json` 不在 (新机器、
    换了卷、被清过), 服务一启动就 `force=false` 走全量重建, 同样清空所有租户。

跨租户的**破坏**比跨租户读取更严重, 而且触发者往往是无意的。修法是把
`clear_all()` 整个删掉, 换成 `remove_tenant(shared_tenant_id)` —— 语料重建只该动
语料自己那份数据。这里用的就是评审给的复现路径, 只是把它变成回归测试。
"""

from pathlib import Path

import pytest

from app.config import Settings
from app.infrastructure.index_manifest import SCHEMA_VERSION
from app.infrastructure.vector_store import ChromaVectorStore
from app.services.rag_service import RagService


class ConstEmbedding:
    """恒定向量: 本测试只关心"谁的文档还在", 不关心检索质量"""

    def embed(self, texts):
        return [[1.0, 0.5] for _ in texts]


class StubLLM:
    def invoke(self, messages):
        return "x"


def _settings(tmp_path: Path) -> Settings:
    corpus = tmp_path / "corpus"
    corpus.mkdir(exist_ok=True)
    (corpus / "policy.txt").write_text("共享语料内容。" * 50, encoding="utf-8")
    return Settings(
        _env_file=None,
        chroma_dir=str(tmp_path / "chroma"),
        corpus_dir=str(corpus),
        sop_path=str(tmp_path / "nonexistent-sop.txt"),   # 只导入 corpus
        collection_name="tenant_safety_kb",
        retrieval_mode="vector",                          # 不依赖 BM25, 更快
        chunk_size=200,
        chunk_overlap=0,
    )


def _store(settings: Settings) -> ChromaVectorStore:
    return ChromaVectorStore(
        persist_dir=settings.chroma_dir,
        collection_name=settings.collection_name,
        embeddings=ConstEmbedding(),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        retrieval_mode=settings.retrieval_mode,
    )


@pytest.fixture
def env(tmp_path):
    settings = _settings(tmp_path)
    store = _store(settings)
    # 两个租户各自上传的私有文档
    store.add_document("alice-secret", "甲的内部手册",
                       "甲的机密内容。" * 30, tenant_id="alice")
    store.add_document("bob-secret", "乙的内部手册",
                       "乙的机密内容。" * 30, tenant_id="bob")
    assert store.count() > 2
    return settings, store


def _doc_ids(store, tenant):
    """只看该租户**自己的**文档。

    `list_documents(tenant)` 会按设计带上共享库的文档 (所有人都该检索到语料),
    所以断言"他租户的文档还在"必须把共享库过滤掉, 否则会把
    "语料重建成功" 误读成 "他租户的文档还在"。
    """
    return sorted(d["doc_id"] for d in store.list_documents(tenant)
                  if d.get("tenant_id") == tenant)


def test_force_rebuild_keeps_other_tenants_documents(env):
    """force=true 全量重建: 语料重导, 但各租户上传的文档必须原样都在"""
    settings, store = env
    rag = RagService(store, StubLLM(), settings)

    rag.ingest(force=True)

    assert _doc_ids(store, "alice") == ["alice-secret"]
    assert _doc_ids(store, "bob") == ["bob-secret"]


def test_startup_path_without_manifest_keeps_other_tenants_documents(env):
    """启动路径 (force=false, 仅因清单缺失而全量重建) 同样不得清空他租户

    这条比上一条更危险: 触发它不需要任何请求, 服务起来就发生了。
    """
    settings, store = env
    manifest = settings.manifest_path
    assert not manifest.exists()          # 新机器/换卷后的常态
    rag = RagService(store, StubLLM(), settings)

    rag.ingest()                          # 等价于 lifespan 里的调用

    assert _doc_ids(store, "alice") == ["alice-secret"]
    assert _doc_ids(store, "bob") == ["bob-secret"]


def test_contract_change_rebuild_keeps_other_tenants_documents(env):
    """改了分块参数 → 索引契约变更 → 全量重建, 也不能波及他租户"""
    settings, store = env
    settings.chunk_size = 123             # 模拟 chunk_size 被改
    rag = RagService(store, StubLLM(), settings)

    rag.ingest()

    assert _doc_ids(store, "alice") == ["alice-secret"]
    assert _doc_ids(store, "bob") == ["bob-secret"]


def test_shared_corpus_is_actually_rebuilt(env):
    """只保住他租户还不够 —— 共享语料本身必须真的重建 (别把修复做成什么都不删)"""
    settings, store = env
    rag = RagService(store, StubLLM(), settings)
    rag.ingest(force=True)

    shared = _doc_ids(store, settings.shared_tenant_id)
    assert shared, "共享语料应当已入库"
    manifest = settings.manifest_path
    assert manifest.exists()
    assert manifest.read_text(encoding="utf-8").count('"docs"') == 1


def test_remove_tenant_does_not_touch_shared_corpus(env):
    """remove_tenant 的边界: 删某个租户不影响共享库"""
    settings, store = env
    rag = RagService(store, StubLLM(), settings)
    rag.ingest(force=True)
    shared_before = _doc_ids(store, settings.shared_tenant_id)

    store.remove_tenant("alice")

    assert _doc_ids(store, "alice") == []
    assert _doc_ids(store, settings.shared_tenant_id) == shared_before
    assert _doc_ids(store, "bob") == ["bob-secret"]


def test_vector_store_has_no_clear_all_primitive():
    """**刻意不提供"清空全部"**: 没有这个方法, 就不会有人再误用它

    这条测试是"用删能力来修 bug"的守卫 —— 谁把 clear_all 加回来, 这里会红。
    """
    from app.infrastructure import vector_store

    assert not hasattr(vector_store.ChromaVectorStore, "clear_all")
    assert not hasattr(vector_store.VectorStore, "clear_all")
    assert SCHEMA_VERSION >= 1            # 顺手确认模块可正常导入
