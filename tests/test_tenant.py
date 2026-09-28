"""P4 多租户隔离 — 注册表 / 向量库隔离 / HTTP 鉴权集成"""

import json

import pytest
from fastapi import Depends, FastAPI, Request
from fastapi.testclient import TestClient

from app.api.deps import enforce_rate_limit, get_tenant_id
from app.config import Settings
from app.core.exceptions import register_exception_handlers
from app.core.security import verify_api_key
from app.infrastructure.vector_store import ChromaVectorStore
from app.services.tenant_service import Tenant, TenantRegistry


def make_settings(**overrides):
    base = dict(
        tenant_mode=True,
        tenants_path="/nonexistent/tenants.json",
        shared_tenant_id="shared",
        api_key=None,
    )
    base.update(overrides)
    return Settings(_env_file=None, **base)


class KeywordEmbedding:
    def embed(self, texts):
        return [[1.0, 0.5] for _ in texts]


def _store(tmp_path):
    return ChromaVectorStore(
        persist_dir=str(tmp_path),
        collection_name="test_kb",
        embeddings=KeywordEmbedding(),
        chunk_size=200,
        chunk_overlap=0,
        retrieval_mode="hybrid",
        shared_tenant_id="shared",
    )


# ── 租户注册表 ───────────────────────────────────────────
def test_registry_load_and_resolve(tmp_path):
    f = tmp_path / "tenants.json"
    f.write_text(json.dumps([
        {"api_key": "key-a", "tenant_id": "net-001", "name": "网点A"},
        {"api_key": "key-b", "tenant_id": "net-002", "name": "网点B"},
    ]), encoding="utf-8")
    reg = TenantRegistry(f)
    assert reg.resolve("key-a") == Tenant("key-a", "net-001", "网点A")
    assert reg.resolve("key-b").tenant_id == "net-002"
    assert reg.resolve("unknown") is None
    assert len(reg.list_tenants()) == 2


def test_registry_missing_file_empty():
    reg = TenantRegistry(None)
    assert reg.resolve("anything") is None
    assert reg.list_tenants() == []


def test_registry_reload():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        f = __import__("pathlib").Path(d) / "t.json"
        f.write_text('[{"api_key": "k1", "tenant_id": "t1"}]', encoding="utf-8")
        reg = TenantRegistry(f)
        assert len(reg.list_tenants()) == 1
        f.write_text('[{"api_key": "k1", "tenant_id": "t1"}, {"api_key": "k2", "tenant_id": "t2"}]',
                     encoding="utf-8")
        assert reg.reload() == 2


# ── 向量库租户隔离 ───────────────────────────────────────
def test_docs_isolated_by_tenant(tmp_path):
    store = _store(tmp_path)
    store.add_document("doc-a", "A网点手册", "包裹破损赔偿500元罚款。", tenant_id="net-001")
    store.add_document("doc-b", "B网点手册", "包裹遗失需24小时找寻。", tenant_id="net-002")
    store.add_document("sop", "共享SOP", "包裹破损需拍照留存证据。", tenant_id="shared")

    # 各租户只看到自己的 + 共享库
    docs_a = {d["doc_id"] for d in store.list_documents(tenant_id="net-001")}
    assert docs_a == {"doc-a", "sop"}
    docs_b = {d["doc_id"] for d in store.list_documents(tenant_id="net-002")}
    assert docs_b == {"doc-b", "sop"}


def test_retrieve_filters_by_tenant(tmp_path):
    store = _store(tmp_path)
    store.add_document("doc-a", "A", "包裹破损赔偿500元。", tenant_id="net-001")
    store.add_document("doc-b", "B", "包裹遗失24小时找寻。", tenant_id="net-002")
    # BM25 决定性 (恒定向量): 查询 500 → 只应命中 net-001 的 doc-a
    chunks = store.retrieve("500元", top_k=5, tenant_id="net-001")
    assert all(c.metadata.get("tenant_id") in ("net-001", "shared") for c in chunks)
    assert any(c.metadata.get("doc_id") == "doc-a" for c in chunks)
    assert not any(c.metadata.get("doc_id") == "doc-b" for c in chunks)


def test_remove_only_own_tenant(tmp_path):
    store = _store(tmp_path)
    store.add_document("doc", "A", "内容A", tenant_id="net-001")
    store.add_document("doc", "B", "内容B", tenant_id="net-002")
    removed = store.remove_document("doc", tenant_id="net-001")
    assert removed >= 1
    # net-002 的 doc 还在
    docs_b = {d["doc_id"] for d in store.list_documents(tenant_id="net-002")}
    assert "doc" in docs_b


# ── HTTP 层: 租户鉴权 ────────────────────────────────────
def _make_app(registry):
    app = FastAPI()
    register_exception_handlers(app)
    app.state.tenant_registry = registry

    @app.get("/api/v1/whoami")
    async def whoami(
        request: Request,
        _: None = Depends(verify_api_key),
        tenant_id: str = Depends(get_tenant_id),
    ):
        return {"tenant_id": tenant_id}

    return app


def test_tenant_key_auth_ok_and_401():
    import tempfile
    from pathlib import Path

    from app.config import get_settings

    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "tenants.json"
        f.write_text('[{"api_key": "secret-a", "tenant_id": "net-001", "name": "A"}]',
                     encoding="utf-8")
        app = _make_app(TenantRegistry(f))
        app.dependency_overrides[get_settings] = lambda: make_settings(
            tenants_path=str(f)
        )
        client = TestClient(app)
        ok = client.get("/api/v1/whoami", headers={"X-API-Key": "secret-a"})
        assert ok.status_code == 200
        assert ok.json()["tenant_id"] == "net-001"
        bad = client.get("/api/v1/whoami", headers={"X-API-Key": "wrong"})
        assert bad.status_code == 401
        assert bad.json()["error"]["code"] == "unauthorized"


def test_non_tenant_mode_defaults_to_default_tenant():
    from fastapi import Request

    settings = make_settings(tenant_mode=False)
    assert settings.tenant_mode is False
    # 单租户模式下 tenant_id 回退 default
    app = FastAPI()

    @app.get("/x")
    async def x(_: None = Depends(verify_api_key), tenant_id: str = Depends(get_tenant_id)):
        return {"tenant_id": tenant_id}

    app.state.tenant_registry = TenantRegistry(None)
    # verify_api_key 依赖 get_settings → 用注入的 settings
    app.dependency_overrides = {}
    import app.api.deps as deps_mod
    from app.config import get_settings

    app.dependency_overrides[get_settings] = lambda: settings
    client = TestClient(app)
    r = client.get("/x")
    assert r.status_code == 200
    assert r.json()["tenant_id"] == "default"


def test_get_tenant_id_fails_closed_without_verify_api_key():
    """路由漏挂 verify_api_key 时必须报错, **不能回退默认租户**

    回退的话, 一个"忘了加依赖"的疏忽就直接变成跨租户读写默认租户的数据,
    而且没有任何报错 —— 静默的越权比 500 危险得多。
    """
    app = FastAPI()

    @app.get("/leaky")
    async def leaky(tenant_id: str = Depends(get_tenant_id)):  # 故意不挂 verify_api_key
        return {"tenant_id": tenant_id}

    r = TestClient(app, raise_server_exceptions=False).get("/leaky")
    assert r.status_code == 500          # 不是 200 + 悄悄读默认租户


def test_get_tenant_id_uses_what_verify_api_key_wrote():
    """正常链路不受影响: verify_api_key 写了什么就读到什么"""
    app = FastAPI()

    @app.get("/x")
    async def x(_: None = Depends(verify_api_key), tenant_id: str = Depends(get_tenant_id)):
        return {"tenant_id": tenant_id}

    app.state.tenant_registry = TenantRegistry(None)
    from app.config import get_settings

    app.dependency_overrides[get_settings] = lambda: make_settings(tenant_mode=False)
    r = TestClient(app).get("/x")
    assert r.status_code == 200
    assert r.json()["tenant_id"] == "default"
