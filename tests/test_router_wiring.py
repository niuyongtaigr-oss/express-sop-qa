"""真实路由的鉴权与限流接线 — 只测依赖本身是不够的

审计发现的盲区: `tests/test_tenant.py` 与 `tests/test_rate_limit.py` 都用**自建的
小型 FastAPI app** 测 `verify_api_key` / `enforce_rate_limit` 这两个依赖, 但没有任何
测试碰过**真实路由**。后果是: 把 `dependencies=[Depends(verify_api_key), ...]` 从
`endpoints/chat.py` / `rag.py` / `eval.py` 上整个删掉, 322 个测试**全绿** ——
/chat、/chat/stream、六个 /rag/*、两个 /eval/* 全线变成匿名可调用, 而测试毫无反应。

这正是"桩镜像了实现假设"的典型: 依赖是好的, 接线没了。所以这里只针对**装配**
(接线) 断言, 用真实 app 走 HTTP。

两个方向都要测:
  · 业务路由**必须**要鉴权 (漏挂 = 匿名可用)
  · 探活/指标路由**必须**不需要鉴权 (多挂 = 编排器探针全 401, 服务被判死)
"""

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.core.rate_limit import RateLimiter
from app.main import create_app

KEY = "k" * 40

# 探针与抓取类接口: 编排器不会带密钥, 这几个**应当**匿名可访问
ANONYMOUS_ALLOWED = {"/api/v1/health", "/api/v1/health/ready", "/api/v1/metrics"}


class _StubRag:
    """只满足本文件要打的那些处理函数, 避免依赖 app.state 里的重对象"""

    ready = True
    indexed_chunks = 1
    kb_version = 0

    def supported_document_extensions(self):
        return [".txt"]


def _app(*, api_key=KEY, rate_limit_enabled=False, burst=20):
    app = create_app()
    settings = Settings(_env_file=None, api_key=api_key, env="dev",
                        rate_limit_enabled=rate_limit_enabled,
                        rate_limit_per_min=60, rate_limit_burst=burst)
    app.dependency_overrides[get_settings] = lambda: settings
    # 不走 lifespan, 手工放上处理函数需要的最小状态
    app.state.rag_service = _StubRag()
    app.state.rate_limiter = RateLimiter(enabled=rate_limit_enabled, per_min=60,
                                         burst=burst)
    return app


def _business_routes(client: TestClient) -> list[tuple[str, str]]:
    """从 OpenAPI 里取出所有非豁免的 (method, path), 路径参数填占位值

    自动取材的意义: 将来**新增**一个业务路由, 它会自动进入这份清单被要求鉴权,
    不需要谁记得回来改测试。
    """
    spec = client.get("/openapi.json").json()
    out: list[tuple[str, str]] = []
    for path, ops in spec["paths"].items():
        if path in ANONYMOUS_ALLOWED:
            continue
        concrete = path.replace("{task_id}", "t1").replace("{doc_id}", "d1")
        for method in ops:
            out.append((method.upper(), concrete))
    return sorted(out)


# ── 业务路由必须鉴权 ─────────────────────────────────────

def test_every_business_route_rejects_missing_key():
    client = TestClient(_app(), raise_server_exceptions=False)
    routes = _business_routes(client)
    assert len(routes) >= 10, f"业务路由太少, 取材逻辑可能坏了: {routes}"

    leaked = []
    for method, path in routes:
        r = client.request(method, path)
        if r.status_code != 401:
            leaked.append(f"{method} {path} -> {r.status_code}")
    assert not leaked, "这些业务路由居然不需要鉴权: " + "; ".join(leaked)


def test_every_business_route_rejects_wrong_key():
    client = TestClient(_app(), raise_server_exceptions=False)
    leaked = [
        f"{m} {p}" for m, p in _business_routes(client)
        if client.request(m, p, headers={"X-API-Key": "wrong"}).status_code != 401
    ]
    assert not leaked, "错误密钥竟然放行: " + "; ".join(leaked)


def test_no_auth_configured_means_open():
    """反方向: 没配 key (dev) 时不该 401 —— 否则本地开发全废

    这里只断言"不是 401", 因为处理函数本身缺状态会 500, 与鉴权无关。
    """
    client = TestClient(_app(api_key=None), raise_server_exceptions=False)
    for method, path in _business_routes(client):
        assert client.request(method, path).status_code != 401, f"{method} {path}"


# ── 探针/指标必须免鉴权 ──────────────────────────────────

def test_anonymous_routes_are_not_401():
    """探针带密钥才让访问, 编排器会把服务判死"""
    client = TestClient(_app(), raise_server_exceptions=False)
    for path in sorted(ANONYMOUS_ALLOWED):
        assert client.get(path).status_code != 401, f"{path} 竟然要鉴权"


# ── 限流接线 ─────────────────────────────────────────────

def test_real_router_is_rate_limited_with_retry_after():
    """真实路由上的限流接线: burst=1 → 第二次必须 429 且带 Retry-After

    这是"接线存在"的端到端证据 —— 依赖单测证明不了路由上挂没挂它。
    """
    client = TestClient(_app(rate_limit_enabled=True, burst=1),
                        raise_server_exceptions=False)
    h = {"X-API-Key": KEY}
    first = client.get("/api/v1/rag/docs/formats", headers=h)
    assert first.status_code == 200, first.text

    second = client.get("/api/v1/rag/docs/formats", headers=h)
    assert second.status_code == 429
    assert int(second.headers["Retry-After"]) >= 1
    assert second.json()["error"]["code"] == "rate_limited"


def test_rate_limit_is_per_tenant_on_real_router(tmp_path):
    """多租户下, 一个租户把自己的额度打满, 不该牵连另一个租户

    单租户模式只有一个有效密钥, 谈不上"按密钥隔离", 所以这条必须走 tenant_mode
    才能测到真实语义。
    """
    import json

    from app.services.tenant_service import TenantRegistry

    tenants = tmp_path / "tenants.json"
    tenants.write_text(json.dumps([
        {"api_key": "a" * 40, "tenant_id": "net-001", "name": "甲"},
        {"api_key": "b" * 40, "tenant_id": "net-002", "name": "乙"},
    ]), encoding="utf-8")

    app = _app(api_key=None, rate_limit_enabled=True, burst=1)
    app.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None, env="dev", tenant_mode=True, rate_limit_enabled=True,
        rate_limit_per_min=60, rate_limit_burst=1,
    )
    app.state.tenant_registry = TenantRegistry(tenants)
    client = TestClient(app, raise_server_exceptions=False)

    a = {"X-API-Key": "a" * 40}
    b = {"X-API-Key": "b" * 40}
    assert client.get("/api/v1/rag/docs/formats", headers=a).status_code == 200
    assert client.get("/api/v1/rag/docs/formats", headers=a).status_code == 429
    assert client.get("/api/v1/rag/docs/formats", headers=b).status_code == 200


def test_auth_runs_before_rate_limit():
    """没带密钥的请求必须是 401, 不该被记进限流桶

    否则任何人用垃圾密钥就能把某个桶打满, 影响真正持密钥的调用方。
    """
    client = TestClient(_app(rate_limit_enabled=True, burst=1),
                        raise_server_exceptions=False)
    for _ in range(5):
        assert client.get("/api/v1/rag/docs/formats").status_code == 401
    # 真正的密钥仍有完整的 burst
    assert client.get("/api/v1/rag/docs/formats",
                      headers={"X-API-Key": KEY}).status_code == 200


@pytest.mark.parametrize("path", sorted(ANONYMOUS_ALLOWED))
def test_anonymous_routes_are_not_rate_limited(path):
    """探针被限流 = 编排器拿不到状态, 进而摘流量/重启, 与限流的目的相反"""
    client = TestClient(_app(rate_limit_enabled=True, burst=1),
                        raise_server_exceptions=False)
    for _ in range(3):
        assert client.get(path).status_code != 429
