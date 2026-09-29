"""存活 / 就绪探针必须分开 (P2)

两者的失败含义相反, 混在一个接口里会让编排器做错事:
  · 存活失败 → 重启进程。依赖 (Ollama/索引) 挂了时重启毫无帮助, 还会掐断
    在途请求、引发重启风暴。
  · 就绪失败 → 摘掉流量, 不动进程。这才是依赖故障时该做的。

原先只有一个 /health, 无论依赖是否可用都返回 200 —— 负载均衡无法据此摘流量。
"""

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.deps import get_rag_service, get_settings
from app.api.v1.endpoints.health import router as health_router
from app.config import Settings


class StubRag:
    def __init__(self, ready=True, chunks=91):
        self.ready = ready
        self.indexed_chunks = chunks


class OkProbe:
    async def get(self, url):
        return httpx.Response(200, request=httpx.Request("GET", url))


class DownProbe:
    async def get(self, url):
        raise httpx.ConnectError("connection refused")


class ExplodingProbe:
    """被调用就炸 —— 用来证明存活探针**根本没碰依赖**"""

    async def get(self, url):
        raise AssertionError("存活探针不得访问网络/依赖")


def _client(rag=None, probe=None, **setting_overrides):
    app = FastAPI()
    app.include_router(health_router, prefix="/api/v1")
    settings = Settings(_env_file=None, **setting_overrides)
    app.dependency_overrides[get_rag_service] = lambda: rag or StubRag()
    app.dependency_overrides[get_settings] = lambda: settings
    app.state.ollama_probe = probe or OkProbe()
    return TestClient(app)


# ── 存活: 恒定 200, 且不碰依赖 ────────────────────────────

def test_liveness_ok_even_when_everything_is_down():
    """Ollama 挂了 + 索引空 → 进程仍然活着, 存活探针必须 200"""
    c = _client(rag=StubRag(ready=False, chunks=0), probe=DownProbe())
    r = c.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_liveness_never_touches_dependencies():
    """存活探针不得发起网络调用 —— 探针会在编排器侧高频执行"""
    c = _client(probe=ExplodingProbe())
    assert c.get("/api/v1/health").status_code == 200


def test_liveness_reports_env_and_auth():
    """运维要能一眼看到自己有没有裸奔"""
    c = _client(env="prod", api_key="k" * 24)
    body = c.get("/api/v1/health").json()
    assert body["env"] == "prod"
    assert body["auth"] == "enabled"


# ── 就绪: 依赖不齐就 503 ─────────────────────────────────

def test_readiness_ok_when_dependencies_are_up():
    c = _client()
    r = c.get("/api/v1/health/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert body["ollama"] == "up"
    assert body["reasons"] == []


def test_readiness_503_when_ollama_is_down():
    c = _client(probe=DownProbe())
    r = c.get("/api/v1/health/ready")
    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "not_ready"
    assert body["ollama"] == "down"
    assert any("Ollama" in reason for reason in body["reasons"])


def test_readiness_503_when_index_is_empty():
    c = _client(rag=StubRag(ready=False, chunks=0))
    r = c.get("/api/v1/health/ready")
    assert r.status_code == 503
    assert any("索引" in reason for reason in r.json()["reasons"])


def test_readiness_lists_all_reasons_not_just_the_first():
    """一次把缺的东西都列出来, 省得运维修一个跑一次"""
    c = _client(rag=StubRag(ready=False, chunks=0), probe=DownProbe())
    reasons = c.get("/api/v1/health/ready").json()["reasons"]
    assert len(reasons) == 2


# ── 指标刷新: 只挂 liveness 的部署也要拿得到 kb_chunks ────
# 拆分前 /health 每次都刷新这两个 Gauge; 拆分后如果只在 ready 里写, 那么只配
# liveness 探针的部署 (K8s 的典型用法) 会看到 ollama_up / kb_chunks 恒为 0。

def _gauge(name: str) -> float:
    from prometheus_client import generate_latest
    for line in generate_latest().decode("utf-8").splitlines():
        if line.startswith(name + " "):
            return float(line.split()[-1])
    raise AssertionError(f"指标不存在: {name}")


def test_liveness_refreshes_kb_chunks():
    """kb_chunks 是本地状态, 不需要探依赖 —— 存活探针就该刷新它"""
    c = _client(rag=StubRag(chunks=42))
    c.get("/api/v1/health")
    assert _gauge("kb_chunks") == 42


def test_liveness_does_not_touch_ollama_gauge():
    """ollama_up 只能由就绪探针写 —— 存活探针不探网络, 更不能把它写成 0

    写成 0 会让"只是没人调就绪探针"看起来像"Ollama 挂了"。
    """
    from app.core.metrics import OLLAMA_UP

    OLLAMA_UP.set(1)
    c = _client(rag=StubRag(), probe=DownProbe())
    c.get("/api/v1/health")          # 存活探针: 即便上游不可达也不该动这个 Gauge
    assert _gauge("ollama_up") == 1

    c.get("/api/v1/health/ready")    # 就绪探针才会真的去探
    assert _gauge("ollama_up") == 0
