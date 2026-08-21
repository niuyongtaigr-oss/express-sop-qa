"""P3-D 细粒度限流与配额 — RateLimiter 单元 + 依赖/异常集成"""

import time

import pytest
from fastapi import Depends, FastAPI, Request
from fastapi.testclient import TestClient

from app.core.exceptions import RateLimitExceeded, register_exception_handlers
from app.core.rate_limit import RateLimiter
from app.api.deps import enforce_rate_limit


# ── RateLimiter 单元 ─────────────────────────────────────
def test_disabled_always_allows():
    rl = RateLimiter(enabled=False, per_min=10, burst=5)
    for _ in range(20):
        ok, retry, remaining = rl.allow("k")
        assert ok is True
        assert retry is None


def test_burst_then_throttle():
    rl = RateLimiter(enabled=True, per_min=60, burst=2)  # 1 token/s, 容量 2
    assert rl.allow("k")[0] is True
    assert rl.allow("k")[0] is True
    ok, retry, _ = rl.allow("k")
    assert ok is False
    assert retry is not None and retry >= 1  # 需等待补充
    # 补充 1.5s 后恢复
    time.sleep(1.6)
    assert rl.allow("k")[0] is True


def test_per_key_isolation():
    rl = RateLimiter(enabled=True, per_min=60, burst=1)
    assert rl.allow("a")[0] is True
    assert rl.allow("a")[0] is False
    assert rl.allow("b")[0] is True  # 另一个 Key 不受影响


def test_daily_quota_exceeded():
    rl = RateLimiter(enabled=True, per_min=1000, burst=1000, daily_quota=2)
    assert rl.allow("k")[0] is True
    assert rl.allow("k")[0] is True
    ok, retry, remaining = rl.allow("k")
    assert ok is False
    assert remaining == 0
    assert retry is not None and retry > 0  # 距次日零点


def test_quota_returns_remaining():
    rl = RateLimiter(enabled=True, per_min=1000, burst=1000, daily_quota=5)
    _, _, r1 = rl.allow("k")
    _, _, r2 = rl.allow("k")
    assert r1 == 4 and r2 == 3


def test_sweep_removes_idle_keys():
    rl = RateLimiter(enabled=True, per_min=60, burst=2)
    rl.allow("idle")
    # 伪造 last_access 为很久以前
    rl._buckets["idle"].last_access = time.monotonic() - 7200
    assert rl.sweep(idle_s=3600) == 1
    assert rl.size() == 0


# ── 依赖 + 异常集成 (HTTP 429 + Retry-After) ─────────────
def _make_app(limiter):
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/api/v1/business")
    async def business(_: None = Depends(enforce_rate_limit)):
        return {"ok": True}

    app.state.rate_limiter = limiter
    return app


def test_429_with_retry_after_header():
    limiter = RateLimiter(enabled=True, per_min=60, burst=1)
    app = _make_app(limiter)
    client = TestClient(app)
    r1 = client.get("/api/v1/business", headers={"X-API-Key": "k1"})
    assert r1.status_code == 200
    r2 = client.get("/api/v1/business", headers={"X-API-Key": "k1"})
    assert r2.status_code == 429
    assert r2.json()["error"]["code"] == "rate_limited"
    assert int(r2.headers["Retry-After"]) >= 1
    # 其他 Key 不受影响
    r3 = client.get("/api/v1/business", headers={"X-API-Key": "k2"})
    assert r3.status_code == 200


def test_rate_limited_error_has_retry_after():
    exc = RateLimitExceeded("限流", retry_after_s=5)
    assert exc.status_code == 429
    assert exc.retry_after_s == 5
