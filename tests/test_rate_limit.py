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


# ── 配置错误与淘汰语义 (P2) ──────────────────────────────
# 限流器里两处"看起来能用、实际是错的"的地方:
#   1. per_min=0 → 算 Retry-After 时除零, 被限流的请求变成 500
#   2. 表满时按 LRU 淘汰任意桶 → 把限流/配额白送回去

def test_zero_per_min_is_rejected_at_construction():
    """per_min=0 会让 Retry-After 除零 (500 而不是 429) —— 必须启动就报错"""
    with pytest.raises(ValueError, match="per_min"):
        RateLimiter(enabled=True, per_min=0, burst=5)


def test_zero_per_min_rejected_by_settings():
    """配置层也要挡: SOP_QA_RATE_LIMIT_PER_MIN=0 不应等到运行时才炸"""
    from pydantic import ValidationError

    from app.config import Settings

    with pytest.raises(ValidationError):
        Settings(_env_file=None, rate_limit_per_min=0)


def test_zero_max_keys_is_rejected():
    with pytest.raises(ValueError, match="max_keys"):
        RateLimiter(enabled=True, per_min=60, max_keys=0)


def test_eviction_never_resets_daily_quota():
    """把自己的桶挤出表再回来, 不能重置当日配额

    旧实现按 LRU 淘汰任意桶 —— 刷够 max_keys 个别的 Key 就能让自己满血复活。
    """
    rl = RateLimiter(enabled=True, per_min=6000, burst=1000,
                     daily_quota=1, max_keys=2)
    assert rl.allow("victim")[0] is True          # 用掉当日唯一名额
    for i in range(5):
        rl.allow(f"f{i}")                         # 试图把 victim 挤出表
    ok, _, remaining = rl.allow("victim")
    assert ok is False and remaining == 0         # 配额没有被重置


def test_eviction_never_gifts_tokens():
    """把自己的桶挤出表再回来, 不能重新拿到一整桶令牌"""
    rl = RateLimiter(enabled=True, per_min=60, burst=1, max_keys=2,
                     daily_quota=1000)
    assert rl.allow("victim")[0] is True
    assert rl.allow("victim")[0] is False         # 桶空
    for i in range(5):
        rl.allow(f"f{i}")
    assert rl.allow("victim")[0] is False         # 仍然限流中


def test_token_table_wedge_self_heals():
    """表满且没有桶已回满时拒绝新 Key (fail-closed) —— 但必须能自愈

    否则一次洪峰就会永久性地挡住所有新 Key。
    """
    rl = RateLimiter(enabled=True, per_min=6000, burst=2, max_keys=2,
                     daily_quota=1000)
    assert rl.allow("a")[0] is True
    assert rl.allow("b")[0] is True
    assert rl.allow("c")[0] is False              # 表满, a/b 都还没回满
    time.sleep(0.05)                              # burst=2, rate=100/s → 20ms 回满
    assert rl.allow("c")[0] is True               # 淘汰已回满的桶后放行


def test_daily_table_prunes_rolled_over_days():
    """当日用量表满时, 只清跨天的记录 (它们本来就会归零)"""
    from datetime import date

    rl = RateLimiter(enabled=True, per_min=6000, burst=100, max_keys=2,
                     daily_quota=100)
    assert rl._daily_max == 8
    for i in range(8):
        rl._daily[f"old{i}"] = [date(2020, 1, 1), 5]
    assert rl.allow("newbie")[0] is True          # 跨天记录被清掉, 腾出位置
    assert "old0" not in rl._daily


def test_daily_table_full_of_today_rejects_new_keys():
    """全是今天的记录时一条都不能丢 (丢了就是配额重置) → 拒绝新 Key"""
    from datetime import date

    rl = RateLimiter(enabled=True, per_min=6000, burst=100, max_keys=2,
                     daily_quota=100)
    for i in range(rl._daily_max):
        rl._daily[f"k{i}"] = [date.today(), 1]
    assert rl.allow("newbie")[0] is False
    assert rl._daily_saturated_day == date.today()
    # 已有记录的 Key 不受影响
    assert rl.allow("k0")[0] is True
    # 已确认占满后不再每次全表扫
    assert rl.allow("another")[0] is False


def test_denied_key_does_not_consume_a_daily_slot():
    """被令牌表拒绝的新 Key **不占**当日记账位

    写反了顺序的话 —— 先占记账位再查令牌 —— 记账表会被一堆"从没被服务过"的 Key
    填满; 一旦填满, `_daily_saturated_day` 会在当天一直生效, 于是**所有全新 Key
    都被封到次日零点**, 哪怕令牌桶早就回满可以淘汰了。令牌表本来"秒级自愈", 这样
    一写就变成"当天不可用"。
    """
    rl = RateLimiter(enabled=True, per_min=6000, burst=2, max_keys=2,
                     daily_quota=1000)
    assert rl.allow("a")[0] is True
    assert rl.allow("b")[0] is True
    used_before = len(rl._daily)

    for i in range(20):
        assert rl.allow(f"probe{i}")[0] is False     # 全被令牌表拒绝

    assert len(rl._daily) == used_before, "被拒的 Key 不该占当日记账位"
    assert rl._daily_saturated_day is None, "不该因为被拒的 Key 把当天记成已满"

    time.sleep(0.05)                                 # burst=2, rate=100/s → 20ms 回满
    assert rl.allow("probe0")[0] is True, "令牌桶回满后新 Key 必须立刻可用"


def test_alert_hint_names_a_real_setting():
    """告警让运维去调 SOP_QA_RATE_LIMIT_MAX_KEYS —— 这个环境变量必须真的存在

    调不了的告警等于没有告警, 所以这条断言把"告警文本"和"配置项"绑在一起。
    """
    from app.config import Settings
    from app.core.rate_limit import DAILY_MAX_HINT

    assert DAILY_MAX_HINT == "SOP_QA_RATE_LIMIT_MAX_KEYS"
    field = DAILY_MAX_HINT.removeprefix("SOP_QA_").lower()
    assert field in Settings.model_fields


def test_max_keys_is_configurable():
    """上界必须可调 —— 否则告警里那句"请调大"是空话"""
    from pydantic import ValidationError

    from app.config import Settings

    assert Settings(_env_file=None).rate_limit_max_keys == 10000
    assert Settings(_env_file=None, rate_limit_max_keys=7).rate_limit_max_keys == 7
    with pytest.raises(ValidationError):
        Settings(_env_file=None, rate_limit_max_keys=0)
