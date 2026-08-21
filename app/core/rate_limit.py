"""细粒度限流与配额 — RateLimiter (P3-D)

在全局信号量 (防打挂 Ollama) 之上, 按调用方 (X-API-Key / 客户端 IP) 做:
  1. 令牌桶限流: 每 Key 每分钟 N 次, 突发上限 burst (超出返回 429 + Retry-After)
  2. 每日配额: 每 Key 每天 M 次, 跨天自动重置

线程安全; 空闲 Key 定期/惰性清理, 桶数有上限 (LRU 淘汰)。
🏭 Java 对标: Guava RateLimiter + 调用方配额表
"""

import logging
import math
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from threading import Lock
from datetime import date

logger = logging.getLogger(__name__)


@dataclass
class _Bucket:
    tokens: float
    capacity: float
    updated: float = field(default_factory=time.monotonic)
    last_access: float = field(default_factory=time.monotonic)
    day: date = field(default_factory=date.today)
    daily_used: int = 0


class RateLimiter:
    """按 Key 的令牌桶 + 每日配额"""

    def __init__(
        self,
        enabled: bool = False,
        per_min: int = 60,
        burst: int = 20,
        daily_quota: int = 1000,
        max_keys: int = 10000,
    ):
        self._enabled = enabled
        self._rate = per_min / 60.0  # 每秒补充令牌数
        self._capacity = float(burst)
        self._daily_quota = daily_quota
        self._max_keys = max_keys
        self._buckets: "OrderedDict[str, _Bucket]" = OrderedDict()
        self._lock = Lock()

    # ── 对外 ─────────────────────────────────────────────
    def allow(self, key: str) -> tuple[bool, float | None, int | None]:
        """检查请求是否放行。

        返回 (ok, retry_after_s, daily_remaining):
          ok=True            → 放行, retry_after=None
          ok=False, 限流     → retry_after=秒数 (令牌桶空)
          ok=False, 配额尽   → retry_after=距次日零点秒数, daily_remaining=0
        """
        if not self._enabled:
            return True, None, None
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                if len(self._buckets) >= self._max_keys:
                    self._buckets.popitem(last=False)  # LRU 淘汰
                bucket = _Bucket(tokens=self._capacity, capacity=self._capacity)
                self._buckets[key] = bucket
            bucket.last_access = time.monotonic()
            self._buckets.move_to_end(key)

            # 令牌补充
            now = time.monotonic()
            bucket.tokens = min(
                bucket.capacity,
                bucket.tokens + (now - bucket.updated) * self._rate,
            )
            bucket.updated = now

            if bucket.tokens < 1.0:
                retry = math.ceil((1.0 - bucket.tokens) / self._rate)
                return False, max(retry, 1), None

            # 每日配额 (跨天重置)
            today = date.today()
            if bucket.day != today:
                bucket.day = today
                bucket.daily_used = 0
            if bucket.daily_used >= self._daily_quota:
                retry = self._seconds_until_midnight()
                return False, max(retry, 1), 0

            bucket.tokens -= 1.0
            bucket.daily_used += 1
            return True, None, self._daily_quota - bucket.daily_used

    # ── 维护 ─────────────────────────────────────────────
    def sweep(self, idle_s: float = 3600.0) -> int:
        """清理空闲过久的 Key (返回清理数)"""
        with self._lock:
            now = time.monotonic()
            idle = [k for k, b in self._buckets.items()
                    if now - b.last_access > idle_s]
            for k in idle:
                del self._buckets[k]
        if idle:
            logger.info("rate_limiter_sweep removed=%d active=%d",
                        len(idle), len(self._buckets))
        return len(idle)

    def size(self) -> int:
        with self._lock:
            return len(self._buckets)

    @staticmethod
    def _seconds_until_midnight() -> int:
        from datetime import datetime, timedelta

        now = datetime.now()
        midnight = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return int((midnight - now).total_seconds()) + 1
