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
from datetime import date
from itertools import islice
from threading import Lock

logger = logging.getLogger(__name__)

# 需要腾位置时最多扫描多少个桶找"已回满"的 (限制单次开销, 见 _make_room)
_EVICT_SCAN = 64

# 当日用量记账表的容量 = max_keys 的倍数 (每条只有 [日期, 次数] 两个字段)
_DAILY_MULTIPLIER = 4

# 告警里报给运维的配置项名 (对应 Settings.rate_limit_max_keys)
DAILY_MAX_HINT = "SOP_QA_RATE_LIMIT_MAX_KEYS"


@dataclass
class _Bucket:
    """令牌桶 —— **只放令牌状态**; 当日配额在 RateLimiter._daily (淘汰语义不同)"""

    tokens: float
    capacity: float
    updated: float = field(default_factory=time.monotonic)
    last_access: float = field(default_factory=time.monotonic)


class RateLimiter:
    """按 Key 的令牌桶 + 每日配额

    两份状态**分开存放**, 因为它们的淘汰语义完全相反:

      · `_buckets` (令牌桶) 可以丢 —— 前提是丢的时候桶已经回满: 对方下次访问
        本来就会拿到一个满桶, 丢掉等于什么都没丢。反过来, 淘汰一个还没回满的桶
        就是把限流白送回去。所以表满时只淘汰"已回满"的桶, 扫不到就拒绝新 Key
        (fail-closed), 新 Key 在 LRU 桶回满后自动恢复 (通常是秒级, 自愈)。
      · `_daily` (当日用量) **不能丢** —— 丢了就是配额重置。一个 Key 只要刷够
        别的 Key 把自己的记录挤出表, 就能无限刷下去。因此只在跨天时清理; 若表满
        且全是今天的记录, 拒绝新 Key 并告警 (调大 rate_limit_max_keys)。
    """

    def __init__(
        self,
        enabled: bool = False,
        per_min: int = 60,
        burst: int = 20,
        daily_quota: int = 1000,
        max_keys: int = 10000,
    ):
        if per_min <= 0:
            # 为 0 时令牌永不补充, 且算 Retry-After 时会除零 (500 而不是 429)
            raise ValueError(
                f"per_min 必须 > 0 (收到 {per_min}): 为 0 时令牌永不补充, "
                "且计算 Retry-After 会除零崩溃"
            )
        if max_keys < 1:
            raise ValueError(f"max_keys 必须 >= 1 (收到 {max_keys})")
        self._enabled = enabled
        self._rate = per_min / 60.0  # 每秒补充令牌数
        self._capacity = float(burst)
        self._daily_quota = daily_quota
        self._max_keys = max_keys
        # 用量记账每条只有两个字段, 所以容量可以比令牌桶宽裕一些
        self._daily_max = max_keys * _DAILY_MULTIPLIER
        self._buckets: "OrderedDict[str, _Bucket]" = OrderedDict()
        self._daily: dict[str, list] = {}  # key -> [date, used]
        # 当日记账表已确认占满 (避免每次新 Key 都全表扫一遍找跨天记录)
        self._daily_saturated_day: date | None = None
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
        now = time.monotonic()
        today = date.today()
        with self._lock:
            # 1) 令牌桶 —— **先查令牌, 再碰当日用量表**。顺序很关键: 被限流拒绝的
            #    请求不配占一个当日用量名额。反过来写的话, 一个灌满令牌表的新 Key
            #    会先占掉记账位再被拒, 于是记账表被这些"从没被服务过"的 Key 填满,
            #    一旦填满 `_daily_saturated_day` 会在当天一直生效 —— 就算令牌桶早
            #    已回满可淘汰, 所有全新 Key 也会被封到次日零点。那是把令牌表的
            #    "秒级自愈"放大成"当天不可用"。
            bucket = self._buckets.get(key)
            if bucket is None:
                if len(self._buckets) >= self._max_keys:
                    wait = self._make_room(now)
                    if wait is not None:
                        return False, wait, None
                bucket = _Bucket(tokens=self._capacity, capacity=self._capacity,
                                 updated=now, last_access=now)
                self._buckets[key] = bucket
            bucket.last_access = now
            self._buckets.move_to_end(key)

            bucket.tokens = min(
                bucket.capacity,
                bucket.tokens + (now - bucket.updated) * self._rate,
            )
            bucket.updated = now

            if bucket.tokens < 1.0:
                retry = math.ceil((1.0 - bucket.tokens) / self._rate)
                return False, max(retry, 1), None

            # 2) 到这里说明这个请求**有令牌可用**, 它才值得占一个当日用量名额。
            #    记不下就不能放行 —— 放行等于绕过配额。
            if key not in self._daily:
                if not self._ensure_daily_room(today):
                    logger.warning(
                        "限流: 当日用量表已满 (%d 个被服务过的 Key 且无跨天记录可清), "
                        "拒绝新 Key —— 请调大 %s", self._daily_max, DAILY_MAX_HINT,
                    )
                    return False, float(max(self._seconds_until_midnight(), 1)), 0
                self._daily[key] = [today, 0]
            rec = self._daily[key]
            if rec[0] != today:  # 跨天重置
                rec[0], rec[1] = today, 0

            # 3) 每日配额
            if rec[1] >= self._daily_quota:
                return False, max(self._seconds_until_midnight(), 1), 0

            bucket.tokens -= 1.0
            rec[1] += 1
            return True, None, self._daily_quota - rec[1]

    # ── 内部 (调用方需持锁) ───────────────────────────────
    def _ensure_daily_room(self, today: date) -> bool:
        """确认当日用量表还能再记一个 Key。

        只有跨天的记录可以丢 —— 它们本来就会在下次被访问时归零。今天的记录一条
        都不能丢: 丢哪条就是给哪个 Key 重置配额。
        """
        if len(self._daily) < self._daily_max:
            return True
        if self._daily_saturated_day == today:
            return False  # 今天已确认占满, 不再每次全表扫
        stale = [k for k, (day, _) in self._daily.items() if day != today]
        for k in stale:
            del self._daily[k]
        if len(self._daily) < self._daily_max:
            return True
        self._daily_saturated_day = today
        return False

    def _make_room(self, now: float) -> float | None:
        """令牌桶表满, 尝试为新 Key 腾位置。

        返回 None = 已腾出; 返回秒数 = 没腾出来, 建议对方这么久后再试。

        只淘汰**已经回满**的桶 (见类 docstring)。扫描从最久未访问的一端开始并
        限定条数: 需要腾位置时表通常是满的, 全表扫描会让每个新 Key 都变成 O(n)。
        """
        waits: list[float] = []
        for k, b in islice(self._buckets.items(), _EVICT_SCAN):
            if b.tokens + (now - b.updated) * self._rate >= b.capacity:
                del self._buckets[k]
                return None
            waits.append((b.capacity - b.tokens) / self._rate - (now - b.updated))
        # 扫不到就保守取最小值: 最久未访问的桶也会是最早回满的那个
        return float(max(min(waits), 1.0)) if waits else 1.0

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
