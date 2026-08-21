"""答案缓存 — ChatCache (P3-A)

对「无会话 (session_id=None) 的相同问题」做短 TTL 缓存, 显著降低 Ollama 压力:
- key = sha256(question + kb_version): 知识库变更后旧缓存自动失效
- LRU + TTL, 线程安全 (多线程调用图 / 事件循环共用)
- 只缓存成功回答; 超时降级/异常不入缓存

🏭 Java 对标: Caffeine 短 TTL 缓存 + 写操作失效
"""

import hashlib
import logging
import threading
import time
from collections import OrderedDict
from typing import Any

logger = logging.getLogger(__name__)


class ChatCache:
    """进程内答案缓存 — LRU + TTL, 线程安全"""

    def __init__(self, ttl_s: float = 300.0, max_entries: int = 512):
        self._ttl_s = ttl_s
        self._max = max_entries
        self._data: "OrderedDict[str, tuple[float, dict]]" = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def key_for(question: str, kb_version: int, tenant_id: str = "default") -> str:
        """缓存键: 问题 + 知识库版本 + 租户 (版本/租户变化即不同 key)"""
        raw = f"{kb_version}|{tenant_id}|{question}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, key: str) -> dict | None:
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            ts, value = item
            if time.time() - ts > self._ttl_s:
                del self._data[key]
                return None
            self._data.move_to_end(key)  # LRU 刷新
            return value

    def set(self, key: str, value: dict) -> None:
        with self._lock:
            self._data[key] = (time.time(), value)
            self._data.move_to_end(key)
            while len(self._data) > self._max:
                self._data.popitem(last=False)

    def clear(self) -> int:
        with self._lock:
            n = len(self._data)
            self._data.clear()
            return n

    def size(self) -> int:
        with self._lock:
            return len(self._data)

    def stats(self) -> dict:
        """缓存统计 (命中率需外部累计)"""
        with self._lock:
            return {"size": len(self._data), "max": self._max, "ttl_s": self._ttl_s}


class CacheStats:
    """缓存命中统计 (进程内计数器)"""

    def __init__(self):
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def hit(self) -> None:
        with self._lock:
            self.hits += 1

    def miss(self) -> None:
        with self._lock:
            self.misses += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            total = self.hits + self.misses
            return {
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(self.hits / total, 4) if total else None,
            }
