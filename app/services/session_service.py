"""多轮会话记忆存储 — SessionStore

进程内内存态 (dict + OrderedDict LRU), 线程安全 (threading.Lock):

- 每个 session_id 保存最近 N 轮 (user/assistant) 消息, 供编排图注入上下文
- 空闲 TTL 过期: 访问时惰性清理 + 后台 sweep() 定期清扫
- 上限保护: 超过 max_sessions 按 LRU 淘汰最久未访问的会话

⚠️ 内存态限制: 单实例有效, 进程重启即失。多实例共享请换 Redis
   (SessionStore 接口不变, 换实现即可)。

🏭 Java 对标: 会话级缓存 (Caffeine LRU + TTL)
"""

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# 会话内一条消息: role=user/assistant, content=文本
Turn = dict


def _new_turn(role: str, content: str) -> Turn:
    return {"role": role, "content": content}


@dataclass
class _Session:
    """单个会话: 消息列表 + 最近访问时间"""

    turns: list[Turn] = field(default_factory=list)
    last_access: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.last_access = time.time()


class SessionStore:
    """多会话存储 — LRU + TTL, 线程安全"""

    def __init__(
        self,
        ttl_s: float = 1800.0,
        max_turns: int = 10,
        max_sessions: int = 2000,
    ):
        self._ttl_s = ttl_s
        self._max_turns = max_turns
        self._max_sessions = max_sessions
        self._sessions: "OrderedDict[str, _Session]" = OrderedDict()
        self._lock = threading.Lock()

    # ── 读 ───────────────────────────────────────────────
    def get_history(self, session_id: str | None, limit: int | None = None) -> list[Turn]:
        """返回该会话的历史消息 (最新的在前), 空会话返回 []"""
        if not session_id:
            return []
        with self._lock:
            sess = self._sessions.get(session_id)
            if sess is None:
                return []
            sess.touch()
            self._sessions.move_to_end(session_id)  # LRU: 访问即刷新
            turns = sess.turns
            if limit is not None and limit > 0:
                turns = turns[-limit:]
            # 保持 user→assistant 顺序返回 (最旧在前), 供提示词拼接
            return list(turns)

    # ── 写 ───────────────────────────────────────────────
    def add_turn(self, session_id: str | None, user_text: str, assistant_text: str) -> None:
        """记录一轮问答 (user + assistant)"""
        if not session_id:
            return
        with self._lock:
            sess = self._sessions.get(session_id)
            if sess is None:
                sess = _Session()
                self._sessions[session_id] = sess
            sess.turns.append(_new_turn("user", user_text))
            sess.turns.append(_new_turn("assistant", assistant_text))
            # 只保留最近 max_turns 轮
            if len(sess.turns) > self._max_turns * 2:
                sess.turns = sess.turns[-(self._max_turns * 2):]
            sess.touch()
            self._sessions.move_to_end(session_id)
            # 写入时同步做 LRU 淘汰 (sweep 兜底, 防止两次清理间无界增长)
            while len(self._sessions) > self._max_sessions:
                self._sessions.popitem(last=False)

    def clear(self, session_id: str | None) -> bool:
        """清空指定会话, 返回是否存在"""
        if not session_id:
            return False
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    # ── 维护 ─────────────────────────────────────────────
    def count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def sweep(self, now: float | None = None) -> int:
        """清理: 1) TTL 过期会话 2) 超出 max_sessions 的 LRU 尾部。
        返回清理的会话数。"""
        now = now or time.time()
        removed = 0
        with self._lock:
            expired = [
                sid for sid, s in self._sessions.items()
                if now - s.last_access > self._ttl_s
            ]
            for sid in expired:
                del self._sessions[sid]
                removed += 1
            while len(self._sessions) > self._max_sessions:
                # OrderedDict 迭代顺序 = 插入序, 头部最旧
                self._sessions.popitem(last=False)
                removed += 1
        if removed:
            logger.info("session_sweep removed=%d active=%d", removed, len(self._sessions))
        return removed
