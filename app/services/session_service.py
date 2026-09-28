"""多轮会话记忆存储 — SessionStore

进程内内存态 (dict + OrderedDict LRU), 线程安全 (threading.Lock):

- 每个会话保存最近 N 轮 (user/assistant) 消息, 供编排图注入上下文
- **会话键是 (tenant_id, session_id) 复合键** —— session_id 由调用方提供,
  不与租户绑定的话, 多租户下 A 传 B 的 session_id 就能把 B 的对话历史塞进
  自己的提示词 (越权读取)。复合键把两者绑死, 租户之间无法互相命中。
- 空闲 TTL 过期: 访问时惰性清理 + 后台 sweep() 定期清扫
- 上限保护: 超过 max_sessions 按 LRU 淘汰最久未访问的会话

⚠️ 两点边界, 需要部署方知晓:
  1. **内存态**: 单实例有效, 进程重启即失。多实例共享请换 Redis
     (SessionStore 接口不变, 换实现即可)。
  2. **只隔离到租户, 不隔离到最终用户**: 单租户模式下所有调用方同属
     default 租户, 谁能给出同一个 session_id 谁就能读到该会话的历史。
     所以 session_id **必须由调用方生成成不可猜测的值** (建议 UUID4),
     接口层已强制长度 ≥16 (见 schemas/chat.py)。若需要用户级隔离,
     得先引入用户身份概念 —— 当前 API 没有。

🏭 Java 对标: 会话级缓存 (Caffeine LRU + TTL), 键含租户维度
"""

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# 会话内一条消息: role=user/assistant, content=文本
Turn = dict

# 会话键: (tenant_id, session_id)
SessionKey = tuple[str, str]

_DEFAULT_TENANT = "default"


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
    """多会话存储 — 按租户隔离, LRU + TTL, 线程安全"""

    def __init__(
        self,
        ttl_s: float = 1800.0,
        max_turns: int = 10,
        max_sessions: int = 2000,
    ):
        self._ttl_s = ttl_s
        self._max_turns = max_turns
        self._max_sessions = max_sessions
        self._sessions: "OrderedDict[SessionKey, _Session]" = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def _key(tenant_id: str, session_id: str) -> SessionKey:
        """会话键 —— 用元组而非字符串拼接, 避免 tenant/session 里的分隔符造成歧义"""
        return (tenant_id or _DEFAULT_TENANT, session_id)

    # ── 读 ───────────────────────────────────────────────
    def get_history(
        self, tenant_id: str, session_id: str | None, limit: int | None = None
    ) -> list[Turn]:
        """返回该租户下该会话的历史消息 (**最旧在前**, 供提示词拼接); 空会话返回 []"""
        if not session_id:
            return []
        key = self._key(tenant_id, session_id)
        with self._lock:
            sess = self._sessions.get(key)
            if sess is None:
                return []
            sess.touch()
            self._sessions.move_to_end(key)  # LRU: 访问即刷新
            turns = sess.turns
            if limit is not None and limit > 0:
                turns = turns[-limit:]
            return list(turns)

    # ── 写 ───────────────────────────────────────────────
    def add_turn(
        self,
        tenant_id: str,
        session_id: str | None,
        user_text: str,
        assistant_text: str,
    ) -> None:
        """记录一轮问答 (user + assistant)"""
        if not session_id:
            return
        key = self._key(tenant_id, session_id)
        with self._lock:
            sess = self._sessions.get(key)
            if sess is None:
                sess = _Session()
                self._sessions[key] = sess
            sess.turns.append(_new_turn("user", user_text))
            sess.turns.append(_new_turn("assistant", assistant_text))
            # 只保留最近 max_turns 轮
            if len(sess.turns) > self._max_turns * 2:
                sess.turns = sess.turns[-(self._max_turns * 2):]
            sess.touch()
            self._sessions.move_to_end(key)
            # 写入时同步做 LRU 淘汰 (sweep 兜底, 防止两次清理间无界增长)
            while len(self._sessions) > self._max_sessions:
                self._sessions.popitem(last=False)

    def clear(self, tenant_id: str, session_id: str | None) -> bool:
        """清空指定租户下的会话, 返回是否存在"""
        if not session_id:
            return False
        with self._lock:
            return self._sessions.pop(self._key(tenant_id, session_id), None) is not None

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
                key for key, s in self._sessions.items()
                if now - s.last_access > self._ttl_s
            ]
            for key in expired:
                del self._sessions[key]
                removed += 1
            while len(self._sessions) > self._max_sessions:
                # OrderedDict 迭代顺序 = 插入序, 头部最旧
                self._sessions.popitem(last=False)
                removed += 1
            active = len(self._sessions)
        if removed:
            logger.info("session_sweep removed=%d active=%d", removed, active)
        return removed
