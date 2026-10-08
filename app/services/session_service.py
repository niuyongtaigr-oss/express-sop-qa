"""多轮会话记忆存储 — SessionStore

进程内内存态 (dict + OrderedDict LRU), 线程安全 (threading.Lock):

- 每个会话保存最近 N 轮 (user/assistant) 消息, 供编排图注入上下文
- **会话键是 (tenant_id, user_id, session_id) 三元组** —— session_id 由调用方提供,
  不与租户绑定的话, 多租户下 A 传 B 的 session_id 就能把 B 的对话历史塞进
  自己的提示词 (越权读取); 同理, 同一租户内不绑 user_id 就会用户之间互相命中。
  三元组把三者绑死。
- `user_id` 为空串表示"这次调用没有用户维度"(例如未启用访问控制的本地开发)。
  此时会话退化成租户级 —— 这是**退化**, 不是等价设计。
- 空闲 TTL 过期: 访问时惰性清理 + 后台 sweep() 定期清扫
- 上限保护: 超过 max_sessions 按 LRU 淘汰最久未访问的会话

⚠️ 两点边界, 需要部署方知晓:
  1. **内存态**: 单实例有效, 进程重启即失。多实例共享请换 Redis
     (SessionStore 接口不变, 换实现即可)。
  2. **用户维度取决于身份从哪来**: 带 user_id 时按三元组隔离; 若部署方式拿不到
     用户身份 (单租户 + 一把共享 key), user_id 为空, 隔离就退化成"只到租户"。
     要拿到可信的 user_id, 用多租户模式 (每用户一把 key, tenants.json 里带
     user_id 字段), 或未启用访问控制的本地开发用 X-User-Id 断言。
     session_id 仍必须由调用方生成成不可猜测的值 (建议 UUID4), 接口层强制 ≥16。

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

# 会话键: (tenant_id, user_id, session_id)
# 为什么必须带 user_id: 只按 (tenant, session) 索引时, 同一租户内两个用户只要
# session_id 相同就会共享历史 —— 原先靠"session_id 不可猜测"兜着。引入用户维度
# 之后这里必须一起升, 否则用户级隔离是假的。
SessionKey = tuple[str, str, str]

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
    def _key(tenant_id: str, user_id: str, session_id: str) -> SessionKey:
        """会话键 —— 用元组而非字符串拼接, 避免各段里的分隔符造成歧义"""
        return (tenant_id or _DEFAULT_TENANT, user_id or "", session_id)

    # ── 读 ───────────────────────────────────────────────
    def get_history(
        self,
        tenant_id: str,
        user_id: str,
        session_id: str | None,
        limit: int | None = None,
    ) -> list[Turn]:
        """返回该 (租户, 用户) 下该会话的历史 (**最旧在前**, 供提示词拼接); 空会话返回 []

        `user_id` 是**必填位置参数**, 不给默认值: 少传一个参数应该当场 TypeError,
        而不是静默退化成"租户级会话" —— 后者是跨用户串历史, 不会报错。
        """
        if not session_id:
            return []
        key = self._key(tenant_id, user_id, session_id)
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
        user_id: str,
        session_id: str | None,
        user_text: str,
        assistant_text: str,
    ) -> None:
        """记录一轮问答 (user + assistant)"""
        if not session_id:
            return
        key = self._key(tenant_id, user_id, session_id)
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

    def clear(self, tenant_id: str, user_id: str, session_id: str | None) -> bool:
        """清空该 (租户, 用户) 下的指定会话, 返回是否存在"""
        if not session_id:
            return False
        with self._lock:
            return self._sessions.pop(
                self._key(tenant_id, user_id, session_id), None
            ) is not None

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
