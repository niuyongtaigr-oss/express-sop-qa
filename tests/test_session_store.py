"""P0.1 会话记忆存储 — SessionStore 单元测试

会话键是 **(tenant_id, user_id, session_id) 三元组**: session_id 由调用方提供,
不绑租户则 A 能读 B 的会话; 同一租户内不绑 user_id 则同租户的两个用户会互相命中。
两层都要绑死。
"""

import time

from app.services.session_service import SessionStore

T = "tenant-a"
U = "user-1"          # 用户维度: 会话键是 (tenant, user, session)
T2 = "tenant-b"
S = "session-abcdefgh"


def test_add_and_get_history_roundtrip():
    s = SessionStore(max_turns=10, max_sessions=100, ttl_s=60)
    s.add_turn(T, U, S, "你好", "你好，有什么可以帮你？")
    s.add_turn(T, U, S, "包裹破损了", "请先拍照留存，然后联系网点")
    hist = s.get_history(T, U, S)
    assert hist == [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好，有什么可以帮你？"},
        {"role": "user", "content": "包裹破损了"},
        {"role": "assistant", "content": "请先拍照留存，然后联系网点"},
    ]


def test_none_session_id_is_noop():
    s = SessionStore()
    s.add_turn(T, U, None, "q", "a")
    assert s.get_history(T, U, None) == []
    assert s.count() == 0


def test_max_turns_trims_oldest():
    s = SessionStore(max_turns=2, max_sessions=10, ttl_s=60)
    s.add_turn(T, U, S, "q1", "a1")
    s.add_turn(T, U, S, "q2", "a2")
    s.add_turn(T, U, S, "q3", "a3")  # 超限 → 丢最旧一轮
    hist = s.get_history(T, U, S)
    assert hist == [
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "q3"},
        {"role": "assistant", "content": "a3"},
    ]


def test_ttl_sweep_expires_idle_sessions():
    s = SessionStore(max_turns=10, max_sessions=100, ttl_s=10)
    s.add_turn(T, U, S, "q", "a")
    assert s.count() == 1
    # 模拟时间流逝: 该会话最后一次访问在 20s 前
    s._sessions[(T, U, S)].last_access = time.time() - 20   # 键是三元组
    removed = s.sweep()
    assert removed == 1
    assert s.count() == 0
    assert s.get_history(T, U, S) == []


def test_max_sessions_lru_eviction():
    s = SessionStore(max_turns=10, max_sessions=2, ttl_s=1000)
    s.add_turn(T, U, "a" * 16, "q", "a")
    s.add_turn(T, U, "b" * 16, "q", "a")
    # 访问 a 刷新其 LRU 位置
    s.get_history(T, U, "a" * 16)
    s.add_turn(T, U, "c" * 16, "q", "a")  # 超限 → 淘汰最久未访问的 b
    assert s.get_history(T, U, "a" * 16) != []
    assert s.get_history(T, U, "b" * 16) == []
    assert s.get_history(T, U, "c" * 16) != []


def test_clear_removes_session():
    s = SessionStore()
    s.add_turn(T, U, S, "q", "a")
    assert s.clear(T, U, S) is True
    assert s.clear(T, U, S) is False
    assert s.count() == 0


# ── 租户隔离 (安全回归) ──────────────────────────────────
# 修复前: 会话只按 session_id 索引, 多租户下 tenant-b 传 tenant-a 的 session_id
# 就能把对方的对话历史读出来, 并塞进自己的提示词。

def test_same_session_id_isolated_between_tenants():
    """同一 session_id 在不同租户下必须是两份互不可见的历史"""
    s = SessionStore()
    s.add_turn(T, U, S, "我是租户A", "收到A")
    s.add_turn(T2, U, S, "我是租户B", "收到B")

    assert [t["content"] for t in s.get_history(T, U, S)] == ["我是租户A", "收到A"]
    assert [t["content"] for t in s.get_history(T2, U, S)] == ["我是租户B", "收到B"]
    assert s.count() == 2          # 两条独立会话, 不是同一条被覆盖


def test_unknown_tenant_cannot_read_another_tenants_session():
    """B 租户拿 A 的 session_id 查询 → 空 (而不是读到 A 的历史)"""
    s = SessionStore()
    s.add_turn(T, U, S, "租户A 的机密问题", "租户A 的回答")
    assert s.get_history(T2, U, S) == []
    assert s.clear(T2, U, S) is False          # 也不能删掉对方的会话
    assert s.get_history(T, U, S) != []        # A 的会话完好


def test_tenant_isolation_survives_lru_eviction():
    """LRU 淘汰按 (租户, 会话) 计数, 不会因另一个租户活跃而误伤"""
    s = SessionStore(max_turns=10, max_sessions=2, ttl_s=1000)
    s.add_turn(T, U, S, "q1", "a1")
    s.add_turn(T2, U, S, "q2", "a2")
    assert s.count() == 2
    assert s.get_history(T, U, S) != [] and s.get_history(T2, U, S) != []
