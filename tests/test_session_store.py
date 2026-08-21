"""P0.1 会话记忆存储 — SessionStore 单元测试"""

import time

from app.services.session_service import SessionStore


def test_add_and_get_history_roundtrip():
    s = SessionStore(max_turns=10, max_sessions=100, ttl_s=60)
    s.add_turn("s1", "你好", "你好，有什么可以帮你？")
    s.add_turn("s1", "包裹破损了", "请先拍照留存，然后联系网点")
    hist = s.get_history("s1")
    assert hist == [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好，有什么可以帮你？"},
        {"role": "user", "content": "包裹破损了"},
        {"role": "assistant", "content": "请先拍照留存，然后联系网点"},
    ]


def test_none_session_id_is_noop():
    s = SessionStore()
    s.add_turn(None, "q", "a")
    assert s.get_history(None) == []
    assert s.count() == 0


def test_max_turns_trims_oldest():
    s = SessionStore(max_turns=2, max_sessions=10, ttl_s=60)
    s.add_turn("s1", "q1", "a1")
    s.add_turn("s1", "q2", "a2")
    s.add_turn("s1", "q3", "a3")  # 超限 → 丢最旧一轮
    hist = s.get_history("s1")
    assert hist == [
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "q3"},
        {"role": "assistant", "content": "a3"},
    ]


def test_ttl_sweep_expires_idle_sessions():
    s = SessionStore(max_turns=10, max_sessions=100, ttl_s=10)
    s.add_turn("s1", "q", "a")
    assert s.count() == 1
    # 模拟时间流逝: s1 最后一次访问在 20s 前
    s._sessions["s1"].last_access = time.time() - 20
    removed = s.sweep()
    assert removed == 1
    assert s.count() == 0
    assert s.get_history("s1") == []


def test_max_sessions_lru_eviction():
    s = SessionStore(max_turns=10, max_sessions=2, ttl_s=1000)
    s.add_turn("a", "q", "a")
    s.add_turn("b", "q", "a")
    # 访问 a 刷新其 LRU 位置
    s.get_history("a")
    s.add_turn("c", "q", "a")  # 超限 → 淘汰最久未访问的 b
    assert s.get_history("a") != []
    assert s.get_history("b") == []
    assert s.get_history("c") != []


def test_clear_removes_session():
    s = SessionStore()
    s.add_turn("s1", "q", "a")
    assert s.clear("s1") is True
    assert s.clear("s1") is False
    assert s.count() == 0
