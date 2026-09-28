"""P2-C 用户反馈闭环 — FeedbackStore 单元测试"""

import logging

from app.services.feedback_service import FeedbackStore


def test_append_and_stats(tmp_path):
    store = FeedbackStore(tmp_path / "fb.jsonl")
    assert store.append({"question": "q1", "rating": 5}) == 1
    assert store.append({"question": "q2", "rating": 1, "comment": "答错了"}) == 2
    assert store.append({"question": "q3", "rating": 3}) == 3
    stats = store.stats()
    assert stats["total"] == 3
    assert stats["avg_rating"] == 3.0
    assert stats["negative_count"] == 1
    assert stats["negative_rate"] == round(1 / 3, 4)


def test_negative_questions_returns_recent(tmp_path):
    store = FeedbackStore(tmp_path / "fb.jsonl")
    for i in range(5):
        store.append({"question": f"坏问题{i}", "rating": 1 if i % 2 else 4})
    neg = store.negative_questions(limit=2)
    assert [n["question"] for n in neg] == ["坏问题3", "坏问题1"]  # 最近优先, 限 2 条


def test_persists_across_instances(tmp_path):
    p = tmp_path / "fb.jsonl"
    FeedbackStore(p).append({"question": "q", "rating": 2})
    store2 = FeedbackStore(p)
    assert store2.stats()["total"] == 1
    assert store2.stats()["negative_count"] == 1


def test_negative_feedback_logs_warning(tmp_path, caplog):
    store = FeedbackStore(tmp_path / "fb.jsonl")
    with caplog.at_level(logging.WARNING):
        store.append({"question": "破损怎么办", "rating": 1, "comment": "不对"})
    assert any("feedback_negative" in r.message for r in caplog.records)
    assert any("破损怎么办" in r.message for r in caplog.records)


def test_empty_stats(tmp_path):
    stats = FeedbackStore(tmp_path / "fb.jsonl").stats()
    assert stats == {"total": 0, "avg_rating": None,
                     "negative_count": 0, "negative_rate": 0.0}
    assert FeedbackStore(tmp_path / "fb.jsonl").negative_questions() == []


def test_feedback_schema_validation():
    from pydantic import ValidationError

    from app.schemas.feedback import FeedbackRequest

    ok = FeedbackRequest(question="q", rating=3)
    assert ok.rating == 3
    try:
        FeedbackRequest(question="q", rating=6)  # 越界
        raise AssertionError("应校验失败")
    except ValidationError:
        pass


def test_file_growth_is_bounded_by_compaction(tmp_path):
    """JSONL 原本只追加不回收 —— 内存有上限 (max_entries), 磁盘没有。

    每累计 max_entries 次写入压缩一次: 文件重写为当前保留的条目。
    """
    p = tmp_path / "fb.jsonl"
    store = FeedbackStore(p, max_entries=10)
    for i in range(25):
        store.append({"question": f"q{i}", "rating": 5})

    lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) < 25          # 从不压缩的话这里就是 25
    assert len(lines) <= 20         # 上界 = max_entries + 一轮未压缩的写入
    assert store.stats()["total"] == 10


def test_compacted_file_is_still_valid_jsonl(tmp_path):
    """压缩后必须是合法 JSONL, 且保留的是**最近**的条目, 且无残留 .tmp"""
    import json

    p = tmp_path / "fb.jsonl"
    store = FeedbackStore(p, max_entries=5)
    for i in range(12):
        store.append({"question": f"q{i}", "rating": 5})

    entries = [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert entries[-1]["question"] == "q11"          # 最新的还在
    assert all(set(e) >= {"question", "rating", "ts"} for e in entries)
    assert not (tmp_path / "fb.jsonl.tmp").exists()  # 原子替换, 不留半截文件

    # 压缩后仍能被新实例读回
    assert FeedbackStore(p, max_entries=5).stats()["total"] == 5
