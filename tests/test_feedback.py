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


# ── 租户隔离 ─────────────────────────────────────────────
# 反馈里带用户原话 (question/answer), 与会话历史是同一类数据。会话已按租户隔离,
# 反馈不能是例外 —— 否则多租户下 A 能看到 B 的低分问题原话。

def test_stats_are_tenant_scoped(tmp_path):
    store = FeedbackStore(tmp_path / "fb.jsonl")
    store.append({"question": "A 的问题", "rating": 1}, tenant_id="net-001")
    store.append({"question": "B 的问题", "rating": 5}, tenant_id="net-002")
    store.append({"question": "B 的另一条", "rating": 5}, tenant_id="net-002")

    assert store.stats("net-001")["total"] == 1
    assert store.stats("net-002")["total"] == 2
    assert store.stats("net-003")["total"] == 0


def test_negative_questions_do_not_leak_across_tenants(tmp_path):
    store = FeedbackStore(tmp_path / "fb.jsonl")
    store.append({"question": "A 的隐私问题", "rating": 1}, tenant_id="net-001")
    store.append({"question": "B 的低分问题", "rating": 1}, tenant_id="net-002")

    b = store.negative_questions(10, "net-002")
    assert [n["question"] for n in b] == ["B 的低分问题"]
    assert store.negative_questions(10, "net-001")[0]["question"] == "A 的隐私问题"


def test_tenant_id_is_persisted(tmp_path):
    import json

    p = tmp_path / "fb.jsonl"
    FeedbackStore(p).append({"question": "q", "rating": 5}, tenant_id="net-001")
    line = json.loads(p.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert line["tenant_id"] == "net-001"
    assert FeedbackStore(p).stats("net-001")["total"] == 1


def test_legacy_entries_without_tenant_go_to_default_tenant(tmp_path):
    """本次改动之前的数据没有 tenant_id —— 那时只有单租户, 归 DEFAULT_TENANT"""
    import json

    p = tmp_path / "fb.jsonl"
    p.write_text(json.dumps({"question": "老数据", "rating": 1, "ts": 1.0}) + "\n",
                 encoding="utf-8")
    store = FeedbackStore(p)
    assert store.stats("default")["total"] == 1
    assert store.stats("net-001")["total"] == 0        # 真实租户看不到老数据
    assert store.negative_questions(10, "default")[0]["question"] == "老数据"


def test_append_without_tenant_defaults_to_default_tenant(tmp_path):
    store = FeedbackStore(tmp_path / "fb.jsonl")
    store.append({"question": "q", "rating": 4})
    assert store.stats("default")["total"] == 1


# ── 接口级: 租户上下文真的透传下去了 ─────────────────────
# 只测 FeedbackStore 的话, 端点忘了传 tenant_id 测试依然全绿 (参数有默认值)。

def _client(store, tenant: str):
    from fastapi import FastAPI, Request
    from fastapi.testclient import TestClient

    from app.api.deps import enforce_rate_limit, get_feedback_store, get_tenant_id
    from app.api.v1.endpoints.feedback import router as feedback_router
    from app.core.security import verify_api_key

    async def _fake_verify(request: Request) -> None:
        request.state.tenant_id = tenant          # 模拟 verify_api_key 的写入

    app = FastAPI()
    app.include_router(feedback_router, prefix="/api/v1")
    app.dependency_overrides[verify_api_key] = _fake_verify
    app.dependency_overrides[enforce_rate_limit] = lambda: None
    app.dependency_overrides[get_feedback_store] = lambda: store
    del get_tenant_id                              # 真实依赖, 不覆盖
    return TestClient(app)


def test_endpoint_passes_tenant_to_store(tmp_path):
    store = FeedbackStore(tmp_path / "fb.jsonl")
    client = _client(store, "net-001")

    assert client.post("/api/v1/chat/feedback",
                       json={"question": "破损怎么赔", "rating": 2}).status_code == 200
    # 真的写进了 net-001, 而不是默认租户
    assert store.stats("net-001")["total"] == 1
    assert store.stats("default")["total"] == 0


def test_endpoint_stats_are_tenant_scoped(tmp_path):
    store = FeedbackStore(tmp_path / "fb.jsonl")
    store.append({"question": "B 的问题", "rating": 1}, tenant_id="net-002")
    store.append({"question": "默认租户的记录", "rating": 3}, tenant_id="default")

    # 必须同时有"默认租户"的记录: 否则请求方即使丢掉 tenant_id 回退到默认租户,
    # 结果也照样是 0, 测试就变成了永远通过
    body = _client(store, "net-001").get("/api/v1/chat/feedback/stats").json()
    assert body["total"] == 0
    assert body["recent_negative"] == []           # 看不到 B 与默认租户的低分问题原话

    own = _client(store, "default").get("/api/v1/chat/feedback/stats").json()
    assert own["total"] == 1                       # 确实按当前租户查
