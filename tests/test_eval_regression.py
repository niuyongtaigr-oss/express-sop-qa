"""P2-B 评测回归与对比 — EvalHistory + compare diff"""

import json

from app.services.eval_service import EvalHistory, EvalService

from test_eval_service import JudgeLLM, StubRag, make_settings


# ── EvalHistory ──────────────────────────────────────────
def test_history_append_and_latest_before(tmp_path):
    h = EvalHistory(tmp_path / "hist.jsonl", max_entries=5)
    h.append({"ts": 1.0, "hit_rate": 0.5, "judge": True})
    h.append({"ts": 2.0, "hit_rate": 0.8, "judge": True})
    assert h.count() == 2
    prev = h.latest_before(2.5, judge=True)
    assert prev["hit_rate"] == 0.8
    assert h.latest_before(1.5, judge=True)["hit_rate"] == 0.5
    assert h.latest_before(0.5, judge=True) is None
    # judge 开关隔离
    assert h.latest_before(2.5, judge=False) is None


def test_history_persists_across_instances(tmp_path):
    p = tmp_path / "hist.jsonl"
    EvalHistory(p).append({"ts": 1.0, "hit_rate": 0.5})
    h2 = EvalHistory(p)
    assert h2.count() == 1
    assert h2.latest_before(2.0)["hit_rate"] == 0.5


def test_history_corrupt_line_ignored(tmp_path):
    p = tmp_path / "hist.jsonl"
    p.write_text('{"ts": 1.0, "hit_rate": 0.5}\nnot-json\n{"ts": 2.0, "hit_rate": 0.7}\n',
                 encoding="utf-8")
    h = EvalHistory(p)
    assert h.count() == 2


# ── 回归对比 (compare) ───────────────────────────────────
def test_first_run_no_compare_then_delta(tmp_path):
    settings = make_settings(eval_history_path=str(tmp_path / "hist.jsonl"))
    rag, llm = StubRag(), JudgeLLM()
    svc = EvalService(rag, llm, settings)

    r1 = svc.run(top_k=3)
    assert r1["compare"] is None  # 首次无基准

    # 第二次: 检索片段变化 → hit_rate 变化 → 有 delta
    rag2, llm2 = StubRag(chunk_text="与期望无关的内容"), JudgeLLM()
    svc2 = EvalService(rag2, llm2, settings)
    r2 = svc2.run(top_k=3)
    comp = r2["compare"]
    assert comp is not None
    assert "vs_ts" in comp and "hit_rate_delta" in comp
    assert comp["hit_rate_delta"] == round(r2["hit_rate"] - r1["hit_rate"], 4)


def test_compare_isolated_by_judge_flag(tmp_path):
    settings = make_settings(eval_history_path=str(tmp_path / "hist.jsonl"))
    rag, llm = StubRag(), JudgeLLM()
    svc = EvalService(rag, llm, settings)
    svc.run(top_k=3)  # judge=True 写入历史
    # judge=False 的评测不应拿 judge=True 的结果做基准
    r2 = svc.run(top_k=3, judge=False)
    assert r2["compare"] is None


def test_history_max_entries(tmp_path):
    p = tmp_path / "hist.jsonl"
    h = EvalHistory(p, max_entries=3)
    for i in range(6):
        h.append({"ts": float(i), "hit_rate": i / 10})
    assert h.count() == 3
    assert h.latest_before(99.0)["ts"] == 5.0  # 最新一条保留


def test_compare_visible_in_task_payload(tmp_path):
    """compare 字段应出现在 /eval/tasks 响应结构 (Schema 层)"""
    from app.schemas.eval import EvalTaskResponse

    payload = {
        "task_id": "abc", "status": "done", "hit_rate": 0.8,
        "compare": {"vs_ts": 123.0, "hit_rate_delta": 0.1},
        "details": [],
    }
    resp = EvalTaskResponse(**payload)
    assert resp.compare["hit_rate_delta"] == 0.1
