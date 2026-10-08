"""长期记忆评测的打分与汇总 (纯逻辑, 不调 LLM)

评测脚本本身也要被测: 它的打分错了, README 里那张表就是错的, 而且不会有人发现。
这里只测纯函数 —— 真实跑一遍要十分钟, 那是 `scripts/run_memory_eval.py` 的事。
"""

import json
from pathlib import Path

from scripts.run_memory_eval import aggregate, score_answer


# ── score_answer ─────────────────────────────────────────

def test_hits_when_expected_keyword_present():
    assert score_answer("您常驻上海。", ["上海"], [])[0] is True


def test_fails_when_no_expected_keyword_present():
    ok, why = score_answer("我不知道。", ["上海"], [])
    assert ok is False and "未命中" in why


def test_expect_any_is_or_not_and():
    """expect_any 是"至少一个", 不是"全都要" """
    assert score_answer("您负责华东。", ["华东", "上海"], [])[0] is True


def test_forbidden_keyword_beats_missing_keyword():
    """出现旧值比"没答出来"更严重 → 报错信息要指向前者

    新旧并存时答案里通常两个都有, 这时必须报"出现了不该出现的", 否则排查方向
    会被带偏成"召回没命中"。
    """
    ok, why = score_answer("您原来常驻上海，现在在杭州。", ["杭州"], ["上海"])
    assert ok is False
    assert "不该出现" in why and "上海" in why


def test_empty_expect_any_means_no_requirement():
    """只给 expect_none 的探针(隔离用例)不该因为"没命中关键词"而失败"""
    ok, _ = score_answer("我不知道您住在哪里。", [], ["上海"])
    assert ok is True


def test_isolation_probe_fails_on_leak():
    ok, why = score_answer("您常驻上海。", [], ["上海"])
    assert ok is False and "不该出现" in why


# ── aggregate ────────────────────────────────────────────

def _probes(*specs):
    """(id, kind, expect_any, expect_none, dimension)"""
    return [
        {"id": i, "kind": k, "user": "u", "expect_any": ea, "expect_none": en,
         "dimension": (s[4] if len(s) > 4 else None)}
        for s in specs for i, k, ea, en in [s[:4]]
    ]


def test_aggregate_counts_only_check_steps():
    probes = _probes(("a", "check", ["x"], [], "recall"), ("b", "turn", [], []),
                     ("c", "no_memory", [], []))
    results = [{"passed": True, "memories_before": 0, "memories_after": 0},
               {"passed": True, "memories_before": 0, "memories_after": 1},
               {"passed": True, "memories_before": 1, "memories_after": 2}]
    s = aggregate(probes, results)
    assert s["checks"] == 1 and s["recall_rate"] == 1.0
    assert s["false_memory_rate"] == 1.0     # 那一轮确实多记了一条


def test_aggregate_false_memory_rate_zero_when_nothing_stored():
    probes = _probes(("a", "no_memory", [], []), ("b", "no_memory", [], []))
    results = [{"passed": True, "memories_before": 0, "memories_after": 0},
               {"passed": True, "memories_before": 3, "memories_after": 3}]
    assert aggregate(probes, results)["false_memory_rate"] == 0.0


def test_aggregate_rates_are_none_when_not_measured():
    """**没测**和**测了得 0 分**是两件事

    分母为 0 时返回 None(打印成"未测"), 不能返回 0.0 —— 那会让人以为这个维度
    已经验证过且表现完美。
    """
    probes = _probes(("only-turn", "turn", [], []))
    results = [{"passed": True, "memories_before": 0, "memories_after": 1}]
    s = aggregate(probes, results)
    assert s["recall_rate"] is None
    assert s["isolation_rate"] is None
    assert s["conflict_rate"] is None
    assert s["false_memory_rate"] is None


def test_aggregate_separates_isolation_and_conflict():
    """隔离与时效要分开算: 一个是"别人的不许漏", 一个是"旧的必须被换掉" """
    probes = _probes(
        ("after-move-city", "check", ["杭州"], ["上海"], "conflict"),
        ("cross-user-leak", "check", [], ["上海", "华东"], "isolation"),
        ("recall-city", "check", ["上海"], [], "recall"),
    )
    results = [
        {"passed": True, "memories_before": 0, "memories_after": 0},   # 时效 OK
        {"passed": False, "memories_before": 0, "memories_after": 0},  # 泄漏了
        {"passed": True, "memories_before": 0, "memories_after": 0},
    ]
    s = aggregate(probes, results)
    # 每个维度各一条探针, 所以各算各的: 召回那条过了、泄漏那条没过、时效那条过了
    assert s["recall_rate"] == 1.0
    assert s["isolation_rate"] == 0.0
    assert s["conflict_rate"] == 1.0


# ── 用例文件本身 ──────────────────────────────────────────

def test_cases_file_is_well_formed():
    """用例文件的结构错误必须现在就被发现 —— 跑到一半才炸会浪费十分钟机器时间"""
    cases = json.loads(
        Path("data/memory_eval_cases.json").read_text(encoding="utf-8")
    )
    probes = cases["probes"]
    assert probes, "用例不能为空"
    ids = [p["id"] for p in probes]
    assert len(ids) == len(set(ids)), f"用例 id 重复: {ids}"

    for p in probes:
        assert p["kind"] in ("turn", "check", "no_memory"), p
        assert p["user"] and p["question"], p
        if p["kind"] == "check":
            assert p.get("expect_any") or p.get("expect_none"), \
                f"{p['id']}: check 至少要有一个断言, 否则它永远是绿的"
            assert p.get("dimension") in ("recall", "conflict", "isolation"), \
                f"{p['id']}: check 必须声明 dimension, 否则指标会把它漏掉"

    kinds = {p["kind"] for p in probes}
    assert {"turn", "check", "no_memory"} <= kinds, "四个维度都要有用例覆盖"


def test_cases_cover_all_four_metrics():
    """四个维度缺一个, README 那张表就会出现"未测" """
    probes = json.loads(
        Path("data/memory_eval_cases.json").read_text(encoding="utf-8")
    )["probes"]
    assert any(p["kind"] == "no_memory" for p in probes), "缺误记率用例"
    assert any(p.get("expect_none") for p in probes if p["kind"] == "check"), "缺隔离用例"
    assert any(p["id"].startswith("after-move") for p in probes), "缺冲突消解用例"
    assert any(p["kind"] == "check" and p.get("expect_any")
               and not p["id"].startswith("after-move") for p in probes), "缺召回用例"


# ── score_memory: 断言要打在存储上, 不是打在模型的措辞上 ──

def test_score_memory_detects_wrong_stored_content():
    """实测教训: 只看答案会被模型自己编的引用骗过去

    答案里出现「根据条款3」, 让 expect_any=["条款"] 白白通过; 而记忆库里那条真实
    偏好已经被"回答请友好简洁"(助手自己的承诺)覆盖了。所以必须断言存储内容。
    """
    from scripts.run_memory_eval import score_memory

    ok, why = score_memory(["回答请友好简洁"], ["条款号"], ["友好简洁"])
    assert ok is False and "不该有" in why


def test_score_memory_passes_on_correct_content():
    from scripts.run_memory_eval import score_memory

    ok, _ = score_memory(["用户要求回答必须带上条款号", "用户现居杭州"],
                         ["条款号"], ["友好简洁", "上海"])
    assert ok is True


def test_score_memory_expect_any_is_any_of():
    from scripts.run_memory_eval import score_memory

    assert score_memory(["用户负责浙江区域"], ["浙江", "江苏"], [])[0] is True
    assert score_memory(["用户负责福建区域"], ["浙江", "江苏"], [])[0] is False


def test_cases_declare_memory_assertions_for_stateful_checks():
    """会改变记忆状态的探针(偏好/冲突/隔离)必须断言存储内容

    只断言答案措辞的探针在"记忆被写坏"时依然可能是绿的 —— 那就成了假绿。
    """
    probes = json.loads(
        Path("data/memory_eval_cases.json").read_text(encoding="utf-8")
    )["probes"]
    by_id = {p["id"]: p for p in probes}
    for pid in ("recall-preference", "preference-survives-update",
                "after-move-city", "after-move-region", "cross-user-leak"):
        p = by_id[pid]
        assert p.get("expect_memory_any") or p.get("expect_memory_none"), \
            f"{pid}: 必须断言记忆内容, 否则可能是假绿"
