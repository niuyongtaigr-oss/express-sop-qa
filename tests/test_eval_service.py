"""P2-A LLM-as-Judge 评测服务测试 (Stub 驱动, 无需 Ollama)"""

import json

from app.config import Settings
from app.services.eval_service import EvalJudgement, EvalService

# 内置评测集: 6 用例 (与 data/eval_cases.json 一致)
BUILTIN_N = 6


def make_settings(**overrides):
    base = dict(eval_judge=True, retrieval_mode="hybrid", chunk_size=200, top_k=3)
    base.update(overrides)
    return Settings(_env_file=None, **base)


class StubRag:
    def __init__(self, chunk_text="包裹破损需异常登记并拍照留存"):
        self._chunk_text = chunk_text
        self.retrieves: list = []
        self.generates: list = []

    def retrieve(self, query, top_k=None):
        self.retrieves.append(query)
        return [{
            "content": self._chunk_text,
            "metadata": {"doc_id": "sop", "title": "默认", "tags": "破损"},
            "distance": 0.1,
            "similarity": 0.9,
        }]

    def generate(self, query, chunks, history=None):
        self.generates.append(query)
        return "包裹破损请立即异常登记并拍照留存。"

    @property
    def indexed_chunks(self):
        return 2


class JudgeLLM:
    """structured_invoke 返回固定评分"""

    def __init__(self, judgement=None):
        self._j = judgement or EvalJudgement(faithfulness=0.8, completeness=0.7, reason="有依据, 部分遗漏")

    def structured_invoke(self, schema, messages):
        assert schema is EvalJudgement
        return self._j

    def invoke(self, messages):
        return "stub"

    async def astream(self, messages):
        yield "s"


def test_hit_rate_and_judge_scores():
    rag, llm = StubRag(), JudgeLLM()
    result = EvalService(rag, llm, make_settings(), record_history=False).run(top_k=3)
    assert result["top_k"] == 3
    assert result["judge"] is True
    assert result["judged_cases"] == BUILTIN_N
    assert result["faithfulness_avg"] == 0.8
    assert result["completeness_avg"] == 0.7
    assert result["hit_rate"] >= 0.0
    # 每条用例都有答案与评分
    assert all(d.get("answer") for d in result["details"])
    assert all(d.get("faithfulness") == 0.8 for d in result["details"])
    # 生成只走 generate, 不走 ask
    assert len(rag.generates) == BUILTIN_N
    # config 快照
    assert result["config"]["retrieval_mode"] == "hybrid"
    assert result["config"]["kb_chunks"] == 2


def test_no_judge_skips_generation():
    rag, llm = StubRag(), JudgeLLM()
    result = EvalService(rag, llm, make_settings(), record_history=False).run(top_k=3, judge=False)
    assert result["judge"] is False
    assert result["judged_cases"] == 0
    assert result["faithfulness_avg"] is None
    assert result["completeness_avg"] is None
    assert rag.generates == []  # 不生成答案
    assert all(d.get("answer") is None for d in result["details"])


def test_hit_rate_depends_on_keywords():
    # 检索片段包含 "遗失" 但不含 "500" → fine-1 用例不命中
    rag = StubRag(chunk_text="包裹遗失需24小时内完成找寻")
    result = EvalService(rag, JudgeLLM(), make_settings()).run(top_k=3, judge=False)
    by_id = {d["case_id"]: d for d in result["details"]}
    assert by_id["lost-1"]["hit"] is True      # 含 "遗失"
    assert by_id["fine-1"]["hit"] is False     # 不含 "500"
    assert 0 < result["hit_rate"] < 1.0


def test_judge_error_does_not_kill_run():
    class BrokenJudge(JudgeLLM):
        def structured_invoke(self, schema, messages):
            raise RuntimeError("judge down")

    rag, llm = StubRag(), BrokenJudge()
    result = EvalService(rag, llm, make_settings(), record_history=False).run(top_k=3)
    assert result["judged_cases"] == 0
    assert result["faithfulness_avg"] is None
    assert any("judge_error" in (d.get("reason") or "") for d in result["details"])
    assert result["hit_rate"] >= 0.0  # 检索部分不受影响


def test_cases_from_file(tmp_path):
    cases = [{
        "id": "custom-1",
        "question": "自定义问题?",
        "expect_keywords": ["破损"],
        "required_points": ["拍照"],
    }]
    f = tmp_path / "cases.json"
    f.write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")
    rag, llm = StubRag(), JudgeLLM()
    svc = EvalService(rag, llm, make_settings(eval_cases_path=str(f)), record_history=False)
    result = svc.run(top_k=3, judge=False)
    assert len(result["details"]) == 1
    assert result["details"][0]["case_id"] == "custom-1"


def test_cases_fallback_when_file_missing(tmp_path):
    missing = tmp_path / "nope.json"
    rag, llm = StubRag(), JudgeLLM()
    svc = EvalService(rag, llm, make_settings(eval_cases_path=str(missing)), record_history=False)
    result = svc.run(top_k=3, judge=False)
    assert len(result["details"]) == BUILTIN_N
