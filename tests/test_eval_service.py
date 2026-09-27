"""P2-A LLM-as-Judge 评测服务测试 (Stub 驱动, 无需 Ollama)"""

import json

from app.config import Settings
from app.services.eval_service import EvalJudgement, EvalService

# 内置评测集: 6 用例 (与 data/eval_cases.json 一致)
# 其中 5 条检索用例 (参与 hit_rate) + 1 条拒答用例 (expect_refusal)
BUILTIN_N = 6
BUILTIN_RETRIEVAL_N = 5
BUILTIN_REFUSAL_N = 1


def make_settings(**overrides):
    base = dict(eval_judge=True, retrieval_mode="hybrid", chunk_size=200, top_k=3)
    base.update(overrides)
    return Settings(_env_file=None, **base)


class StubRag:
    def __init__(
        self,
        chunk_text="包裹破损需异常登记并拍照留存",
        answer="包裹破损请立即异常登记并拍照留存。",
    ):
        self._chunk_text = chunk_text
        self._answer = answer
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
        return self._answer

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
    # 拒答用例不走 LLM 打分 (required_points 为空, 评分无意义), 只判「是否如实拒答」
    assert result["judged_cases"] == BUILTIN_RETRIEVAL_N
    assert result["refusal_total"] == BUILTIN_REFUSAL_N
    assert result["refusal_checked"] == BUILTIN_REFUSAL_N
    assert result["faithfulness_avg"] == 0.8
    assert result["completeness_avg"] == 0.7
    assert result["hit_rate"] >= 0.0
    # 每条用例都有答案 (含拒答用例)
    assert all(d.get("answer") for d in result["details"])
    # 只有检索用例带 faithfulness 评分
    scored = [d for d in result["details"] if d["kind"] == "retrieval"]
    assert len(scored) == BUILTIN_RETRIEVAL_N
    assert all(d["faithfulness"] == 0.8 for d in scored)
    # 默认 stub 回答不含拒答标记 → 拒答用例判为「未拒答」(即编造)
    assert result["refusal_accuracy"] == 0.0
    # 生成只走 generate, 不走 ask; 检索用例 + 拒答用例各生成一次
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


# ── 拒答用例 (expect_refusal) ────────────────────────────
# 背景: 知识库中确实没有答案的问题, 系统应如实拒答而非编造。
# 这是企业知识库的底线指标 —— 宁可说「不知道」, 不能一本正经地胡说。


def test_refusal_passes_when_model_refuses():
    """如实拒答 → refusal_accuracy = 1.0"""
    rag = StubRag(answer="根据现有知识库无法回答此问题。")
    result = EvalService(rag, JudgeLLM(), make_settings(), record_history=False).run(top_k=3)
    assert result["refusal_accuracy"] == 1.0
    refusals = [d for d in result["details"] if d["kind"] == "refusal"]
    assert len(refusals) == BUILTIN_REFUSAL_N
    assert refusals[0]["refused"] is True


def test_refusal_fails_when_model_hallucinates():
    """知识库无答案却硬答 → refusal_accuracy = 0.0"""
    rag = StubRag(answer="请拨打客服热线 400-000-0000 申请特殊处理。")
    result = EvalService(rag, JudgeLLM(), make_settings(), record_history=False).run(top_k=3)
    assert result["refusal_accuracy"] == 0.0
    refusals = [d for d in result["details"] if d["kind"] == "refusal"]
    assert refusals[0]["refused"] is False


def test_refusal_excluded_from_hit_rate():
    """拒答用例不参与 hit_rate: 分母只算检索用例, 不会被无解的用例拉低"""
    # 检索片段覆盖全部 5 条检索用例的关键词
    rag = StubRag(chunk_text="破损 异常登记 理赔 500 遗失 24小时 拍照留存 省区审核")
    result = EvalService(rag, JudgeLLM(), make_settings(), record_history=False).run(
        top_k=3, judge=False
    )
    assert result["hit_rate"] == 1.0          # 若拒答用例计入分母, 这里会是 1.0 以下
    refusals = [d for d in result["details"] if d["kind"] == "refusal"]
    assert refusals[0]["hit"] is None          # 拒答用例不出 hit 结论


def test_refusal_skipped_without_judge():
    """--no-judge 不生成回答 → 拒答用例无从判定, 指标为 None 而非算作失败"""
    rag = StubRag()
    result = EvalService(rag, JudgeLLM(), make_settings(), record_history=False).run(
        top_k=3, judge=False
    )
    assert result["refusal_accuracy"] is None
    assert result["refusal_checked"] == 0
    assert result["refusal_total"] == BUILTIN_REFUSAL_N
    assert rag.generates == []


# ── API DTO 兼容性 (回归防护) ────────────────────────────


def test_api_dto_accepts_refusal_detail():
    """回归: 拒答用例的 hit=None 必须能通过 EvalTaskResponse 校验。

    端点实现是 EvalTaskResponse(task_id=..., **tasks[task_id]), 而
    EvalCaseDetail.hit 原为必填 bool —— 引入拒答用例后若不同步放宽为
    bool | None, 每次带 judge 的评测轮询都会 500。
    """
    from app.schemas.eval import EvalTaskResponse

    rag = StubRag(answer="根据现有知识库无法回答此问题。")
    result = EvalService(rag, JudgeLLM(), make_settings(), record_history=False).run(top_k=3)

    resp = EvalTaskResponse(task_id="abc12345", status="done", **result)

    assert resp.refusal_accuracy == 1.0
    assert resp.refusal_total == BUILTIN_REFUSAL_N
    assert resp.details is not None
    refusal = [d for d in resp.details if d.kind == "refusal"][0]
    assert refusal.hit is None
    assert refusal.refused is True
    retrieval = [d for d in resp.details if d.kind == "retrieval"][0]
    assert retrieval.hit is not None
    assert retrieval.refused is None
