"""Eval 资源 DTO — /eval/run, /eval/tasks/{task_id}"""

from pydantic import BaseModel


class EvalRunResponse(BaseModel):
    """提交评测任务的响应"""

    task_id: str
    status: str  # pending


class EvalCaseDetail(BaseModel):
    """单条评测用例结果"""

    case_id: str = ""
    question: str
    expect: str = ""
    hit: bool
    top_similarity: float = 0.0
    answer: str | None = None          # LLM 生成答案 (启用 judge 时有)
    faithfulness: float | None = None  # 忠实性 0-1
    completeness: float | None = None  # 完整性 0-1
    reason: str | None = None          # 评审理由 / judge_error 说明


class EvalTaskResponse(BaseModel):
    """评测任务状态/结果"""

    task_id: str
    status: str  # pending / running / done / error
    hit_rate: float | None = None
    faithfulness_avg: float | None = None
    completeness_avg: float | None = None
    judged_cases: int | None = None
    top_k: int | None = None
    judge: bool | None = None
    config: dict | None = None
    compare: dict | None = None  # 回归对比: {vs_ts, hit_rate_delta, ...}
    details: list[EvalCaseDetail] | None = None
    detail: str | None = None  # status=error 时的错误说明
