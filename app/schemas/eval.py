"""Eval 资源 DTO — /eval/run, /eval/tasks/{task_id}"""

from pydantic import BaseModel


class EvalRunResponse(BaseModel):
    """提交评测任务的响应"""

    task_id: str
    status: str  # pending


class EvalCaseDetail(BaseModel):
    """单条评测用例结果"""

    question: str
    expect: str
    hit: bool
    top_similarity: float


class EvalTaskResponse(BaseModel):
    """评测任务状态/结果"""

    task_id: str
    status: str  # pending / running / done / error
    hit_rate: float | None = None
    top_k: int | None = None
    details: list[EvalCaseDetail] | None = None
    detail: str | None = None  # status=error 时的错误说明
