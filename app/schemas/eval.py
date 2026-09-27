"""Eval 资源 DTO — /eval/run, /eval/tasks/{task_id}"""

from pydantic import BaseModel


class EvalRunResponse(BaseModel):
    """提交评测任务的响应"""

    task_id: str
    status: str  # pending


class EvalCaseDetail(BaseModel):
    """单条评测用例结果

    两类用例结果字段不同:
      - kind="retrieval": hit(bool) + 可选 faithfulness/completeness
      - kind="refusal":    hit=None (不参与检索指标) + refused(bool)
    """

    case_id: str = ""
    question: str
    kind: str = "retrieval"            # retrieval | refusal
    expect: str = ""
    hit: bool | None = None            # 检索用例: Top-K 是否覆盖期望关键词; 拒答用例恒为 None
    refused: bool | None = None        # 拒答用例: 是否如实拒答 (未编造)
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
    refusal_accuracy: float | None = None  # 拒答准确率 (judge 关闭时为 None)
    refusal_checked: int | None = None     # 实际评估的拒答用例数
    refusal_total: int | None = None       # 评测集中的拒答用例总数
    faithfulness_avg: float | None = None
    completeness_avg: float | None = None
    judged_cases: int | None = None
    top_k: int | None = None
    judge: bool | None = None
    config: dict | None = None
    compare: dict | None = None  # 回归对比: {vs_ts, hit_rate_delta, ...}
    details: list[EvalCaseDetail] | None = None
    detail: str | None = None  # status=error 时的错误说明
