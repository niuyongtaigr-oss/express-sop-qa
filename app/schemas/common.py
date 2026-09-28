"""通用 DTO — 统一错误响应 / 健康检查响应"""

from pydantic import BaseModel


class ErrorDetail(BaseModel):
    """统一错误体: {"error": {...}}"""

    code: str
    message: str
    trace_id: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


class HealthResponse(BaseModel):
    """存活探针 (liveness): 进程还活着 —— 不包含依赖状态 (那是 readiness 的事)"""

    status: str
    indexed_chunks: int = 0
    env: str = "dev"        # dev / prod
    auth: str = "disabled"  # enabled / tenant / disabled —— 让运维看得见是否裸奔


class ReadyResponse(BaseModel):
    """就绪探针 (readiness): 依赖齐备才 ready, 否则 503 + reasons 说明缺什么"""

    status: str             # ready / not_ready
    indexed_chunks: int = 0
    ollama: str = "down"    # up / down
    reasons: list[str] = []
