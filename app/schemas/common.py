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
    """健康检查: 服务存活 + 知识库索引状态 + Ollama 可达性 + 访问控制状态"""

    status: str
    indexed_chunks: int
    ollama: str  # up / down
    env: str = "dev"        # dev / prod
    auth: str = "disabled"  # enabled / tenant / disabled —— 让运维看得见是否裸奔
