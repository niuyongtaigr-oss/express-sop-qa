"""业务异常体系 + 全局异常处理器

统一错误响应格式:
  {"error": {"code": "...", "message": "...", "trace_id": "..."}}

🏭 Java 对标: @RestControllerAdvice + 自定义 BusinessException 体系
"""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.logging import get_trace_id

logger = logging.getLogger(__name__)


class AppError(Exception):
    """业务异常基类 — 子类定义 status_code 与错误码"""

    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class KnowledgeBaseNotReady(AppError):
    """知识库未建索引 (调用查询类接口前置检查失败)"""

    status_code = 409
    code = "knowledge_base_not_ready"


class DegradedError(AppError):
    """服务降级 (超时/上游不可用), 由上层转成友好响应"""

    status_code = 503
    code = "service_degraded"


class UnauthorizedError(AppError):
    """API Key 校验失败"""

    status_code = 401
    code = "unauthorized"


class RateLimitExceeded(AppError):
    """调用方限流/配额超限 (令牌桶空或日配额尽)"""

    status_code = 429
    code = "rate_limited"

    def __init__(self, message: str, retry_after_s: float | None = None):
        super().__init__(message)
        self.retry_after_s = retry_after_s


def _error_payload(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message, "trace_id": get_trace_id()}}


def register_exception_handlers(app: FastAPI) -> None:
    """注册全局异常处理器 — 所有异常统一转成标准错误响应"""

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        logger.warning("业务异常: %s %s", exc.code, exc.message)
        headers = {}
        # 限流 429 附带 Retry-After (秒), 供客户端退避
        if isinstance(exc, RateLimitExceeded) and exc.retry_after_s is not None:
            headers["Retry-After"] = str(int(exc.retry_after_s))
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_payload(exc.code, exc.message),
            headers=headers,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=_error_payload("validation_error", str(exc.errors())),
        )

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
        # 兜底: 未预期异常记完整堆栈, 但不把内部细节暴露给客户端
        logger.exception("未处理异常: %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content=_error_payload("internal_error", "服务内部错误, 请稍后重试"),
        )
