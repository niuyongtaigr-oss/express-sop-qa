"""请求日志中间件 — trace_id / method / path / status / elapsed

每个请求生成 trace_id 写入 contextvars, 处理完成后打一条结构化访问日志。
🏭 Java 对标: Servlet Filter + MDC 埋点
"""

import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.logging import new_trace_id

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """请求级日志: 入口生成 trace_id, 出口记录耗时与状态码"""

    async def dispatch(self, request: Request, call_next) -> Response:
        trace_id = new_trace_id()
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
        response.headers["X-Trace-Id"] = trace_id
        logger.info(
            "http_access",
            extra={
                "fields": {
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "elapsed_ms": elapsed_ms,
                }
            },
        )
        return response
