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
from app.core.metrics import HTTP_DURATION, HTTP_REQUESTS

logger = logging.getLogger(__name__)


def _route_path(request: Request) -> str:
    """请求的路由模板路径 (避免 /chat/{id} 之类的高基数路径撑爆指标)"""
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    return path or request.url.path


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """请求级日志 + Prometheus 指标: 入口生成 trace_id, 出口记录耗时与状态码"""

    async def dispatch(self, request: Request, call_next) -> Response:
        trace_id = new_trace_id()
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            # 异常路径也记录指标 (500), 再向上抛给异常处理器
            elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
            path = _route_path(request)
            HTTP_REQUESTS.labels(request.method, path, "500").inc()
            HTTP_DURATION.labels(request.method, path).observe(elapsed_ms / 1000)
            raise
        elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
        response.headers["X-Trace-Id"] = trace_id
        path = _route_path(request)
        HTTP_REQUESTS.labels(request.method, path, str(response.status_code)).inc()
        HTTP_DURATION.labels(request.method, path).observe(elapsed_ms / 1000)
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
