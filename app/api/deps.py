"""FastAPI 依赖注入 — 从 app.state 取 lifespan 里构建好的服务实例

服务实例在应用启动时 (lifespan) 构建一次并挂到 app.state,
请求处理时通过 Depends 注入, 不用全局单例。

🏭 Java 对标: @Autowired 从 ApplicationContext 取 Bean
"""

from fastapi import Depends, Request

from app.config import get_settings  # re-export, 路由统一从 deps 拿依赖
from app.core.exceptions import RateLimitExceeded
from app.core.rate_limit import RateLimiter
from app.services.chat_service import ChatService
from app.services.eval_service import EvalService
from app.services.feedback_service import FeedbackStore
from app.services.rag_service import RagService
from app.services.session_service import SessionStore

__all__ = ["get_settings", "get_rag_service", "get_chat_service",
           "get_eval_service", "get_eval_tasks", "get_session_store",
           "get_feedback_store", "get_rate_limiter", "enforce_rate_limit"]


def get_rag_service(request: Request) -> RagService:
    return request.app.state.rag_service


def get_chat_service(request: Request) -> ChatService:
    return request.app.state.chat_service


def get_eval_service(request: Request) -> EvalService:
    return request.app.state.eval_service


def get_eval_tasks(request: Request) -> dict:
    """评测任务登记表 (内存态, 进程重启即失效)"""
    return request.app.state.eval_tasks


def get_session_store(request: Request) -> SessionStore:
    """多轮会话存储 (内存态 LRU+TTL, 由 lifespan 创建)"""
    return request.app.state.sessions


def get_feedback_store(request: Request) -> FeedbackStore:
    """用户反馈存储 (JSONL 持久化, 由 lifespan 创建)"""
    return request.app.state.feedback_store


def get_rate_limiter(request: Request) -> RateLimiter:
    """按 Key/IP 限流器 (由 lifespan 创建)"""
    return request.app.state.rate_limiter


async def enforce_rate_limit(
    request: Request,
    limiter: RateLimiter = Depends(get_rate_limiter),
) -> None:
    """业务路由依赖: 按 X-API-Key (或客户端 IP) 限流 + 日配额 (P3-D)"""
    key = request.headers.get("X-API-Key") or (
        request.client.host if request.client else "unknown"
    )
    ok, retry_after, _ = limiter.allow(key)
    if not ok:
        raise RateLimitExceeded(
            "请求过于频繁, 已触发限流, 请稍后重试",
            retry_after_s=retry_after,
        )
