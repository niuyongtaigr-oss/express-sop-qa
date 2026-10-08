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
           "get_feedback_store", "get_rate_limiter", "enforce_rate_limit",
           "get_tenant_id", "get_user_id", "get_memory_service"]


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


def get_memory_service(request: Request):
    """长期记忆服务 (由 lifespan 创建; memory_enabled=false 时它不写不读)"""
    return request.app.state.memory_service


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


def get_user_id(request: Request) -> str:
    """当前请求的用户 (用户级会话/记忆的隔离维度), 由 verify_api_key 写入

    **取不到就返回空串, 不回落到租户**: 空串表示"这次调用没有用户维度", 用户级
    功能据此**关闭**(而不是把私人数据记到租户上让全租户可见)。回落到租户看似
    更"可用", 但那是把隔离边界悄悄放宽, 属于方向性错误。

    与 get_tenant_id 的区别: 租户取不到要报错(否则会读写默认租户的数据); 用户
    取不到是一个**合法的部署形态**(单租户 + 一把共享 key), 所以返回空串。
    """
    return getattr(request.state, "user_id", "") or ""


def get_tenant_id(request: Request) -> str:
    """当前请求的租户 (P4 多租户): 由 verify_api_key 写入 request.state

    **fail-closed**: 取不到就报错, 绝不回退默认租户。回退的后果是——任何新接口
    只要漏挂 verify_api_key, 就会静默地去读写默认租户的知识库与会话: 一个"忘了
    加依赖"的疏忽直接变成跨租户越权, 而且没有任何报错。宁可 500 也不能默默读错
    数据。本函数只负责读取, 租户上下文必须由 verify_api_key 写入。
    """
    tenant_id = getattr(request.state, "tenant_id", None)
    if not tenant_id:
        raise RuntimeError(
            "request.state.tenant_id 未设置: 该路由缺少 verify_api_key 依赖。"
            "get_tenant_id 不会回退到默认租户 (那会造成静默的跨租户访问)。"
        )
    return tenant_id
