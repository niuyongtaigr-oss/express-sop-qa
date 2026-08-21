"""FastAPI 依赖注入 — 从 app.state 取 lifespan 里构建好的服务实例

服务实例在应用启动时 (lifespan) 构建一次并挂到 app.state,
请求处理时通过 Depends 注入, 不用全局单例。

🏭 Java 对标: @Autowired 从 ApplicationContext 取 Bean
"""

from fastapi import Request

from app.config import get_settings  # re-export, 路由统一从 deps 拿依赖
from app.services.chat_service import ChatService
from app.services.eval_service import EvalService
from app.services.rag_service import RagService

__all__ = ["get_settings", "get_rag_service", "get_chat_service",
           "get_eval_service", "get_eval_tasks"]


def get_rag_service(request: Request) -> RagService:
    return request.app.state.rag_service


def get_chat_service(request: Request) -> ChatService:
    return request.app.state.chat_service


def get_eval_service(request: Request) -> EvalService:
    return request.app.state.eval_service


def get_eval_tasks(request: Request) -> dict:
    """评测任务登记表 (内存态, 进程重启即失效)"""
    return request.app.state.eval_tasks
