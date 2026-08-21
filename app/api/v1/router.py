"""API v1 路由汇总 — 所有业务接口挂在 /api/v1 前缀下

版本化路由: 后续不兼容变更开 /api/v2, 老版本平滑下线。
"""

from fastapi import APIRouter

from app.api.v1.endpoints import chat, eval, feedback, health, rag

api_v1_router = APIRouter(prefix="/api/v1")
api_v1_router.include_router(health.router)
api_v1_router.include_router(chat.router)
api_v1_router.include_router(rag.router)
api_v1_router.include_router(eval.router)
api_v1_router.include_router(feedback.router)
