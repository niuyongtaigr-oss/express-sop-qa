"""应用入口 — create_app() 应用工厂 + lifespan 生命周期

启动流程 (lifespan):
  1. 初始化结构化日志
  2. 构建 infrastructure 层 (Embedding / LLM / Chroma 向量库)
  3. 构建 RagService 并加载索引 (已有索引则跳过重建)
  4. 装配 LangGraph 编排图 + ChatService (限流/超时降级)
  5. 服务实例挂到 app.state, 由 api/deps.py 注入到路由

运行方式:
  uvicorn app.main:app --port 8000
  或 python3 -m app.main
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.v1.router import api_v1_router
from app.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import setup_logging
from app.core.middleware import RequestLoggingMiddleware

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期: 启动时建/加载索引并装配编排图"""
    settings = get_settings()
    setup_logging(settings.log_level)
    logger.info("应用启动中... llm=%s embed=%s", settings.llm_model, settings.embed_model)

    # 重依赖延迟到启动时加载 (infrastructure 工厂内部懒 import)
    from app.agents.graph import build_chat_graph
    from app.infrastructure.embeddings import create_embedding_client
    from app.infrastructure.llm import create_llm_client
    from app.infrastructure.vector_store import create_vector_store
    from app.services.chat_service import ChatService
    from app.services.eval_service import EvalService
    from app.services.feedback_service import FeedbackStore
    from app.services.rag_service import RagService
    from app.services.session_service import SessionStore

    embeddings = create_embedding_client(settings)
    llm = create_llm_client(settings)
    vector_store = create_vector_store(settings, embeddings)

    rag_service = RagService(vector_store, llm, settings)
    # 建/加载索引是同步重活, 丢线程池不阻塞事件循环
    n, rebuilt = await asyncio.to_thread(rag_service.ingest, False)
    logger.info("知识库就绪: %d chunks (rebuilt=%s)", n, rebuilt)

    graph = build_chat_graph(rag_service, llm, settings)

    sessions = SessionStore(
        ttl_s=settings.session_ttl_s,
        max_turns=settings.session_max_turns,
        max_sessions=settings.session_max_sessions,
    )
    # 后台定期清理过期会话 (asyncio 任务, 关闭时取消)
    stop_sweep = asyncio.Event()

    async def _session_sweeper():
        from app.core.metrics import SESSION_ACTIVE

        while not stop_sweep.is_set():
            try:
                await asyncio.wait_for(stop_sweep.wait(), settings.session_sweep_interval_s)
            except asyncio.TimeoutError:
                await asyncio.to_thread(sessions.sweep)
                SESSION_ACTIVE.set(sessions.count())

    sweep_task = asyncio.create_task(_session_sweeper())

    app.state.rag_service = rag_service
    app.state.chat_service = ChatService(
        graph, sessions, settings,
        get_kb_version=lambda: rag_service.kb_version,
    )
    app.state.eval_service = EvalService(rag_service, llm, settings)
    app.state.eval_tasks = {}
    app.state.sessions = sessions
    app.state.feedback_store = FeedbackStore(settings.feedback_file)
    logger.info("应用就绪")
    try:
        yield
    finally:
        stop_sweep.set()
        sweep_task.cancel()
        logger.info("应用已关闭")


def create_app() -> FastAPI:
    """应用工厂 — 创建并装配 FastAPI 实例"""
    app = FastAPI(
        title="快递 SOP 智能问答系统",
        description="FastAPI + LangGraph 编排 + RAG 知识库 (生产级分层架构)",
        version="1.0.0",
        lifespan=lifespan,
    )
    register_exception_handlers(app)
    app.add_middleware(RequestLoggingMiddleware)
    app.include_router(api_v1_router)
    return app


# uvicorn app.main:app 直接引用
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
