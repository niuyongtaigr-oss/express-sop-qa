"""应用入口 — create_app() 应用工厂 + lifespan 生命周期

启动流程 (lifespan):
  1. 初始化结构化日志
  2. 构建 infrastructure 层 (Embedding / LLM / Chroma 向量库)
  3. 构建 RagService 并按索引清单增量重建索引 (语料未变则跳过, 见 P1.9)
  4. 装配 LangGraph 编排图 + ChatService (限流/超时降级)
  5. 服务实例挂到 app.state, 由 api/deps.py 注入到路由

运行方式:
  uvicorn app.main:app --port 8000
  或 python3 -m app.main
"""

import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse

from app.api.v1.router import api_v1_router
from app.config import Settings, get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import setup_logging
from app.core.middleware import RequestLoggingMiddleware

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期: 启动时建/加载索引并装配编排图"""
    settings = get_settings()
    setup_logging(settings.log_level)
    # 启动自检: prod 下未配置访问控制直接拒绝启动 (fail-closed)
    from app.core.security import assert_secure_settings

    assert_secure_settings(settings)
    logger.info("应用启动中... env=%s llm=%s embed=%s",
                settings.env, settings.llm_model, settings.embed_model)

    # 重依赖延迟到启动时加载 (infrastructure 工厂内部懒 import)
    from app.agents.graph import build_chat_graph
    from app.core.rate_limit import RateLimiter
    from app.infrastructure.embeddings import create_embedding_client
    from app.infrastructure.llm import create_intent_llm, create_llm_client
    from app.infrastructure.vector_store import create_vector_store
    from app.services.chat_service import ChatService
    from app.services.eval_service import EvalService
    from app.services.feedback_service import FeedbackStore
    from app.services.rag_service import RagService
    from app.services.session_service import SessionStore
    from app.services.tenant_service import TenantRegistry

    embeddings = create_embedding_client(settings)
    llm = create_llm_client(settings)
    # P3-C 多模型路由: 配置 intent_model 时意图识别走独立小模型
    intent_llm = create_intent_llm(settings)
    vector_store = create_vector_store(settings, embeddings)

    rag_service = RagService(vector_store, llm, settings)
    # 建/加载索引是同步重活, 丢线程池不阻塞事件循环
    n, rebuilt = await asyncio.to_thread(rag_service.ingest, False)
    logger.info("知识库就绪: %d chunks (rebuilt=%s)", n, rebuilt)

    graph = build_chat_graph(rag_service, llm, settings, intent_llm=intent_llm)

    sessions = SessionStore(
        ttl_s=settings.session_ttl_s,
        max_turns=settings.session_max_turns,
        max_sessions=settings.session_max_sessions,
    )
    rate_limiter = RateLimiter(
        enabled=settings.rate_limit_enabled,
        per_min=settings.rate_limit_per_min,
        burst=settings.rate_limit_burst,
        daily_quota=settings.rate_quota_daily,
    )
    # 后台定期清理过期会话/限流桶 (asyncio 任务, 关闭时取消)
    stop_sweep = asyncio.Event()

    async def _session_sweeper():
        from app.core.metrics import SESSION_ACTIVE

        while not stop_sweep.is_set():
            try:
                await asyncio.wait_for(stop_sweep.wait(), settings.session_sweep_interval_s)
            except asyncio.TimeoutError:
                # 单次清理失败不能让后台任务死掉 —— 否则清理静默停止且无告警,
                # 会话与限流桶会一直堆积到进程重启。
                try:
                    await asyncio.to_thread(sessions.sweep)
                    await asyncio.to_thread(rate_limiter.sweep)
                    SESSION_ACTIVE.set(sessions.count())
                except Exception:
                    logger.exception("后台清理失败, 本轮跳过")

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
    app.state.rate_limiter = rate_limiter
    # 探活复用一个 httpx client —— /health 每次新建连接会让高频探针放大开销
    app.state.ollama_probe = httpx.AsyncClient(timeout=2.0)
    # P4 多租户: tenant_mode 时加载租户清单 (X-API-Key → tenant_id)
    app.state.tenant_registry = TenantRegistry(
        settings.tenants_file if settings.tenant_mode else None
    )
    if settings.tenant_mode:
        logger.info("多租户模式已开启 (%d 租户)",
                    len(app.state.tenant_registry.list_tenants()))
    logger.info("应用就绪")
    try:
        yield
    finally:
        stop_sweep.set()
        # 取消后要 await, 否则任务可能仍在跑 (取消不彻底)
        sweep_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await sweep_task
        await app.state.ollama_probe.aclose()
        logger.info("应用已关闭")


def _setup_cors(app: FastAPI, settings: Settings) -> None:
    """按配置挂载 CORS 中间件 —— 默认**关闭** (同源部署不需要)。

    为什么默认关: CORS 是"允许别的源在浏览器里调用本服务", 每开一个源都是多一份
    暴露面。同源部署的前端根本不需要它, 所以默认空 = 不挂中间件。

    关于 `*`: 本服务的鉴权走 `X-API-Key` 请求头, 不依赖 Cookie, 因此**不设置
    allow_credentials** —— 不带凭据时通配源不会把登录态泄漏给任意站点 (第三方
    页面拿不到用户浏览器里本服务的 localStorage, 也拿不到 API Key)。反过来说,
    一旦有人想用 Cookie 鉴权, 就必须改成显式列出来源, 所以这里干脆不提供
    allow_credentials 开关: 把"通配源 + 带凭据"这个经典组合从配置面上删掉, 比写
    一句文档提醒更可靠。
    """
    origins = [o.strip() for o in settings.cors_allow_origins.split(",") if o.strip()]
    if not origins:
        return
    if "*" in origins:
        logger.warning(
            "CORS 允许任意来源 (SOP_QA_CORS_ALLOW_ORIGINS=*): 任何网站都能在浏览器里"
            "调用本服务。生产环境请改为显式列出来源。"
        )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,   # 鉴权走 X-API-Key 请求头, 不用 Cookie
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],       # 需要放行 X-API-Key / Content-Type
    )


def _mount_console(app: FastAPI) -> None:
    """挂载极简管理台 (/admin) —— 单文件 HTML, 无外部依赖。

    为什么需要它: `/rag/docs` 这些接口都能用, 但目标用户是企业管理员, 不会用
    curl。缺一个能"看见并操作知识库"的页面, 功能再全也交付不出去。

    为什么页面本身可以不鉴权: 它**自身不持有任何数据** —— 没有密钥、没有语料,
    内容全靠 JS 实时调同源 API 拉取; 密钥由使用者在页面里输入, 只存在自己浏览器
    的 sessionStorage 里。所以真正需要管的访问控制仍在 API 层由 verify_api_key
    负责 (prod 下未配置访问控制会直接拒绝启动)。

    与 CORS 默认关闭是配套的: 页面与 API 同源, 因此不需要开 CORS。
    """
    html = Path(__file__).resolve().parent / "static" / "admin.html"

    @app.get("/admin", include_in_schema=False)
    async def admin_console() -> FileResponse:
        return FileResponse(html, media_type="text/html")

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse("/admin")


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
    _setup_cors(app, get_settings())
    _mount_console(app)
    app.include_router(api_v1_router)
    return app


# uvicorn app.main:app 直接引用
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
