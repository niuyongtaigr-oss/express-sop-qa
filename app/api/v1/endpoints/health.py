"""健康检查端点 — 存活 (liveness) 与就绪 (readiness) **分开**

为什么必须分开: 两者的失败含义相反。
  · 存活探针失败 → 编排器**重启**进程。进程本身没坏、只是依赖 (Ollama/索引)
    不可用时, 重启毫无帮助, 还会掐断在途请求并引发重启风暴。
  · 就绪探针失败 → 负载均衡**摘掉流量**, 不动进程。这才是依赖故障时的正确动作。

原先只有一个 /health, 且无论依赖是否可用都返回 200 —— 负载均衡无法据此摘流量,
只能把请求继续送进来撞超时。因此:

  GET /api/v1/health        存活: 不打网络, 只回答"进程还能服务", 恒定 200
  GET /api/v1/health/ready  就绪: 探 Ollama + 查索引, 不可用返回 503

探活接口不做 API Key 校验 (编排器不会带密钥), 所以它们只应暴露在集群内网,
不要直接挂到公网。
"""

import logging

import httpx
from fastapi import APIRouter, Depends, Request, Response

from app.api.deps import get_rag_service, get_settings
from app.config import Settings
from app.core.metrics import KB_CHUNKS, OLLAMA_UP
from app.core.security import auth_status
from app.schemas.common import HealthResponse, ReadyResponse
from app.services.rag_service import RagService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(
    rag_service: RagService = Depends(get_rag_service),
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    """存活探针 (liveness): 进程能响应即 200 —— **刻意不探依赖**

    这里既不查 Ollama 也不查索引可用性: 依赖不可用时进程依然健康, 重启解决不了
    问题。依赖状态请查 /health/ready。

    `indexed_chunks` 是进程内的本地状态 (不产生网络调用), 放在这里只是为了方便
    一眼看到"索引是不是空的"。
    """
    return HealthResponse(
        status="ok",
        indexed_chunks=rag_service.indexed_chunks,
        env=settings.env,
        auth=auth_status(settings),
    )


async def _probe_ollama(request: Request, settings: Settings) -> str:
    """探 Ollama 可达性 (复用 lifespan 创建的 client, 不每次新建连接)"""
    try:
        r = await request.app.state.ollama_probe.get(
            f"{settings.ollama_base_url}/api/tags"
        )
        return "up" if r.status_code == 200 else "down"
    except httpx.HTTPError:
        return "down"


@router.get(
    "/health/ready",
    response_model=ReadyResponse,
    responses={503: {"model": ReadyResponse, "description": "依赖不可用, 请摘流量"}},
)
async def health_ready(
    request: Request,
    response: Response,
    rag_service: RagService = Depends(get_rag_service),
    settings: Settings = Depends(get_settings),
) -> ReadyResponse:
    """就绪探针 (readiness): 依赖齐备才 200, 否则 503 (负载均衡据此摘流量)

    同时刷新 Prometheus 的依赖状态指标 —— 探依赖是这里的事, 存活探针不做。
    """
    ollama = await _probe_ollama(request, settings)
    chunks = rag_service.indexed_chunks
    OLLAMA_UP.set(1 if ollama == "up" else 0)
    KB_CHUNKS.set(chunks)

    reasons: list[str] = []
    if not rag_service.ready or chunks <= 0:
        reasons.append("知识库未建索引 (POST /api/v1/rag/ingest)")
    if ollama != "up":
        reasons.append(f"Ollama 不可达 ({settings.ollama_base_url})")

    if reasons:
        # 用 response.status_code 而不是抛 HTTPException: 后者会走异常处理器,
        # 响应体变成统一错误格式, 探针拿不到结构化的 not_ready 详情。
        response.status_code = 503
        logger.warning("readiness 未通过: %s", "; ".join(reasons))
        return ReadyResponse(
            status="not_ready", indexed_chunks=chunks, ollama=ollama, reasons=reasons
        )
    return ReadyResponse(status="ready", indexed_chunks=chunks, ollama=ollama)
