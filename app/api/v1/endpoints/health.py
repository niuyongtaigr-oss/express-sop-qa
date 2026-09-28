"""健康检查端点 — GET /api/v1/health

报告服务存活状态 + 知识库索引状态 + Ollama 可达性。
探活接口, 不做 API Key 校验。
"""

import logging

import httpx
from fastapi import APIRouter, Depends

from app.api.deps import get_rag_service, get_settings
from app.config import Settings
from app.core.metrics import KB_CHUNKS, OLLAMA_UP
from app.core.security import auth_status
from app.schemas.common import HealthResponse
from app.services.rag_service import RagService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(
    rag_service: RagService = Depends(get_rag_service),
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    """服务存活 + 知识库索引状态 + Ollama 可达性 (同时刷新 Prometheus 状态指标)"""
    ollama = "down"
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"{settings.ollama_base_url}/api/tags")
            ollama = "up" if r.status_code == 200 else "down"
    except httpx.HTTPError:
        ollama = "down"
    OLLAMA_UP.set(1 if ollama == "up" else 0)
    KB_CHUNKS.set(rag_service.indexed_chunks)
    return HealthResponse(
        status="ok",
        indexed_chunks=rag_service.indexed_chunks,
        ollama=ollama,
        env=settings.env,
        auth=auth_status(settings),
    )
