"""知识库端点 — POST /api/v1/rag/query, POST /api/v1/rag/ingest

/rag/query:  绕过意图编排, 直接检索 + 生成 (调试/评测用)
/rag/ingest: 从 SOP 文档重建索引 (?force=true 强制重建)
"""

import asyncio
import logging

from fastapi import APIRouter, Depends

from app.api.deps import get_rag_service, get_settings
from app.config import Settings
from app.core.exceptions import DegradedError
from app.core.security import verify_api_key
from app.schemas.rag import IngestResponse, RagQueryRequest, RagQueryResponse
from app.services.rag_service import RagService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["rag"], dependencies=[Depends(verify_api_key)])


@router.post("/rag/query", response_model=RagQueryResponse)
async def rag_query(
    req: RagQueryRequest,
    rag_service: RagService = Depends(get_rag_service),
    settings: Settings = Depends(get_settings),
) -> RagQueryResponse:
    """知识库直通查询 (检索 + 生成), 带超时控制"""
    try:
        # 检索/生成是同步阻塞调用, 丢线程池; 超时抛降级异常 (503)
        result = await asyncio.wait_for(
            asyncio.to_thread(rag_service.ask, req.query, req.top_k),
            timeout=settings.rag_timeout_s,
        )
    except asyncio.TimeoutError:
        raise DegradedError(
            f"知识库查询超时 (>{settings.rag_timeout_s:.0f}s), 请稍后重试"
        )
    return RagQueryResponse(**result)


@router.post("/rag/ingest", response_model=IngestResponse)
async def rag_ingest(
    force: bool = False,
    rag_service: RagService = Depends(get_rag_service),
) -> IngestResponse:
    """重建知识库索引。默认幂等 (已有索引则跳过); force=true 强制重建"""
    n, rebuilt = await asyncio.to_thread(rag_service.ingest, force)
    return IngestResponse(indexed_chunks=n, rebuilt=rebuilt)
