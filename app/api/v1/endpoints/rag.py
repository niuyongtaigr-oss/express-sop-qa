"""知识库端点 — /rag/query, /rag/ingest, /rag/docs

/rag/query:    绕过意图编排, 直接检索 + 生成 (调试/评测用)
/rag/ingest:   导入默认 SOP 文档 (force=true 清空后重建)
/rag/docs:     多文档管理: 清单 / 增量导入 (upsert) / 删除 (P1.6)
"""

import asyncio
import logging
import re
import uuid

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_rag_service, get_settings
from app.config import Settings
from app.core.exceptions import DegradedError
from app.core.security import verify_api_key
from app.schemas.rag import (
    DocAddRequest,
    DocAddResponse,
    DocDeleteResponse,
    IngestResponse,
    ListDocsResponse,
    RagQueryRequest,
    RagQueryResponse,
)
from app.services.rag_service import RagService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["rag"], dependencies=[Depends(verify_api_key)])

# doc_id 只允许安全字符 (与向量库 id 生成保持一致)
_DOC_ID_SAFE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


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
    """导入默认 SOP 文档。默认幂等 (已有文档则跳过); force=true 清空后重建"""
    n, rebuilt = await asyncio.to_thread(rag_service.ingest, force)
    return IngestResponse(indexed_chunks=n, rebuilt=rebuilt)


@router.get("/rag/docs", response_model=ListDocsResponse)
async def rag_list_docs(
    rag_service: RagService = Depends(get_rag_service),
) -> ListDocsResponse:
    """知识库文档清单 (doc_id / title / chunk 数)"""
    docs = await asyncio.to_thread(rag_service.list_documents)
    return ListDocsResponse(
        documents=docs,
        total_chunks=sum(d["chunk_count"] for d in docs),
    )


@router.post("/rag/docs", response_model=DocAddResponse)
async def rag_add_doc(
    req: DocAddRequest,
    rag_service: RagService = Depends(get_rag_service),
) -> DocAddResponse:
    """增量导入/覆盖一篇文档 (upsert: 同 doc_id 旧 chunk 先删后加)"""
    doc_id = req.doc_id or f"doc-{uuid.uuid4().hex[:12]}"
    if not _DOC_ID_SAFE.match(doc_id):
        raise HTTPException(
            status_code=422,
            detail="doc_id 仅允许字母/数字/._- , 长度 ≤64",
        )
    n = await asyncio.to_thread(
        rag_service.add_document, doc_id, req.title, req.content
    )
    return DocAddResponse(doc_id=doc_id, title=req.title, indexed_chunks=n)


@router.delete("/rag/docs/{doc_id}", response_model=DocDeleteResponse)
async def rag_delete_doc(
    doc_id: str,
    rag_service: RagService = Depends(get_rag_service),
) -> DocDeleteResponse:
    """删除一篇文档及其全部 chunk"""
    removed = await asyncio.to_thread(rag_service.remove_document, doc_id)
    if removed == 0:
        raise HTTPException(status_code=404, detail=f"文档不存在: {doc_id}")
    return DocDeleteResponse(doc_id=doc_id, removed_chunks=removed)
