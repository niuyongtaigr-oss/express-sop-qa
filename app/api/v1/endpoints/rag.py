"""知识库端点 — /rag/query, /rag/ingest, /rag/docs, /rag/docs/upload

/rag/query:        绕过意图编排, 直接检索 + 生成 (调试/评测用)
/rag/ingest:       按清单比对语料, 增量重建索引 (force=true 全量重建) (P1.9)
/rag/docs:         多文档管理: 清单 / 增量导入 (upsert) / 删除 (P1.6)
/rag/docs/upload:  上传真实文件 (PDF/Word/Excel/CSV/文本) 解析入库 (P1.7)
"""

import asyncio
import logging
import re
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.api.deps import (
    enforce_rate_limit,
    get_rag_service,
    get_settings,
    get_tenant_id,
)
from app.config import Settings
from app.core.exceptions import DegradedError, DocumentParseError
from app.core.security import verify_api_key
from app.schemas.rag import (
    DocAddRequest,
    DocAddResponse,
    DocDeleteResponse,
    DocFormatsResponse,
    DocUploadResponse,
    IngestResponse,
    ListDocsResponse,
    RagQueryRequest,
    RagQueryResponse,
)
from app.services.rag_service import RagService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["rag"], dependencies=[Depends(verify_api_key), Depends(enforce_rate_limit)])

# doc_id 只允许安全字符 (与向量库 id 生成保持一致)
_DOC_ID_SAFE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


@router.post("/rag/query", response_model=RagQueryResponse)
async def rag_query(
    req: RagQueryRequest,
    rag_service: RagService = Depends(get_rag_service),
    settings: Settings = Depends(get_settings),
    tenant_id: str = Depends(get_tenant_id),
) -> RagQueryResponse:
    """知识库直通查询 (检索 + 生成, 按租户隔离), 带超时控制"""
    try:
        # 检索/生成是同步阻塞调用, 丢线程池; 超时抛降级异常 (503)
        result = await asyncio.wait_for(
            asyncio.to_thread(
                rag_service.ask, req.query, req.top_k, None, tenant_id
            ),
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
    """比对语料清单并重建索引: 默认增量 (只重建变化的文档), force=true 全量重建

    注意: force=true 会先清空再逐篇写入, 期间其他请求可能读到不完整索引 ——
    建议挑低峰期执行, 详见 README「索引维护与并发约束」。
    """
    n, rebuilt = await asyncio.to_thread(rag_service.ingest, force)
    return IngestResponse(indexed_chunks=n, rebuilt=rebuilt)


@router.get("/rag/docs", response_model=ListDocsResponse)
async def rag_list_docs(
    rag_service: RagService = Depends(get_rag_service),
    tenant_id: str = Depends(get_tenant_id),
) -> ListDocsResponse:
    """知识库文档清单 (本租户 + 共享库; doc_id / title / chunk 数)"""
    docs = await asyncio.to_thread(rag_service.list_documents, tenant_id)
    return ListDocsResponse(
        documents=docs,
        total_chunks=sum(d["chunk_count"] for d in docs),
    )


@router.post("/rag/docs", response_model=DocAddResponse)
async def rag_add_doc(
    req: DocAddRequest,
    rag_service: RagService = Depends(get_rag_service),
    tenant_id: str = Depends(get_tenant_id),
) -> DocAddResponse:
    """增量导入/覆盖一篇文档到本租户 (upsert: 同 doc_id 旧 chunk 先删后加)"""
    doc_id = req.doc_id or f"doc-{uuid.uuid4().hex[:12]}"
    if not _DOC_ID_SAFE.match(doc_id):
        raise HTTPException(
            status_code=422,
            detail="doc_id 仅允许字母/数字/._- , 长度 ≤64",
        )
    n = await asyncio.to_thread(
        rag_service.add_document, doc_id, req.title, req.content, tenant_id
    )
    return DocAddResponse(doc_id=doc_id, title=req.title, indexed_chunks=n)


@router.delete("/rag/docs/{doc_id}", response_model=DocDeleteResponse)
async def rag_delete_doc(
    doc_id: str,
    rag_service: RagService = Depends(get_rag_service),
    tenant_id: str = Depends(get_tenant_id),
) -> DocDeleteResponse:
    """删除本租户的一篇文档及其全部 chunk (不能删共享库文档)"""
    removed = await asyncio.to_thread(
        rag_service.remove_document, doc_id, tenant_id
    )
    if removed == 0:
        raise HTTPException(status_code=404, detail=f"文档不存在: {doc_id}")
    return DocDeleteResponse(doc_id=doc_id, removed_chunks=removed)


@router.get("/rag/docs/formats", response_model=DocFormatsResponse)
async def rag_doc_formats(
    rag_service: RagService = Depends(get_rag_service),
) -> DocFormatsResponse:
    """当前支持的文档格式 (前端据此设置上传 accept)"""
    return DocFormatsResponse(extensions=rag_service.supported_document_extensions())


@router.post("/rag/docs/upload", response_model=DocUploadResponse)
async def rag_upload_doc(
    file: UploadFile = File(..., description="待入库文档"),
    doc_id: str | None = Form(default=None),
    title: str | None = Form(default=None),
    rag_service: RagService = Depends(get_rag_service),
    settings: Settings = Depends(get_settings),
    tenant_id: str = Depends(get_tenant_id),
) -> DocUploadResponse:
    """上传真实文件 (PDF / Word / Excel / CSV / 文本) → 解析 → 清洗 → 切块入库

    与 POST /rag/docs 的区别: 那个收纯文本正文, 这个收**原始文件字节**,
    由文档解析层负责格式适配、中文编码识别与文本清洗。
    """
    filename = file.filename or "document"
    if doc_id and not _DOC_ID_SAFE.match(doc_id):
        raise HTTPException(
            status_code=422,
            detail="doc_id 仅允许字母/数字/._- , 长度 ≤64",
        )
    # 多读 1 字节用于判定超限, 避免把超大文件整个读进内存
    limit = settings.doc_max_bytes
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise DocumentParseError(f"文件过大 (>{limit / 1e6:.0f} MB), 请拆分后上传")

    result = await asyncio.to_thread(
        rag_service.ingest_upload, filename, data, doc_id, title, tenant_id
    )
    return DocUploadResponse(
        doc_id=result["doc_id"],
        title=result["title"],
        indexed_chunks=result["chunks"],
        source=result["source"],
    )
