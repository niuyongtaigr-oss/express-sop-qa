"""RAG 资源 DTO — /rag/query, /rag/ingest, /rag/docs"""

from pydantic import BaseModel, Field

from app.schemas.chat import SourceItem


class RagQueryRequest(BaseModel):
    """知识库直通查询请求"""

    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)


class RagQueryResponse(BaseModel):
    """知识库直通查询响应 (检索 + 生成)"""

    answer: str
    sources: list[SourceItem]


class IngestResponse(BaseModel):
    """重建索引响应"""

    indexed_chunks: int
    rebuilt: bool  # True=本次实际导入; False=索引已存在, 跳过


class DocInfo(BaseModel):
    """知识库文档清单项"""

    doc_id: str
    title: str
    chunk_count: int


class DocAddRequest(BaseModel):
    """新增/覆盖一篇文档"""

    doc_id: str | None = Field(
        default=None, max_length=64,
        description="文档 ID (缺省自动生成; 已存在则覆盖)",
    )
    title: str = Field(min_length=1, max_length=128)
    content: str = Field(min_length=1, description="文档正文")


class DocAddResponse(BaseModel):
    doc_id: str
    title: str
    indexed_chunks: int


class DocUploadResponse(BaseModel):
    """文件上传解析入库响应"""

    doc_id: str
    title: str
    indexed_chunks: int
    source: dict = Field(
        default_factory=dict,
        description="解析溯源: filename / ext / pages / sheets / encoding / "
                    "chars / chars_raw / truncated 等",
    )


class DocFormatsResponse(BaseModel):
    """当前支持的文档格式 (前端上传前限制 accept, 并由服务端如实告知)"""

    extensions: list[str]


class DocDeleteResponse(BaseModel):
    doc_id: str
    removed_chunks: int


class ListDocsResponse(BaseModel):
    documents: list[DocInfo]
    total_chunks: int
