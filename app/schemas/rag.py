"""RAG 资源 DTO — /rag/query, /rag/ingest"""

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
    rebuilt: bool  # True=本次实际重建; False=索引已存在, 跳过
