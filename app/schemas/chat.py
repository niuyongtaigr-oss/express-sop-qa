"""Chat 资源 DTO — /chat, /chat/stream"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """智能问答请求"""

    question: str = Field(min_length=1, max_length=2000, description="用户问题")
    session_id: str | None = Field(
        default=None, max_length=64, description="会话 ID (多轮记忆, 同 ID 共享上下文)"
    )


class SourceItem(BaseModel):
    """引用来源 (检索命中的知识块)"""

    content: str
    doc_id: str = ""
    title: str = ""
    tags: str = ""
    similarity: float = 0.0


class ChatResponse(BaseModel):
    """智能问答响应"""

    answer: str
    intent: str  # rag_qa / direct / multi_hop / degraded
    sources: list[SourceItem] = []
    trace_id: str
    elapsed_ms: float
    session_id: str | None = None
