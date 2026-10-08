"""Chat 资源 DTO — /chat, /chat/stream"""

from pydantic import BaseModel, Field

from app.schemas.memory import MemoryItemDTO


class ChatRequest(BaseModel):
    """智能问答请求"""

    question: str = Field(min_length=1, max_length=2000, description="用户问题")
    session_id: str | None = Field(
        default=None,
        min_length=16,
        max_length=64,
        description=(
            "会话 ID (多轮记忆, 同 ID 共享上下文)。由调用方生成, **必须不可猜测**"
            "(建议 UUID4) —— 服务端只按租户隔离, 不区分租户内的最终用户, "
            "能给出同一个 ID 的人就能读到该会话历史。"
            "注意: 这里的长度下限只是挡住 \"1\" 这类显然可猜的值, "
            "**长度不等于不可猜测** —— 真正的不可猜测由调用方负责。"
        ),
    )


class SourceItem(BaseModel):
    """引用来源 (检索命中的知识块)

    `doc_id` + `chunk_index` 让引用**可定位**: 前端能据此跳到原文对应段落, 人工
    也能复核模型有没有真的引用这一段。"引用了哪个文档"与"引用了哪一段"是两件
    事 —— 只给文档级出处时, 一段被误引用的回答看起来完全正常。
    """

    content: str
    doc_id: str = ""
    title: str = ""
    tags: str = ""
    chunk_index: int | None = Field(
        default=None, description="该块在文档内的序号 (0 起), 用于定位原文段落"
    )
    similarity: float = 0.0

    @classmethod
    def from_chunk(cls, chunk: dict) -> "SourceItem":
        """检索命中的 chunk → DTO

        只此一处做这个映射: 原先 `RagService` 与 multi_hop 节点各写一份,
        加字段时漏改其中一份不会有任何报错, 只是某个入口悄悄少一个字段。
        """
        meta = chunk.get("metadata") or {}
        return cls(
            content=chunk.get("content", ""),
            doc_id=meta.get("doc_id") or "",
            title=meta.get("title") or "",
            tags=meta.get("tags") or "",
            chunk_index=meta.get("chunk_index"),
            similarity=round(chunk.get("similarity") or 0.0, 4),
        )


class ChatResponse(BaseModel):
    """智能问答响应"""

    answer: str
    intent: str  # rag_qa / direct / multi_hop / degraded
    sources: list[SourceItem] = []
    # 本轮召回并喂给模型的长期记忆 —— 暴露出来是为了让"系统记住了什么"可见:
    # 记忆错了却看不见, 就没法排查(它不像引用那样有原文可核对)
    memories: list[MemoryItemDTO] = []
    trace_id: str
    elapsed_ms: float
    session_id: str | None = None
    cached: bool = False  # 是否命中答案缓存 (P3-A)