"""知识库服务 — ingest / retrieve / ask

组合 infrastructure 层 (VectorStore + LLMClient), 对上层暴露干净的业务接口。
本层不碰 HTTP、不碰框架细节, 只做业务编排。

🏭 Java 对标: Service 层 (注入 Repository 与外部客户端)
"""

import logging

from app.config import Settings
from app.core.exceptions import KnowledgeBaseNotReady
from app.infrastructure.llm import LLMClient
from app.infrastructure.vector_store import RetrievedChunk, VectorStore

logger = logging.getLogger(__name__)

# RAG 生成用 system prompt (严格基于参考文档回答)
_RAG_SYSTEM_PROMPT = (
    "你是一个快递行业的知识库助手。请严格根据下面提供的参考文档回答问题。\n"
    "规则:\n"
    "1. 如果参考文档中有答案，直接引用相关内容\n"
    "2. 如果文档中部分相关但不完整，基于文档推断并标注「根据文档推断」\n"
    "3. 如果文档完全不相关，诚实回答「根据现有知识库无法回答此问题」\n"
    "4. 回答简洁专业，不超过 200 字"
)


class RagService:
    """SOP 知识库服务 — 对上层暴露的唯一知识库入口"""

    def __init__(self, vector_store: VectorStore, llm: LLMClient, settings: Settings):
        self._store = vector_store
        self._llm = llm
        self._settings = settings

    # ── 建索引 ───────────────────────────────────────────
    def ingest(self, force: bool = False) -> tuple[int, bool]:
        """从 SOP 文档建索引

        已有索引且未指定 force 时跳过重建 (Chroma 持久化, 重启不丢)。
        返回 (chunk 数, 是否实际重建)。
        """
        existing = self._store.count()
        if existing > 0 and not force:
            logger.info("索引已存在 (%d chunks), 跳过重建", existing)
            return existing, False
        text = self._settings.sop_file.read_text(encoding="utf-8")
        n = self._store.ingest([text])
        logger.info("知识库索引重建完成: %d chunks, source=%s",
                    n, self._settings.sop_file)
        return n, True

    @property
    def indexed_chunks(self) -> int:
        return self._store.count()

    @property
    def ready(self) -> bool:
        return self._store.count() > 0

    # ── 查询接口 ─────────────────────────────────────────
    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """纯检索: 返回 [{content, metadata, distance, similarity}], 不调 LLM"""
        self._require_ready()
        chunks = self._store.retrieve(query, top_k or self._settings.top_k)
        return [
            {
                "content": c.content,
                "metadata": c.metadata,
                "distance": c.distance,
                "similarity": c.similarity,
            }
            for c in chunks
        ]

    def ask(self, query: str, top_k: int | None = None) -> dict:
        """检索 + 生成一站式问答, 返回 {answer, sources}"""
        chunks = self.retrieve(query, top_k)
        answer = self.generate(query, chunks)
        return {"answer": answer, "sources": self._to_sources(chunks)}

    def generate(self, query: str, chunks: list[dict]) -> str:
        """生成: 检索结果 + 用户问题 → LLM 回答 (multi_hop 节点也复用)"""
        from langchain_core.messages import HumanMessage, SystemMessage

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            tags = chunk.get("metadata", {}).get("tags", "")
            context_parts.append(f"[参考文档 {i}] (标签: {tags})\n{chunk['content']}")
        context = "\n\n".join(context_parts) if context_parts else "(无检索结果)"
        user_prompt = (
            f"参考文档:\n{context}\n\n"
            f"用户问题: {query}\n\n"
            f"请根据上述参考文档回答用户问题:"
        )
        return self._llm.invoke([
            SystemMessage(content=_RAG_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])

    # ── 内部 ─────────────────────────────────────────────
    def _require_ready(self) -> None:
        if not self.ready:
            raise KnowledgeBaseNotReady("知识库未建索引, 请先调用 POST /rag/ingest")

    @staticmethod
    def _to_sources(chunks: list[dict]) -> list[dict]:
        return [
            {
                "content": c["content"],
                "tags": c.get("metadata", {}).get("tags", ""),
                "similarity": round(c["similarity"], 4),
            }
            for c in chunks
        ]
