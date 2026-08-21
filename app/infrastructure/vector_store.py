"""向量库封装 — Chroma PersistentClient 持久化实现

职责与 RAGPipeline 参考实现等价:
  ingest   — 分块 (RecursiveCharacterTextSplitter) → 打标签 → 向量化 → 重建集合
  retrieve — 查询向量化 → 语义搜索 → Top-K (cosine distance → similarity)

持久化到 data/chroma/, 进程重启索引不丢; 与内存版 Client() 的本质区别。
🏭 Java 对标: 对 ES 封装的 Repository 层
"""

import logging
from dataclasses import dataclass, field
from typing import Protocol

from app.infrastructure.embeddings import EmbeddingClient

logger = logging.getLogger(__name__)

# chunk 分类标签关键词 (快递 SOP 业务词)
_TAG_KEYWORDS = ["破损", "遗失", "拦截", "理赔", "赔偿", "罚款"]


@dataclass
class RetrievedChunk:
    """检索命中的知识块"""

    content: str
    metadata: dict = field(default_factory=dict)
    distance: float = 0.0
    similarity: float = 0.0


class VectorStore(Protocol):
    """向量库协议 — 业务层依赖的抽象"""

    def count(self) -> int:
        """当前集合中的 chunk 数"""
        ...

    def ingest(self, texts: list[str]) -> int:
        """全量重建索引: 分块 → 向量化 → 入库, 返回 chunk 数"""
        ...

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        """语义检索 Top-K"""
        ...


class ChromaVectorStore:
    """Chroma 实现 — PersistentClient 持久化 + cosine 距离"""

    def __init__(
        self,
        persist_dir: str,
        collection_name: str,
        embeddings: EmbeddingClient,
        chunk_size: int = 200,
        chunk_overlap: int = 40,
    ):
        # 重依赖延迟到实例化时加载 (不用知识库的进程不必付出启动成本)
        import chromadb
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        self._embeddings = embeddings
        self._collection_name = collection_name
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", " ", ""],
        )
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self._collection.count()

    def ingest(self, texts: list[str]) -> int:
        """全量重建: 删旧集合 → 分块 → 打标签 → 向量化 → 批量入库"""
        from langchain_core.documents import Document

        docs = [Document(page_content=t) for t in texts]
        chunks = self._text_splitter.split_documents(docs)
        chunk_texts = [c.page_content for c in chunks]
        if not chunk_texts:
            logger.warning("ingest 输入分块结果为空")
            return 0

        # 为每个 chunk 补充分类标签与序号
        metadatas = []
        for i, ct in enumerate(chunk_texts):
            tags = [kw for kw in _TAG_KEYWORDS if kw in ct]
            metadatas.append({
                "tags": " + ".join(tags) if tags else "其他",
                "chunk_index": i,
            })

        embeddings = self._embeddings.embed(chunk_texts)

        # 重建集合 (幂等: 先删后建)
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._collection.add(
            documents=chunk_texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=[f"{self._collection_name}-{i}" for i in range(len(chunk_texts))],
        )
        logger.info("向量库重建完成: %d chunks", len(chunk_texts))
        return len(chunk_texts)

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        """查询向量化 → 语义搜索 → Top-K (cosine distance → similarity)"""
        total = self._collection.count()
        if total == 0:
            return []
        q_emb = self._embeddings.embed([query])[0]
        results = self._collection.query(
            query_embeddings=[q_emb],
            n_results=min(top_k, total),
            include=["documents", "metadatas", "distances"],
        )
        return [
            RetrievedChunk(
                content=results["documents"][0][i],
                metadata=results["metadatas"][0][i],
                distance=results["distances"][0][i],
                similarity=1 - results["distances"][0][i],
            )
            for i in range(len(results["ids"][0]))
        ]


def create_vector_store(settings, embeddings: EmbeddingClient) -> VectorStore:
    """向量库工厂 — 按配置创建实现实例 (当前仅 Chroma)"""
    settings.chroma_path.mkdir(parents=True, exist_ok=True)
    return ChromaVectorStore(
        persist_dir=str(settings.chroma_path),
        collection_name=settings.collection_name,
        embeddings=embeddings,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
