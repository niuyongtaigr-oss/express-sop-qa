"""向量库封装 — Chroma PersistentClient 持久化实现 (多文档 + 混合检索)

职责:
  add_document    — 增量导入单篇文档 (分块 → 打标签 → 向量化 → upsert)
  remove_document — 按 doc_id 删除文档及全部 chunk
  list_documents  — 列出知识库文档清单 (从 chunk 元数据聚合, 无需独立注册表)
  retrieve        — 混合检索: 向量 Top-K + BM25 Top-K → RRF 融合 (可切纯向量)
  clear_all       — 清空全部文档 (force 重建用)

持久化到 data/chroma/, 进程重启索引不丢; 与内存版 Client() 的本质区别。
P1.5 混合检索: cosine 向量召回擅长语义相关, BM25 擅长精确术语/编号,
  RRF (Reciprocal Rank Fusion) 融合两者排序, 提升整体召回质量。
🏭 Java 对标: 对 ES 封装的 Repository 层
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Protocol

from app.infrastructure.bm25 import BM25Index
from app.infrastructure.embeddings import EmbeddingClient

logger = logging.getLogger(__name__)

# chunk 分类标签关键词 (快递 SOP 业务词)
_TAG_KEYWORDS = ["破损", "遗失", "拦截", "理赔", "赔偿", "罚款"]

# RRF 融合常数 (标准取值 60)
_RRF_K = 60

# Chroma id 只保留安全字符 (防止 doc_id 含路径分隔符等)
_ID_SAFE = re.compile(r"[^A-Za-z0-9_.-]")


@dataclass
class RetrievedChunk:
    """检索命中的知识块

    `score_kind` 说明 `similarity` 是**哪种分**, 它决定了这个数能不能和阈值直接
    比较 (见 nodes._is_confident):
      · "cosine"    — 向量余弦 (1 - 距离), 与相关度同量纲, 可与阈值比较
      · "bm25_norm" — BM25 分数的**归一化排名映射**, 恒落在 (0.4, 0.9], 且该模式
                      下的最高分**恒为 0.9**, 与真实相关度无关
    """

    content: str
    metadata: dict = field(default_factory=dict)
    distance: float = 0.0
    similarity: float = 0.0
    score_kind: str = "cosine"


class VectorStore(Protocol):
    """向量库协议 — 业务层依赖的抽象"""

    def count(self) -> int:
        """当前集合中的 chunk 数"""
        ...

    def add_document(
        self, doc_id: str, title: str, text: str, tenant_id: str = "default"
    ) -> int:
        """增量导入/覆盖单篇文档, 返回该文档的 chunk 数"""
        ...

    def remove_document(self, doc_id: str, tenant_id: str = "default") -> int:
        """删除文档及其全部 chunk, 返回删除数"""
        ...

    def list_documents(self, tenant_id: str = "default") -> list[dict]:
        """文档清单: [{doc_id, title, chunk_count}]"""
        ...

    def clear_all(self) -> None:
        """清空全部文档 (重建场景)"""
        ...

    def retrieve(
        self, query: str, top_k: int = 3, tenant_id: str = "default"
    ) -> list[RetrievedChunk]:
        """语义检索 Top-K (hybrid 模式为向量+BM25 融合, 按租户过滤)"""
        ...


class ChromaVectorStore:
    """Chroma 实现 — PersistentClient 持久化 + cosine 距离 + 可选 BM25 混合

    P4 多租户: 每个 chunk 带 tenant_id 元数据; 检索/文档操作按
    tenant_id + shared_tenant_id (共享库) 过滤。
    """

    def __init__(
        self,
        persist_dir: str,
        collection_name: str,
        embeddings: EmbeddingClient,
        chunk_size: int = 200,
        chunk_overlap: int = 40,
        retrieval_mode: str = "hybrid",
        shared_tenant_id: str = "shared",
    ):
        # 重依赖延迟到实例化时加载 (不用知识库的进程不必付出启动成本)
        import chromadb
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        self._embeddings = embeddings
        self._collection_name = collection_name
        self._mode = retrieval_mode
        self._shared_tenant = shared_tenant_id
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
        self._bm25: BM25Index | None = None
        if self._mode in ("hybrid", "bm25"):
            self._rebuild_bm25()

    # ── 文档管理 ─────────────────────────────────────────
    def count(self) -> int:
        return self._collection.count()

    def add_document(
        self, doc_id: str, title: str, text: str, tenant_id: str = "default"
    ) -> int:
        """增量导入/覆盖单篇文档: 先删旧 chunk → 分块 → 打标签 → 向量化 → 入库"""
        from langchain_core.documents import Document

        safe_id = _ID_SAFE.sub("_", doc_id) or "doc"
        self._delete_chunks_by_doc(safe_id, tenant_id)  # upsert 语义: 覆盖旧版本

        docs = [Document(page_content=text)]
        chunks = self._text_splitter.split_documents(docs)
        chunk_texts = [c.page_content for c in chunks]
        if not chunk_texts:
            logger.warning("add_document 分块结果为空 doc_id=%s", doc_id)
            return 0

        metadatas = []
        for i, ct in enumerate(chunk_texts):
            tags = [kw for kw in _TAG_KEYWORDS if kw in ct]
            metadatas.append({
                "doc_id": safe_id,
                "title": title,
                "tenant_id": tenant_id,
                "tags": " + ".join(tags) if tags else "其他",
                "chunk_index": i,
            })

        embeddings = self._embeddings.embed(chunk_texts)
        self._collection.add(
            documents=chunk_texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=[f"{self._collection_name}-{tenant_id}-{safe_id}-{i}"
                 for i in range(len(chunk_texts))],
        )
        logger.info("文档导入完成 doc_id=%s tenant=%s chunks=%d",
                    doc_id, tenant_id, len(chunk_texts))
        self._rebuild_bm25()
        return len(chunk_texts)

    def remove_document(self, doc_id: str, tenant_id: str = "default") -> int:
        safe_id = _ID_SAFE.sub("_", doc_id) or "doc"
        ids = self._chunk_ids_by_doc(safe_id, tenant_id)
        if ids:
            self._collection.delete(ids=ids)
            logger.info("文档已删除 doc_id=%s tenant=%s chunks=%d",
                        doc_id, tenant_id, len(ids))
            self._rebuild_bm25()
        return len(ids)

    def list_documents(self, tenant_id: str = "default") -> list[dict]:
        """从 chunk 元数据聚合文档清单 (本租户 + 共享租户)"""
        if self._collection.count() == 0:
            return []
        data = self._collection.get(
            where={"tenant_id": {"$in": [tenant_id, self._shared_tenant]}},
            include=["metadatas"],
        )
        metas = data.get("metadatas") or []
        by_doc: dict[str, dict] = {}
        order: list[str] = []
        for m in metas:
            doc_id = (m or {}).get("doc_id", "unknown")
            if doc_id not in by_doc:
                by_doc[doc_id] = {
                    "doc_id": doc_id,
                    "title": (m or {}).get("title", ""),
                    "tenant_id": (m or {}).get("tenant_id", ""),
                    "chunk_count": 0,
                }
                order.append(doc_id)
            by_doc[doc_id]["chunk_count"] += 1
        return [by_doc[d] for d in order]

    def clear_all(self) -> None:
        """清空全部文档 (重建场景)"""
        total = self._collection.count()
        if total:
            data = self._collection.get(include=[])
            ids = data.get("ids") or []
            if ids:
                self._collection.delete(ids=ids)
        self._bm25 = BM25Index([], [])
        logger.info("知识库已清空 (removed=%d)", total)

    # ── 检索 ─────────────────────────────────────────────
    def retrieve(
        self, query: str, top_k: int = 3, tenant_id: str = "default"
    ) -> list[RetrievedChunk]:
        """检索 (按 retrieval_mode 分派), 按租户过滤

        hybrid: 向量 + BM25 → RRF 融合 (默认)
        vector: 纯向量
        bm25:   纯 BM25 —— 保留此模式是为了能**量化混合检索的增益到底来自哪里**,
                否则只有 hybrid 与 vector 两组数字时, 无法判断 BM25 是否真的在起作用
        """
        total = self._collection.count()
        if total == 0:
            return []
        tenant_filter = {"tenant_id": {"$in": [tenant_id, self._shared_tenant]}}
        if self._mode == "hybrid" and self._bm25 is not None:
            vec = self._query_vector(query, top_k=top_k * 3, where=tenant_filter)
            bm_all = self._bm25.top_k(query, k=top_k * 3)
            # BM25 结果也按租户过滤 (元数据含 tenant_id)
            bm = [
                (pos, score) for pos, score in bm_all
                if self._bm25.metadatas[pos].get("tenant_id") in
                (tenant_id, self._shared_tenant)
            ]
            return self._hybrid_merge(vec, bm, top_k)
        if self._mode == "bm25" and self._bm25 is not None:
            return self._query_bm25(query, top_k, tenant_id)
        return self._query_vector(query, top_k=top_k, where=tenant_filter)

    def _query_bm25(
        self, query: str, top_k: int, tenant_id: str
    ) -> list[RetrievedChunk]:
        """纯 BM25 检索 (评测对照用); 分数归一化映射到 (0.4, 0.9] 作为近似相似度"""
        if self._bm25 is None:
            return []
        bm = [
            (pos, score) for pos, score in self._bm25.top_k(query, k=top_k * 3)
            if self._bm25.metadatas[pos].get("tenant_id") in
            (tenant_id, self._shared_tenant)
        ][:top_k]
        if not bm:
            return []
        max_score = max(score for _, score in bm) or 1.0
        return [
            RetrievedChunk(
                content=self._bm25.texts[pos],
                metadata=self._bm25.metadatas[pos],
                distance=0.0,
                similarity=round(0.4 + 0.5 * (score / max_score), 4),
                score_kind="bm25_norm",
            )
            for pos, score in bm
        ]

    # ── 内部: 向量查询 / RRF 融合 / BM25 维护 ─────────────
    def _query_vector(
        self, query: str, top_k: int, where: dict | None = None
    ) -> list[RetrievedChunk]:
        total = self._collection.count()
        if total == 0:
            return []
        q_emb = self._embeddings.embed([query])[0]
        results = self._collection.query(
            query_embeddings=[q_emb],
            n_results=min(top_k, total),
            where=where,
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

    def _hybrid_merge(
        self,
        vec: list[RetrievedChunk],
        bm: list[tuple[int, float]],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """RRF 融合两个 Top-N 列表; BM25 独有命中用归一化分数映射相似度"""
        rrf: dict[str, float] = {}
        vec_by_content: dict[str, RetrievedChunk] = {}
        for rank, c in enumerate(vec, 1):
            vec_by_content[c.content] = c
            rrf[c.content] = rrf.get(c.content, 0.0) + 1.0 / (_RRF_K + rank)

        bm_meta: dict[str, dict] = {}
        bm_score: dict[str, float] = {}
        max_bm = 1.0
        for rank, (pos, score) in enumerate(bm, 1):
            content = self._bm25.texts[pos]
            bm_meta[content] = self._bm25.metadatas[pos]
            bm_score[content] = score
            max_bm = max(max_bm, score)
            rrf[content] = rrf.get(content, 0.0) + 1.0 / (_RRF_K + rank)

        ranked = sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        merged: list[RetrievedChunk] = []
        for content, _ in ranked:
            c = vec_by_content.get(content)
            if c is not None:
                merged.append(c)
                continue
            # BM25 独有命中: 归一化 BM25 分数映射到 (0.4, 0.9] 作为近似相似度
            norm = bm_score.get(content, 0.0) / max_bm
            merged.append(RetrievedChunk(
                content=content,
                metadata=bm_meta.get(content, {}),
                distance=0.0,
                similarity=round(0.4 + 0.5 * norm, 4),
                score_kind="bm25_norm",
            ))
        return merged

    def _rebuild_bm25(self) -> None:
        """从当前集合全量重建 BM25 (语料小, 变更后重建成本可忽略)"""
        total = self._collection.count()
        if total == 0:
            self._bm25 = BM25Index([], [])
            return
        data = self._collection.get(include=["documents", "metadatas"])
        texts = data.get("documents") or []
        metas = data.get("metadatas") or []
        self._bm25 = BM25Index(texts, metas)

    # ── 内部: doc 过滤 ───────────────────────────────────
    def _chunk_ids_by_doc(self, doc_id: str, tenant_id: str) -> list[str]:
        data = self._collection.get(
            where={"$and": [{"doc_id": doc_id}, {"tenant_id": tenant_id}]},
            include=[],
        )
        return data.get("ids") or []

    def _delete_chunks_by_doc(self, doc_id: str, tenant_id: str) -> None:
        ids = self._chunk_ids_by_doc(doc_id, tenant_id)
        if ids:
            self._collection.delete(ids=ids)


def create_vector_store(settings, embeddings: EmbeddingClient) -> VectorStore:
    """向量库工厂 — 按配置创建实现实例 (当前仅 Chroma)"""
    settings.chroma_path.mkdir(parents=True, exist_ok=True)
    return ChromaVectorStore(
        persist_dir=str(settings.chroma_path),
        collection_name=settings.collection_name,
        embeddings=embeddings,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        retrieval_mode=settings.retrieval_mode,
        shared_tenant_id=settings.shared_tenant_id,
    )
