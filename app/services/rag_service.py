"""知识库服务 — ask / retrieve / 文档管理

组合 infrastructure 层 (VectorStore + LLMClient), 对上层暴露干净的业务接口。
本层不碰 HTTP、不碰框架细节, 只做业务编排。

P0.1 多轮记忆: ask/generate 支持注入会话历史, 回答时结合上文上下文。
P1.6 文档管理: 多文档增量导入/删除/清单 (doc_id 维度), 替代单文件全量重建。

🏭 Java 对标: Service 层 (注入 Repository 与外部客户端)
"""

import hashlib
import logging
import re
from pathlib import Path

from app.config import Settings
from app.core.exceptions import KnowledgeBaseNotReady
from app.infrastructure.document_loader import (
    DocumentLoader,
    create_document_loader,
)
from app.infrastructure.index_manifest import DocFingerprint, IndexManifest
from app.infrastructure.llm import LLMClient
from app.infrastructure.reranker import Reranker, create_reranker
from app.infrastructure.vector_store import RetrievedChunk, VectorStore

logger = logging.getLogger(__name__)

# 可直接用作 Chroma doc_id 的字符集 (与 vector_store._ID_SAFE 保持一致)
_ASCII_SAFE = re.compile(r"[A-Za-z0-9_.-]+")


def derive_doc_id(filename: str) -> str:
    """从文件名派生 Chroma 安全的 doc_id

    向量库层把非 [A-Za-z0-9_.-] 的字符统一替换为下划线, 因此中文文件名会被
    压成同样的下划线串而互相冲突 (upsert 时误删别的文档)。这里对非 ASCII
    文件名改用内容哈希 ID, 可读名称仍保留在 title 与 metadata 中。
    """
    stem = Path(filename).stem.strip() or "document"
    if _ASCII_SAFE.fullmatch(stem):
        return stem
    return "doc_" + hashlib.sha1(stem.encode("utf-8")).hexdigest()[:10]

# RAG 生成用 system prompt (严格基于参考文档回答)
_RAG_SYSTEM_PROMPT = (
    "你是一个快递行业的知识库助手。请严格根据下面提供的参考文档回答问题。\n"
    "规则:\n"
    "1. 如果参考文档中有答案，直接引用相关内容\n"
    "2. 如果文档中部分相关但不完整，基于文档推断并标注「根据文档推断」\n"
    "3. 如果文档完全不相关，诚实回答「根据现有知识库无法回答此问题」\n"
    "4. 回答简洁专业，不超过 200 字\n"
    "如有对话历史，请结合历史理解当前问题 (如指代「它/那/这个」)。"
)


def _history_text(history: list[dict] | None, limit: int = 6) -> str:
    """会话历史 → 提示词文本 (最旧在前)"""
    if not history:
        return ""
    parts = []
    for turn in history[-limit:]:
        role = "用户" if turn.get("role") == "user" else "助手"
        parts.append(f"{role}: {turn.get('content', '')}")
    return "\n".join(parts)


class RagService:
    """SOP 知识库服务 — 对上层暴露的唯一知识库入口"""

    def __init__(
        self,
        vector_store: VectorStore,
        llm: LLMClient,
        settings: Settings,
        document_loader: DocumentLoader | None = None,
        reranker: Reranker | None = None,
    ):
        self._store = vector_store
        self._llm = llm
        self._settings = settings
        self._version = 0  # 知识库版本: 任何写入变更 +1 (答案缓存失效用)
        # 文档解析器 (PDF/Word/Excel/文本): 缺省按配置创建; 显式传入便于测试替换
        self._loader = document_loader or create_document_loader(settings)
        # 重排器 (RRF 融合之后的精排): 关闭时为 NoopReranker, 调用路径保持一致
        self._reranker = reranker or create_reranker(settings, llm)

    @property
    def kb_version(self) -> int:
        return self._version

    # ── 建索引 / 文档管理 ────────────────────────────────
    def _source_documents(self) -> list[tuple[str, str, Path]]:
        """当前应当被索引的全部源文档: [(doc_id, title, path)]

        默认 SOP + data/corpus/ 下的语料文件。语料目录里每篇一个文档。
        """
        sources: list[tuple[str, str, Path]] = []
        sop = self._settings.sop_file
        if sop.exists():
            sources.append((self._settings.sop_doc_id, sop.stem, sop))

        corpus = self._settings.corpus_dir_path
        if corpus.is_dir():
            for path in sorted(corpus.iterdir()):
                if not path.is_file() or path.name.startswith("."):
                    continue
                sources.append((derive_doc_id(path.name), path.stem, path))
        return sources

    def _ingest_one(self, doc_id: str, title: str, path: Path) -> int:
        """导入单篇源文档 (统一走文档解析层), 返回 chunk 数; 失败返回 0

        注意: 即使是 .txt 也必须走解析层 —— 清洗 (去控制字符 / 合并被字距误判
        拆开的汉字 / 压缩空白) 是解析层的一部分, 绕过它会静默改变入库文本,
        进而改变切块结果。语料是纯文本不是跳过解析的理由。
        """
        try:
            doc = self._loader.load(path.name, path.read_bytes())
        except Exception as e:  # 单篇失败不拖垮整库导入
            logger.warning("语料导入失败, 跳过: %s (%s)", path.name, e)
            return 0
        return self._store.add_document(
            doc_id=doc_id,
            title=title,
            text=doc.text,
            tenant_id=self._settings.shared_tenant_id,  # 语料进共享库
        )

    def ingest(self, force: bool = False) -> tuple[int, bool]:
        """增量导入知识库: 只重建「指纹变了 / 新增 / 删除」的文档

        判定依据是 `data/chroma/index_manifest.json` 里的**文档级指纹**, 而不是
        「集合里有没有 chunk」。后者会漏掉三类静默失败 (详见 index_manifest 模块
        的 docstring): 元数据结构变了、语料改了、分块参数改了。

        返回 (chunk 总数, 是否实际发生了写入)。

        force=True 时无条件全量重建 (清单缺失 / 索引契约变更时也走全量)。
        """
        sources = self._source_documents()
        current = IndexManifest.build(
            [(doc_id, path) for doc_id, _title, path in sources],
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
        )
        titles = {doc_id: title for doc_id, title, _p in sources}

        previous = IndexManifest.load(self._settings.manifest_path)
        existing = self._store.count()

        # 全量重建的三种情形: 显式 force / 无清单 / 索引契约(schema+分块参数)变了
        if force or previous is None or not previous.contract_matches(
            self._settings.chunk_size, self._settings.chunk_overlap
        ):
            reason = (
                "force" if force
                else "无清单" if previous is None
                else f"索引契约变更 (manifest schema={previous.schema_version}, "
                     f"chunk={previous.chunk_size}/{previous.chunk_overlap})"
            )
            if existing > 0:
                self._store.clear_all()
            for doc_id, title, path in sources:
                n = self._ingest_one(doc_id, title, path)
                current.docs[doc_id] = DocFingerprint(
                    sha1=current.docs[doc_id].sha1, chunks=n
                )
            current.save(self._settings.manifest_path)
            self._version += 1
            total = self._store.count()   # 以索引实际状态为准, 不依赖累加
            logger.info("知识库全量重建完成: %d chunks, 原因=%s", total, reason)
            return total, True

        # 增量: 只处理变化的部分
        rebuild, removed = previous.diff(current)
        # 集合被清空但清单还在 (例如调过 /rag/docs 删除接口) → 也要补建
        if existing == 0:
            rebuild = sorted(current.docs)

        if not rebuild and not removed:
            logger.info("索引与语料一致 (%d chunks), 跳过导入", existing)
            return existing, False

        # 未重建的文档沿用上一版清单里的 chunk 数 —— build() 只算语料指纹,
        # 不知道 chunk 数(索引产物), 不沿用会把它清零, 清单从此与实际不符
        for doc_id in current.docs:
            if doc_id not in rebuild and doc_id in previous.docs:
                current.docs[doc_id] = DocFingerprint(
                    sha1=current.docs[doc_id].sha1,
                    chunks=previous.docs[doc_id].chunks,
                )

        for doc_id in removed:
            removed_n = self._store.remove_document(
                doc_id, tenant_id=self._settings.shared_tenant_id
            )
            logger.info("语料已删除, 移除索引: doc_id=%s chunks=%d", doc_id, removed_n)

        for doc_id in rebuild:
            path = next(p for d, _t, p in sources if d == doc_id)
            n = self._ingest_one(doc_id, titles[doc_id], path)
            current.docs[doc_id] = DocFingerprint(
                sha1=current.docs[doc_id].sha1, chunks=n
            )

        current.save(self._settings.manifest_path)
        self._version += 1
        total = self._store.count()       # 以索引实际状态为准
        logger.info(
            "知识库增量更新完成: 重建 %d 篇 %s, 删除 %d 篇 %s, 共 %d chunks",
            len(rebuild), rebuild, len(removed), removed, total,
        )
        return total, True

    def add_document(
        self, doc_id: str, title: str, content: str, tenant_id: str = "default"
    ) -> int:
        """增量导入/覆盖一篇文档 (upsert 语义), 返回 chunk 数"""
        n = self._store.add_document(doc_id, title, content, tenant_id=tenant_id)
        self._version += 1
        logger.info("add_document doc_id=%s tenant=%s chunks=%d", doc_id, tenant_id, n)
        return n

    def supported_document_extensions(self) -> list[str]:
        """文档解析层支持的扩展名 (供前端限制上传 accept, 以及错误提示)"""
        return self._loader.supported_extensions()

    def ingest_upload(
        self,
        filename: str,
        data: bytes,
        doc_id: str | None = None,
        title: str | None = None,
        tenant_id: str = "default",
    ) -> dict:
        """解析上传文件 (PDF / Word / Excel / 文本) 并增量入库。

        与 add_document 的区别: 入参是**原始文件字节**, 由文档解析层负责
        格式适配、编码识别与文本清洗 —— 客户的实际资料是 PDF / Word / Excel,
        不是现成的纯文本。

        解析失败抛 DocumentParseError (由上层转 400, 不吞掉原因)。
        返回 {doc_id, title, chunks, source, truncated}。
        """
        doc = self._loader.load(filename, data)
        resolved_id = doc_id or derive_doc_id(filename)
        resolved_title = title or Path(filename).stem or filename
        chunks = self.add_document(
            resolved_id, resolved_title, doc.text, tenant_id=tenant_id
        )
        logger.info(
            "上传文档入库 doc_id=%s title=%s chunks=%d chars=%d",
            resolved_id, resolved_title, chunks, doc.metadata.get("chars", 0),
        )
        return {
            "doc_id": resolved_id,
            "title": resolved_title,
            "chunks": chunks,
            "source": doc.metadata,
            "truncated": bool(doc.metadata.get("truncated")),
        }

    def remove_document(self, doc_id: str, tenant_id: str = "default") -> int:
        """删除一篇文档及其全部 chunk (仅限本租户), 返回删除数"""
        n = self._store.remove_document(doc_id, tenant_id=tenant_id)
        if n:
            self._version += 1
        logger.info("remove_document doc_id=%s tenant=%s removed=%d", doc_id, tenant_id, n)
        return n

    def list_documents(self, tenant_id: str = "default") -> list[dict]:
        return self._store.list_documents(tenant_id=tenant_id)

    @property
    def indexed_chunks(self) -> int:
        return self._store.count()

    @property
    def ready(self) -> bool:
        return self._store.count() > 0

    # ── 查询接口 ─────────────────────────────────────────
    def retrieve(
        self, query: str, top_k: int | None = None, tenant_id: str = "default"
    ) -> list[dict]:
        """纯检索: 粗排 (向量/BM25/RRF) → 可选精排 (重排) → top_k

        返回 [{content, metadata, distance, similarity}]。

        重排开启时会先多取候选 (rerank_candidates) 再精排截断 —— 重排的价值
        正在于「从更大的候选池里挑出真正相关的少数几条」, 只取 top_k 个候选
        再重排是没有意义的。重排本身需要 LLM, 因此该路径不再是"不调 LLM"。
        """
        self._require_ready()
        k = top_k or self._settings.top_k
        fetch_k = max(self._settings.rerank_candidates, k) if self._settings.rerank_enabled else k
        chunks = self._store.retrieve(query, top_k=fetch_k, tenant_id=tenant_id)
        if self._settings.rerank_enabled:
            chunks = self._reranker.rerank(query, chunks, k)
        return [
            {
                "content": c.content,
                "metadata": c.metadata,
                "distance": c.distance,
                "similarity": c.similarity,
            }
            for c in chunks
        ]

    def ask(
        self,
        query: str,
        top_k: int | None = None,
        history: list[dict] | None = None,
        tenant_id: str = "default",
    ) -> dict:
        """检索 + 生成一站式问答 (可带会话历史), 返回 {answer, sources}"""
        chunks = self.retrieve(query, top_k, tenant_id=tenant_id)
        answer = self.generate(query, chunks, history=history)
        return {"answer": answer, "sources": self._to_sources(chunks)}

    def generate(
        self,
        query: str,
        chunks: list[dict],
        history: list[dict] | None = None,
    ) -> str:
        """生成: 检索结果 + 会话历史 + 用户问题 → LLM 回答 (multi_hop 节点也复用)"""
        from langchain_core.messages import HumanMessage, SystemMessage

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            meta = chunk.get("metadata", {})
            tags = meta.get("tags", "")
            title = meta.get("title", "")
            context_parts.append(
                f"[参考文档 {i}] (来源: {title or '-'}, 标签: {tags})\n{chunk['content']}"
            )
        context = "\n\n".join(context_parts) if context_parts else "(无检索结果)"
        history_text = _history_text(history)
        history_block = f"\n\n对话历史:\n{history_text}" if history_text else ""
        user_prompt = (
            f"参考文档:\n{context}{history_block}\n\n"
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
            raise KnowledgeBaseNotReady("知识库未建索引, 请先调用 POST /rag/ingest 或 POST /rag/docs")

    @staticmethod
    def _to_sources(chunks: list[dict]) -> list[dict]:
        return [
            {
                "content": c["content"],
                "doc_id": c.get("metadata", {}).get("doc_id", ""),
                "title": c.get("metadata", {}).get("title", ""),
                "tags": c.get("metadata", {}).get("tags", ""),
                "similarity": round(c["similarity"], 4),
            }
            for c in chunks
        ]
