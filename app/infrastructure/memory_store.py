"""长期记忆存储 — MemoryStore (P5 用户级记忆)

和知识库 (`vector_store.py`) 是**两件事**, 所以分成两个 collection:

  · 知识库: 世界知识 (法规/SOP), 读多写少, 由运维放语料, 基本不变
  · 长期记忆: 关于**某个人**的事实与偏好, 由对话写进去, 一直变、会冲突、会过期

混在一个 collection 里会同时坏掉两件事: 检索串味(把"某用户说的偏好"当成法规引用),
以及删除语义冲突(删一条记忆 vs 删一篇文档)。

隔离维度是 **(tenant_id, user_id)** 两级, 每一次读写都强制带上 —— 记忆里装的是
个人信息, 少一个维度就是跨用户泄漏。

🏭 Java 对标: 用户画像表 (带租户+用户双维索引) + 向量检索
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Protocol

logger = logging.getLogger(__name__)

# 记忆种类: 事实 / 偏好 / 经历
KIND_FACT = "fact"
KIND_PREFERENCE = "preference"
KIND_EPISODE = "episode"
KINDS = (KIND_FACT, KIND_PREFERENCE, KIND_EPISODE)


@dataclass
class MemoryItem:
    """一条长期记忆

    `memory_id` 是稳定标识 —— 冲突消解时 UPDATE/DELETE 都靠它, 所以不能像知识库
    那样用内容哈希当 id (内容一改 id 就变了)。
    """

    text: str
    tenant_id: str
    user_id: str
    kind: str = KIND_FACT
    # 事实的"槽位" (例如 居住地 / 负责区域 / 回答格式偏好)。
    # 为什么需要它: 冲突消解如果只靠向量相似度, "常驻上海"和"已搬到杭州"这类
    # **同槽位但语义相反**的新旧事实永远碰不到面 —— 它们在向量空间里几乎不相似。
    # 靠 key 精确匹配才能把该比的两条摆到一起, 让 LLM 判 UPDATE 还是 ADD。
    key: str = ""
    importance: float = 0.5
    source_session: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    memory_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])

    def to_metadata(self) -> dict:
        return {
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "kind": self.kind,
            "key": self.key,
            "importance": float(self.importance),
            "source_session": self.source_session,
            "created_at": float(self.created_at),
            "updated_at": float(self.updated_at),
        }

    @classmethod
    def from_metadata(cls, memory_id: str, text: str, meta: dict) -> "MemoryItem":
        return cls(
            memory_id=memory_id,
            text=text,
            tenant_id=meta.get("tenant_id", ""),
            user_id=meta.get("user_id", ""),
            kind=meta.get("kind", KIND_FACT),
            key=meta.get("key", ""),
            importance=float(meta.get("importance", 0.5)),
            source_session=meta.get("source_session", ""),
            created_at=float(meta.get("created_at", 0.0)),
            updated_at=float(meta.get("updated_at", 0.0)),
        )


class MemoryStore(Protocol):
    """长期记忆存储协议 — 业务层依赖的抽象"""

    def add(self, item: MemoryItem) -> None:
        """写入一条记忆 (同 memory_id 则覆盖)"""
        ...

    def update(self, item: MemoryItem) -> bool:
        """更新一条记忆的内容/元数据, 不存在返回 False"""
        ...

    def search(
        self, query: str, tenant_id: str, user_id: str, top_k: int = 5
    ) -> list[tuple[MemoryItem, float]]:
        """按语义检索该 (租户, 用户) 的记忆, 返回 [(记忆, 相似度)]"""
        ...

    def find_by_key(self, key: str, tenant_id: str, user_id: str) -> list[MemoryItem]:
        """按事实槽位精确匹配 (冲突消解用, 见 MemoryItem.key 的说明)"""
        ...

    def list(self, tenant_id: str, user_id: str, limit: int = 100) -> list[MemoryItem]:
        """列出该 (租户, 用户) 的记忆 (新→旧)"""
        ...

    def delete(self, memory_id: str, tenant_id: str, user_id: str) -> bool:
        """删除一条记忆; 必须带租户+用户, 否则就是越权删"""
        ...

    def delete_all(self, tenant_id: str, user_id: str) -> int:
        """清空该 (租户, 用户) 的全部记忆, 返回删除数"""
        ...

    def count(self, tenant_id: str, user_id: str) -> int:
        ...

    def purge_older_than(self, deadline: float) -> int:
        """删除 updated_at 早于 deadline 的记忆, 返回条数。

        **这是唯一一个不带 (tenant, user) 的方法**, 因为它是维护操作而不是业务
        读写 —— 业务侧任何一次按 id 的删除/更新都会先校验归属 (见 delete/update)。
        """
        ...


class ChromaMemoryStore:
    """Chroma 实现 — 独立 collection, 元数据带 tenant_id + user_id

    嵌入复用知识库那套 `EmbeddingClient`, 但 collection 完全独立。
    """

    def __init__(
        self,
        persist_dir: str,
        collection_name: str,
        embeddings,
    ):
        import chromadb

        self._embeddings = embeddings
        self._collection_name = collection_name
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ── 写 ───────────────────────────────────────────────
    def add(self, item: MemoryItem) -> None:
        emb = self._embeddings.embed([item.text])[0]
        self._collection.upsert(
            ids=[item.memory_id],
            documents=[item.text],
            embeddings=[emb],
            metadatas=[item.to_metadata()],
        )

    def update(self, item: MemoryItem) -> bool:
        if not self._exists(item.memory_id, item.tenant_id, item.user_id):
            return False
        item.updated_at = time.time()
        self.add(item)   # upsert: 同 id 覆盖
        return True

    def delete(self, memory_id: str, tenant_id: str, user_id: str) -> bool:
        """删除前先校验归属 —— id 猜对了也不能删别人的"""
        if not self._exists(memory_id, tenant_id, user_id):
            return False
        self._collection.delete(ids=[memory_id])
        return True

    def delete_all(self, tenant_id: str, user_id: str) -> int:
        ids = self._ids_of(tenant_id, user_id)
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    def _exists(self, memory_id: str, tenant_id: str, user_id: str) -> bool:
        data = self._collection.get(ids=[memory_id], include=["metadatas"])
        metas = data.get("metadatas") or []
        if not metas:
            return False
        m = metas[0] or {}
        return m.get("tenant_id") == tenant_id and m.get("user_id") == user_id

    def _ids_of(self, tenant_id: str, user_id: str) -> list[str]:
        data = self._collection.get(
            where={"$and": [{"tenant_id": tenant_id}, {"user_id": user_id}]},
            include=[],
        )
        return list(data.get("ids") or [])

    # ── 读 ───────────────────────────────────────────────
    def search(
        self, query: str, tenant_id: str, user_id: str, top_k: int = 5
    ) -> list[tuple[MemoryItem, float]]:
        if self._collection.count() == 0 or not query.strip():
            return []
        emb = self._embeddings.embed([query])[0]
        res = self._collection.query(
            query_embeddings=[emb],
            # 多取一些再按 (租户, 用户) 过滤后的结果截断: 过滤发生在向量库内部,
            # 这里的 n_results 是"过滤前"的上限, 取太小可能在多租户下取空
            n_results=min(max(top_k * 4, top_k), self._collection.count()),
            where={"$and": [{"tenant_id": tenant_id}, {"user_id": user_id}]},
            include=["documents", "metadatas", "distances"],
        )
        items: list[tuple[MemoryItem, float]] = []
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        ids = (res.get("ids") or [[]])[0]
        for i in range(len(ids)):
            item = MemoryItem.from_metadata(ids[i], docs[i], metas[i] or {})
            items.append((item, round(1 - dists[i], 4)))
        # 相似度为主, 重要性做微调 —— 重要的事即使相似度略低也该浮上来
        items.sort(key=lambda p: (p[1] + 0.15 * p[0].importance), reverse=True)
        return items[:top_k]

    def find_by_key(self, key: str, tenant_id: str, user_id: str) -> list[MemoryItem]:
        if not key or self._collection.count() == 0:
            return []
        data = self._collection.get(
            where={"$and": [
                {"tenant_id": tenant_id}, {"user_id": user_id}, {"key": key},
            ]},
            include=["documents", "metadatas"],
        )
        ids = data.get("ids") or []
        docs = data.get("documents") or []
        metas = data.get("metadatas") or []
        return [MemoryItem.from_metadata(ids[i], docs[i], metas[i] or {})
                for i in range(len(ids))]

    def list(self, tenant_id: str, user_id: str, limit: int = 100) -> list[MemoryItem]:
        data = self._collection.get(
            where={"$and": [{"tenant_id": tenant_id}, {"user_id": user_id}]},
            include=["documents", "metadatas"],
        )
        ids = data.get("ids") or []
        docs = data.get("documents") or []
        metas = data.get("metadatas") or []
        items = [
            MemoryItem.from_metadata(ids[i], docs[i], metas[i] or {})
            for i in range(len(ids))
        ]
        items.sort(key=lambda m: m.updated_at, reverse=True)
        return items[:limit]

    def count(self, tenant_id: str, user_id: str) -> int:
        return len(self._ids_of(tenant_id, user_id))

    def purge_older_than(self, deadline: float) -> int:
        if self._collection.count() == 0:
            return 0
        data = self._collection.get(
            where={"updated_at": {"$lt": float(deadline)}}, include=[]
        )
        ids = list(data.get("ids") or [])
        if ids:
            self._collection.delete(ids=ids)
            logger.info("长期记忆 TTL 清理: removed=%d", len(ids))
        return len(ids)


def create_memory_store(settings, embeddings) -> MemoryStore:
    """工厂: 按配置创建记忆存储 (重依赖延迟到实例化)"""
    return ChromaMemoryStore(
        persist_dir=settings.chroma_dir,
        collection_name=settings.memory_collection_name,
        embeddings=embeddings,
    )
