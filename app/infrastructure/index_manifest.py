"""索引清单 — 记录「每篇源文档的指纹」与索引契约, 支撑增量重建

为什么需要它: 原先 `ingest()` 只看 `count() > 0` 就跳过, 于是下面三种不一致
都发现不了, 而且**全部是静默失败**:

  1. 元数据结构变了 (如某版本新增 `tenant_id` 字段) —— 旧 chunk 缺字段, 会被
     租户过滤器**全部排除** → 检索返回空列表, 但没有任何报错。
     这是最难排查的一类: 不报错, 只是什么都不返回。
  2. 语料文件改了 (新增法规 / 修订条款) —— 索引仍是旧的, 检索结果过时。
  3. 分块参数改了 (chunk_size / chunk_overlap) —— 索引与当前配置不符。

设计取舍 —— 文档级而非索引级:
  - 索引级指纹: 改一篇文档就要全量重建 (语料变大后很浪费)
  - 文档级清单: 只重建「指纹变了 / 新增 / 删除」的那几篇
  记录里同时存 schema_version 与分块参数 —— 它们变了同样必须重建。

指纹用**文件内容的 sha1** 而不是 (大小, mtime):
  - 内容是精确的; (大小, mtime) 会被 `touch` 或重新 clone 误触发全量重建
  - 成本: 知识库语料通常在百 MB 以内, sha1 在启动时只需几十毫秒
  - 若语料大到无法承受, 再退回 (大小, mtime) —— 代价是误触发变多

🏭 Java 对标: 增量构建的任务输入快照 (Gradle task inputs fingerprint)
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# chunk 元数据结构 / 分块行为变更时手动 +1 —— 版本不一致会触发全量重建
SCHEMA_VERSION = 2

# 清单文件名 (放在 Chroma 持久化目录内, 与索引同生命周期)
MANIFEST_FILENAME = "index_manifest.json"


def file_sha1(path: Path) -> str:
    """文件内容 sha1 (分块读取, 避免大文件一次性读进内存)"""
    h = hashlib.sha1()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class DocFingerprint:
    """单篇源文档的指纹"""

    sha1: str
    chunks: int


@dataclass
class IndexManifest:
    """索引清单 — 本次索引对应的语料契约

    docs 的 key 是 doc_id (与向量库中 chunk 的 doc_id 元数据一致)。
    """

    schema_version: int = SCHEMA_VERSION
    chunk_size: int = 0
    chunk_overlap: int = 0
    docs: dict[str, DocFingerprint] = field(default_factory=dict)

    # ── 构造 ─────────────────────────────────────────────
    @classmethod
    def build(
        cls,
        sources: list[tuple[str, Path]],
        chunk_size: int,
        chunk_overlap: int,
    ) -> "IndexManifest":
        """按当前语料文件构造清单 (sources: [(doc_id, path)])"""
        # 显式传入 SCHEMA_VERSION: dataclass 字段默认值在类定义时就已绑定,
        # 依赖默认值会导致「常量和实例值各说各话」
        manifest = cls(
            schema_version=SCHEMA_VERSION,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        for doc_id, path in sources:
            try:
                manifest.docs[doc_id] = DocFingerprint(
                    sha1=file_sha1(path), chunks=0
                )
            except OSError as e:
                logger.warning("语料文件读取失败, 跳过: %s (%s)", path, e)
        return manifest

    # ── 比较 ─────────────────────────────────────────────
    def contract_matches(self, chunk_size: int, chunk_overlap: int) -> bool:
        """索引契约 (schema 版本 + 分块参数) 是否一致

        不一致时增量毫无意义 —— 全部 chunk 都得按新契约重建。
        """
        return (
            self.schema_version == SCHEMA_VERSION
            and self.chunk_size == chunk_size
            and self.chunk_overlap == chunk_overlap
        )

    def diff(self, current: "IndexManifest") -> tuple[list[str], list[str]]:
        """与当前语料对比。

        返回 (需要重建的 doc_id, 需要删除的 doc_id):
          - 需要重建: 新增的 + sha1 变化的
          - 需要删除: 清单里有、当前语料里没有的
        chunks 数不参与比较 —— 那是索引产物, 不是语料特征。
        """
        rebuild = [
            doc_id for doc_id, fp in current.docs.items()
            if doc_id not in self.docs or self.docs[doc_id].sha1 != fp.sha1
        ]
        removed = [doc_id for doc_id in self.docs if doc_id not in current.docs]
        return sorted(rebuild), sorted(removed)

    @property
    def total_chunks(self) -> int:
        return sum(fp.chunks for fp in self.docs.values())

    # ── 持久化 ───────────────────────────────────────────
    @classmethod
    def load(cls, path: Path) -> "IndexManifest | None":
        """读取清单; 不存在或损坏时返回 None (调用方按「需要全量重建」处理)"""
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            docs = {
                doc_id: DocFingerprint(**fp)
                for doc_id, fp in raw.pop("docs", {}).items()
            }
            return cls(docs=docs, **raw)
        except (json.JSONDecodeError, TypeError, OSError) as e:
            logger.warning("索引清单解析失败, 将按全量重建处理: %s (%s)", path, e)
            return None

    def save(self, path: Path) -> None:
        """写清单 (先写临时文件再原子替换, 避免中断留下半截 JSON)"""
        payload = asdict(self)
        payload["docs"] = {k: asdict(v) for k, v in self.docs.items()}
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            tmp.replace(path)
        except OSError as e:
            logger.warning("索引清单写入失败: %s (%s)", path, e)
