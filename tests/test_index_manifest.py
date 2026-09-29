"""索引清单与增量重建测试 (纯内存替身, 无需 Ollama)

覆盖的核心场景: 原先 `ingest()` 只看 `count() > 0` 就跳过, 导致三类静默失败。
这里的用例逐条锁住「这几种情况必须触发重建」。
"""

import json
from pathlib import Path

import pytest

from app.config import Settings
from app.infrastructure.index_manifest import (
    SCHEMA_VERSION,
    DocFingerprint,
    IndexManifest,
    file_sha1,
)
from app.services.rag_service import RagService


# ── 替身: 只记录调用的最小 VectorStore ───────────────────

class FakeStore:
    """按文本长度估算 chunk 数, 记录每一次写入/删除/清空"""

    CHUNK = 200

    def __init__(self):
        self.docs: dict[str, tuple[str, str, str]] = {}   # doc_id -> (title, text, tenant)
        self.clears = 0

    @classmethod
    def _chunks(cls, text: str) -> int:
        return max(1, -(-len(text) // cls.CHUNK))

    def count(self) -> int:
        return sum(self._chunks(t) for _ti, t, _te in self.docs.values())

    def add_document(self, doc_id, title, text, tenant_id="default"):
        self.docs[doc_id] = (title, text, tenant_id)
        return self._chunks(text)

    def remove_document(self, doc_id, tenant_id="default"):
        if doc_id not in self.docs:
            return 0
        n = self._chunks(self.docs[doc_id][1])
        del self.docs[doc_id]
        return n

    def remove_tenant(self, tenant_id):
        """只删该租户 —— 与真实实现一致 (清空全部是跨租户破坏)"""
        self.clears += 1
        for doc_id in [d for d, (_t, _x, te) in self.docs.items() if te == tenant_id]:
            del self.docs[doc_id]
        return 0


# ── 夹具: 临时语料目录 + 临时 Chroma 目录 ────────────────

@pytest.fixture
def env(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    sop = tmp_path / "sop.txt"
    sop.write_text("演示文档: 破损件需异常登记。", encoding="utf-8")
    settings = Settings(
        _env_file=None,
        corpus_dir=str(corpus),
        sop_path=str(sop),
        chroma_dir=str(tmp_path / "chroma"),
    )
    store = FakeStore()
    rag = RagService(store, None, settings)
    return rag, store, settings, corpus


def write_doc(corpus: Path, name: str, text: str) -> Path:
    path = corpus / name
    path.write_text(text, encoding="utf-8")
    return path


# ── IndexManifest 单元 ───────────────────────────────────

def test_manifest_roundtrip(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("内容", encoding="utf-8")
    m = IndexManifest.build([("a", f)], chunk_size=200, chunk_overlap=40)
    m.docs["a"] = DocFingerprint(sha1=m.docs["a"].sha1, chunks=3)
    m.save(tmp_path / "m.json")

    loaded = IndexManifest.load(tmp_path / "m.json")
    assert loaded is not None
    assert loaded.chunk_size == 200 and loaded.chunk_overlap == 40
    assert loaded.docs["a"].chunks == 3
    assert loaded.total_chunks == 3


def test_manifest_load_missing_returns_none(tmp_path):
    assert IndexManifest.load(tmp_path / "nope.json") is None


def test_manifest_load_corrupt_returns_none(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{ 这不是 json", encoding="utf-8")
    assert IndexManifest.load(bad) is None


def test_manifest_save_is_atomic(tmp_path):
    """先写 .tmp 再 replace —— 中断不会留下半截 JSON"""
    m = IndexManifest(chunk_size=200, chunk_overlap=40)
    target = tmp_path / "m.json"
    m.save(target)
    assert target.exists()
    assert not (tmp_path / "m.json.tmp").exists()


def test_file_sha1_changes_with_content(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("版本一", encoding="utf-8")
    before = file_sha1(f)
    f.write_text("版本二", encoding="utf-8")
    assert file_sha1(f) != before


def test_sha1_ignores_mtime(tmp_path):
    """只看内容 —— touch 一下不该触发重建"""
    import os
    f = tmp_path / "a.txt"
    f.write_text("内容", encoding="utf-8")
    before = file_sha1(f)
    os.utime(f, (0, 0))
    assert file_sha1(f) == before


def test_contract_matches_detects_schema_and_chunk_change():
    m = IndexManifest(schema_version=SCHEMA_VERSION, chunk_size=200, chunk_overlap=40)
    assert m.contract_matches(200, 40)
    assert not m.contract_matches(100, 40)          # chunk_size 变了
    assert not m.contract_matches(200, 10)          # chunk_overlap 变了
    old = IndexManifest(schema_version=SCHEMA_VERSION - 1, chunk_size=200, chunk_overlap=40)
    assert not old.contract_matches(200, 40)        # schema 版本变了


def test_diff_finds_added_modified_removed():
    old = IndexManifest(docs={
        "keep": DocFingerprint("same", 1),
        "change": DocFingerprint("old", 1),
        "gone": DocFingerprint("x", 1),
    })
    new = IndexManifest(docs={
        "keep": DocFingerprint("same", 9),      # chunks 变但 sha1 未变 → 不算变化
        "change": DocFingerprint("new", 1),
        "added": DocFingerprint("y", 1),
    })
    rebuild, removed = old.diff(new)
    assert rebuild == ["added", "change"]
    assert removed == ["gone"]


# ── RagService.ingest: 增量行为 ──────────────────────────

def test_first_ingest_is_full(env):
    rag, store, _s, corpus = env
    write_doc(corpus, "法规A.txt", "甲" * 500)
    total, changed = rag.ingest()
    assert changed is True
    assert len(store.docs) == 2          # sop + 法规A
    assert "sop" in store.docs
    assert total > 0


def test_second_ingest_skips_when_unchanged(env):
    """核心回归: 语料没变时必须跳过, 不能每次都全量重建"""
    rag, store, _s, corpus = env
    write_doc(corpus, "法规A.txt", "甲" * 500)
    rag.ingest()
    first_count = store.count()
    total, changed = rag.ingest()
    assert changed is False
    assert total == first_count


def test_modified_doc_triggers_partial_rebuild(env):
    """改一篇只重建那一篇 —— 这是文档级清单相对「索引级指纹」的价值"""
    rag, store, s, corpus = env
    write_doc(corpus, "法规A.txt", "甲" * 500)
    write_doc(corpus, "法规B.txt", "乙" * 500)
    rag.ingest()
    clears_before = store.clears

    write_doc(corpus, "法规A.txt", "甲" * 900)      # 只改 A
    total, changed = rag.ingest()

    assert changed is True
    assert store.clears == clears_before            # 关键: 没有发生全量清空
    assert len(store.docs) == 3                     # sop + A + B
    assert any(t == "法规B" for t, _x, _y in store.docs.values())   # B 未被清掉
    manifest = IndexManifest.load(s.manifest_path)
    assert manifest is not None
    assert manifest.total_chunks == total


def test_removed_doc_is_dropped_from_index(env):
    rag, store, s, corpus = env
    write_doc(corpus, "法规A.txt", "甲" * 300)
    b = write_doc(corpus, "法规B.txt", "乙" * 300)
    rag.ingest()
    assert len(store.docs) == 3

    b.unlink()
    total, changed = rag.ingest()
    assert changed is True
    assert len(store.docs) == 2
    assert not any(t == "法规B" for t, _x, _y in store.docs.values())


def test_chunk_size_change_forces_full_rebuild(env):
    """分块参数变了 → 增量没有意义, 必须全量"""
    rag, store, s, corpus = env
    write_doc(corpus, "法规A.txt", "甲" * 500)
    rag.ingest()
    clears_before = store.clears

    rag._settings = s.model_copy(update={"chunk_size": 100})
    total, changed = rag.ingest()
    assert changed is True
    assert store.clears == clears_before + 1        # 发生了全量清空


def test_schema_version_change_forces_full_rebuild(env, monkeypatch):
    """元数据结构变了 (如新增 tenant_id) → 旧 chunk 必须全部作废"""
    rag, store, s, corpus = env
    write_doc(corpus, "法规A.txt", "甲" * 500)
    rag.ingest()

    import app.infrastructure.index_manifest as im
    monkeypatch.setattr(im, "SCHEMA_VERSION", im.SCHEMA_VERSION + 1)
    clears_before = store.clears
    _total, changed = rag.ingest()
    assert changed is True
    assert store.clears == clears_before + 1


def test_empty_collection_with_manifest_triggers_rebuild(env):
    """集合被清空但清单还在 (例如调过删除接口) → 必须补建, 不能跳过"""
    rag, store, _s, corpus = env
    write_doc(corpus, "法规A.txt", "甲" * 500)
    rag.ingest()

    store.docs.clear()                # 模拟「清单说有、索引没有」
    _total, changed = rag.ingest()
    assert changed is True
    assert len(store.docs) == 2


def test_force_rebuilds_everything(env):
    rag, store, _s, corpus = env
    write_doc(corpus, "法规A.txt", "甲" * 500)
    rag.ingest()
    clears_before = store.clears
    _total, changed = rag.ingest(force=True)
    assert changed is True
    assert store.clears == clears_before + 1


def test_manifest_written_into_chroma_dir(env):
    """清单与索引同生命周期: 索引目录被删 → 清单一起消失 → 下次自然全量"""
    rag, _store, s, corpus = env
    write_doc(corpus, "法规A.txt", "甲" * 500)
    rag.ingest()
    assert s.manifest_path.exists()
    assert s.manifest_path.parent == s.chroma_path
    saved = json.loads(s.manifest_path.read_text(encoding="utf-8"))
    assert saved["schema_version"] == SCHEMA_VERSION
    assert saved["chunk_size"] == s.chunk_size
    assert "sop" in saved["docs"]


# ── 语料导入必须走解析层 (回归) ──────────────────────────
# 曾经为省一次文件读取, 对 .txt 直接 read_text 而绕过解析层, 结果文本清洗
# 被静默跳过 —— chunk 数变化才暴露出来。下面两条守住这个路径。

def test_txt_corpus_goes_through_normalizer(env):
    rag, store, _s, corpus = env
    # 用一个独特标记选中文档 —— fixture 的 sop.txt 里也有「破损」二字
    (corpus / "脏语料.txt").write_bytes("脏 语 料\x00 破 损 件 需 备 案".encode("utf-8"))
    rag.ingest()
    text = next(t for _ti, t, _te in store.docs.values() if "脏语料" in t)
    assert "\x00" not in text                          # 控制字符已去除
    assert "破损件需备案" in text                        # 被字距拆开的汉字已合并


def test_gbk_corpus_is_decoded(env):
    """GBK 语料要能在导入路径上被正确解码"""
    rag, store, _s, corpus = env
    (corpus / "gbk语料.txt").write_bytes("网点破损件处理规范".encode("gbk"))
    rag.ingest()
    assert any("网点破损件处理规范" in t for _ti, t, _te in store.docs.values())
