"""P3-A 答案缓存 — ChatCache 单元 + ChatService 集成 (Stub 图)"""

import asyncio
import time

import pytest

from app.config import Settings
from app.services.cache_service import ChatCache
from app.services.chat_service import ChatService
from app.services.session_service import SessionStore


def make_settings(**overrides):
    base = dict(cache_enabled=True, cache_ttl_s=300, cache_max_entries=8,
                max_concurrency=4, chat_timeout_s=10)
    base.update(overrides)
    return Settings(_env_file=None, **base)


class CountingGraph:
    """记录 invoke 次数, 返回固定结果"""

    def __init__(self, answer="默认答案"):
        self.answer = answer
        self.calls = 0

    def invoke(self, inputs):
        self.calls += 1
        return {
            "answer": self.answer,
            "intent": "rag_qa",
            "sources": [{"content": "c"}],
        }

    async def astream_events(self, inputs, version="v2"):
        yield {"event": "on_chain_end", "name": "intent",
               "metadata": {"langgraph_node": "intent"},
               "data": {"output": {"intent": "rag_qa", "intent_reason": "r"}}}


# ── ChatCache 单元 ───────────────────────────────────────
def test_cache_get_set_ttl():
    c = ChatCache(ttl_s=0.05, max_entries=8)
    key = ChatCache.key_for("破损怎么赔", 1)
    assert c.get(key) is None
    c.set(key, {"answer": "x"})
    assert c.get(key)["answer"] == "x"
    time.sleep(0.08)
    assert c.get(key) is None  # TTL 过期


def test_cache_lru_eviction():
    c = ChatCache(ttl_s=300, max_entries=2)
    k1, k2, k3 = "a", "b", "c"
    c.set(k1, {"answer": "1"})
    c.set(k2, {"answer": "2"})
    c.get(k1)  # 刷新 k1
    c.set(k3, {"answer": "3"})  # 淘汰 k2
    assert c.get(k1) is not None
    assert c.get(k2) is None
    assert c.get(k3) is not None


def test_cache_key_includes_kb_version():
    assert ChatCache.key_for("问题", 1) != ChatCache.key_for("问题", 2)
    assert ChatCache.key_for("问题A", 1) != ChatCache.key_for("问题B", 1)


def test_cache_clear_and_stats():
    c = ChatCache(max_entries=8)
    c.set("k", {"answer": "x"})
    assert c.size() == 1
    assert c.clear() == 1
    assert c.size() == 0
    assert c.stats()["max"] == 8


# ── ChatService 集成 ─────────────────────────────────────
@pytest.mark.asyncio
async def test_stateless_second_call_hits_cache():
    graph, sessions = CountingGraph(), SessionStore()
    svc = ChatService(graph, sessions, make_settings())
    r1 = await svc.chat("破损怎么赔")
    r2 = await svc.chat("破损怎么赔")
    assert r1["cached"] is False
    assert r2["cached"] is True
    assert graph.calls == 1  # 第二次未调图
    assert r2["answer"] == r1["answer"]
    assert r2["elapsed_ms"] == 0.0
    assert svc.cache_stats["hits"] == 1
    assert svc.cache_stats["misses"] == 1


@pytest.mark.asyncio
async def test_session_requests_never_cached():
    graph, sessions = CountingGraph(), SessionStore()
    svc = ChatService(graph, sessions, make_settings())
    r1 = await svc.chat("破损怎么赔", session_id="s1")
    r2 = await svc.chat("破损怎么赔", session_id="s1")
    assert r1["cached"] is False and r2["cached"] is False
    assert graph.calls == 2  # 有会话 → 不缓存, 每次都调图


@pytest.mark.asyncio
async def test_kb_version_change_invalidates_cache():
    version = {"v": 1}
    graph, sessions = CountingGraph(), SessionStore()
    svc = ChatService(graph, sessions, make_settings(),
                      get_kb_version=lambda: version["v"])
    await svc.chat("破损怎么赔")
    version["v"] = 2  # 模拟知识库变更
    r2 = await svc.chat("破损怎么赔")
    assert r2["cached"] is False
    assert graph.calls == 2


@pytest.mark.asyncio
async def test_degraded_not_cached():
    class DegradedGraph(CountingGraph):
        def invoke(self, inputs):
            self.calls += 1
            return {"answer": "超时提示", "intent": "degraded", "sources": []}

    graph, sessions = DegradedGraph(), SessionStore()
    svc = ChatService(graph, sessions, make_settings())
    await svc.chat("问题")
    await svc.chat("问题")
    assert graph.calls == 2  # 降级结果不入缓存


@pytest.mark.asyncio
async def test_cache_disabled_by_config():
    graph, sessions = CountingGraph(), SessionStore()
    svc = ChatService(graph, sessions, make_settings(cache_enabled=False))
    assert svc._cache is None
    await svc.chat("问题")
    await svc.chat("问题")
    assert graph.calls == 2


@pytest.mark.asyncio
async def test_cache_stats_shape():
    graph, sessions = CountingGraph(), SessionStore()
    svc = ChatService(graph, sessions, make_settings())
    await svc.chat("q")
    stats = svc.cache_stats
    assert stats["hits"] == 0 and stats["misses"] == 1
    assert stats["hit_rate"] == 0.0
    assert stats["size"] == 1
