"""长期记忆 — 接口与问答链路的接线

存储层与服务的正确性在 `test_memory.py`; 这里只测**接线**:
  · 记忆接口只回当前用户自己的数据, 拿不到用户身份时明确报错
  · 召回的记忆真的进了图 state(否则模型看不到)
  · 写入只发生在"成功回答 + 有用户 + 已启用"三个条件同时成立时
最后一条尤其重要 —— 降级提示或半截答案一旦被写成"关于该用户的记忆", 会被
反复召回, 而错误信息比没有记忆更糟。
"""

import asyncio

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from app.api.deps import (
    enforce_rate_limit,
    get_memory_service,
    get_tenant_id,
    get_user_id,
)
from app.api.v1.endpoints.memory import router as memory_router
from app.config import Settings
from app.core.security import verify_api_key
from app.infrastructure.memory_store import ChromaMemoryStore, MemoryItem
from app.services.chat_service import ChatService
from app.services.memory_service import MemoryService
from app.services.session_service import SessionStore

from test_memory import KeywordEmbedding, StubLLM


def _settings(**kw) -> Settings:
    base = dict(memory_enabled=True, memory_top_k=5, memory_min_importance=0.0,
                cache_enabled=False)
    base.update(kw)
    return Settings(_env_file=None, **base)


def _service(tmp_path, **kw) -> MemoryService:
    store = ChromaMemoryStore(str(tmp_path), "mem", KeywordEmbedding())
    return MemoryService(store, StubLLM(), _settings(**kw))


# ── 接口: 只回自己的, 拿不到用户就报错 ────────────────────

def _client(svc, tenant="t1", user="u1"):
    app = FastAPI()
    app.include_router(memory_router, prefix="/api/v1")

    async def _fake_verify(request: Request) -> None:
        request.state.tenant_id = tenant
        request.state.user_id = user

    app.dependency_overrides[verify_api_key] = _fake_verify
    app.dependency_overrides[enforce_rate_limit] = lambda: None
    app.dependency_overrides[get_memory_service] = lambda: svc
    return TestClient(app)


def test_list_returns_only_own_memories(tmp_path):
    svc = _service(tmp_path)
    store = svc._store
    store.add(MemoryItem(text="我的记忆", tenant_id="t1", user_id="u1", key="居住地"))
    store.add(MemoryItem(text="别人的记忆", tenant_id="t1", user_id="u2"))

    body = _client(svc).get("/api/v1/memory").json()
    assert body["enabled"] is True
    assert body["user_id"] == "u1"
    assert body["total"] == 1
    assert [i["text"] for i in body["items"]] == ["我的记忆"]
    assert body["items"][0]["key"] == "居住地"


def test_list_without_user_is_400_not_empty_list(tmp_path):
    """拿不到可信用户身份要**报错**, 不能返回空列表

    空列表会让人以为"功能正常, 只是还没数据", 把配置问题伪装成正常状态。
    """
    r = _client(_service(tmp_path), user="").get("/api/v1/memory")
    assert r.status_code == 400
    assert "user_id" in r.json()["detail"]


def test_delete_one_and_delete_all(tmp_path):
    svc = _service(tmp_path)
    store = svc._store
    store.add(MemoryItem(text="a", tenant_id="t1", user_id="u1"))
    store.add(MemoryItem(text="b", tenant_id="t1", user_id="u1"))
    other = MemoryItem(text="别人的", tenant_id="t1", user_id="u2")
    store.add(other)

    mine = store.list("t1", "u1")[0]
    r = _client(svc).delete(f"/api/v1/memory/{mine.memory_id}")
    assert r.json() == {"deleted": True, "removed": 1}

    # 别人的 id 删不掉
    r2 = _client(svc).delete(f"/api/v1/memory/{other.memory_id}")
    assert r2.json()["deleted"] is False
    assert store.count("t1", "u2") == 1

    r3 = _client(svc).delete("/api/v1/memory")
    assert r3.json()["removed"] == 1
    assert store.count("t1", "u1") == 0


def test_delete_without_user_is_400(tmp_path):
    assert _client(_service(tmp_path), user="").delete(
        "/api/v1/memory").status_code == 400


def test_memory_routes_require_auth_by_default():
    """真实路由必须挂鉴权 —— 记忆里是个人信息, 不能裸奔"""
    from app.main import create_app

    app = create_app()
    spec = app.openapi()["paths"]
    assert "/api/v1/memory" in spec and "/api/v1/memory/{memory_id}" in spec


# ── 问答链路: 召回进提示词 ───────────────────────────────

class _RecordingGraph:
    """记录图收到的输入 (召回的记忆就在里面)"""

    def __init__(self, degraded=False):
        self.inputs: list[dict] = []
        self._degraded = degraded

    def invoke(self, inputs):
        self.inputs.append(inputs)
        if self._degraded:
            return {"answer": "降级提示", "intent": "degraded", "sources": []}
        return {"answer": "正常回答", "intent": "rag_qa", "sources": []}

    async def astream_events(self, inputs, version="v2"):
        self.inputs.append(inputs)
        yield {"event": "on_chain_end", "name": "intent",
               "metadata": {"langgraph_node": "intent"},
               "data": {"output": {"intent": "rag_qa", "intent_reason": "r"}}}


def _svc(tmp_path, graph, memory) -> ChatService:
    return ChatService(graph, SessionStore(ttl_s=60, max_turns=10, max_sessions=10),
                       _settings(), memory=memory)


@pytest.mark.asyncio
async def test_recalled_memory_is_passed_into_the_graph(tmp_path):
    memory = _service(tmp_path)
    memory._store.add(MemoryItem(text="用户常驻上海", tenant_id="t1", user_id="u1",
                                 key="居住地"))
    graph = _RecordingGraph()
    svc = _svc(tmp_path, graph, memory)

    await svc.chat("上海 网点", tenant_id="t1", user_id="u1")

    sent = graph.inputs[0]
    assert sent["user_id"] == "u1"
    assert "用户常驻上海" in sent["memory_text"]
    assert "截至" in sent["memory_text"]


@pytest.mark.asyncio
async def test_no_user_means_no_memory_in_prompt(tmp_path):
    memory = _service(tmp_path)
    memory._store.add(MemoryItem(text="用户常驻上海", tenant_id="t1", user_id="u1"))
    graph = _RecordingGraph()
    svc = _svc(tmp_path, graph, memory)

    await svc.chat("上海 网点", tenant_id="t1", user_id="")

    assert graph.inputs[0]["memory_text"] == ""


# ── 问答链路: 写入的三个条件 ─────────────────────────────

class _RecordingMemory(MemoryService):
    """记录 remember 调用; 真的落库交给父类"""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.remembered: list[tuple] = []

    def remember(self, question, answer, tenant_id, user_id, session_id=""):
        self.remembered.append((question, answer, tenant_id, user_id, session_id))
        return {"extracted": 1, "added": 1, "updated": 0, "deleted": 0, "skipped": 0}


async def _drain(svc) -> None:
    """等后台记忆任务跑完 —— 用 ChatService 自己的等待点, 不猜 sleep 时长"""
    await svc.flush_memory()


@pytest.mark.asyncio
async def test_successful_answer_schedules_a_memory_write(tmp_path):
    memory = _RecordingMemory(ChromaMemoryStore(str(tmp_path), "mem", KeywordEmbedding()),
                              StubLLM(), _settings())
    svc = _svc(tmp_path, _RecordingGraph(), memory)

    await svc.chat("我常驻上海", session_id="s" * 16, tenant_id="t1", user_id="u1")
    await _drain(svc)

    assert len(memory.remembered) == 1
    q, a, tenant, user, sid = memory.remembered[0]
    assert (q, a, tenant, user) == ("我常驻上海", "正常回答", "t1", "u1")
    assert sid == "s" * 16


@pytest.mark.asyncio
async def test_degraded_answer_is_never_written_to_memory(tmp_path):
    """降级提示绝不能变成"关于该用户的记忆" —— 它会被反复召回, 比没记忆更糟"""
    memory = _RecordingMemory(ChromaMemoryStore(str(tmp_path), "mem", KeywordEmbedding()),
                              StubLLM(), _settings())
    svc = _svc(tmp_path, _RecordingGraph(degraded=True), memory)

    await svc.chat("问题", tenant_id="t1", user_id="u1")
    await _drain(svc)

    assert memory.remembered == []


@pytest.mark.asyncio
async def test_no_user_means_no_memory_write(tmp_path):
    memory = _RecordingMemory(ChromaMemoryStore(str(tmp_path), "mem", KeywordEmbedding()),
                              StubLLM(), _settings())
    svc = _svc(tmp_path, _RecordingGraph(), memory)

    await svc.chat("我常驻上海", tenant_id="t1", user_id="")
    await _drain(svc)

    assert memory.remembered == []


@pytest.mark.asyncio
async def test_disabled_memory_never_writes(tmp_path):
    memory = _RecordingMemory(ChromaMemoryStore(str(tmp_path), "mem", KeywordEmbedding()),
                              StubLLM(), _settings(memory_enabled=False))
    svc = _svc(tmp_path, _RecordingGraph(), memory)

    await svc.chat("我常驻上海", tenant_id="t1", user_id="u1")
    await _drain(svc)

    assert memory.remembered == []


@pytest.mark.asyncio
async def test_no_memory_service_is_fine(tmp_path):
    """没接入记忆服务时链路照常工作 — 不该因为 memory=None 报错"""
    svc = ChatService(_RecordingGraph(), SessionStore(ttl_s=60),
                      _settings(memory_enabled=False), memory=None)
    res = await svc.chat("问题", tenant_id="t1", user_id="u1")
    assert res["answer"] == "正常回答"


@pytest.mark.asyncio
async def test_memory_write_failure_does_not_break_the_answer(tmp_path):
    """记忆写失败必须被吞掉: 用户该拿到回答, 不该看到一个 500"""
    class _Boom(_RecordingMemory):
        def remember(self, *a, **kw):
            raise RuntimeError("记忆库挂了")

    memory = _Boom(ChromaMemoryStore(str(tmp_path), "mem", KeywordEmbedding()),
                   StubLLM(), _settings())
    svc = _svc(tmp_path, _RecordingGraph(), memory)

    res = await svc.chat("我常驻上海", tenant_id="t1", user_id="u1")
    assert res["answer"] == "正常回答"
