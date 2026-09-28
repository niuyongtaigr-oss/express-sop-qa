"""P0.2 真流式测试 — astream_events 事件解析 + 真实图事件结构 + 全链路

- _handle_event: 合成 LangGraph 事件 → SSE 事件映射 (纯单元)
- 真实 LangGraph 图 + StubLLM: 验证 node 级 on_chain_end 的 name/langgraph_node 约定
- ChatService.stream: 注入伪图 (产出合成事件), 验证全链路 (done/记忆回写/错误)
"""

import asyncio

import pytest

from app.agents.graph import build_chat_graph
from app.agents.state import IntentDecision
from app.config import Settings
from app.services.chat_service import ChatService
from app.services.session_service import SessionStore

from test_graph_nodes import StubLLM, StubRag, make_settings


# ── _handle_event 单元测试 ────────────────────────────────
def _token_event(node, text):
    class _Chunk:
        content = text
    return {"event": "on_chat_model_stream",
            "metadata": {"langgraph_node": node},
            "data": {"chunk": _Chunk()}}


def _node_end_event(node, output):
    return {"event": "on_chain_end", "name": node,
            "metadata": {"langgraph_node": node}, "data": {"output": output}}


@pytest.mark.asyncio
async def test_handle_event_maps_deltas_intent_sources():
    parts: list[str] = []
    events = [
        _node_end_event("intent", {"intent": "rag_qa", "intent_reason": "涉及理赔"}),
        _token_event("intent", "不该推送给前端的 token"),
        _token_event("rag_qa", "理赔"),
        _token_event("rag_qa", "流程"),
        _node_end_event("rag_qa", {"answer": "理赔流程", "sources": [{"content": "c"}]}),
    ]
    out = []
    for ev in events:
        async for o in ChatService._handle_event(ev, parts):
            out.append(o)
    types = [o["type"] for o in out]
    assert types == ["intent", "answer_delta", "answer_delta", "sources"]
    assert out[0]["intent"] == "rag_qa"
    assert out[1]["delta"] == "理赔"
    # intent 节点的 token 被过滤
    assert "".join(parts) == "理赔流程"
    assert out[-1]["sources"] == [{"content": "c"}]


@pytest.mark.asyncio
async def test_handle_event_ignores_inner_chain_end():
    """内部子链 (name != langgraph_node) 不产生 sources 事件"""
    out = []
    inner = {"event": "on_chain_end", "name": "ChatOllama",
             "metadata": {"langgraph_node": "rag_qa"},
             "data": {"output": {"answer": "x"}}}
    async for o in ChatService._handle_event(inner, []):
        out.append(o)
    assert out == []


# ── 真实 LangGraph 图: 验证 node 完成事件约定 ────────────
@pytest.mark.asyncio
async def test_real_graph_astream_events_node_outputs():
    llm, rag = StubLLM(intent="rag_qa"), StubRag()
    graph = build_chat_graph(rag, llm, make_settings())
    seen = {}
    async for ev in graph.astream_events(
        {"question": "包裹破损怎么办?"}, version="v2"
    ):
        if ev["event"] == "on_chain_end":
            node = ev.get("metadata", {}).get("langgraph_node")
            if node and ev.get("name") == node:
                seen[node] = ev["data"].get("output")
    assert seen.get("intent", {}).get("intent") == "rag_qa"
    assert seen.get("rag_qa", {}).get("answer") == "rag answer"


# ── ChatService.stream 全链路 (伪图) ─────────────────────
class FakeGraph:
    """产出合成 astream_events 的伪图"""

    def __init__(self, events, fail=False):
        self._events = events
        self._fail = fail

    async def astream_events(self, inputs, version="v2"):
        if self._fail:
            raise RuntimeError("graph broken")
        yield _node_end_event("intent", {"intent": "rag_qa", "intent_reason": "r"})
        for ev in self._events:
            yield ev


@pytest.mark.asyncio
async def test_stream_full_pipeline_with_memory():
    sessions = SessionStore()
    graph = FakeGraph([
        _token_event("rag_qa", "理赔"),
        _token_event("rag_qa", "流程"),
        _node_end_event("rag_qa", {"answer": "理赔流程", "sources": [{"content": "c"}]}),
    ])
    svc = ChatService(graph, sessions, make_settings())
    out = [ev async for ev in svc.stream("包裹破损怎么理赔?", session_id="s1", tenant_id="tenant")]
    types = [o["type"] for o in out]
    assert types == ["intent", "answer_delta", "answer_delta", "sources", "done"]
    # 记忆回写: user + assistant 各一条
    hist = sessions.get_history("tenant", "s1")
    assert hist[0] == {"role": "user", "content": "包裹破损怎么理赔?"}
    assert hist[1] == {"role": "assistant", "content": "理赔流程"}
    assert out[-1]["intent"] == "rag_qa"


@pytest.mark.asyncio
async def test_stream_error_event_still_done():
    sessions = SessionStore()
    svc = ChatService(FakeGraph([], fail=True), sessions, make_settings())
    out = [ev async for ev in svc.stream("你好", session_id="s1", tenant_id="tenant")]
    types = [o["type"] for o in out]
    assert types == ["error", "done"]
    assert "graph broken" in out[0]["error"]
    # 出错不污染记忆
    assert sessions.get_history("tenant", "s1") == []


@pytest.mark.asyncio
async def test_stream_timeout_fallback():
    class SlowGraph:
        async def astream_events(self, inputs, version="v2"):
            await asyncio.sleep(0.2)
            yield _node_end_event("intent", {"intent": "rag_qa", "intent_reason": "r"})

    sessions = SessionStore()
    svc = ChatService(SlowGraph(), sessions, make_settings(chat_timeout_s=0.05))
    out = [ev async for ev in svc.stream("问题", session_id="s1", tenant_id="tenant")]
    assert out[0]["type"] == "answer_delta"
    assert "超时" in out[0]["delta"]
    assert out[-1]["type"] == "done"
    assert out[-1]["intent"] == "degraded"
    assert sessions.get_history("tenant", "s1") == []  # 降级不回写


# ── 客户端断连 (回归) ────────────────────────────────────
# 事件收尾曾经写在 finally 里: 客户端中途断连时 aclose() 会在 finally 处抛
# GeneratorExit, 而 finally 又试图 yield → RuntimeError:
# "async generator ignored GeneratorExit"。收尾必须在 finally 之外。


@pytest.mark.asyncio
async def test_stream_survives_client_disconnect():
    """回归: 中途断连不得抛 GeneratorExit 相关异常"""
    sessions = SessionStore()
    svc = ChatService(
        FakeGraph([_token_event("rag_qa", "理赔"), _token_event("rag_qa", "流程")]),
        sessions,
        make_settings(),
    )
    gen = svc.stream("包裹破损怎么理赔?", session_id="s1", tenant_id="tenant")
    first = await anext(gen)
    assert first["type"] == "intent"

    # 模拟客户端断开: 关掉生成器。修复前这里抛
    # RuntimeError: async generator ignored GeneratorExit
    await gen.aclose()


@pytest.mark.asyncio
async def test_stream_disconnect_does_not_write_partial_answer():
    """断连时 answer_parts 是半截的, 不能写进历史污染下一轮上下文"""
    sessions = SessionStore()
    svc = ChatService(
        FakeGraph([_token_event("rag_qa", "理赔")]),
        sessions,
        make_settings(),
    )
    gen = svc.stream("包裹破损怎么理赔?", session_id="s1", tenant_id="tenant")
    await anext(gen)                     # intent
    got = await anext(gen)               # answer_delta: "理赔"
    assert got["delta"] == "理赔"
    await gen.aclose()                   # 断连

    assert sessions.get_history("tenant", "s1") == []
