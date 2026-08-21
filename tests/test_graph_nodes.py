"""编排图节点测试 — 意图路由 / 多轮记忆注入 / LLM query 改写 (P0.1+P0.4)

用 StubLLM + StubRag 驱动真实 LangGraph 图, 无需 Ollama。
"""

import pytest

from app.agents.graph import build_chat_graph
from app.agents.state import IntentDecision, RewriteQuery
from app.config import Settings


def make_settings(**overrides):
    base = dict(
        multi_hop_max_rounds=2,
        multi_hop_similarity_threshold=0.6,
        top_k=3,
    )
    base.update(overrides)
    return Settings(_env_file=None, **base)


class StubLLM:
    """协议级桩: 记录收到的消息, 按 schema 返回预设结构化输出"""

    def __init__(self, intent="rag_qa", rewrite=None):
        self.intent = intent
        self.rewrite = rewrite
        self.invoked: list = []  # 每次调用的 messages

    def invoke(self, messages):
        self.invoked.append(("invoke", messages))
        return "stub answer"

    def structured_invoke(self, schema, messages):
        self.invoked.append(("structured", schema, messages))
        if schema is IntentDecision:
            return IntentDecision(intent=self.intent, reason="测试原因")
        if schema is RewriteQuery:
            return self.rewrite or RewriteQuery(rewritten_query="破损 理赔 流程 时限")
        raise AssertionError(f"未知 schema: {schema}")

    async def astream(self, messages):
        for ch in ["你", "好"]:
            yield ch


class StubRag:
    """桩 RAG 服务: 记录 ask/retrieve/generate 入参"""

    def __init__(self, similarity=0.9):
        self.similarity = similarity
        self.asks: list = []
        self.retrieves: list = []
        self.generates: list = []

    def _chunk(self):
        return {
            "content": "包裹破损需拍照留存证据并联系网点",
            "metadata": {"doc_id": "sop", "title": "默认", "tags": "破损"},
            "distance": 0.1,
            "similarity": self.similarity,
        }

    def ask(self, query, top_k=None, history=None):
        self.asks.append((query, history))
        return {
            "answer": "rag answer",
            "sources": [{"content": "c", "doc_id": "sop", "title": "默认", "tags": "", "similarity": 0.9}],
        }

    def retrieve(self, query, top_k=None):
        self.retrieves.append(query)
        return [self._chunk()]

    def generate(self, query, chunks, history=None):
        self.generates.append((query, chunks, history))
        return "multi answer"


def _build(rag, llm):
    return build_chat_graph(rag, llm, make_settings())


def test_routes_to_rag_qa():
    llm, rag = StubLLM(intent="rag_qa"), StubRag()
    result = _build(rag, llm).invoke({"question": "包裹破损了怎么办?"})
    assert result["intent"] == "rag_qa"
    assert result["answer"] == "rag answer"
    assert rag.asks  # rag_qa 节点调用了 rag_service.ask
    assert not rag.generates


def test_routes_to_direct():
    llm, rag = StubLLM(intent="direct"), StubRag()
    result = _build(rag, llm).invoke({"question": "你好"})
    assert result["intent"] == "direct"
    assert result["answer"] == "stub answer"
    assert not rag.asks  # 不走知识库


def test_history_passed_to_rag_ask():
    llm, rag = StubLLM(intent="rag_qa"), StubRag()
    history = [{"role": "user", "content": "包裹破损了"}, {"role": "assistant", "content": "请拍照留存"}]
    _build(rag, llm).invoke({"question": "那理赔要多久?", "history": history})
    _, h = rag.asks[0]
    assert h == history


def test_history_passed_to_direct_llm():
    llm, rag = StubLLM(intent="direct"), StubRag()
    history = [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好"}]
    _build(rag, llm).invoke({"question": "谢谢", "history": history})
    kinds = [entry[0] for entry in llm.invoked]
    assert "structured" in kinds  # 意图识别
    invoke_msgs = [entry[1] for entry in llm.invoked if entry[0] == "invoke"][0]
    # 历史被注入到直接回答的提示词中
    contents = [getattr(m, "content", "") for m in invoke_msgs]
    assert any("你好" in c for c in contents)


def test_multi_hop_llm_rewrite_second_retrieve():
    """首轮相似度不足 → LLM 改写 query → 用改写后的 query 再检索"""
    llm = StubLLM(
        intent="multi_hop",
        rewrite=RewriteQuery(rewritten_query="破损 理赔 流程 时限", keep_original=False),
    )
    rag = StubRag(similarity=0.3)  # 低于阈值 0.6
    result = _build(rag, llm).invoke({"question": "包裹破损怎么办"})
    assert rag.retrieves[0] == "包裹破损怎么办"
    assert rag.retrieves[1] == "破损 理赔 流程 时限"
    assert result["rounds"] == 2
    assert result["answer"] == "multi answer"
    assert rag.generates  # 最终走 generate


def test_multi_hop_keep_original_stops_early():
    """改写器判定无需改写 → 提前停止, 不重复检索"""
    llm = StubLLM(intent="multi_hop", rewrite=RewriteQuery(keep_original=True))
    rag = StubRag(similarity=0.3)
    result = _build(rag, llm).invoke({"question": "理赔流程是什么"})
    assert len(rag.retrieves) == 1  # 只检索了一轮
    assert result["rounds"] == 1


def test_multi_hop_high_similarity_stops_first_round():
    """首轮命中即超过阈值 → 不触发改写"""
    llm = StubLLM(intent="multi_hop", rewrite=RewriteQuery(rewritten_query="x"))
    rag = StubRag(similarity=0.95)
    result = _build(rag, llm).invoke({"question": "包裹破损怎么办"})
    assert len(rag.retrieves) == 1
    assert result["rounds"] == 1
