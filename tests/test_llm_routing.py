"""P3-C 多模型/供应商路由 — 工厂 + 分意图注入测试"""

import pytest

from app.agents.graph import build_chat_graph
from app.agents.state import IntentDecision
from app.config import Settings
from app.infrastructure.llm import (
    OpenAILLMClient,
    OllamaLLMClient,
    create_intent_llm,
    create_llm_client,
)

from test_graph_nodes import StubRag


def make_settings(**overrides):
    base = dict(
        llm_provider="ollama", llm_model="qwen2.5:7b",
        ollama_base_url="http://localhost:11434",
        intent_model=None,
        openai_base_url="https://api.openai.com/v1",
        openai_api_key=None, openai_model=None,
        multi_hop_max_rounds=2, multi_hop_similarity_threshold=0.6, top_k=3,
    )
    base.update(overrides)
    return Settings(_env_file=None, **base)


# ── 工厂 ─────────────────────────────────────────────────
def test_factory_ollama_default():
    client = create_llm_client(make_settings())
    assert isinstance(client, OllamaLLMClient)
    assert client._model == "qwen2.5:7b"
    assert client._base_url == "http://localhost:11434"


def test_factory_openai_provider():
    client = create_llm_client(make_settings(
        llm_provider="openai",
        openai_api_key="sk-test",
        openai_base_url="https://api.deepseek.com/v1",
    ))
    assert isinstance(client, OpenAILLMClient)
    assert client._model == "qwen2.5:7b"  # openai_model 未配置 → 回退 llm_model
    assert client._base_url == "https://api.deepseek.com/v1"
    assert client._api_key == "sk-test"


def test_factory_openai_model_override():
    client = create_llm_client(make_settings(
        llm_provider="openai", openai_model="deepseek-chat",
    ))
    assert client._model == "deepseek-chat"


def test_factory_explicit_model_and_provider():
    client = create_llm_client(
        make_settings(), model="llama3:8b", provider="ollama"
    )
    assert client._model == "llama3:8b"


def test_create_intent_llm_none_when_unconfigured():
    assert create_intent_llm(make_settings()) is None


def test_create_intent_llm_with_intent_model():
    client = create_intent_llm(make_settings(intent_model="qwen2.5:3b"))
    assert isinstance(client, OllamaLLMClient)
    assert client._model == "qwen2.5:3b"


# ── 分意图注入 (图装配) ─────────────────────────────────
class IntentOnlyLLM:
    """只实现 structured_invoke (意图识别专用桩)"""

    def __init__(self, intent="rag_qa"):
        self.intent = intent
        self.structured_calls = 0
        self.invoke_calls = 0

    def structured_invoke(self, schema, messages):
        self.structured_calls += 1
        assert schema is IntentDecision
        return IntentDecision(intent=self.intent, reason="r")

    def invoke(self, messages):
        self.invoke_calls += 1
        return "should-not-be-called"

    async def astream(self, messages):
        yield "x"


class AnswerOnlyLLM:
    """只实现 invoke (回答专用桩)"""

    def __init__(self):
        self.invoke_calls = 0

    def invoke(self, messages):
        self.invoke_calls += 1
        return "回答"

    def structured_invoke(self, schema, messages):
        raise AssertionError("回答模型不应做结构化输出")

    async def astream(self, messages):
        yield "x"


def test_intent_llm_routes_intent_node():
    """配置 intent_llm 后, 意图识别走小模型, 回答走大模型"""
    intent_llm = IntentOnlyLLM(intent="direct")
    answer_llm = AnswerOnlyLLM()
    rag = StubRag()
    graph = build_chat_graph(rag, answer_llm, make_settings(), intent_llm=intent_llm)
    result = graph.invoke({"question": "你好"})
    assert intent_llm.structured_calls == 1
    assert intent_llm.invoke_calls == 0  # 意图模型不负责回答
    assert answer_llm.invoke_calls == 1  # direct 节点用回答模型
    assert result["intent"] == "direct"
    assert result["answer"] == "回答"


def test_no_intent_llm_falls_back_to_answer_llm():
    """未配置 intent_model → 意图识别与回答共用同一客户端"""
    class DualLLM(IntentOnlyLLM, AnswerOnlyLLM):
        pass

    dual = DualLLM(intent="direct")
    rag = StubRag()
    graph = build_chat_graph(rag, dual, make_settings(), intent_llm=None)
    graph.invoke({"question": "你好"})
    assert dual.structured_calls == 1
    assert dual.invoke_calls == 1  # direct 节点用同一客户端回答
