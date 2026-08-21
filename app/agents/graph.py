"""LangGraph 编排图装配 — build_chat_graph

图结构:

                 ┌→ rag_qa     (知识库问答, 走 RagService)
     intent ─────┼→ direct     (闲聊/通用问题, LLM 直接回答)
     (意图识别)   └→ multi_hop  (复杂问题, 多轮检索后回答)

依赖通过工厂参数注入 (rag_service / llm / settings), 不用全局单例。
"""

from langgraph.graph import END, START, StateGraph

from app.agents.nodes import (
    make_direct_node,
    make_intent_node,
    make_multi_hop_node,
    make_rag_qa_node,
)
from app.agents.state import ChatState
from app.config import Settings
from app.infrastructure.llm import LLMClient
from app.services.rag_service import RagService


def build_chat_graph(
    rag_service: RagService,
    llm: LLMClient,
    settings: Settings,
    intent_llm: LLMClient | None = None,
):
    """装配并编译编排图 (依赖注入入口)

    P3-C 多模型路由: intent_llm 单独指定时, 意图识别走小模型 (省成本);
    缺省 None 则与回答共用 llm。
    """

    def route(state: ChatState) -> str:
        """条件边: 按意图路由 (返回节点名, 不落 state)"""
        return state["intent"]

    intent_client = intent_llm or llm
    builder = StateGraph(ChatState)
    builder.add_node("intent", make_intent_node(intent_client))
    builder.add_node("rag_qa", make_rag_qa_node(rag_service))
    builder.add_node("direct", make_direct_node(llm))
    builder.add_node("multi_hop", make_multi_hop_node(rag_service, llm, settings))

    builder.add_edge(START, "intent")
    builder.add_conditional_edges("intent", route)
    builder.add_edge("rag_qa", END)
    builder.add_edge("direct", END)
    builder.add_edge("multi_hop", END)
    return builder.compile()
