"""LangGraph 节点工厂 — intent / rag_qa / direct / multi_hop

每个节点是一个闭包工厂: 注入依赖 (llm / rag_service / settings),
返回节点函数。不用全局单例, 方便测试与多实例部署。

三种意图的选型判断:
  rag_qa    — 问题明确涉及快递 SOP 业务 (破损/理赔/拦截/罚款...)
  direct    — 打招呼、闲聊、与业务无关的通用问题
  multi_hop — 涉及多个条款组合/需要跨文档推理的复杂问题
"""

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.state import ChatState, IntentDecision
from app.config import Settings
from app.infrastructure.llm import LLMClient
from app.services.rag_service import RagService

logger = logging.getLogger(__name__)


def make_intent_node(llm: LLMClient):
    """意图识别节点: with_structured_output 约束模型输出意图类别"""

    def intent_node(state: ChatState) -> dict:
        decision = llm.structured_invoke(IntentDecision, [
            SystemMessage(content=(
                "你是快递客服系统的意图识别器。判断用户问题属于哪类:\n"
                "- rag_qa: 明确涉及快递 SOP 业务 (破损/遗失/理赔/拦截/罚款/操作流程)\n"
                "- direct: 打招呼、闲聊、与快递业务无关的通用问题\n"
                "- multi_hop: 需要组合多个条款/跨场景推理的复杂业务问题"
            )),
            HumanMessage(content=state["question"]),
        ])
        logger.info("intent=%s reason=%s", decision.intent, decision.reason)
        return {"intent": decision.intent, "intent_reason": decision.reason}

    return intent_node


def make_rag_qa_node(rag_service: RagService):
    """知识库问答节点: 检索 + 生成"""

    def rag_qa_node(state: ChatState) -> dict:
        result = rag_service.ask(state["question"])
        return {"answer": result["answer"], "sources": result["sources"]}

    return rag_qa_node


def make_direct_node(llm: LLMClient):
    """直接回答节点: 闲聊/通用问题, 不走知识库"""

    def direct_node(state: ChatState) -> dict:
        answer = llm.invoke([
            SystemMessage(content=(
                "你是快递客服助手, 友好简洁地回答。"
                "与业务无关的问题简短回应并引导回业务。"
            )),
            HumanMessage(content=state["question"]),
        ])
        return {"answer": answer, "sources": []}

    return direct_node


def make_multi_hop_node(rag_service: RagService, settings: Settings):
    """多轮检索节点: 检索 → 相似度阈值判断 → 改写 query 再检索 (最多 N 轮)

    N 与阈值均走配置; 首轮相似度超过阈值即认为信息足够, 提前停止。
    """

    def multi_hop_node(state: ChatState) -> dict:
        all_chunks: list[dict] = []
        query = state["question"]
        rounds = 0
        for rnd in range(1, settings.multi_hop_max_rounds + 1):
            rounds = rnd
            chunks = rag_service.retrieve(query)
            all_chunks.extend(chunks)
            # 命中度足够 → 停止; 否则换表述再检索一轮
            if chunks and chunks[0]["similarity"] > settings.multi_hop_similarity_threshold:
                break
            query = f"{state['question']} 处理流程 赔偿 时限"
        answer = rag_service.generate(
            state["question"], all_chunks[: settings.top_k * 2]
        )
        # 去重 sources (按内容前缀)
        seen, sources = set(), []
        for c in all_chunks:
            key = c["content"][:50]
            if key not in seen:
                seen.add(key)
                sources.append({
                    "content": c["content"],
                    "tags": c.get("metadata", {}).get("tags", ""),
                    "similarity": round(c["similarity"], 4),
                })
        return {"answer": answer, "sources": sources, "rounds": rounds}

    return multi_hop_node
