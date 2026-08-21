"""LangGraph 节点工厂 — intent / rag_qa / direct / multi_hop

每个节点是一个闭包工厂: 注入依赖 (llm / rag_service / settings),
返回节点函数。不用全局单例, 方便测试与多实例部署。

三种意图的选型判断:
  rag_qa    — 问题明确涉及快递 SOP 业务 (破损/理赔/拦截/罚款...)
  direct    — 打招呼、闲聊、与业务无关的通用问题
  multi_hop — 涉及多个条款组合/需要跨文档推理的复杂问题

P0.1 多轮记忆: 各节点通过 state["history"] (SessionStore 注入的最近 N 轮)
             拼接到提示词, 支持「那理赔要多久?」这类指代追问。
P0.4 LLM 改写: multi_hop 节点首轮命中不足时, 由 LLM 结合历史与已检索
             片段改写 query (结构化输出 RewriteQuery), 替代硬编码拼接。
"""

import logging

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.agents.state import ChatState, IntentDecision, RewriteQuery
from app.config import Settings
from app.infrastructure.llm import LLMClient
from app.services.rag_service import RagService

logger = logging.getLogger(__name__)

# 意图识别 system prompt (支持结合对话历史)
_INTENT_SYSTEM = (
    "你是快递客服系统的意图识别器。判断用户问题属于哪类:\n"
    "- rag_qa: 明确涉及快递 SOP 业务 (破损/遗失/理赔/拦截/罚款/操作流程)\n"
    "- direct: 打招呼、闲聊、与快递业务无关的通用问题\n"
    "- multi_hop: 需要组合多个条款/跨场景推理的复杂业务问题\n"
    "如果当前问题是针对上一轮对话的追问 (如「那理赔要多久?」), 请结合"
    "对话历史推断其业务意图, 不要因为表述不完整而误判为 direct。"
)

# 直接回答节点 system prompt (支持结合对话历史)
_DIRECT_SYSTEM = (
    "你是快递客服助手, 友好简洁地回答。\n"
    "与业务无关的问题简短回应并引导回业务。"
    "如有对话历史, 请保持上下文连贯, 但不要编造历史中不存在的信息。"
)

# 多轮检索 query 改写的 system prompt
_REWRITE_SYSTEM = (
    "你是快递知识库的检索查询改写器。多轮检索中, 上一轮检索结果不足以回答"
    "用户问题, 请把「用户问题」(必要时结合对话历史与已检索片段) 改写为一个"
    "更适合检索的查询:\n"
    "1. 补充缺失的业务维度 (如 处理流程/赔偿/时限/责任方/罚款)\n"
    "2. 保持简洁, 只保留检索所需关键词, 不要口语化\n"
    "3. 若原问题已经足够明确、没有可补充的信息, 置 keep_original=true"
)


def _history_messages(history: list[dict] | None) -> list[BaseMessage]:
    """把会话历史转成 langchain 消息 (越界/未知 role 直接丢弃)"""
    if not history:
        return []
    msgs: list[BaseMessage] = []
    for turn in history[-6:]:  # 最多带最近 6 条, 防上下文膨胀
        role = turn.get("role")
        content = turn.get("content", "")
        if not content:
            continue
        if role == "user":
            msgs.append(HumanMessage(content=content))
        elif role == "assistant":
            msgs.append(AIMessage(content=content))
    return msgs


def make_intent_node(llm: LLMClient):
    """意图识别节点: with_structured_output 约束模型输出意图类别 (含历史)"""

    def intent_node(state: ChatState) -> dict:
        decision = llm.structured_invoke(
            IntentDecision,
            [SystemMessage(content=_INTENT_SYSTEM)]
            + _history_messages(state.get("history"))
            + [HumanMessage(content=state["question"])],
        )
        logger.info("intent=%s reason=%s", decision.intent, decision.reason)
        return {"intent": decision.intent, "intent_reason": decision.reason}

    return intent_node


def make_rag_qa_node(rag_service: RagService):
    """知识库问答节点: 检索 + 生成 (生成时带对话历史)"""

    def rag_qa_node(state: ChatState) -> dict:
        result = rag_service.ask(
            state["question"], history=state.get("history")
        )
        return {"answer": result["answer"], "sources": result["sources"]}

    return rag_qa_node


def make_direct_node(llm: LLMClient):
    """直接回答节点: 闲聊/通用问题, 不走知识库 (带对话历史)"""

    def direct_node(state: ChatState) -> dict:
        answer = llm.invoke(
            [SystemMessage(content=_DIRECT_SYSTEM)]
            + _history_messages(state.get("history"))
            + [HumanMessage(content=state["question"])],
        )
        return {"answer": answer, "sources": []}

    return direct_node


def make_multi_hop_node(rag_service: RagService, llm: LLMClient, settings: Settings):
    """多轮检索节点: 检索 → 相似度阈值判断 → LLM 改写 query 再检索 (最多 N 轮)

    N 与阈值均走配置; 首轮相似度超过阈值即认为信息足够, 提前停止。
    改写由 LLM 生成 (RewriteQuery), 若模型判定无需改写则提前停止, 防死循环。
    """

    def multi_hop_node(state: ChatState) -> dict:
        all_chunks: list[dict] = []
        query = state["question"]
        history = state.get("history")
        rounds = 0
        for rnd in range(1, settings.multi_hop_max_rounds + 1):
            rounds = rnd
            chunks = rag_service.retrieve(query)
            all_chunks.extend(chunks)
            # 命中度足够 → 停止; 否则 LLM 改写 query 再检索一轮
            if chunks and chunks[0]["similarity"] > settings.multi_hop_similarity_threshold:
                break
            if rnd == settings.multi_hop_max_rounds:
                break  # 已是最后一轮, 不再改写
            decision = llm.structured_invoke(
                RewriteQuery,
                [SystemMessage(content=_REWRITE_SYSTEM)]
                + _history_messages(history)
                + [
                    HumanMessage(content=(
                        f"用户问题: {state['question']}\n"
                        f"当前检索查询: {query}\n"
                        f"已检索到的片段:\n"
                        + "\n".join(
                            f"- {c['content'][:80]}"
                            for c in chunks[:3]
                        )
                        + "\n\n请输出改写后的检索查询 (或 keep_original=true):"
                    )),
                ],
            )
            new_query = (decision.rewritten_query or "").strip()
            if decision.keep_original or not new_query or new_query == query:
                logger.info("rewrite 判定无需改写, 提前停止 (round=%d)", rnd)
                break
            query = new_query
            logger.info("multi_hop round=%d query改写: %s → %s", rnd, state["question"], query)
        answer = rag_service.generate(
            state["question"], all_chunks[: settings.top_k * 2], history=history
        )
        # 去重 sources (按内容前缀), 字段与 RagService._to_sources 对齐
        seen, sources = set(), []
        for c in all_chunks:
            key = c["content"][:50]
            if key not in seen:
                seen.add(key)
                meta = c.get("metadata", {})
                sources.append({
                    "content": c["content"],
                    "doc_id": meta.get("doc_id", ""),
                    "title": meta.get("title", ""),
                    "tags": meta.get("tags", ""),
                    "similarity": round(c["similarity"], 4),
                })
        return {"answer": answer, "sources": sources, "rounds": rounds}

    return multi_hop_node
