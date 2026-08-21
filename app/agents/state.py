"""LangGraph 编排状态定义 — ChatState + 意图识别结构化输出"""

from typing import Literal, TypedDict

from pydantic import BaseModel


class ChatState(TypedDict, total=False):
    """编排图在节点间传递的状态"""

    question: str
    history: list[dict]    # 多轮会话历史: [{"role": "user"|"assistant", "content": str}]
    intent: str            # rag_qa / direct / multi_hop
    intent_reason: str
    answer: str
    sources: list[dict]
    rounds: int            # multi_hop 已检索轮数


class IntentDecision(BaseModel):
    """意图识别结构化输出 (with_structured_output 的约束 schema)"""

    intent: Literal["rag_qa", "direct", "multi_hop"]
    reason: str


class RewriteQuery(BaseModel):
    """多轮检索的 query 改写输出 (LLM 改写替代硬编码拼接)"""

    rewritten_query: str = ""
    keep_original: bool = False  # 原问题已足够明确/无法改写时置 True, 提前停止
