"""重排层 — 在 RRF 融合之后, 用更强的相关性判断给候选重新排序

为什么需要这一层: 向量与 BM25 都是"粗排", 只比较 query 与 chunk 的**表示**
(向量距离 / 词频), 没有真正读一遍内容。实测中 RRF 融合后 hybrid 的 R@1
(0.70) 反而低于纯 BM25 (0.80) —— 等权融合会把向量的头部结果顶上来。
重排是精排, 逐个"读"候选与问题的匹配度, 专治低 k 精度。

为什么用 LLM 而不是专用 cross-encoder (bge-reranker 等):
  - 专用 reranker 需要 torch + 数百 MB 模型, 而本项目的部署承诺是
    「Ollama 本地推理、依赖尽量轻」—— 为 rerank 引入 torch 不划算
  - LLM 已经在链路里了 (qwen2.5:7b), 复用它是零新增依赖
  - 代价要说清楚: **精度不如专用 cross-encoder, 且每次查询多一次 LLM 调用**

成本控制的关键取舍 —— 列表式 (listwise) 而非逐条式 (pointwise):
  逐条打分要对每个候选调一次 LLM, N 个候选就是 N 次调用; 列表式把全部候选
  放进**一次**调用, 让模型一次性输出所有分数。重排本身就会放大延迟,
  再乘以候选数就没法用了。

🏭 Java 对标: 查询后置处理器 (重排 chain 中的 Rescorer)
"""

from __future__ import annotations

import logging
from typing import Protocol

from pydantic import BaseModel, Field

from app.infrastructure.llm import LLMClient
from app.infrastructure.vector_store import RetrievedChunk

logger = logging.getLogger(__name__)


class RerankScore(BaseModel):
    """单条候选的相关性打分"""

    index: int = Field(description="候选片段编号 (与输入中的 [编号] 对应)")
    relevance: float = Field(ge=0, le=1, description="该片段回答问题的相关度 0-1")


class RerankResult(BaseModel):
    """列表式重排的结构化输出"""

    scores: list[RerankScore] = Field(default_factory=list, description="每个候选的分数")


_RERANK_SYSTEM = (
    "你是检索结果相关性评审员。给定用户问题和若干候选文档片段, 为每个片段打 "
    "0-1 分, 表示它在多大程度上能回答该问题。\n"
    "评分标准:\n"
    "- 1.0: 直接包含问题所需的答案\n"
    "- 0.5: 部分相关, 需要与其他片段结合才能回答\n"
    "- 0.0: 与问题无关\n"
    "严格评分: 不含所需信息的片段一律给低分, 不要因为领域相同就给高分。\n"
    "必须为**每一个**候选编号给出分数。"
)


class Reranker(Protocol):
    """重排协议 — 输入候选列表, 输出重排后的前 top_k 条"""

    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        ...


class NoopReranker:
    """关闭重排时使用 — 直接截断

    保留这个实现 (而不是在调用处写 if) 是为了让「开/关重排」走同一条代码路径,
    少一处分支就少一处只在某个配置下才暴露的 bug。
    """

    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        return chunks[:top_k]


class LLMReranker:
    """LLM 列表式重排 — 一次调用给全部候选打分"""

    def __init__(
        self,
        llm: LLMClient,
        max_candidates: int = 12,
        max_chars_per_candidate: int = 300,
    ):
        self._llm = llm
        self._max_candidates = max_candidates
        self._max_chars = max_chars_per_candidate

    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        candidates = chunks[: self._max_candidates]
        if len(candidates) <= 1:
            return candidates[:top_k]

        listing = "\n\n".join(
            f"[{i}] {c.content[: self._max_chars]}" for i, c in enumerate(candidates)
        )
        messages = [
            _system_message(),
            _human_message(f"用户问题: {query}\n\n候选片段:\n{listing}\n\n请为每个片段打分。"),
        ]

        try:
            result = self._llm.structured_invoke(RerankResult, messages)
        except Exception as e:
            # 重排失败绝不能拖垮检索 —— 回退到粗排顺序, 只是精度退回原样
            logger.warning("重排调用失败, 回退粗排顺序: %s: %s", type(e).__name__, e)
            return candidates[:top_k]

        scores = {s.index: s.relevance for s in result.scores if 0 <= s.index < len(candidates)}
        if not scores:
            logger.warning("重排未返回任何有效分数, 回退粗排顺序")
            return candidates[:top_k]

        # 稳定排序: 同分保持粗排原序; 未打分的按 0 分沉底
        order = sorted(range(len(candidates)), key=lambda i: (-scores.get(i, 0.0), i))
        ranked: list[RetrievedChunk] = []
        for i in order[:top_k]:
            chunk = candidates[i]
            chunk.metadata = {**chunk.metadata, "rerank_score": round(scores.get(i, 0.0), 4)}
            ranked.append(chunk)
        logger.info(
            "重排完成: 候选 %d → 输出 %d, 最高分 %.2f",
            len(candidates), len(ranked), max(scores.values()),
        )
        return ranked


def _system_message():
    from langchain_core.messages import SystemMessage

    return SystemMessage(content=_RERANK_SYSTEM)


def _human_message(text: str):
    from langchain_core.messages import HumanMessage

    return HumanMessage(content=text)


def create_reranker(settings, llm: LLMClient) -> Reranker:
    """重排工厂 — 按配置创建 (关闭时返回 NoopReranker)"""
    if not settings.rerank_enabled:
        return NoopReranker()
    logger.info(
        "启用 LLM 重排: candidates=%d, 每候选截断 %d 字符",
        settings.rerank_candidates, settings.rerank_max_chars,
    )
    return LLMReranker(
        llm,
        max_candidates=settings.rerank_candidates,
        max_chars_per_candidate=settings.rerank_max_chars,
    )
