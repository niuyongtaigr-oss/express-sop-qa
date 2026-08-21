"""检索质量评测服务 — 命中率 (hit_rate)

内置一组「问题 → 期望关键词」测试集, 逐题检索, 看 Top-K 结果
是否包含期望关键词, 统计命中率。评测不调 LLM, 只测检索链路。

🏭 Java 对标: 离线批量质检 Job
"""

import logging

from app.services.rag_service import RagService

logger = logging.getLogger(__name__)

# 测试集: (问题, 期望关键词) — 围绕 data/sop.txt 内容
EVAL_CASES: list[tuple[str, str]] = [
    ("包裹破损了怎么处理?", "破损"),
    ("理赔的流程是什么?", "理赔"),
    ("暴力分拣致破损罚多少钱?", "500"),
    ("拦截件处理时限是多久?", "拦截"),
    ("包裹遗失怎么赔偿?", "遗失"),
    ("客户投诉客服不跟进怎么办?", "投诉"),
]


class EvalService:
    """检索命中率评测"""

    def __init__(self, rag_service: RagService):
        self._rag = rag_service

    def run(self, top_k: int = 3) -> dict:
        """跑一轮评测, 返回 {hit_rate, top_k, details}"""
        details = []
        hits = 0
        for question, keyword in EVAL_CASES:
            chunks = self._rag.retrieve(question, top_k=top_k)
            hit = any(keyword in c["content"] for c in chunks)
            hits += hit
            details.append({
                "question": question,
                "expect": keyword,
                "hit": hit,
                "top_similarity": round(chunks[0]["similarity"], 4) if chunks else 0,
            })
        hit_rate = hits / len(EVAL_CASES) if EVAL_CASES else 0.0
        logger.info("eval_done hit_rate=%.4f cases=%d top_k=%d",
                    hit_rate, len(EVAL_CASES), top_k)
        return {"hit_rate": round(hit_rate, 4), "top_k": top_k, "details": details}
