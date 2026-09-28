"""用户反馈端点 — POST /chat/feedback, GET /chat/feedback/stats

反馈落盘 JSONL (FeedbackStore), 负面反馈打告警日志;
stats 返回总量/均分/负面率 + 最近低分问题, 供客服运营与评测集扩充。

**按租户隔离**: 反馈里带用户原话 (question/answer), 与会话历史是同一类数据 ——
写入与查询都带上当前租户, 否则多租户下 A 能看到 B 的低分问题。
"""

import asyncio
import logging

from fastapi import APIRouter, Depends

from app.api.deps import (
    enforce_rate_limit,
    get_feedback_store,
    get_tenant_id,
)
from app.core.security import verify_api_key
from app.schemas.feedback import FeedbackRequest, FeedbackResponse, FeedbackStats
from app.services.feedback_service import FeedbackStore

logger = logging.getLogger(__name__)

router = APIRouter(tags=["feedback"], dependencies=[Depends(verify_api_key), Depends(enforce_rate_limit)])


@router.post("/chat/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    req: FeedbackRequest,
    store: FeedbackStore = Depends(get_feedback_store),
    tenant_id: str = Depends(get_tenant_id),
) -> FeedbackResponse:
    """提交用户对回答的反馈 (1-5 分 + 可选纠错文本)"""
    total = await asyncio.to_thread(
        store.append,
        {
            "question": req.question,
            "session_id": req.session_id,
            "answer": req.answer,
            "rating": req.rating,
            "comment": req.comment,
        },
        tenant_id,
    )
    return FeedbackResponse(stored=True, total=total)


@router.get("/chat/feedback/stats", response_model=FeedbackStats)
async def feedback_stats(
    store: FeedbackStore = Depends(get_feedback_store),
    tenant_id: str = Depends(get_tenant_id),
) -> FeedbackStats:
    """本租户的反馈统计 (总量/均分/负面率) + 最近低分问题"""
    stats = await asyncio.to_thread(store.stats, tenant_id)
    recent = await asyncio.to_thread(store.negative_questions, 10, tenant_id)
    return FeedbackStats(**stats, recent_negative=recent)
