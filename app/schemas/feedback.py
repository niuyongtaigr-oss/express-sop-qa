"""反馈资源 DTO — /chat/feedback"""

from pydantic import BaseModel, Field


class FeedbackRequest(BaseModel):
    """提交一条用户反馈"""

    question: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, max_length=64)
    answer: str | None = Field(default=None, max_length=4000)
    rating: int = Field(ge=1, le=5, description="1-5 分, ≤2 视为负面")
    comment: str | None = Field(default=None, max_length=500, description="纠错/补充文本")


class FeedbackResponse(BaseModel):
    stored: bool
    total: int


class FeedbackStats(BaseModel):
    total: int
    avg_rating: float | None
    negative_count: int
    negative_rate: float
    recent_negative: list[dict] = []
