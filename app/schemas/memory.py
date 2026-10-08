"""长期记忆 DTO — /api/v1/memory

记忆里装的是**个人信息**, 所以这些接口一律只回当前用户自己的记忆:
隔离维度是 (tenant_id, user_id), 两者都来自认证 (见 README「身份与隔离」)。
"""

from pydantic import BaseModel, Field


class MemoryItemDTO(BaseModel):
    """一条长期记忆 (对外表示)"""

    memory_id: str
    text: str
    kind: str = Field(description="fact / preference / episode")
    key: str = Field(default="", description="事实槽位 (冲突消解用, 便于人核对)")
    importance: float = 0.0
    created_at: float = 0.0
    updated_at: float = 0.0


class MemoryListResponse(BaseModel):
    """列出当前用户的记忆"""

    enabled: bool = Field(description="长期记忆总开关是否打开")
    user_id: str = Field(description="当前请求的用户维度; 空串表示本次调用没有用户")
    total: int
    items: list[MemoryItemDTO] = []


class MemoryForgetResponse(BaseModel):
    """删除结果"""

    deleted: bool = False
    removed: int = 0
