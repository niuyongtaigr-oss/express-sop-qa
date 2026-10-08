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

    @classmethod
    def from_item(cls, item) -> "MemoryItemDTO":
        """从 MemoryItem 映射 —— 只此一处, 别在端点里各写一份

        (与 `SourceItem.from_chunk` 同一个理由: 两处映射就会有两处漂移,
        加字段时漏掉一处不会报错, 只会让某个接口悄悄少返回一个字段。)
        """
        return cls(
            memory_id=item.memory_id, text=item.text, kind=item.kind, key=item.key,
            importance=item.importance, created_at=item.created_at,
            updated_at=item.updated_at,
        )


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
