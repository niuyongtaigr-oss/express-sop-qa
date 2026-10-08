"""长期记忆端点 — 查看 / 删除自己的记忆 (P5)

为什么必须有这几个接口: 记忆里装的是个人信息, **能写就必须能删** —— 用户要求
"忘掉我说过的事"时, 服务方得真的有办法删掉。这不是可选项, 是合规底线。

隔离: 全部操作都带 (tenant_id, user_id), 且 user_id 来自认证。拿不到可信
user_id 时**明确报错**而不是返回空列表 —— 后者会让人以为"记忆功能是好的, 只是
没数据", 把一个配置问题伪装成正常状态。
"""

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import (
    enforce_rate_limit,
    get_memory_service,
    get_tenant_id,
    get_user_id,
)
from app.core.security import verify_api_key
from app.schemas.memory import (
    MemoryForgetResponse,
    MemoryItemDTO,
    MemoryListResponse,
)
from app.services.memory_service import MemoryService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["memory"], dependencies=[Depends(verify_api_key), Depends(enforce_rate_limit)])

_NO_USER = (
    "本次调用没有可信的用户身份, 用户级记忆不可用。"
    "请在多租户模式下用绑定了 user_id 的 API Key (data/tenants.json), "
    "或仅在本地开发时用 X-User-Id 请求头断言。"
)


def _require_user(user_id: str) -> str:
    if not user_id:
        raise HTTPException(status_code=400, detail=_NO_USER)
    return user_id


@router.get("/memory", response_model=MemoryListResponse)
async def list_memory(
    memory: MemoryService = Depends(get_memory_service),
    tenant_id: str = Depends(get_tenant_id),
    user_id: str = Depends(get_user_id),
) -> MemoryListResponse:
    """查看**当前用户**的长期记忆 (按重要性与新旧排序)"""
    uid = _require_user(user_id)
    items = await asyncio.to_thread(memory.list_memories, tenant_id, uid)
    return MemoryListResponse(
        enabled=memory.enabled,
        user_id=uid,
        total=len(items),
        items=[
            MemoryItemDTO(
                memory_id=m.memory_id, text=m.text, kind=m.kind, key=m.key,
                importance=m.importance, created_at=m.created_at,
                updated_at=m.updated_at,
            )
            for m in items
        ],
    )


@router.delete("/memory/{memory_id}", response_model=MemoryForgetResponse)
async def forget_one(
    memory_id: str,
    memory: MemoryService = Depends(get_memory_service),
    tenant_id: str = Depends(get_tenant_id),
    user_id: str = Depends(get_user_id),
) -> MemoryForgetResponse:
    """删除自己的一条记忆 (别人的删不掉 —— 归属校验在存储层)"""
    uid = _require_user(user_id)
    deleted = await asyncio.to_thread(memory.forget, memory_id, tenant_id, uid)
    return MemoryForgetResponse(deleted=deleted, removed=1 if deleted else 0)


@router.delete("/memory", response_model=MemoryForgetResponse)
async def forget_all(
    memory: MemoryService = Depends(get_memory_service),
    tenant_id: str = Depends(get_tenant_id),
    user_id: str = Depends(get_user_id),
) -> MemoryForgetResponse:
    """清空**自己**的全部记忆 ("请忘掉关于我的一切")"""
    uid = _require_user(user_id)
    removed = await asyncio.to_thread(memory.forget_all, tenant_id, uid)
    logger.info("用户清空了自己的长期记忆 tenant=%s user=%s removed=%d",
                tenant_id, uid, removed)
    return MemoryForgetResponse(deleted=removed > 0, removed=removed)
