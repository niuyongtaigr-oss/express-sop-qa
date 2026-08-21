"""检索评测端点 — POST /api/v1/eval/run, GET /api/v1/eval/tasks/{task_id}

评测较慢 (逐题检索), 采用后台任务方式: 提交立即返回 task_id, 轮询结果。
任务登记表在内存中 (app.state.eval_tasks), 进程重启即失效。

TODO(可选优化): 任务状态持久化到 Redis/SQLite, 支持多实例共享
"""

import asyncio
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_eval_service, get_eval_tasks, get_settings
from app.config import Settings
from app.core.security import verify_api_key
from app.schemas.eval import EvalRunResponse, EvalTaskResponse
from app.services.eval_service import EvalService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["eval"], dependencies=[Depends(verify_api_key)])


async def _run_eval_task(
    task_id: str,
    tasks: dict,
    eval_service: EvalService,
    top_k: int,
) -> None:
    """后台执行评测: 检索是同步阻塞调用, 丢线程池"""
    tasks[task_id]["status"] = "running"
    try:
        result = await asyncio.to_thread(eval_service.run, top_k)
        tasks[task_id] = {"status": "done", **result}
    except Exception as e:
        logger.exception("评测任务失败 task_id=%s", task_id)
        tasks[task_id] = {"status": "error", "detail": f"{type(e).__name__}: {e}"}


@router.post("/eval/run", response_model=EvalRunResponse)
async def eval_run(
    tasks: dict = Depends(get_eval_tasks),
    eval_service: EvalService = Depends(get_eval_service),
    settings: Settings = Depends(get_settings),
) -> EvalRunResponse:
    """提交检索命中率评测 (后台任务), 立即返回 task_id"""
    task_id = uuid.uuid4().hex[:8]
    tasks[task_id] = {"status": "pending"}
    asyncio.create_task(_run_eval_task(task_id, tasks, eval_service, settings.top_k))
    return EvalRunResponse(task_id=task_id, status="pending")


@router.get("/eval/tasks/{task_id}", response_model=EvalTaskResponse)
async def eval_task(
    task_id: str,
    tasks: dict = Depends(get_eval_tasks),
) -> EvalTaskResponse:
    """查询评测任务状态/结果"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="评测任务不存在")
    return EvalTaskResponse(task_id=task_id, **tasks[task_id])
