"""评测任务登记表的内存上界 (P2)

`/eval/run` 每次调用都往进程内 dict 里加一条且永不清理 —— 服务长跑就是
内存泄漏。这里给登记表加上限, 超出后按插入序淘汰最旧的。

两组测试: 单元级的淘汰语义 + 接口级的**调用点确实接上了**。
只有前者的话, 把 `_prune_tasks(tasks)` 那行删掉测试依然全绿。
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.deps import enforce_rate_limit, get_settings
from app.api.v1.endpoints.eval import (
    _MAX_EVAL_TASKS,
    _prune_tasks,
    router as eval_router,
)
from app.config import Settings
from app.core.security import verify_api_key


def test_prune_keeps_registry_bounded():
    tasks = {f"t{i}": {"status": "done"} for i in range(_MAX_EVAL_TASKS + 10)}
    _prune_tasks(tasks)
    assert len(tasks) == _MAX_EVAL_TASKS


def test_prune_evicts_oldest_first():
    """dict 保持插入序, 从头部删即最旧 —— 刚提交的任务都在尾部, 不会被误删"""
    tasks = {f"t{i}": {"status": "done"} for i in range(_MAX_EVAL_TASKS + 3)}
    _prune_tasks(tasks)
    assert "t0" not in tasks                       # 最旧的三条被淘汰
    assert "t2" not in tasks
    assert "t3" in tasks
    assert f"t{_MAX_EVAL_TASKS + 2}" in tasks      # 最新的一定在


def test_prune_is_noop_below_limit():
    tasks = {f"t{i}": {"status": "done"} for i in range(3)}
    _prune_tasks(tasks)
    assert len(tasks) == 3


# ── 接口级: /eval/run 真的会触发淘汰 ──────────────────────

class _StubEvalService:
    """不跑真实检索 —— 这里只关心任务登记表的大小"""

    def run(self, top_k=5):
        return {"hit_rate": 1.0, "total": 0, "details": []}


def _make_client() -> tuple[FastAPI, TestClient]:
    app = FastAPI()
    app.include_router(eval_router, prefix="/api/v1")
    # 认证/限流与本测试无关, 直接放行; settings 单独给一份小配置
    app.dependency_overrides[verify_api_key] = lambda: None
    app.dependency_overrides[enforce_rate_limit] = lambda: None
    app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None, top_k=1)
    app.state.eval_service = _StubEvalService()
    app.state.eval_tasks = {}
    return app, TestClient(app)


def test_eval_run_endpoint_prunes_registry():
    """连打 _MAX_EVAL_TASKS+5 次 → 登记表不超过上限 (删掉调用点那行就会失败)"""
    app, client = _make_client()
    over = 5
    for _ in range(_MAX_EVAL_TASKS + over):
        assert client.post("/api/v1/eval/run").status_code == 200

    tasks = app.state.eval_tasks
    assert len(tasks) == _MAX_EVAL_TASKS
    assert len(tasks) < _MAX_EVAL_TASKS + over


# ── 被裁剪的任务不得复活 / 不得 KeyError ─────────────────
# 登记表是有上限的内存结构, 而任务在跑完之前随时可能被裁掉。原先 `_run_eval_task`
# 第一行就是 `tasks[task_id]["status"] = "running"`, 末行又直接赋值 —— 于是一个
# "跑到一半被裁掉"的任务会把自己的结果塞回去, 上限定不住; 而一个"还没开始就被
# 裁掉"的任务会抛 KeyError (这个异常还没人 await, 变成无人认领的报错)。

import pytest

from app.api.v1.endpoints.eval import _run_eval_task


class _StubEval:
    def __init__(self, on_run=None, exc=None):
        self.calls = 0
        self._on_run = on_run
        self._exc = exc

    def run(self, top_k=5):
        self.calls += 1
        if self._on_run:
            self._on_run()
        if self._exc:
            raise self._exc
        return {"hit_rate": 1.0, "total": 1, "details": []}


@pytest.mark.asyncio
async def test_missing_task_is_a_noop_not_a_keyerror():
    tasks: dict = {}
    await _run_eval_task("ghost", tasks, _StubEval(), 1)   # 不抛异常
    assert tasks == {}


@pytest.mark.asyncio
async def test_pruned_task_does_not_resurrect_itself():
    """不复活 —— 否则"上限 50"就不是硬上限"""
    tasks = {"keep": {"status": "pending"}}

    await _run_eval_task("ghost", tasks, _StubEval(), 1)
    assert "ghost" not in tasks
    assert list(tasks) == ["keep"]


@pytest.mark.asyncio
async def test_task_pruned_while_running_does_not_write_back():
    tasks = {"ghost": {"status": "pending"}}

    def _prune_during_run():
        tasks.clear()                      # 模拟并发裁剪把它挤掉

    await _run_eval_task("ghost", tasks, _StubEval(on_run=_prune_during_run), 1)
    assert "ghost" not in tasks


@pytest.mark.asyncio
async def test_normal_task_still_writes_result():
    """别把修复做成"什么都不写" """
    tasks = {"t1": {"status": "pending"}}
    await _run_eval_task("t1", tasks, _StubEval(), 1)
    assert tasks["t1"]["status"] == "done"
    assert tasks["t1"]["hit_rate"] == 1.0


@pytest.mark.asyncio
async def test_error_is_recorded_when_task_still_present():
    tasks = {"t1": {"status": "pending"}}
    await _run_eval_task("t1", tasks, _StubEval(exc=RuntimeError("boom")), 1)
    assert tasks["t1"]["status"] == "error"
    assert "boom" in tasks["t1"]["detail"]
