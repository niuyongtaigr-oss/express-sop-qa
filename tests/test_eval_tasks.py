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
