"""用户身份链路 — user_id 从哪来、以及它凭什么可信

引入用户级记忆的前提是：**user_id 必须来自认证结果，不能由调用方断言**。
理由：user_id 天然可猜（工号、姓名），而长期记忆里装的是某个人的事实与偏好 ——
断言式 user_id 等于"猜到工号就能翻别人档案"，比会话越权更严重。

所以这里有两条硬约束，各配一条测试：
  1. 租户模式下，user_id 只认 tenants.json 里绑定的那列
  2. 未启用访问控制（仅本地开发）时，才允许 X-User-Id 断言；
     一旦配了 API Key 或开了租户模式，这条断言通道必须**自动关闭**
"""

import json

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.api.deps import get_tenant_id, get_user_id
from app.config import Settings, get_settings
from app.core.security import verify_api_key
from app.services.tenant_service import TenantRegistry


def _settings(**kw) -> Settings:
    base = dict(env="dev", api_key=None, tenant_mode=False)
    base.update(kw)
    return Settings(_env_file=None, **base)


def _client(settings: Settings, registry: TenantRegistry | None = None) -> TestClient:
    app = FastAPI()

    @app.get("/whoami")
    async def whoami(
        _: None = Depends(verify_api_key),
        tenant_id: str = Depends(get_tenant_id),
        user_id: str = Depends(get_user_id),
    ):
        return {"tenant_id": tenant_id, "user_id": user_id}

    app.dependency_overrides[get_settings] = lambda: settings
    app.state.tenant_registry = registry or TenantRegistry(None)
    return TestClient(app, raise_server_exceptions=False)


def _registry(tmp_path, entries) -> TenantRegistry:
    f = tmp_path / "tenants.json"
    f.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
    return TenantRegistry(f)


# ── 租户模式: user_id 只认清单里绑定的 ────────────────────

def test_user_id_comes_from_the_key_mapping(tmp_path):
    reg = _registry(tmp_path, [
        {"api_key": "k-user", "tenant_id": "net-001", "user_id": "u-1001", "name": "张三"},
        {"api_key": "k-tenant", "tenant_id": "net-001", "name": "网点公共账号"},
    ])
    c = _client(_settings(tenant_mode=True), reg)

    r = c.get("/whoami", headers={"X-API-Key": "k-user"})
    assert r.json() == {"tenant_id": "net-001", "user_id": "u-1001"}

    # 清单里没绑 user_id 的 key → 空串 (表示"没有用户维度"), 不是报错
    r2 = c.get("/whoami", headers={"X-API-Key": "k-tenant"})
    assert r2.json() == {"tenant_id": "net-001", "user_id": ""}


def test_asserted_user_id_is_ignored_in_tenant_mode(tmp_path):
    """**关键安全断言**: 租户模式下请求头里的 X-User-Id 必须被无视

    否则任何持有效租户密钥的人, 只要把 X-User-Id 换成别人的工号, 就能读写
    那个人的长期记忆 —— user_id 就成了摆设。
    """
    reg = _registry(tmp_path, [
        {"api_key": "k1", "tenant_id": "net-001", "user_id": "u-1001"},
    ])
    c = _client(_settings(tenant_mode=True), reg)

    r = c.get("/whoami", headers={"X-API-Key": "k1", "X-User-Id": "u-9999"})
    assert r.json()["user_id"] == "u-1001", "请求头断言竟然覆盖了认证结果!"


def test_asserted_user_id_is_ignored_when_api_key_enabled(tmp_path):
    """单租户但配了 API Key: 同样不接受断言(这时没有任何可信的用户来源)"""
    c = _client(_settings(api_key="x" * 40))
    r = c.get("/whoami", headers={"X-API-Key": "x" * 40, "X-User-Id": "u-1001"})
    assert r.json()["user_id"] == ""


# ── 未启用访问控制(仅本地开发): 允许断言 ──────────────────

def test_asserted_user_id_is_accepted_only_without_access_control():
    """没有任何鉴权时不存在可信身份来源, 允许用请求头断言 —— 否则本地没法开发

    这条通道只在"未启用访问控制"时存在; prod 下无 key 会直接拒绝启动,
    所以它到不了生产。
    """
    c = _client(_settings())
    assert c.get("/whoami").json()["user_id"] == ""
    assert c.get("/whoami", headers={"X-User-Id": "u-1001"}).json()["user_id"] == "u-1001"
    # 空白要被归一掉, 否则会出现 user_id=" " 这种"看起来有值"的维度
    assert c.get("/whoami", headers={"X-User-Id": "   "}).json()["user_id"] == ""


def test_get_user_id_returns_empty_not_error(tmp_path):
    """取不到用户是**合法形态**(单租户一把共享 key), 所以返回空串。

    与 get_tenant_id 的 fail-closed 刻意不同: 租户取不到会读错数据, 必须报错;
    用户取不到只是"没有用户维度", 用户级功能据此关闭即可。
    """
    from fastapi import Request

    app = FastAPI()

    @app.get("/u")
    async def u(request: Request):
        return {"user_id": get_user_id(request)}

    assert TestClient(app).get("/u").json()["user_id"] == ""


# ── 会话层: 用户维度真的被隔离 ───────────────────────────

def test_sessions_are_isolated_between_users_in_same_tenant():
    """同一租户、同一 session_id、不同用户 → 互不可见

    这是引入 user_id 的核心目的。原先只按 (tenant, session) 索引时, 同租户内
    两个用户只要 session_id 撞上就会共享历史。
    """
    from app.services.session_service import SessionStore

    s = SessionStore(max_turns=10, max_sessions=100, ttl_s=60)
    sid = "session-abcdefgh"
    s.add_turn("t", "u-1", sid, "我是甲", "收到甲")
    s.add_turn("t", "u-2", sid, "我是乙", "收到乙")

    assert [x["content"] for x in s.get_history("t", "u-1", sid)] == ["我是甲", "收到甲"]
    assert [x["content"] for x in s.get_history("t", "u-2", sid)] == ["我是乙", "收到乙"]
    assert s.get_history("t", "u-3", sid) == []


def test_clear_only_affects_that_user():
    from app.services.session_service import SessionStore

    s = SessionStore(max_turns=10, max_sessions=100, ttl_s=60)
    sid = "session-abcdefgh"
    s.add_turn("t", "u-1", sid, "甲的问题", "甲的答案")
    s.add_turn("t", "u-2", sid, "乙的问题", "乙的答案")

    assert s.clear("t", "u-1", sid) is True
    assert s.get_history("t", "u-1", sid) == []
    assert s.get_history("t", "u-2", sid) != []      # 乙不受影响
    assert s.clear("t", "u-1", sid) is False          # 已经没了


def test_empty_user_id_is_a_distinct_dimension():
    """user_id="" 是**一个维度取值**, 不是"通配" —— 不能与具名用户互相命中"""
    from app.services.session_service import SessionStore

    s = SessionStore(max_turns=10, max_sessions=100, ttl_s=60)
    sid = "session-abcdefgh"
    s.add_turn("t", "", sid, "匿名调用的问题", "回答")
    s.add_turn("t", "u-1", sid, "甲的提问", "回答")

    assert [x["content"] for x in s.get_history("t", "", sid)] == ["匿名调用的问题", "回答"]
    assert [x["content"] for x in s.get_history("t", "u-1", sid)] == ["甲的提问", "回答"]


@pytest.mark.asyncio
async def test_chat_service_passes_user_id_to_sessions():
    """端到端: 走 ChatService 时 user_id 真的传到了会话存储

    只测 SessionStore 的话, ChatService 忘了传 user_id 测试依然全绿 ——
    而"忘了传"恰好就是这次改造要防的失败模式。
    """
    from app.services.chat_service import ChatService
    from app.services.session_service import SessionStore

    class _Graph:
        def invoke(self, inputs):
            return {"answer": "答", "intent": "rag_qa", "sources": []}

    sessions = SessionStore(max_turns=10, max_sessions=100, ttl_s=60)
    svc = ChatService(_Graph(), sessions, _settings(cache_enabled=False))
    sid = "session-abcdefgh"

    await svc.chat("甲的问题", session_id=sid, tenant_id="t", user_id="u-1")
    await svc.chat("乙的问题", session_id=sid, tenant_id="t", user_id="u-2")

    assert [x["content"] for x in sessions.get_history("t", "u-1", sid)][0] == "甲的问题"
    assert [x["content"] for x in sessions.get_history("t", "u-2", sid)][0] == "乙的问题"
