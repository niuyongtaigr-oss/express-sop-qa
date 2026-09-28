"""访问控制与启动自检 — 安全默认值回归

背景: 原实现是 fail-open —— 未配置 SOP_QA_API_KEY 时直接放行, 只打一条
warning。配合 docker-compose 的默认值 (无 key + 限流关闭), 使用者照 README
起容器得到的是一台无鉴权的公网服务, 而唯一的信号是日志里一行 warning。
这里锁住「prod 必须拒绝启动」这个行为。
"""

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.core.security import assert_secure_settings, auth_status
from app.schemas.chat import ChatRequest

KEY = "x" * 48


def _s(**kw) -> Settings:
    base = dict(env="dev", api_key=None, tenant_mode=False)
    base.update(kw)
    return Settings(_env_file=None, **base)


# ── 启动自检 ─────────────────────────────────────────────

def test_prod_without_api_key_refuses_to_start():
    with pytest.raises(RuntimeError, match="拒绝启动"):
        assert_secure_settings(_s(env="prod"))


def test_prod_error_message_is_actionable():
    """错误信息要能直接照做 —— 否则运维只会去 Google"""
    with pytest.raises(RuntimeError) as e:
        assert_secure_settings(_s(env="prod"))
    msg = str(e.value)
    assert "SOP_QA_API_KEY" in msg
    assert "SOP_QA_ENV=dev" in msg


def test_prod_with_api_key_starts():
    assert_secure_settings(_s(env="prod", api_key=KEY))


def test_prod_with_tenant_mode_starts():
    """多租户模式靠租户清单鉴别, 不需要 api_key"""
    assert_secure_settings(_s(env="prod", tenant_mode=True))


def test_dev_without_api_key_starts():
    """本地开发保持宽松 —— 但必须显式处于 dev"""
    assert_secure_settings(_s(env="dev"))


@pytest.mark.parametrize("value", ["PROD", " prod ", "Prod"])
def test_env_value_is_normalized(value):
    with pytest.raises(RuntimeError):
        assert_secure_settings(_s(env=value))


# ── /health 暴露的访问控制状态 ───────────────────────────

def test_auth_status_reflects_configuration():
    assert auth_status(_s()) == "disabled"
    assert auth_status(_s(api_key=KEY)) == "enabled"
    assert auth_status(_s(tenant_mode=True)) == "tenant"


# ── session_id 长度下限 ──────────────────────────────────
# 服务端只按租户隔离, 不区分租户内的最终用户 —— 所以 session_id 必须不可
# 猜测, 否则同一租户内谁猜到 ID 谁就能读到那段对话历史。长度下限是底线。

def test_session_id_optional():
    assert ChatRequest(question="q").session_id is None


def test_short_session_id_is_rejected():
    for sid in ("s1", "1", "a" * 15):
        with pytest.raises(ValidationError):
            ChatRequest(question="q", session_id=sid)


def test_reasonable_session_id_accepted():
    import uuid

    sid = str(uuid.uuid4())
    assert ChatRequest(question="q", session_id=sid).session_id == sid
    assert len(sid) >= 16


def test_session_id_max_length_still_enforced():
    with pytest.raises(ValidationError):
        ChatRequest(question="q", session_id="a" * 65)
