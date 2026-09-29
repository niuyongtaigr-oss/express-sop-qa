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


# ── env 取值必须校验 (否则拼错就静默 fail-open) ───────────
# 启动自检只认精确的 "prod"。若 env 不做取值校验, `Production` / `prd` / 空串都会
# 落进 dev 分支 —— 一个 fail-closed 的安全闸门被拼写错误变成了 fail-open。

@pytest.mark.parametrize("value,expected", [
    ("PROD", "prod"), (" prod ", "prod"), ("Prod", "prod"),
    ("DEV", "dev"), ("dev", "dev"), (" Dev ", "dev"),
])
def test_env_is_normalized(value, expected):
    assert _s(env=value).env == expected


@pytest.mark.parametrize("bad", [
    "Production", "production", "prd", "prod-1", "PRODUCTION", "", "   ", "test",
])
def test_env_typo_is_rejected_at_construction(bad):
    """拼错必须在**构造配置时**就报错, 而不是启动后才让人以为受保护了"""
    with pytest.raises(ValidationError, match="env 取值非法"):
        _s(env=bad)


def test_env_typo_cannot_silently_disable_the_gate():
    """反证: 若拼错被放过, 它就会走到 assert_secure_settings 的 dev 分支

    这条锁住的是"拼错 ≠ 放行"这个性质本身。
    """
    with pytest.raises(ValidationError):
        _s(env="Production")          # 构造就失败 → 根本到不了启动自检
    # 而合法的 prod 归一化后必须被拒绝启动 (没有 key)
    assert _s(env="PROD").env == "prod"
    with pytest.raises(RuntimeError, match="拒绝启动"):
        assert_secure_settings(_s(env="PROD"))
