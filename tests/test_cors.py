"""CORS 配置 (P2)

原先**完全没有 CORS 配置** —— 浏览器端无法直接调用本服务 (同源部署则无影响,
所以不算 bug, 但也没有任何办法解决)。

默认关闭, 因为每允许一个来源就多一份暴露面, 而同源部署的前端不需要它。
配置面上的一个刻意取舍: **不提供 allow_credentials 开关** —— 本服务鉴权走
X-API-Key 请求头而非 Cookie, 不需要凭据; 把"通配来源 + 允许凭据"这个经典错配
从配置面上删掉, 比写一句文档提醒更可靠。
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import _setup_cors

TRIVIAL = "/api/v1/definitely-not-a-route"   # 404 也带 CORS 头, 且不依赖 app.state
ALLOWED = "https://admin.example.com"
OTHER = "https://evil.example.com"


def _app(value: str) -> FastAPI:
    app = FastAPI()

    @app.get("/ping")
    async def ping():
        return {"ok": True}

    _setup_cors(app, Settings(_env_file=None, cors_allow_origins=value))
    return app


# ── 默认关闭 ─────────────────────────────────────────────

def test_default_is_disabled():
    """空配置 = 不挂中间件: 连 CORS 头都不该有"""
    r = TestClient(_app("")).get(TRIVIAL, headers={"Origin": ALLOWED})
    assert "access-control-allow-origin" not in r.headers


def test_setup_cors_does_not_add_middleware_when_empty():
    """用来防止"配了空字符串却仍然挂上中间件"这种看似无害的改动"""
    app = FastAPI()
    _setup_cors(app, Settings(_env_file=None, cors_allow_origins="   ,  ,"))
    assert not any("CORSMiddleware" in str(m.cls) for m in app.user_middleware)


# ── 显式来源 ─────────────────────────────────────────────

def test_listed_origin_is_allowed():
    r = TestClient(_app(ALLOWED)).get(TRIVIAL, headers={"Origin": ALLOWED})
    assert r.headers.get("access-control-allow-origin") == ALLOWED


def test_unlisted_origin_gets_no_header():
    """没列出来的来源必须拿不到 CORS 头, 否则等于配了 *"""
    r = TestClient(_app(ALLOWED)).get(TRIVIAL, headers={"Origin": OTHER})
    assert "access-control-allow-origin" not in r.headers


def test_multiple_origins_are_parsed_from_csv():
    app = _app(f"{ALLOWED} , https://ops.example.com")
    for origin in (ALLOWED, "https://ops.example.com"):
        r = TestClient(app).get(TRIVIAL, headers={"Origin": origin})
        assert r.headers.get("access-control-allow-origin") == origin


# ── 预检 ─────────────────────────────────────────────────

def test_preflight_allows_api_key_header():
    """X-API-Key 会让浏览器发预检 —— 不放行这个头, 前端一个请求都发不出去"""
    r = TestClient(_app(ALLOWED)).options(
        TRIVIAL,
        headers={
            "Origin": ALLOWED,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "x-api-key,content-type",
        },
    )
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") == ALLOWED
    assert "POST" in r.headers.get("access-control-allow-methods", "")


# ── 凭据绝不开启 ─────────────────────────────────────────

def test_credentials_are_never_allowed_even_with_wildcard():
    """通配来源 + 允许凭据 = 任意站点携带用户 Cookie 调本服务

    本服务用 X-API-Key 鉴权, 不需要凭据, 所以配置面上根本没有这个开关。
    """
    r = TestClient(_app("*")).get(TRIVIAL, headers={"Origin": OTHER})
    assert r.headers.get("access-control-allow-origin") in ("*", OTHER)
    assert "access-control-allow-credentials" not in r.headers


# ── 真的接到了 create_app 上 ─────────────────────────────

def test_create_app_wires_cors(monkeypatch):
    """只测 _setup_cors 的话, create_app 忘了调用它测试依然全绿"""
    import app.main as main_mod

    monkeypatch.setattr(
        main_mod, "get_settings",
        lambda: Settings(_env_file=None, cors_allow_origins=ALLOWED),
    )
    r = TestClient(main_mod.create_app()).get(TRIVIAL, headers={"Origin": ALLOWED})
    assert r.headers.get("access-control-allow-origin") == ALLOWED


def test_create_app_default_has_no_cors(monkeypatch):
    import app.main as main_mod

    monkeypatch.setattr(
        main_mod, "get_settings",
        lambda: Settings(_env_file=None, cors_allow_origins=""),
    )
    r = TestClient(main_mod.create_app()).get(TRIVIAL, headers={"Origin": ALLOWED})
    assert "access-control-allow-origin" not in r.headers


# ── 交付默认值必须在"照文档复制"之后依然成立 ──────────────
# `.env.example` 里若把示例写在**值为空**的那一行后面, dotenv 不会剥离行内注释:
#   SOP_QA_CORS_ALLOW_ORIGINS=        # 例: https://admin.example.com
# 解析出来是 "# 例: https://admin.example.com" —— 一个非空字符串, 于是 CORS 中间件
# 被挂上了, 而文档承诺的是"留空即关闭"。照文档 `cp .env.example .env` 的人拿到的
# 就是与文档相反的行为。

ENV_EXAMPLE = Path(__file__).resolve().parent.parent / ".env.example"


def test_env_example_has_no_empty_value_with_inline_comment():
    """整份 .env.example 都不许出现"值为空 + 行内注释" —— 那类默认值全是假的"""
    from dotenv import dotenv_values

    values = dotenv_values(ENV_EXAMPLE)
    fake_empty = {k: v for k, v in values.items()
                  if v is not None and v.lstrip().startswith("#")}
    assert not fake_empty, f"这些配置项看起来是空的, 其实不是: {fake_empty}"


def test_copied_env_example_really_disables_cors():
    """端到端: 照文档复制 .env.example 之后, CORS 必须真的是关闭的"""
    settings = Settings(_env_file=ENV_EXAMPLE)
    assert settings.cors_allow_origins == ""

    app = FastAPI()
    _setup_cors(app, settings)
    assert not any("CORSMiddleware" in str(m.cls) for m in app.user_middleware)
