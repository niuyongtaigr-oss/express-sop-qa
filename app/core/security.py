"""API Key 校验依赖 — X-API-Key header + 多租户解析 (P4)

两种模式:
  单租户 (tenant_mode=False):
    - 配置了 SOP_QA_API_KEY: 业务接口必须携带正确的 X-API-Key, 否则 401
    - 未配置: 放行并日志告警 (仅限本地开发)
  多租户 (tenant_mode=True):
    - X-API-Key 必须是 data/tenants.json 中某个租户的密钥 (401 拒绝未知密钥)
    - 校验通过后把 tenant_id 写入 request.state, 下游依赖取用 (检索/文档按租户隔离)
    - 不再使用 SOP_QA_API_KEY

密钥只来自环境变量 / .env / 租户清单, 仓库内零密钥。

🏭 Java 对标: Spring Security ApiKeyFilter / OncePerRequestFilter + 租户上下文
"""

import hmac
import logging

from fastapi import Depends, Request

from app.config import Settings, get_settings
from app.core.exceptions import UnauthorizedError
from app.services.tenant_service import DEFAULT_TENANT, TenantRegistry

logger = logging.getLogger(__name__)

# 未配置密钥时的告警只打一次, 避免每个请求刷日志
_warned_no_key = False


async def verify_api_key(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> None:
    """FastAPI 依赖: 校验 X-API-Key (挂在需要保护的路由上) 并解析租户上下文"""
    global _warned_no_key
    provided = request.headers.get("X-API-Key", "")

    if settings.tenant_mode:
        # 多租户: key → 租户映射
        registry: TenantRegistry = request.app.state.tenant_registry
        tenant = registry.resolve(provided or None)
        if tenant is None:
            raise UnauthorizedError("缺少或错误的 X-API-Key (未识别的租户密钥)")
        request.state.tenant_id = tenant.tenant_id
        request.state.tenant_name = tenant.name
        return

    # 单租户: 可选校验 SOP_QA_API_KEY
    request.state.tenant_id = DEFAULT_TENANT
    request.state.tenant_name = ""
    if not settings.api_key:
        if not _warned_no_key:
            _warned_no_key = True
            logger.warning("未配置 SOP_QA_API_KEY, API Key 校验处于放行状态 (仅限本地开发)")
        return
    # compare_digest 防时序侧信道
    if not provided or not hmac.compare_digest(provided, settings.api_key):
        raise UnauthorizedError("缺少或错误的 X-API-Key")


# ── 访问控制自检 ─────────────────────────────────────────

def auth_status(settings: Settings) -> str:
    """当前访问控制状态 — 供 /health 暴露, 让运维看得见自己有没有裸奔"""
    if settings.tenant_mode:
        return "tenant"
    return "enabled" if settings.api_key else "disabled"


def assert_secure_settings(settings: Settings) -> None:
    """启动自检: 生产模式必须配置访问控制 (fail-closed)

    为什么需要它: 默认是 fail-open —— 未配置 SOP_QA_API_KEY 时直接放行, 只打
    一条 warning 日志。对一个承载企业内部知识库的服务, 这是不可接受的交付形态:

      · docker-compose 的默认值是「无 key + 限流关闭」
      · 使用者照 README 起容器, 得到的是一台无鉴权、无限流的公网服务
      · 而唯一的信号是日志里一行 warning —— 没人会看见

    所以在 prod 下不再"警告后放行", 而是**拒绝启动**, 并在错误信息里给出
    具体怎么办。本地开发请显式设置 SOP_QA_ENV=dev。
    """
    if settings.env.strip().lower() != "prod":
        return
    if settings.tenant_mode:
        return  # 多租户模式靠租户清单鉴别, 不需要 api_key
    if settings.api_key:
        return
    raise RuntimeError(
        "生产模式 (SOP_QA_ENV=prod) 必须配置访问控制, 拒绝启动。\n"
        "  单租户: 设置 SOP_QA_API_KEY —— 可用 `openssl rand -hex 24` 生成\n"
        "  多租户: 设置 SOP_QA_TENANT_MODE=true 并提供 data/tenants.json\n"
        "  本地开发若确实不需要校验: 显式设置 SOP_QA_ENV=dev"
    )
