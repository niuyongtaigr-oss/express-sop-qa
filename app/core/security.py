"""API Key 校验依赖 — X-API-Key header

- 配置了 SOP_QA_API_KEY: 业务接口必须携带正确的 X-API-Key, 否则 401
- 未配置: 放行并日志告警 (仅限本地开发, 生产环境务必配置)
- 密钥只来自环境变量 / .env, 仓库内零密钥

🏭 Java 对标: Spring Security ApiKeyFilter / OncePerRequestFilter
"""

import hmac
import logging

from fastapi import Depends, Request

from app.config import Settings, get_settings
from app.core.exceptions import UnauthorizedError

logger = logging.getLogger(__name__)

# 未配置密钥时的告警只打一次, 避免每个请求刷日志
_warned_no_key = False


async def verify_api_key(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> None:
    """FastAPI 依赖: 校验 X-API-Key (挂在需要保护的路由上)"""
    global _warned_no_key
    if not settings.api_key:
        if not _warned_no_key:
            _warned_no_key = True
            logger.warning("未配置 SOP_QA_API_KEY, API Key 校验处于放行状态 (仅限本地开发)")
        return
    provided = request.headers.get("X-API-Key", "")
    # compare_digest 防时序侧信道
    if not provided or not hmac.compare_digest(provided, settings.api_key):
        raise UnauthorizedError("缺少或错误的 X-API-Key")
