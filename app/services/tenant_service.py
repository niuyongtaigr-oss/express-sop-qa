"""租户注册表 — TenantRegistry (P4 多租户)

把 X-API-Key 映射到租户 (tenant_id), 实现按租户隔离:
- 每个租户一把独立的 API Key (data/tenants.json: [{api_key, tenant_id, name}])
- 检索/文档管理只在该租户 + 共享租户 (shared) 范围内生效
- 关闭 tenant_mode 时无租户概念, 全部走默认租户 (default)

🏭 Java 对标: 租户表 + Key→租户映射 (Saas 多租户)
"""

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_TENANT = "default"


@dataclass(frozen=True)
class Tenant:
    api_key: str
    tenant_id: str
    name: str = ""


class TenantRegistry:
    """租户清单 — JSON 文件加载, 线程安全"""

    def __init__(self, path: Path | None = None):
        self._path = path
        self._by_key: dict[str, Tenant] = {}
        self._lock = threading.Lock()
        if path is not None:
            self.reload()

    def reload(self) -> int:
        """(重新)加载租户清单, 返回租户数"""
        if self._path is None or not self._path.exists():
            logger.warning("租户清单不存在: %s (多租户模式下将拒绝所有请求)",
                           self._path)
            with self._lock:
                self._by_key = {}
            return 0
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            by_key = {}
            for item in data or []:
                key = str(item.get("api_key", "")).strip()
                if not key:
                    continue
                by_key[key] = Tenant(
                    api_key=key,
                    tenant_id=str(item.get("tenant_id", "")).strip() or DEFAULT_TENANT,
                    name=str(item.get("name", "")),
                )
            with self._lock:
                self._by_key = by_key
            logger.info("租户清单加载: %s (%d 租户)", self._path, len(by_key))
            return len(by_key)
        except (json.JSONDecodeError, OSError) as e:
            logger.error("租户清单解析失败: %s", e)
            with self._lock:
                self._by_key = {}
            return 0

    def resolve(self, api_key: str | None) -> Tenant | None:
        """按 API Key 解析租户; 未找到返回 None"""
        if not api_key:
            return None
        with self._lock:
            return self._by_key.get(api_key)

    def list_tenants(self) -> list[Tenant]:
        with self._lock:
            return list(self._by_key.values())
