"""监控指标端点 — GET /api/v1/metrics (Prometheus 抓取)

与 /health 一样不做 API Key 校验 (探活/抓取类接口)。
"""

from fastapi import APIRouter, Response

from app.core.metrics import CONTENT_TYPE_LATEST, metrics_response

router = APIRouter(tags=["metrics"])


@router.get("/metrics")
async def metrics() -> Response:
    """Prometheus 文本格式指标 (application/openmetrics-text)"""
    return Response(metrics_response(), media_type=CONTENT_TYPE_LATEST)
