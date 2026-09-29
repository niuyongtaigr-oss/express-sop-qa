"""P3-B Prometheus 监控指标 — 指标生成 + 业务埋点集成"""

import asyncio

import pytest
from prometheus_client import generate_latest

from app.core.metrics import (
    CHAT_DEGRADED,
    CHAT_REQUESTS,
    HTTP_REQUESTS,
    metrics_response,
)
from app.services.chat_service import ChatService
from app.services.session_service import SessionStore


def _text() -> str:
    return generate_latest().decode("utf-8")


def test_metrics_endpoint_payload_contains_our_metrics():
    text = metrics_response().decode("utf-8")
    for name in (
        "http_requests_total",
        "http_request_duration_seconds",
        "chat_requests_total",
        "chat_duration_seconds",
        "chat_degraded_total",
        "chat_cache_hits_total",
        "ollama_up",
        "kb_chunks",
        "session_active",
        "eval_hit_rate",
    ):
        assert name in text, f"缺少指标: {name}"


def test_http_requests_counter_labels():
    HTTP_REQUESTS.labels("POST", "/api/v1/chat", "200").inc()
    HTTP_REQUESTS.labels("GET", "/api/v1/health", "200").inc()
    text = _text()
    assert 'http_requests_total{method="POST",path="/api/v1/chat",status="200"} 1.0' in text
    assert 'http_requests_total{method="GET",path="/api/v1/health",status="200"} 1.0' in text


class CountingGraph:
    def __init__(self, intent="rag_qa", degraded=False):
        self.intent = intent
        self.degraded = degraded
        self.calls = 0

    def invoke(self, inputs):
        self.calls += 1
        return {
            "answer": "答案",
            "intent": "degraded" if self.degraded else self.intent,
            "sources": [],
        }


def _settings(**kw):
    from app.config import Settings

    base = dict(cache_enabled=False, max_concurrency=4, chat_timeout_s=10)
    base.update(kw)
    return Settings(_env_file=None, **base)


def _line_value(text: str, prefix: str) -> float | None:
    """解析 Prometheus 文本中某指标行的数值"""
    for line in text.splitlines():
        if line.startswith(prefix):
            return float(line.rsplit(" ", 1)[-1])
    return None


@pytest.mark.asyncio
async def test_chat_metrics_recorded():
    svc = ChatService(CountingGraph(intent="multi_hop"), SessionStore(), _settings())
    await svc.chat("复杂问题")
    text = _text()
    assert any('intent="multi_hop"' in line for line in text.splitlines())


@pytest.mark.asyncio
async def test_degraded_metrics_recorded():
    """必须断言计数器**真的涨了**

    原先断言 `_line_value(...) is not None` —— 但 chat_degraded_total 是无标签
    Counter, prometheus_client 在零次自增时也会输出 `chat_degraded_total 0.0`,
    所以那个断言恒真: 把 `CHAT_DEGRADED.inc()` 整行删掉测试照样绿。
    无标签指标一律用"前后差值"断言, 不能用"存在性"。
    """
    before = _line_value(_text(), "chat_degraded_total ") or 0.0
    svc = ChatService(CountingGraph(degraded=True), SessionStore(), _settings())
    await svc.chat("问题")
    text = _text()
    assert any('intent="degraded"' in line for line in text.splitlines())
    assert _line_value(text, "chat_degraded_total ") == before + 1


@pytest.mark.asyncio
async def test_cache_metrics_recorded():
    svc = ChatService(CountingGraph(), SessionStore(), _settings(cache_enabled=True))
    await svc.chat("问题")
    await svc.chat("问题")  # 第二次命中
    text = _text()
    assert _line_value(text, "chat_cache_misses_total ") >= 1
    assert _line_value(text, "chat_cache_hits_total ") >= 1
