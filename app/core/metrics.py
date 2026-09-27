"""Prometheus 指标定义 — 全局单例 (进程内)

指标清单:
  http_requests_total{method,path,status}    HTTP 请求计数 (中间件)
  http_request_duration_seconds{method,path} HTTP 耗时直方图
  chat_requests_total{intent}                问答请求计数 (含 degraded)
  chat_duration_seconds                      问答耗时直方图
  chat_degraded_total                        超时降级计数
  chat_cache_hits_total / chat_cache_misses_total  答案缓存命中/未命中
  ollama_up                                  上游可达性 (health 探活写入)
  kb_chunks                                  知识库 chunk 数
  session_active                             活跃会话数
  eval_hit_rate / eval_faithfulness_avg / eval_completeness_avg  最近一轮评测
  eval_refusal_accuracy                      最近一轮评测拒答准确率
"""

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# ── HTTP 层 ──────────────────────────────────────────────
HTTP_REQUESTS = Counter(
    "http_requests_total", "HTTP 请求计数", ["method", "path", "status"]
)
HTTP_DURATION = Histogram(
    "http_request_duration_seconds", "HTTP 请求耗时",
    ["method", "path"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
)

# ── 问答层 ───────────────────────────────────────────────
CHAT_REQUESTS = Counter(
    "chat_requests_total", "问答请求计数", ["intent"]
)
CHAT_DURATION = Histogram(
    "chat_duration_seconds", "问答耗时", buckets=(1, 2, 5, 10, 20, 40, 60, 120)
)
CHAT_DEGRADED = Counter("chat_degraded_total", "超时降级计数")
CACHE_HITS = Counter("chat_cache_hits_total", "答案缓存命中")
CACHE_MISSES = Counter("chat_cache_misses_total", "答案缓存未命中")

# ── 基础设施/状态 ───────────────────────────────────────
OLLAMA_UP = Gauge("ollama_up", "Ollama 可达性 (1=up 0=down)")
KB_CHUNKS = Gauge("kb_chunks", "知识库 chunk 数")
SESSION_ACTIVE = Gauge("session_active", "活跃会话数")

# ── 评测 ─────────────────────────────────────────────────
EVAL_HIT_RATE = Gauge("eval_hit_rate", "最近一轮评测检索命中率")
EVAL_FAITHFULNESS = Gauge("eval_faithfulness_avg", "最近一轮评测忠实性均值")
EVAL_COMPLETENESS = Gauge("eval_completeness_avg", "最近一轮评测完整性均值")
EVAL_REFUSAL_ACCURACY = Gauge(
    "eval_refusal_accuracy", "最近一轮评测拒答准确率 (知识库无答案时应如实拒答)"
)


def metrics_response() -> bytes:
    """/metrics 端点响应体"""
    return generate_latest()


__all__ = [
    "CONTENT_TYPE_LATEST", "metrics_response",
    "HTTP_REQUESTS", "HTTP_DURATION",
    "CHAT_REQUESTS", "CHAT_DURATION", "CHAT_DEGRADED",
    "CACHE_HITS", "CACHE_MISSES",
    "OLLAMA_UP", "KB_CHUNKS", "SESSION_ACTIVE",
    "EVAL_HIT_RATE", "EVAL_FAITHFULNESS", "EVAL_COMPLETENESS",
    "EVAL_REFUSAL_ACCURACY",
]
