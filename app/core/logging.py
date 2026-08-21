"""结构化日志 — JSON 行输出 + trace_id 上下文 (contextvars)

- 每条日志是单行 JSON, 方便采集到 ELK/Loki
- trace_id 存于 contextvars, 同一请求链路自动携带, 无需逐层传参

🏭 Java 对标: logback JSON encoder + MDC(traceId)
"""

import contextvars
import json
import logging
import time
import uuid

# 请求级 trace_id 上下文 (由请求日志中间件写入, 各层读取)
_trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "trace_id", default="-"
)


def new_trace_id() -> str:
    """生成新的 trace_id 并写入上下文"""
    trace_id = uuid.uuid4().hex[:12]
    _trace_id_var.set(trace_id)
    return trace_id


def set_trace_id(trace_id: str) -> None:
    _trace_id_var.set(trace_id)


def get_trace_id() -> str:
    return _trace_id_var.get()


class JsonFormatter(logging.Formatter):
    """把 LogRecord 格式化为单行 JSON"""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": round(time.time(), 3),
            "level": record.levelname,
            "logger": record.name,
            "trace_id": get_trace_id(),
            "msg": record.getMessage(),
        }
        # 业务字段通过 log_event 的 extra 传入
        fields = getattr(record, "fields", None)
        if fields:
            payload.update(fields)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(level: str = "INFO") -> None:
    """初始化根 logger (应用启动时调用一次)"""
    root = logging.getLogger()
    root.setLevel(level.upper())
    # 避免重复添加 handler (如 uvicorn --reload)
    if not any(isinstance(h.formatter, JsonFormatter) for h in root.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_event(logger: logging.Logger, event: str, **fields) -> None:
    """打一条结构化业务日志: {"event": ..., 业务字段...}"""
    logger.info(event, extra={"fields": fields})
