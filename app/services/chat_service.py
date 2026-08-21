"""问答编排服务 — /chat 链路入口

职责: 调用 LangGraph 编排图, 外面包一层生产化能力:
  - 限流: asyncio.Semaphore (并发数走配置, 超限排队)
  - 超时降级: asyncio.wait_for + fallback 友好提示
  - 阻塞调用 (LLM/Chroma) 丢线程池, 不卡事件循环

🏭 Java 对标: Resilience4j Bulkhead + TimeLimiter + fallback
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from app.config import Settings
from app.core.logging import get_trace_id

logger = logging.getLogger(__name__)

# 超时降级时的友好提示 (fallback)
_FALLBACK_ANSWER = "当前咨询人数较多或问题较复杂，处理超时。请稍后重试，或尝试简化您的问题。"

# SSE 分段推送的分段标点
_SSE_SEG_CHARS = "。!?；\n"


class ChatService:
    """智能问答编排服务 — 包装 LangGraph 图 + 限流/超时/降级"""

    def __init__(self, graph: Any, settings: Settings):
        self._graph = graph
        self._settings = settings
        # 进程级信号量: 超过 max_concurrency 的请求排队,
        # 防止突发流量把 Ollama (本地单点) 打挂
        self._semaphore = asyncio.Semaphore(settings.max_concurrency)

    async def chat(self, question: str) -> dict:
        """非流式问答: 意图识别 → 路由 → 回答 (带限流 + 超时降级)"""
        start = time.perf_counter()
        result = await self._invoke_with_guard(question)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info("chat_done intent=%s elapsed_ms=%.1f",
                    result.get("intent", "unknown"), elapsed_ms)
        return {
            "answer": result.get("answer", ""),
            "intent": result.get("intent", "unknown"),
            "sources": result.get("sources", []),
            "trace_id": get_trace_id(),
            "elapsed_ms": elapsed_ms,
        }

    async def stream(self, question: str) -> AsyncIterator[dict]:
        """流式问答: 同 /chat 链路, 整体生成后按标点分段推送 (SSE)

        TODO(优化): 换 graph.astream_events 真流式 (LLM token 级输出)
        """
        try:
            result = await self._invoke_with_guard(question)
            answer = result.get("answer", "")
            trace_id = get_trace_id()
            seg = ""
            for ch in answer:
                seg += ch
                if ch in _SSE_SEG_CHARS or len(seg) >= 30:
                    yield {"delta": seg, "trace_id": trace_id}
                    seg = ""
                    await asyncio.sleep(0.05)
            if seg:
                yield {"delta": seg, "trace_id": trace_id}
            yield {"intent": result.get("intent", "unknown")}
        except Exception as e:  # SSE 通道内异常转成错误事件, 不断连
            logger.exception("stream 处理失败")
            yield {"error": f"{type(e).__name__}: {e}"}

    # ── 内部: 限流 + 超时降级 ─────────────────────────────
    async def _invoke_with_guard(self, question: str) -> dict:
        async def _run() -> dict:
            async with self._semaphore:
                # graph.invoke 是同步阻塞调用, 丢线程池执行
                return await asyncio.to_thread(
                    self._graph.invoke, {"question": question}
                )

        try:
            return await asyncio.wait_for(
                _run(), timeout=self._settings.chat_timeout_s
            )
        except asyncio.TimeoutError:
            # 超时降级: 不抛错, 返回友好提示 (fallback)
            logger.warning("chat 超时降级 (>%ss)",
                           self._settings.chat_timeout_s)
            return {
                "answer": _FALLBACK_ANSWER,
                "intent": "degraded",
                "sources": [],
            }
