"""问答编排服务 — /chat 链路入口

职责: 调用 LangGraph 编排图, 外面包一层生产化能力:
  - 限流: asyncio.Semaphore (并发数走配置, 超限排队)
  - 超时降级: asyncio.timeout + fallback 友好提示
  - 阻塞调用 (LLM/Chroma) 丢线程池, 不卡事件循环
  - 多轮记忆: SessionStore 注入历史 + 回答后回写 (P0.1)

真流式 (P0.2): /chat/stream 不再「先生成完再按标点切」, 而是直接消费
  graph.astream_events 的 on_chat_model_stream token 事件 (LLM token 级输出),
  SSE 事件带 type 字段:
    {"type":"intent", ...} → {"type":"answer_delta", "delta":...} → 
    {"type":"sources", ...} → {"type":"done", ...}  (出错: {"type":"error", ...})

🏭 Java 对标: Resilience4j Bulkhead + TimeLimiter + fallback
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from app.config import Settings
from app.core.logging import get_trace_id
from app.services.session_service import SessionStore

logger = logging.getLogger(__name__)

# 超时降级时的友好提示 (fallback)
_FALLBACK_ANSWER = "当前咨询人数较多或问题较复杂，处理超时。请稍后重试，或尝试简化您的问题。"

# 产生答案 token 的节点 (intent 节点的结构化输出 token 不推送给前端)
_ANSWER_NODES = {"rag_qa", "direct", "multi_hop"}


class ChatService:
    """智能问答编排服务 — 包装 LangGraph 图 + 限流/超时/降级 + 会话记忆"""

    def __init__(self, graph: Any, sessions: SessionStore, settings: Settings):
        self._graph = graph
        self._sessions = sessions
        self._settings = settings
        # 进程级信号量: 超过 max_concurrency 的请求排队,
        # 防止突发流量把 Ollama (本地单点) 打挂
        self._semaphore = asyncio.Semaphore(settings.max_concurrency)

    async def chat(self, question: str, session_id: str | None = None) -> dict:
        """非流式问答: 意图识别 → 路由 → 回答 (带限流 + 超时降级 + 记忆)"""
        start = time.perf_counter()
        result = await self._invoke_with_guard(question, session_id)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        intent = result.get("intent", "unknown")
        answer = result.get("answer", "")
        # 记忆回写: 只有真实回答才入库, 降级提示不污染历史
        if intent != "degraded":
            self._sessions.add_turn(session_id, question, answer)
        logger.info("chat_done intent=%s elapsed_ms=%.1f session=%s",
                    intent, elapsed_ms, session_id)
        return {
            "answer": answer,
            "intent": intent,
            "sources": result.get("sources", []),
            "trace_id": get_trace_id(),
            "elapsed_ms": elapsed_ms,
            "session_id": session_id,
        }

    async def stream(self, question: str, session_id: str | None = None) -> AsyncIterator[dict]:
        """流式问答 (SSE): 真 token 级流式, 事件见模块 docstring"""
        history = self._sessions.get_history(session_id)
        answer_parts: list[str] = []
        intent = "unknown"
        degraded = False
        failed = False
        try:
            async with self._semaphore:
                try:
                    async with asyncio.timeout(self._settings.chat_timeout_s):
                        async for ev in self._graph.astream_events(
                            {"question": question, "history": history},
                            version="v2",
                        ):
                            async for out in self._handle_event(ev, answer_parts):
                                if out.get("type") == "intent":
                                    intent = out["intent"]
                                yield out
                except asyncio.TimeoutError:
                    degraded = True
                    intent = "degraded"
                    logger.warning("stream 超时降级 (>%ss)", self._settings.chat_timeout_s)
                    yield {"type": "answer_delta", "delta": _FALLBACK_ANSWER}
        except Exception as e:  # SSE 通道内异常转成错误事件, 不断连
            failed = True
            logger.exception("stream 处理失败")
            yield {"type": "error", "error": f"{type(e).__name__}: {e}"}
        finally:
            # 只有真实完成才回写记忆; 超时降级/异常不污染历史
            if not degraded and not failed:
                self._sessions.add_turn(
                    session_id, question, "".join(answer_parts)
                )
            yield {
                "type": "done",
                "trace_id": get_trace_id(),
                "intent": intent,
            }

    # ── 内部: astream_events 事件解析 ────────────────────
    @staticmethod
    async def _handle_event(ev: dict, answer_parts: list[str]) -> AsyncIterator[dict]:
        """把 LangGraph astream_events 事件映射为对外 SSE 事件"""
        if ev["event"] == "on_chat_model_stream":
            node = ev.get("metadata", {}).get("langgraph_node")
            if node in _ANSWER_NODES:
                chunk = ev["data"].get("chunk")
                text = getattr(chunk, "content", "") or ""
                if text:
                    answer_parts.append(text)
                    yield {"type": "answer_delta", "delta": text}
            return
        if ev["event"] == "on_chain_end":
            node = ev.get("metadata", {}).get("langgraph_node")
            # 节点自身完成事件: name 与 langgraph_node 一致 (过滤内部子链)
            if node and ev.get("name") == node:
                output = ev["data"].get("output") or {}
                if node == "intent":
                    yield {
                        "type": "intent",
                        "intent": output.get("intent", "unknown"),
                        "reason": output.get("intent_reason", ""),
                    }
                elif node in _ANSWER_NODES:
                    event: dict = {
                        "type": "sources",
                        "sources": output.get("sources", []),
                    }
                    if output.get("rounds") is not None:
                        event["rounds"] = output["rounds"]
                    yield event

    # ── 内部: 限流 + 超时降级 ─────────────────────────────
    async def _invoke_with_guard(self, question: str, session_id: str | None) -> dict:
        async def _run() -> dict:
            async with self._semaphore:
                # 先取历史再进线程池 (读取很快, 不占信号量窗口太久)
                history = self._sessions.get_history(session_id)
                # graph.invoke 是同步阻塞调用, 丢线程池执行
                return await asyncio.to_thread(
                    self._graph.invoke,
                    {"question": question, "history": history},
                )

        try:
            return await asyncio.wait_for(
                _run(), timeout=self._settings.chat_timeout_s
            )
        except asyncio.TimeoutError:
            # 超时降级: 不抛错, 返回友好提示 (fallback)
            logger.warning("chat 超时降级 (>%ss)", self._settings.chat_timeout_s)
            return {
                "answer": _FALLBACK_ANSWER,
                "intent": "degraded",
                "sources": [],
            }
