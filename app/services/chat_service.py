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
from collections.abc import AsyncIterator, Callable
from typing import Any

from app.config import Settings
from app.core.logging import get_trace_id
from app.core.metrics import (
    CACHE_HITS,
    CACHE_MISSES,
    CHAT_DEGRADED,
    CHAT_DURATION,
    CHAT_REQUESTS,
)
from app.services.cache_service import CacheStats, ChatCache
from app.services.session_service import SessionStore

logger = logging.getLogger(__name__)

# 超时降级时的友好提示 (fallback)
_FALLBACK_ANSWER = "当前咨询人数较多或问题较复杂，处理超时。请稍后重试，或尝试简化您的问题。"

# 链路异常时的友好提示。与流式路径保持一致: 流式发 error 事件后仍给 done,
# 非流式原先直接 500 —— 同一产品两种体验, 这里补齐。
_ERROR_ANSWER = "服务暂时不可用，请稍后重试。如持续出现请联系管理员。"

# 产生答案 token 的节点 (intent 节点的结构化输出 token 不推送给前端)
_ANSWER_NODES = {"rag_qa", "direct", "multi_hop"}


class ChatService:
    """智能问答编排服务 — 包装 LangGraph 图 + 限流/超时/降级 + 会话记忆 + 答案缓存"""

    def __init__(
        self,
        graph: Any,
        sessions: SessionStore,
        settings: Settings,
        get_kb_version: Callable[[], int] | None = None,
    ):
        self._graph = graph
        self._sessions = sessions
        self._settings = settings
        self._get_kb_version = get_kb_version or (lambda: 0)
        # 进程级信号量: 超过 max_concurrency 的请求排队,
        # 防止突发流量把 Ollama (本地单点) 打挂
        self._semaphore = asyncio.Semaphore(settings.max_concurrency)
        # 答案缓存 (P3-A): 无会话相同问题短 TTL 缓存
        self._cache: ChatCache | None = None
        self._cache_stats = CacheStats()
        if settings.cache_enabled:
            self._cache = ChatCache(
                ttl_s=settings.cache_ttl_s,
                max_entries=settings.cache_max_entries,
            )
        # 超时链路自检: chat_timeout_s 走 asyncio.wait_for 取消的是**协程**, 而
        # asyncio.to_thread 里的**线程不可取消**。只有 LLM 自己先超时, 线程才会
        # 真正结束、信号量释放才与"实际占用"一致。否则超时后线程仍在跑, 实际
        # 并发会超过 max_concurrency —— 信号量形同虚设。
        if settings.llm_timeout_s >= settings.chat_timeout_s:
            logger.warning(
                "llm_timeout_s (%.0fs) >= chat_timeout_s (%.0fs): LLM 会晚于外层超时, "
                "取消协程后线程仍在跑 → 并发上限失效。建议 llm_timeout_s 明显小于 "
                "chat_timeout_s。",
                settings.llm_timeout_s, settings.chat_timeout_s,
            )

    @property
    def cache_stats(self) -> dict:
        """缓存命中统计 (供 /metrics 与排查)"""
        stats = self._cache_stats.snapshot()
        if self._cache is not None:
            stats.update(self._cache.stats())
        return stats

    def clear_cache(self) -> int:
        """清空答案缓存 (知识库变更时由上层调用)"""
        return self._cache.clear() if self._cache else 0

    async def chat(
        self,
        question: str,
        session_id: str | None = None,
        tenant_id: str = "default",
    ) -> dict:
        """非流式问答: 意图识别 → 路由 → 回答 (限流 + 超时降级 + 记忆 + 缓存)"""
        start = time.perf_counter()
        cached = False
        cache_key = None
        # 只缓存无会话的请求: 有会话时答案依赖历史, 不能复用
        if self._cache is not None and session_id is None:
            cache_key = ChatCache.key_for(
                question, self._get_kb_version(), tenant_id
            )
            hit = self._cache.get(cache_key)
            if hit is not None:
                self._cache_stats.hit()
                CACHE_HITS.inc()
                cached = True
                result = hit
            else:
                self._cache_stats.miss()
                CACHE_MISSES.inc()

        if not cached:
            result = await self._invoke_with_guard(question, session_id, tenant_id)
            # 真实回答且无会话才写缓存; 降级/异常不入缓存
            if (
                self._cache is not None
                and cache_key is not None
                and result.get("intent") != "degraded"
            ):
                self._cache.set(cache_key, result)

        elapsed_ms = round((time.perf_counter() - start) * 1000, 2) if not cached else 0.0
        intent = result.get("intent", "unknown")
        answer = result.get("answer", "")
        # 记忆回写: 只有真实回答才入库, 降级提示不污染历史
        if not cached and intent != "degraded":
            self._sessions.add_turn(tenant_id, session_id, question, answer)
        # Prometheus 指标
        CHAT_REQUESTS.labels(intent).inc()
        if not cached:
            CHAT_DURATION.observe(elapsed_ms / 1000)
        if intent == "degraded":
            CHAT_DEGRADED.inc()
        logger.info("chat_done intent=%s elapsed_ms=%.1f session=%s cached=%s",
                    intent, elapsed_ms, session_id, cached)
        return {
            "answer": answer,
            "intent": intent,
            "sources": result.get("sources", []),
            "trace_id": get_trace_id(),
            "elapsed_ms": elapsed_ms,
            "session_id": session_id,
            "cached": cached,
        }

    async def stream(
        self,
        question: str,
        session_id: str | None = None,
        tenant_id: str = "default",
    ) -> AsyncIterator[dict]:
        """流式问答 (SSE): 真 token 级流式, 事件见模块 docstring

        事件收尾放在 try/finally **之外**, 这是刻意的:
        若把 `done` 放在 finally 里 yield, 客户端中途断连时 aclose() 会在
        finally 处抛 GeneratorExit, 而 finally 又试图 yield —— 触发
        `RuntimeError: async generator ignored GeneratorExit`。
        放到 finally 之外后, GeneratorExit 直接向上传播, 生成器干净关闭。

        副作用 (也是想要的行为): 断连时既不发 `done`, 也不回写会话记忆 ——
        此时 answer_parts 是半截的, 写进历史只会污染下一轮上下文。
        """
        history = self._sessions.get_history(tenant_id, session_id)
        answer_parts: list[str] = []
        intent = "unknown"
        degraded = False
        failed = False
        try:
            async with self._semaphore:
                try:
                    async with asyncio.timeout(self._settings.chat_timeout_s):
                        async for ev in self._graph.astream_events(
                            {
                                "question": question,
                                "history": history,
                                "tenant_id": tenant_id,
                            },
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

        # 只有走完正常路径 (非降级、非异常、且客户端没断连) 才回写记忆
        if not degraded and not failed:
            self._sessions.add_turn(
                tenant_id, session_id, question, "".join(answer_parts)
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
    async def _invoke_with_guard(
        self, question: str, session_id: str | None, tenant_id: str = "default"
    ) -> dict:
        async def _run() -> dict:
            async with self._semaphore:
                # 先取历史再进线程池 (读取很快, 不占信号量窗口太久)
                history = self._sessions.get_history(tenant_id, session_id)
                # graph.invoke 是同步阻塞调用, 丢线程池执行
                return await asyncio.to_thread(
                    self._graph.invoke,
                    {
                        "question": question,
                        "history": history,
                        "tenant_id": tenant_id,
                    },
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
        except Exception:
            # 非超时异常原先直接冒到 API 层变 500, 而流式路径是转成 error 事件后
            # 正常收尾 —— 同一份产品两条链路容错不一致。这里补齐。
            # 用 logger.exception 保留完整堆栈: 兜底是为了用户体验, 不是为了藏 bug。
            logger.exception("chat 处理失败, 已降级返回")
            return {
                "answer": _ERROR_ANSWER,
                "intent": "degraded",
                "sources": [],
            }
