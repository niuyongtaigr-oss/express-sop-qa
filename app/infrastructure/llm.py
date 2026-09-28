"""LLM 客户端抽象 — Protocol + Ollama/OpenAI 兼容实现 + 懒加载工厂 (P3-C)

业务层只依赖 LLMClient 协议, 换实现 (OpenAI/通义/DeepSeek...) 只需新增一个
实现类和工厂分支, 不改任何业务代码。
P3-C 多模型路由: 意图识别用小模型省成本, 回答用大模型 — 工厂按 model 参数
创建任意模型的客户端, 编排层分开注入 intent_llm / answer_llm。

🏭 Java 对标: 接口 + @Bean 工厂 (面向接口编程, DIP)
"""

import logging
from collections.abc import AsyncIterator
from typing import Protocol, Sequence, TypeVar

from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from app.config import Settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMClient(Protocol):
    """LLM 客户端协议 — 业务层依赖的唯一抽象"""

    def invoke(self, messages: Sequence[BaseMessage]) -> str:
        """普通对话: 消息列表 → 文本回答"""
        ...

    def structured_invoke(
        self, schema: type[T], messages: Sequence[BaseMessage]
    ) -> T:
        """结构化输出: 按 Pydantic schema 约束模型输出 (意图识别等场景)"""
        ...

    def astream(self, messages: Sequence[BaseMessage]) -> AsyncIterator[str]:
        """流式对话: 消息列表 → token 级文本增量 (SSE 真流式用)"""
        ...
        yield  # pragma: no cover - Protocol 仅声明


class OllamaLLMClient:
    """ChatOllama 实现 — 懒加载底层客户端 (首次调用才初始化)

    ⚠️ 超时必须走 `client_kwargs={"timeout": ...}`。
    直接传 `ChatOllama(timeout=...)` 会被**静默忽略** (该参数不在 ChatOllama
    的字段里, 也不会透传到底层 ollama.Client) —— 那样超时形同没设。
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        temperature: float = 0,
        timeout_s: float | None = None,
    ):
        self._model = model
        self._base_url = base_url
        self._temperature = temperature
        self._timeout_s = timeout_s
        self._chat = None  # ChatOllama 懒加载

    def _get_chat(self):
        if self._chat is None:
            from langchain_ollama import ChatOllama

            kwargs = {}
            if self._timeout_s:
                kwargs["client_kwargs"] = {"timeout": self._timeout_s}
            self._chat = ChatOllama(
                model=self._model,
                base_url=self._base_url,
                temperature=self._temperature,
                **kwargs,
            )
        return self._chat

    def invoke(self, messages: Sequence[BaseMessage]) -> str:
        resp = self._get_chat().invoke(list(messages))
        return str(resp.content)

    async def astream(self, messages: Sequence[BaseMessage]) -> AsyncIterator[str]:
        """token 级流式输出 (ChatOllama.astream → AIMessageChunk)"""
        async for chunk in self._get_chat().astream(list(messages)):
            text = getattr(chunk, "content", "")
            if text:
                yield str(text)

    def structured_invoke(
        self, schema: type[T], messages: Sequence[BaseMessage]
    ) -> T:
        router = self._get_chat().with_structured_output(schema)
        return router.invoke(list(messages))


class OpenAILLMClient:
    """OpenAI 兼容实现 (ChatOpenAI) — 通义/DeepSeek/本地 vLLM 均走 OpenAI 协议

    懒加载 langchain_openai (仅 openai provider 且首次调用时才 import,
    纯 Ollama 部署无需安装该依赖)。
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str | None = None,
        temperature: float = 0,
        timeout_s: float | None = None,
    ):
        self._model = model
        self._base_url = base_url
        self._api_key = api_key or "sk-no-key"  # 本地 vLLM 等可能不需要 key
        self._temperature = temperature
        self._timeout_s = timeout_s
        self._chat = None

    def _get_chat(self):
        if self._chat is None:
            from langchain_openai import ChatOpenAI

            kwargs = {}
            if self._timeout_s:
                kwargs["timeout"] = self._timeout_s
            self._chat = ChatOpenAI(
                model=self._model,
                base_url=self._base_url,
                api_key=self._api_key,
                temperature=self._temperature,
                **kwargs,
            )
        return self._chat

    def invoke(self, messages: Sequence[BaseMessage]) -> str:
        resp = self._get_chat().invoke(list(messages))
        return str(resp.content)

    async def astream(self, messages: Sequence[BaseMessage]) -> AsyncIterator[str]:
        async for chunk in self._get_chat().astream(list(messages)):
            text = getattr(chunk, "content", "")
            if text:
                yield str(text)

    def structured_invoke(
        self, schema: type[T], messages: Sequence[BaseMessage]
    ) -> T:
        router = self._get_chat().with_structured_output(schema)
        return router.invoke(list(messages))


def create_llm_client(
    settings: Settings,
    model: str | None = None,
    provider: str | None = None,
) -> LLMClient:
    """LLM 工厂 — 按配置创建指定 provider/model 的客户端

    model/provider 缺省走配置; provider=ollama|openai (OpenAI 兼容协议)。
    """
    provider = provider or settings.llm_provider
    if model is None:
        # openai provider 时回答模型优先 openai_model, 否则回退 llm_model
        model = settings.openai_model or settings.llm_model if provider == "openai" else settings.llm_model
    if provider == "openai":
        logger.info("创建 OpenAI 兼容 LLM: model=%s base_url=%s timeout=%ss",
                    model, settings.openai_base_url, settings.llm_timeout_s)
        return OpenAILLMClient(
            model=model,
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
            timeout_s=settings.llm_timeout_s,
        )
    logger.info("创建 Ollama LLM: model=%s base_url=%s timeout=%ss",
                model, settings.ollama_base_url, settings.llm_timeout_s)
    return OllamaLLMClient(
        model=model,
        base_url=settings.ollama_base_url,
        timeout_s=settings.llm_timeout_s,
    )


def create_intent_llm(settings: Settings) -> LLMClient | None:
    """分意图路由: 配置了 intent_model 才创建独立的意图识别小模型客户端

    未配置返回 None → 编排层回退用 answer_llm (保持原行为)。
    """
    if not settings.intent_model:
        return None
    return create_llm_client(settings, model=settings.intent_model)
