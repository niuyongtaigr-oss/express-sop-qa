"""LLM 客户端抽象 — Protocol + ChatOllama 实现 + 懒加载工厂

业务层只依赖 LLMClient 协议, 换实现 (OpenAI/通义/...) 只需新增一个
实现类和工厂分支, 不改任何业务代码。

🏭 Java 对标: 接口 + @Bean 工厂 (面向接口编程, DIP)
"""

import logging
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


class OllamaLLMClient:
    """ChatOllama 实现 — 懒加载底层客户端 (首次调用才初始化)"""

    def __init__(self, model: str, base_url: str, temperature: float = 0):
        self._model = model
        self._base_url = base_url
        self._temperature = temperature
        self._chat = None  # ChatOllama 懒加载

    def _get_chat(self):
        if self._chat is None:
            from langchain_ollama import ChatOllama

            self._chat = ChatOllama(
                model=self._model,
                base_url=self._base_url,
                temperature=self._temperature,
            )
        return self._chat

    def invoke(self, messages: Sequence[BaseMessage]) -> str:
        resp = self._get_chat().invoke(list(messages))
        return str(resp.content)

    def structured_invoke(
        self, schema: type[T], messages: Sequence[BaseMessage]
    ) -> T:
        router = self._get_chat().with_structured_output(schema)
        return router.invoke(list(messages))


def create_llm_client(settings: Settings) -> LLMClient:
    """LLM 工厂 — 按配置创建实现实例 (当前仅 Ollama)"""
    logger.info("创建 LLM 客户端: model=%s base_url=%s",
                settings.llm_model, settings.ollama_base_url)
    return OllamaLLMClient(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
    )
