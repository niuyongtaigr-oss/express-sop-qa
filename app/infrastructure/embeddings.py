"""Embedding 客户端抽象 — Protocol + Ollama 实现 + 工厂

🏭 Java 对标: 接口 + @Bean 工厂 (与 llm.py 同一模式)
"""

import logging
from typing import Protocol

from app.config import Settings

logger = logging.getLogger(__name__)


class EmbeddingClient(Protocol):
    """Embedding 客户端协议 — 文本批量向量化"""

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class OllamaEmbeddingClient:
    """Ollama Embedding 实现 (懒加载, 默认 bge-m3)"""

    def __init__(self, model: str, base_url: str):
        self._model = model
        self._base_url = base_url
        self._embedder = None

    def _get_embedder(self):
        if self._embedder is None:
            from langchain_ollama import OllamaEmbeddings

            self._embedder = OllamaEmbeddings(
                model=self._model, base_url=self._base_url
            )
        return self._embedder

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._get_embedder().embed_documents(texts)


def create_embedding_client(settings: Settings) -> EmbeddingClient:
    """Embedding 工厂 — 按配置创建实现实例 (当前仅 Ollama)"""
    logger.info("创建 Embedding 客户端: model=%s", settings.embed_model)
    return OllamaEmbeddingClient(
        model=settings.embed_model,
        base_url=settings.ollama_base_url,
    )
