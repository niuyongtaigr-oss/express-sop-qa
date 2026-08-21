"""轻量 BM25 索引 — 无第三方依赖, 适配中文 (字符 bigram 分词)

用于混合检索 (Hybrid): 与向量检索结果做 RRF (Reciprocal Rank Fusion) 融合,
弥补纯向量检索对精确术语/编号 (如「500元」「24小时」「3倍」) 召回不足的问题。

分词策略: 去掉空白后按相邻字符生成 bigram (CJK 无词边界时的常用做法),
过滤含标点的 bigram, 长度=1 的文本退化为单字 token。
语料规模小 (SOP chunk 数百级), 每次变更全量重建成本可忽略。
"""

import logging
import math
import re

logger = logging.getLogger(__name__)

# 会被过滤掉的字符: 中英文标点/空白/数字边界符
_PUNCT = set("。，、；：？！…—～《》「」『』（）()【】[]{}<>\"'`·,.!?;:/-_%")

_K1 = 1.5   # BM25 词频饱和参数
_B = 0.75   # 文档长度归一化参数


def tokenize(text: str) -> list[str]:
    """中文友好分词: 字符 bigram (单字符文本退化为单字)"""
    t = "".join(ch for ch in text if ch not in _PUNCT and not ch.isspace())
    if len(t) <= 1:
        return [t] if t else []
    return [t[i : i + 2] for i in range(len(t) - 1)]


class BM25Index:
    """静态 BM25 索引 — 建好后可多次查询 top_k"""

    def __init__(
        self,
        texts: list[str],
        metadatas: list[dict] | None = None,
        k1: float = _K1,
        b: float = _B,
    ):
        self.texts = texts
        # 与 texts 对齐的元数据 (供混合检索把 BM25 独有命中还原成完整 chunk)
        self.metadatas = metadatas or [{} for _ in texts]
        self._k1 = k1
        self._b = b
        self._avgdl = 0.0
        self._doc_freqs: list[dict[str, int]] = []  # 每篇的 term 频次
        self._df: dict[str, int] = {}               # term 的文档频率
        self._idf: dict[str, float] = {}
        self._build()

    def _build(self) -> None:
        n_docs = len(self.texts)
        if n_docs == 0:
            return
        total_len = 0
        for text in self.texts:
            tokens = tokenize(text)
            total_len += len(tokens)
            freq: dict[str, int] = {}
            for tok in tokens:
                freq[tok] = freq.get(tok, 0) + 1
            self._doc_freqs.append(freq)
            for tok in set(freq):
                self._df[tok] = self._df.get(tok, 0) + 1
        self._avgdl = total_len / n_docs
        for tok, df in self._df.items():
            # 经典 BM25 idf (平滑, 避免负值)
            self._idf[tok] = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))

    def _score(self, doc_idx: int, query_tokens: list[str]) -> float:
        freq = self._doc_freqs[doc_idx]
        dl = sum(freq.values())
        denom = self._k1 * (1 - self._b + self._b * dl / max(self._avgdl, 1e-9))
        score = 0.0
        for tok in query_tokens:
            f = freq.get(tok, 0)
            if f:
                score += self._idf.get(tok, 0.0) * (f * (self._k1 + 1)) / (f + denom)
        return score

    def top_k(self, query: str, k: int) -> list[tuple[int, float]]:
        """返回 [(doc_idx, bm25_score)] 按分数降序, 最多 k 条"""
        if not self.texts:
            return []
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        scored = [
            (i, self._score(i, q_tokens)) for i in range(len(self.texts))
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(i, s) for i, s in scored if s > 0][:k]
