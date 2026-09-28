"""用户反馈存储 — FeedbackStore (P2-C)

把用户对回答的 👍/👎 + 纠错文本落盘到 JSONL (data/feedback.jsonl):
- 负面反馈 (rating ≤ 2) 打结构化告警日志, 便于人工跟进
- stats / negative_questions 供统计与评测集扩充素材 (低分问题 → 新评测用例)

**按租户隔离**: 每条记录带 tenant_id, stats / negative_questions 只统计本租户。
原先三者都是全局的 —— 多租户下 A 能看到 B 的低分问题, 而那些 question 是用户
原话。会话已经按租户隔离了, 反馈是同一类数据, 不能只在会话上做隔离。

线程安全, 保留最近 max_entries 条 (**全局**上限, 见 _load 的说明)。
🏭 Java 对标: 反馈表 (带 tenant_id) + 低分告警 Job
"""

import json
import logging
import threading
import time
from pathlib import Path

from app.services.tenant_service import DEFAULT_TENANT

logger = logging.getLogger(__name__)


class FeedbackStore:
    """反馈存储 — JSONL 持久化 + 统计"""

    def __init__(self, path: Path, max_entries: int = 10000):
        self._path = path
        self._max = max_entries
        self._lock = threading.Lock()
        self._entries: list[dict] = self._load()
        # 距上次压缩的写入数: 文件是 append-only, 不压缩会无限增长
        self._writes_since_compact = 0

    def _load(self) -> list[dict]:
        if not self._path.exists():
            return []
        entries: list[dict] = []
        try:
            lines = self._path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        for line in lines[-self._max * 2:]:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return entries[-self._max:]

    def append(self, entry: dict, tenant_id: str = DEFAULT_TENANT) -> int:
        """写入一条反馈 (含 ts 与 tenant_id), 返回当前总数; 负面反馈打告警日志"""
        entry["ts"] = time.time()
        entry["tenant_id"] = tenant_id or DEFAULT_TENANT
        with self._lock:
            self._entries.append(entry)
            self._entries = self._entries[-self._max:]
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            total = len(self._entries)
            # 每累计 max_entries 次写入压缩一次: 把文件重写为当前保留的条目。
            # 否则 JSONL 只追加不回收 —— 内存有上限, 磁盘没有。
            self._writes_since_compact += 1
            if self._writes_since_compact >= self._max:
                self._compact_locked()
                self._writes_since_compact = 0
        rating = entry.get("rating", 0)
        if rating <= 2:
            logger.warning(
                "feedback_negative tenant=%s rating=%d question=%s comment=%s",
                entry["tenant_id"], rating,
                entry.get("question", ""), entry.get("comment", ""),
            )
        return total

    def _compact_locked(self) -> None:
        """把 JSONL 重写为当前保留的条目 (调用方需已持锁)"""
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        try:
            with tmp.open("w", encoding="utf-8") as f:
                for e in self._entries:
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")
            tmp.replace(self._path)  # 原子替换, 中断不会留下半截文件
            logger.info("feedback 已压缩至 %d 条", len(self._entries))
        except OSError as e:
            logger.warning("feedback 压缩失败: %s", e)

    def _of_tenant(self, tenant_id: str) -> list[dict]:
        """本租户的记录。

        老数据 (本次改动之前写入) 没有 tenant_id 字段 —— 那时只有单租户模式,
        因此归到 DEFAULT_TENANT。多租户下真实租户 id 来自租户清单, 不会借此
        看到别的租户。
        """
        want = tenant_id or DEFAULT_TENANT
        return [e for e in self._entries
                if (e.get("tenant_id") or DEFAULT_TENANT) == want]

    def stats(self, tenant_id: str = DEFAULT_TENANT) -> dict:
        with self._lock:
            entries = self._of_tenant(tenant_id)
            total = len(entries)
            if total == 0:
                return {"total": 0, "avg_rating": None,
                        "negative_count": 0, "negative_rate": 0.0}
            ratings = [e.get("rating", 0) for e in entries]
            negative = sum(1 for r in ratings if r <= 2)
            return {
                "total": total,
                "avg_rating": round(sum(ratings) / total, 2),
                "negative_count": negative,
                "negative_rate": round(negative / total, 4),
            }

    def negative_questions(self, limit: int = 10,
                           tenant_id: str = DEFAULT_TENANT) -> list[dict]:
        """本租户最近的低分问题 (评测集扩充素材)"""
        with self._lock:
            neg = [e for e in self._of_tenant(tenant_id) if e.get("rating", 0) <= 2]
        neg = neg[-limit:]
        return [
            {
                "question": e.get("question", ""),
                "rating": e.get("rating", 0),
                "comment": e.get("comment", ""),
                "ts": e.get("ts"),
            }
            for e in reversed(neg)
        ]
