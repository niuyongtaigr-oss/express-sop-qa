"""长期记忆服务 — 抽取 / 冲突消解 / 召回 (P5 用户级记忆)

短期记忆（会话历史）是"把最近几轮原样带上"；长期记忆要难得多，因为**写比读难**：

  ① 抽取：从这轮对话里提炼值得记的事实（不是把原话塞进去）
  ② 合并：新事实与旧事实冲突时怎么办 —— ADD / UPDATE / DELETE / NOOP
  ③ 遗忘：重要性 + 上限淘汰（另见 store 侧的清理）

第 ② 步是分水岭：只会 ADD 的系统会同时留下"用户常驻上海"和"用户搬到杭州了"，
之后回答里就出现精神分裂。所以这里让 LLM 在**同一批候选**上输出操作序列，
而不是无脑追加。

两条不可动摇的约束：
  · **user_id 为空 → 全程不写不读**。把私人记忆记到租户维度上等于全租户可见,
    那是把隔离边界悄悄放宽, 比"没这个功能"更糟。
  · **记忆是尽力而为**: 抽取/消解失败只记日志并返回统计, 绝不把异常抛给问答链路。
    记忆写失败不该让用户拿不到回答。

🏭 Java 对标: 用户画像更新 Job（抽取 → 去重合并 → 落库）+ 读时召回
"""

import logging
import time
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.config import Settings
from app.infrastructure.llm import LLMClient
from app.infrastructure.memory_store import (
    KIND_FACT,
    KIND_PREFERENCE,
    MemoryItem,
    MemoryStore,
)

logger = logging.getLogger(__name__)

_EXTRACT_SYSTEM = (
    "你负责从客服对话里提炼**值得长期记住**的关于该用户的信息。\n"
    "分开记两类, 分别放进两个字段:\n"
    "- facts: 稳定的事实 (例如「用户负责华东区网点」「用户所在网点有 30 人」)\n"
    "- preferences: 用户对回答或服务提出的**要求** (例如「以后回答请带上条款号」\n"
    "  「不要用表格」「回答简短点」)。只要用户提了要求就必须抽出来, 再小也算 ——\n"
    "  这次不记, 下次就不会照做, 比不知道更让人恼火。\n"
    "另有 episode (值得记住的经历, 放进 facts 并把 kind 写成 episode)。\n"
    "**只记「用户」说的话**。「助手」的回答一律不是用户信息 —— 助手说「我会尽量简洁地\n"
    "回答您」是助手的承诺, 不是用户的要求; 把它记成用户偏好等于凭空给用户安要求,\n"
    "而且会覆盖掉用户真正提过的要求。\n"
    "**不要记**: 寒暄、一次性查询、知识库本身的内容(法规条文)、助手说的话、你不确定的信息。\n"
    "importance 取 0~1: 越稳定、越会影响后续服务的越高; 拿不准就给 0.3 以下。\n"
    "每条都必须给出 key(槽位名): 用简短名词短语概括「这条说的是哪件事」,\n"
    "例如「居住地」「负责区域」「回答格式要求」「所在网点」。**同一个槽位的不同版本\n"
    "必须用同一个 key** —— 系统靠它发现「新旧版本冲突」, 留空就等于放弃了这件事。\n"
    "\n"
    "示例 1 —— 用户说「我上个月调到杭州了，现在负责浙江这边」:\n"
    '  {"facts": [{"text": "用户现居杭州", "kind": "fact", "key": "居住地", "importance": 0.8},\n'
    '             {"text": "用户负责浙江区域", "kind": "fact", "key": "负责区域", "importance": 0.7}],\n'
    '   "preferences": []}\n'
    "  (这条没说要求, 所以 preferences 是空列表 —— 空是正常结果, 不要硬凑)\n"
    "示例 2 —— 用户说「我常驻上海，负责华东区的网点。以后回答请带上条款号」:\n"
    '  {"facts": [{"text": "用户常驻上海", "kind": "fact", "key": "居住地", "importance": 0.8},\n'
    '             {"text": "用户负责华东区网点", "kind": "fact", "key": "负责区域", "importance": 0.7}],\n'
    '   "preferences": [{"text": "用户要求回答必须带上条款号", "kind": "preference",\n'
    '                    "key": "回答格式要求", "importance": 0.9}]}\n'
    "  **preferences 不是空的** —— 用户提了要求。最容易被漏掉的就是这种「以后…请…」。\n"
    "\n"
    "没有值得记的就两个字段都返回空列表 —— 宁可不记, 也不要记噪声。\n"
    "text 必须是**一句完整的话**(主语+内容), key 只是**两三个字的槽位名**,\n"
    "两者绝不能相同, 也绝不能把 key 写进 text。"
)

_CONSOLIDATE_SYSTEM = (
    "你在维护一份用户长期记忆。给定「新提取的事实」和「已有的相关记忆」, 判断该怎么落库。\n"
    "对每条新事实输出一个操作:\n"
    "- ADD: 全新信息, 已有记忆里没有\n"
    "- UPDATE: 与某条已有记忆是同一件事但更准确/更新了 —— 必须给出 target_id 和**合并后的\n"
    "  完整句子**(例如已有「用户常驻上海」, 新信息是「调到杭州」, 新文本应是「用户现居杭州」;\n"
    "  不许写成「居住地: 杭州」这种 字段名:值 的形式, 也不要丢主语)\n"
    "- DELETE: 某条已有记忆被新信息证伪/取代, 且没有替代内容 —— 必须给出 target_id\n"
    "- NOOP: 已有记忆已经包含了这条信息, 无需改动\n"
    "每种操作都要给出 importance(0~1): ADD 与 UPDATE 直接沿用新事实的分值即可 ——\n"
    "这一项会决定记忆被召回的优先级和淘汰顺序, 别随手填 0.3。\n"
    "关键: **同一件事的新旧版本不要并存**(例如「常驻上海」与「已搬到杭州」应 UPDATE),\n"
    "但不同的事必须分开保留。判断是否同一件事, 首先看 key 是否相同 —— 同 key\n"
    "就是同一个槽位, 应当 UPDATE 而不是 ADD。拿不准时选 ADD。"
)


# ── 结构化输出 schema ────────────────────────────────────
class MemoryFact(BaseModel):
    """抽取出的一条候选事实"""

    text: str = Field(description="一句话, 自包含(不要用「他/这个」这类指代)")
    kind: Literal["fact", "preference", "episode"] = KIND_FACT
    key: str = Field(
        description="这条事实的槽位名(必填), 用于识别新旧版本。用简短名词短语, "
                    "例如: 居住地 / 负责区域 / 回答格式偏好 / 所在网点 / 常用查询类型。"
                    "**同一件事的不同版本必须用同一个 key** —— 它是发现冲突的唯一线索。",
    )
    importance: float = Field(default=0.3, ge=0.0, le=1.0)


class MemoryExtraction(BaseModel):
    """抽取结果 —— **事实与偏好是分开的两个字段, 不靠 kind 区分**

    实测(qwen2.5:7b): 两类放在同一个列表里、用 kind 区分时, 偏好会被事实整个吞掉 ——
    提示词里加了偏好示例、列了偏好信号词, 依然一条偏好都抽不出来。改成两个独立字段
    后模型才会分别考虑。类别由**字段**决定(代码强制 kind), 不给模型选错的机会。
    """

    facts: list[MemoryFact] = Field(
        default=[], description="用户陈述的稳定事实(居住地/负责区域/所在网点/规模等)"
    )
    preferences: list[MemoryFact] = Field(
        default=[],
        description="用户对回答或服务提出的要求, 例如「以后回答请带上条款号」「不要用表格」。"
                    "**只要提了要求就必须抽出来**, 再小也算。",
    )


class MemoryOp(BaseModel):
    """一条落库操作"""

    op: Literal["ADD", "UPDATE", "DELETE", "NOOP"]
    target_id: str | None = None
    text: str | None = None
    kind: Literal["fact", "preference", "episode"] | None = None
    key: str | None = None
    # 可空是刻意的: 有默认值时模型经常不填, 而"没填"会被当成 0.3 覆盖掉原值 ——
    # 实测两条 importance=0.8/0.7 的记忆被压平成了 0.3, 重要性就废了。
    # None = "我没意见", 由代码决定沿用旧值还是采用新事实的分值。
    importance: float | None = Field(default=None, ge=0.0, le=1.0)
    reason: str = ""


class ConsolidationPlan(BaseModel):
    ops: list[MemoryOp] = []


def _fact_is_sane(fact: MemoryFact) -> bool:
    """抽取结果的最低质量门

    实测: 7B 模型会把槽位名直接填进 text (得到 text="居住地"), 整句信息全丢。
    这种"看起来有值、实际没信息"的结果必须挡在落库之前 —— 提示词已经给了样例,
    但模型行为不稳定, 所以代码侧再兜一层。
    """
    text = (fact.text or "").strip()
    if len(text) < 5:                     # "居住地" 这种只有槽位名
        return False
    if fact.key and text == fact.key.strip():
        return False
    return True


def _update_text_is_sane(text: str, key: str) -> bool:
    """UPDATE 的新文本必须是一句完整的话, 不能是「字段名: 值」的退化形式"""
    t = (text or "").strip()
    if len(t) < 5:
        return False
    if key and t == key.strip():
        return False
    if key and t.startswith(f"{key.strip()}:") and len(t) < len(key) + 6:
        return False
    return True


class MemoryService:
    """长期记忆的读写编排 — 所有方法都要求 (tenant_id, user_id)"""

    def __init__(self, store: MemoryStore, llm: LLMClient, settings: Settings):
        self._store = store
        self._llm = llm
        self._settings = settings
        # 本轮新事实的最高分值: 模型没给 importance 时的兜底
        self._fallback_importance = 0.3

    # ── 读: 召回 ─────────────────────────────────────────
    def recall(self, query: str, tenant_id: str, user_id: str) -> list[MemoryItem]:
        """召回该用户的长期记忆 (未启用 / 无用户维度 → 空)"""
        if not self.enabled or not user_id:
            return []
        try:
            pairs = self._store.search(
                query, tenant_id, user_id, top_k=self._settings.memory_top_k
            )
        except Exception:
            logger.exception("记忆召回失败, 按无记忆继续")
            return []
        floor = self._settings.memory_recall_min_similarity
        return [item for item, sim in pairs if sim >= floor]

    def format_memories(self, items: list[MemoryItem]) -> str:
        """记忆 → 提示词段落。

        **必须标注"截至日期"**: 不标时间, 模型会把三个月前的偏好当成当前事实。
        与「参考文档」分开成两段也是刻意的 —— 一个是用户说过的话, 一个是权威语料,
        混在一起模型就分不清该以谁为准。
        """
        if not items:
            return ""
        today = datetime.now().strftime("%Y-%m-%d")
        lines = [f"关于该用户的已知信息 (截至 {today}, 可能已过时, 以用户当前说法为准):"]
        for m in items:
            lines.append(f"- [{m.kind}] {m.text}")
        return "\n".join(lines)

    # ── 写: 抽取 + 冲突消解 ───────────────────────────────
    def remember(
        self,
        question: str,
        answer: str,
        tenant_id: str,
        user_id: str,
        session_id: str = "",
    ) -> dict:
        """从一轮问答里沉淀长期记忆。尽力而为: 任何失败都只记日志。

        调用方必须保证: 只在**真实回答**后调用(降级/断连/半截答案不要调) ——
        否则错误内容会被固化成"关于该用户的记忆", 而且会被反复召回。
        """
        stats = {"extracted": 0, "added": 0, "updated": 0, "deleted": 0, "skipped": 0}
        if not self.enabled or not user_id:
            return stats

        facts = self._extract(question, answer)
        facts = [f for f in facts if f.importance >= self._settings.memory_min_importance]
        stats["extracted"] = len(facts)
        if not facts:
            return stats

        self._fallback_importance = max((f.importance for f in facts), default=0.3)
        candidates = self._candidates(facts, tenant_id, user_id)
        ops = self._consolidate(facts, candidates)
        if ops is None:      # 消解失败 → 退化成逐条 ADD? 不: 宁可这轮不记
            stats["skipped"] = len(facts)
            return stats

        by_id = {m.memory_id: m for m in candidates}
        for op in ops:
            try:
                self._apply(op, by_id, tenant_id, user_id, session_id, stats)
            except Exception:
                logger.exception("记忆落库失败 op=%s", op.op)
        self._enforce_limit(tenant_id, user_id)
        logger.info("长期记忆写入 tenant=%s user=%s %s", tenant_id, user_id, stats)
        return stats

    def _extract(self, question: str, answer: str) -> list[MemoryFact]:
        from langchain_core.messages import HumanMessage, SystemMessage

        try:
            out = self._llm.structured_invoke(
                MemoryExtraction,
                [SystemMessage(content=_EXTRACT_SYSTEM),
                 HumanMessage(content=f"【用户】{question}\n【助手】{answer}")],
            )
        except Exception:
            logger.warning("记忆抽取失败, 本轮不记", exc_info=True)
            return []
        raw = list(out.facts) + list(out.preferences)
        facts = [f for f in raw if _fact_is_sane(f)]
        if len(facts) != len(raw):
            logger.warning("丢弃 %d 条退化抽取结果 (text 过短或等于 key)",
                           len(raw) - len(facts))
        # kind 由"来自哪个字段"决定 —— 模型把偏好写进 facts 也纠正得回来
        for f in out.preferences:
            if _fact_is_sane(f):
                f.kind = KIND_PREFERENCE
        return facts

    def _candidates(
        self, facts: list[MemoryFact], tenant_id: str, user_id: str
    ) -> list[MemoryItem]:
        """为每条新事实找语义相近的已有记忆, 去重后作为消解上下文"""
        seen: dict[str, MemoryItem] = {}
        for f in facts:
            # 两条路都要走: 向量相似找"说得像的", key 匹配找"同一个槽位的"。
            # 只靠前者会漏掉「常驻上海 → 已搬到杭州」这类**同槽位反义**的冲突
            # (它们在向量空间里几乎不相似), 于是新旧并存、回答自相矛盾。
            if f.key:
                try:
                    for item in self._store.find_by_key(f.key, tenant_id, user_id):
                        seen[item.memory_id] = item
                except Exception:
                    logger.warning("记忆槽位检索失败 key=%s", f.key, exc_info=True)
            try:
                for item, _sim in self._store.search(f.text, tenant_id, user_id, top_k=3):
                    seen[item.memory_id] = item
            except Exception:
                logger.warning("记忆候选检索失败", exc_info=True)
        return list(seen.values())

    def _consolidate(
        self, facts: list[MemoryFact], candidates: list[MemoryItem]
    ) -> list[MemoryOp] | None:
        from langchain_core.messages import HumanMessage, SystemMessage

        if not candidates:      # 没有任何旧记忆 → 直接全部 ADD, 省一次 LLM 调用
            return [
                MemoryOp(op="ADD", text=f.text, kind=f.kind, key=f.key,
                         importance=f.importance)
                for f in facts
            ]
        new_block = "\n".join(
            f"- ({f.kind}, key={f.key or '-'}, importance={f.importance}) {f.text}"
            for f in facts
        )
        old_block = "\n".join(
            f"- id={m.memory_id} ({m.kind}, key={m.key or '-'}) {m.text}"
            for m in candidates
        )
        try:
            out = self._llm.structured_invoke(
                ConsolidationPlan,
                [SystemMessage(content=_CONSOLIDATE_SYSTEM),
                 HumanMessage(content=f"新提取的事实:\n{new_block}\n\n已有的相关记忆:\n{old_block}")],
            )
            return list(out.ops)
        except Exception:
            logger.warning("记忆冲突消解失败, 本轮不记", exc_info=True)
            return None

    def _apply(
        self,
        op: MemoryOp,
        by_id: dict[str, MemoryItem],
        tenant_id: str,
        user_id: str,
        session_id: str,
        stats: dict,
    ) -> None:
        if op.op == "NOOP":
            stats["skipped"] += 1
            return
        if op.op == "ADD":
            if not op.text:
                stats["skipped"] += 1
                return
            self._store.add(MemoryItem(
                text=op.text, tenant_id=tenant_id, user_id=user_id,
                kind=op.kind or KIND_FACT, key=op.key or "",
                # 模型没给分值 → 退回"本轮新事实里最高的那个", 而不是拍一个默认值
                importance=(op.importance if op.importance is not None
                            else self._fallback_importance),
                source_session=session_id,
            ))
            stats["added"] += 1
            return

        target = by_id.get(op.target_id or "")
        # 目标必须来自**本次召回的候选**且属于当前用户 —— 防模型编造 id 或指向别人的记忆
        if target is None or target.tenant_id != tenant_id or target.user_id != user_id:
            logger.warning("记忆操作目标非法, 跳过 op=%s target=%s", op.op, op.target_id)
            stats["skipped"] += 1
            return
        if op.op == "UPDATE":
            if not op.text or not _update_text_is_sane(op.text, op.key or target.key):
                # 实测退化案例: 模型把新文本写成「居住地: 杭州」这种 字段名:值,
                # 甚至直接回写槽位名。宁可保留旧值 —— 覆盖成垃圾比不更新更糟。
                logger.warning("拒绝退化的 UPDATE 文本: %r", op.text)
                stats["skipped"] += 1
                return
            target.text = op.text
            target.kind = op.kind or target.kind
            target.key = op.key or target.key
            if op.importance is not None:      # 没给就保留原分值, 不覆盖成默认值
                target.importance = op.importance
            self._store.update(target)
            stats["updated"] += 1
        elif op.op == "DELETE":
            if self._store.delete(target.memory_id, tenant_id, user_id):
                stats["deleted"] += 1

    def _enforce_limit(self, tenant_id: str, user_id: str) -> None:
        """每个用户的记忆条数上限: 超了按 (重要性, 新旧) 淘汰最差的

        没有上限的记忆库 = 越用越慢、越用越吵, 而且召回质量单调下降。
        """
        limit = self._settings.memory_max_items
        items = self._store.list(tenant_id, user_id, limit=limit + 50)
        if len(items) <= limit:
            return
        keep = sorted(items, key=lambda m: (m.importance, m.updated_at), reverse=True)[:limit]
        keep_ids = {m.memory_id for m in keep}
        for m in items:
            if m.memory_id not in keep_ids:
                self._store.delete(m.memory_id, tenant_id, user_id)

    # ── 维护 / 对外查询 ───────────────────────────────────
    @property
    def enabled(self) -> bool:
        return bool(self._settings.memory_enabled)

    def list_memories(self, tenant_id: str, user_id: str) -> list[MemoryItem]:
        if not user_id:
            return []
        return self._store.list(tenant_id, user_id)

    def forget(self, memory_id: str, tenant_id: str, user_id: str) -> bool:
        """删除一条记忆 (合规要求: 用户要求删就必须能删)"""
        if not user_id:
            return False
        return self._store.delete(memory_id, tenant_id, user_id)

    def forget_all(self, tenant_id: str, user_id: str) -> int:
        if not user_id:
            return 0
        return self._store.delete_all(tenant_id, user_id)

    def sweep(self) -> int:
        """清理陈旧的长期记忆 (按 updated_at 的 TTL, 兜底用)

        记忆和会话不同: 会话过期就是"这次对话结束了", 记忆过期是"这件事可能不再
        成立"。所以 TTL 给得长 (默认半年), 且**只按最后更新时间**判断 —— 一条被
        反复召回但从未更新的偏好, 说明它一直有效, 该留着。
        """
        if not self.enabled:
            return 0
        deadline = time.time() - self._settings.memory_ttl_days * 86400
        try:
            return self._store.purge_older_than(deadline)
        except Exception:
            logger.exception("记忆 TTL 清理失败, 本轮跳过")
            return 0
