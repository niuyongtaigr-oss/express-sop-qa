"""用户级长期记忆 — 存储隔离 + 抽取/消解/遗忘

这个模块和知识库最大的不同是**它会写入个人信息**, 所以测试的重点不是"能不能
召回", 而是**边界**: 跨用户不可见、删别人的删不掉、没用户维度时全程不写不读。
"""

import time

import pytest

from app.config import Settings
from app.infrastructure.memory_store import ChromaMemoryStore, MemoryItem
from app.services.memory_service import (
    ConsolidationPlan,
    MemoryExtraction,
    MemoryFact,
    MemoryOp,
    MemoryService,
)


class KeywordEmbedding:
    """按关键词维度的向量 —— 让"相似"这件事可控, 便于断言排序与召回"""

    _DIMS = {"上海": 0, "杭州": 1, "理赔": 2, "条款": 3, "破损": 4}

    def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * 6
            v[5] = 0.1
            for kw, i in self._DIMS.items():
                if kw in t:
                    v[i] = 1.0
            out.append(v)
        return out


def _store(tmp_path) -> ChromaMemoryStore:
    return ChromaMemoryStore(
        persist_dir=str(tmp_path),
        collection_name="test_memory",
        embeddings=KeywordEmbedding(),
    )


def _settings(**kw) -> Settings:
    base = dict(memory_enabled=True, memory_top_k=5, memory_min_importance=0.0)
    base.update(kw)
    return Settings(_env_file=None, **base)


def _item(text, tenant="t1", user="u1", **kw) -> MemoryItem:
    return MemoryItem(text=text, tenant_id=tenant, user_id=user, **kw)


class StubLLM:
    """按 schema 返回预设结构化输出; 可指定抛错以验证失败隔离"""

    def __init__(self, extraction=None, plan=None, error=None):
        self._extraction = extraction
        self._plan = plan
        self._error = error
        self.schemas: list = []

    def structured_invoke(self, schema, messages):
        self.schemas.append(schema)
        if self._error is not None:
            raise self._error
        if schema is MemoryExtraction:
            return self._extraction or MemoryExtraction(facts=[])
        if schema is ConsolidationPlan:
            return self._plan or ConsolidationPlan(ops=[])
        raise AssertionError(f"未知 schema: {schema}")


# ── 存储层: 隔离是硬要求 ─────────────────────────────────

def test_store_isolates_between_users(tmp_path):
    """同一租户、不同用户 → 互相看不到 (这是引入 user_id 的全部意义)"""
    s = _store(tmp_path)
    s.add(_item("用户常驻上海", user="u1"))
    s.add(_item("用户常驻杭州", user="u2"))

    a = s.list("t1", "u1")
    b = s.list("t1", "u2")
    assert [m.text for m in a] == ["用户常驻上海"]
    assert [m.text for m in b] == ["用户常驻杭州"]
    assert s.count("t1", "u1") == 1 and s.count("t1", "u2") == 1
    assert s.count("t1", "u3") == 0

    hits = s.search("上海 理赔", "t1", "u1", top_k=5)
    assert all(m.user_id == "u1" for m, _ in hits)


def test_store_isolates_between_tenants(tmp_path):
    s = _store(tmp_path)
    s.add(_item("甲的偏好", tenant="t1", user="u1"))
    s.add(_item("乙的偏好", tenant="t2", user="u1"))
    assert [m.text for m in s.list("t1", "u1")] == ["甲的偏好"]
    assert [m.text for m in s.list("t2", "u1")] == ["乙的偏好"]


def test_delete_and_update_require_matching_owner(tmp_path):
    """id 猜对了也不能动别人的记忆 —— 越权删比越权读更严重"""
    s = _store(tmp_path)
    item = _item("用户的机密偏好", tenant="t1", user="u1")
    s.add(item)

    assert s.delete(item.memory_id, "t1", "u2") is False       # 同租户换个用户
    assert s.delete(item.memory_id, "t2", "u1") is False       # 换个租户
    assert s.count("t1", "u1") == 1                            # 还在

    item.text = "被改过"
    assert s.update(item) is True
    assert s.list("t1", "u1")[0].text == "被改过"

    assert s.delete(item.memory_id, "t1", "u1") is True
    assert s.count("t1", "u1") == 0


def test_find_by_key_matches_slot(tmp_path):
    """按槽位精确匹配 —— 冲突消解就靠它把"同槽位的新旧版本"摆到一起"""
    s = _store(tmp_path)
    s.add(_item("用户常驻上海", key="居住地"))
    s.add(_item("用户偏好先给条款号", key="回答格式"))

    same_slot = s.find_by_key("居住地", "t1", "u1")
    assert [m.text for m in same_slot] == ["用户常驻上海"]
    assert s.find_by_key("不存在的槽位", "t1", "u1") == []
    # 同 key 但换了用户 → 仍然看不到
    assert s.find_by_key("居住地", "t1", "u2") == []


def test_delete_all_only_clears_that_user(tmp_path):
    s = _store(tmp_path)
    s.add(_item("a", user="u1"))
    s.add(_item("b", user="u1"))
    s.add(_item("c", user="u2"))
    assert s.delete_all("t1", "u1") == 2
    assert s.count("t1", "u1") == 0
    assert s.count("t1", "u2") == 1


def test_purge_older_than_is_the_only_unscoped_op(tmp_path):
    """TTL 清理是维护操作, 刻意不带隔离维度 (业务侧删除仍必须校验归属)"""
    s = _store(tmp_path)
    old = _item("很久以前的事", user="u1")
    old.updated_at = time.time() - 400 * 86400
    s.add(old)
    s.add(_item("最近的事", user="u2"))

    assert s.purge_older_than(time.time() - 180 * 86400) == 1
    assert s.count("t1", "u1") == 0
    assert s.count("t1", "u2") == 1          # 新的不受影响


# ── 服务层: 门控 ─────────────────────────────────────────

def test_no_user_means_no_read_no_write(tmp_path):
    """user_id 为空 → 全程不写不读。**不回落到租户** —— 那等于全租户可见"""
    store = _store(tmp_path)
    svc = MemoryService(store, StubLLM(
        extraction=MemoryExtraction(facts=[MemoryFact(text="常驻上海", key="居住地", importance=1.0)])
    ), _settings())

    stats = svc.remember("我常驻上海", "好的", tenant_id="t1", user_id="")
    assert stats["extracted"] == 0 and stats["added"] == 0
    assert store.count("t1", "") == 0
    assert svc.recall("上海", "t1", "") == []


def test_disabled_memory_does_nothing(tmp_path):
    store = _store(tmp_path)
    svc = MemoryService(store, StubLLM(
        extraction=MemoryExtraction(facts=[MemoryFact(text="常驻上海", key="居住地", importance=1.0)])
    ), _settings(memory_enabled=False))
    assert svc.remember("我常驻上海", "好的", "t1", "u1")["added"] == 0
    assert store.count("t1", "u1") == 0


def test_low_importance_facts_are_dropped(tmp_path):
    store = _store(tmp_path)
    svc = MemoryService(store, StubLLM(extraction=MemoryExtraction(facts=[
        MemoryFact(text="用户常驻上海", importance=0.9, key="测试槽位"),
        MemoryFact(text="用户刚才问了个一次性问题", key="临时", importance=0.05),
    ])), _settings(memory_min_importance=0.3))

    stats = svc.remember("q", "a", "t1", "u1")
    assert stats["extracted"] == 1 and stats["added"] == 1
    assert [m.text for m in store.list("t1", "u1")] == ["用户常驻上海"]


# ── 服务层: 抽取与消解 ───────────────────────────────────

def test_remember_adds_extracted_facts_with_key_and_session(tmp_path):
    store = _store(tmp_path)
    svc = MemoryService(store, StubLLM(extraction=MemoryExtraction(facts=[
        MemoryFact(text="用户常驻上海", kind="fact", key="居住地", importance=0.8),
    ])), _settings())

    stats = svc.remember("我常驻上海", "好的", "t1", "u1", session_id="s-1")
    assert stats["added"] == 1
    m = store.list("t1", "u1")[0]
    assert (m.key, m.source_session, m.user_id) == ("居住地", "s-1", "u1")


def test_same_slot_conflict_updates_instead_of_duplicating(tmp_path):
    """**核心用例**: 同槽位的新值必须 UPDATE 掉旧值

    只靠向量相似度找不到这条冲突 —— "常驻上海"和"已搬到杭州"在向量空间里几乎
    不相似。能命中是因为按 key(居住地) 精确匹配把两条摆到了一起。
    """
    store = _store(tmp_path)
    store.add(_item("用户常驻上海", key="居住地", importance=0.5))
    old_id = store.list("t1", "u1")[0].memory_id

    svc = MemoryService(store, StubLLM(
        extraction=MemoryExtraction(facts=[
            MemoryFact(text="用户已搬到杭州", key="居住地", importance=0.8),
        ]),
        plan=ConsolidationPlan(ops=[
            MemoryOp(op="UPDATE", target_id=old_id, text="用户已搬到杭州",
                     key="居住地", importance=0.8),
        ]),
    ), _settings())

    stats = svc.remember("我搬到杭州了", "好的", "t1", "u1")
    assert stats["updated"] == 1 and stats["added"] == 0
    texts = [m.text for m in store.list("t1", "u1")]
    assert texts == ["用户已搬到杭州"], "新旧版本并存 → 回答会自相矛盾"


def test_candidates_include_same_slot_even_when_vectors_differ(tmp_path):
    """候选必须包含"同槽位但语义相反"的旧记忆 —— 否则冲突永远发现不了

    注意要造足够的干扰项: 记忆只有一两条时, 向量检索无论如何都会把它带出来,
    那样测不出 key 的作用。这里放三条与新事实更像的记忆, 把同槽位那条挤出 top-k,
    再断言"只有走 key 这条路才捞得回来"。
    """
    store = _store(tmp_path)
    store.add(_item("用户常驻上海", key="居住地"))
    for distractor in ("用户问过理赔流程", "用户问过条款号", "用户反馈过破损"):
        store.add(_item(distractor))

    fact = MemoryFact(text="用户已搬到杭州, 理赔/条款/破损 都关心", key="居住地")

    vector_only = [m.text for m, _ in store.search(fact.text, "t1", "u1", top_k=3)]
    assert "用户常驻上海" not in vector_only, "这条前提不成立就测不出 key 的作用"

    svc = MemoryService(store, StubLLM(extraction=MemoryExtraction(facts=[])), _settings())
    cands = [c.text for c in svc._candidates([fact], "t1", "u1")]
    assert "用户常驻上海" in cands, "同槽位的旧记忆必须进入消解上下文"


def test_apply_refuses_target_owned_by_another_user(tmp_path):
    """纵深防御: 即使 by_id 里混进了别人的记忆, `_apply` 也必须拒绝

    主防线是"候选来自带 (tenant, user) 过滤的检索", 正常路径绕不过去 —— 但边界
    检查不能只押在上游: 哪天有人给 `_candidates` 加了一条跨用户召回, 这里必须还
    拦得住。直接打私有方法就是为了测这层冗余。
    """
    store = _store(tmp_path)
    victim = _item("别人的记忆", tenant="t1", user="u2", key="居住地")
    store.add(victim)

    svc = MemoryService(store, StubLLM(), _settings())
    stats = {"extracted": 0, "added": 0, "updated": 0, "deleted": 0, "skipped": 0}
    by_id = {victim.memory_id: victim}          # 故意把别人的记忆塞进来

    svc._apply(MemoryOp(op="DELETE", target_id=victim.memory_id),
               by_id, "t1", "u1", "s-1", stats)
    svc._apply(MemoryOp(op="UPDATE", target_id=victim.memory_id, text="被改了"),
               by_id, "t1", "u1", "s-1", stats)

    assert stats["deleted"] == 0 and stats["updated"] == 0 and stats["skipped"] == 2
    assert store.list("t1", "u2")[0].text == "别人的记忆"   # 原样未动


def test_foreign_target_id_is_rejected(tmp_path):
    """模型编造的 target_id(或指向别人的记忆) 必须被拒 —— 不能凭它删改数据"""
    store = _store(tmp_path)
    victim = _item("别人的记忆", tenant="t1", user="u2", key="居住地")
    store.add(victim)
    # 当前用户自己也有一条同槽位记忆 —— 否则候选为空会走"直接 ADD"的短路,
    # 根本轮不到消解, 也就测不到"伪造 target_id"这条路径
    store.add(_item("用户常驻上海", tenant="t1", user="u1", key="居住地"))

    svc = MemoryService(store, StubLLM(
        extraction=MemoryExtraction(facts=[
            MemoryFact(text="用户常驻上海", key="居住地", importance=0.9),
        ]),
        plan=ConsolidationPlan(ops=[
            MemoryOp(op="DELETE", target_id=victim.memory_id),      # 指向别的用户
            MemoryOp(op="UPDATE", target_id="根本没这个id", text="x"),
        ]),
    ), _settings())

    stats = svc.remember("q", "a", "t1", "u1")
    assert stats["deleted"] == 0 and stats["updated"] == 0 and stats["skipped"] == 2
    assert store.count("t1", "u2") == 1        # 受害者的记忆完好


def test_noop_is_counted_not_applied(tmp_path):
    store = _store(tmp_path)
    existing = _item("用户常驻上海", key="居住地")
    store.add(existing)

    svc = MemoryService(store, StubLLM(
        extraction=MemoryExtraction(facts=[MemoryFact(text="用户常驻上海", key="居住地",
                                                      importance=0.9)]),
        plan=ConsolidationPlan(ops=[MemoryOp(op="NOOP", target_id=existing.memory_id)]),
    ), _settings())

    stats = svc.remember("q", "a", "t1", "u1")
    assert stats == {"extracted": 1, "added": 0, "updated": 0, "deleted": 0, "skipped": 1}
    assert store.count("t1", "u1") == 1


# ── 失败隔离: 记忆是尽力而为 ──────────────────────────────

def test_extraction_failure_is_swallowed(tmp_path):
    """抽取失败只记日志 —— 记忆写失败绝不能让用户拿不到回答"""
    store = _store(tmp_path)
    svc = MemoryService(store, StubLLM(error=RuntimeError("LLM 挂了")), _settings())
    stats = svc.remember("q", "a", "t1", "u1")
    assert stats["extracted"] == 0
    assert store.count("t1", "u1") == 0


def test_consolidation_failure_skips_rather_than_blind_add(tmp_path):
    """消解失败时**不记**而不是退化成无脑 ADD

    退化成 ADD 会把"可能冲突的新版本"直接塞进去, 制造出上面那种自相矛盾 ——
    宁可这轮不记, 下一轮再说。
    """
    store = _store(tmp_path)
    store.add(_item("用户常驻上海", key="居住地"))

    class _FailOnPlan(StubLLM):
        def structured_invoke(self, schema, messages):
            if schema is ConsolidationPlan:
                raise RuntimeError("消解调用失败")
            return super().structured_invoke(schema, messages)

    svc = MemoryService(store, _FailOnPlan(extraction=MemoryExtraction(facts=[
        MemoryFact(text="用户已搬到杭州", key="居住地", importance=0.9),
    ])), _settings())

    stats = svc.remember("q", "a", "t1", "u1")
    assert stats["added"] == 0 and stats["skipped"] == 1
    assert [m.text for m in store.list("t1", "u1")] == ["用户常驻上海"]


def test_recall_failure_returns_empty(tmp_path):
    class _BrokenStore:
        def search(self, *a, **kw):
            raise RuntimeError("向量库挂了")

    svc = MemoryService(_BrokenStore(), StubLLM(), _settings())
    assert svc.recall("上海", "t1", "u1") == []


# ── 召回格式 / 上限 / 遗忘 ────────────────────────────────

def test_format_includes_date_and_kind(tmp_path):
    """必须带"截至日期" —— 不标时间, 模型会把三个月前的偏好当成当前事实"""
    svc = MemoryService(_store(tmp_path), StubLLM(), _settings())
    text = svc.format_memories([
        _item("用户常驻上海", kind="fact"),
        _item("用户要求先给条款号", kind="preference"),
    ])
    assert "截至" in text and "2026-" in text
    assert "[fact] 用户常驻上海" in text
    assert "[preference] 用户要求先给条款号" in text
    assert "以用户当前说法为准" in text


def test_format_empty_is_empty_string(tmp_path):
    svc = MemoryService(_store(tmp_path), StubLLM(), _settings())
    assert svc.format_memories([]) == ""


def test_recall_applies_similarity_floor(tmp_path):
    """相似度地板: 够不着就不召回, 别把不相干的记忆塞进提示词

    注意查询词要选得能真正拉开相似度 —— 用"上海"查"用户常驻上海"时向量几乎同向
    (相似度≈1.0), 那种情况下任何地板都拦不住它。
    """
    store = _store(tmp_path)
    store.add(_item("用户常驻上海", key="居住地"))

    high = MemoryService(store, StubLLM(), _settings(memory_recall_min_similarity=0.99))
    assert high.recall("上海 理赔 条款", "t1", "u1") == []   # 相似度被稀释 → 拦下

    low = MemoryService(store, StubLLM(), _settings(memory_recall_min_similarity=0.1))
    assert len(low.recall("上海 理赔 条款", "t1", "u1")) == 1


def test_enforce_limit_keeps_most_important(tmp_path):
    store = _store(tmp_path)
    for i in range(6):
        store.add(_item(f"记忆{i}", importance=i / 10))
    svc = MemoryService(store, StubLLM(), _settings(memory_max_items=3))

    svc._enforce_limit("t1", "u1")

    kept = sorted(m.importance for m in store.list("t1", "u1"))
    assert len(kept) == 3
    assert kept == [0.3, 0.4, 0.5]           # 留下最重要的三条


def test_forget_and_forget_all(tmp_path):
    store = _store(tmp_path)
    store.add(_item("a", user="u1"))
    store.add(_item("b", user="u1"))
    store.add(_item("别人的", user="u2"))
    svc = MemoryService(store, StubLLM(), _settings())

    one = store.list("t1", "u1")[0]
    assert svc.forget(one.memory_id, "t1", "u1") is True
    assert svc.forget(one.memory_id, "t1", "u2") is False      # 别人的删不掉
    assert svc.forget_all("t1", "u1") == 1
    assert store.count("t1", "u1") == 0
    assert store.count("t1", "u2") == 1                        # 不受影响
    assert svc.forget_all("t1", "") == 0                        # 无用户维度 → 不动作


def test_sweep_purges_only_stale(tmp_path):
    store = _store(tmp_path)
    stale = _item("很久以前", user="u1")
    stale.updated_at = time.time() - 400 * 86400
    store.add(stale)
    store.add(_item("最近", user="u1"))

    svc = MemoryService(store, StubLLM(), _settings(memory_ttl_days=180))
    assert svc.sweep() == 1
    assert [m.text for m in store.list("t1", "u1")] == ["最近"]


def test_sweep_disabled_does_nothing(tmp_path):
    store = _store(tmp_path)
    stale = _item("很久以前", user="u1")
    stale.updated_at = time.time() - 400 * 86400
    store.add(stale)
    svc = MemoryService(store, StubLLM(), _settings(memory_enabled=False))
    assert svc.sweep() == 0
    assert store.count("t1", "u1") == 1


# ── 输出质量门: 实测模型会产出退化的抽取/更新 ─────────────
# 这一组不是"想当然的防御", 而是真实 7B 模型跑出来的问题:
#   抽取把槽位名直接填进 text → text="居住地", 整句信息全丢
#   消解把 UPDATE 文本写成「居住地: 上海」这种 字段名:值 → 覆盖掉本来正确的记忆
# 提示词里已经给了样例, 但小模型行为不稳定, 代码侧必须再兜一层。

def test_degenerate_extraction_is_dropped(tmp_path):
    """text 只是槽位名的抽取结果必须丢弃"""
    store = _store(tmp_path)
    svc = MemoryService(store, StubLLM(extraction=MemoryExtraction(facts=[
        MemoryFact(text="居住地", key="居住地", importance=0.9),          # 退化
        MemoryFact(text="用户现居杭州", key="居住地", importance=0.8),     # 正常
    ])), _settings())

    stats = svc.remember("我调到杭州了", "好的", "t1", "u1")
    assert stats["extracted"] == 1 and stats["added"] == 1
    assert [m.text for m in store.list("t1", "u1")] == ["用户现居杭州"]


def test_update_with_degenerate_text_keeps_the_old_value(tmp_path):
    """退化 UPDATE 必须被拒, 并**保留旧值** —— 覆盖成垃圾比不更新更糟"""
    store = _store(tmp_path)
    store.add(_item("用户常驻上海", key="居住地"))
    old_id = store.list("t1", "u1")[0].memory_id

    svc = MemoryService(store, StubLLM(
        extraction=MemoryExtraction(facts=[
            MemoryFact(text="用户现居杭州", key="居住地", importance=0.9),
        ]),
        plan=ConsolidationPlan(ops=[
            MemoryOp(op="UPDATE", target_id=old_id, text="居住地: 上海", key="居住地"),
        ]),
    ), _settings())

    stats = svc.remember("我调到杭州了", "好的", "t1", "u1")
    assert stats["updated"] == 0 and stats["skipped"] == 1
    assert [m.text for m in store.list("t1", "u1")] == ["用户常驻上海"]


def test_update_without_importance_keeps_the_old_score(tmp_path):
    """模型没给 importance 时**不许覆盖**原值

    实测: MemoryOp.importance 有默认值 0.3, 模型不填就变成 0.3, 两条 0.8/0.7 的
    记忆被压平成 0.3 —— importance 驱动召回排序与淘汰, 被压平就等于没有。
    """
    store = _store(tmp_path)
    store.add(_item("用户常驻上海", key="居住地", importance=0.9))
    old_id = store.list("t1", "u1")[0].memory_id

    svc = MemoryService(store, StubLLM(
        extraction=MemoryExtraction(facts=[
            MemoryFact(text="用户现居杭州", key="居住地", importance=0.8),
        ]),
        plan=ConsolidationPlan(ops=[
            MemoryOp(op="UPDATE", target_id=old_id, text="用户现居杭州", key="居住地"),
        ]),
    ), _settings())

    svc.remember("我调到杭州了", "好的", "t1", "u1")
    assert store.list("t1", "u1")[0].importance == 0.9      # 保留原分值


def test_add_without_importance_uses_the_fact_score(tmp_path):
    """ADD 且模型没给分值 → 用本轮新事实里最高的, 而不是拍默认值"""
    store = _store(tmp_path)
    store.add(_item("旧的无关记忆", key="别的槽位"))          # 制造候选, 走消解路径

    svc = MemoryService(store, StubLLM(
        extraction=MemoryExtraction(facts=[
            MemoryFact(text="用户负责浙江区域", key="负责区域", importance=0.7),
        ]),
        plan=ConsolidationPlan(ops=[MemoryOp(op="ADD", text="用户负责浙江区域",
                                              key="负责区域")]),
    ), _settings())

    svc.remember("我负责浙江", "好的", "t1", "u1")
    added = [m for m in store.list("t1", "u1") if m.text == "用户负责浙江区域"][0]
    assert added.importance == 0.7


def test_normal_update_still_applies(tmp_path):
    """别把质量门做成"什么都不改" —— 正常的 UPDATE 必须生效"""
    store = _store(tmp_path)
    store.add(_item("用户常驻上海", key="居住地"))
    old_id = store.list("t1", "u1")[0].memory_id

    svc = MemoryService(store, StubLLM(
        extraction=MemoryExtraction(facts=[
            MemoryFact(text="用户现居杭州", key="居住地", importance=0.9),
        ]),
        plan=ConsolidationPlan(ops=[
            MemoryOp(op="UPDATE", target_id=old_id, text="用户现居杭州", key="居住地"),
        ]),
    ), _settings())

    stats = svc.remember("我调到杭州了", "好的", "t1", "u1")
    assert stats["updated"] == 1
    assert [m.text for m in store.list("t1", "u1")] == ["用户现居杭州"]


# ── 事实与偏好必须分开抽 (实测: 合在一起偏好会被吞掉) ─────
# 真实 7B 模型的实测: 把两类放在同一个列表里、用 kind 区分时, 偏好**一条都抽不出来** ——
# 提示词里加偏好示例、列偏好信号词都没用。改成两个独立字段后模型才会分别考虑。
# 这几条测试锁住这个设计, 防止有人"顺手合并"回一个列表。

def test_preferences_field_is_stored_as_preference_kind(tmp_path):
    """来自 preferences 字段的条目, kind 由**代码**强制, 不信任模型填的 kind"""
    store = _store(tmp_path)
    svc = MemoryService(store, StubLLM(extraction=MemoryExtraction(
        facts=[MemoryFact(text="用户常驻上海", kind="fact", key="居住地", importance=0.8)],
        preferences=[MemoryFact(text="用户要求回答必须带上条款号", kind="fact",  # 模型填错了
                                key="回答格式要求", importance=0.9)],
    )), _settings())

    stats = svc.remember("我常驻上海。以后回答请带上条款号。", "好的", "t1", "u1")
    assert stats["extracted"] == 2 and stats["added"] == 2
    kinds = {m.text: m.kind for m in store.list("t1", "u1")}
    assert kinds["用户要求回答必须带上条款号"] == "preference"      # 被纠正
    assert kinds["用户常驻上海"] == "fact"


def test_preference_survives_alongside_facts(tmp_path):
    """一句话里既有事实又有偏好时, 两边都要落库 (实测的失败模式就是偏好被丢掉)"""
    store = _store(tmp_path)
    svc = MemoryService(store, StubLLM(extraction=MemoryExtraction(
        facts=[
            MemoryFact(text="用户常驻上海", key="居住地", importance=0.8),
            MemoryFact(text="用户负责华东区网点", key="负责区域", importance=0.7),
        ],
        preferences=[
            MemoryFact(text="用户要求回答必须带上条款号", key="回答格式要求", importance=0.9),
        ],
    )), _settings())

    svc.remember("我常驻上海，负责华东区的网点。以后回答请带上条款号。", "好的", "t1", "u1")
    assert len(store.list("t1", "u1")) == 3
    assert sum(1 for m in store.list("t1", "u1") if m.kind == "preference") == 1


def test_degenerate_preference_is_also_dropped(tmp_path):
    """质量门对偏好同样生效 (退化输出不分字段)"""
    store = _store(tmp_path)
    svc = MemoryService(store, StubLLM(extraction=MemoryExtraction(
        facts=[], preferences=[MemoryFact(text="回答格式要求", key="回答格式要求",
                                          importance=0.9)],
    )), _settings())

    stats = svc.remember("以后回答请带上条款号", "好的", "t1", "u1")
    assert stats["extracted"] == 0 and store.count("t1", "u1") == 0
