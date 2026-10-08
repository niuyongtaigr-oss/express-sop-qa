"""长期记忆跨会话评测 — 命令行入口

量的是**记忆这件事本身有没有用、有没有害**, 而不是检索命中率:

  · 召回率    前一阶段说过的事实, 后一阶段问起时答得出吗
  · 时效性    事实更新后旧值有没有被消解掉 (新旧并存 = 回答自相矛盾)
  · 隔离      别人的记忆会不会漏给当前用户
  · 误记率    不该记的对话(寒暄/知识库问答)有没有被写进记忆库

最后一项最容易被忽略, 却是最要命的: 记忆会被**反复召回**, 记错一条的影响是持续的。

做法要点:
  · 用**临时 chroma 目录**, 不碰正式索引与正式记忆库 (与 run_eval.py 同一套纪律)
  · 用例是**有序脚本**而非独立用例: 真实 LLM 下每轮问答约 40s、每次抽取约 20s,
    逐条独立跑要 20 分钟; 复用 setup 才能在一次运行里量完四个维度
  · 每轮之后 `await chat.flush_memory()` 等后台写入落定 —— 不猜 sleep 秒数,
    否则慢机器上会出现随机失败

用法:
    python scripts/run_memory_eval.py
    python scripts/run_memory_eval.py --cases data/memory_eval_cases.json --out /tmp/mem.json
"""

import argparse
import asyncio
import json
import logging
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.graph import build_chat_graph  # noqa: E402
from app.config import Settings  # noqa: E402
from app.infrastructure.embeddings import create_embedding_client  # noqa: E402
from app.infrastructure.llm import create_intent_llm, create_llm_client  # noqa: E402
from app.infrastructure.memory_store import create_memory_store  # noqa: E402
from app.infrastructure.vector_store import create_vector_store  # noqa: E402
from app.services.chat_service import ChatService  # noqa: E402
from app.services.memory_service import MemoryService  # noqa: E402
from app.services.rag_service import RagService  # noqa: E402
from app.services.session_service import SessionStore  # noqa: E402

TENANT = "default"


# ── 打分 (纯函数, 便于单测) ──────────────────────────────
def score_answer(
    answer: str, expect_any: list[str], expect_none: list[str]
) -> tuple[bool, str]:
    """判定一次探针的答案, 返回 (是否通过, 原因)

    语义: `expect_any` 是"至少命中一个"(空列表表示不要求命中);
    `expect_none` 是"一个都不许出现"。两者同时给出时, 后者优先报错 ——
    出现旧值是比"没答出来"更严重的问题(说明记忆里新旧并存)。
    """
    hit_none = [k for k in expect_none if k in answer]
    if hit_none:
        return False, f"出现了不该出现的 {hit_none}"
    if expect_any and not any(k in answer for k in expect_any):
        return False, f"未命中任何期望 {expect_any}"
    return True, "ok"


def score_memory(
    texts: list[str], expect_any: list[str], expect_none: list[str]
) -> tuple[bool, str]:
    """判定记忆库里**实际存了什么**, 返回 (是否通过, 原因)

    为什么不能只看答案: 实测答案里出现「根据条款3」这种模型自己编的引用, 让
    `expect_any=["条款"]` 白白通过 —— 而记忆库里那条真实偏好已经被一条错误内容
    覆盖掉了("回答请友好简洁", 来自助手自己的承诺)。**断言要打在存储上, 不是
    打在模型的措辞上**, 否则测的是模型的嘴, 不是系统的状态。
    """
    blob = "\n".join(texts)
    hit_none = [k for k in expect_none if k in blob]
    if hit_none:
        return False, f"记忆里出现了不该有的 {hit_none}"
    if expect_any and not any(k in blob for k in expect_any):
        return False, f"记忆里没有期望的 {expect_any}"
    return True, "ok"


def aggregate(probes: list[dict], results: list[dict]) -> dict:
    """把逐条结果汇总成四个指标

    分母为 0 时返回 None 而不是 0.0 —— "没测"和"测了是 0 分"是两件事, 混在一起
    会让人以为某个维度已经验证过了。
    """
    def rate(dimension: str | None) -> float | None:
        """按**用例显式声明的维度**汇总。

        不要用"有没有 expect_none"之类的特征去猜维度: 时效检查(旧值不许出现)与
        隔离检查(别人的不许出现)都带 expect_none, 一猜就会把两者混成一个指标 ——
        实测这么写会让 isolation_rate 变成两个维度的平均值, 数字看着正常但是错的。
        """
        picked = [
            r for r, p in zip(results, probes)
            if p["kind"] == "check" and (dimension is None or p.get("dimension") == dimension)
        ]
        if not picked:
            return None
        return sum(1 for r in picked if r["passed"]) / len(picked)

    no_mem = [r for r, p in zip(results, probes) if p["kind"] == "no_memory"]
    return {
        "checks": len([p for p in probes if p["kind"] == "check"]),
        "recall_rate": rate("recall"),
        "isolation_rate": rate("isolation"),
        "conflict_rate": rate("conflict"),
        "false_memory_rate": (
            sum(1 for r in no_mem if r["memories_after"] > r["memories_before"]) / len(no_mem)
            if no_mem else None
        ),
    }


# ── 运行 ─────────────────────────────────────────────────
async def run(cases_file: Path, keep_db: bool = False) -> dict:
    settings = Settings(_env_file=None, memory_enabled=True, cache_enabled=False)
    embeddings = create_embedding_client(settings)
    llm = create_llm_client(settings)

    with tempfile.TemporaryDirectory(prefix="mem-eval-") as tmp:
        cfg = settings.model_copy(update={"chroma_dir": tmp})
        store = create_vector_store(cfg, embeddings)
        rag = RagService(store, llm, cfg)
        rag.ingest(force=True)

        memory_store = create_memory_store(cfg, embeddings)
        memory = MemoryService(memory_store, llm, cfg)
        graph = build_chat_graph(rag, llm, cfg, intent_llm=create_intent_llm(cfg))
        chat = ChatService(
            graph,
            SessionStore(ttl_s=3600, max_turns=10, max_sessions=100),
            cfg,
            memory=memory,
        )

        probes = json.loads(cases_file.read_text(encoding="utf-8"))["probes"]
        results: list[dict] = []
        print(f"🧠 长期记忆跨会话评测 — 临时记忆库, {len(probes)} 步\n")
        print(f"  {'id':<30}{'type':<11}{'结果':<6}说明")
        print("  " + "-" * 86)

        for probe in probes:
            user = probe["user"]
            before = memory_store.count(TENANT, user)
            res = await chat.chat(
                probe["question"],
                session_id=f"eval-{probe['id']}"[:32].ljust(16, "0"),
                tenant_id=TENANT,
                user_id=user,
            )
            await chat.flush_memory()
            answer = res.get("answer", "")
            after = memory_store.count(TENANT, user)

            memories_now = [m.text for m in memory_store.list(TENANT, user)]
            if probe["kind"] == "check":
                passed, why = score_answer(
                    answer, probe.get("expect_any", []), probe.get("expect_none", [])
                )
                if passed and (probe.get("expect_memory_any")
                               or probe.get("expect_memory_none")):
                    passed, why = score_memory(
                        memories_now, probe.get("expect_memory_any", []),
                        probe.get("expect_memory_none", []),
                    )
            else:
                passed, why = True, probe["kind"]

            results.append({
                "id": probe["id"],
                "kind": probe["kind"],
                "user": user,
                "question": probe["question"],
                "answer": answer,
                "intent": res.get("intent", ""),
                "passed": passed,
                "reason": why,
                "memories_before": before,
                "memories_after": after,
                "memories_now": memories_now,
            })
            mark = "✅" if passed else "❌"
            print(f"  {probe['id']:<30}{probe['kind']:<11}{mark:<6}{why}")

        summary = aggregate(probes, results)
        memories = {
            u: [{"text": m.text, "key": m.key, "kind": m.kind,
                 "importance": m.importance}
                for m in memory_store.list(TENANT, u)]
            for u in sorted({p["user"] for p in probes})
        }

    return {"summary": summary, "results": results, "memories": memories}


def _fmt(value: float | None) -> str:
    return "未测" if value is None else f"{value:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="长期记忆跨会话评测")
    parser.add_argument("--cases", default="data/memory_eval_cases.json")
    parser.add_argument("--out", default=None, help="把完整结果写到 JSON 文件")
    parser.add_argument("--verbose", action="store_true", help="打印每条答案")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    payload = asyncio.run(run(Path(args.cases)))

    s = payload["summary"]
    print("\n  ── 指标 ──")
    print(f"  召回率        : {_fmt(s['recall_rate'])}   (说过的事, 之后问起答得出吗)")
    print(f"  时效性        : {_fmt(s['conflict_rate'])}   (事实更新后旧值有没有消解掉)")
    print(f"  隔离          : {_fmt(s['isolation_rate'])}   (别人的记忆有没有漏出来)")
    print(f"  误记率(越低越好): {_fmt(s['false_memory_rate'])}   (寒暄/知识库问答被误记的比例)")

    if args.verbose:
        print("\n  ── 逐条答案 ──")
        for r in payload["results"]:
            print(f"\n  [{r['id']}] intent={r['intent']} {'✅' if r['passed'] else '❌'}")
            print(f"    Q: {r['question']}")
            print(f"    A: {r['answer'][:160]}")

    print("\n  ── 最终记忆库 ──")
    for user, items in payload["memories"].items():
        print(f"  {user}: {len(items)} 条")
        for m in items:
            print(f"    · [{m['kind']}] key={m['key']!r} imp={m['importance']} :: {m['text']}")

    if args.out:
        Path(args.out).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n  完整结果已写入 {args.out}")

    failed = [r["id"] for r in payload["results"] if not r["passed"]]
    if failed:
        print(f"\n  ❌ 未通过: {failed}")
        sys.exit(1)
    print("\n  ✅ 全部通过")


if __name__ == "__main__":
    main()
