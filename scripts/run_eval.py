#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""命令行评测入口 — 不启动 HTTP 服务, 直接跑评测 (检索命中率 + 答案质量 + 拒答准确率)

运行方式 (从项目根目录, 需 Ollama 运行中):
  python3 scripts/run_eval.py                # 完整评测 (含 LLM 评分, 较慢)
  python3 scripts/run_eval.py --no-judge     # 只测检索命中率 (快)
  python3 scripts/run_eval.py --force-reingest   # 强制重建索引后评测
  python3 scripts/run_eval.py --top-k 5
  python3 scripts/run_eval.py --scan-top-k "1,3,5"        # top_k 参数扫描 (P2-B)
  python3 scripts/run_eval.py --compare-mode              # hybrid / vector / bm25 对照 (P2-B)
  python3 scripts/run_eval.py --compare-mode --rerank     # 同上, 但启用 LLM 精排
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

# 脚本直接运行时 sys.path[0] 是 scripts/, 插入项目根目录使 app 包可导入
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from app.core.logging import setup_logging
from app.infrastructure.embeddings import create_embedding_client
from app.infrastructure.llm import create_llm_client
from app.infrastructure.vector_store import ChromaVectorStore, create_vector_store
from app.services.eval_service import EvalService
from app.services.rag_service import RagService


def _print_details(result: dict) -> None:
    for d in result["details"]:
        # 拒答用例无 hit 概念, 只判「是否如实拒答」
        if d.get("kind") == "refusal":
            refused = d.get("refused")
            mark = {True: "🚫", False: "⚠️"}.get(refused, "•")
            note = {True: "如实拒答", False: "未拒答(编造)", None: "未评估"}[refused]
            print(f"  {mark} [REFUSE] {d['question']} "
                  f"(sim={d['top_similarity']}) | {note}")
            continue
        mark = "✅" if d["hit"] else "❌"
        extra = ""
        if d.get("faithfulness") is not None:
            extra = (f" | 忠实={d['faithfulness']:.2f} 完整={d['completeness']:.2f}"
                     f" | {d.get('reason', '')[:40]}")
        print(f"  {mark} [{d['expect']}] {d['question']} (sim={d['top_similarity']}){extra}")


def _print_summary(result: dict) -> None:
    print(f"🎯 检索命中率: {result['hit_rate']:.0%} (top_k={result['top_k']})")
    if result.get("refusal_accuracy") is not None:
        print(f"🚫 拒答准确率: {result['refusal_accuracy']:.0%} "
              f"({result.get('refusal_checked', 0)}/{result.get('refusal_total', 0)} 条)"
              " — 知识库无答案时应如实拒答, 而非编造")
    if result.get("faithfulness_avg") is not None:
        print(f"⭐ 答案质量: 忠实性={result['faithfulness_avg']:.2f} "
              f"完整性={result['completeness_avg']:.2f} (judged={result['judged_cases']})")
    comp = result.get("compare")
    if comp:
        print(f"📈 对比上次: hit_rate {comp['hit_rate_delta']:+.2f}"
              + (f" | 拒答 {comp['refusal_accuracy_delta']:+.2f}"
                 if comp.get("refusal_accuracy_delta") is not None else "")
              + (f" | 忠实 {comp.get('faithfulness_delta', 0):+.2f}"
                 f" | 完整 {comp.get('completeness_delta', 0):+.2f}"
                 if comp.get("faithfulness_delta") is not None else ""))


def main() -> None:
    parser = argparse.ArgumentParser(description="检索命中率 + 答案质量评测 (命令行)")
    parser.add_argument("--force-reingest", action="store_true", help="强制重建索引")
    parser.add_argument("--top-k", type=int, default=None, help="检索 Top-K (默认走配置)")
    parser.add_argument("--no-judge", action="store_true", help="跳过 LLM 答案评分, 只测检索")
    parser.add_argument("--scan-top-k", type=str, default=None,
                        help='top_k 参数扫描: 逗号分隔, 如 "1,3,5" (不调 LLM)')
    parser.add_argument("--compare-mode", action="store_true",
                        help="对照 hybrid / vector / bm25 检索模式 (临时索引, 不污染正式库)")
    parser.add_argument("--rerank", action="store_true",
                        help="启用 LLM 精排 (配合 --compare-mode 可对照重排增益)")
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.log_level)

    embeddings = create_embedding_client(settings)
    llm = create_llm_client(settings)

    if args.compare_mode:
        _compare_modes(settings, embeddings, llm, args.top_k, rerank=args.rerank)
        return
    if args.scan_top_k:
        _scan_top_k(settings, embeddings, llm, args.scan_top_k)
        return

    vector_store = create_vector_store(settings, embeddings)
    rag_service = RagService(vector_store, llm, settings)

    print("📚 加载索引...")
    n, rebuilt = rag_service.ingest(force=args.force_reingest)
    print(f"   {n} chunks (rebuilt={rebuilt})")

    print("📊 开始评测...")
    result = EvalService(rag_service, llm, settings).run(
        top_k=args.top_k or settings.top_k,
        judge=not args.no_judge,
    )
    _print_details(result)
    print()
    _print_summary(result)


def _scan_top_k(settings, embeddings, llm, csv: str) -> None:
    """top_k 参数扫描: 不调 LLM (纯检索), 不写评测历史"""
    vector_store = create_vector_store(settings, embeddings)
    rag_service = RagService(vector_store, llm, settings)
    n, _ = rag_service.ingest(force=False)
    print(f"📚 索引 {n} chunks, top_k 扫描开始...")
    svc = EvalService(rag_service, llm, settings, record_history=False)
    print(f"  {'top_k':<6}{'hit_rate':<10}{'avg_sim':<10}")
    for k in [int(x) for x in csv.split(",") if x.strip()]:
        result = svc.run(top_k=k, judge=False)
        sims = [d["top_similarity"] for d in result["details"]]
        avg_sim = sum(sims) / len(sims) if sims else 0
        print(f"  {k:<6}{result['hit_rate']:<10.2f}{avg_sim:<10.3f}")


RECALL_KS = (1, 3, 5, 10)


def _recall_at_k(rag, cases: list[dict]) -> dict[str, float]:
    """对每个 k 统计「前 k 条里含期望关键词」的比例 (Recall@k)。

    比单一 hit_rate 信息量大得多: hit_rate 只在某一个 k 上取一个点, 看不出
    两种检索方式「谁头部更准、谁覆盖更全」。重排的价值恰恰体现在低 k 上,
    所以必须分开看 R@1 / R@3。
    """
    ks = [k for k in RECALL_KS]
    ranks: list[int] = []
    for case in cases:
        chunks = rag.retrieve(case["question"], top_k=max(ks), tenant_id="shared")
        keywords = case["expect_keywords"]
        rank = next(
            (i for i, c in enumerate(chunks, 1) if any(k in c["content"] for k in keywords)),
            0,
        )
        ranks.append(rank)
    n = len(ranks) or 1
    return {f"R@{k}": sum(1 for r in ranks if 0 < r <= k) / n for k in ks}


def _compare_modes(settings, embeddings, llm, top_k, rerank: bool = False) -> None:
    """检索模式对照: hybrid / vector / bm25, 各自用独立临时索引, 不污染正式库

    三种都跑才能量化「混合检索的增益来自哪里」—— 只有 hybrid 与 vector 两组
    数字时, 无法判断 BM25 到底有没有起作用。

    rerank=True 时同时启用精排, 可直接对照「重排把低 k 精度抬了多少」。
    """
    mode_settings = settings.model_copy(update={"rerank_enabled": rerank})
    tag = " + 精排(LLM rerank)" if rerank else " (仅粗排)"
    print(f"🆚 检索模式对比{tag} — 临时索引, 语料为 data/corpus/\n")

    cases = [
        c for c in json.loads(
            settings.eval_cases_file.read_text(encoding="utf-8")
        ) if not c.get("expect_refusal")
    ]
    header = f"  {'mode':<8}{'chunks':<9}" + "".join(f"{'R@'+str(k):<8}" for k in RECALL_KS)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for mode in ("hybrid", "vector", "bm25"):
        with tempfile.TemporaryDirectory(prefix="sopqa-cmp-") as d:
            store = ChromaVectorStore(
                persist_dir=d,
                collection_name=settings.collection_name,
                embeddings=embeddings,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                retrieval_mode=mode,
            )
            rag = RagService(store, llm, mode_settings)
            n, _ = rag.ingest(force=True)
            recall = _recall_at_k(rag, cases)
            print(f"  {mode:<8}{n:<9}" + "".join(
                f"{recall[f'R@{k}']:<8.2f}" for k in RECALL_KS
            ))


if __name__ == "__main__":
    main()
