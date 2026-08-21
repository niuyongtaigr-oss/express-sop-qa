#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""命令行评测入口 — 不启动 HTTP 服务, 直接跑评测 (检索命中率 + LLM-as-Judge)

运行方式 (从项目根目录, 需 Ollama 运行中):
  python3 scripts/run_eval.py                # 完整评测 (含 LLM 评分, 较慢)
  python3 scripts/run_eval.py --no-judge     # 只测检索命中率 (快)
  python3 scripts/run_eval.py --force-reingest   # 强制重建索引后评测
  python3 scripts/run_eval.py --top-k 5
"""

import argparse
import sys
from pathlib import Path

# 脚本直接运行时 sys.path[0] 是 scripts/, 插入项目根目录使 app 包可导入
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from app.core.logging import setup_logging
from app.infrastructure.embeddings import create_embedding_client
from app.infrastructure.llm import create_llm_client
from app.infrastructure.vector_store import create_vector_store
from app.services.eval_service import EvalService
from app.services.rag_service import RagService


def main() -> None:
    parser = argparse.ArgumentParser(description="检索命中率 + 答案质量评测 (命令行)")
    parser.add_argument("--force-reingest", action="store_true", help="强制重建索引")
    parser.add_argument("--top-k", type=int, default=None, help="检索 Top-K (默认走配置)")
    parser.add_argument("--no-judge", action="store_true", help="跳过 LLM 答案评分, 只测检索")
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.log_level)

    embeddings = create_embedding_client(settings)
    llm = create_llm_client(settings)
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
    for d in result["details"]:
        mark = "✅" if d["hit"] else "❌"
        extra = ""
        if d.get("faithfulness") is not None:
            extra = (f" | 忠实={d['faithfulness']:.2f} 完整={d['completeness']:.2f}"
                     f" | {d.get('reason', '')[:40]}")
        print(f"  {mark} [{d['expect']}] {d['question']} (sim={d['top_similarity']}){extra}")

    print(f"\n🎯 检索命中率: {result['hit_rate']:.0%} (top_k={result['top_k']})")
    if result.get("faithfulness_avg") is not None:
        print(f"⭐ 答案质量: 忠实性={result['faithfulness_avg']:.2f} "
              f"完整性={result['completeness_avg']:.2f} (judged={result['judged_cases']})")


if __name__ == "__main__":
    main()
