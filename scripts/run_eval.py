#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""命令行评测入口 — 不启动 HTTP 服务, 直接跑检索命中率评测

运行方式 (从项目根目录, 需 Ollama 运行中):
  python3 scripts/run_eval.py
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
    parser = argparse.ArgumentParser(description="检索命中率评测 (命令行)")
    parser.add_argument("--force-reingest", action="store_true", help="强制重建索引")
    parser.add_argument("--top-k", type=int, default=None, help="检索 Top-K (默认走配置)")
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
    result = EvalService(rag_service).run(top_k=args.top_k or settings.top_k)
    for d in result["details"]:
        mark = "✅" if d["hit"] else "❌"
        print(f"  {mark} [{d['expect']}] {d['question']} (sim={d['top_similarity']})")
    print(f"\n🎯 命中率: {result['hit_rate']:.0%} (top_k={result['top_k']})")


if __name__ == "__main__":
    main()
