# 快递 SOP 智能问答系统

生产级架构的 FastAPI + LangGraph + RAG 项目：意图识别路由 → 知识库问答 / 直接回答 / 多轮检索，
面向有 Java/PHP 后端背景、转向 AI Agent 开发的读者。

## 架构

```
                ┌────────────── FastAPI (app/main.py) ──────────────┐
                │        所有业务接口挂在 /api/v1 前缀下             │
                │  /health /chat /chat/stream /rag/* /eval/*         │
                └───────────────┬────────────────────────────────────┘
              core/ 横切层: 结构化 JSON 日志 + trace_id │ 请求日志中间件
                          统一异常处理 │ X-API-Key 鉴权 (可选)
                                │
                     ┌──────────▼───────────┐
                     │  services/ 业务层     │  不碰 HTTP 细节
                     │  chat_service: 限流   │  (Semaphore) + 超时降级
                     │  (wait_for+fallback) │
                     └──────────┬───────────┘
                     ┌──────────▼───────────┐
                     │  agents/ LangGraph    │  意图识别 → 条件路由
                     │  (依赖注入装配)       │
                     └───┬───────┬──────┬───┘
              rag_qa ────┘       │      └──── multi_hop (检索→阈值判断→改写再检索)
              direct ────────────┘
                                │
                     ┌──────────▼───────────┐
                     │  services/rag_service │  ingest / retrieve / ask
                     └──────────┬───────────┘
        ┌───────────────────────┼────────────────────────┐
┌───────▼────────┐   ┌──────────▼──────────┐   ┌─────────▼─────────┐
│ infrastructure │   │  infrastructure     │   │  infrastructure   │
│ llm.py         │   │  embeddings.py      │   │  vector_store.py  │
│ LLMClient 协议 │   │  EmbeddingClient    │   │  VectorStore 协议 │
│ ChatOllama 实现│   │  Ollama 实现(bge-m3)│   │  Chroma 持久化实现│
└────────────────┘   └─────────────────────┘   └───────────────────┘
     全部走 Protocol 抽象 + 工厂注入: 换实现不改业务层
```

## 分层职责

| 目录 | 职责 |
|------|------|
| `app/main.py` | 应用工厂 `create_app()` + lifespan（启动时建/加载索引） |
| `app/config.py` | pydantic-settings 集中配置，环境变量前缀 `SOP_QA_`，仓库零密钥 |
| `app/core/` | 横切关注点：结构化日志+trace_id、统一异常、请求日志中间件、API Key 鉴权 |
| `app/schemas/` | Pydantic DTO，按资源分文件（chat/rag/eval/common） |
| `app/infrastructure/` | 外部系统适配层：LLM / Embedding / 向量库，Protocol 抽象 + 工厂 |
| `app/services/` | 业务逻辑层：RAG 服务、问答编排（限流/超时降级）、检索评测 |
| `app/agents/` | LangGraph 编排：ChatState、节点工厂、图装配（依赖注入，无全局单例） |
| `app/api/` | HTTP 层：依赖注入 + 版本化路由 `/api/v1` |
| `scripts/run_eval.py` | 命令行评测入口（不启动 HTTP） |
| `data/` | `sop.txt` 知识库源文档；`chroma/` 为 Chroma 运行时产物（gitignore） |

## 接口清单（前缀 `/api/v1`）

| 方法 | 路径 | 功能 | 鉴权 |
|------|------|------|------|
| GET | `/api/v1/health` | 健康检查：服务存活 + 索引 chunk 数 + Ollama 可达性 | 否 |
| POST | `/api/v1/chat` | 智能问答主接口（意图识别→路由→回答），限流+超时降级 | 是* |
| POST | `/api/v1/chat/stream` | 同上，SSE 分段推送（`data: {"delta": ...}`，结束 `data: [DONE]`） | 是* |
| POST | `/api/v1/rag/query` | 知识库直通查询（绕过编排，检索+生成，调试/评测用） | 是* |
| POST | `/api/v1/rag/ingest` | 重建索引（默认幂等跳过；`?force=true` 强制重建） | 是* |
| POST | `/api/v1/eval/run` | 提交检索命中率评测（后台任务），返回 task_id | 是* |
| GET | `/api/v1/eval/tasks/{task_id}` | 轮询评测结果（status/hit_rate/details） | 是* |

\* 配置了 `SOP_QA_API_KEY` 时需携带 `X-API-Key` 请求头；未配置则放行并日志告警（仅本地开发）。

### 请求/响应示例

```bash
# 智能问答
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"question": "包裹破损了怎么申请理赔?"}'
# → {"answer": "...", "intent": "rag_qa", "sources": [{"content","tags","similarity"}],
#    "trace_id": "...", "elapsed_ms": 123}

# 知识库直通
curl -X POST http://127.0.0.1:8000/api/v1/rag/query \
  -H 'Content-Type: application/json' -d '{"query": "理赔流程", "top_k": 3}'

# 评测（异步任务）
curl -X POST http://127.0.0.1:8000/api/v1/eval/run        # → {"task_id": "...", "status": "pending"}
curl http://127.0.0.1:8000/api/v1/eval/tasks/<task_id>    # → {"status": "done", "hit_rate": ..., "details": [...]}
```

统一错误响应：`{"error": {"code": "...", "message": "...", "trace_id": "..."}}`

## 配置

全部走环境变量 / `.env`（前缀 `SOP_QA_`），示例见 `.env.example`。关键项：

- `SOP_QA_LLM_MODEL` / `SOP_QA_EMBED_MODEL`：Ollama 模型（默认 qwen2.5:7b / bge-m3）
- `SOP_QA_MAX_CONCURRENCY` / `SOP_QA_CHAT_TIMEOUT_S`：限流并发数 / 超时秒数
- `SOP_QA_MULTI_HOP_MAX_ROUNDS` / `SOP_QA_MULTI_HOP_SIMILARITY_THRESHOLD`：多轮检索参数
- `SOP_QA_API_KEY`：可选 API Key 鉴权

## 本地启动

前置：Ollama 运行中（`localhost:11434`），已拉取 `qwen2.5:7b` 和 `bge-m3`。

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # 按需修改

uvicorn app.main:app --port 8000        # 或 python3 -m app.main
# 文档: http://127.0.0.1:8000/docs

# 命令行评测（不启动 HTTP）
python3 scripts/run_eval.py
```

启动时（lifespan）自动加载/构建 Chroma 索引（持久化在 `data/chroma/`，已有索引不重建）。

## 设计要点

- **安全**：配置全走 pydantic-settings + 环境变量；Pydantic 入参校验（长度/范围）；
  SSE 输出 `json.dumps` 序列化；异常捕获具体化并记日志；统一错误响应格式。
- **扩展性**：LLM/Embedding/向量库 Protocol 抽象 + 工厂注入；路由版本化；服务层与 HTTP 层解耦。
- **生产化**：`asyncio.Semaphore` 限流（超限排队）、`asyncio.wait_for` + fallback 超时降级、
  结构化 JSON 日志 + contextvars trace_id 全链路串联。
