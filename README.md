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
| `app/infrastructure/` | 外部系统适配层：LLM / Embedding / 向量库 / 文档解析，Protocol 抽象 + 工厂 |
| `app/services/` | 业务逻辑层：RAG 服务、问答编排（限流/超时降级）、检索评测 |
| `app/agents/` | LangGraph 编排：ChatState、节点工厂、图装配（依赖注入，无全局单例） |
| `app/api/` | HTTP 层：依赖注入 + 版本化路由 `/api/v1` |
| `scripts/run_eval.py` | 命令行评测入口（不启动 HTTP） |
| `data/` | `sop.txt` 知识库源文档；`chroma/` 为 Chroma 运行时产物（gitignore） |

## 接口清单（前缀 `/api/v1`）

| 方法 | 路径 | 功能 | 鉴权 |
|------|------|------|------|
| GET | `/api/v1/health` | 健康检查：服务存活 + 索引 chunk 数 + Ollama 可达性 | 否 |
| GET | `/api/v1/metrics` | Prometheus 指标（HTTP/chat/缓存/评测等，见 `app/core/metrics.py`） | 否 |
| POST | `/api/v1/chat` | 智能问答主接口（意图识别→路由→回答），限流+超时降级，支持 `session_id` 多轮记忆 + 答案缓存 | 是* |
| POST | `/api/v1/chat/stream` | 同上，SSE 真流式（token 级，事件带 `type` 字段，见下） | 是* |
| POST | `/api/v1/chat/feedback` | 用户反馈（1-5 分 + 纠错文本），负面自动告警 | 是* |
| GET | `/api/v1/chat/feedback/stats` | 反馈统计（均分/负面率/最近低分问题） | 是* |
| POST | `/api/v1/rag/query` | 知识库直通查询（绕过编排，检索+生成，调试/评测用） | 是* |
| POST | `/api/v1/rag/ingest` | 导入默认 SOP 文档（默认幂等跳过；`?force=true` 清空重建） | 是* |
| GET | `/api/v1/rag/docs` | 知识库文档清单（doc_id / title / chunk 数） | 是* |
| POST | `/api/v1/rag/docs` | 增量导入/覆盖文档（upsert：同 doc_id 旧 chunk 先删后加） | 是* |
| DELETE | `/api/v1/rag/docs/{doc_id}` | 删除文档及其全部 chunk | 是* |
| GET | `/api/v1/rag/docs/formats` | 支持的文档格式（前端设置上传 accept 用） | 是* |
| POST | `/api/v1/rag/docs/upload` | **上传文件**（PDF/Word/Excel/CSV/文本）解析并入库 | 是* |
| POST | `/api/v1/eval/run` | 提交评测（检索命中率 + LLM-as-Judge 答案质量 + 拒答准确率），后台任务返回 task_id | 是* |
| GET | `/api/v1/eval/tasks/{task_id}` | 轮询评测结果（hit_rate/拒答准确率/忠实性/完整性 + 回归对比 compare） | 是* |

\* 配置了 `SOP_QA_API_KEY` 时需携带 `X-API-Key` 请求头；未配置则放行并日志告警（仅本地开发）。
启用 `SOP_QA_RATE_LIMIT_ENABLED=true` 后，上述业务接口额外按 `X-API-Key`/IP 限流（429 + `Retry-After`）。

### 请求/响应示例

```bash
# 智能问答 (带会话 ID, 多轮记忆共享上下文)
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"question": "包裹破损了怎么申请理赔?", "session_id": "sess-001"}'
# → {"answer": "...", "intent": "rag_qa", "sources": [{"content","doc_id","title","tags","similarity"}],
#    "trace_id": "...", "elapsed_ms": 123, "session_id": "sess-001"}

# 知识库直通
curl -X POST http://127.0.0.1:8000/api/v1/rag/query \
  -H 'Content-Type: application/json' -d '{"query": "理赔流程", "top_k": 3}'

# 多文档管理 (P1.6)
curl -X GET  http://127.0.0.1:8000/api/v1/rag/docs
curl -X POST http://127.0.0.1:8000/api/v1/rag/docs \
  -H 'Content-Type: application/json' \
  -d '{"doc_id": "claim-rules", "title": "理赔细则", "content": "理赔时限..."}'
curl -X DELETE http://127.0.0.1:8000/api/v1/rag/docs/claim-rules

# 上传文件入库 (P1.7) — PDF / Word / Excel / CSV / 文本
curl -X GET  http://127.0.0.1:8000/api/v1/rag/docs/formats   # 支持的扩展名
curl -X POST http://127.0.0.1:8000/api/v1/rag/docs/upload \
  -F 'file=@理赔制度.pdf' -F 'title=理赔制度'
# → {"doc_id":"...","title":"理赔制度","indexed_chunks":12,
#    "source":{"ext":".pdf","pages":8,"pages_with_text":8,"chars":6321,"truncated":false}}

# 评测（异步任务）
curl -X POST http://127.0.0.1:8000/api/v1/eval/run        # → {"task_id": "...", "status": "pending"}
curl http://127.0.0.1:8000/api/v1/eval/tasks/<task_id>    # → {"status": "done", "hit_rate": ..., "details": [...]}
```

### SSE 流式事件（`/chat/stream`，P0.2 真流式）

逐 token 推送（不再按标点切段），事件均为 `data: {json}` 行 + 空行，结束 `data: [DONE]`：

```
data: {"type": "intent", "intent": "rag_qa", "reason": "..."}
data: {"type": "answer_delta", "delta": "根据"}      # 若干条 token 增量
data: {"type": "answer_delta", "delta": "申通..."}
data: {"type": "sources", "sources": [...], "rounds": 1}   # multi_hop 时有 rounds
data: {"type": "done", "trace_id": "...", "intent": "rag_qa"}
data: [DONE]
```

超时降级插入 `{"type":"answer_delta","delta":"友好提示"}` 且 intent=degraded；出错插入 `{"type":"error","error":"..."}`。

统一错误响应：`{"error": {"code": "...", "message": "...", "trace_id": "..."}}`

## 配置

全部走环境变量 / `.env`（前缀 `SOP_QA_`），示例见 `.env.example`。关键项：

- `SOP_QA_LLM_MODEL` / `SOP_QA_EMBED_MODEL`：Ollama 模型（默认 qwen2.5:7b / bge-m3）
- `SOP_QA_LLM_PROVIDER` / `SOP_QA_INTENT_MODEL`：`ollama`|`openai`（OpenAI 兼容协议）切换；
  可配置意图识别专用小模型（分意图路由，省成本）
- `SOP_QA_RETRIEVAL_MODE`：`hybrid`（向量+BM25 RRF 融合，默认）/ `vector`（纯向量）
- `SOP_QA_SESSION_TTL_S` / `SOP_QA_SESSION_MAX_TURNS`：会话记忆过期时间 / 最大轮数
- `SOP_QA_CACHE_ENABLED` / `SOP_QA_CACHE_TTL_S`：无会话答案缓存开关 / 过期秒数
- `SOP_QA_RATE_LIMIT_ENABLED` / `SOP_QA_RATE_LIMIT_PER_MIN` / `SOP_QA_RATE_QUOTA_DAILY`：按 Key 限流/配额
- `SOP_QA_MAX_CONCURRENCY` / `SOP_QA_CHAT_TIMEOUT_S`：限流并发数 / 超时秒数
- `SOP_QA_MULTI_HOP_MAX_ROUNDS` / `SOP_QA_MULTI_HOP_SIMILARITY_THRESHOLD`：多轮检索参数
- `SOP_QA_EVAL_JUDGE` / `SOP_QA_EVAL_CASES_PATH`：LLM-as-Judge 开关 / 评测集路径
- `SOP_QA_API_KEY`：可选 API Key 鉴权

## 本地启动

前置：Ollama 运行中（`localhost:11434`），已拉取 `qwen2.5:7b` 和 `bge-m3`。

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # 按需修改

uvicorn app.main:app --port 8000        # 或 python3 -m app.main
# 文档: http://127.0.0.1:8000/docs
# 指标: http://127.0.0.1:8000/api/v1/metrics

# 命令行评测（不启动 HTTP）
python3 scripts/run_eval.py                    # 完整评测 (含 LLM 评分)
python3 scripts/run_eval.py --no-judge         # 只测检索命中率
python3 scripts/run_eval.py --scan-top-k "1,3,5"   # top_k 参数扫描
python3 scripts/run_eval.py --compare-mode     # hybrid vs vector 对比
```

启动时（lifespan）自动加载/构建 Chroma 索引（持久化在 `data/chroma/`，已有索引不重建）。

## Docker 部署 (P3-E)

```bash
docker compose up -d
# 首次会自动拉取 Ollama 模型 (qwen2.5:7b + bge-m3, 约 6GB), 完成后 app 才启动
curl http://localhost:8000/api/v1/health
```

- 组成：`ollama`（推理服务）+ `ollama-pull`（一次性拉模型）+ `app`（本服务）
- 数据卷持久化：Chroma 索引 / 评测历史 / 用户反馈（`app-data`）、Ollama 模型（`ollama-models`）
- 端口 `SOP_QA_APP_PORT` 可改；模型名可用 `SOP_QA_LLM_MODEL` / `SOP_QA_EMBED_MODEL` 覆盖
- 也可单独构建镜像：`docker build -t express-sop-qa .`
- 受限网络（容器内无法访问 PyPI）时用离线构建：
  `pip download -r requirements.txt -d wheels && docker build --build-arg ... `（在宿主机预下载
  wheels 后 `COPY wheels /wheels` + `pip install --no-index --find-links=/wheels -r requirements.txt`）

## 设计要点

- **安全**：配置全走 pydantic-settings + 环境变量；Pydantic 入参校验（长度/范围）；
  SSE 输出 `json.dumps` 序列化；异常捕获具体化并记日志；统一错误响应格式。
- **扩展性**：LLM/Embedding/向量库 Protocol 抽象 + 工厂注入；路由版本化；服务层与 HTTP 层解耦。
- **生产化**：`asyncio.Semaphore` 限流（超限排队）、`asyncio.wait_for`/`asyncio.timeout` + fallback 超时降级、
  结构化 JSON 日志 + contextvars trace_id 全链路串联。
- **P0.1 多轮记忆**：`session_id` → `SessionStore`（内存 LRU + TTL + 后台清扫），
  历史注入意图识别/直接回答/知识库生成节点，支持「那理赔要多久?」类追问。
- **P0.2 真流式**：`graph.astream_events` 消费 `on_chat_model_stream` token 事件，SSE 按 token 推送。
- **P0.4 LLM 改写**：multi_hop 首轮命中不足时由 LLM 结构化输出改写检索 query（替代硬编码拼接），
  `keep_original=true` 提前停止防死循环。
- **P1.5 混合检索**：向量 Top-N + BM25 Top-N → RRF 融合（无第三方依赖，字符 bigram 分词适配中文），
  精确术语/编号召回更稳。
- **P1.6 多文档知识库**：`/rag/docs` 增量导入/覆盖/删除，chunk 元数据带 doc_id/title，
  替代单文件全量重建。
- **P1.7 文档解析层**：`/rag/docs/upload` 直接收**原始文件**——PDF / Word / Excel /
  CSV / 文本，解析 → 清洗 → 切块入库。企业知识库的原始资料是 PDF 规范、Word 制度、
  Excel 台账，没有这一层，客户的资料根本进不来。
  - 中文编码自动识别（BOM → utf-8 → gb18030）：国内资料 GBK 极常见，按 utf-8 硬读
    会**静默产出乱码**，是最隐蔽的一类脏数据。
  - 文本清洗：去控制字符、合并被字距误判拆开的汉字（`理 赔 流 程` → `理赔流程`）、
    压缩空白。真实 PDF 抽出的文本几乎都需要过一遍。
  - 表格渲染成「表头: 值」的行文本，比 markdown 表更适合 RAG——切块后列名上下文不丢。
  - 扫描件（无文本层 PDF）**明确报错**提示需要 OCR，而不是静默产出空文档。
  - 体积 / 字符数上限兜底，中文文件名走哈希 doc_id 避免撞 id。
  - 各解析库懒加载，缺失时给出可执行的安装提示。
- **P2-A LLM-as-Judge**：评测集 `data/eval_cases.json`（期望关键词 + 要求要点）；
  答案质量按忠实性/完整性 LLM 结构化评分，HTTP 后台任务 + CLI 双入口。
- **P2-A′ 拒答测试**：评测集支持 `expect_refusal: true` 用例——知识库中确实没有答案的
  问题，系统应如实拒答而非编造。单独统计 `refusal_accuracy`，守住企业知识库的底线
  （宁可说「不知道」，不能一本正经地胡说）。该类用例不参与 `hit_rate`，因其属生成侧指标。
- **P2-B 评测回归**：评测历史 JSONL 落盘，`/eval/tasks` 返回与上次评测的 delta
  （hit_rate/拒答准确率/忠实性/完整性）；CLI 支持 `--scan-top-k` / `--compare-mode` 参数扫描。
- **P2-C 反馈闭环**：`/chat/feedback` 落盘 JSONL，负面反馈（≤2 分）结构化告警；
  `stats` 提供均分/负面率/低分问题（评测集扩充素材）。
- **P3-A 答案缓存**：无会话相同问题短 TTL 缓存，key 含知识库版本号（变更自动失效），
  命中率统计进 `/metrics`。
- **P3-B 监控指标**：`/metrics` 暴露 Prometheus 指标（QPS/耗时分位/意图分布/降级/
  缓存/评测/上游状态）。
- **P3-C 多模型路由**：`SOP_QA_LLM_PROVIDER` 切换 Ollama / OpenAI 兼容服务；
  `SOP_QA_INTENT_MODEL` 让意图识别走小模型省成本。
- **P3-D 按 Key 限流**：`X-API-Key`/IP 令牌桶 + 每日配额，429 + `Retry-After`，
  与全局信号量叠加使用。
- **P3-E Docker 部署**：`docker compose up -d` 一键起 app + Ollama（自动拉模型）。
- **P4 多租户隔离**：`SOP_QA_TENANT_MODE=true` 后 `X-API-Key` 映射租户
  （`data/tenants.json`），检索/文档按租户隔离；默认 SOP 进共享库（所有租户可检），
  租户私有文档互不可见。

## 测试

```bash
pip install -r requirements-dev.txt
.venv/bin/python -m pytest tests/ -q     # 无需 Ollama (Stub LLM/向量库驱动)
```

## 隐私自检（本仓库是公开的）

提交过的东西会**永久留在 git 历史里**——要清掉得改写历史 + force push。所以
「不要带私密信息进去」被做成了可执行的检查，而不是一句口头约定：

```bash
python3 scripts/check_privacy.py               # 检查全部已跟踪文件
python3 scripts/check_privacy.py --staged      # 只检查暂存区
python3 scripts/check_privacy.py --history     # 检查全部提交历史（message + diff + 历史路径）
python3 scripts/check_privacy.py --list-rules  # 打印当前生效的规则

# 装成 pre-commit 钩子（推荐）
ln -sf ../../scripts/check_privacy.py .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit
```

`tests/test_repo_hygiene.py` 会在每次 pytest 时跑同一套检查，CI 也能拦住。

检查范围：手机号 / 邮箱 / 身份证号 / 本地绝对路径 / 常见密钥样式 / 私密文件名。

**`--history` 为什么必要**：工作区干净 ≠ 历史干净。敏感信息可能在某次提交里
出现过、后来又被删掉——当前文件里查不到，但它仍然留在变更记录里，只有改写
历史才能清掉。**提交信息（commit message）同样要查**：改了文件内容却忘了改
message 是很常见的疏漏，而 message 一样会永久留在公开仓库里。

三个设计细节：

- **检查脚本里不含任何具体敏感词**。把姓名、手机号写进检查脚本，等于换个文件
  继续泄露。所以脚本只放「类别」正则，项目相关的自定义词通过环境变量
  `SOP_QA_PRIVACY_TERMS`（逗号分隔）或本地 `.privacy-terms`（每行一个，已
  gitignore）提供 —— 规则可以公开，「要防什么」留在本地。
- **命中报告不回显原文**。检查输出会进 CI 日志，若把命中的手机号原样打出来，
  这个检查本身就成了新的泄露渠道。报告只给「类别 + 文件:行号 + 长度」。
- **正则不匹配自身源码**，否则自检永远失败、没人会认真对待它；测试里也有
  一条用例专门守住这点。
