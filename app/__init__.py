"""快递 SOP 智能问答系统 — 应用包

生产级分层架构:
  api/            HTTP 接口层 (FastAPI 路由, 版本化 /api/v1)
  services/       业务逻辑层 (不碰 HTTP 细节)
  agents/         LangGraph 编排 (意图识别 → 路由)
  infrastructure/ 外部系统适配层 (LLM / Embedding / 向量库, Protocol 抽象)
  core/           横切关注点 (日志/异常/中间件/安全)
  schemas/        Pydantic DTO
  config.py       集中配置 (pydantic-settings)
"""
