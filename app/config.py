"""集中配置管理 — pydantic-settings

所有可调参数收敛到本模块, 从环境变量 / .env 读取 (前缀 SOP_QA_),
其他模块一律通过 get_settings() 获取, 禁止模块级硬编码常量。

🏭 Java 对标: application.yml + @ConfigurationProperties
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录 (app/ 的上一级), 用于把相对路径配置解析成绝对路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """应用配置 — 环境变量前缀 SOP_QA_, 也支持 .env 文件"""

    model_config = SettingsConfigDict(
        env_prefix="SOP_QA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── 应用 ────────────────────────────────────────────
    log_level: str = "INFO"

    # ── Ollama / 模型 ────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:7b"
    embed_model: str = "bge-m3"

    # ── 多模型路由 (P3-C) ────────────────────────────────
    llm_provider: str = "ollama"        # ollama | openai (OpenAI 兼容协议)
    intent_model: str | None = None     # 意图识别专用模型 (None=与回答共用 llm_model)
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str | None = None   # OpenAI 兼容服务密钥 (vLLM 本地可省略)
    openai_model: str | None = None     # OpenAI provider 时的回答模型 (缺省=llm_model)

    # ── 知识库 ───────────────────────────────────────────
    sop_path: str = "data/sop.txt"        # SOP 文档 (相对项目根目录, 默认文档)
    sop_doc_id: str = "sop"               # 默认文档 ID (sop.txt 导入时使用)
    chroma_dir: str = "data/chroma"       # Chroma 持久化目录
    collection_name: str = "sop_knowledge"
    chunk_size: int = 200
    chunk_overlap: int = 40
    top_k: int = 3
    retrieval_mode: str = "hybrid"        # hybrid=向量+BM25 融合(RRF) / vector=纯向量

    # ── 多轮会话 (记忆) ──────────────────────────────────
    session_ttl_s: int = 1800             # 会话空闲过期时间 (秒)
    session_max_turns: int = 10           # 单会话保留的最大轮数 (超出丢最旧)
    session_max_sessions: int = 2000      # 内存会话上限 (LRU 淘汰)
    session_sweep_interval_s: int = 60    # 后台清理过期会话的间隔 (秒)

    # ── 编排 ─────────────────────────────────────────────
    multi_hop_max_rounds: int = 2               # 多轮检索最大轮数 (防死循环)
    multi_hop_similarity_threshold: float = 0.6  # 首轮命中即停止的相似度阈值

    # ── 评测 (P2) ────────────────────────────────────────
    eval_cases_path: str = "data/eval_cases.json"  # 结构化评测集 (不存在则用内置)
    eval_judge: bool = True                         # 是否启用 LLM-as-Judge 质量评分
    eval_history_path: str = "data/eval_history.jsonl"  # 评测历史 (回归对比用)
    feedback_path: str = "data/feedback.jsonl"          # 用户反馈落盘 (P2-C)

    # ── 生产化 (限流/超时) ───────────────────────────────
    max_concurrency: int = 4        # 全局并发上限 (asyncio.Semaphore)
    chat_timeout_s: float = 60.0    # /chat 单次请求超时
    rag_timeout_s: float = 30.0     # /rag/query 单次请求超时

    # ── 答案缓存 (P3) ────────────────────────────────────
    cache_enabled: bool = True      # 无会话相同问题的短 TTL 答案缓存
    cache_ttl_s: float = 300.0      # 缓存过期秒数
    cache_max_entries: int = 512    # 缓存条目上限 (LRU)

    # ── 细粒度限流/配额 (P3-D) ───────────────────────────
    rate_limit_enabled: bool = False   # 按 Key/IP 限流总开关 (生产开启)
    rate_limit_per_min: int = 60       # 每 Key 每分钟请求上限
    rate_limit_burst: int = 20         # 令牌桶突发上限
    rate_quota_daily: int = 1000       # 每 Key 每日配额

    # ── 安全 (可选) ──────────────────────────────────────
    # 配置后业务接口必须携带 X-API-Key; 不配置则放行并日志告警 (仅本地开发)
    api_key: str | None = None

    # ── 派生路径 (相对路径 → 绝对路径) ────────────────────
    @property
    def sop_file(self) -> Path:
        p = Path(self.sop_path)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def chroma_path(self) -> Path:
        p = Path(self.chroma_dir)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def eval_cases_file(self) -> Path:
        p = Path(self.eval_cases_path)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def eval_history_file(self) -> Path:
        p = Path(self.eval_history_path)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def feedback_file(self) -> Path:
        p = Path(self.feedback_path)
        return p if p.is_absolute() else PROJECT_ROOT / p


@lru_cache
def get_settings() -> Settings:
    """配置缓存单例 — 进程内只解析一次环境变量"""
    return Settings()
