"""检索与回答质量评测服务 — hit_rate + LLM-as-Judge + 拒答测试 (P2)

三个维度:
  1. 检索命中率 (hit_rate): 检索链路质量 — Top-K 结果是否包含期望关键词 (不调 LLM)
  2. 答案质量 (LLM-as-Judge): 生成链路质量 — 对每个用例实际生成回答, 由评审 LLM
     按两个维度结构化打分 (0-1):
       - faithfulness  忠实性: 回答是否都能从参考文档中找到依据 (未编造)
       - completeness  完整性: 回答是否覆盖 required_points 要求要点
  3. 拒答准确率 (refusal_accuracy): 知识库中确实没有答案的问题, 系统应如实拒答
     而非编造。企业知识库最怕「一本正经地胡说」, 故单独设一类用例守住这条底线。

用例分两类 (见 data/eval_cases.json 与 _BUILTIN_CASES):
  - 检索用例: 有 expect_keywords —— 参与 hit_rate, judge 开启时额外打分
  - 拒答用例: 有 expect_refusal=True —— 不参与 hit_rate (那是检索指标, 检索器
    对任何问题都会返回 Top-K, 无法据此判定), 判定需真实生成回答, 故仅在
    judge 开启时评估

评测集: data/eval_cases.json (结构化), 文件不存在时回退内置用例。
评测较慢 (每用例 = 1 次生成 + 1 次评审), HTTP 走后台任务; CLI 可 --no-judge。

🏭 Java 对标: 离线批量质检 Job + 人工抽检
"""

import json
import logging
import threading
import time
from pathlib import Path

from pydantic import BaseModel, Field

from app.config import Settings
from app.core.metrics import (
    EVAL_COMPLETENESS,
    EVAL_FAITHFULNESS,
    EVAL_HIT_RATE,
    EVAL_REFUSAL_ACCURACY,
)
from app.infrastructure.llm import LLMClient
from app.services.rag_service import RagService

logger = logging.getLogger(__name__)


class EvalJudgement(BaseModel):
    """LLM-as-Judge 结构化输出: 两个质量维度 0-1 分 + 理由"""

    faithfulness: float = Field(ge=0, le=1, description="忠实性: 回答有文档依据的程度")
    completeness: float = Field(ge=0, le=1, description="完整性: 覆盖要求要点的程度")
    reason: str = Field(default="", description="评分理由 (简洁)")


# 内置评测集 (data/eval_cases.json 不存在时的回退)
# ⚠️ 必须与 data/eval_cases.json 保持一致 —— tests 里有一条用例专门守住这点,
#    否则回退集会悄悄与真实评测集漂移 (回退时指标口径就变了)。
_BUILTIN_CASES: list[dict] = [
    {"id": "complaint-sla",
     "question": "快递企业接到用户投诉后，最迟多久要处理并告知用户？",
     "expect_keywords": ["7日"],
     "required_points": ["自接到投诉之日起7日内予以处理", "并告知用户"]},
    {"id": "site-filing",
     "question": "开办快递末端网点，需要在多少天内向邮政管理部门备案？",
     "expect_keywords": ["20日"],
     "required_points": ["自开办之日起20日内", "向所在地邮政管理部门备案"]},
    {"id": "stop-notice",
     "question": "快递企业停止经营，应当提前多久向社会公告？",
     "expect_keywords": ["10日"],
     "required_points": ["提前10日向社会公告", "书面告知邮政管理部门",
                         "交回快递业务经营许可证"]},
    {"id": "user-info-penalty",
     "question": "快递企业出售、泄露用户信息会面临什么处罚？",
     "expect_keywords": ["没收违法所得"],
     "required_points": ["责令改正", "没收违法所得", "处1万元以上5万元以下罚款",
                         "情节严重的处5万元以上10万元以下罚款",
                         "可责令停业整顿直至吊销经营许可证"]},
    {"id": "uninsured-loss",
     "question": "未保价的快件发生丢失，赔偿责任怎么确定？",
     "expect_keywords": ["未保价"],
     "required_points": ["依照民事法律的有关规定确定赔偿责任"]},
    {"id": "locker-consent",
     "question": "未经用户同意，快递员能把快件放到智能快件箱吗？",
     "expect_keywords": ["智能快件箱"],
     "required_points": ["未经用户同意不得擅自将快件投递到智能快件箱、"
                         "快递服务站等末端服务设施"]},
    {"id": "network-outage",
     "question": "快递服务网络阻断后，应当在多久内向邮政管理部门报告？",
     "expect_keywords": ["24小时"],
     "required_points": ["24小时内向邮政管理部门报告", "并向社会公告"]},
    {"id": "throwing-penalty",
     "question": "快递企业抛扔、踩踏快件会被怎么处罚？",
     "expect_keywords": ["抛扔", "踩踏"],
     "required_points": ["责令改正", "予以警告或者通报批评", "可以并处1万元以下罚款",
                         "情节严重的处1万元以上3万元以下罚款"]},
    {"id": "appeal-channel",
     "question": "用户对投诉处理结果不满意，还有什么救济途径？",
     "expect_keywords": ["申诉"],
     "required_points": ["可以提出快递服务质量申诉", "邮政管理部门对申诉实施调解"]},
    {"id": "green-packaging",
     "question": "对快递包装有哪些绿色环保要求？",
     "expect_keywords": ["过度包装"],
     "required_points": ["推进快递包装标准化、循环化、减量化、无害化",
                         "避免过度包装"]},
    # 拒答用例: 语料中没有冷链温度规定, 期望系统如实拒答而非编造
    {"id": "cold-chain",
     "question": "生鲜快递的冷链运输温度标准是多少度？",
     "expect_refusal": True},
]

# 拒答判定标记: 回答中出现任一即视为「如实拒答」(与 rag_service 的 system prompt 对齐)
_REFUSAL_MARKERS: tuple[str, ...] = (
    "无法回答", "没有相关", "不包含", "未提及", "无法确定", "无相关",
)

# 评审 LLM 的 system prompt
_JUDGE_SYSTEM = (
    "你是快递客服答案质量评审员。根据提供的「参考文档」评判助手回答的质量, "
    "从两个维度各打 0-1 分:\n"
    "1. faithfulness 忠实性: 回答内容是否都能从参考文档中找到依据, 没有依据的"
    "   内容 (编造/臆测) 越多分越低\n"
    "2. completeness 完整性: 回答是否覆盖了「要求要点」中的全部要点, 遗漏越多分越低\n"
    "评分要严格: 只有明确出现在文档/要点里的才算达标。"
)


class EvalHistory:
    """评测历史 — JSONL 持久化 (回归对比: 与最近一次同配置评测做 diff)

    线程安全; 只保留最近 max_entries 条; 进程重启不丢 (落盘)。
    """

    def __init__(self, path: Path, max_entries: int = 50):
        self._path = path
        self._max = max_entries
        self._lock = threading.Lock()
        self._entries: list[dict] = self._load()

    def _load(self) -> list[dict]:
        if not self._path.exists():
            return []
        entries: list[dict] = []
        try:
            lines = self._path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        for line in lines[-self._max * 2:]:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return entries[-self._max:]

    def append(self, entry: dict) -> None:
        with self._lock:
            self._entries.append(entry)
            self._entries = self._entries[-self._max:]
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def latest_before(self, ts: float, judge: bool | None = None) -> dict | None:
        """最近一次 ts 之前的评测摘要 (可限定 judge 开关一致)"""
        with self._lock:
            for e in reversed(self._entries):
                if e.get("ts", 0) < ts and (judge is None or e.get("judge") == judge):
                    return e
        return None

    def count(self) -> int:
        with self._lock:
            return len(self._entries)


class EvalService:
    """检索命中率 + 答案质量评测"""

    def __init__(
        self,
        rag_service: RagService,
        llm: LLMClient,
        settings: Settings,
        history_path: Path | None = None,
        record_history: bool = True,
    ):
        self._rag = rag_service
        self._llm = llm
        self._settings = settings
        self._cases = self._load_cases()
        self._history = EvalHistory(history_path or settings.eval_history_file)
        self._record_history = record_history

    # ── 评测集 ───────────────────────────────────────────
    def _load_cases(self) -> list[dict]:
        candidate = self._settings.eval_cases_file
        if candidate.exists():
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
                if isinstance(data, list) and data:
                    logger.info("评测集加载: %s (%d 用例)", candidate, len(data))
                    return data
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("评测集解析失败, 回退内置: %s", e)
        logger.info("评测集不存在, 使用内置 %d 用例", len(_BUILTIN_CASES))
        return _BUILTIN_CASES

    # ── 评测执行 ─────────────────────────────────────────
    def run(self, top_k: int = 3, judge: bool | None = None) -> dict:
        """跑一轮评测。

        judge=None 时按配置 (settings.eval_judge) 决定是否启用 LLM 评分。

        用例分两类, 分开统计 (见模块 docstring):
          - 检索用例 (expect_keywords): 参与 hit_rate, judge 开启时额外打分
          - 拒答用例 (expect_refusal): 不参与 hit_rate, 仅在 judge 开启时评估

        返回 {hit_rate, refusal_accuracy, faithfulness_avg, completeness_avg,
              top_k, judge, config, details}
        """
        use_judge = self._settings.eval_judge if judge is None else judge
        now = time.time()
        # 回归对比基准: 本次之前最近一次同 judge 设置的评测
        baseline = self._history.latest_before(now, judge=use_judge)
        details = []
        hits = 0
        answerable = 0       # 参与 hit_rate 的检索用例数
        judged = 0
        f_sum = c_sum = 0.0
        refusal_checked = 0  # 实际评估的拒答用例数
        refusal_passed = 0
        for case in self._cases:
            question = case["question"]
            chunks = self._rag.retrieve(question, top_k=top_k)
            top_sim = round(chunks[0]["similarity"], 4) if chunks else 0.0

            # ── 拒答用例: 知识库无答案, 期望系统如实拒答而非编造 ──
            if case.get("expect_refusal"):
                detail = {
                    "case_id": case.get("id", ""),
                    "question": question,
                    "kind": "refusal",
                    "expect": "REFUSE",
                    "hit": None,  # 拒答用例不参与检索命中率
                    "top_similarity": top_sim,
                }
                if use_judge and chunks:
                    try:
                        answer = self._rag.generate(question, chunks)
                        refused = any(m in answer for m in _REFUSAL_MARKERS)
                        refusal_checked += 1
                        refusal_passed += refused
                        detail.update({"answer": answer, "refused": refused})
                    except Exception as e:  # 单用例失败不拖垮整轮
                        logger.exception("拒答用例评估失败 question=%s", question)
                        detail["reason"] = f"eval_error: {type(e).__name__}: {e}"
                details.append(detail)
                continue

            # ── 检索用例: Top-K 是否覆盖期望关键词 ──
            joined = "".join(c["content"] for c in chunks)
            hit = any(kw in joined for kw in case.get("expect_keywords", []))
            answerable += 1
            hits += hit
            detail = {
                "case_id": case.get("id", ""),
                "question": question,
                "kind": "retrieval",
                "expect": "+".join(case.get("expect_keywords", [])),
                "hit": hit,
                "top_similarity": top_sim,
            }
            if use_judge and chunks:
                try:
                    answer = self._rag.generate(question, chunks)
                    j = self._judge(question, chunks, answer, case.get("required_points", []))
                    detail.update({
                        "answer": answer,
                        "faithfulness": round(j.faithfulness, 4),
                        "completeness": round(j.completeness, 4),
                        "reason": j.reason,
                    })
                    judged += 1
                    f_sum += j.faithfulness
                    c_sum += j.completeness
                except Exception as e:  # 单用例评审失败不拖垮整轮
                    logger.exception("用例评审失败 question=%s", question)
                    detail["reason"] = f"judge_error: {type(e).__name__}: {e}"
            details.append(detail)

        n = len(self._cases)
        hit_rate = hits / answerable if answerable else 0.0
        refusal_accuracy = refusal_passed / refusal_checked if refusal_checked else None
        result = {
            "hit_rate": round(hit_rate, 4),
            "refusal_accuracy": (
                round(refusal_accuracy, 4) if refusal_accuracy is not None else None
            ),
            "refusal_checked": refusal_checked,
            "refusal_total": n - answerable,
            "faithfulness_avg": round(f_sum / judged, 4) if judged else None,
            "completeness_avg": round(c_sum / judged, 4) if judged else None,
            "judged_cases": judged,
            "top_k": top_k,
            "judge": use_judge,
            "config": {
                "retrieval_mode": self._settings.retrieval_mode,
                "chunk_size": self._settings.chunk_size,
                "kb_chunks": self._rag.indexed_chunks,
            },
            "compare": self._diff_vs(
                baseline, hit_rate, judged, f_sum, c_sum,
                refusal_accuracy, refusal_checked,
            ),
            "details": details,
        }
        if self._record_history:
            self._history.append({
                "ts": now,
                "hit_rate": result["hit_rate"],
                "refusal_accuracy": result["refusal_accuracy"],
                "faithfulness_avg": result["faithfulness_avg"],
                "completeness_avg": result["completeness_avg"],
                "judged_cases": judged,
                "top_k": top_k,
                "judge": use_judge,
                "config": result["config"],
            })
        # Prometheus: 刷新最近一轮评测指标
        EVAL_HIT_RATE.set(result["hit_rate"])
        if result["refusal_accuracy"] is not None:
            EVAL_REFUSAL_ACCURACY.set(result["refusal_accuracy"])
        if result["faithfulness_avg"] is not None:
            EVAL_FAITHFULNESS.set(result["faithfulness_avg"])
            EVAL_COMPLETENESS.set(result["completeness_avg"])
        logger.info(
            "eval_done hit_rate=%.4f refusal=%s (%d/%d) faithfulness=%s completeness=%s "
            "cases=%d judged=%d",
            hit_rate, result["refusal_accuracy"], refusal_passed, refusal_checked,
            result["faithfulness_avg"], result["completeness_avg"], n, judged,
        )
        return result

    # ── 内部: 回归对比 ───────────────────────────────────
    @staticmethod
    def _diff_vs(
        baseline: dict | None,
        hit_rate: float,
        judged: int,
        f_sum: float,
        c_sum: float,
        refusal_accuracy: float | None = None,
        refusal_checked: int = 0,
    ) -> dict | None:
        """与基准评测的差值 (delta), 无基准返回 None"""
        if not baseline:
            return None
        diff = {
            "vs_ts": baseline.get("ts"),
            "hit_rate_delta": round(hit_rate - baseline.get("hit_rate", 0.0), 4),
        }
        if refusal_checked and baseline.get("refusal_accuracy") is not None:
            diff["refusal_accuracy_delta"] = round(
                refusal_accuracy - baseline["refusal_accuracy"], 4
            )
        if judged:
            f_avg = round(f_sum / judged, 4)
            c_avg = round(c_sum / judged, 4)
            if baseline.get("faithfulness_avg") is not None:
                diff["faithfulness_delta"] = round(f_avg - baseline["faithfulness_avg"], 4)
            if baseline.get("completeness_avg") is not None:
                diff["completeness_delta"] = round(c_avg - baseline["completeness_avg"], 4)
        return diff

    # ── 内部: LLM 评审 ───────────────────────────────────
    def _judge(
        self,
        question: str,
        chunks: list[dict],
        answer: str,
        required_points: list[str],
    ) -> EvalJudgement:
        from langchain_core.messages import HumanMessage, SystemMessage

        refs = "\n".join(f"[{i + 1}] {c['content']}" for i, c in enumerate(chunks[:5]))
        points = "\n".join(f"- {p}" for p in required_points) or "(无明确要点)"
        human = (
            f"用户问题: {question}\n\n"
            f"参考文档:\n{refs}\n\n"
            f"要求要点:\n{points}\n\n"
            f"助手回答:\n{answer}\n\n"
            "请给出两个维度的评分与理由:"
        )
        return self._llm.structured_invoke(
            EvalJudgement,
            [SystemMessage(content=_JUDGE_SYSTEM), HumanMessage(content=human)],
        )
