"""P2-A LLM-as-Judge 评测服务测试 (Stub 驱动, 无需 Ollama)

用例集通过 tmp_path 现场构造后显式传入, **不依赖 data/eval_cases.json 的具体
内容** —— 否则每次调整真实语料/评测集, 这里就会跟着红一片。
真实评测集与内置回退集的一致性由 test_eval_cases_file_matches_builtin 守住。
"""

import json

from app.config import Settings
from app.services.eval_service import (
    _BUILTIN_CASES,
    EvalJudgement,
    EvalService,
)


def make_settings(**overrides):
    base = dict(eval_judge=True, retrieval_mode="hybrid", chunk_size=200, top_k=3)
    base.update(overrides)
    return Settings(_env_file=None, **base)


def write_cases(tmp_path, cases: list[dict]) -> str:
    """把用例集写到临时文件, 返回路径 (供 make_settings(eval_cases_path=...))"""
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")
    return str(path)


# 两条检索用例 + 一条拒答用例的固定小集: 计数与断言都明确, 不受真实评测集影响
CASES_2_RETRIEVAL_1_REFUSAL = [
    {"id": "r-hit", "question": "破损怎么处理?",
     "expect_keywords": ["异常登记"], "required_points": ["异常登记"]},
    {"id": "r-miss", "question": "罚款多少?",
     "expect_keywords": ["500"], "required_points": ["500元罚款"]},
    {"id": "x-refuse", "question": "冷链温度标准?", "expect_refusal": True},
]
N_RETRIEVAL = 2
N_REFUSAL = 1
N_ALL = N_RETRIEVAL + N_REFUSAL



class StubRag:
    def __init__(
        self,
        chunk_text="包裹破损需异常登记并拍照留存",
        answer="包裹破损请立即异常登记并拍照留存。",
    ):
        self._chunk_text = chunk_text
        self._answer = answer
        self.retrieves: list = []
        self.generates: list = []

    def retrieve(self, query, top_k=None):
        self.retrieves.append(query)
        return [{
            "content": self._chunk_text,
            "metadata": {"doc_id": "sop", "title": "默认", "tags": "破损"},
            "distance": 0.1,
            "similarity": 0.9,
        }]

    def generate(self, query, chunks, history=None):
        self.generates.append(query)
        return self._answer

    @property
    def indexed_chunks(self):
        return 2


class JudgeLLM:
    """structured_invoke 返回固定评分"""

    def __init__(self, judgement=None):
        self._j = judgement or EvalJudgement(faithfulness=0.8, completeness=0.7, reason="有依据, 部分遗漏")

    def structured_invoke(self, schema, messages):
        assert schema is EvalJudgement
        return self._j

    def invoke(self, messages):
        return "stub"

    async def astream(self, messages):
        yield "s"


def test_hit_rate_and_judge_scores(tmp_path):
    rag, llm = StubRag(), JudgeLLM()
    settings = make_settings(eval_cases_path=write_cases(tmp_path, CASES_2_RETRIEVAL_1_REFUSAL))
    result = EvalService(rag, llm, settings, record_history=False).run(top_k=3)
    assert result["top_k"] == 3
    assert result["judge"] is True
    # 拒答用例不走 LLM 打分 (required_points 为空, 评分无意义), 只判「是否如实拒答」
    assert result["judged_cases"] == N_RETRIEVAL
    assert result["refusal_total"] == N_REFUSAL
    assert result["refusal_checked"] == N_REFUSAL
    assert result["faithfulness_avg"] == 0.8
    assert result["completeness_avg"] == 0.7
    assert result["hit_rate"] >= 0.0
    # 每条用例都有答案 (含拒答用例)
    assert all(d.get("answer") for d in result["details"])
    # 只有检索用例带 faithfulness 评分
    scored = [d for d in result["details"] if d["kind"] == "retrieval"]
    assert len(scored) == N_RETRIEVAL
    assert all(d["faithfulness"] == 0.8 for d in scored)
    # 默认 stub 回答不含拒答标记 → 拒答用例判为「未拒答」(即编造)
    assert result["refusal_accuracy"] == 0.0
    # 生成只走 generate, 不走 ask; 检索用例 + 拒答用例各生成一次
    assert len(rag.generates) == N_ALL
    # config 快照
    assert result["config"]["retrieval_mode"] == "hybrid"
    assert result["config"]["kb_chunks"] == 2


def test_no_judge_skips_generation(tmp_path):
    rag, llm = StubRag(), JudgeLLM()
    settings = make_settings(eval_cases_path=write_cases(tmp_path, CASES_2_RETRIEVAL_1_REFUSAL))
    result = EvalService(rag, llm, settings, record_history=False).run(top_k=3, judge=False)
    assert result["judge"] is False
    assert result["judged_cases"] == 0
    assert result["faithfulness_avg"] is None
    assert result["completeness_avg"] is None
    assert rag.generates == []  # 不生成答案
    assert all(d.get("answer") is None for d in result["details"])


def test_hit_rate_depends_on_keywords(tmp_path):
    # 检索片段含 "异常登记" 但不含 "500" → r-miss 用例不命中
    rag = StubRag(chunk_text="破损需立即异常登记并拍照留存")
    settings = make_settings(eval_cases_path=write_cases(tmp_path, CASES_2_RETRIEVAL_1_REFUSAL))
    result = EvalService(rag, JudgeLLM(), settings).run(top_k=3, judge=False)
    by_id = {d["case_id"]: d for d in result["details"]}
    assert by_id["r-hit"]["hit"] is True      # 含 "异常登记"
    assert by_id["r-miss"]["hit"] is False    # 不含 "500"
    assert result["hit_rate"] == 0.5          # 2 条检索用例命中 1 条


def test_judge_error_does_not_kill_run(tmp_path):
    class BrokenJudge(JudgeLLM):
        def structured_invoke(self, schema, messages):
            raise RuntimeError("judge down")

    rag, llm = StubRag(), BrokenJudge()
    settings = make_settings(eval_cases_path=write_cases(tmp_path, CASES_2_RETRIEVAL_1_REFUSAL))
    result = EvalService(rag, llm, settings, record_history=False).run(top_k=3)
    assert result["judged_cases"] == 0
    assert result["faithfulness_avg"] is None
    assert any("judge_error" in (d.get("reason") or "") for d in result["details"])
    assert result["hit_rate"] >= 0.0  # 检索部分不受影响


def test_cases_from_file(tmp_path):
    cases = [{
        "id": "custom-1",
        "question": "自定义问题?",
        "expect_keywords": ["破损"],
        "required_points": ["拍照"],
    }]
    rag, llm = StubRag(), JudgeLLM()
    svc = EvalService(rag, llm, make_settings(eval_cases_path=write_cases(tmp_path, cases)),
                      record_history=False)
    result = svc.run(top_k=3, judge=False)
    assert len(result["details"]) == 1
    assert result["details"][0]["case_id"] == "custom-1"


def test_cases_fallback_when_file_missing(tmp_path):
    missing = tmp_path / "nope.json"
    rag, llm = StubRag(), JudgeLLM()
    svc = EvalService(rag, llm, make_settings(eval_cases_path=str(missing)), record_history=False)
    result = svc.run(top_k=3, judge=False)
    assert len(result["details"]) == len(_BUILTIN_CASES)


def test_eval_cases_file_matches_builtin():
    """data/eval_cases.json 必须与 _BUILTIN_CASES 语义一致。

    两者是同一份评测集的两个副本 (文件是主用, 内置是缺文件时的回退)。
    一旦漂移, 「回退时」的指标口径就和正常跑不一致了, 而且不会有人察觉。

    比较时忽略 `_` 开头的键 (如 _comment) —— 那是文档性说明, 不参与判分。
    """
    from app.config import Settings as S

    def semantic(cases: list[dict]) -> list[dict]:
        return [
            {k: v for k, v in case.items() if not k.startswith("_")}
            for case in cases
        ]

    path = S(_env_file=None).eval_cases_file
    file_cases = json.loads(path.read_text(encoding="utf-8"))
    assert semantic(file_cases) == semantic(_BUILTIN_CASES)


# ── 拒答用例 (expect_refusal) ────────────────────────────
# 背景: 知识库中确实没有答案的问题, 系统应如实拒答而非编造。
# 这是企业知识库的底线指标 —— 宁可说「不知道」, 不能一本正经地胡说。


def test_refusal_passes_when_model_refuses(tmp_path):
    """如实拒答 → refusal_accuracy = 1.0"""
    rag = StubRag(answer="根据现有知识库无法回答此问题。")
    settings = make_settings(eval_cases_path=write_cases(tmp_path, CASES_2_RETRIEVAL_1_REFUSAL))
    result = EvalService(rag, JudgeLLM(), settings, record_history=False).run(top_k=3)
    assert result["refusal_accuracy"] == 1.0
    refusals = [d for d in result["details"] if d["kind"] == "refusal"]
    assert len(refusals) == N_REFUSAL
    assert refusals[0]["refused"] is True


def test_refusal_fails_when_model_hallucinates(tmp_path):
    """知识库无答案却硬答 → refusal_accuracy = 0.0"""
    rag = StubRag(answer="冷链运输温度应保持在 0 到 4 摄氏度。")
    settings = make_settings(eval_cases_path=write_cases(tmp_path, CASES_2_RETRIEVAL_1_REFUSAL))
    result = EvalService(rag, JudgeLLM(), settings, record_history=False).run(top_k=3)
    assert result["refusal_accuracy"] == 0.0
    refusals = [d for d in result["details"] if d["kind"] == "refusal"]
    assert refusals[0]["refused"] is False


def test_refusal_excluded_from_hit_rate(tmp_path):
    """拒答用例不参与 hit_rate: 分母只算检索用例, 不会被无解的用例拉低"""
    # 检索片段覆盖全部 2 条检索用例的关键词
    rag = StubRag(chunk_text="破损需异常登记，罚款 500 元")
    settings = make_settings(eval_cases_path=write_cases(tmp_path, CASES_2_RETRIEVAL_1_REFUSAL))
    result = EvalService(rag, JudgeLLM(), settings, record_history=False).run(
        top_k=3, judge=False
    )
    assert result["hit_rate"] == 1.0           # 若拒答用例计入分母, 这里会低于 1.0
    refusals = [d for d in result["details"] if d["kind"] == "refusal"]
    assert refusals[0]["hit"] is None          # 拒答用例不出 hit 结论


def test_refusal_skipped_without_judge(tmp_path):
    """--no-judge 不生成回答 → 拒答用例无从判定, 指标为 None 而非算作失败"""
    rag = StubRag()
    settings = make_settings(eval_cases_path=write_cases(tmp_path, CASES_2_RETRIEVAL_1_REFUSAL))
    result = EvalService(rag, JudgeLLM(), settings, record_history=False).run(
        top_k=3, judge=False
    )
    assert result["refusal_accuracy"] is None
    assert result["refusal_checked"] == 0
    assert result["refusal_total"] == N_REFUSAL
    assert rag.generates == []


# ── API DTO 兼容性 (回归防护) ────────────────────────────


def test_api_dto_accepts_refusal_detail(tmp_path):
    """回归: 拒答用例的 hit=None 必须能通过 EvalTaskResponse 校验。

    端点实现是 EvalTaskResponse(task_id=..., **tasks[task_id]), 而
    EvalCaseDetail.hit 原为必填 bool —— 引入拒答用例后若不同步放宽为
    bool | None, 每次带 judge 的评测轮询都会 500。
    """
    from app.schemas.eval import EvalTaskResponse

    rag = StubRag(answer="根据现有知识库无法回答此问题。")
    settings = make_settings(eval_cases_path=write_cases(tmp_path, CASES_2_RETRIEVAL_1_REFUSAL))
    result = EvalService(rag, JudgeLLM(), settings, record_history=False).run(top_k=3)

    resp = EvalTaskResponse(task_id="abc12345", status="done", **result)

    assert resp.refusal_accuracy == 1.0
    assert resp.refusal_total == N_REFUSAL
    assert resp.details is not None
    refusal = [d for d in resp.details if d.kind == "refusal"][0]
    assert refusal.hit is None
    assert refusal.refused is True
    retrieval = [d for d in resp.details if d.kind == "retrieval"][0]
    assert retrieval.hit is not None
    assert retrieval.refused is None


# ── 临时索引必须连 manifest 一起搬走 ─────────────────────
# 踩过的坑 (`scripts/run_eval.py --compare-mode`): 只换了 store 的 persist_dir, 而
# manifest 的位置是从 settings.chroma_dir 推出来的 —— 于是 force 重建把**新语料的
# 指纹**写进了正式索引的 manifest, chunk 却落在临时目录。正式索引还留着旧内容,
# 清单却说"已是最新", 下次 ingest() 判定无变化 → **正式索引永久静默陈旧**。
# 这正是清单机制本来要防的那类失败。

def test_temp_index_settings_moves_manifest_together(tmp_path):
    """临时索引的 manifest 必须落在临时目录里, 且不动正式索引的清单"""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from run_eval import _temp_index_settings

    real = Settings(_env_file=None, chroma_dir=str(tmp_path / "real"))
    real_manifest = real.manifest_path

    for rerank in (False, True):
        tmp_cfg = _temp_index_settings(real, "/tmp/sopqa-cmp-xyz", rerank)
        assert tmp_cfg.rerank_enabled is rerank
        # manifest 跟着临时目录走
        assert str(tmp_cfg.manifest_path).startswith("/tmp/sopqa-cmp-xyz")
        assert tmp_cfg.manifest_path != real_manifest
        # 正式配置没被改动
        assert real.manifest_path == real_manifest
        assert real.chroma_dir.endswith("real")
        assert real.rerank_enabled is False


def test_temp_index_context_keeps_manifest_with_the_index(tmp_path):
    """上下文管理器交出的配置, manifest 必须落在临时索引目录里

    这条守的是**调用点**: 只测 `_temp_index_settings` 的话, `_compare_mode` 里忘了
    把临时目录传进去 (原缺陷) 测试照样绿。改成上下文管理器后目录与配置由同一个
    对象交出, 那个错误在调用点已经写不出来了。
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from run_eval import _temp_index

    class _NoEmbed:
        def embed(self, texts):
            return [[1.0, 0.0] for _ in texts]

    real = Settings(_env_file=None, chroma_dir=str(tmp_path / "real"))
    real_manifest = real.manifest_path

    with _temp_index(real, _NoEmbed(), "vector", False) as (store, cfg):
        assert cfg.manifest_path != real_manifest
        assert str(cfg.manifest_path).startswith(cfg.chroma_dir)
        assert cfg.rerank_enabled is False
        assert store.count() == 0            # 临时索引是空的

    # 正式配置与正式清单全程未被改动
    assert real.chroma_dir.endswith("real")
    assert real.manifest_path == real_manifest
    assert not real_manifest.exists()


def test_temp_index_settings_without_dir_only_changes_rerank(tmp_path):
    """不传目录时只改 rerank, manifest 位置保持不动"""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from run_eval import _temp_index_settings

    real = Settings(_env_file=None, chroma_dir=str(tmp_path / "real"))
    cfg = _temp_index_settings(real, None, True)
    assert cfg.rerank_enabled is True
    assert cfg.manifest_path == real.manifest_path
