"""P1.5 混合检索 — BM25 索引单元测试"""

from app.infrastructure.bm25 import BM25Index, tokenize


def test_tokenize_chinese_bigram():
    toks = tokenize("包裹破损赔偿500元")
    assert "包裹" in toks
    assert "破损" in toks
    assert "赔偿" in toks
    # 标点被过滤 (bigram 滑窗: 破损/损遗/遗失)
    toks2 = tokenize("破损,遗失。")
    assert set(toks2) == {"破损", "损遗", "遗失"}


def test_bm25_ranks_keyword_match_first():
    corpus = [
        "包裹破损赔偿500元罚款",
        "包裹遗失连续3天未扫描更新",
        "今天是晴天适合出门散步",
    ]
    idx = BM25Index(corpus)
    hits = idx.top_k("500元罚款怎么处理", k=3)
    assert hits, "应返回非空结果"
    top_idx, top_score = hits[0]
    assert top_idx == 0 and top_score > 0
    assert corpus[top_idx] == corpus[0]


def test_bm25_exact_term_beats_semantic_irrelevant():
    corpus = [
        "理赔流程需要提交照片和运单号",
        "今天天气不错",
    ]
    idx = BM25Index(corpus)
    hits = idx.top_k("理赔流程", k=2)
    assert hits[0][0] == 0


def test_bm25_empty_corpus_and_empty_query():
    assert BM25Index([]).top_k("x", 3) == []
    idx = BM25Index(["一句话"])
    assert idx.top_k("", 3) == []
    assert idx.top_k("、、", 3) == []  # 全标点 → 无 token


def test_bm25_keeps_metadatas_aligned():
    texts = ["包裹破损赔偿500元"]
    metas = [{"doc_id": "sop", "title": "默认文档"}]
    idx = BM25Index(texts, metas)
    assert idx.metadatas[0]["doc_id"] == "sop"
