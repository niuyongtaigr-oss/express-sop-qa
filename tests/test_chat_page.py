"""对话页 (GET /) —— 页面纪律 + 事件契约

这个项目的产品形态就是问答, 但接口此前只能靠 curl / Swagger 试; 没有页面, 别人
无法在两分钟内看到它。页面本身**不持有任何数据**(无密钥、无语料), 内容全靠 JS 调
同源 API, 所以它可以不鉴权 —— 下面几条测试守住这个性质。

最要紧的一条是**事件契约**: 页面手工解析 SSE 并按 `type` 分支。服务端若新增/改名
一个事件类型而页面没跟上, 前端会**静默丢掉**那段内容 (不报错、只是少了东西)。
所以这里用真实图跑一遍 `ChatService.stream()`, 把实际发出的 type 与页面分支处理的
type 对齐。
"""

import re

from fastapi.testclient import TestClient

from app.main import create_app

APP = create_app()
PAGE = TestClient(APP).get("/").text


def _body() -> str:
    """页面去掉 HTML 注释后的内容 —— 注释里提到某个词不代表它是可执行代码"""
    return re.sub(r"<!--.*?-->", "", PAGE, flags=re.S)


# ── 挂载 ─────────────────────────────────────────────────

def test_chat_page_is_served_at_root():
    r = TestClient(APP).get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "快递 SOP 智能问答" in r.text


def test_admin_page_still_served():
    """加对话页不能把管理台挤掉"""
    r = TestClient(APP).get("/admin")
    assert r.status_code == 200
    assert "管理台" in r.text


def test_pages_are_not_in_openapi_schema():
    spec = TestClient(APP).get("/openapi.json").json()
    assert "/" not in spec["paths"]
    assert "/admin" not in spec["paths"]


# ── 页面纪律 (与管理台同规格) ─────────────────────────────

def test_page_has_no_external_resources():
    """零外部依赖: 受限网络 (无 CDN) 里也要能打开"""
    text = _body()
    assert "http://" not in text
    assert "https://" not in text
    assert "<script src=" not in text
    assert "<link " not in text


def test_page_never_injects_server_data_as_html():
    """回答与引用都来自服务端 —— 只能 textContent, 不能拼 innerHTML

    否则知识库里一篇标题带 <script> 的文档, 就能在访问者浏览器里执行脚本。
    """
    text = _body()
    for bad in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write"):
        assert bad not in text, f"页面用了 {bad}"
    assert "textContent" in text


def test_page_keeps_key_in_session_storage_only():
    text = _body()
    assert "sessionStorage" in text
    assert "localStorage" not in text
    assert "document.cookie" not in text


def test_page_contains_no_api_key_or_corpus():
    text = _body()
    assert "SOP_QA_API_KEY" not in text
    assert not re.search(r"[A-Za-z0-9]{32,}", text), "疑似硬编码的长密钥/摘要"


# ── 调用的接口必须真实存在 ───────────────────────────────

def test_every_api_path_called_by_the_page_exists():
    paths = set(TestClient(APP).get("/openapi.json").json()["paths"])
    called = set(re.findall(r'"(/api/v1/[A-Za-z0-9/_-]*)"', _body()))
    assert called, "没提取到任何接口路径, 正则该更新了"
    for lit in sorted(called):
        assert any(lit == p or p.startswith(lit.rstrip("/") + "/") for p in paths), (
            f"页面调用了不存在的接口: {lit}"
        )


def test_page_uses_the_streaming_endpoint():
    """对话页必须走 SSE 流式 —— 走非流式的话 CPU 推理 40+ 秒全程白屏"""
    assert '"/api/v1/chat/stream"' in _body()


def test_page_sends_a_session_id_and_the_api_key():
    text = _body()
    assert "session_id" in text
    assert "X-API-Key" in text
    # 会话 ID 必须不可猜测且 ≥16 位 (接口层有 min_length 校验)
    assert "randomUUID" in text


# ── 事件契约: 服务端发什么, 页面就得认什么 ────────────────

def _page_handled_types() -> set[str]:
    return set(re.findall(r'ev\.type === "(\w+)"', _body()))


def test_page_handles_every_event_type_the_server_emits():
    """用**真实图**跑一遍 stream(), 把实际事件类型与页面分支对齐

    页面是按 type 手工分支的: 服务端新增/改名一个事件而页面没跟上, 那段内容会被
    静默丢掉 —— 不报错、只是少了东西, 这比崩掉更难发现。
    """
    import asyncio

    from app.agents.graph import build_chat_graph
    from app.services.chat_service import ChatService
    from app.services.session_service import SessionStore
    from test_graph_nodes import StubLLM, StubRag, make_settings

    settings = make_settings()
    graph = build_chat_graph(StubRag(), StubLLM(intent="rag_qa"), settings)
    svc = ChatService(graph, SessionStore(), settings)

    async def collect() -> set[str]:
        seen: set[str] = set()
        async for ev in svc.stream("包裹破损怎么办"):
            seen.add(ev["type"])
        return seen

    emitted = asyncio.run(collect())
    assert emitted, "这条用例没收到任何事件, 契约就白测了"

    handled = _page_handled_types()
    missing = emitted - handled
    assert not missing, f"页面没有处理这些事件类型, 内容会被静默丢掉: {missing}"


def test_page_also_handles_error_and_done():
    """错误与收尾事件也要有分支 (happy path 测不到它们)"""
    handled = _page_handled_types()
    assert {"done", "error"} <= handled


def test_page_parses_sse_frames_conservatively():
    """手工解析 SSE 的两个要点, 缺一个就会丢帧或卡死

      · 必须按 `\\n\\n` 切帧并在读完一块后保留残余 (网络分块不保证按帧对齐)
      · 必须识别 `data: [DONE]` 结束标记
    """
    text = _body()
    assert r'"\n\n"' in text or "'\\n\\n'" in text
    assert "[DONE]" in text
