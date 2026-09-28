"""管理台页面 (P5)

`/rag/docs` 这些接口都能用, 但目标用户是企业管理员, 不会用 curl —— 缺一个能
"看见并操作知识库"的页面, 功能再全也交付不出去。

这个页面**自身不持有任何数据**: 没有密钥、没有语料, 内容全靠 JS 实时调同源 API
拉取。所以页面本身可以公开, 访问控制仍在 API 层。下面几条测试都是为了守住这个
性质 (页面里一旦混进数据或密钥, 它就从"静态壳"变成了泄漏面)。
"""

import re

from fastapi.testclient import TestClient

from app.main import create_app

HTML = create_app()


def _body() -> str:
    """页面去掉 HTML 注释后的内容 —— 注释里出现某个词不代表它是可执行代码"""
    text = TestClient(HTML).get("/admin").text
    return re.sub(r"<!--.*?-->", "", text, flags=re.S)


def test_admin_page_is_served():
    r = TestClient(HTML).get("/admin")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "管理台" in r.text


def test_root_redirects_to_admin():
    r = TestClient(HTML).get("/", follow_redirects=False)
    assert r.status_code in (307, 308)
    assert r.headers["location"] == "/admin"


def test_admin_is_not_in_openapi_schema():
    """它是页面不是接口, 不该出现在 /docs 的接口清单里"""
    spec = TestClient(HTML).get("/openapi.json").json()
    assert "/admin" not in spec["paths"]
    assert "/" not in spec["paths"]


# ── 页面不得携带数据/密钥 ─────────────────────────────────

def test_page_has_no_external_resources():
    """零外部依赖: 受限网络 (无 CDN) 里也必须能打开"""
    r = TestClient(HTML).get("/admin")
    assert "http://" not in r.text.replace("http://www.w3.org", "")
    assert "https://" not in r.text
    assert "<script src=" not in r.text
    assert "<link " not in r.text


def test_page_contains_no_api_key_or_corpus():
    """页面是静态壳 —— 不含任何密钥, 也不含任何语料内容"""
    r = TestClient(HTML).get("/admin")
    text = r.text
    assert "SOP_QA_API_KEY" not in text
    assert not re.search(r"[A-Za-z0-9]{32,}", text), "疑似硬编码的长密钥/摘要"
    # 语料样例不该出现在页面里
    for word in ("破损", "理赔", "第二十八条"):
        assert word not in text


def test_key_is_kept_in_session_storage_only():
    """密钥存 sessionStorage (关标签页即消失), 不写 localStorage、不写 Cookie"""
    text = _body()
    assert "sessionStorage" in text
    assert "localStorage" not in text
    assert "document.cookie" not in text


# ── 渲染必须防 XSS ───────────────────────────────────────

def test_page_does_not_inject_server_data_as_html():
    """文档标题来自服务端 —— 必须用 textContent 渲染, 不能拼 innerHTML

    否则一篇标题里带 <script> 的文档就能在管理员浏览器里执行脚本。
    """
    text = _body()
    # 允许的 innerHTML 用法是……没有
    assert "innerHTML" not in text
    assert "outerHTML" not in text
    assert "insertAdjacentHTML" not in text
    assert "document.write" not in text
    # 服务端数据一律走 textContent
    assert "textContent" in text


# ── 页面里调的每个接口都必须真实存在 ─────────────────────
# 页面是纯 JS, 拼错一个 URL 不会在启动时报错 —— 只会有人在点按钮时看到 404。
# 用 OpenAPI 清单把这类拼写错误挡在提交前。

def test_every_api_path_called_by_the_page_exists():
    page = _body()
    paths = set(TestClient(HTML).get("/openapi.json").json()["paths"])

    called = set(re.findall(r'"(/api/v1/[A-Za-z0-9/_-]*)"', page))
    assert called, "没提取到任何接口路径, 正则该更新了"

    for lit in sorted(called):
        assert any(lit == p or p.startswith(lit.rstrip("/") + "/") for p in paths), (
            f"页面调用了不存在的接口: {lit}"
        )


def test_page_calls_the_documented_endpoints():
    """反向确认: 关键能力都真的调了 (防止"页面看起来能用但其实没接上")"""
    page = _body()
    for path in ("/api/v1/rag/docs", "/api/v1/rag/docs/upload",
                 "/api/v1/rag/docs/formats", "/api/v1/rag/ingest",
                 "/api/v1/chat/feedback/stats"):
        assert f'"{path}"' in page, f"页面没有调用 {path}"


def test_in_progress_messages_are_visible():
    """不带 kind 的提示 (如"上传解析中…") 必须显示得出来

    原先写的是 `el.className = kind || ""` —— className="" 会命中外层 #msg 的
    display:none, 于是"重建/上传进行中"这类提示**根本不显示**: 用户点完按钮只
    看到界面没反应, 会以为坏了, 然后重复点击。
    """
    text = _body()
    assert re.search(r"#msg\.info\s*\{[^}]*display:\s*block", text), "缺少可见的 info 样式"
    assert re.search(r'el\.className\s*=\s*kind\s*\|\|\s*"info"', text), \
        'say() 的默认状态必须是可见的 "info", 不能是 ""'
