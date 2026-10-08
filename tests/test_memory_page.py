"""长期记忆页 (GET /memory) —— 页面纪律

记忆里装的是**个人信息**, 所以这个页面的纪律比管理台还严一档:

  · 只有 API 的话, "系统记住了我什么"只能靠 curl 看 —— 看不到就等于没有
  · 合规要求"能写就必须能删", 删除入口得是人能点到的东西, 不是文档里一行 curl
  · 记忆正文来自用户自己说过的话, 里面完全可能出现 `<script>` —— 一旦拼 HTML,
    下次打开这个页面就会执行它

下面几条测试守住这些性质 (与对话页同一套规格)。
"""

import re

from fastapi.testclient import TestClient

from app.main import create_app

APP = create_app()
PAGE = TestClient(APP).get("/memory").text


def _body() -> str:
    """去掉 HTML 注释后的内容 —— 注释里提到某个词不代表它是可执行代码"""
    return re.sub(r"<!--.*?-->", "", PAGE, flags=re.S)


# ── 挂载 ─────────────────────────────────────────────────

def test_memory_page_is_served():
    r = TestClient(APP).get("/memory")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "我的长期记忆" in r.text


def test_other_pages_still_served():
    """加页面不能把已有页面挤掉"""
    assert "快递 SOP 智能问答" in TestClient(APP).get("/").text
    assert "管理台" in TestClient(APP).get("/admin").text


def test_memory_page_is_not_in_openapi_schema():
    spec = TestClient(APP).get("/openapi.json").json()
    assert "/memory" not in spec["paths"], "页面不该出现在接口清单里"


def test_memory_api_still_in_openapi():
    """页面藏起来了, 但它的接口必须还在接口清单里"""
    spec = TestClient(APP).get("/openapi.json").json()
    assert "/api/v1/memory" in spec["paths"]
    assert "/api/v1/memory/{memory_id}" in spec["paths"]


# ── 页面纪律 ─────────────────────────────────────────────

def test_page_has_no_external_resources():
    """零外部依赖: 受限网络(无 CDN)里也要能打开"""
    text = _body()
    assert "http://" not in text
    assert "https://" not in text
    assert "<script src=" not in text
    assert "<link " not in text


def test_page_never_builds_html_from_data():
    """**最要紧的一条**: 记忆正文来自用户说的话, 绝不拼 HTML

    用户在对话里说一句 `<img src=x onerror=...>`, 如果这个页面拼 innerHTML,
    下次打开就会执行它 —— 存储型 XSS 的教科书场景, 而且数据源就是用户自己。
    """
    text = _body()
    for bad in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write"):
        assert bad not in text, f"页面里出现了 {bad}, 存储型 XSS 的口子"


def test_page_has_a_textcontent_only_helper():
    text = _body()
    assert "textContent" in text
    assert re.search(r'function el\(tag, cls, text\)', text), "缺少统一的建元素助手"


def test_page_uses_no_persistent_storage():
    """密钥只进 sessionStorage(关标签页即消失), 不进 localStorage / Cookie"""
    text = _body()
    assert "sessionStorage" in text
    assert "localStorage" not in text
    assert "document.cookie" not in text


def test_page_does_not_embed_secrets():
    """页面里不能有**密钥的值** (请求头名、`type="password"` 这类是正常的)

    一开始我把这条写成"不许出现 password 字样", 结果被 `<input type="password">`
    误伤 —— 断言要盯住"值", 不是盯住词。
    """
    text = _body()
    assert "X-API-Key" in text                       # 只是请求头名
    assert 'value = apiKey()' in text                # 密钥值只从 sessionStorage 来
    assert not re.search(r'id="key"[^>]*value="[^"]+"', text), "密钥输入框被预填了值"
    assert not re.search(r"\b[A-Za-z0-9_-]{32,}\b", text), "页面里出现了疑似密钥的长串"


# ── 接口调用 ─────────────────────────────────────────────

def test_every_api_path_the_page_calls_exists():
    """页面里写死的接口路径都要在接口清单里存在

    这类错误在浏览器里表现为"点了没反应", 排查成本高; 在这里是一行断言。
    """
    spec = TestClient(APP).get("/openapi.json").json()["paths"]
    called = set(re.findall(r'"(/api/v1/[^"]+)"', _body()))
    assert called, "页面没有调用任何接口?"
    for path in called:
        normalized = re.sub(r'/" \+.*$', "", path)          # 去掉拼接部分
        assert normalized in spec or normalized.rstrip("/") in spec, \
            f"页面调用了不存在的接口: {normalized}"


def test_page_supports_both_delete_forms():
    """逐条删除与全部清空都要有 (合规: 用户要求忘掉就必须能删)"""
    text = _body()
    assert "DELETE" in text
    assert "/api/v1/memory/" in text, "缺少按 id 删除"
    assert '"/api/v1/memory"' in text, "缺少清空全部"


def test_page_surfaces_the_no_identity_error():
    """拿不到用户身份时服务端返回 400, 页面必须把它显示出来

    这个错误最容易被吞掉(显示成"还没有任何记忆"), 而它其实是配置问题 ——
    用户会以为功能正常只是没数据, 然后一直等。
    """
    text = _body()
    assert "e.message" in text or "detail" in text
    assert "msg(\"err\"" in text or "msg('err'" in text


def test_page_sends_the_dev_user_id_header():
    """本地开发时页面必须能带上 X-User-Id

    否则这个页面在"未启用访问控制"的本地环境里完全用不了 —— 身份只认认证结果,
    而本地没有任何认证, 服务端就只会返回 400。
    """
    text = _body()
    assert "X-User-Id" in text
    assert 'sessionStorage.getItem("sop_qa_user_id")' in text
    assert "sop_qa_user_id" in text


def test_page_labels_the_asserted_identity_as_dev_only():
    """断言式身份必须在界面上写明"仅本地开发", 并说明启用访问控制后会被忽略

    这一条是给人看的: 不能让人以为"填个工号就能查到别人的记忆"。服务端确实会忽略
    它(见 test_user_identity.py), 但界面上也必须说清楚, 否则就是在误导使用者。
    """
    text = _body()
    assert "仅本地开发" in text
    assert "忽略" in text or "无效" in text


def test_chat_page_links_to_the_memory_page():
    """对话页要有入口 —— 藏起来的页面等于没有"""
    chat = TestClient(APP).get("/").text
    assert 'href="/memory"' in chat


def test_page_has_no_markdown_asterisks_in_visible_text():
    """可见文本里不能出现 Markdown 星号 —— HTML 不会把它渲染成粗体

    这是在截图里肉眼发现的: 页面上直接显示「**当前身份**」。注释里的星号无所谓
    (是给读代码的人看的), 但用户看得见的地方必须是真标签。
    """
    body = _body()
    body = re.sub(r"<script.*?</script>", "", body, flags=re.S)   # 去掉脚本(含 JS 注释)
    assert "**" not in body, "可见文本里出现了 Markdown 星号, 应改成 <strong>"
