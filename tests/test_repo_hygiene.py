"""仓库隐私自检 (pytest / CI 入口)

本仓库是公开的, 提交过的东西会永久留在 git 历史里。这里复用
scripts/check_privacy.py 的规则, 让每次跑测试都顺手拦一遍 —— 详见该脚本的
模块 docstring (含「为什么检查脚本里没有具体敏感词」的设计说明)。
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.check_privacy import (  # noqa: E402  (需先补 sys.path)
    CONTENT_RULES,
    Hit,
    active_rules,
    commit_identities,
    format_identities,
    format_report,
    identity_content_hits,
    identity_hits,
    scan,
    scan_history,
    scan_staged,
    scan_tracked,
)


# ── 测试样本 ─────────────────────────────────────────────
# 样本必须按段拼接: 本文件里不能出现「连续 11 位数字」「完整邮箱」这类字面量,
# 否则它自己就会被 check_privacy 命中 —— 自检必须自洽, 不能给自己开后门。


def _sample(*parts: str) -> str:
    """把样本拼成目标形态; 源文件里只留下分段片段"""
    return "".join(parts)


_FAKE_PHONE = _sample("138", "0013", "8000")
_FAKE_ID = _sample("110101", "19900307", "1234")
_FAKE_EMAIL = _sample("someone", "@qq", ".com")
_FAKE_PATH = _sample("/home/", "someone", "/x")
_FAKE_TOKEN = _sample("sk-", "abcdefghij", "klmnopqrst", "uvwxyz012345")


# ── 仓库本体检查 ─────────────────────────────────────────

def test_tracked_files_carry_no_private_info():
    """已跟踪文件不得包含个人信息 / 私密文件名"""
    hits = scan_tracked()
    assert not hits, format_report(hits)


def test_staged_files_carry_no_private_info():
    """暂存区同样检查 (不能只靠本地 pre-commit 钩子)"""
    hits = scan_staged()
    assert not hits, format_report(hits)


def test_commit_history_carries_no_private_info():
    """提交历史 (message + diff + 历史路径名) 同样不得含个人信息。

    工作区干净 ≠ 历史干净: 改过文件内容却忘了改 commit message 是很常见的
    疏漏, 而 message 一样会永久留在公开仓库的变更记录里。
    """
    hits = scan_history()
    assert not hits, format_report(hits)


def test_custom_terms_rule_tracks_configuration(monkeypatch, tmp_path):
    """配置了本地敏感词就必须出现该规则; 没配置就不能凭空出现。

    不能直接断言「当前环境没有自定义词」—— 本地开发时 .privacy-terms 往往是
    存在的, 那种写法会让测试随环境时红时绿。这里把 PROJECT_ROOT 指到空目录,
    测的是机制本身。
    """
    import scripts.check_privacy as cp

    monkeypatch.setattr(cp, "PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("SOP_QA_PRIVACY_TERMS", raising=False)
    assert "自定义敏感词" not in [name for name, _ in cp.active_rules()]

    (tmp_path / ".privacy-terms").write_text("某内部代号\n", encoding="utf-8")
    assert "自定义敏感词" in [name for name, _ in cp.active_rules()]


# ── 报告本身不得泄露 ─────────────────────────────────────

def test_report_never_echoes_matched_text():
    """命中报告只给位置, 不回显原文。

    检查输出会进 CI 日志 —— 如果它把命中的手机号原样打出来, 这个检查本身
    就成了新的泄露渠道。
    """
    fake_phone = _FAKE_PHONE
    report = format_report([Hit(path="f.txt", line=7, rule="手机号", length=len(fake_phone))])
    assert "f.txt:7" in report          # 位置: 可定位
    assert "手机号" in report            # 类别: 可判断
    assert fake_phone not in report     # 原文: 绝不出现


# ── 检出能力 (防止规则被改坏后静默失效) ──────────────────

def _scan_text(tmp_path: Path, name: str, text: str) -> list[Hit]:
    (tmp_path / name).write_text(text, encoding="utf-8")
    return scan([name], root=tmp_path)


def test_detects_phone_number(tmp_path):
    hits = _scan_text(tmp_path, "a.txt", f"联系电话 {_FAKE_PHONE}")
    assert [h.rule for h in hits] == ["手机号"]


def test_detects_email_but_allows_example_domain(tmp_path):
    assert _scan_text(tmp_path, "a.txt", f"联系 {_FAKE_EMAIL}")[0].rule == "邮箱"
    assert _scan_text(tmp_path, "b.txt", "联系 you@example.com") == []


def test_detects_id_card_and_absolute_path(tmp_path):
    rules = {
        h.rule
        for h in _scan_text(tmp_path, "a.txt", f"证件 {_FAKE_ID}\n路径 {_FAKE_PATH}")
    }
    assert rules == {"身份证号", "本地绝对路径"}


def test_detects_secret_like_token(tmp_path):
    hits = _scan_text(tmp_path, "a.txt", f"key = {_FAKE_TOKEN}")
    assert hits and hits[0].rule == "密钥样式"


def test_detects_private_filename(tmp_path):
    hits = scan(["\u7b80\u5386-2026.pdf"], root=tmp_path)  # 个人材料类文件名
    assert hits and hits[0].line == 0
    assert "私密文件名" in hits[0].rule


def test_skips_binary_files(tmp_path):
    (tmp_path / "blob.bin").write_bytes(b"\x00\x01\x02 " + _FAKE_PHONE.encode())
    assert scan(["blob.bin"], root=tmp_path) == []


def test_clean_text_passes(tmp_path):
    assert _scan_text(tmp_path, "a.txt", "破损件需立即异常登记并拍照留存。") == []


def test_rules_do_not_self_match():
    """规则源码不得命中自己 —— 否则自检永远失败, 没人会认真对待它"""
    import re

    for _name, pattern in CONTENT_RULES:
        assert re.search(pattern.pattern, pattern.pattern) is None, (
            f"规则自命中: {pattern.pattern}"
        )


# ── 提交身份字段 ─────────────────────────────────────────
# 这个字段和文件内容一样会公开显示在 GitHub 的每一次提交上, 却最容易没人看 ——
# 用工作邮箱提交作品集仓库, 就是把任职单位域名挂在公开页面上。
# 原先的检查完全没扫它 (只扫 message / diff / 路径)。

def test_commit_identities_are_surfaced():
    """必须真的读得到身份, 而且计数单位是"提交数"而不是"作者+提交者"的两倍"""
    import scripts.check_privacy as cp

    items = commit_identities()
    assert items, "仓库里应当有提交"
    commits = int(cp._git("rev-list", "--all", "--count"))
    # 每个提交贡献一次计数 (作者与提交者是同一身份时只算一次)
    assert sum(c for _, c in items) == commits
    assert max(c for _, c in items) <= commits


def test_identity_does_not_fail_by_default(monkeypatch):
    """未配置允许清单时只提示不判失败

    每个仓库的每次提交都必然带邮箱, 一律失败会让这个检查永远红着 ——
    而永远红的检查等于没有检查。
    """
    monkeypatch.delenv("SOP_QA_PRIVACY_ALLOWED_IDENTITIES", raising=False)
    assert identity_hits() == []


def test_identity_allowlist_enables_strict_mode(monkeypatch):
    """配置了允许清单, 不在清单里的身份就要判失败"""
    monkeypatch.setenv("SOP_QA_PRIVACY_ALLOWED_IDENTITIES", "nobody@example.com")
    hits = identity_hits()
    assert hits and all("允许清单" in h.rule for h in hits)


def test_identity_allowlist_match_passes(monkeypatch):
    monkeypatch.setenv(
        "SOP_QA_PRIVACY_ALLOWED_IDENTITIES",
        ",".join(ident for ident, _ in commit_identities()),
    )
    assert identity_hits() == []


def test_phone_in_identity_is_flagged_without_allowlist(monkeypatch):
    """身份里出现手机号 —— 即使没配允许清单也必须判失败"""
    monkeypatch.delenv("SOP_QA_PRIVACY_ALLOWED_IDENTITIES", raising=False)
    hits = identity_content_hits(_sample("某人 <", _FAKE_PHONE, "@qq.com>"))
    assert [h.rule for h in hits] == ["身份含手机号"]


def test_plain_email_identity_is_not_a_content_hit():
    """普通邮箱身份不算内容命中 —— 否则每次提交都命中, 检查永远红"""
    assert identity_content_hits("someone <someone@example.com>") == []


def test_identity_report_hides_identities_that_hit_rules(monkeypatch):
    """身份命中敏感规则时, 报告里不得回显该身份 (与 format_report 一致)"""
    import scripts.check_privacy as cp

    monkeypatch.setattr(
        cp, "commit_identities",
        lambda: [(_sample("某人 <", _FAKE_PHONE, "@qq.com>"), 3)],
    )
    report = format_identities()
    assert _FAKE_PHONE not in report
    assert "已隐去" in report


def test_identity_report_shows_normal_identities(monkeypatch):
    """正常身份要照原样显示出来 —— 看不见就没法判断"""
    import scripts.check_privacy as cp

    monkeypatch.setattr(cp, "commit_identities",
                        lambda: [("someone <someone@example.com>", 2)])
    assert "someone <someone@example.com>" in format_identities()
