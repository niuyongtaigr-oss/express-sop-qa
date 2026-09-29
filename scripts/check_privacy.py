#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""仓库隐私自检 — 防止个人信息随提交进入公开仓库

本仓库是公开的, 一旦提交, 内容就永久留在 git 历史里 (要清掉得改写历史 +
force push)。所以把「不要带私密信息进去」做成可执行的检查, 而不是一句口头约定。

用法:
  python3 scripts/check_privacy.py               # 检查全部已跟踪文件
  python3 scripts/check_privacy.py --staged      # 只检查暂存区 (pre-commit 用)
  python3 scripts/check_privacy.py --commit-msg F # 检查一条提交信息 (commit-msg 钩子用)
  python3 scripts/check_privacy.py --history     # 检查**全部提交历史** (message + diff + 作者身份)
  python3 scripts/check_privacy.py --identities  # 只看提交作者身份 (会公开显示的那个字段)
  python3 scripts/check_privacy.py --list-rules  # 打印当前生效的规则

--history 为什么必要:
  工作区干净 ≠ 历史干净。敏感信息可能在某次提交里出现过、后来又被删掉 ——
  当前文件里查不到, 但它仍然留在变更记录里, 只有改写历史才能清掉。
  同理, **提交信息**也要查: 改了文件内容却忘了改 commit message 是很常见的疏漏。

还有一个更容易被忽略的字段: **提交的作者身份 (name <email>)**。它由 git config 决定,
和文件内容一样会公开显示在 GitHub 的每一次提交上, 却没有任何"提交前看一眼"的习惯 ——
用工作邮箱提交作品集仓库, 就是把任职单位域名挂在了公开页面上。本脚本原先只扫
message/diff/路径, 完全没看这个字段。

装成钩子 (推荐**两个都装**):
# 装成钩子 —— **不要用 ln -sf 软链**, 用安装脚本:
#   scripts/install_git_hooks.sh
# 为什么不能用软链: git 调 commit-msg 钩子时会把"提交信息文件路径"作为 $1 传进来,
# 软链过去的本脚本用的是 argparse, 收到这个位置参数会报错退出 2 —— 结果是**每一次
# 提交都被拦死**; pre-commit 也一样传不了 --staged。安装脚本写的是转发脚本。

为什么 commit-msg 也要装: `--staged` 看的是**暂存的文件内容**, 而提交信息不在暂存
区里 —— 只看暂存区就完全拦不住"代码没问题、但提交信息里写了敏感词"。本仓库真的
踩过这个坑 (写提交信息时复述了一个敏感词), 事后才发现, 而提交信息同样会永久留在
公开仓库的变更记录里。

CI / pytest:
  tests/test_repo_hygiene.py 会在每次跑测试时执行同一套检查。

设计要点 — 为什么本脚本里没有「具体敏感词」:
  把姓名/手机号写进检查脚本, 等于换个文件继续泄露。所以本脚本只包含
  「类别」正则 (手机号长什么样、邮箱长什么样), 不含任何具体值。项目相关的
  自定义敏感词由下面两个渠道提供, 二者都不进仓库:
    1) 环境变量 SOP_QA_PRIVACY_TERMS (逗号分隔)
    2) 仓库根的 .privacy-terms (每行一个, 已 gitignore)
  这样检查规则本身可以公开, 而「要防什么」留在本地。

另一个细节: 命中报告 **不回显命中的原文**。检查脚本的输出会进 CI 日志,
如果把命中的手机号原样打出来, 这个检查本身就变成了泄露渠道。
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── 内容规则 ─────────────────────────────────────────────
# 只描述「类别」, 不含具体值。注意各正则不会匹配到自身源码 (字符类里含 '[' 等)。
CONTENT_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("手机号", re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")),
    # 域名部分必须允许**多级**: 「[\w-]+\.[A-Za-z]{2,}」这种写法遇到
    # users.noreply.github.com 会只匹配到 "…@users.noreply" —— 匹配被截断,
    # 于是下面按 endswith 做白名单判断永远失败, 把正常文档误报成泄露。
    ("邮箱", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}")),
    ("身份证号", re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")),
    ("本地绝对路径", re.compile(r"/home/[A-Za-z0-9_.-]+/|/mnt/[a-z]/|[A-Za-z]:\\\\[Uu]sers\\\\")),
    ("密钥样式", re.compile(
        r"sk-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}"
        r"|-----BEGIN [A-Z ]*PRIVATE KEY-----"
    )),
]

# 邮箱白名单: 文档里出现示例邮箱是正常的, 不应误报
EMAIL_ALLOWED_DOMAINS = (
    "example.com", "example.org", "example.net",
    "users.noreply.github.com", "localhost",
)

# ── 路径规则 (只作用于文件名, 不作用于内容) ──────────────
# 用 unicode 转义书写, 使本脚本源码自身不含这些字面词 —— 保证自检时不会自命中
PATH_RULE = re.compile("\u7b80\u5386|resume|\u8bc1\u4ef6|\u8eab\u4efd\u8bc1", re.IGNORECASE)
PATH_RULE_LABEL = "私密文件名(个人材料类)"

# 二进制与不需要扫描的路径
SKIP_DIR_PREFIXES = (".venv/", "venv/", "node_modules/", "data/chroma/", ".git/")


@dataclass(frozen=True)
class Hit:
    """一条命中 — 只记位置, 不记原文"""

    path: str
    line: int          # 0 表示命中在文件名上
    rule: str
    length: int        # 命中串长度, 便于定位是哪类内容


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout


def _custom_terms() -> list[str]:
    """本地自定义敏感词 (环境变量 + .privacy-terms), 二者都不进仓库"""
    terms = [t.strip() for t in os.environ.get("SOP_QA_PRIVACY_TERMS", "").split(",")]
    terms = [t for t in terms if t]
    local = PROJECT_ROOT / ".privacy-terms"
    if local.exists():
        for line in local.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                terms.append(line)
    return terms


def active_rules() -> list[tuple[str, re.Pattern[str]]]:
    """当前生效的全部内容规则 (内置类别 + 本地自定义词)"""
    rules = list(CONTENT_RULES)
    terms = _custom_terms()
    if terms:
        rules.append((
            "自定义敏感词",
            re.compile("|".join(re.escape(t) for t in terms)),
        ))
    return rules


def _is_binary(data: bytes) -> bool:
    return b"\x00" in data[:8192]


def _target_files(staged: bool) -> list[str]:
    if staged:
        out = _git("diff", "--cached", "--name-only", "--diff-filter=ACM")
    else:
        out = _git("ls-files")
    return [
        f for f in out.splitlines()
        if f.strip() and not f.startswith(SKIP_DIR_PREFIXES)
    ]


def scan(
    files: list[str],
    rules: list[tuple[str, re.Pattern[str]]] | None = None,
    root: Path = PROJECT_ROOT,
) -> list[Hit]:
    """扫描指定文件列表, 返回命中 (不含原文)。

    root 可替换, 便于测试在临时目录里验证检出能力。
    """
    rules = rules if rules is not None else active_rules()
    hits: list[Hit] = []
    for rel in files:
        if PATH_RULE.search(Path(rel).name):
            hits.append(Hit(path=rel, line=0, rule=PATH_RULE_LABEL, length=len(rel)))
        path = root / rel
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if _is_binary(data):
            continue
        text = data.decode("utf-8", errors="ignore")
        for lineno, line in enumerate(text.splitlines(), 1):
            for rule_name, pattern in rules:
                match = pattern.search(line)
                if not match:
                    continue
                if rule_name == "邮箱" and match.group(0).endswith(EMAIL_ALLOWED_DOMAINS):
                    continue
                hits.append(
                    Hit(path=rel, line=lineno, rule=rule_name, length=len(match.group(0)))
                )
    return hits


def scan_tracked() -> list[Hit]:
    """检查全部已跟踪文件 (pytest / CI 入口)"""
    return scan(_target_files(staged=False))


def scan_staged() -> list[Hit]:
    """检查暂存区 (pre-commit 入口)"""
    return scan(_target_files(staged=True))


# ── 历史扫描 ─────────────────────────────────────────────
# 当前文件干净 ≠ 提交日志干净: 敏感信息可能在某次提交里出现过、后来被删掉。
# 那种情况下工作区查不到, 但它仍在变更记录里, 只有改写历史才能清掉。

_HISTORY_MARK = "\x1e"  # 记录分隔符, 正文里不会出现

# 允许的提交身份 (逗号分隔的完整 "name <email>" 或仅 email)。两种模式的差别:
#   已配置 → 不在清单里的身份**判失败** (严格模式, 适合 CI)
#   未配置 → 只打印出来让人看一眼, 不判失败
# 为什么默认不判失败: 每个仓库的每次提交都必然带一个邮箱, 一律失败等于让这个检查
# 永远红着 —— 而永远红的检查等于没有检查。
IDENTITY_ENV = "SOP_QA_PRIVACY_ALLOWED_IDENTITIES"


def commit_identities() -> list[tuple[str, int]]:
    """本仓库全部提交的作者/提交者身份及出现次数 (按次数降序)"""
    # 作者与提交者用 \x1f 分隔, 每个提交内去重后再计数 —— 否则一个提交会被数两次,
    # 报出来的 "× 52" 会让人以为身份出现了 52 次
    raw = _git("log", "--all", "--format=%an <%ae>%x1f%cn <%ce>")
    counts: Counter[str] = Counter()
    for line in raw.splitlines():
        for ident in {p.strip() for p in line.split("\x1f") if p.strip()}:
            counts[ident] += 1
    return counts.most_common()


def allowed_identities() -> set[str]:
    return {t.strip().lower()
            for t in os.environ.get(IDENTITY_ENV, "").split(",") if t.strip()}


def identity_content_hits(ident: str) -> list[Hit]:
    """身份里出现手机号/身份证/本地路径 —— 判失败。那不是正常的提交身份。

    邮箱与密钥样式两条规则**不适用**于身份字段: 身份里必然有邮箱, 用邮箱规则去判
    等于每次都命中。
    """
    hits: list[Hit] = []
    for rule_name, pattern in CONTENT_RULES:
        if rule_name in ("邮箱", "密钥样式"):
            continue
        if pattern.search(ident):
            hits.append(Hit(path="提交作者身份", line=0,
                            rule=f"身份含{rule_name}", length=len(ident)))
    return hits


def identity_hits() -> list[Hit]:
    """身份字段的问题 —— 两类判定标准不同:

      · 身份里出现手机号/身份证/本地路径: **判失败** (identity_content_hits)。
      · 身份只是一个没被允许清单收入的工作邮箱: **不判失败**, 由 --identities
        打印出来供人判断。用工作邮箱提交作品集仓库未必是错, 但必须是**有意识**
        的选择; 工具能做的是把它摆到眼前。
    """
    allowed = allowed_identities()
    hits: list[Hit] = []
    for ident, _ in commit_identities():
        hits.extend(identity_content_hits(ident))
        if allowed and ident.lower() not in allowed:
            hits.append(Hit(path="提交作者身份", line=0,
                            rule="不在允许清单内 (原文已隐去)",
                            length=len(ident)))
    return hits


def format_identities() -> str:
    items = commit_identities()
    if not items:
        return "提交身份: (无提交)"
    lines = ["提交身份 (这个字段同样会公开显示在 GitHub 上):"]
    for ident, count in items:
        # 身份原则上照原样打印 (它本来就在 GitHub 上公开)。但如果它自己命中了敏感
        # 规则, 就不能再回显 —— 与 format_report 的"不回显命中原文"保持一致。
        shown = "[已隐去: 该身份命中敏感规则]" if identity_content_hits(ident) else ident
        lines.append(f"  · {shown}  × {count}")
    if not allowed_identities():
        lines.append(
            f"  → 若其中某个不属于本仓库, 现在改还来得及 (只影响之后的提交):\n"
            f"      git config user.email '<你希望公开的邮箱>'\n"
            f"    要严格拦住: 设 {IDENTITY_ENV}='<可接受的完整身份>' (逗号分隔)\n"
            f"    注意: 已提交的身份无法靠「补一次提交」改掉 —— 必须改写历史并 force push,\n"
            f"    而旧提交在 GitHub 上按精确 SHA 仍可解析。"
        )
    return "\n".join(lines)


def _history_bodies() -> list[tuple[str, str]]:
    """返回 [(sha, 该提交的 message + patch 全文), ...]

    格式串必须带上 %B —— 只写 %H 会把默认输出里的 commit message 挤掉,
    于是「提交信息里的敏感词」整类漏检 (这正是本函数曾经踩过的坑)。
    """
    dump = _git("log", "--all", "-p", f"--format={_HISTORY_MARK}%H%n%B")
    entries: list[tuple[str, str]] = []
    for chunk in dump.split(_HISTORY_MARK)[1:]:
        sha, _, body = chunk.partition("\n")
        entries.append((sha.strip(), body))
    return entries


def _match_line(line: str, rules: list[tuple[str, re.Pattern[str]]]) -> list[Hit]:
    """单行匹配 (与 scan 共用同一套判定与白名单)"""
    found: list[Hit] = []
    for rule_name, pattern in rules:
        match = pattern.search(line)
        if not match:
            continue
        if rule_name == "邮箱" and match.group(0).endswith(EMAIL_ALLOWED_DOMAINS):
            continue
        found.append(Hit(path="", line=0, rule=rule_name, length=len(match.group(0))))
    return found


_DIFF_FILE = re.compile(r"^\+\+\+ b/(.+)$")


def scan_commit_message(path: Path) -> list[Hit]:
    """检查一条提交信息文本 (commit-msg 钩子入口)。

    git 传进来的是 COMMIT_EDITMSG, 里面除了用户写的内容, 还有一大段以 `#` 开头的
    模板与状态行 —— 那些在提交时会被丢掉, 不参与判定 (否则模板里随便出现一个
    邮箱域名就会误报)。
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    rules = active_rules()
    hits: list[Hit] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("#"):
            continue
        for hit in _match_line(line, rules):
            hits.append(Hit(path="(commit message)", line=lineno,
                            rule=hit.rule, length=hit.length))
    return hits


def scan_history(rules: list[tuple[str, re.Pattern[str]]] | None = None) -> list[Hit]:
    """扫描全部提交历史: 每个提交的 message + diff, 以及历史路径名。

    message 也要查 —— 改了文件内容却忘了改 commit message 是很常见的疏漏,
    而 message 同样会永久留在公开仓库的变更记录里。

    两个实现细节:
      - diff 的内容行带 `+` / `-` / 空格前缀。**必须去掉前缀再匹配**, 否则前缀
        会与紧邻的装饰器写法连成一体, 被邮箱规则误判成一个邮箱地址。
      - 跟踪 `+++ b/<path>` 以报出具体文件, 而不是只给一个提交内的行号。
    """
    rules = rules if rules is not None else active_rules()
    hits: list[Hit] = []
    for sha, body in _history_bodies():
        current = "(commit message)"
        for lineno, line in enumerate(body.splitlines(), 1):
            match = _DIFF_FILE.match(line)
            if match:
                current = match.group(1)
                continue
            probe = line[1:] if line[:1] in "+-" else line
            for hit in _match_line(probe, rules):
                hits.append(
                    Hit(path=f"{sha[:7]}:{current}", line=lineno,
                        rule=hit.rule, length=hit.length)
                )

    # 历史中出现过的所有路径 (文件可能已被删除, 但路径仍留在变更记录里)
    for path in _git("log", "--all", "--pretty=format:", "--name-only").splitlines():
        path = path.strip()
        if path and PATH_RULE.search(Path(path).name):
            hits.append(
                Hit(path=f"历史路径 {path}", line=0, rule=PATH_RULE_LABEL, length=len(path))
            )
    return hits


def format_report(hits: list[Hit]) -> str:
    """生成报告 — 不回显命中的原文, 避免检查输出本身成为泄露渠道"""
    if not hits:
        return "隐私自检通过: 未发现个人信息 / 私密文件名"
    lines = [f"隐私自检发现 {len(hits)} 处问题 (命中内容已隐去, 请自行核对):", ""]
    for hit in hits:
        where = hit.path if hit.line == 0 else f"{hit.path}:{hit.line}"
        lines.append(f"  ✗ [{hit.rule}] {where}  (长度 {hit.length})")
    lines += [
        "",
        "处理建议:",
        "  · 换成中性示例 (本仓库是快递 SOP 项目, 用行业内的词即可)",
        "  · 确属误报: 加进 EMAIL_ALLOWED_DOMAINS, 或在 .privacy-terms 里调整自定义词",
        "  · 已提交的敏感信息: 不要只补一次提交, 必须改写历史 (git commit --amend",
        "    或 git filter-repo) 后 force push —— 否则它会一直留在变更记录里",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="仓库隐私自检 (公开仓库防泄露)")
    parser.add_argument("--staged", action="store_true", help="只检查暂存区 (pre-commit 用)")
    parser.add_argument("--history", action="store_true",
                        help="检查全部提交历史 (message + diff + 历史路径名 + 作者身份)")
    parser.add_argument("--identities", action="store_true",
                        help="只打印提交的作者身份 (这个字段同样会公开显示)")
    parser.add_argument("--commit-msg", metavar="FILE",
                        help="检查一个提交信息文件 (commit-msg 钩子用)")
    parser.add_argument("--list-rules", action="store_true", help="打印当前生效的规则")
    parser.add_argument("--quiet", action="store_true", help="仅在发现问题时输出")
    args = parser.parse_args(argv)

    if args.list_rules:
        print("内容规则:")
        for name, pattern in active_rules():
            print(f"  · {name}: {pattern.pattern[:70]}")
        print(f"路径规则:\n  · {PATH_RULE_LABEL}: {PATH_RULE.pattern}")
        terms = _custom_terms()
        print(f"本地自定义词: {len(terms)} 个" if terms else "本地自定义词: 无")
        return 0

    if args.commit_msg:
        hits = scan_commit_message(Path(args.commit_msg))
        if hits or not args.quiet:
            print(format_report(hits))
        return 1 if hits else 0

    if args.identities:
        print(format_identities())
        hits = identity_hits()
        if hits:
            print()
            print(format_report(hits))
            return 1
        return 0

    if args.history:
        hits = scan_history() + identity_hits()
        # 身份字段单独打印: 它是"给你看一眼"的信息, 与会回显隐去的命中项不同
        if not args.quiet:
            print(format_identities())
            print()
        if not hits:
            print("历史自检通过: 提交记录中未发现个人信息 / 私密路径名 / 可疑提交身份")
            return 0
        print(format_report(hits))
        return 1

    hits = scan_staged() if args.staged else scan_tracked()
    if hits or not args.quiet:
        print(format_report(hits))
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
