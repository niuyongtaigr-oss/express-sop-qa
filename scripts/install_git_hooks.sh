#!/usr/bin/env bash
# 安装隐私自检的 git 钩子 —— 把「不要带私密信息进去」变成提交时自动执行
#
# 为什么不是一个 ln -sf 软链到检查脚本:
#   git 调 commit-msg 钩子时会把**提交信息文件的路径**作为 $1 传进来, 而检查脚本
#   用的是 argparse —— 它会把这个位置参数当成无法识别的参数, 直接报错退出 2。
#   结果是**每一次提交都被拦死**, 而且报错信息看起来像是脚本本身坏了。
#   pre-commit 用软链也传不了 --staged, 只能退化成扫描全部已跟踪文件 (更慢且与
#   文档描述的行为不一致)。
# 所以钩子里必须是一行真正的转发脚本。本脚本负责生成它们, 可重复执行。
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
HOOK_DIR="$(git rev-parse --git-path hooks)"
CHECK="scripts/check_privacy.py"

if [ ! -f "$ROOT/$CHECK" ]; then
  echo "找不到 $CHECK, 请在仓库根目录执行" >&2
  exit 1
fi

install_hook() {
  # 拆成多条 local: bash 在执行单条 local 前会先把所有词展开, 写在同一条里时
  # $name 还没被赋值 (set -u 下直接报 unbound variable)
  local name="$1"
  local body="$2"
  local target="$HOOK_DIR/$name"

  # **必须先 unlink 再写**。这里踩过一次坑: 老文档让人把钩子做成
  # `ln -sf ../../scripts/check_privacy.py .git/hooks/pre-commit`, 于是
  # $target 是个**指向检查脚本的软链** —— `> "$target"` 会沿着软链写,
  # 把 scripts/check_privacy.py 的内容直接覆盖成钩子内容 (而钩子又去 exec 它
  # 自己 → 无限递归)。对软链要删掉再建新文件, 不能就地覆盖。
  if [ -L "$target" ]; then
    rm -f "$target"
  elif [ -e "$target" ] && ! grep -q "check_privacy" "$target" 2>/dev/null; then
    echo "  已存在非本项目的 $name 钩子, 备份为 $name.bak"
    mv "$target" "$target.bak"
  elif [ -e "$target" ]; then
    rm -f "$target"
  fi

  printf '%s\n' "$body" > "$target"
  chmod +x "$target"
  echo "  已安装 $name"
}

install_hook pre-commit '#!/usr/bin/env bash
# 由 scripts/install_git_hooks.sh 生成: 只检查暂存区 (快)
root="$(git rev-parse --show-toplevel)"
exec "$root/scripts/check_privacy.py" --staged'

install_hook commit-msg '#!/usr/bin/env bash
# 由 scripts/install_git_hooks.sh 生成: 检查提交信息本身
# pre-commit 看的是暂存的文件内容, 提交信息不在暂存区里 —— 少了这个钩子,
# "代码没问题但提交信息里写了敏感词"完全拦不住。
root="$(git rev-parse --show-toplevel)"
exec "$root/scripts/check_privacy.py" --commit-msg "$1"'

# 兜底自检: 万一把检查脚本本身写坏了, 当场报出来而不是等下次提交才发现
if ! head -1 "$ROOT/$CHECK" | grep -q "python"; then
  echo "!! $CHECK 的首行不是 python shebang —— 它被写坏了, 请 git checkout 恢复" >&2
  exit 1
fi

echo
echo "完成。自检三个入口:"
echo "  提交时自动:  $HOOK_DIR/pre-commit, $HOOK_DIR/commit-msg"
echo "  手动全量:    python3 $CHECK"
echo "  手动查历史:  python3 $CHECK --history"
