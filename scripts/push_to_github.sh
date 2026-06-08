#!/usr/bin/env bash

## 运行指南
#
# 从项目根目录执行：
#   bash scripts/push_to_github.sh "your commit message"
#
# 行为：
# - 若工作区有改动：git add -A && git commit -m "<message>"
# - 无论是否产生新提交：推送当前分支到已配置的 upstream（若未配置 upstream，则提示如何设置）

set -euo pipefail

commit_message="${1:-update}"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Error: current directory is not a git repository." >&2
  exit 1
fi

branch="$(git rev-parse --abbrev-ref HEAD)"

git add -A

if git diff --cached --quiet; then
  echo "No staged changes. Skip commit."
else
  git commit -m "${commit_message}"
fi

if git rev-parse --abbrev-ref --symbolic-full-name '@{u}' >/dev/null 2>&1; then
  git push
else
  echo "Error: no upstream configured for branch '${branch}'." >&2
  echo "Hint: run one of the following once:" >&2
  echo "  git push -u origin ${branch}" >&2
  echo "  git branch --set-upstream-to=origin/${branch} ${branch}" >&2
  exit 2
fi

