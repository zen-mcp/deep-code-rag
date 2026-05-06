#!/usr/bin/env bash
set -euo pipefail

base_branch="${1:-develop}"
current_branch="$(git branch --show-current)"

if [[ "$current_branch" == "develop" || "$current_branch" == "main" ]]; then
  echo "✅ Skipping rebase for $current_branch"
  exit 0
fi

git fetch origin "$base_branch"

behind_count="$(git rev-list --count "$current_branch..origin/$base_branch")"
if [ "$behind_count" -eq 0 ]; then
  echo "✅ Branch $current_branch is up to date with $base_branch"
  exit 0
fi

if ! git rebase "origin/$base_branch"; then
  echo "❌ Rebase conflict detected."
  echo "    1. Resolve conflicts manually"
  echo "    2. Run: 'git rebase --continue' (or '--abort')"
  echo "    3. Run: git push"
  git rebase --abort 2>/dev/null || true
  exit 1
fi

echo "✅ Branch $current_branch rebased successfully."
