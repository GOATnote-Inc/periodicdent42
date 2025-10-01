#!/bin/sh
set -eu

read_cmd() {
  if [ -n "${HOOK_COMMAND:-}" ]; then
    printf '%s' "$HOOK_COMMAND"
    return 0
  fi
  if [ "$#" -gt 0 ]; then
    printf '%s' "$*"
    return 0
  fi
  if [ ! -t 0 ]; then
    cat
    return 0
  fi
  printf ''
}

CMD="$(read_cmd "$@")"
CMD_TRIMMED=$(printf '%s' "$CMD" | tr '\n' ' ' | sed 's/^ *//;s/ *$//')

if [ -z "$CMD_TRIMMED" ]; then
  exit 0
fi

lower() {
  printf '%s' "$1" | tr 'A-Z' 'a-z'
}

CMD_LOWER=$(lower "$CMD_TRIMMED")

case "$CMD_LOWER" in
  *'rm -rf'*|*'rm -fr'*|*'rm -r '*|*'rm -f /'*|*'shred -'*|*'mkfs '*|*'dd if='*' of='/dev/'*)
    echo "Command blocked: destructive file operation detected." >&2
    exit 20 ;;
  *'git push --force'*|*'git push -f'*|*'git reset --hard'*|*'git clean -fd'*|*'git checkout -f'*)
    echo "Command blocked: destructive git operation requires manual review." >&2
    exit 20 ;;
  *'docker '*' --privileged'*)
    echo "Command blocked: running Docker with --privileged is not allowed." >&2
    exit 20 ;;
  *'curl '*'| bash'*|*'curl '*'|sh'*|*'wget '*' -O - | bash'*|*'wget '*' -O - | sh'*)
    echo "Command blocked: remote script execution via curl|bash is denied." >&2
    exit 20 ;;
  *'npm install -g '*|*'pnpm install -g '*|*'yarn global '*|*'pip install --user '*|*'pip install -U '*|*'pip3 install -U '*|*'pip install --upgrade '*|*'pip3 install --upgrade '*)
    echo "Command blocked: global or in-place package upgrades are not permitted." >&2
    exit 20 ;;
  *'gcloud auth application-default login'*)
    echo "Command blocked: use Workload Identity instead of ADC login." >&2
    exit 20 ;;
  *'.env'*'cat '*|*'cat '*'.env'*|*'grep '*'.env'*|*'less '*'.env'*)
    echo "Command blocked: direct access to .env files is restricted." >&2
    exit 20 ;;
esac

PACKAGE_PATTERNS='pip install|pip3 install|python -m pip install|npm install|pnpm install|yarn install|poetry add|poetry install|uv pip install|cargo install'

ALLOWLIST_FILE=".cursor/allowlist.json"
ALLOW_MATCH=0
if [ -f "$ALLOWLIST_FILE" ]; then
  ALLOW_MATCH=$(python - "$ALLOWLIST_FILE" "$CMD_TRIMMED" <<'PY'
import json
import re
import sys
from pathlib import Path
allow_file = Path(sys.argv[1])
command = sys.argv[2]
try:
    data = json.loads(allow_file.read_text())
except Exception:
    print(0)
    raise SystemExit
patterns = data.get("allow", [])
for pattern in patterns:
    if re.fullmatch(pattern, command):
        print(1)
        raise SystemExit
print(0)
PY
)
fi

if [ "$ALLOW_MATCH" = "1" ]; then
  exit 0
fi

if printf '%s' "$CMD_LOWER" | grep -Eq "$(printf '%s' "$PACKAGE_PATTERNS" | sed 's/ /|/g')"; then
  echo "Package operations require confirmation unless explicitly allowlisted." >&2
  exit 10
fi

echo "Command requires approval: not found in allowlist. Use sandboxed execution or confirm." >&2
exit 10
