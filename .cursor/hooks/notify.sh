#!/bin/sh
set -eu

ROOT_DIR="${HOOK_ROOT:-$(pwd)}"
AUDIT_FILE="${ROOT_DIR}/.cursor/audit.log"
NOTICE_FILE="${ROOT_DIR}/.cursor/notify.log"

RECENT_CHANGES="No edits recorded."
if [ -f "$AUDIT_FILE" ]; then
  RECENT_CHANGES=$(tail -n 5 "$AUDIT_FILE" 2>/dev/null || echo "No audit entries.")
fi

CURSOR_RECENT_CHANGES="$RECENT_CHANGES" python - "$NOTICE_FILE" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone

notice_file = sys.argv[1]
summary = os.environ.get("HOOK_SESSION_SUMMARY", "Session complete.")
blocked = [cmd.strip() for cmd in os.environ.get("HOOK_BLOCKED_COMMANDS", "").split('\n') if cmd.strip()]
pending = [cmd.strip() for cmd in os.environ.get("HOOK_PENDING_CONFIRMATIONS", "").split('\n') if cmd.strip()]
recent = os.environ.get("CURSOR_RECENT_CHANGES", "").split('\n')
entry = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "summary": summary,
    "blocked_commands": blocked,
    "pending_confirmations": pending,
    "recent_audit": recent[-5:],
}
os.makedirs(os.path.dirname(notice_file), exist_ok=True)
with open(notice_file, "a", encoding="utf-8") as fh:
    fh.write(json.dumps(entry) + "\n")
print("=== Session Summary ===")
print(summary)
if blocked:
    print("Blocked:")
    for cmd in blocked:
        print(f" - {cmd}")
if pending:
    print("Pending confirmation:")
    for cmd in pending:
        print(f" - {cmd}")
print("Recent edits tracked in .cursor/audit.log")
PY
