#!/bin/sh
set -eu

ROOT_DIR="${HOOK_ROOT:-$(pwd)}"
AUDIT_FILE="${ROOT_DIR}/.cursor/audit.log"
FILE_PATH="${HOOK_FILEPATH:-unknown}"
USER_NAME="${CURSOR_USER:-${USER:-unknown}}"

diff_content="$(cat)"

DIFF_CONTENT="$diff_content" CURSOR_AUDIT_FILE="$AUDIT_FILE" CURSOR_AUDIT_USER="$USER_NAME" CURSOR_AUDIT_PATH="$FILE_PATH" python - <<'PY'
import hashlib
import json
import os
from datetime import datetime, timezone

audit_file = os.environ["CURSOR_AUDIT_FILE"]
file_path = os.environ.get("CURSOR_AUDIT_PATH", "unknown")
user_name = os.environ.get("CURSOR_AUDIT_USER", "unknown")
raw = os.environ.get("DIFF_CONTENT", "")
summary = raw.strip().replace('\n', ' ')
if len(summary) > 200:
    summary = summary[:197] + "..."
fingerprint = hashlib.sha256(raw.encode()).hexdigest()
record = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "user": user_name,
    "file": file_path,
    "summary": summary,
    "sha256": fingerprint,
}
os.makedirs(os.path.dirname(audit_file), exist_ok=True)
with open(audit_file, "a", encoding="utf-8") as fh:
    fh.write(json.dumps(record) + "\n")
PY
