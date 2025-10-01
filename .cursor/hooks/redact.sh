#!/bin/sh
set -eu

FILE_PATH="${HOOK_FILEPATH:-unknown}"
ROOT_DIR="${HOOK_ROOT:-$(pwd)}"
LOG_FILE="${ROOT_DIR}/.cursor/redaction.log"

python - "$FILE_PATH" "$LOG_FILE" <<'PY'
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone

file_path = sys.argv[1]
log_file = sys.argv[2]
content = sys.stdin.read()

patterns = [
    re.compile(r"(?i)(api[_-]?key|token|secret|password|authorization|auth[_-]?token)\s*[:=]\s*(['\"]?)([A-Za-z0-9\-_/=]{12,})\2"),
    re.compile(r"(?i)bearer\s+([A-Za-z0-9\-_.=]{16,})"),
    re.compile(r"['\"](AIza[0-9A-Za-z\-_]{35})['\"]"),
    re.compile(r"['\"](ya29\.[0-9A-Za-z\-_]+)['\"]"),
    re.compile(r"(['\"]?)(?:ssh-rsa|ssh-ed25519) [A-Za-z0-9+/=]{20,}(['\"]?)"),
]

redacted = content
findings = []

def apply(pattern, text):
    def replacer(match):
        if match.lastindex:
            for idx in range(1, match.lastindex + 1):
                candidate = match.group(idx)
                if candidate and len(candidate) >= 12:
                    digest = hashlib.sha256(candidate.encode()).hexdigest()[:12]
                    findings.append({
                        "fingerprint": digest,
                        "span": [match.start(idx), match.end(idx)],
                        "pattern": pattern.pattern,
                    })
                    placeholder = f"[REDACTED::{digest}]"
                    start, end = match.span(idx)
                    return text[match.start():start] + placeholder + text[end:match.end()]
        candidate = match.group(0)
        digest = hashlib.sha256(candidate.encode()).hexdigest()[:12]
        findings.append({
            "fingerprint": digest,
            "span": [match.start(), match.end()],
            "pattern": pattern.pattern,
        })
        return f"[REDACTED::{digest}]"
    return pattern.sub(replacer, text)

for pattern in patterns:
    redacted = apply(pattern, redacted)

sys.stdout.write(redacted)

if findings:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "file": file_path,
        "count": len(findings),
        "findings": findings,
        "sha256": hashlib.sha256(redacted.encode()).hexdigest(),
    }
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
PY
