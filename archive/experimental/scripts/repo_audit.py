"""Lightweight static audit heuristics for the repository."""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

EXCLUDE_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", "out"}
PY_SUFFIX = ".py"
TODO_PATTERN = re.compile(r"TODO|FIXME", re.IGNORECASE)


@dataclass
class Finding:
    path: str
    category: str
    issue: str
    evidence: str
    recommendation: str
    effort: int
    priority: str


def iter_python_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.py"):
        if any(part in EXCLUDE_DIRS for part in path.parts):
            continue
        yield path


def has_tests_for(path: Path, tests_root: Path) -> bool:
    module_name = path.stem
    if not tests_root.exists():
        return False
    pattern = re.compile(rf"\b{re.escape(module_name)}\b")
    for test_file in tests_root.rglob("test_*.py"):
        if pattern.search(test_file.read_text(encoding="utf-8")):
            return True
    return False


def scan_file(path: Path) -> dict[str, int | bool]:
    text = path.read_text(encoding="utf-8")
    todo_matches = TODO_PATTERN.findall(text)
    long_functions = 0
    current_length = 0
    inside_def = False
    for line in text.splitlines():
        if line.strip().startswith("def "):
            inside_def = True
            current_length = 1
            continue
        if inside_def:
            if line.startswith("def ") or line.startswith("class "):
                inside_def = False
                if current_length > 100:
                    long_functions += 1
            else:
                current_length += 1
    return {"todo_count": len(todo_matches), "long_functions": long_functions}


def build_findings(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    tests_root = root / "tests"
    for py_file in iter_python_files(root):
        relative = py_file.relative_to(root)
        metrics = scan_file(py_file)
        if relative.parts and relative.parts[0] in {"services", "apps", "app"}:
            if not has_tests_for(py_file, tests_root):
                findings.append(
                    Finding(
                        path=str(relative),
                        category="Testing",
                        issue="Missing dedicated unit tests",
                        evidence="No references found in tests/ directory",
                        recommendation="Add focused unit tests covering public behaviour",
                        effort=2,
                        priority="High" if "api" in relative.parts else "Medium",
                    )
                )
        if metrics["todo_count"]:
            findings.append(
                Finding(
                    path=str(relative),
                    category="Code Hygiene",
                    issue="Lingering TODO/FIXME comment",
                    evidence=f"Found {metrics['todo_count']} TODO markers",
                    recommendation="Document ownership or resolve TODO before release",
                    effort=1,
                    priority="Medium",
                )
            )
        if metrics["long_functions"]:
            findings.append(
                Finding(
                    path=str(relative),
                    category="Maintainability",
                    issue="Functions exceeding 100 lines",
                    evidence=f"Detected {metrics['long_functions']} long function(s)",
                    recommendation="Refactor into smaller helpers to aid readability",
                    effort=3,
                    priority="Medium",
                )
            )
    return findings


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lightweight repo audit")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("docs/audit.json"))
    args = parser.parse_args()

    findings = [asdict(finding) for finding in build_findings(args.root)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump({"findings": findings}, handle, indent=2)


if __name__ == "__main__":
    main()
