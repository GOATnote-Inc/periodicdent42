"""Generate dependency graph information for the repository."""
from __future__ import annotations

import argparse
import ast
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

EXCLUDE_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", "out"}
SUPPORTED_SUFFIXES = {".py", ".ts", ".tsx"}

IMPORT_RE = re.compile(r"^\s*import.+from\s+['\"](?P<module>[^'\"]+)['\"]")
IMPORT_SIDE_RE = re.compile(r"^\s*from\s+['\"](?P<module>[^'\"]+)['\"]\s+import")


def iter_source_files(root: Path) -> Iterable[Path]:
    """Yield repository source files that we want to analyse."""

    for path in root.rglob("*"):
        if path.is_dir():
            if path.name in EXCLUDE_DIRS:
                continue
            continue
        if path.suffix in SUPPORTED_SUFFIXES and not any(part in EXCLUDE_DIRS for part in path.parts):
            yield path


def parse_python(path: Path) -> set[str]:
    """Return imported module names for a Python file."""

    imports: set[str] = set()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return imports
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


def parse_ts(path: Path) -> set[str]:
    """Return imported module names for a TypeScript/TSX file."""

    imports: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return imports
    for line in lines:
        match = IMPORT_RE.match(line) or IMPORT_SIDE_RE.match(line)
        if match:
            imports.add(match.group("module"))
    return imports


def normalize_target(module: str, root_packages: set[str]) -> str | None:
    """Reduce an import string to a top-level package if it exists in the repo."""

    for package in sorted(root_packages, key=len, reverse=True):
        if module == package or module.startswith(f"{package}.") or module.startswith(f"{package}/"):
            return package
    return None


def build_graph(root: Path) -> dict[str, list[str]]:
    """Construct a mapping of source file to packages it depends on."""

    root_packages = {
        path.name
        for path in root.iterdir()
        if path.is_dir() and path.name not in EXCLUDE_DIRS and not path.name.startswith(".")
    }
    graph: dict[str, list[str]] = defaultdict(list)
    for source_path in iter_source_files(root):
        suffix = source_path.suffix
        imports = parse_python(source_path) if suffix == ".py" else parse_ts(source_path)
        for module in imports:
            target = normalize_target(module, root_packages)
            if target is None:
                continue
            graph[str(source_path.relative_to(root))].append(target)
    return graph


def to_mermaid(graph: dict[str, list[str]]) -> str:
    """Serialise the dependency graph into a Mermaid flowchart."""

    lines = ["flowchart TD"]
    for source, targets in sorted(graph.items()):
        node_name = source.replace("/", "_").replace(".", "_")
        for target in sorted(set(targets)):
            target_node = target.replace("/", "_").replace(".", "_")
            lines.append(f"    {node_name} --> {target_node}")
    if len(lines) == 1:
        lines.append("    Empty[No internal dependencies detected]")
    return "\n".join(lines)


def main() -> None:
    """Command-line entry point."""

    parser = argparse.ArgumentParser(description="Generate repository dependency graph")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--json", type=Path, default=Path("docs/dependency_graph.json"))
    parser.add_argument("--mermaid", type=Path, default=Path("docs/ARCHITECTURE_MAP.md"))
    args = parser.parse_args()

    graph = build_graph(args.root)
    payload = {"generated": True, "edges": graph}
    args.json.parent.mkdir(parents=True, exist_ok=True)
    with args.json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    mermaid_diagram = to_mermaid(graph)
    content = (
        "# Architecture Map\n\n"
        "## Service Dependencies\n\n"
        "```mermaid\n"
        f"{mermaid_diagram}\n"
        "```\n\n"
        "## Run Targets\n\n"
        "- `make run.api` — start the FastAPI RAG service on :8000.\n"
        "- `make run.web` — launch the marketing/demo shell.\n"
        "- `make demo` — boot the Next.js demos workspace.\n"
        "- `make graph` — refresh this dependency map.\n"
        "- `make audit` — regenerate audit findings JSON.\n\n"
        "## Data Model\n\n"
        "```mermaid\n"
        "erDiagram\n"
        "    ExperimentRun {\n"
        "        string id PK\n"
        "        string query\n"
        "        json context\n"
        "        json flash_response\n"
        "        json pro_response\n"
        "        float flash_latency_ms\n"
        "        float pro_latency_ms\n"
        "        datetime created_at\n"
        "        string user_id\n"
        "    }\n"
        "    InstrumentRun {\n"
        "        string id PK\n"
        "        string instrument_id\n"
        "        string sample_id\n"
        "        string campaign_id\n"
        "        string status\n"
        "        json metadata_json\n"
        "        string notes\n"
        "        datetime created_at\n"
        "        datetime updated_at\n"
        "    }\n"
        "    ExperimentRun ||--o{ InstrumentRun : logs\n"
        "```\n"
    )
    args.mermaid.parent.mkdir(parents=True, exist_ok=True)
    args.mermaid.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
