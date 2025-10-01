from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

QUALITY_ORDER = {"bad": 0, "okay": 1, "good": 2}
DOMAINS = ["xrd", "transport", "synthesis", "superconductor", "catalyst", "generic"]


@dataclass
class DatasetConfig:
    output_dir: Path
    splits: Dict[str, float]
    domains: List[str]
    min_quality: str
    seed: int

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "DatasetConfig":
        return cls(
            output_dir=Path(str(data["output_dir"])),
            splits={str(k): float(v) for k, v in dict(data["splits"]).items()},
            domains=[str(d) for d in data["domains"]],
            min_quality=str(data["min_quality"]),
            seed=int(data["seed"]),
        )


def synthetic_corpus(seed: int) -> Iterable[Dict[str, object]]:
    rng = np.random.default_rng(seed)
    base_date = datetime(2024, 1, 1)
    goals = {
        "superconductor": "maximize critical temperature",
        "transport": "improve charge mobility",
        "xrd": "sharpen (110) diffraction peak",
        "synthesis": "increase powder yield",
        "catalyst": "optimize turnover frequency",
        "generic": "reduce cycle time",
    }
    for domain in DOMAINS:
        for idx in range(40):
            date = base_date + timedelta(days=idx)
            context = {
                "recent_runs": [
                    {
                        "params": {"anneal_temp_C": 450 + 10 * rng.normal()},
                        "measurement": rng.normal(50, 5),
                        "qc_pass": bool(rng.random() > 0.1),
                        "outcome": rng.normal(50, 3),
                    }
                    for _ in range(3)
                ],
                "domain": domain,
                "task_id": f"{domain}_{idx:03d}",
                "timestamp": date.isoformat(),
            }
            constraints = {
                "max_temp_C": 900,
                "max_ramp_C_per_min": 10,
                "max_voltage_V": 5,
            }
            candidate_space = {
                "anneal_temp_C": [350, 950],
                "anneal_time_min": [5, 120],
                "doping_pct": [0.0, 0.2],
            }
            plan_text = (
                f"Run anneal at {450 + 5 * np.sin(idx)} C followed by gradual cool down."
            )
            rationale_text = (
                "Hypothesize improved lattice ordering from controlled anneal;"
                " monitor conductivity and adjust doping."
            )
            quality = rng.choice(["good", "okay", "bad"], p=[0.4, 0.45, 0.15])
            trajectory = [
                {
                    "params": {
                        "anneal_temp_C": float(450 + 20 * rng.normal()),
                        "anneal_time_min": float(rng.uniform(10, 80)),
                        "doping_pct": float(rng.uniform(0.0, 0.2)),
                    },
                    "measurement": float(rng.normal(60, 8)),
                    "qc": {"passed": bool(rng.random() > 0.1)},
                    "outcome": float(rng.normal(55, 6)),
                    "success": bool(rng.random() > 0.2),
                }
                for _ in range(2)
            ]
            label = {
                "action": "anneal",
                "anneal_temp_C": float(450 + 15 * np.cos(idx / 4)),
                "anneal_time_min": float(30 + 5 * rng.normal()),
                "doping_pct": float(max(0.0, min(0.2, rng.normal(0.05, 0.02)))),
            }
            tags = [domain]
            if domain == "superconductor":
                tags.append("transport")
            record = {
                "goal": goals[domain],
                "context": context,
                "constraints": constraints,
                "candidate_space": candidate_space,
                "plan_text": plan_text,
                "rationale_text": rationale_text,
                "trajectory": trajectory,
                "label": label,
                "quality": quality,
                "tags": tags,
                "domain": domain,
                "task": "materials_campaign",
                "created_at": date.isoformat(),
            }
            yield record


def stratified_split(records: List[Dict[str, object]], splits: Dict[str, float]) -> Dict[str, List[Dict[str, object]]]:
    total = float(sum(splits.values()))
    normalized = {name: frac / total for name, frac in splits.items()}
    buckets: Dict[str, List[Dict[str, object]]] = {name: [] for name in splits}
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for record in records:
        domain = str(record["domain"])
        grouped.setdefault(domain, []).append(record)
    for domain_records in grouped.values():
        domain_records.sort(key=lambda r: r["created_at"])
        n = len(domain_records)
        start = 0
        for split_name, frac in normalized.items():
            end = start + int(round(frac * n))
            buckets[split_name].extend(domain_records[start:end])
            start = end
        if start < n:
            buckets[next(iter(splits))].extend(domain_records[start:])
    return buckets


def filter_records(records: Iterable[Dict[str, object]], domains: List[str], min_quality: str) -> List[Dict[str, object]]:
    allowed = {d for d in domains}
    threshold = QUALITY_ORDER[min_quality]
    filtered = [
        record
        for record in records
        if record["domain"] in allowed and QUALITY_ORDER[record["quality"]] >= threshold
    ]
    return filtered


def save_split(records: Iterable[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build processed planner dataset")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config_dict = json.loads(Path(args.config).read_text()) if config_path.suffix == ".json" else None
    if config_dict is None:
        import yaml  # type: ignore

        config_dict = yaml.safe_load(config_path.read_text())
    config = DatasetConfig.from_dict(config_dict)

    all_records = list(synthetic_corpus(config.seed))
    filtered_records = filter_records(all_records, config.domains, config.min_quality)
    buckets = stratified_split(filtered_records, config.splits)

    for split_name, split_records in buckets.items():
        save_split(split_records, config.output_dir / f"{split_name}.jsonl")

    print(f"Wrote dataset with {sum(len(v) for v in buckets.values())} records to {config.output_dir}")


if __name__ == "__main__":
    main()
