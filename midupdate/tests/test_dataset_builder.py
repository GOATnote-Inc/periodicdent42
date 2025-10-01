from __future__ import annotations

import json
from pathlib import Path

from training.build_dataset import DatasetConfig, filter_records, save_split, stratified_split, synthetic_corpus

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"


def test_dataset_contains_required_fields():
    train_path = DATA_DIR / "train.jsonl"
    assert train_path.exists()
    with train_path.open("r", encoding="utf-8") as fh:
        sample = json.loads(next(iter(fh)))
    for field in [
        "goal",
        "context",
        "constraints",
        "candidate_space",
        "plan_text",
        "rationale_text",
        "trajectory",
        "label",
        "quality",
        "tags",
    ]:
        assert field in sample
