from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict
from types import SimpleNamespace

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pytest

from training.build_dataset import DatasetConfig, filter_records, save_split, stratified_split, synthetic_corpus
from training.train import run_training

DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_REGISTRY = BASE_DIR / "training" / "model_registry"


@pytest.fixture(scope="session", autouse=True)
def prepare_artifacts() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    config = DatasetConfig(
        output_dir=DATA_DIR,
        splits={"train": 0.7, "val": 0.15, "test": 0.15},
        domains=["superconductor", "xrd", "synthesis"],
        min_quality="okay",
        seed=123,
    )
    records = list(synthetic_corpus(config.seed))
    filtered = filter_records(records, config.domains, config.min_quality)
    buckets = stratified_split(filtered, config.splits)
    for split_name, split_records in buckets.items():
        save_split(split_records, config.output_dir / f"{split_name}.jsonl")

    cfg = SimpleNamespace(
        data_path=str(DATA_DIR),
        lr=0.01,
        epochs=1,
        model_name="test-model",
        checkpoint_dir=str(MODEL_REGISTRY),
    )
    run_training(cfg)


def pytest_addoption(parser):
    parser.addoption("--cov", action="store", default=None)
    parser.addoption("--cov-report", action="store", default=None)
