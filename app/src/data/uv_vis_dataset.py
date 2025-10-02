"""High-fidelity UV-Vis reference dataset loader."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from src.utils.settings import settings


@dataclass
class UvVisSample:
    """Single UV-Vis sample entry."""

    sample_id: str
    analyte: str
    composition: Dict[str, float]
    wavelengths_nm: List[float]
    absorbance: List[float]
    labels: Dict[str, float]
    metadata: Dict[str, float]


class UvVisReferenceLibrary:
    """In-memory dataset wrapper with sampling utilities."""

    def __init__(self, samples: List[UvVisSample]) -> None:
        if not samples:
            raise ValueError("UV-Vis reference library requires at least one sample")
        self._samples = samples

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._samples)

    def __iter__(self) -> Iterator[UvVisSample]:  # pragma: no cover - trivial
        return iter(self._samples)

    def sample_batch(self, *, size: int) -> List[UvVisSample]:
        if size <= 0:
            raise ValueError("size must be positive")
        if size > len(self._samples):
            return list(self._samples)
        return self._samples[:size]

    def get(self, sample_id: str) -> Optional[UvVisSample]:
        return next((sample for sample in self._samples if sample.sample_id == sample_id), None)


def _default_dataset_path() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        candidate = parent / "configs" / "uv_vis_reference_library.json"
        if candidate.exists():
            return candidate
    return current.parents[2] / "configs" / "uv_vis_reference_library.json"


def load_reference_library(path: Optional[Path] = None) -> UvVisReferenceLibrary:
    """Load UV-Vis dataset from disk."""

    dataset_path = path
    if dataset_path is None:
        if settings.UV_VIS_DATASET_PATH:
            dataset_path = Path(settings.UV_VIS_DATASET_PATH)
        else:
            dataset_path = _default_dataset_path()

    if not dataset_path.exists():
        raise FileNotFoundError(f"UV-Vis dataset not found at {dataset_path}")

    with dataset_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    samples = [
        UvVisSample(
            sample_id=entry["sample_id"],
            analyte=entry["analyte"],
            composition=entry["composition"],
            wavelengths_nm=entry["wavelength_nm"],
            absorbance=entry["absorbance"],
            labels=entry["labels"],
            metadata=entry.get("metadata", {}),
        )
        for entry in raw
    ]
    return UvVisReferenceLibrary(samples)
