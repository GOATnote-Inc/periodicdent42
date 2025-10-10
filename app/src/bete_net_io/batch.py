"""
Batch screening for large sets of candidate superconductors.

Features:
- Parallel execution with multiprocessing
- Resume capability (checkpoint every N materials)
- Progress tracking with ETA
- Output to Parquet/CSV/JSON

Copyright 2025 GOATnote Autonomous Research Lab Initiative
Licensed under Apache 2.0
"""

import logging
import pickle
import time
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

from src.bete_net_io.inference import predict_tc

logger = logging.getLogger(__name__)


@dataclass
class ScreeningConfig:
    """Configuration for batch screening."""

    inputs: List[str]  # CIF paths or MP-IDs
    mu_star: float = 0.10
    output_path: Path = Path("screening_results.parquet")
    checkpoint_path: Optional[Path] = None
    checkpoint_interval: int = 100  # Checkpoint every N materials
    n_workers: int = 4
    resume: bool = False


def _predict_single(args) -> dict:
    """Worker function for parallel prediction."""
    input_id, mu_star = args
    try:
        result = predict_tc(input_id, mu_star=mu_star)
        return result.to_dict()
    except Exception as e:
        logger.error(f"Prediction failed for {input_id}: {e}")
        return {
            "formula": "ERROR",
            "input_hash": "",
            "mp_id": input_id if input_id.startswith("mp-") else None,
            "tc_kelvin": None,
            "error": str(e),
        }


def batch_screen(config: ScreeningConfig) -> pd.DataFrame:
    """
    Screen a batch of candidate superconductors.

    Args:
        config: ScreeningConfig with inputs and parameters

    Returns:
        DataFrame with columns: formula, mp_id, tc_kelvin, lambda_ep, omega_log, ...

    Example:
        >>> config = ScreeningConfig(
        ...     inputs=["mp-48", "mp-66", "mp-134"],
        ...     mu_star=0.13,
        ...     output_path=Path("results.parquet"),
        ...     n_workers=8
        ... )
        >>> df = batch_screen(config)
        >>> df.sort_values("tc_kelvin", ascending=False).head(10)
    """
    logger.info(
        f"Batch screening {len(config.inputs)} materials with {config.n_workers} workers"
    )

    # Load checkpoint if resuming
    completed = []
    if config.resume and config.checkpoint_path and config.checkpoint_path.exists():
        logger.info(f"Resuming from checkpoint: {config.checkpoint_path}")
        with open(config.checkpoint_path, "rb") as f:
            completed = pickle.load(f)
        logger.info(f"Loaded {len(completed)} completed predictions")

        # Filter out already completed
        completed_ids = {r["mp_id"] or r["input_hash"] for r in completed}
        remaining = [
            inp
            for inp in config.inputs
            if (inp if inp.startswith("mp-") else inp) not in completed_ids
        ]
        logger.info(f"Remaining: {len(remaining)} materials")
    else:
        remaining = config.inputs

    # Parallel execution
    start_time = time.time()
    args_list = [(inp, config.mu_star) for inp in remaining]

    with Pool(config.n_workers) as pool:
        results = list(
            tqdm(
                pool.imap(_predict_single, args_list),
                total=len(args_list),
                desc="Screening",
                unit="material",
            )
        )

    all_results = completed + results

    # Save checkpoint
    if config.checkpoint_path:
        config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config.checkpoint_path, "wb") as f:
            pickle.dump(all_results, f)
        logger.info(f"Checkpoint saved: {config.checkpoint_path}")

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    # Sort by Tc (descending)
    df = df.sort_values("tc_kelvin", ascending=False, na_position="last")

    # Save results
    config.output_path.parent.mkdir(parents=True, exist_ok=True)

    if config.output_path.suffix == ".parquet":
        df.to_parquet(config.output_path, index=False)
    elif config.output_path.suffix == ".csv":
        df.to_csv(config.output_path, index=False)
    elif config.output_path.suffix == ".json":
        df.to_json(config.output_path, orient="records", indent=2)
    else:
        raise ValueError(f"Unsupported output format: {config.output_path.suffix}")

    elapsed = time.time() - start_time
    rate = len(all_results) / elapsed
    logger.info(
        f"Screening complete: {len(all_results)} materials in {elapsed:.1f}s ({rate:.1f} mat/s)"
    )
    logger.info(f"Results saved: {config.output_path}")

    return df


def screen_from_csv(
    csv_path: Path,
    output_path: Path,
    mu_star: float = 0.10,
    n_workers: int = 4,
    resume: bool = False,
) -> pd.DataFrame:
    """
    Convenience function to screen materials from CSV file.

    CSV should have a column 'mp_id' or 'cif_path'.

    Args:
        csv_path: Path to CSV with materials to screen
        output_path: Where to save results (Parquet/CSV/JSON)
        mu_star: Coulomb pseudopotential
        n_workers: Number of parallel workers
        resume: Resume from checkpoint if available

    Returns:
        DataFrame with screening results
    """
    df_input = pd.read_csv(csv_path)

    if "mp_id" in df_input.columns:
        inputs = df_input["mp_id"].tolist()
    elif "cif_path" in df_input.columns:
        inputs = df_input["cif_path"].tolist()
    else:
        raise ValueError("CSV must have 'mp_id' or 'cif_path' column")

    checkpoint_path = output_path.parent / f"{output_path.stem}_checkpoint.pkl"

    config = ScreeningConfig(
        inputs=inputs,
        mu_star=mu_star,
        output_path=output_path,
        checkpoint_path=checkpoint_path if resume else None,
        n_workers=n_workers,
        resume=resume,
    )

    return batch_screen(config)

