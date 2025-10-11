"""
Reproducibility Utilities (NeurIPS/ICLR Compliance)

Comprehensive random seed management for PyTorch, NumPy, and DataLoader.

Protocol Contract:
- Set ALL sources of randomness (Python, NumPy, PyTorch, CUDA, DataLoader workers)
- Use deterministic algorithms when available
- Document any known sources of non-determinism
"""

import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Optional, Any
import logging
import os

logger = logging.getLogger(__name__)


def set_all_seeds(seed: int, deterministic: bool = True, warn_tf32: bool = True) -> None:
    """
    Set all random seeds for reproducibility.
    
    Sets seeds for:
    - Python random module
    - NumPy
    - PyTorch (CPU and CUDA)
    - CUDA (deterministic algorithms, cuDNN)
    
    Args:
        seed: Random seed (integer)
        deterministic: If True, enable deterministic algorithms (may impact performance)
        warn_tf32: If True, warn about TF32 precision (can cause non-determinism)
    
    Example:
        >>> from src.utils.reproducibility import set_all_seeds
        >>> set_all_seeds(42, deterministic=True)
        >>> # All subsequent random operations will be reproducible
    """
    # Python random module
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch CPU
    torch.manual_seed(seed)
    
    # PyTorch CUDA (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
        
        if deterministic:
            # Enable deterministic operations (may slow down)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
            logger.info(
                "CUDA deterministic mode enabled "
                "(cudnn.deterministic=True, cudnn.benchmark=False)"
            )
        else:
            logger.warning(
                "CUDA deterministic mode DISABLED. "
                "Results may vary across runs!"
            )
    
    # Enable deterministic algorithms in PyTorch
    if deterministic:
        try:
            # PyTorch 1.8+
            torch.use_deterministic_algorithms(True)
            logger.info("PyTorch deterministic algorithms enabled")
        except AttributeError:
            logger.warning(
                "torch.use_deterministic_algorithms() not available "
                "(requires PyTorch 1.8+)"
            )
    
    # TF32 precision (Ampere GPUs)
    if warn_tf32 and torch.cuda.is_available():
        # TF32 can introduce non-determinism in matrix multiplications
        if torch.cuda.get_device_capability()[0] >= 8:  # Ampere or newer
            if torch.backends.cuda.matmul.allow_tf32:
                logger.warning(
                    "TF32 precision enabled (allow_tf32=True). "
                    "This may cause slight non-determinism. "
                    "To disable: torch.backends.cuda.matmul.allow_tf32 = False"
                )
    
    # Set environment variables for additional determinism
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    logger.info(
        f"All random seeds set to {seed} "
        f"(deterministic={deterministic})"
    )


def seed_worker(worker_id: int) -> None:
    """
    Seed DataLoader worker for reproducibility.
    
    Must be passed to DataLoader via worker_init_fn parameter.
    
    Args:
        worker_id: Worker ID (automatically provided by DataLoader)
    
    Example:
        >>> from torch.utils.data import DataLoader
        >>> from src.utils.reproducibility import seed_worker
        >>> 
        >>> loader = DataLoader(
        ...     dataset,
        ...     batch_size=32,
        ...     worker_init_fn=seed_worker,  # Seed each worker
        ...     generator=torch.Generator().manual_seed(42)
        ... )
    """
    # Get worker seed from PyTorch
    worker_seed = torch.initial_seed() % 2**32
    
    # Seed worker's random number generators
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
    logger.debug(f"DataLoader worker {worker_id} seeded with {worker_seed}")


def create_reproducible_dataloader(
    dataset: Any,
    batch_size: int,
    seed: int = 42,
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs
) -> DataLoader:
    """
    Create DataLoader with reproducibility guarantees.
    
    Args:
        dataset: PyTorch Dataset
        batch_size: Batch size
        seed: Random seed for shuffling
        shuffle: Whether to shuffle data
        num_workers: Number of workers for data loading
        **kwargs: Additional DataLoader arguments
    
    Returns:
        DataLoader with seeded generator and worker_init_fn
    
    Example:
        >>> loader = create_reproducible_dataloader(
        ...     dataset, batch_size=32, seed=42, shuffle=True, num_workers=4
        ... )
        >>> # Guaranteed to produce same batches across runs with same seed
    """
    # Create seeded generator
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
        **kwargs
    )


def verify_reproducibility(
    fn: callable,
    seed: int = 42,
    n_runs: int = 2,
    rtol: float = 1e-5,
    atol: float = 1e-8
) -> bool:
    """
    Verify that a function produces reproducible results.
    
    Args:
        fn: Function to test (should return a numeric value or array)
        seed: Random seed
        n_runs: Number of runs to compare (default: 2)
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison
    
    Returns:
        True if all runs produce identical results
    
    Example:
        >>> def my_random_function():
        ...     return torch.randn(100).sum().item()
        >>> 
        >>> is_reproducible = verify_reproducibility(my_random_function, seed=42)
        >>> print(f"Reproducible: {is_reproducible}")
    """
    results = []
    
    for i in range(n_runs):
        # Reset all seeds
        set_all_seeds(seed, deterministic=True)
        
        # Run function
        result = fn()
        results.append(result)
        
        logger.debug(f"Run {i+1}/{n_runs}: result={result}")
    
    # Compare all results
    reference = results[0]
    
    for i, result in enumerate(results[1:], start=2):
        if isinstance(reference, (int, float)):
            # Scalar comparison
            if not np.isclose(reference, result, rtol=rtol, atol=atol):
                logger.error(
                    f"NON-REPRODUCIBLE: Run 1={reference}, Run {i}={result}"
                )
                return False
        elif isinstance(reference, np.ndarray):
            # Array comparison
            if not np.allclose(reference, result, rtol=rtol, atol=atol):
                max_diff = np.abs(reference - result).max()
                logger.error(
                    f"NON-REPRODUCIBLE: Run 1 vs Run {i}, max_diff={max_diff}"
                )
                return False
        elif isinstance(reference, torch.Tensor):
            # Tensor comparison
            if not torch.allclose(reference, result, rtol=rtol, atol=atol):
                max_diff = (reference - result).abs().max().item()
                logger.error(
                    f"NON-REPRODUCIBLE: Run 1 vs Run {i}, max_diff={max_diff}"
                )
                return False
        else:
            # Generic comparison
            if reference != result:
                logger.error(
                    f"NON-REPRODUCIBLE: Run 1={reference}, Run {i}={result}"
                )
                return False
    
    logger.info(f"✅ REPRODUCIBLE: {n_runs} runs produced identical results")
    return True


def get_random_state_snapshot() -> dict:
    """
    Capture current random state for all RNG sources.
    
    Returns:
        Dictionary with RNG states
    
    Example:
        >>> snapshot = get_random_state_snapshot()
        >>> # ... do some random operations ...
        >>> restore_random_state(snapshot)
        >>> # RNG state restored
    """
    snapshot = {
        'python_random': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    
    if torch.cuda.is_available():
        snapshot['torch_cuda'] = torch.cuda.get_rng_state_all()
    
    return snapshot


def restore_random_state(snapshot: dict) -> None:
    """
    Restore random state from snapshot.
    
    Args:
        snapshot: State dictionary from get_random_state_snapshot()
    
    Example:
        >>> snapshot = get_random_state_snapshot()
        >>> # ... do some random operations ...
        >>> restore_random_state(snapshot)
        >>> # RNG state restored to snapshot
    """
    random.setstate(snapshot['python_random'])
    np.random.set_state(snapshot['numpy'])
    torch.set_rng_state(snapshot['torch'])
    
    if 'torch_cuda' in snapshot and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(snapshot['torch_cuda'])
    
    logger.debug("Random state restored from snapshot")


# Legacy compatibility
def set_seed(seed: int) -> None:
    """
    Legacy function for backward compatibility.
    
    Use set_all_seeds() for new code (more comprehensive).
    """
    logger.warning(
        "set_seed() is deprecated. Use set_all_seeds() for full reproducibility."
    )
    set_all_seeds(seed, deterministic=True)

