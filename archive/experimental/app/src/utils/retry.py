"""
Async retry utilities with exponential backoff and jitter.

Used for non-blocking audit logging with resilience against transient failures.
"""

import asyncio
import random
import logging
from typing import Callable, TypeVar, Optional, Tuple

logger = logging.getLogger(__name__)

T = TypeVar('T')


async def retry_async(
    func: Callable[[], T],
    *,
    attempts: int = 5,
    backoff: Tuple[float, float] = (0.25, 3.0),
    jitter_factor: float = 0.1,
    error_msg: str = "Retry operation"
) -> Optional[T]:
    """
    Retry an async function with exponential backoff and jitter.
    
    Strategy:
    - Initial delay: backoff[0] seconds
    - Maximum delay: backoff[1] seconds
    - Delay doubles on each retry (exponential)
    - Random jitter added to prevent thundering herd
    
    Args:
        func: Async callable to retry (no arguments)
        attempts: Maximum number of attempts (default: 5)
        backoff: (min_delay, max_delay) in seconds
        jitter_factor: Jitter as fraction of delay (0.1 = 10% jitter)
        error_msg: Context for error logging
    
    Returns:
        Result of func() if successful, None if all retries exhausted
    
    Example:
        >>> async def save_to_db():
        ...     return await db.insert(record)
        >>> 
        >>> result = await retry_async(
        ...     save_to_db,
        ...     attempts=5,
        ...     backoff=(0.25, 3.0)
        ... )
    
    Timing Example:
        Attempt 1: immediate
        Attempt 2: wait 0.25s + jitter
        Attempt 3: wait 0.50s + jitter
        Attempt 4: wait 1.00s + jitter
        Attempt 5: wait 2.00s + jitter
        Attempt 6: wait 3.00s + jitter (capped)
        
    Total wait time (worst case): ~7s for 5 retries
    """
    min_delay, max_delay = backoff
    
    for attempt in range(attempts):
        try:
            return await func()
        
        except asyncio.CancelledError:
            # Don't retry if task was cancelled
            logger.warning(f"{error_msg}: Cancelled on attempt {attempt + 1}/{attempts}")
            raise
        
        except Exception as e:
            is_final_attempt = (attempt == attempts - 1)
            
            if is_final_attempt:
                # Final attempt failed - log error and return None
                logger.error(
                    f"{error_msg}: All {attempts} attempts failed. Last error: {e}",
                    exc_info=True
                )
                return None
            
            # Calculate delay with exponential backoff
            base_delay = min(min_delay * (2 ** attempt), max_delay)
            jitter = random.uniform(0, base_delay * jitter_factor)
            total_delay = base_delay + jitter
            
            logger.warning(
                f"{error_msg}: Attempt {attempt + 1}/{attempts} failed with {e.__class__.__name__}: {e}. "
                f"Retrying in {total_delay:.2f}s..."
            )
            
            await asyncio.sleep(total_delay)
    
    # Should never reach here (loop either returns or raises)
    return None


async def retry_with_circuit_breaker(
    func: Callable[[], T],
    *,
    attempts: int = 3,
    backoff: Tuple[float, float] = (0.1, 1.0),
    circuit_threshold: int = 5,
    circuit_timeout: float = 60.0
) -> Optional[T]:
    """
    Retry with circuit breaker pattern (future enhancement).
    
    Not implemented in v1 - simple retry_async is sufficient for audit logging.
    
    Would track:
    - Recent failure rate
    - Open circuit if failure rate > threshold
    - Automatic reset after timeout
    """
    # TODO: Implement circuit breaker if audit logging failures become systematic
    return await retry_async(func, attempts=attempts, backoff=backoff)
