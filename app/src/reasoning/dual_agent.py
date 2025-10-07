"""
Dual-model AI agent using Gemini 2.5 Flash + Pro in parallel.

Flash provides instant preliminary responses (<2s).
Pro provides verified, high-accuracy responses (10-30s).

Enhanced with:
- Configurable timeouts (FLASH_TIMEOUT_S, PRO_TIMEOUT_S)
- Proper cancellation on timeout/error
- Structured error types for SSE streaming
- Metrics collection for observability
"""

import asyncio
import time
from typing import Dict, Any, Tuple
import logging

from src.services.vertex import get_flash_model, get_pro_model
from src.utils.settings import settings
from src.utils.metrics import (
    observe_latency,
    increment_timeout,
    increment_error,
    increment_cancellation
)

logger = logging.getLogger(__name__)


class DualModelAgent:
    """
    Parallel Fast + Accurate AI reasoning with Gemini 2.5 Flash and Pro.
    
    Moat: INTERPRETABILITY + TIME - Instant feedback with verified accuracy.
    """
    
    def __init__(self, project_id: str, location: str):
        self.project_id = project_id
        self.location = location
        
        # Get model instances
        self.flash_model = get_flash_model()
        self.pro_model = get_pro_model()
        
        logger.info("DualModelAgent initialized")
    
    def start_query_tasks(
        self,
        prompt: str,
        context: Dict[str, Any],
    ) -> Tuple[asyncio.Task, asyncio.Task]:
        """
        Launch Flash and Pro queries concurrently with timeouts.
        
        Each task is wrapped with asyncio.wait_for() to enforce timeout.
        Timeouts are configurable via environment variables:
        - FLASH_TIMEOUT_S (default: 5s)
        - PRO_TIMEOUT_S (default: 45s)
        
        Returns:
            (flash_task, pro_task) - Both wrapped with timeout enforcement
        """
        flash_timeout = settings.FLASH_TIMEOUT_S
        pro_timeout = settings.PRO_TIMEOUT_S
        
        # Wrap each query with timeout
        async def flash_with_timeout():
            try:
                return await asyncio.wait_for(
                    self._query_flash(prompt, context),
                    timeout=flash_timeout
                )
            except asyncio.TimeoutError:
                logger.warning(f"Flash timeout after {flash_timeout}s")
                increment_timeout("flash")
                raise

        async def pro_with_timeout():
            try:
                return await asyncio.wait_for(
                    self._query_pro(prompt, context),
                    timeout=pro_timeout
                )
            except asyncio.TimeoutError:
                logger.warning(f"Pro timeout after {pro_timeout}s")
                increment_timeout("pro")
                raise

        flash_task = asyncio.create_task(flash_with_timeout())
        pro_task = asyncio.create_task(pro_with_timeout())
        
        return flash_task, pro_task

    async def query_parallel(
        self,
        prompt: str,
        context: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Execute both models in parallel.
        
        Args:
            prompt: User query or task
            context: Additional context (domain, constraints, etc.)
        
        Returns:
            (flash_response, pro_response) - Flash completes first, Pro follows
        """
        # Launch both models simultaneously
        flash_task, pro_task = self.start_query_tasks(prompt, context)
        
        # Await Flash first for immediate UI update
        flash_response = await flash_task
        
        # Pro completes in background
        pro_response = await pro_task
        
        return flash_response, pro_response
    
    async def _query_flash(self, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fast preliminary response using Gemini 2.5 Flash.
        
        Target latency: <2 seconds
        """
        start_time = time.time()
        
        try:
            # Enhanced prompt with context
            enhanced_prompt = self._build_flash_prompt(prompt, context)
            
            # Generate with Flash settings (speed optimized)
            response = await asyncio.to_thread(
                self.flash_model.generate_content,
                enhanced_prompt,
                generation_config={
                    "temperature": 0.7,
                    "max_output_tokens": 1024,
                    "top_p": 0.9,
                }
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Record metrics
            observe_latency("flash", latency_ms)
            
            # Try to get usage metadata if available
            usage = {}
            try:
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    usage = {
                        "input_tokens": response.usage_metadata.prompt_token_count,
                        "output_tokens": response.usage_metadata.candidates_token_count,
                    }
            except Exception:
                pass
            
            return {
                "model": "gemini-2.5-flash",
                "content": response.text,
                "latency_ms": round(latency_ms, 2),
                "is_preliminary": True,
                "confidence": "medium",
                "usage": usage
            }
        
        except Exception as e:
            logger.error(f"Flash model error: {e}")
            increment_error(e.__class__.__name__)
            return {
                "model": "gemini-2.5-flash",
                "content": f"Error: {str(e)}",
                "latency_ms": (time.time() - start_time) * 1000,
                "is_preliminary": True,
                "confidence": "error",
                "error": str(e),
                "error_class": e.__class__.__name__
            }
    
    async def _query_pro(self, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Accurate, verified response using Gemini 2.5 Pro.
        
        Target latency: 10-30 seconds (acceptable for verification)
        """
        start_time = time.time()
        
        try:
            # Enhanced prompt with scientific rigor
            enhanced_prompt = self._build_pro_prompt(prompt, context)
            
            # Generate with Pro settings (accuracy optimized)
            response = await asyncio.to_thread(
                self.pro_model.generate_content,
                enhanced_prompt,
                generation_config={
                    "temperature": 0.2,  # Lower temp for consistency
                    "max_output_tokens": 8192,  # More detailed
                    "top_p": 0.95,
                }
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Record metrics
            observe_latency("pro", latency_ms)
            
            # Parse reasoning steps
            reasoning_steps = self._extract_reasoning_steps(response.text)
            
            # Try to get usage metadata if available
            usage = {}
            try:
                if hasattr(response, 'usage_metadata'):
                    usage = {
                        "input_tokens": response.usage_metadata.prompt_token_count,
                        "output_tokens": response.usage_metadata.candidates_token_count,
                    }
            except Exception:
                pass
            
            return {
                "model": "gemini-2.5-pro",
                "content": response.text,
                "latency_ms": round(latency_ms, 2),
                "is_preliminary": False,
                "confidence": "high",
                "reasoning_steps": reasoning_steps,
                "usage": usage
            }
        
        except Exception as e:
            logger.error(f"Pro model error: {e}")
            increment_error(e.__class__.__name__)
            return {
                "model": "gemini-2.5-pro",
                "content": f"Error: {str(e)}",
                "latency_ms": (time.time() - start_time) * 1000,
                "is_preliminary": False,
                "confidence": "error",
                "error": str(e),
                "error_class": e.__class__.__name__
            }
    
    def _build_flash_prompt(self, prompt: str, context: Dict[str, Any]) -> str:
        """Build prompt optimized for Flash (concise, actionable)."""
        context_str = "\n".join([f"{k}: {v}" for k, v in context.items()])
        
        return f"""Context:
{context_str}

Task: {prompt}

Provide a QUICK preliminary analysis suitable for immediate user feedback.
Be concise and actionable. Note: This is a fast preview; a more detailed analysis will follow.
"""
    
    def _build_pro_prompt(self, prompt: str, context: Dict[str, Any]) -> str:
        """Build prompt optimized for Pro (comprehensive, rigorous)."""
        context_str = "\n".join([f"{k}: {v}" for k, v in context.items()])
        
        return f"""Context:
{context_str}

Task: {prompt}

Provide a COMPREHENSIVE, scientifically rigorous analysis with:
1. Step-by-step reasoning
2. Confidence intervals and uncertainty quantification (where applicable)
3. Alternative approaches considered
4. Key assumptions and limitations
5. Expected outcomes with rationale

This is the FINAL verified response for scientific use.
"""
    
    def _extract_reasoning_steps(self, text: str) -> list:
        """
        Extract reasoning steps from Pro response for audit trail.
        
        Simple implementation: split by paragraphs or numbered lists.
        """
        # Split by double newlines (paragraphs)
        steps = [s.strip() for s in text.split("\n\n") if s.strip()]
        
        # Limit to first 5 steps for brevity
        return steps[:5]

