"""
Dual-model AI agent using Gemini 2.5 Flash + Pro in parallel.

Flash provides instant preliminary responses (<2s).
Pro provides verified, high-accuracy responses (10-30s).
"""

import asyncio
import time
from typing import Dict, Any, Tuple
import logging

from src.services.vertex import get_flash_model, get_pro_model

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
        flash_task = asyncio.create_task(self._query_flash(prompt, context))
        pro_task = asyncio.create_task(self._query_pro(prompt, context))
        
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
            
            return {
                "model": "gemini-2.5-flash",
                "content": response.text,
                "latency_ms": round(latency_ms, 2),
                "is_preliminary": True,
                "confidence": "medium",
                "usage": {
                    "input_tokens": response.usage_metadata.prompt_token_count,
                    "output_tokens": response.usage_metadata.candidates_token_count,
                }
            }
        
        except Exception as e:
            logger.error(f"Flash model error: {e}")
            return {
                "model": "gemini-2.5-flash",
                "content": f"Error: {str(e)}",
                "latency_ms": (time.time() - start_time) * 1000,
                "is_preliminary": True,
                "confidence": "error",
                "error": str(e)
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
            
            # Parse reasoning steps
            reasoning_steps = self._extract_reasoning_steps(response.text)
            
            return {
                "model": "gemini-2.5-pro",
                "content": response.text,
                "latency_ms": round(latency_ms, 2),
                "is_preliminary": False,
                "confidence": "high",
                "reasoning_steps": reasoning_steps,
                "usage": {
                    "input_tokens": response.usage_metadata.prompt_token_count,
                    "output_tokens": response.usage_metadata.candidates_token_count,
                }
            }
        
        except Exception as e:
            logger.error(f"Pro model error: {e}")
            return {
                "model": "gemini-2.5-pro",
                "content": f"Error: {str(e)}",
                "latency_ms": (time.time() - start_time) * 1000,
                "is_preliminary": False,
                "confidence": "error",
                "error": str(e)
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

