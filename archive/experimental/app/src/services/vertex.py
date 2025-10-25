"""
Vertex AI service wrapper for Gemini models.
"""

import vertexai
from vertexai.preview.generative_models import GenerativeModel
from typing import Optional
import logging

logger = logging.getLogger(__name__)

_initialized = False


def init_vertex(project_id: str, location: str) -> None:
    """
    Initialize Vertex AI SDK.
    
    Should be called once at app startup.
    """
    global _initialized
    
    if _initialized:
        logger.debug("Vertex AI already initialized")
        return
    
    try:
        vertexai.init(project=project_id, location=location)
        _initialized = True
        logger.info(f"Vertex AI initialized: project={project_id}, location={location}")
    except Exception as e:
        logger.error(f"Failed to initialize Vertex AI: {e}")
        raise


def get_flash_model(model_name: str = "gemini-2.5-flash") -> GenerativeModel:
    """
    Get Gemini 2.5 Flash model instance.
    
    Fast, cost-effective model for preliminary responses.
    """
    if not _initialized:
        raise RuntimeError("Vertex AI not initialized. Call init_vertex() first.")
    
    return GenerativeModel(model_name)


def get_pro_model(model_name: str = "gemini-2.5-pro") -> GenerativeModel:
    """
    Get Gemini 2.5 Pro model instance.
    
    High-accuracy model for verified responses.
    """
    if not _initialized:
        raise RuntimeError("Vertex AI not initialized. Call init_vertex() first.")
    
    return GenerativeModel(model_name)


def is_initialized() -> bool:
    """Check if Vertex AI has been initialized."""
    return _initialized

