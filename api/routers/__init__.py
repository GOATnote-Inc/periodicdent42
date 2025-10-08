"""
API Routers

Collection of FastAPI routers for different API endpoints.
"""

from .auth import router as auth_router

__all__ = ["auth_router"]

