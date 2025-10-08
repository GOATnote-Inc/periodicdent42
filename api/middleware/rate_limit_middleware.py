"""
Rate Limit Middleware

Middleware to attach user information to request state for rate limiting.
"""

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from jose import jwt, JWTError
import os

# Import JWT configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "CHANGE_THIS_SECRET_KEY_IN_PRODUCTION")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware to extract user information for rate limiting.
    
    This middleware runs before rate limiting to identify authenticated users
    and apply appropriate rate limits.
    """
    
    async def dispatch(self, request: Request, call_next):
        """
        Extract user information from JWT token if present.
        
        Sets request.state.user if token is valid.
        """
        # Try to extract token from Authorization header
        auth_header = request.headers.get("Authorization")
        
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            
            try:
                # Decode token (without full validation - just for rate limiting)
                payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
                
                # Create a simple user object with just the ID
                class SimpleUser:
                    def __init__(self, user_id):
                        self.id = user_id
                
                user_id = payload.get("user_id")
                if user_id:
                    request.state.user = SimpleUser(user_id)
            
            except JWTError:
                # Invalid token - treat as anonymous
                pass
        
        # Continue with request
        response = await call_next(request)
        return response

