"""
API Dependencies

Reusable dependencies for endpoint authorization and validation.
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from .auth import verify_token, get_user_by_id, oauth2_scheme
from .database import get_db
from .models.user import User

# OAuth2 scheme for token authentication
# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """
    Dependency to get the current authenticated user
    
    Usage:
        @app.get("/protected")
        def protected_endpoint(current_user: User = Depends(get_current_user)):
            return {"user": current_user.email}
    
    Args:
        token: JWT token from Authorization header
        db: Database session
    
    Returns:
        User object
    
    Raises:
        HTTPException 401: If token is invalid or user not found
    """
    token_data = verify_token(token, token_type="access")
    
    user = get_user_by_id(db, user_id=token_data.user_id)
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user",
        )
    
    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """
    Dependency to get current active user
    
    Usage:
        @app.get("/protected")
        def protected_endpoint(user: User = Depends(get_current_active_user)):
            return {"user": user.email}
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    return current_user


async def get_current_superuser(
    current_user: User = Depends(get_current_user),
) -> User:
    """
    Dependency to require superuser access
    
    Usage:
        @app.delete("/users/{user_id}")
        def delete_user(user_id: int, admin: User = Depends(get_current_superuser)):
            ...
    """
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    return current_user

