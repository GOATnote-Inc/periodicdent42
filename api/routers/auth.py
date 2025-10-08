"""
Authentication Router

API endpoints for user authentication and management.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from ..auth import (
    authenticate_user,
    create_access_token,
    create_refresh_token,
    verify_token,
    create_user,
    get_user_by_email,
)
from ..database import get_db
from ..dependencies import get_current_user
from ..schemas.auth import (
    UserCreate,
    UserResponse,
    Token,
    TokenRefresh,
    UserLogin,
)
from ..models.user import User

router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user_create: UserCreate, db: Session = Depends(get_db)):
    """
    Register a new user
    
    - **email**: Valid email address
    - **password**: Must be at least 8 characters with 1 number and 1 special character
    
    Returns the created user (without password)
    """
    # Check if user already exists
    existing_user = get_user_by_email(db, email=user_create.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create user
    db_user = create_user(
        db,
        email=user_create.email,
        password=user_create.password
    )
    
    return db_user


@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """
    Login with email and password
    
    Returns access token and refresh token
    
    - **username**: User email (OAuth2 spec uses 'username' field)
    - **password**: User password
    """
    # Authenticate user
    user = authenticate_user(db, email=form_data.username, password=form_data.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create tokens
    access_token = create_access_token(
        data={"sub": user.email, "user_id": user.id}
    )
    refresh_token = create_refresh_token(
        data={"sub": user.email, "user_id": user.id}
    )
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": 1440 * 60  # 24 hours in seconds
    }


@router.post("/login/json", response_model=Token)
async def login_json(user_login: UserLogin, db: Session = Depends(get_db)):
    """
    Login with JSON body (alternative to form-data)
    
    - **email**: User email
    - **password**: User password
    """
    user = authenticate_user(db, email=user_login.email, password=user_login.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(
        data={"sub": user.email, "user_id": user.id}
    )
    refresh_token = create_refresh_token(
        data={"sub": user.email, "user_id": user.id}
    )
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": 1440 * 60
    }


@router.post("/refresh", response_model=Token)
async def refresh_token_endpoint(
    token_refresh: TokenRefresh,
    db: Session = Depends(get_db)
):
    """
    Refresh access token using refresh token
    
    - **refresh_token**: Valid refresh token
    
    Returns new access token and refresh token
    """
    # Verify refresh token
    token_data = verify_token(token_refresh.refresh_token, token_type="refresh")
    
    # Get user
    from ..auth import get_user_by_id
    user = get_user_by_id(db, user_id=token_data.user_id)
    
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
    
    # Create new tokens
    access_token = create_access_token(
        data={"sub": user.email, "user_id": user.id}
    )
    refresh_token = create_refresh_token(
        data={"sub": user.email, "user_id": user.id}
    )
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": 1440 * 60
    }


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """
    Get current user information
    
    Requires authentication
    """
    return current_user


@router.post("/logout")
async def logout(current_user: User = Depends(get_current_user)):
    """
    Logout (client should discard tokens)
    
    Note: JWTs are stateless, so true logout requires token blacklisting
    or short expiration times. For now, clients should discard the token.
    """
    return {"message": "Successfully logged out. Please discard your tokens."}

