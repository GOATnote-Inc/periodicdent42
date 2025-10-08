"""
Authentication Logic

JWT token generation, validation, and user authentication.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
import os

from .models.user import User
from .schemas.auth import TokenData
from .database import get_db

# JWT Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "CHANGE_THIS_SECRET_KEY_IN_PRODUCTION")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))  # 24 hours
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))  # 7 days

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token
    
    Args:
        data: Dictionary containing token payload
        expires_delta: Optional custom expiration time
    
    Returns:
        Encoded JWT token string
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "type": "access"
    })
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def create_refresh_token(data: dict) -> str:
    """
    Create a JWT refresh token
    
    Args:
        data: Dictionary containing token payload
    
    Returns:
        Encoded JWT refresh token string
    """
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "type": "refresh"
    })
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str, token_type: str = "access") -> TokenData:
    """
    Verify and decode a JWT token
    
    Args:
        token: JWT token string
        token_type: Expected token type ("access" or "refresh")
    
    Returns:
        TokenData with decoded payload
    
    Raises:
        HTTPException: If token is invalid or expired
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        # Verify token type
        if payload.get("type") != token_type:
            raise credentials_exception
        
        email: str = payload.get("sub")
        user_id: int = payload.get("user_id")
        
        if email is None or user_id is None:
            raise credentials_exception
        
        token_data = TokenData(email=email, user_id=user_id)
        return token_data
    
    except JWTError:
        raise credentials_exception


def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
    """
    Authenticate a user by email and password
    
    Args:
        db: Database session
        email: User email
        password: Plain text password
    
    Returns:
        User object if authentication successful, None otherwise
    """
    user = db.query(User).filter(User.email == email).first()
    
    if not user:
        return None
    
    if not user.check_password(password):
        return None
    
    if not user.is_active:
        return None
    
    return user


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Get user by email"""
    return db.query(User).filter(User.email == email).first()


def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
    """Get user by ID"""
    return db.query(User).filter(User.id == user_id).first()


def create_user(db: Session, email: str, password: str) -> User:
    """
    Create a new user
    
    Args:
        db: Database session
        email: User email
        password: Plain text password (will be hashed)
    
    Returns:
        Created User object
    """
    hashed_password = User.get_password_hash(password)
    db_user = User(email=email, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

