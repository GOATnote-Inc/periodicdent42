"""
Application settings with environment variable loading and Secret Manager fallback.
"""

from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings with GCP integration."""
    
    # Core GCP settings
    PROJECT_ID: str = "periodicdent42"
    LOCATION: str = "us-central1"
    ENVIRONMENT: str = "development"
    
    # Vertex AI models
    GEMINI_FLASH_MODEL: str = "gemini-2.5-flash"
    GEMINI_PRO_MODEL: str = "gemini-2.5-pro"
    
    # Database (Cloud SQL)
    GCP_SQL_INSTANCE: Optional[str] = None  # Format: project:region:instance
    DB_USER: str = "ard_user"
    DB_PASSWORD: Optional[str] = None
    DB_NAME: str = "ard_intelligence"
    DB_HOST: str = "localhost"  # For local development
    DB_PORT: int = 5432
    
    # Cloud Storage
    GCS_BUCKET: Optional[str] = None  # Set by Terraform or env
    
    # Server settings
    PORT: int = 8080
    LOG_LEVEL: str = "INFO"
    
    # Feature flags
    ENABLE_METRICS: bool = True
    ENABLE_TRACING: bool = True
    
    # Security settings
    API_KEY: Optional[str] = None  # API key for authentication (set in Secret Manager)
    ALLOWED_ORIGINS: str = ""  # Comma-separated list of allowed CORS origins
    ENABLE_AUTH: bool = False  # Enable in production to require API key auth
    RATE_LIMIT_PER_MINUTE: int = 60  # Max requests per IP per minute
    
    class Config:
        env_file = ".env"
        case_sensitive = True


def get_secret_from_manager(secret_id: str, project_id: str) -> Optional[str]:
    """
    Fetch secret from Google Secret Manager.
    
    Only called when running on GCP and secret not in env.
    """
    try:
        from google.cloud import secretmanager
        
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
        
        response = client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8")
    except Exception as e:
        print(f"Warning: Could not fetch secret {secret_id}: {e}")
        return None


def load_settings() -> Settings:
    """
    Load settings with Secret Manager fallback for production.
    """
    settings = Settings()
    
    # If running on GCP (detected by GOOGLE_CLOUD_PROJECT env var)
    if os.getenv("GOOGLE_CLOUD_PROJECT"):
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT", settings.PROJECT_ID)
        
        # Fetch missing secrets from Secret Manager
        if not settings.DB_PASSWORD:
            settings.DB_PASSWORD = get_secret_from_manager("db-password", project_id)
        
        if not settings.GCS_BUCKET:
            settings.GCS_BUCKET = get_secret_from_manager("gcs-bucket", project_id)
        
        if not settings.API_KEY and settings.ENABLE_AUTH:
            settings.API_KEY = get_secret_from_manager("api-key", project_id)
    
    return settings


# Singleton instance
settings = load_settings()

