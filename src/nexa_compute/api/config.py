from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

from nexa_compute.utils.secrets import get_secret_manager


class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Nexa Forge"

    # Security
    SECRET_KEY: str = "development_secret_key"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days

    # Database
    DATABASE_URL: str = "sqlite:///./var/nexa_forge.db"

    # Workers
    WORKER_TIMEOUT_SECONDS: int = 60

    # Storage
    STORAGE_BACKEND: str = "local"  # local, s3
    S3_BUCKET: Optional[str] = None

    # Rate limiting
    RATE_LIMIT_WINDOW_SECONDS: int = 60
    RATE_LIMIT_REQUESTS_PER_WINDOW: int = 120

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, extra="ignore")


@lru_cache()
def get_settings():
    settings = Settings()
    # Ensure secrets manager is initialized and required secrets are validated.
    get_secret_manager()
    return settings
