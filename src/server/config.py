"""Environment configuration loader for Nexa API."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env file from project root
ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = ROOT / ".env"

if ENV_FILE.exists():
    load_dotenv(ENV_FILE)


class Config:
    """Configuration from environment variables."""
    
    # API Keys
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    HUGGINGFACE_TOKEN: str = os.getenv("HUGGINGFACE_TOKEN", "")
    WANDB_API_KEY: str = os.getenv("WANDB_API_KEY", "")
    
    # Cloud Storage
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    S3_BUCKET: str = os.getenv("S3_BUCKET", "nexa-artifacts")
    S3_REGION: str = os.getenv("S3_REGION", "us-east-1")
    
    # DigitalOcean Spaces (S3-compatible)
    DO_SPACES_KEY: str = os.getenv("DO_SPACES_KEY", "")
    DO_SPACES_SECRET: str = os.getenv("DO_SPACES_SECRET", "")
    DO_SPACES_BUCKET: str = os.getenv("DO_SPACES_BUCKET", "nexa-artifacts")
    DO_SPACES_REGION: str = os.getenv("DO_SPACES_REGION", "nyc3")
    DO_SPACES_ENDPOINT: str = os.getenv("DO_SPACES_ENDPOINT", f"https://{DO_SPACES_REGION}.digitaloceanspaces.com")
    
    # GPU Providers
    PRIME_INTELLECT_API_KEY: str = os.getenv("PRIME_INTELLECT_API_KEY", "")
    PRIME_INTELLECT_API_URL: str = os.getenv("PRIME_INTELLECT_API_URL", "https://api.primeintellect.ai")
    
    # VPS Configuration
    VPS_HOST: str = os.getenv("VPS_HOST", "localhost")
    VPS_API_URL: str = os.getenv("VPS_API_URL", "http://localhost:8000")
    
    # SSH Configuration
    SSH_KEY_PATH: Optional[Path] = Path(os.getenv("SSH_KEY_PATH", "~/.ssh/id_rsa")).expanduser() if os.getenv("SSH_KEY_PATH") else None
    
    # Repository
    REPO_URL: str = os.getenv("REPO_URL", "https://github.com/DarkStarStrix/Nexa_Compute.git")
    
    @classmethod
    def validate(cls) -> list[str]:
        """Validate required configuration."""
        missing = []
        
        # Check critical keys
        if not cls.OPENROUTER_API_KEY:
            missing.append("OPENROUTER_API_KEY")
        
        # Check storage (either AWS or DO Spaces)
        has_aws = cls.AWS_ACCESS_KEY_ID and cls.AWS_SECRET_ACCESS_KEY
        has_do = cls.DO_SPACES_KEY and cls.DO_SPACES_SECRET
        
        if not (has_aws or has_do):
            missing.append("AWS credentials or DO_SPACES credentials")
        
        return missing
    
    @classmethod
    def get_storage_backend(cls) -> str:
        """Determine which storage backend to use."""
        if cls.DO_SPACES_KEY and cls.DO_SPACES_SECRET:
            return "do_spaces"
        elif cls.AWS_ACCESS_KEY_ID and cls.AWS_SECRET_ACCESS_KEY:
            return "s3"
        else:
            return "local"


__all__ = ["Config"]
