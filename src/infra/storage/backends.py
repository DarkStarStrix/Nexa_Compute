"""Artifact storage backends (S3, DO Spaces, local)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class StorageBackend:
    """Base class for storage backends."""
    
    def upload(self, local_path: Path, remote_key: str) -> str:
        """Upload file and return URI."""
        raise NotImplementedError
    
    def download(self, remote_key: str, local_path: Path) -> None:
        """Download file from storage."""
        raise NotImplementedError
    
    def exists(self, remote_key: str) -> bool:
        """Check if file exists."""
        raise NotImplementedError


class LocalStorage(StorageBackend):
    """Local filesystem storage (for testing)."""
    
    def __init__(self, base_dir: Path = Path("artifacts")):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def upload(self, local_path: Path, remote_key: str) -> str:
        """Copy to local artifacts directory."""
        dest = self.base_dir / remote_key
        dest.parent.mkdir(parents=True, exist_ok=True)
        
        import shutil
        shutil.copy2(local_path, dest)
        
        logger.info(f"Uploaded {local_path} to {dest}")
        return f"file://{dest.absolute()}"
    
    def download(self, remote_key: str, local_path: Path) -> None:
        """Copy from local artifacts directory."""
        src = self.base_dir / remote_key
        
        import shutil
        shutil.copy2(src, local_path)
        
        logger.info(f"Downloaded {src} to {local_path}")
    
    def exists(self, remote_key: str) -> bool:
        """Check if file exists."""
        return (self.base_dir / remote_key).exists()


class S3Storage(StorageBackend):
    """AWS S3 or DigitalOcean Spaces storage."""
    
    def __init__(
        self,
        bucket: str,
        access_key: str,
        secret_key: str,
        region: str = "us-east-1",
        endpoint_url: Optional[str] = None
    ):
        self.bucket = bucket
        self.region = region
        
        try:
            import boto3
            self.s3 = boto3.client(
                's3',
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name=region,
                endpoint_url=endpoint_url
            )
            logger.info(f"Initialized S3 storage: {bucket}")
        except ImportError:
            logger.error("boto3 not installed. Install with: pip install boto3")
            raise
    
    def upload(self, local_path: Path, remote_key: str) -> str:
        """Upload to S3/Spaces."""
        self.s3.upload_file(str(local_path), self.bucket, remote_key)
        
        uri = f"s3://{self.bucket}/{remote_key}"
        logger.info(f"Uploaded {local_path} to {uri}")
        return uri
    
    def download(self, remote_key: str, local_path: Path) -> None:
        """Download from S3/Spaces."""
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self.s3.download_file(self.bucket, remote_key, str(local_path))
        
        logger.info(f"Downloaded s3://{self.bucket}/{remote_key} to {local_path}")
    
    def exists(self, remote_key: str) -> bool:
        """Check if object exists."""
        try:
            self.s3.head_object(Bucket=self.bucket, Key=remote_key)
            return True
        except:
            return False


def get_storage_backend(backend_type: str = "local", **kwargs) -> StorageBackend:
    """Factory function to get storage backend."""
    if backend_type == "local":
        return LocalStorage(**kwargs)
    elif backend_type in ["s3", "do_spaces"]:
        return S3Storage(**kwargs)
    else:
        raise ValueError(f"Unknown storage backend: {backend_type}")


__all__ = ["StorageBackend", "LocalStorage", "S3Storage", "get_storage_backend"]
