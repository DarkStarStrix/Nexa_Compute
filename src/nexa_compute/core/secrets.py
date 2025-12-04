"""Secrets management helpers supporting multiple backends."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional

LOGGER = logging.getLogger(__name__)

try:  # Optional dependency
    import boto3  # type: ignore
except ImportError:  # pragma: no cover
    boto3 = None  # type: ignore

try:  # Optional dependency
    import hvac  # type: ignore
except ImportError:  # pragma: no cover
    hvac = None  # type: ignore


class SecretResolutionError(RuntimeError):
    """Raised when a secret cannot be resolved."""


@dataclass
class SecretsConfig:
    backend: str = "env"
    env_prefix: str = ""
    aws_region: Optional[str] = None
    aws_prefix: str = ""
    vault_addr: Optional[str] = None
    vault_token: Optional[str] = None
    vault_mount_path: str = "secret/data"
    required_secrets: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_env(cls) -> "SecretsConfig":
        required = tuple(
            filter(
                None,
                (os.getenv("NEXA_REQUIRED_SECRETS", "") or "")
                .replace(" ", "")
                .split(","),
            )
        )
        return cls(
            backend=os.getenv("NEXA_SECRETS_BACKEND", "env").lower(),
            env_prefix=os.getenv("NEXA_SECRETS_ENV_PREFIX", ""),
            aws_region=os.getenv("NEXA_SECRETS_AWS_REGION"),
            aws_prefix=os.getenv("NEXA_SECRETS_AWS_PREFIX", ""),
            vault_addr=os.getenv("VAULT_ADDR"),
            vault_token=os.getenv("VAULT_TOKEN"),
            vault_mount_path=os.getenv("NEXA_SECRETS_VAULT_MOUNT", "secret/data"),
            required_secrets=required,
        )


class SecretManager:
    """Resolve secrets from env vars, AWS Secrets Manager, or Hashicorp Vault."""

    def __init__(self, config: SecretsConfig) -> None:
        self.config = config
        self._cache: Dict[str, str] = {}

    def get(self, name: str, *, required: bool = True, default: Optional[str] = None) -> Optional[str]:
        if name in self._cache:
            return self._cache[name]

        resolver = {
            "env": self._from_env,
            "aws": self._from_aws,
            "vault": self._from_vault,
        }.get(self.config.backend)

        if resolver is None:
            raise SecretResolutionError(f"Unsupported secrets backend: {self.config.backend}")

        value = resolver(name)
        if value is None:
            if required:
                raise SecretResolutionError(f"Secret '{name}' not found in backend '{self.config.backend}'.")
            return default

        self._cache[name] = value
        return value

    def load_many(self, mapping: Mapping[str, bool]) -> Dict[str, Optional[str]]:
        """Load multiple secrets, returning a dictionary."""
        return {key: self.get(key, required=required) for key, required in mapping.items()}

    def validate_required(self, required: Optional[Iterable[str]] = None) -> None:
        """Ensure all required secrets are available."""
        targets = tuple(required or self.config.required_secrets)
        for name in targets:
            self.get(name, required=True)

    def refresh(self, name: str) -> Optional[str]:
        """Refresh a cached secret."""
        self._cache.pop(name, None)
        return self.get(name, required=False)

    def _from_env(self, name: str) -> Optional[str]:
        env_name = f"{self.config.env_prefix}{name}".upper()
        return os.getenv(env_name) or os.getenv(name)

    def _from_aws(self, name: str) -> Optional[str]:
        if boto3 is None:
            raise SecretResolutionError("boto3 is not installed; cannot use AWS secrets backend.")
        region = self.config.aws_region or os.getenv("AWS_REGION")
        if not region:
            raise SecretResolutionError("AWS region not configured for secrets backend.")

        client = boto3.client("secretsmanager", region_name=region)  # type: ignore[var-annotated]
        secret_id = f"{self.config.aws_prefix}{name}"
        try:
            response = client.get_secret_value(SecretId=secret_id)
        except Exception as exc:  # pragma: no cover - boto errors
            raise SecretResolutionError(f"Unable to fetch AWS secret '{secret_id}': {exc}") from exc

        secret = response.get("SecretString")
        if secret and secret.startswith("{"):
            try:
                payload = json.loads(secret)
                return payload.get(name) or payload.get("value")
            except json.JSONDecodeError:
                return secret
        return secret

    def _from_vault(self, name: str) -> Optional[str]:
        if hvac is None:
            raise SecretResolutionError("hvac is not installed; cannot use Vault backend.")
        if not self.config.vault_addr or not self.config.vault_token:
            raise SecretResolutionError("Vault address/token missing.")

        client = hvac.Client(url=self.config.vault_addr, token=self.config.vault_token)  # type: ignore[var-annotated]
        secret_path = f"{self.config.vault_mount_path.rstrip('/')}/{name}"
        try:
            response = client.secrets.kv.v2.read_secret_version(path=secret_path)
        except Exception as exc:  # pragma: no cover - hvac errors
            raise SecretResolutionError(f"Unable to read Vault secret '{secret_path}': {exc}") from exc
        return response["data"]["data"].get("value") or response["data"]["data"].get(name)


_DEFAULT_MANAGER: Optional[SecretManager] = None


def get_secret_manager() -> SecretManager:
    global _DEFAULT_MANAGER
    if _DEFAULT_MANAGER is None:
        config = SecretsConfig.from_env()
        _DEFAULT_MANAGER = SecretManager(config)
        try:
            _DEFAULT_MANAGER.validate_required()
        except SecretResolutionError as exc:
            LOGGER.error("Required secret missing: %s", exc)
            raise
    return _DEFAULT_MANAGER


def get_secret(name: str, *, required: bool = True, default: Optional[str] = None) -> Optional[str]:
    """Convenience helper to fetch a secret via the default manager."""
    return get_secret_manager().get(name, required=required, default=default)


__all__ = [
    "SecretManager",
    "SecretResolutionError",
    "SecretsConfig",
    "get_secret_manager",
    "get_secret",
]

