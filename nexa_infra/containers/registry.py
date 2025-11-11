"""Registry definitions for curated Nexa container images."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable


PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
BOOTSTRAP_SCRIPT: Path = PROJECT_ROOT / "bin" / "nexa-bootstrap.sh"
DOCKER_DIR: Path = PROJECT_ROOT / "docker"


@dataclass(frozen=True)
class ContainerSpec:
    """Describe a supported container image."""

    key: str
    repository: str
    default_tag: str
    description: str
    shm_size: str
    dockerfile: Path
    context: Path = PROJECT_ROOT

    @property
    def image(self) -> str:
        """Return the default image reference."""

        return f"{self.repository}:{self.default_tag}"

    def ref(self, tag: str | None = None) -> str:
        """Return an image reference for the given tag (defaulting to the default tag)."""

        return f"{self.repository}:{tag or self.default_tag}"

    def release_tags(self, variant: str, include_latest: bool = True, include_date: bool = True) -> Iterable[str]:
        """Return the canonical release tags for this container."""

        tags = [variant]
        if include_latest:
            tags.append("latest")
        if include_date:
            snapshot = datetime.now(timezone.utc).strftime("%Y%m%d")
            tags.append(snapshot)
        return tags


_CONTAINERS: Dict[str, ContainerSpec] = {
    "train-light": ContainerSpec(
        key="train-light",
        repository="ghcr.io/nexa/nexa_light",
        default_tag="latest",
        description="HF/TRL stack for <=20B parameter jobs",
        shm_size="16g",
        dockerfile=DOCKER_DIR / "train-light.Dockerfile",
    ),
    "train-heavy": ContainerSpec(
        key="train-heavy",
        repository="ghcr.io/nexa/nexa_heavy",
        default_tag="latest",
        description="Axolotl/DeepSpeed stack for >20B or multi-node jobs",
        shm_size="32g",
        dockerfile=DOCKER_DIR / "train-heavy.Dockerfile",
    ),
    "infer": ContainerSpec(
        key="infer",
        repository="ghcr.io/nexa/nexa_infer",
        default_tag="latest",
        description="vLLM/TensorRT-LLM inference stack",
        shm_size="8g",
        dockerfile=DOCKER_DIR / "infer.Dockerfile",
    ),
}


def available_containers() -> Iterable[ContainerSpec]:
    """Yield the known container specifications."""

    return _CONTAINERS.values()


def get_container(key: str) -> ContainerSpec:
    """Return a container specification by key."""

    try:
        return _CONTAINERS[key]
    except KeyError as exc:
        known = ", ".join(sorted(_CONTAINERS))
        raise ValueError(f"Unknown container target '{key}'. Expected one of: {known}") from exc


__all__ = ["ContainerSpec", "available_containers", "get_container", "PROJECT_ROOT", "BOOTSTRAP_SCRIPT"]


