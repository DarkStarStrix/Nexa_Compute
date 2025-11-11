"""Project registry utilities for NexaCompute."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

_SLUG_PATTERN = re.compile(r"^[a-z0-9_]+$")

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PROJECTS_DIR = _REPO_ROOT / "projects"


class ProjectRegistryError(RuntimeError):
    """Raised when project metadata cannot be resolved."""


def _validate_slug(slug: str) -> None:
    if not _SLUG_PATTERN.match(slug):
        raise ProjectRegistryError(f"Invalid project slug '{slug}'. Must match {_SLUG_PATTERN.pattern}.")


@dataclass(frozen=True)
class ProjectMetadata:
    """Lightweight representation of a registered project."""

    slug: str
    display_name: str
    root: Path
    manifest: Dict[str, Any]

    @property
    def manifests_dir(self) -> Path:
        return self.root / "manifests"

    @property
    def configs_dir(self) -> Path:
        return self.root / "configs"

    @property
    def docs_dir(self) -> Path:
        return self.root / "docs"

    @property
    def pipelines_dir(self) -> Path:
        return self.root / "pipelines"

    def _path_from_manifest(self, key: str) -> Path:
        paths = self.manifest.get("paths", {})
        relative = paths.get(key)
        if not relative:
            raise ProjectRegistryError(f"Manifest for '{self.slug}' missing paths['{key}'].")
        return (_REPO_ROOT / relative).resolve()

    @property
    def raw_data_dir(self) -> Path:
        return self._path_from_manifest("data_raw")

    @property
    def processed_data_dir(self) -> Path:
        return self._path_from_manifest("data_processed")

    @property
    def artifacts_dir(self) -> Path:
        return self._path_from_manifest("artifacts")

    @property
    def configs_path(self) -> Path:
        return self._path_from_manifest("configs")

    @property
    def docs_path(self) -> Path:
        return self._path_from_manifest("docs")


class ProjectRegistry:
    """Filesystem-backed registry of NexaCompute projects."""

    def __init__(self, projects_dir: Path | None = None) -> None:
        self._projects_dir = projects_dir or _PROJECTS_DIR
        self._cache: Dict[str, ProjectMetadata] = {}
        self.refresh()

    def refresh(self) -> None:
        """Reload metadata for all projects."""

        self._cache.clear()
        if not self._projects_dir.exists():
            return
        for candidate in self._projects_dir.iterdir():
            if not candidate.is_dir() or candidate.name.startswith("_"):
                continue
            metadata = self._load_metadata(candidate)
            if metadata:
                self._cache[metadata.slug] = metadata

    def list(self) -> List[ProjectMetadata]:
        """Return metadata for all registered projects."""

        return list(self._cache.values())

    def slugs(self) -> List[str]:
        """Return the list of registered project slugs."""

        return sorted(self._cache.keys())

    def get(self, slug: str) -> ProjectMetadata:
        """Return metadata for a given project slug."""

        _validate_slug(slug)
        try:
            return self._cache[slug]
        except KeyError as exc:
            raise ProjectRegistryError(f"Project '{slug}' is not registered.") from exc

    def ensure_registered(self, slug: str) -> None:
        """Validate that the slug is present in the registry."""

        self.get(slug)

    def _load_metadata(self, project_dir: Path) -> Optional[ProjectMetadata]:
        manifest_path = project_dir / "manifests" / "project_manifest.json"
        if not manifest_path.exists():
            raise ProjectRegistryError(
                f"Project '{project_dir.name}' missing manifest: {manifest_path}"
            )
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        slug = data.get("project_slug")
        if not isinstance(slug, str):
            raise ProjectRegistryError(f"Manifest {manifest_path} missing 'project_slug'.")
        _validate_slug(slug)
        if slug != project_dir.name:
            raise ProjectRegistryError(
                f"Manifest slug '{slug}' does not match directory '{project_dir.name}'."
            )
        display_name = data.get("display_name") or slug
        return ProjectMetadata(slug=slug, display_name=display_name, root=project_dir, manifest=data)


DEFAULT_PROJECT_REGISTRY = ProjectRegistry()


def assert_project_exists(slug: str) -> ProjectMetadata:
    """Convenience helper to validate a project slug and return metadata."""

    return DEFAULT_PROJECT_REGISTRY.get(slug)


def validate_project_slug(slug: str) -> None:
    """Public wrapper for slug validation."""

    _validate_slug(slug)

