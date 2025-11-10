"""SQLite-backed registry for models, datasets, and pipeline runs.

The registry is responsible for mapping logical names to concrete artifact URIs
and tracking pipeline run metadata. It follows the contract described in
``docs/Spec_V2.md``.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional, Sequence

from .artifacts import ArtifactError, promote as promote_artifact_pointer

__all__ = [
    "RegistryError",
    "ModelRegistry",
    "RunStatus",
    "create_registry",
    "DEFAULT_DB_PATH",
    "get_registry",
    "register",
    "resolve",
    "promote",
    "promote_pointer_uri",
]
DEFAULT_DB_PATH = Path("registry/models.db")


def get_registry(db_path: Optional[Path] = None) -> ModelRegistry:
    """Return a :class:`ModelRegistry` instance for ``db_path``."""

    target_path = (db_path or DEFAULT_DB_PATH).resolve()
    return ModelRegistry(target_path)


def register(
    name: str,
    uri: str,
    meta: Mapping[str, object],
    *,
    db_path: Optional[Path] = None,
    version: Optional[str] = None,
    tags: Optional[Sequence[str]] = None,
) -> str:
    """Register an artifact with the default registry database."""

    registry = get_registry(db_path)
    return registry.register(name, uri, meta, version=version, tags=tags)


def resolve(reference: str, *, db_path: Optional[Path] = None) -> str:
    """Resolve a reference against the default registry database."""

    registry = get_registry(db_path)
    return registry.resolve(reference)


def promote(name: str, version: str, tag: str, *, db_path: Optional[Path] = None) -> None:
    """Promote ``version`` under ``name`` to ``tag`` in the default registry."""

    registry = get_registry(db_path)
    registry.promote(name, version, tag)


class RegistryError(RuntimeError):
    """Raised when registry operations fail."""


def _utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path = db_path.resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def create_registry(db_path: Path) -> None:
    """Initialise the registry database if it does not already exist."""

    with _connect(db_path) as conn:
        conn.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                version TEXT NOT NULL,
                uri TEXT NOT NULL,
                meta_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(name, version)
            );
            CREATE TABLE IF NOT EXISTS model_tags (
                name TEXT NOT NULL,
                tag TEXT NOT NULL,
                version TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (name, tag),
                FOREIGN KEY(name, version) REFERENCES models(name, version)
            );
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                spec_json TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                ended_at TEXT
            );
            """
        )


def _latest_version(conn: sqlite3.Connection, name: str) -> Optional[str]:
    cursor = conn.execute(
        "SELECT version FROM models WHERE name = ? ORDER BY created_at DESC, id DESC LIMIT 1",
        (name,),
    )
    row = cursor.fetchone()
    return row["version"] if row else None


def _version_exists(conn: sqlite3.Connection, name: str, version: str) -> bool:
    cursor = conn.execute("SELECT 1 FROM models WHERE name = ? AND version = ? LIMIT 1", (name, version))
    return cursor.fetchone() is not None


def _increment_semver(current: Optional[str]) -> str:
    if not current:
        return "1.0.0"
    try:
        major, minor, patch = (int(part) for part in current.split("."))
    except ValueError as exc:  # pragma: no cover - schema should prevent this
        raise RegistryError(f"cannot increment malformed semver: {current}") from exc
    patch += 1
    return f"{major}.{minor}.{patch}"


def _is_semver(version: str) -> bool:
    parts = version.split(".")
    if len(parts) != 3:
        return False
    return all(part.isdigit() for part in parts)


@dataclass(frozen=True)
class RunStatus:
    """Lifecycle information for a pipeline run."""

    run_id: str
    status: str
    started_at: str
    ended_at: Optional[str] = None


class ModelRegistry:
    """Facade for interacting with the registry database."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path.resolve()
        create_registry(self.db_path)

    def register(
        self,
        name: str,
        uri: str,
        meta: Mapping[str, object],
        *,
        version: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
    ) -> str:
        """Register a new model artifact and return its version."""

        payload = json.dumps(meta, sort_keys=True)
        created_at = _utcnow()
        tags = list(tags or [])
        with _connect(self.db_path) as conn, conn:  # context manager commits automatically
            latest = _latest_version(conn, name)
            version_to_use = version or _increment_semver(latest)

            if version and _version_exists(conn, name, version):
                raise RegistryError(f"model {name} version {version} already exists")

            conn.execute(
                "INSERT INTO models (name, version, uri, meta_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (name, version_to_use, uri, payload, created_at),
            )

            all_tags = {"latest", *tags}
            for tag in all_tags:
                conn.execute(
                    """
                    INSERT INTO model_tags (name, tag, version, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(name, tag) DO UPDATE SET version = excluded.version, updated_at = excluded.updated_at
                    """,
                    (name, tag, version_to_use, created_at),
                )

        return version_to_use

    def resolve(self, reference: str) -> str:
        """Resolve a model reference (``name[:selector]``) to a concrete URI."""

        if ":" in reference:
            name, selector = reference.split(":", 1)
        else:
            name, selector = reference, "latest"
        name = name.strip()
        selector = selector.strip()

        if not name:
            raise RegistryError("model reference must include a name")

        with _connect(self.db_path) as conn:
            if _is_semver(selector):
                row = conn.execute(
                    "SELECT uri FROM models WHERE name = ? AND version = ?",
                    (name, selector),
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    SELECT m.uri
                    FROM model_tags t
                    JOIN models m ON m.name = t.name AND m.version = t.version
                    WHERE t.name = ? AND t.tag = ?
                    """,
                    (name, selector),
                ).fetchone()

        if not row:
            raise RegistryError(f"model reference not found: {reference}")

        return row["uri"]

    def promote(self, name: str, version: str, tag: str) -> None:
        """Promote an existing version to a new tag."""

        tag = tag.strip()
        if not tag:
            raise RegistryError("tag must not be empty")

        with _connect(self.db_path) as conn, conn:
            if not _version_exists(conn, name, version):
                raise RegistryError(f"model {name} version {version} is not registered")

            conn.execute(
                """
                INSERT INTO model_tags (name, tag, version, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(name, tag) DO UPDATE SET version = excluded.version, updated_at = excluded.updated_at
                """,
                (name, tag, version, _utcnow()),
            )

    def list_versions(self, name: str) -> list[str]:
        """Return all versions registered for ``name`` ordered by creation time."""

        with _connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT version FROM models WHERE name = ? ORDER BY created_at ASC, id ASC",
                (name,),
            ).fetchall()
        return [row["version"] for row in rows]

    def record_run(self, run_id: str, spec: Mapping[str, object], status: str) -> None:
        """Insert a run row, overwriting existing metadata if the run already exists."""

        with _connect(self.db_path) as conn, conn:
            conn.execute(
                """
                INSERT INTO runs (id, spec_json, status, started_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET spec_json = excluded.spec_json, status = excluded.status, started_at = excluded.started_at
                """,
                (run_id, json.dumps(spec, sort_keys=True), status, _utcnow()),
            )

    def update_run_status(self, run_id: str, status: str, *, ended: bool = False) -> None:
        """Update the status (and optionally the end timestamp) for a run."""

        if ended:
            ended_at = _utcnow()
        else:
            ended_at = None

        with _connect(self.db_path) as conn, conn:
            if ended:
                conn.execute(
                    "UPDATE runs SET status = ?, ended_at = ? WHERE id = ?",
                    (status, ended_at, run_id),
                )
            else:
                conn.execute(
                    "UPDATE runs SET status = ? WHERE id = ?",
                    (status, run_id),
                )

    def get_run(self, run_id: str) -> Optional[RunStatus]:
        """Return run status information if the run exists."""

        with _connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT id, status, started_at, ended_at FROM runs WHERE id = ?",
                (run_id,),
            ).fetchone()

        if not row:
            return None
        return RunStatus(run_id=row["id"], status=row["status"], started_at=row["started_at"], ended_at=row["ended_at"])


def promote_pointer_uri(pointer_uri: str, artifact_path: Path) -> None:
    """Promote an artifact pointer using the artifact protocol."""

    pointer_path = Path(pointer_uri)
    try:
        promote_artifact_pointer(pointer_path, artifact_path)
    except ArtifactError as exc:  # pragma: no cover - promotion failure is rare
        raise RegistryError(str(exc)) from exc

