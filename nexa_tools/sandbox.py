"""Execution sandbox for running user-provided Python snippets safely."""

from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List
from uuid import uuid4

from nexa_compute.core.exceptions import ResourceError


@dataclass(frozen=True)
class SandboxResult:
    """Outcome of executing Python code inside the sandbox."""

    stdout: str
    stderr: str
    artifacts: List[str]


class SandboxRunner:
    """Lightweight sandbox that executes Python code in a temporary directory or container."""

    def __init__(
        self,
        *,
        python_executable: str | None = None,
        artifact_root: Path | None = None,
        use_docker: bool | None = None,
        sandbox_image: str | None = None,
        cpu_limit: float = 1.0,
        memory_limit: str = "1g",
    ) -> None:
        self._python = python_executable or sys.executable
        self._artifact_root = Path(artifact_root or Path("artifacts") / "tool_runs").resolve()
        self._artifact_root.mkdir(parents=True, exist_ok=True)
        self._use_docker = use_docker if use_docker is not None else True
        self._sandbox_image = sandbox_image or os.getenv("NEXA_SANDBOX_IMAGE", "ghcr.io/nexa/nexa_sandbox_py:latest")
        self._cpu_limit = cpu_limit
        self._memory_limit = memory_limit
        self._max_code_bytes = 64 * 1024

    def run(self, code: str, *, timeout_s: int = 10) -> SandboxResult:
        """Execute the provided Python code with a wall-clock timeout."""

        if not code.strip():
            raise ValueError("Cannot execute empty code payload.")
        self._validate_source(code)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            script_path = tmp_path / "main.py"
            script_path.write_text(code, encoding="utf-8")

            work_dir = tmp_path / "work"
            work_dir.mkdir(parents=True, exist_ok=True)

            run_id = uuid4().hex
            artifact_dir = self._artifact_root / run_id
            artifact_dir.mkdir(parents=True, exist_ok=True)

            try:
                if self._use_docker:
                    completed = self._run_in_container(script_path, work_dir, artifact_dir, timeout_s)
                else:
                    completed = self._run_locally(script_path, tmp_path, timeout_s)
            except subprocess.TimeoutExpired as exc:
                return SandboxResult(stdout=exc.stdout or "", stderr=_format_timeout_error(exc), artifacts=[])

            self._harvest_artifacts(work_dir if self._use_docker else tmp_path, artifact_dir)
            artifacts = self._collect_artifact_paths(artifact_dir)
            return SandboxResult(stdout=completed.stdout, stderr=completed.stderr, artifacts=artifacts)

    def _run_locally(self, script_path: Path, tmp_path: Path, timeout_s: int) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [self._python, str(script_path)],
            cwd=tmp_path,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )

    def _run_in_container(
        self,
        script_path: Path,
        work_dir: Path,
        artifact_dir: Path,
        timeout_s: int,
    ) -> subprocess.CompletedProcess[str]:
        if not shutil.which("docker"):
            raise ResourceError("Docker is required for sandboxed execution but was not found in PATH.")

        script_root = script_path.parent.resolve()
        work_dir = work_dir.resolve()
        artifact_dir = artifact_dir.resolve()

        command = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--cpus",
            str(self._cpu_limit),
            "--memory",
            self._memory_limit,
            "--pids-limit",
            "256",
            "--security-opt",
            "no-new-privileges",
            "-v",
            f"{script_root}:/sandbox_ro:ro",
            "-v",
            f"{work_dir}:/sandbox_work",
            "-v",
            f"{artifact_dir}:/sandbox_artifacts",
            "-e",
            "SANDBOX_OUTPUT_DIR=/sandbox_artifacts",
            self._sandbox_image,
            "bash",
            "-lc",
            "cp -r /sandbox_ro/. /sandbox_work && cd /sandbox_work && python main.py",
        ]
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )

    @staticmethod
    def _harvest_artifacts(source_dir: Path, destination_dir: Path) -> None:
        """Copy any generated files (excluding the script) to the artifact directory."""

        for path in _iter_artifact_candidates(source_dir):
            relative_name = path.name
            destination = destination_dir / relative_name
            if path.is_dir():
                shutil.copytree(path, destination, dirs_exist_ok=True)
            else:
                shutil.copy2(path, destination)

    @staticmethod
    def _collect_artifact_paths(artifact_dir: Path) -> List[str]:
        """Enumerate artifacts under ``artifact_dir``."""
        if not artifact_dir.exists():
            return []
        return [
            f"artifact://{artifact_dir.name}/{path.name}"
            for path in sorted(artifact_dir.iterdir())
            if path.name != ".DS_Store"
        ]

    def _validate_source(self, code: str) -> None:
        """Static analysis to block obviously dangerous payloads."""
        if len(code.encode("utf-8")) > self._max_code_bytes:
            raise ValueError("Sandbox payload exceeds maximum allowed size (64KB).")
        tree = ast.parse(code)
        validator = _SandboxASTValidator()
        validator.visit(tree)


class _SandboxASTValidator(ast.NodeVisitor):
    """Reject imports and calls that may escape the sandbox."""

    BANNED_MODULES = {"os", "subprocess", "sys", "socket", "shutil", "pathlib", "tempfile"}
    BANNED_ATTRS = {"system", "popen", "exec", "eval", "run", "__import__"}

    def visit_Import(self, node: ast.Import) -> None:  # pragma: no cover - trivial
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root in self.BANNED_MODULES:
                raise ValueError(f"Import of module '{root}' is not permitted in the sandbox.")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # pragma: no cover - trivial
        module = (node.module or "").split(".")[0]
        if module in self.BANNED_MODULES:
            raise ValueError(f"Import of module '{module}' is not permitted in the sandbox.")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Attribute):
            if func.attr in self.BANNED_ATTRS:
                raise ValueError(f"Use of '{func.attr}' is not permitted in the sandbox.")
        elif isinstance(func, ast.Name) and func.id in self.BANNED_ATTRS:
            raise ValueError(f"Use of '{func.id}' is not permitted in the sandbox.")
        self.generic_visit(node)


def _iter_artifact_candidates(source_dir: Path) -> Iterable[Path]:
    """Yield files generated by the sandboxed script."""

    for child in source_dir.iterdir():
        if child.name == "main.py":
            continue
        if child.name.startswith("__pycache__"):
            continue
        yield child


def _format_timeout_error(exc: subprocess.TimeoutExpired) -> str:
    """Produce a structured error message for timeout scenarios."""

    return (
        f"Execution timed out after {exc.timeout}s. Partial stdout:\n{exc.stdout or ''}\n"
        f"Partial stderr:\n{exc.stderr or ''}"
    )


__all__ = ["SandboxRunner", "SandboxResult"]

