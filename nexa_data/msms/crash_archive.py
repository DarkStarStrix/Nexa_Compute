"""Remote crash archive for cloud mode crash recovery."""

import json
import logging
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class CrashArchive:
    """Crash archive manager for recovering failed runs."""

    def __init__(self, crash_dumps_root: Path = Path("nexadata_crash_dumps")):
        """Initialize crash archive.

        Args:
            crash_dumps_root: Root directory for crash dumps
        """
        self.crash_dumps_root = Path(crash_dumps_root)
        self.crash_dumps_root.mkdir(parents=True, exist_ok=True)

    def archive_crash(
        self,
        run_id: str,
        remote_host: Optional[str] = None,
        remote_user: Optional[str] = None,
        remote_run_dir: Optional[Path] = None,
        remote_manifest_path: Optional[Path] = None,
    ) -> Path:
        """Archive a crashed run from remote host.

        Args:
            run_id: Run ID to archive
            remote_host: Remote hostname (for SSH)
            remote_user: Remote username (for SSH)
            remote_run_dir: Remote run directory path
            remote_manifest_path: Remote manifest path

        Returns:
            Path to local crash dump directory
        """
        crash_dir = self.crash_dumps_root / f"run_{run_id}"
        crash_dir.mkdir(parents=True, exist_ok=True)

        if remote_host and remote_user:
            self._pull_from_remote(
                run_id,
                remote_host,
                remote_user,
                crash_dir,
                remote_run_dir,
                remote_manifest_path,
            )
        else:
            logger.warning("No remote host specified, crash archive skipped")

        return crash_dir

    def _pull_from_remote(
        self,
        run_id: str,
        remote_host: str,
        remote_user: str,
        local_crash_dir: Path,
        remote_run_dir: Optional[Path],
        remote_manifest_path: Optional[Path],
    ) -> None:
        """Pull crash data from remote host via SSH.

        Args:
            run_id: Run ID
            remote_host: Remote hostname
            remote_user: Remote username
            local_crash_dir: Local crash directory
            remote_run_dir: Remote run directory
            remote_manifest_path: Remote manifest path
        """
        ssh_base = f"{remote_user}@{remote_host}"

        items_to_pull = []

        if remote_manifest_path:
            items_to_pull.append(("manifest", remote_manifest_path))

        if remote_run_dir:
            items_to_pull.extend([
                ("logs", remote_run_dir / "*.log"),
                ("metrics", remote_run_dir / "metrics.json"),
                ("timeseries", remote_run_dir / "resource_timeseries.json"),
                ("shards", remote_run_dir / "train"),
                ("shards", remote_run_dir / "val"),
                ("shards", remote_run_dir / "test"),
                ("validation_reports", remote_run_dir / "validation"),
            ])

        for item_type, remote_path in items_to_pull:
            try:
                local_path = local_crash_dir / item_type
                local_path.mkdir(parents=True, exist_ok=True)

                if remote_path.name.endswith(".json") or remote_path.name.endswith(".log"):
                    remote_cmd = f"cat {remote_path}"
                    local_file = local_path / remote_path.name
                    result = subprocess.run(
                        ["ssh", ssh_base, remote_cmd],
                        capture_output=True,
                        text=True,
                    )
                    if result.returncode == 0:
                        with open(local_file, "w") as f:
                            f.write(result.stdout)
                        logger.info(f"Pulled {item_type}: {remote_path.name}")
                else:
                    remote_cmd = f"tar -czf - -C {remote_path.parent} {remote_path.name}"
                    result = subprocess.run(
                        ["ssh", ssh_base, remote_cmd],
                        capture_output=True,
                    )
                    if result.returncode == 0:
                        tar_path = local_path / f"{remote_path.name}.tar.gz"
                        with open(tar_path, "wb") as f:
                            f.write(result.stdout)
                        logger.info(f"Pulled {item_type}: {remote_path.name} (tar.gz)")

            except Exception as e:
                logger.warning(f"Failed to pull {item_type} from {remote_path}: {e}")

        summary = {
            "run_id": run_id,
            "archived_at": datetime.utcnow().isoformat(),
            "remote_host": remote_host,
            "remote_user": remote_user,
            "items_pulled": [item[0] for item in items_to_pull],
        }

        summary_path = local_crash_dir / "archive_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

    def list_crashes(self) -> List[Dict]:
        """List all crash dumps.

        Returns:
            List of crash dump information dictionaries
        """
        crashes = []

        for crash_dir in sorted(self.crash_dumps_root.glob("run_*")):
            if not crash_dir.is_dir():
                continue

            summary_path = crash_dir / "archive_summary.json"
            if summary_path.exists():
                with open(summary_path) as f:
                    summary = json.load(f)
            else:
                summary = {
                    "run_id": crash_dir.name.replace("run_", ""),
                    "archived_at": datetime.fromtimestamp(
                        crash_dir.stat().st_mtime
                    ).isoformat(),
                }

            crashes.append({
                "run_id": summary.get("run_id", crash_dir.name),
                "archived_at": summary.get("archived_at"),
                "path": str(crash_dir),
            })

        return crashes

    def inspect_crash(self, run_id: str) -> Dict:
        """Inspect a specific crash dump.

        Args:
            run_id: Run ID to inspect

        Returns:
            Crash dump information dictionary
        """
        crash_dir = self.crash_dumps_root / f"run_{run_id}"

        if not crash_dir.exists():
            raise FileNotFoundError(f"Crash dump not found: {crash_dir}")

        info = {
            "run_id": run_id,
            "path": str(crash_dir),
        }

        summary_path = crash_dir / "archive_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            info.update(summary)

        manifest_path = crash_dir / "manifest" / "dataset_manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                info["manifest"] = json.load(f)

        metrics_path = crash_dir / "metrics" / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                info["metrics"] = json.load(f)

        log_files = list(crash_dir.glob("logs/*.log"))
        if log_files:
            info["log_files"] = [str(f) for f in log_files]

        shard_dirs = list(crash_dir.glob("shards/*"))
        if shard_dirs:
            info["shard_directories"] = [str(d) for d in shard_dirs]

        return info

    def clean_old_crashes(self, days: int = 30) -> int:
        """Clean crash dumps older than specified days.

        Args:
            days: Number of days to keep

        Returns:
            Number of crash dumps cleaned
        """
        import time
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        cleaned = 0

        for crash_dir in self.crash_dumps_root.glob("run_*"):
            if crash_dir.is_dir() and crash_dir.stat().st_mtime < cutoff_time:
                shutil.rmtree(crash_dir)
                cleaned += 1
                logger.info(f"Cleaned old crash dump: {crash_dir}")

        return cleaned

