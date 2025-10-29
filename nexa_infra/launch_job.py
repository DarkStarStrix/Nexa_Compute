"""Launch training jobs either locally or through torchrun."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

from nexa_train.train import run_training_job


def launch_training_job(config_path: Path, *, distributed: bool = False, overrides: Optional[list[str]] = None) -> None:
    if distributed:
        cmd = [
            "bash",
            "scripts/launch_ddp.sh",
            "--config",
            str(config_path),
        ]
        if overrides:
            for override in overrides:
                cmd.extend(["--override", override])
        print("[nexa-infra] Launching distributed job:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        return
    print(f"[nexa-infra] Launching single-process training job with {config_path}")
    run_training_job(config_path, overrides=overrides or [])
