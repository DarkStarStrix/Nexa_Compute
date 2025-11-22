"""SSH command executor for remote workers."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Optional, List

logger = logging.getLogger(__name__)


class SSHExecutor:
    """Execute commands on remote workers via SSH."""
    
    def __init__(
        self,
        ssh_host: str,
        ssh_user: str = "root",
        ssh_key_path: Optional[Path] = None
    ):
        self.ssh_host = ssh_host
        self.ssh_user = ssh_user
        self.ssh_key_path = ssh_key_path
    
    def execute(
        self,
        command: str,
        check: bool = True,
        timeout: Optional[int] = None
    ) -> subprocess.CompletedProcess:
        """
        Execute command on remote host.
        
        Args:
            command: Command to execute
            check: Raise exception on non-zero exit
            timeout: Command timeout in seconds
            
        Returns:
            CompletedProcess with stdout/stderr
        """
        ssh_args = self._build_ssh_args()
        ssh_args.append(command)
        
        logger.debug(f"SSH exec on {self.ssh_host}: {command[:100]}...")
        
        return subprocess.run(
            ssh_args,
            capture_output=True,
            text=True,
            check=check,
            timeout=timeout
        )
    
    def execute_async(self, command: str) -> None:
        """
        Execute command asynchronously (fire and forget).
        
        Args:
            command: Command to execute
        """
        ssh_args = self._build_ssh_args()
        ssh_args.append(f"nohup {command} > /dev/null 2>&1 &")
        
        logger.debug(f"SSH async exec on {self.ssh_host}: {command[:100]}...")
        
        subprocess.Popen(
            ssh_args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    
    def upload_file(self, local_path: Path, remote_path: str) -> None:
        """
        Upload file to remote host via SCP.
        
        Args:
            local_path: Local file path
            remote_path: Remote destination path
        """
        scp_args = ["scp"]
        if self.ssh_key_path:
            scp_args.extend(["-i", str(self.ssh_key_path)])
        scp_args.extend([
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            str(local_path),
            f"{self.ssh_user}@{self.ssh_host}:{remote_path}"
        ])
        
        logger.debug(f"SCP upload: {local_path} -> {self.ssh_host}:{remote_path}")
        subprocess.run(scp_args, check=True)
    
    def download_file(self, remote_path: str, local_path: Path) -> None:
        """
        Download file from remote host via SCP.
        
        Args:
            remote_path: Remote file path
            local_path: Local destination path
        """
        scp_args = ["scp"]
        if self.ssh_key_path:
            scp_args.extend(["-i", str(self.ssh_key_path)])
        scp_args.extend([
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            f"{self.ssh_user}@{self.ssh_host}:{remote_path}",
            str(local_path)
        ])
        
        logger.debug(f"SCP download: {self.ssh_host}:{remote_path} -> {local_path}")
        subprocess.run(scp_args, check=True)
    
    def _build_ssh_args(self) -> List[str]:
        """Build SSH command arguments."""
        ssh_args = ["ssh"]
        if self.ssh_key_path:
            ssh_args.extend(["-i", str(self.ssh_key_path)])
        ssh_args.extend([
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            f"{self.ssh_user}@{self.ssh_host}"
        ])
        return ssh_args


__all__ = ["SSHExecutor"]
