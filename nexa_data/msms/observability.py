"""Observability manager for multiple display modes."""

import logging
import subprocess
import threading
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class DisplayMode(str, Enum):
    """Top-level display mode enumeration."""
    
    ARGS = "args"  # Raw args, minimal console output
    CLI = "cli"  # CLI mode with observability options
    TUI = "tui"  # TUI mode with observability options


class ObservabilityLevel(str, Enum):
    """Observability level within CLI/TUI modes."""
    
    SIMPLE = "simple"  # tqdm + logs
    GRANULAR = "granular"  # FastAPI dashboard + tqdm


class ObservabilityManager:
    """Manages observability display based on mode and level."""

    def __init__(
        self,
        display_mode: DisplayMode,
        observability_level: Optional[ObservabilityLevel] = None,
        run_dir: Path = None,
        run_id: str = None,
        dashboard_port: int = 8080,
    ):
        """Initialize observability manager.

        Args:
            display_mode: Top-level display mode (args/cli/tui)
            observability_level: Observability level (simple/granular) - only used for cli/tui modes
            run_dir: Run directory for metrics/logs
            run_id: Run ID
            dashboard_port: Port for FastAPI dashboard (if applicable)
        """
        self.display_mode = display_mode
        self.observability_level = observability_level or ObservabilityLevel.SIMPLE
        self.run_dir = Path(run_dir) if run_dir else None
        self.run_id = run_id
        self.dashboard_port = dashboard_port
        
        self.tui_process: Optional[subprocess.Popen] = None
        self.dashboard_process: Optional[subprocess.Popen] = None
        self.dashboard_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start observability components based on mode."""
        if self.display_mode == DisplayMode.TUI:
            self._start_tui()
        elif (
            self.display_mode in [DisplayMode.CLI, DisplayMode.TUI]
            and self.observability_level == ObservabilityLevel.GRANULAR
        ):
            self._start_dashboard()

    def _start_tui(self) -> None:
        """Start Golang TUI in separate process."""
        tui_binary = Path(__file__).parent / "tui" / "nexa-data-tui"
        
        if not tui_binary.exists():
            logger.warning(
                f"TUI binary not found at {tui_binary}. "
                "Run 'go build' in nexa_data/msms/tui/ to build it."
            )
            return
        
        try:
            self.tui_process = subprocess.Popen(
                [str(tui_binary), "--run-dir", str(self.run_dir), "--run-id", self.run_id],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            logger.info(f"Started TUI (PID: {self.tui_process.pid})")
        except Exception as e:
            logger.error(f"Failed to start TUI: {e}")

    def _start_dashboard(self) -> None:
        """Start FastAPI dashboard in background thread."""
        try:
            from .dashboard import app
            import uvicorn
            
            def run_dashboard():
                uvicorn.run(app, host="0.0.0.0", port=self.dashboard_port, log_level="warning")
            
            self.dashboard_thread = threading.Thread(
                target=run_dashboard,
                daemon=True,
            )
            self.dashboard_thread.start()
            logger.info(f"Started FastAPI dashboard on port {self.dashboard_port}")
        except Exception as e:
            logger.error(f"Failed to start dashboard: {e}")

    def stop(self) -> None:
        """Stop observability components."""
        if self.tui_process:
            try:
                self.tui_process.terminate()
                self.tui_process.wait(timeout=5)
                logger.info("TUI stopped")
            except Exception as e:
                logger.warning(f"Error stopping TUI: {e}")
                if self.tui_process:
                    self.tui_process.kill()

        # Dashboard thread will stop automatically when main process exits (daemon)

    def should_use_tqdm(self) -> bool:
        """Determine if tqdm progress bar should be used."""
        # Use tqdm for CLI/TUI modes, not for raw args mode
        return self.display_mode in [DisplayMode.CLI, DisplayMode.TUI]

    def should_use_minimal_output(self) -> bool:
        """Determine if minimal output should be used."""
        return self.display_mode == DisplayMode.ARGS

    def get_dashboard_url(self) -> Optional[str]:
        """Get dashboard URL if applicable."""
        if (
            self.display_mode in [DisplayMode.CLI, DisplayMode.TUI]
            and self.observability_level == ObservabilityLevel.GRANULAR
        ):
            return f"http://localhost:{self.dashboard_port}"
        return None

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()

