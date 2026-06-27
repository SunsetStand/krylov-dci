"""
Utilities for Krylov-dCI project.

Timing, logging, and I/O helpers.
"""

import time
import os
import sys
from typing import Optional


class Timer:
    """Context manager for wall-clock timing."""

    def __init__(self, label: str = ""):
        self.label = label
        self.start_time = 0.0
        self.elapsed = 0.0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
        if self.label:
            print(f"[Timer] {self.label}: {self.elapsed:.2f} s")

    def get(self) -> float:
        """Get elapsed time so far (seconds)."""
        return time.time() - self.start_time


class Logger:
    """Simple logger with timestamps."""

    def __init__(self, logfile: Optional[str] = None):
        self.logfile = logfile
        if logfile:
            os.makedirs(os.path.dirname(logfile) or ".", exist_ok=True)

    def log(self, msg: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {msg}"
        print(line)
        if self.logfile:
            with open(self.logfile, "a") as f:
                f.write(line + "\n")


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        h = int(seconds / 3600)
        m = int((seconds % 3600) / 60)
        return f"{h}h{m:02d}m"
