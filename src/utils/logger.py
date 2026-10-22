"""
Live Dual-Stream Logging Utility.
Mirrors terminal standard output and standard error simultaneously to the console
and to a timestamped log file in the results directory (UTF-8 encoded).
"""

import atexit
from datetime import datetime
import os
import sys
from typing import TextIO


class DualStream:
    """
    Wraps an existing output stream (e.g., stdout/stderr) and a log file,
    routing writes to both destinations and flushing immediately.
    """

    def __init__(self, stream: TextIO, log_file: TextIO, filter_cr: bool = False):
        self.stream = stream
        self.log_file = log_file
        self.filter_cr = filter_cr

    def write(self, message: str) -> None:
        self.stream.write(message)
        # Avoid cluttering the log file with high-frequency tqdm carriage-return (\r) updates
        if self.filter_cr and "\r" in message and "\n" not in message:
            return
        if not self.log_file.closed:
            self.log_file.write(message)
            self.log_file.flush()

    def flush(self) -> None:
        self.stream.flush()
        if not self.log_file.closed:
            self.log_file.flush()

    def isatty(self) -> bool:
        """Preserves terminal interactive capabilities (for tqdm and curses)."""
        return getattr(self.stream, "isatty", lambda: False)()

    def __getattr__(self, name: str):
        """Delegates all other stream attributes (encoding, errors, fileno, etc.)."""
        return getattr(self.stream, name)


def setup_experiment_logger(output_dir: str = "./results") -> str:
    """
    Configures dual logging to console and a timestamped log file.
    Creates a unique file on every run: experiment_run_YYYY-MM-DD_HH-MM-SS.log.

    Returns:
        The path to the created log file.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"experiment_run_{timestamp}.log"
    log_path = os.path.join(output_dir, log_filename)

    log_file = open(log_path, "w", encoding="utf-8")

    orig_stdout = sys.stdout
    orig_stderr = sys.stderr

    sys.stdout = DualStream(orig_stdout, log_file, filter_cr=False)
    sys.stderr = DualStream(orig_stderr, log_file, filter_cr=True)

    def cleanup():
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        if not log_file.closed:
            log_file.flush()
            log_file.close()

    atexit.register(cleanup)

    return log_path
