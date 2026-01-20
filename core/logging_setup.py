"""
core/logging_setup.py

Central logging configuration.
Writes to console and to a log file under logs_dir.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union


def setup_logging(logs_dir: Union[str, Path], level: int = logging.INFO) -> None:
    """
    Initialise root logging for GaitGuard.

    Parameters
    ----------
    logs_dir : str or Path
        Directory where the main gaitguard.log file will be written.
        Accepts both plain strings (from YAML) and Path objects.
    level : int
        Logging level for the root logger (default: INFO).
    """
    logs_path = Path(logs_dir)
    logs_path.mkdir(parents=True, exist_ok=True)
    log_file = logs_path / "gaitguard.log"

    root = logging.getLogger()
    if root.handlers:
        for h in list(root.handlers):
            root.removeHandler(h)

    root.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(ch)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)
