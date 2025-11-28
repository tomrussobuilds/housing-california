"""
Logger module
-------------
Provides a reusable logger with RotatingFileHandler and console output.
"""

import logging
import sys
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime, timezone

sys.dont_write_bytecode = True


class Logger:
    """Configurable logger with rotating file handler and stdout output.

    Parameters:
        name (str): Logger name.
        log_dir (Path): Directory where log files are stored.
        log_to_file (bool): Enable/disable log file output.
        level (int): Default logging level.
        max_bytes (int): Max size before rotation.
        backup_count (int): Number of rotated files to keep.
    """

    _loggers = {}

    def __init__(
        self,
        name: str = "california_housing_pipeline",
        log_dir: Path = Path("logs"),
        log_to_file: bool = True,
        level: int = logging.INFO,
        max_bytes: int = 5 * 1024 * 1024,
        backup_count: int = 5,
    ):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_to_file = log_to_file
        self.level = logging.DEBUG if os.getenv("DEBUG") == "1" else level
        self.max_bytes = max_bytes
        self.backup_count = backup_count

        if name in Logger._loggers:
            self.logger = Logger._loggers[name]
        else:
            self.logger = logging.getLogger(name)
            self._setup_logger()
            Logger._loggers[name] = self.logger

    def _setup_logger(self):
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )
        self.logger.setLevel(self.level)
        self.logger.propagate = False

        # Console handler
        if not any(isinstance(h, logging.StreamHandler) for h in self.logger.handlers):
            stream = logging.StreamHandler(sys.stdout)
            stream.setFormatter(formatter)
            self.logger.addHandler(stream)

        # Rotating file handler
        if self.log_to_file:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = self.log_dir / f"{self.name}_{timestamp}.log"

            if not any(isinstance(h, RotatingFileHandler) for h in self.logger.handlers):
                file_handler = RotatingFileHandler(
                    filename,
                    maxBytes=self.max_bytes,
                    backupCount=self.backup_count,
                )
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)

    def get_logger(self):
        """Return the configured logger."""
        return self.logger
