import logging
import sys
from typing import Optional
from pathlib import Path
from datetime import datetime
from functools import lru_cache


class ColoredFormatter(logging.Formatter):
    """Colored output formatter."""
    COLORS = {
        'DEBUG': '\033[94m',
        'INFO': '\033[92m',
        'WARNING': '\033[93m',
        'ERROR': '\033[91m',
        'CRITICAL': '\033[91m\033[1m',
        'RESET': '\033[0m'
    }

    def format(self, record):
        if hasattr(sys.stderr, "isatty") and sys.stderr.isatty():
            record.levelname = (
                f"{self.COLORS[record.levelname]}"
                f"{record.levelname}"
                f"{self.COLORS['RESET']}"
            )
        return super().format(record)


@lru_cache(None)
def get_logger(
        name: str,
        log_dir: Optional[str] = "logs",
        level: str = "INFO"
) -> logging.Logger:
    """
    Get configured logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(getattr(logging, level))

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        ))
        logger.addHandler(console_handler)

        # File handler
        if log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(exist_ok=True)

            file_handler = logging.FileHandler(
                log_dir / f"{name}_{datetime.now():%Y%m%d}.log"
            )
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            logger.addHandler(file_handler)

        logger.propagate = False

    return logger