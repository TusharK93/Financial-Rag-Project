
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logger(name: str = "finance_rag") -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, log_level, logging.INFO))
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | "
        "%(module)s:%(lineno)d | %(message)s"
    )

    # Always log to console (Render and other hosts collect stdout).
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional rotating file logs for local development.
    if os.getenv("LOG_TO_FILE", "false").lower() == "true":
        log_dir = Path(os.getenv("LOG_DIR", "logs"))
        log_dir.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_dir / "finance_rag.log",
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


logger = setup_logger()
