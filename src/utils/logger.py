import logging
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_obj: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra"):
            log_obj.update(record.extra)  # type: ignore

        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_obj)


def setup_logger(
    name: str, level: str = "INFO", log_file: str = "logs/app.log"
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Avoid duplicate handlers
    if logger.hasHandlers():
        return logger

    # Ensure log directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    json_formatter = JSONFormatter()

    # File Handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(json_formatter)
    logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(json_formatter)
    logger.addHandler(console_handler)

    return logger


# Convenience function
def get_logger(name: str) -> logging.Logger:
    from src.core.config import get_config

    config = get_config()
    return setup_logger(name, level=config.logging.level, log_file=config.logging.file)
