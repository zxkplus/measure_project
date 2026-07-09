"""
Thread-safe logging system for Measure API.

Usage:
    from measure_api.config import Config
    from measure_api.logger import setup_logging, get_logger

    cfg = Config.load()
    setup_logging(cfg.get("log"))
    logger = get_logger("project")
    logger.info("Template set: center=%s size=%s", center, size)

Log files:
    logs/
    ├── measure_api_YYYY-MM-DD.log
    └── measure_api_YYYY-MM-DD.error.log
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from logging.handlers import TimedRotatingFileHandler
from typing import Any, Optional

# Prevent double-init
_initialized = False
_init_lock = threading.Lock()
_trace_id: threading.local = threading.local()


class TraceIdFilter(logging.Filter):
    """Inject ``trace_id`` from thread-local storage into log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = getattr(_trace_id, "value", "--------")
        return True


def set_trace_id(trace_id: str) -> None:
    """Set the trace ID for the current thread (called at request entry)."""
    _trace_id.value = trace_id


def get_trace_id() -> str:
    """Get the trace ID for the current thread."""
    return getattr(_trace_id, "value", "--------")


def setup_logging(log_config: dict) -> None:
    """
    Configure the root logger.  Safe to call multiple times
    (only the first call takes effect).

    Args:
        log_config: Dict from ``Config.get("log")``, must contain:
            - directory: str        -- log output directory
            - level: str            -- one of DEBUG/INFO/WARNING/ERROR/CRITICAL
            - format: str           -- log format string
            - date_format: str      -- date format string
            - backup_days: int      -- days to keep old log files
            - console_output: bool  -- also print to stderr
    """
    global _initialized
    with _init_lock:
        if _initialized:
            return

        log_dir = log_config.get("directory", "logs")
        log_level = getattr(logging, log_config.get("level", "INFO").upper(), logging.INFO)
        log_fmt = log_config.get(
            "format",
            "[%(asctime)s.%(msecs)03d] [%(levelname)-7s] [%(name)s:%(lineno)d] [%(trace_id)s] %(message)s",
        )
        date_fmt = log_config.get("date_format", "%Y-%m-%d %H:%M:%S")
        backup_days = log_config.get("backup_days", 30)
        console_output = log_config.get("console_output", True)

        os.makedirs(log_dir, exist_ok=True)

        formatter = logging.Formatter(log_fmt, datefmt=date_fmt)
        trace_filter = TraceIdFilter()

        # --- Main log (all levels ≥ configured level) ---
        main_path = os.path.join(log_dir, "measure_api.log")
        main_handler = TimedRotatingFileHandler(
            main_path, when="midnight", interval=1,
            backupCount=backup_days, encoding="utf-8",
        )
        main_handler.setLevel(log_level)
        main_handler.setFormatter(formatter)
        main_handler.addFilter(trace_filter)

        # --- Error-only log (ERROR and above) ---
        error_path = os.path.join(log_dir, "measure_api.error.log")
        error_handler = TimedRotatingFileHandler(
            error_path, when="midnight", interval=1,
            backupCount=backup_days, encoding="utf-8",
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        error_handler.addFilter(trace_filter)

        root = logging.getLogger()
        root.setLevel(logging.DEBUG)  # allow all levels; handlers filter
        root.addHandler(main_handler)
        root.addHandler(error_handler)

        if console_output:
            console = logging.StreamHandler(sys.stderr)
            console.setLevel(log_level)
            console.setFormatter(formatter)
            console.addFilter(trace_filter)
            root.addHandler(console)

        _initialized = True


def update_log_level(level_name: str) -> None:
    """
    Hot-update the log level on all handlers (call from
    ``Config.reload()`` path).
    """
    level = getattr(logging, level_name.upper(), logging.INFO)
    root = logging.getLogger()
    for handler in root.handlers:
        handler.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for the given module name.

    The name is prefixed with ``measure_api.`` for consistency.

    Args:
        name: Short module name, e.g. ``"project"``, ``"server"``.

    Returns:
        ``logging.Logger`` instance.
    """
    return logging.getLogger(f"measure_api.{name}")
