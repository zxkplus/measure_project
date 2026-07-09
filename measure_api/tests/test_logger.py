"""Tests for logger module."""

from __future__ import annotations

import logging
import os
import tempfile
import threading

import pytest

from measure_api.logger import (
    TraceIdFilter,
    get_logger,
    get_trace_id,
    set_trace_id,
    setup_logging,
    update_log_level,
)


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging state before each test."""
    import measure_api.logger as log_mod
    log_mod._initialized = False
    # Remove any handlers added by previous tests
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    yield


@pytest.fixture
def log_dir():
    """Create a temporary log directory."""
    with tempfile.TemporaryDirectory() as tmp:
        yield tmp


def test_setup_and_write(log_dir):
    """Logger writes to file."""
    setup_logging({
        "directory": log_dir,
        "level": "DEBUG",
        "console_output": False,
        "backup_days": 1,
    })
    logger = get_logger("test")
    logger.info("hello world")

    # Check file existence
    main_log = os.path.join(log_dir, "measure_api.log")
    assert os.path.isfile(main_log)
    content = open(main_log).read()
    assert "hello world" in content
    assert "measure_api.test" in content


def test_level_filtering(log_dir):
    """DEBUG messages filtered at INFO level."""
    setup_logging({
        "directory": log_dir,
        "level": "INFO",
        "console_output": False,
        "backup_days": 1,
    })
    logger = get_logger("test")
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")

    main_log = os.path.join(log_dir, "measure_api.log")
    content = open(main_log).read()
    assert "debug message" not in content
    assert "info message" in content
    assert "warning message" in content


def test_error_log_separate(log_dir):
    """ERROR-level messages appear in error log file."""
    setup_logging({
        "directory": log_dir,
        "level": "DEBUG",
        "console_output": False,
        "backup_days": 1,
    })
    logger = get_logger("test")
    logger.warning("warning message")
    logger.error("error message")

    error_log = os.path.join(log_dir, "measure_api.error.log")
    assert os.path.isfile(error_log)
    content = open(error_log).read()
    assert "error message" in content
    assert "warning message" not in content


def test_trace_id_injection(log_dir):
    """Log records include trace_id."""
    setup_logging({
        "directory": log_dir,
        "level": "INFO",
        "console_output": False,
        "backup_days": 1,
    })
    set_trace_id("trk-test123")
    logger = get_logger("test")
    logger.info("check trace id")

    main_log = os.path.join(log_dir, "measure_api.log")
    content = open(main_log).read()
    assert "[trk-test123]" in content or "trk-test123" in content


def test_get_set_trace_id():
    """get_trace_id/set_trace_id round-trip."""
    set_trace_id("abc")
    assert get_trace_id() == "abc"
    set_trace_id("def")
    assert get_trace_id() == "def"


def test_thread_safe_trace_id():
    """Trace ID is per-thread (thread-local)."""
    set_trace_id("main")

    results = {}
    def worker():
        set_trace_id("worker")
        results["tid"] = get_trace_id()

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    assert get_trace_id() == "main"
    assert results["tid"] == "worker"


def test_thread_safe_concurrent_writes(log_dir):
    """Concurrent log writes do not interleave corruptly."""
    setup_logging({
        "directory": log_dir,
        "level": "DEBUG",
        "console_output": False,
        "backup_days": 1,
    })

    n_threads = 10
    n_lines = 50

    def worker(tid: int):
        logger = get_logger(f"worker{tid}")
        for i in range(n_lines):
            logger.info("thread=%d line=%d", tid, i)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    main_log = os.path.join(log_dir, "measure_api.log")
    content = open(main_log).read()
    # All lines should be present
    for tid in range(n_threads):
        for i in range(n_lines):
            assert f"thread={tid} line={i}" in content


def test_update_log_level(log_dir):
    """Hot-update log level works."""
    setup_logging({
        "directory": log_dir,
        "level": "WARNING",
        "console_output": False,
        "backup_days": 1,
    })
    logger = get_logger("test")
    logger.info("should be filtered")
    logger.warning("should appear")

    # Switch to INFO
    update_log_level("INFO")
    logger.info("now visible")

    main_log = os.path.join(log_dir, "measure_api.log")
    content = open(main_log).read()
    assert "should be filtered" not in content
    assert "should appear" in content
    assert "now visible" in content
