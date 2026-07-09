"""
Call recorder — persists API request/response data for reproducibility.

Supports synchronous (immediate write) and asynchronous (background
worker) write modes.  All writes are thread-safe.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from measure_api.logger import get_logger

logger = get_logger("call_recorder")


class CallRecorder:
    """
    Thread-safe API call recorder.

    Each call is saved as an individual JSON file and indexed in a
    per-session trace file for ordered replay.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Args:
            config: Dict from ``cfg.get("call_records", {})``.  Relevant keys:
                - enabled: bool
                - directory: str
                - write_mode: "sync" | "async"
                - async_buffer_size: int
                - async_flush_interval_s: int
                - retention_days: int
                - max_total_size_gb: float
                - record_request: bool
                - record_response: bool
                - record_visual: bool
                - copy_images_on_measure: bool
                - max_image_size_mb: int
        """
        self._config = config
        self._enabled = config.get("enabled", True)
        self._directory = config.get("directory", "call_records")
        self._write_mode = config.get("write_mode", "async")
        self._lock = threading.Lock()
        self._running = True

        # Async mode uses a background writer thread
        self._queue: "queue.Queue[Optional[Dict]]" = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        if self._enabled and self._write_mode == "async":
            self._start_worker()

        # Periodic cleanup thread
        self._cleanup_interval_s = 3600  # run every hour
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop, daemon=True, name="call-recorder-cleanup",
        )
        self._cleanup_thread.start()

        os.makedirs(self._directory, exist_ok=True)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def record(self, call_data: Dict[str, Any]) -> None:
        """
        Record an API call.

        In async mode, enqueues the data for background writing.
        In sync mode, writes immediately.
        """
        if not self._enabled:
            return

        if self._write_mode == "async":
            self._queue.put(call_data)
        else:
            self._write_sync(call_data)

    def flush(self) -> None:
        """Force-flush all pending records (async mode)."""
        if self._write_mode == "async":
            self._queue.join()
            # Give the worker a moment to finish writing
            import time
            time.sleep(0.3)

    def shutdown(self) -> None:
        """Flush and stop the background worker."""
        self._running = False
        if self._worker is not None and self._worker.is_alive():
            self._queue.put(None)  # sentinel
            self._worker.join(timeout=5)

    # ------------------------------------------------------------------
    # Internal — write
    # ------------------------------------------------------------------

    def _write_sync(self, call_data: Dict[str, Any]) -> None:
        """Immediate synchronous write (thread-safe via lock)."""
        with self._lock:
            self._do_write(call_data)

    def _do_write(self, call_data: Dict[str, Any]) -> None:
        """Unlocked internal write."""
        cfg = self._config
        session_id = call_data.get("session_id", "unknown")
        call_id = call_data.get("call_id", "unknown")
        seq = call_data.get("seq", 0)
        function = call_data.get("function", "unknown")

        # Remove large fields if configured
        if not cfg.get("record_request", True):
            call_data.pop("request", None)
        if not cfg.get("record_response", True):
            call_data.pop("response", None)

        # Build date-based subdirectory
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        date_dir = os.path.join(self._directory, date_str)
        os.makedirs(date_dir, exist_ok=True)

        # Build filename
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        # Short session for filename
        sess_short = session_id[:6] if len(session_id) > 6 else session_id
        fname = (
            f"{timestamp}_{seq:04d}_{function}_{sess_short}_{call_id}.json"
        )
        filepath = os.path.join(date_dir, fname)

        # Visual data: separate file if too large
        if not cfg.get("record_visual", False):
            call_data.pop("visual_b64", None)
            call_data.pop("visual_file", None)

        # Write JSON
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(call_data, f, indent=2, ensure_ascii=False,
                          default=str)
        except Exception as e:
            logger.error("Failed to write call record %s: %s", filepath, e)
            return

        # Append to session trace
        self._append_to_trace(date_dir, session_id, call_data, fname)

    def _append_to_trace(
        self, date_dir: str, session_id: str, call_data: Dict[str, Any], fname: str,
    ) -> None:
        """Append a call entry to the session trace file."""
        trace_path = os.path.join(date_dir, f"session_{session_id}_trace.json")

        entry = {
            "seq": call_data.get("seq", 0),
            "call_id": call_data.get("call_id", ""),
            "function": call_data.get("function", ""),
            "file": fname,
            "timestamp": call_data.get("timestamp", ""),
            "elapsed_ms": call_data.get("elapsed_ms", 0),
        }

        try:
            # Read existing trace if present (lock held by caller)
            if os.path.isfile(trace_path):
                with open(trace_path, "r", encoding="utf-8") as f:
                    trace = json.load(f)
            else:
                trace = {
                    "session_id": session_id,
                    "created_at": call_data.get("timestamp", ""),
                    "total_calls": 0,
                    "calls": [],
                }

            trace["total_calls"] += 1
            trace["calls"].append(entry)
            trace["updated_at"] = call_data.get("timestamp", "")

            with open(trace_path, "w", encoding="utf-8") as f:
                json.dump(trace, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error("Failed to update trace %s: %s", trace_path, e)

    # ------------------------------------------------------------------
    # Async worker
    # ------------------------------------------------------------------

    def _start_worker(self) -> None:
        """Start the background writer thread."""
        def worker():
            while self._running:
                try:
                    item = self._queue.get(timeout=1)
                    if item is None:  # sentinel
                        break
                    self._write_sync(item)
                    self._queue.task_done()
                except queue.Empty:
                    pass

        self._worker = threading.Thread(target=worker, daemon=True,
                                        name="call-recorder-writer")
        self._worker.start()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _cleanup_loop(self) -> None:
        """Periodic cleanup of old call records."""
        while self._running:
            time.sleep(self._cleanup_interval_s)
            try:
                self._cleanup_old()
            except Exception as e:
                logger.warning("Cleanup error: %s", e)

    def _cleanup_old(self) -> None:
        """Remove call records that exceed retention or total size limits."""
        retention_days = self._config.get("retention_days", 90)
        max_size_gb = self._config.get("max_total_size_gb", 10)
        max_size_bytes = max_size_gb * (1024**3)

        if not os.path.isdir(self._directory):
            return

        cutoff = time.time() - retention_days * 86400

        # Collect all date directories and their sizes
        entries: list[tuple[str, float]] = []  # (path, mtime)
        total_size = 0

        with self._lock:
            for name in os.listdir(self._directory):
                dir_path = os.path.join(self._directory, name)
                if not os.path.isdir(dir_path):
                    continue
                dir_mtime = os.path.getmtime(dir_path)
                dir_size = self._dir_size(dir_path)
                entries.append((dir_path, dir_mtime, dir_size))
                total_size += dir_size

            # Remove by age
            for dir_path, mtime, _ in sorted(entries, key=lambda x: x[1]):
                if mtime < cutoff:
                    shutil.rmtree(dir_path, ignore_errors=True)
                    total_size -= self._dir_size(dir_path)
                    logger.info("Cleaned up old records: %s", dir_path)

            # Remove by total size (oldest first)
            if total_size > max_size_bytes:
                for dir_path, mtime, _ in sorted(entries, key=lambda x: x[1]):
                    if total_size <= max_size_bytes:
                        break
                    dir_size = self._dir_size(dir_path)
                    shutil.rmtree(dir_path, ignore_errors=True)
                    total_size -= dir_size
                    logger.info("Cleaned up oversized records: %s", dir_path)

    @staticmethod
    def _dir_size(path: str) -> int:
        """Recursive directory size in bytes."""
        total = 0
        for dirpath, _, filenames in os.walk(path):
            for fn in filenames:
                try:
                    total += os.path.getsize(os.path.join(dirpath, fn))
                except OSError:
                    pass
        return total
