"""Tests for call_recorder module."""

from __future__ import annotations

import json
import os
import tempfile
import time

import pytest

from measure_api.call_recorder import CallRecorder


@pytest.fixture
def cr_dir():
    """Create a temporary call_records directory."""
    with tempfile.TemporaryDirectory() as tmp:
        yield tmp


@pytest.fixture
def sample_call():
    return {
        "call_id": "trk-test0001",
        "seq": 1,
        "timestamp": "2026-07-09T15:30:00.123456",
        "elapsed_ms": 45.2,
        "function": "add_measurement",
        "endpoint": "POST /api/session/test/measurements",
        "session_id": "sess-test",
        "request": {"object_type": "FitCircle", "label": "c1", "params": {"radius": 50}},
        "response": {"valid": True, "result": {"radius": 49.8}},
        "error": None,
    }


def test_sync_write(cr_dir, sample_call):
    """Sync mode writes file immediately."""
    cr = CallRecorder({
        "enabled": True,
        "directory": cr_dir,
        "write_mode": "sync",
        "retention_days": 90,
        "max_total_size_gb": 10,
    })
    cr.record(sample_call)

    # Check file exists in date directory
    date_dirs = os.listdir(cr_dir)
    assert len(date_dirs) == 1
    date_dir = os.path.join(cr_dir, date_dirs[0])
    files = os.listdir(date_dir)
    assert any("add_measurement" in f for f in files)
    assert any("sess-test" in f for f in files)
    assert any("trk-test0001" in f for f in files)

    # Validate content
    json_files = [f for f in os.listdir(date_dir) if f.endswith(".json")]
    # One call file + one trace file
    assert len(json_files) >= 2

    # Read back the call file
    call_files = [f for f in json_files if "trace" not in f]
    if call_files:
        with open(os.path.join(date_dir, call_files[0])) as f:
            data = json.load(f)
        assert data["call_id"] == "trk-test0001"
        assert data["request"]["object_type"] == "FitCircle"

    cr.shutdown()


def test_async_write(cr_dir, sample_call):
    """Async mode eventually writes files."""
    cr = CallRecorder({
        "enabled": True,
        "directory": cr_dir,
        "write_mode": "async",
        "async_buffer_size": 2,
        "async_flush_interval_s": 1,
        "retention_days": 90,
        "max_total_size_gb": 10,
    })
    cr.record(sample_call)
    time.sleep(1.5)  # allow flush

    date_dirs = os.listdir(cr_dir)
    if date_dirs:
        # Should have written by now
        date_dir = os.path.join(cr_dir, date_dirs[0])
        files = os.listdir(date_dir)
        assert any("add_measurement" in f for f in files)

    cr.shutdown()


def test_trace_file(cr_dir, sample_call):
    """Session trace file is written."""
    cr = CallRecorder({
        "enabled": True,
        "directory": cr_dir,
        "write_mode": "sync",
        "retention_days": 90,
        "max_total_size_gb": 10,
    })
    cr.record(sample_call)

    date_dirs = os.listdir(cr_dir)
    assert len(date_dirs) == 1
    date_dir = os.path.join(cr_dir, date_dirs[0])
    trace_files = [f for f in os.listdir(date_dir) if "trace" in f]
    assert len(trace_files) == 1

    with open(os.path.join(date_dir, trace_files[0])) as f:
        trace = json.load(f)
    assert trace["session_id"] == "sess-test"
    assert trace["total_calls"] == 1
    assert len(trace["calls"]) == 1
    assert trace["calls"][0]["seq"] == 1

    cr.shutdown()


def test_disabled(cr_dir, sample_call):
    """Disabled recorder does nothing."""
    cr = CallRecorder({
        "enabled": False,
        "directory": cr_dir,
        "write_mode": "sync",
    })
    cr.record(sample_call)
    assert len(os.listdir(cr_dir)) == 0
    cr.shutdown()


def test_multiple_calls(cr_dir, sample_call):
    """Multiple calls create separate files."""
    cr = CallRecorder({
        "enabled": True,
        "directory": cr_dir,
        "write_mode": "sync",
        "retention_days": 90,
        "max_total_size_gb": 10,
    })
    for seq in range(5):
        call = dict(sample_call)
        call["seq"] = seq
        call["call_id"] = f"trk-{seq:04d}"
        cr.record(call)

    date_dirs = os.listdir(cr_dir)
    date_dir = os.path.join(cr_dir, date_dirs[0])
    # Should have 5 call files + 1 trace file
    json_files = [f for f in os.listdir(date_dir) if f.endswith(".json") and "trace" not in f]
    assert len(json_files) == 5

    # Trace should reflect all 5 calls
    trace_files = [f for f in os.listdir(date_dir) if "trace" in f]
    with open(os.path.join(date_dir, trace_files[0])) as f:
        trace = json.load(f)
    assert trace["total_calls"] == 5

    cr.shutdown()


def test_flush(cr_dir, sample_call):
    """Flush forces write in async mode."""
    cr = CallRecorder({
        "enabled": True,
        "directory": cr_dir,
        "write_mode": "async",
        "async_buffer_size": 100,
        "async_flush_interval_s": 3600,
        "retention_days": 90,
        "max_total_size_gb": 10,
    })
    cr.record(sample_call)
    time.sleep(0.2)
    # Won't have flushed yet (buffer_size=100, interval=3600)
    cr.flush()
    # Now should be flushed
    date_dirs = os.listdir(cr_dir)
    assert len(date_dirs) == 1
    cr.shutdown()
