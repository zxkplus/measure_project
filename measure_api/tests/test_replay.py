"""Tests for SessionReplay."""

from __future__ import annotations

import json
import os
import tempfile

import cv2
import numpy as np
import pytest

from measure_api.replay import SessionReplay


@pytest.fixture
def project_dir():
    """Create a temp directory with sample images."""
    tmp = tempfile.mkdtemp(prefix="measure_api_replay_test_")
    img = np.zeros((100, 100), dtype=np.uint8) + 100
    cv2.circle(img, (50, 50), 30, 200, -1)
    cv2.imwrite(os.path.join(tmp, "ref.png"), img)
    return tmp


@pytest.fixture
def session_trace_path(project_dir):
    """Create a sample session trace with call records."""
    date_dir = os.path.join(project_dir, "call_records", "2026-07-09")
    os.makedirs(date_dir, exist_ok=True)

    # Create call files
    calls = [
        {
            "call_id": "trk-0001",
            "seq": 1,
            "timestamp": "2026-07-09T10:00:00",
            "elapsed_ms": 10.0,
            "function": "create_session",
            "endpoint": "POST /api/session",
            "session_id": "sess-replay-test",
            "request": {"project_dir": project_dir},
            "response": {"session_id": "sess-replay-test", "status": "created"},
            "error": None,
        },
        {
            "call_id": "trk-0002",
            "seq": 2,
            "timestamp": "2026-07-09T10:00:01",
            "elapsed_ms": 20.0,
            "function": "load_reference",
            "endpoint": "POST /api/session/sess-replay-test/reference",
            "session_id": "sess-replay-test",
            "request": {"image_path": os.path.join(project_dir, "ref.png")},
            "response": {"width": 100, "height": 100},
            "error": None,
        },
        {
            "call_id": "trk-0003",
            "seq": 3,
            "timestamp": "2026-07-09T10:00:02",
            "elapsed_ms": 30.0,
            "function": "set_template",
            "endpoint": "POST /api/session/sess-replay-test/template",
            "session_id": "sess-replay-test",
            "request": {
                "center": [50, 50],
                "size": [60, 60],
                "angle_deg": 0,
            },
            "response": {"template_shape": [60, 60]},
            "error": None,
        },
    ]

    for call in calls:
        call_id = call["call_id"]
        fname = f"20260709_100000_{call['seq']:04d}_{call['function']}_{call_id}.json"
        with open(os.path.join(date_dir, fname), "w") as f:
            json.dump(call, f)

    # Create trace file
    trace = {
        "session_id": "sess-replay-test",
        "created_at": "2026-07-09T10:00:00",
        "total_calls": 3,
        "calls": [
            {"seq": 1, "call_id": "trk-0001", "function": "create_session",
             "file": "20260709_100000_0001_create_session_trk-0001.json",
             "timestamp": "2026-07-09T10:00:00", "elapsed_ms": 10.0},
            {"seq": 2, "call_id": "trk-0002", "function": "load_reference",
             "file": "20260709_100000_0002_load_reference_trk-0002.json",
             "timestamp": "2026-07-09T10:00:01", "elapsed_ms": 20.0},
            {"seq": 3, "call_id": "trk-0003", "function": "set_template",
             "file": "20260709_100000_0003_set_template_trk-0003.json",
             "timestamp": "2026-07-09T10:00:02", "elapsed_ms": 30.0},
        ],
    }
    trace_path = os.path.join(date_dir, "session_sess-replay-test_trace.json")
    with open(trace_path, "w") as f:
        json.dump(trace, f)

    return trace_path


def test_from_trace(session_trace_path):
    """Load trace from file."""
    replay = SessionReplay.from_trace(session_trace_path)
    assert replay.session_id == "sess-replay-test"
    assert replay.total_calls == 3


def test_from_trace_not_found():
    """Missing trace file raises error."""
    with pytest.raises(FileNotFoundError):
        SessionReplay.from_trace("/nonexistent/trace.json")


def test_replay_all(session_trace_path):
    """Replay all calls."""
    replay = SessionReplay.from_trace(session_trace_path)
    results = replay.replay_all()
    assert len(results) == 3
    # All should succeed
    assert all(r["error"] is None for r in results)
    # Verify set_template (last call) template shape
    assert results[-1]["response"]["template_shape"] == [60, 60]


def test_replay_upto_seq(session_trace_path):
    """Replay up to sequence number 2."""
    replay = SessionReplay.from_trace(session_trace_path)
    results = replay.replay_upto(seq=2)
    assert len(results) == 2
    assert results[0]["seq"] == 1
    assert results[1]["seq"] == 2


def test_replay_upto_call_id(session_trace_path):
    """Replay up to a specific call_id."""
    replay = SessionReplay.from_trace(session_trace_path)
    results = replay.replay_upto(call_id="trk-0002")
    assert len(results) == 2
    assert results[-1]["call_id"] == "trk-0002"


def test_replay_function(session_trace_path):
    """Replay only matching calls."""
    replay = SessionReplay.from_trace(session_trace_path)
    results = replay.replay_function("load_reference")
    assert len(results) >= 1
    assert all(r["function"] == "load_reference" for r in results)


def test_calls_property(session_trace_path):
    """Calls property returns list of call entries."""
    replay = SessionReplay.from_trace(session_trace_path)
    assert len(replay.calls) == 3
    assert replay.calls[0]["seq"] == 1
    assert replay.calls[0]["function"] == "create_session"
