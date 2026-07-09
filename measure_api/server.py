"""
Flask REST server for the Measure API.

All endpoints return JSON.  Each request receives an ``X-Trace-Id`` header
for log/call-record correlation.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Dict, Optional

from flask import Flask, Response, g, jsonify, request

from measure_api.call_recorder import CallRecorder
from measure_api.config import Config
from measure_api.logger import get_logger, set_trace_id
from measure_api.project import MeasureProject

logger = get_logger("server")

# ---------------------------------------------------------------------------
# Session manager
# ---------------------------------------------------------------------------


class SessionManager:
    """In-memory session management for active MeasureProject instances."""

    def __init__(self, max_sessions: int = 10) -> None:
        self._sessions: Dict[str, MeasureProject] = {}
        self._seq_counters: Dict[str, int] = {}
        self._max_sessions = max_sessions

    def create(self, project_dir: str) -> str:
        """Create a new session.  Returns session_id (UUID)."""
        if len(self._sessions) >= self._max_sessions:
            raise RuntimeError(
                f"Maximum sessions ({self._max_sessions}) reached."
            )
        sid = uuid.uuid4().hex[:12]
        project = MeasureProject(os.path.abspath(project_dir))
        self._sessions[sid] = project
        self._seq_counters[sid] = 0
        logger.info("Session created: %s -> %s", sid, project_dir)
        return sid

    def get(self, sid: str) -> MeasureProject:
        """Get a project by session ID.  Raises KeyError if not found."""
        if sid not in self._sessions:
            raise KeyError(f"Session '{sid}' not found")
        return self._sessions[sid]

    def delete(self, sid: str) -> bool:
        """Delete a session.  Returns True if found and removed."""
        if sid in self._sessions:
            del self._sessions[sid]
            self._seq_counters.pop(sid, None)
            logger.info("Session deleted: %s", sid)
            return True
        return False

    def get_next_seq(self, sid: str) -> int:
        """Increment and return the call sequence number for a session."""
        self._seq_counters[sid] = self._seq_counters.get(sid, 0) + 1
        return self._seq_counters[sid]

    def list_sessions(self) -> Dict[str, str]:
        """Return {sid: phase} for all active sessions."""
        return {sid: proj.phase for sid, proj in self._sessions.items()}


# ---------------------------------------------------------------------------
# Flask app factory
# ---------------------------------------------------------------------------


def create_app(cfg: Optional[Config] = None) -> Flask:
    """
    Create and configure the Flask application.

    Args:
        cfg: Config instance.  If None, loaded via ``Config.load()``.
    """
    if cfg is None:
        cfg = Config.load()

    app = Flask(__name__)
    app.config["cfg"] = cfg

    # Session manager (shared across requests)
    max_sessions = cfg.get("server.max_sessions", 10)
    session_manager = SessionManager(max_sessions=max_sessions)
    app.config["session_manager"] = session_manager

    # Call recorder
    call_recorder = CallRecorder(cfg.get("call_records", {}))
    app.config["call_recorder"] = call_recorder

    # ------------------------------------------------------------------
    # Middleware
    # ------------------------------------------------------------------

    @app.before_request
    def _assign_trace_id() -> None:
        """Assign a trace ID for the current request."""
        trace_id = f"trk-{uuid.uuid4().hex[:12]}"
        g.trace_id = trace_id
        g.start_time = time.perf_counter()
        set_trace_id(trace_id)

    @app.after_request
    def _attach_trace_header(response: Response) -> Response:
        """Attach X-Trace-Id to every response."""
        trace_id = getattr(g, "trace_id", "--------")
        response.headers["X-Trace-Id"] = trace_id
        return response

    @app.teardown_request
    def _record_call(exc=None) -> None:
        """Record the API call (request + response + timing)."""
        # Handled inline in each endpoint via _make_response

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    @app.route("/api/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "ok",
            "sessions": len(session_manager.list_sessions()),
        })

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    @app.route("/api/session", methods=["POST"])
    def create_session():
        data = request.get_json(silent=True) or {}
        project_dir = data.get("project_dir", "")
        if not project_dir:
            return jsonify({"error": "project_dir is required"}), 400
        try:
            sid = session_manager.create(project_dir)
            return _ok({"session_id": sid, "status": "created"})
        except Exception as e:
            return _err(str(e), 400)

    @app.route("/api/session/<sid>", methods=["GET"])
    def get_session(sid: str):
        try:
            proj = session_manager.get(sid)
            return _ok(proj.status())
        except KeyError as e:
            return _err(str(e), 404)

    @app.route("/api/session/<sid>", methods=["DELETE"])
    def delete_session(sid: str):
        if session_manager.delete(sid):
            return _ok({"status": "deleted", "session_id": sid})
        return _err(f"Session '{sid}' not found", 404)

    @app.route("/api/sessions", methods=["GET"])
    def list_sessions():
        return _ok({"sessions": session_manager.list_sessions()})

    # ------------------------------------------------------------------
    # Reference image
    # ------------------------------------------------------------------

    @app.route("/api/session/<sid>/reference", methods=["POST"])
    def load_reference(sid: str):
        try:
            proj = session_manager.get(sid)
        except KeyError as e:
            return _err(str(e), 404)
        data = request.get_json(silent=True) or {}
        image_path = data.get("image_path", "")
        if not image_path:
            return _err("image_path is required", 400)
        try:
            result = proj.load_reference(image_path)
            return _record_and_respond(session_manager, call_recorder, sid, result)
        except Exception as e:
            return _err(str(e), 400)

    # ------------------------------------------------------------------
    # Template
    # ------------------------------------------------------------------

    @app.route("/api/session/<sid>/template", methods=["POST"])
    def set_template(sid: str):
        try:
            proj = session_manager.get(sid)
        except KeyError as e:
            return _err(str(e), 404)
        data = request.get_json(silent=True) or {}
        try:
            result = proj.set_template(
                center=tuple(data.get("center", (0, 0))),
                size=tuple(data.get("size", (0, 0))),
                angle_deg=float(data.get("angle_deg", 0)),
                preprocessor=data.get("preprocessor", "raw"),
                match_score_threshold=float(data.get("match_score_threshold", 0.5)),
                angle_range_deg=float(data.get("angle_range_deg", 30)),
                max_matches=int(data.get("max_matches", 0)),
            )
            return _record_and_respond(session_manager, call_recorder, sid, result)
        except Exception as e:
            return _err(str(e), 400)

    # ------------------------------------------------------------------
    # Measurement CRUD
    # ------------------------------------------------------------------

    @app.route("/api/session/<sid>/measurements", methods=["POST"])
    def add_measurement(sid: str):
        try:
            proj = session_manager.get(sid)
        except KeyError as e:
            return _err(str(e), 404)
        data = request.get_json(silent=True) or {}
        object_type = data.get("object_type", "")
        label = data.get("label", "")
        params = data.get("params", {})
        include_visual = request.args.get("include_visual", "false").lower() == "true"
        if not object_type or not label:
            return _err("object_type and label are required", 400)
        try:
            result = proj.add_measurement(object_type, label, params,
                                          include_visual=include_visual)
            return _record_and_respond(session_manager, call_recorder, sid, result)
        except Exception as e:
            return _err(str(e), 400)

    @app.route("/api/session/<sid>/measurements/<label>", methods=["PUT"])
    def update_measurement(sid: str, label: str):
        try:
            proj = session_manager.get(sid)
        except KeyError as e:
            return _err(str(e), 404)
        data = request.get_json(silent=True) or {}
        params = data.get("params", {})
        include_visual = request.args.get("include_visual", "false").lower() == "true"
        try:
            result = proj.update_measurement(label, params,
                                              include_visual=include_visual)
            return _record_and_respond(session_manager, call_recorder, sid, result)
        except Exception as e:
            return _err(str(e), 400)

    @app.route("/api/session/<sid>/measurements/<label>", methods=["DELETE"])
    def delete_measurement(sid: str, label: str):
        try:
            proj = session_manager.get(sid)
        except KeyError as e:
            return _err(str(e), 404)
        try:
            result = proj.remove_measurement(label)
            return _record_and_respond(session_manager, call_recorder, sid, result)
        except Exception as e:
            return _err(str(e), 400)

    @app.route("/api/session/<sid>/measurements", methods=["GET"])
    def list_measurements(sid: str):
        try:
            proj = session_manager.get(sid)
        except KeyError as e:
            return _err(str(e), 404)
        return _ok(proj.list_measurements())

    @app.route("/api/session/<sid>/measurements/<label>", methods=["GET"])
    def get_measurement(sid: str, label: str):
        try:
            proj = session_manager.get(sid)
        except KeyError as e:
            return _err(str(e), 404)
        try:
            result = proj.get_measurement(label)
            return _ok(result)
        except ValueError as e:
            return _err(str(e), 404)

    @app.route("/api/session/<sid>/measurements/test", methods=["POST"])
    def test_measurement(sid: str):
        try:
            proj = session_manager.get(sid)
        except KeyError as e:
            return _err(str(e), 404)
        data = request.get_json(silent=True) or {}
        object_type = data.get("object_type", "")
        label = data.get("label", "test")
        params = data.get("params", {})
        include_visual = request.args.get("include_visual", "false").lower() == "true"
        if not object_type:
            return _err("object_type is required", 400)
        try:
            result = proj.test_measurement(object_type, label, params,
                                           include_visual=include_visual)
            return _record_and_respond(session_manager, call_recorder, sid, result)
        except Exception as e:
            return _err(str(e), 400)

    # ------------------------------------------------------------------
    # Composed measurements
    # ------------------------------------------------------------------

    @app.route("/api/session/<sid>/composed", methods=["POST"])
    def add_composed(sid: str):
        try:
            proj = session_manager.get(sid)
        except KeyError as e:
            return _err(str(e), 404)
        data = request.get_json(silent=True) or {}
        composed_type = data.get("composed_type", "")
        label = data.get("label", "")
        deps = data.get("dependencies", {})
        include_visual = request.args.get("include_visual", "false").lower() == "true"
        if not composed_type or not label:
            return _err("composed_type and label are required", 400)
        try:
            result = proj.add_composed(composed_type, label, deps,
                                       include_visual=include_visual)
            return _record_and_respond(session_manager, call_recorder, sid, result)
        except Exception as e:
            return _err(str(e), 400)

    @app.route("/api/session/<sid>/composed/<label>", methods=["DELETE"])
    def delete_composed(sid: str, label: str):
        try:
            proj = session_manager.get(sid)
        except KeyError as e:
            return _err(str(e), 404)
        try:
            result = proj.remove_composed(label)
            return _record_and_respond(session_manager, call_recorder, sid, result)
        except Exception as e:
            return _err(str(e), 400)

    @app.route("/api/session/<sid>/composed", methods=["GET"])
    def list_composed(sid: str):
        try:
            proj = session_manager.get(sid)
        except KeyError as e:
            return _err(str(e), 404)
        return _ok(proj.list_composed())

    # ------------------------------------------------------------------
    # DAG
    # ------------------------------------------------------------------

    @app.route("/api/session/<sid>/dag", methods=["GET"])
    def dag(sid: str):
        try:
            proj = session_manager.get(sid)
        except KeyError as e:
            return _err(str(e), 404)
        fmt = request.args.get("format", "json")
        return _ok(proj.get_dag(format=fmt))

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    @app.route("/api/session/<sid>/save", methods=["POST"])
    def save_project(sid: str):
        try:
            proj = session_manager.get(sid)
        except KeyError as e:
            return _err(str(e), 404)
        try:
            result = proj.save()
            return _record_and_respond(session_manager, call_recorder, sid, result)
        except Exception as e:
            return _err(str(e), 400)

    @app.route("/api/session/<sid>/load", methods=["POST"])
    def load_project(sid: str):
        try:
            proj = session_manager.get(sid)
        except KeyError as e:
            return _err(str(e), 404)
        try:
            result = proj.load()
            return _record_and_respond(session_manager, call_recorder, sid, result)
        except Exception as e:
            return _err(str(e), 400)

    # ------------------------------------------------------------------
    # Measure
    # ------------------------------------------------------------------

    @app.route("/api/session/<sid>/measure", methods=["POST"])
    def measure(sid: str):
        try:
            proj = session_manager.get(sid)
        except KeyError as e:
            return _err(str(e), 404)
        data = request.get_json(silent=True) or {}
        inspection_image = data.get("inspection_image", "")
        include_visual = request.args.get("include_visual", "false").lower() == "true"
        if not inspection_image:
            return _err("inspection_image is required", 400)
        try:
            result = proj.measure(inspection_image, include_visual=include_visual)
            return _record_and_respond(session_manager, call_recorder, sid, result)
        except Exception as e:
            return _err(str(e), 400)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ok(data: dict) -> Response:
        """Return a 200 JSON response."""
        response = jsonify(data)
        return response

    def _err(message: str, status: int = 400) -> Response:
        """Return an error JSON response."""
        logger.warning("[%s] Error %d: %s", getattr(g, "trace_id", "?"), status, message)
        return jsonify({"error": message}), status

    def _record_and_respond(
        sm: SessionManager,
        recorder: CallRecorder,
        sid: str,
        data: dict,
    ) -> Response:
        """
        Record the API call and return the response.

        This wraps around the SDK call result to persist request/response
        into the call recorder.
        """
        trace_id = getattr(g, "trace_id", "--------")
        start_time = getattr(g, "start_time", time.perf_counter())
        elapsed = (time.perf_counter() - start_time) * 1000
        seq = sm.get_next_seq(sid)

        # Build call record
        call_record = {
            "call_id": trace_id,
            "seq": seq,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.") +
                         f"{time.perf_counter() * 1000 % 1000:06.0f}",
            "elapsed_ms": round(elapsed, 1),
            "function": request.endpoint or "unknown",
            "endpoint": f"{request.method} {request.path}",
            "session_id": sid,
            "request": request.get_json(silent=True) or {},
            "response": data,
            "error": None,
        }

        try:
            recorder.record(call_record)
        except Exception as e:
            logger.warning("Call recording failed: %s", e)

        return jsonify(data)

    return app
