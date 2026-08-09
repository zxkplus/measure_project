"""Thin Flask adapter for the headless v2 engine application interface."""

from __future__ import annotations

import uuid
from typing import Any, Dict, Mapping, Optional, Tuple

from flask import Flask, Response, g, jsonify, request
from werkzeug.exceptions import HTTPException

from measure_api.v2.application import EngineApplication, EngineError
from measure_api.v2.image import SharedMemoryImageReader
from measure_api.v2.operations import TERMINAL_STATES


def create_production_app(
    application: EngineApplication,
    image_reader: Optional[Any] = None,
) -> Flask:
    """Create the production-only `/api/v2` HTTP adapter."""
    app = Flask("measure-api-v2")
    reader = image_reader or SharedMemoryImageReader()

    @app.before_request
    def assign_trace_id() -> None:
        g.trace_id = request.headers.get("X-Trace-Id") or uuid.uuid4().hex

    @app.after_request
    def attach_trace_id(response: Response) -> Response:
        response.headers["X-Trace-Id"] = g.trace_id
        return response

    @app.errorhandler(EngineError)
    def handle_engine_error(error: EngineError) -> Tuple[Response, int]:
        payload = dict(error.error)
        payload["trace_id"] = g.trace_id
        return jsonify({"error": payload}), _error_status(payload["code"])

    @app.errorhandler(Exception)
    def handle_unknown_error(_error: Exception) -> Tuple[Response, int]:
        return (
            jsonify(
                {
                    "error": {
                        "code": "internal.unknown",
                        "category": "internal",
                        "message": "Internal engine error",
                        "retryable": False,
                        "details": {},
                        "trace_id": g.trace_id,
                    }
                }
            ),
            500,
        )

    @app.errorhandler(HTTPException)
    def handle_http_error(error: HTTPException) -> Tuple[Response, int]:
        return (
            jsonify(
                {
                    "error": {
                        "code": "request.route_unavailable",
                        "category": "request",
                        "message": error.description,
                        "retryable": False,
                        "details": {},
                        "trace_id": g.trace_id,
                    }
                }
            ),
            int(error.code or 400),
        )

    @app.get("/api/v2/health")
    def health() -> Response:
        return jsonify({"status": "ok"})

    @app.get("/api/v2/openapi.json")
    def openapi() -> Response:
        return jsonify(openapi_document())

    @app.post("/api/v2/modeling-sessions")
    def create_session() -> Tuple[Response, int]:
        data = _json_body()
        snapshot = reader.snapshot(_mapping(data.get("image"), "image"))
        return jsonify(application.create_modeling_session(snapshot)), 201

    @app.get("/api/v2/modeling-sessions/<session_id>")
    def get_session(session_id: str) -> Response:
        return jsonify(application.get_modeling_session(session_id))

    @app.delete("/api/v2/modeling-sessions/<session_id>")
    def delete_session(session_id: str) -> Response:
        application.delete_modeling_session(session_id)
        return Response(status=204)

    @app.get("/api/v2/modeling-sessions/<session_id>/objects")
    def list_objects(session_id: str) -> Response:
        return jsonify({"objects": application.list_objects(session_id)})

    @app.post("/api/v2/modeling-sessions/<session_id>/objects")
    def create_object(session_id: str) -> Tuple[Response, int]:
        return jsonify(application.create_object(session_id, _json_body())), 201

    @app.get("/api/v2/modeling-sessions/<session_id>/objects/<object_id>")
    def get_object(session_id: str, object_id: str) -> Response:
        return jsonify(application.get_object(session_id, object_id))

    @app.put("/api/v2/modeling-sessions/<session_id>/objects/<object_id>")
    def update_object(session_id: str, object_id: str) -> Response:
        return jsonify(application.update_object(session_id, object_id, _json_body()))

    @app.delete("/api/v2/modeling-sessions/<session_id>/objects/<object_id>")
    def delete_object(session_id: str, object_id: str) -> Response:
        application.delete_object(session_id, object_id)
        return Response(status=204)

    @app.post("/api/v2/modeling-sessions/<session_id>:preview")
    def preview(session_id: str) -> Tuple[Response, int]:
        data = _json_body(required=False)
        image = None
        if "image" in data:
            image = reader.snapshot(_mapping(data["image"], "image"))
        operation = application.preview(
            session_id,
            image,
            timeout_ms=data.get("timeout_ms", 30_000),
            client_request_id=data.get("client_request_id"),
        )
        return _wait_response(application, operation, data.get("wait_ms", 0))

    @app.post("/api/v2/modeling-sessions/<session_id>:publish")
    def publish(session_id: str) -> Tuple[Response, int]:
        data = _json_body()
        operation = application.publish(
            session_id,
            _string(data, "calibration_model_id"),
            timeout_ms=data.get("timeout_ms", 60_000),
            client_request_id=data.get("client_request_id"),
        )
        return _wait_response(application, operation, data.get("wait_ms", 0))

    @app.post("/api/v2/modeling-sessions:import")
    def import_model() -> Tuple[Response, int]:
        package_file = request.files.get("model_package")
        package = package_file.read() if package_file is not None else request.get_data()
        operation = application.import_model(
            package,
            timeout_ms=_query_int("timeout_ms", 60_000),
            client_request_id=request.args.get("client_request_id"),
        )
        return _wait_response(
            application,
            operation,
            _query_int("wait_ms", 0),
        )

    @app.post("/api/v2/measurements")
    def measure() -> Tuple[Response, int]:
        data = _json_body()
        snapshot = reader.snapshot(_mapping(data.get("image"), "image"))
        operation = application.measure(
            _string(data, "calibration_model_id"),
            _string(data, "asset_ref"),
            snapshot,
            timeout_ms=data.get("timeout_ms", 10_000),
            client_request_id=data.get("client_request_id"),
        )
        return _wait_response(application, operation, data.get("wait_ms", 0))

    @app.post("/api/v2/models/<calibration_model_id>:export")
    def export_model(calibration_model_id: str) -> Tuple[Response, int]:
        data = _json_body()
        operation = application.export_model(
            calibration_model_id,
            _string(data, "asset_ref"),
            timeout_ms=data.get("timeout_ms", 60_000),
            client_request_id=data.get("client_request_id"),
        )
        return _wait_response(application, operation, data.get("wait_ms", 0))

    @app.get("/api/v2/operations")
    def list_operations() -> Response:
        return jsonify(
            {
                "operations": application.list_operations(
                    request.args.get("client_request_id")
                )
            }
        )

    @app.get("/api/v2/operations/<operation_id>")
    def get_operation(operation_id: str) -> Response:
        return jsonify(application.get_operation(operation_id))

    @app.post("/api/v2/operations/<operation_id>:cancel")
    def cancel_operation(operation_id: str) -> Response:
        return jsonify(application.cancel_operation(operation_id))

    return app


def _wait_response(
    application: EngineApplication,
    operation: Mapping[str, Any],
    wait_ms: Any,
) -> Tuple[Response, int]:
    if isinstance(wait_ms, bool) or not isinstance(wait_ms, int) or wait_ms < 0:
        raise EngineError("request.wait_invalid", "request", "wait_ms must be non-negative")
    current = application.wait_operation(str(operation["operation_id"]), wait_ms)
    if current["status"] in TERMINAL_STATES:
        return jsonify(current), 200
    response = jsonify(current)
    response.headers["Location"] = f"/api/v2/operations/{current['operation_id']}"
    return response, 202


def _json_body(*, required: bool = True) -> Dict[str, Any]:
    data = request.get_json(silent=True)
    if data is None and not required:
        return {}
    if not isinstance(data, dict):
        raise EngineError("request.json_invalid", "request", "JSON object body is required")
    return data


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EngineError("request.field_invalid", "request", f"{field} must be an object")
    return value


def _string(data: Mapping[str, Any], field: str) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value:
        raise EngineError("request.field_invalid", "request", f"{field} is required")
    return value


def _query_int(name: str, default: int) -> int:
    value = request.args.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise EngineError("request.field_invalid", "request", f"{name} must be an integer") from exc


def _error_status(code: str) -> int:
    if code.endswith(".not_found") or code == "session.expired":
        return 404
    if code in {"object.dependency_conflict", "object.label_conflict"}:
        return 409
    if code.startswith("object.parameters") or code.startswith("model.package"):
        return 422
    if code == "model.asset_unavailable":
        return 422
    if code.startswith("image.") and code not in {"image.descriptor_invalid"}:
        return 422
    if code.startswith("internal.") or code == "operation.manager_unavailable":
        return 500
    return 400


def openapi_document() -> Dict[str, Any]:
    """Return the static production contract and its accepted limits."""
    object_variants = []
    from measure_api.v2.application import _OBJECT_CONTRACTS

    for object_type, contract in _OBJECT_CONTRACTS.items():
        properties = {}
        for name in (*contract.required, *contract.defaults.keys()):
            default = contract.defaults.get(name)
            if name in {"start", "end", "center"}:
                schema: Dict[str, Any] = {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2,
                }
            elif isinstance(default, bool):
                schema = {"type": "boolean"}
            elif isinstance(default, int):
                schema = {"type": "integer"}
            elif isinstance(default, str):
                schema = {"type": "string"}
            else:
                schema = {"type": "number"}
            if name in contract.defaults:
                schema["default"] = default
            properties[name] = schema
        object_variants.append(
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["object_type", "label", "parameters"],
                "properties": {
                    "object_type": {"const": object_type},
                    "label": {"type": "string", "minLength": 1},
                    "parameters": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": list(contract.required),
                        "properties": properties,
                    },
                    "input_object_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": len(contract.input_types),
                        "maxItems": len(contract.input_types),
                    },
                },
            }
        )
    paths: Dict[str, Dict[str, Any]] = {
        path: {}
        for path in (
            "/api/v2/modeling-sessions",
            "/api/v2/modeling-sessions/{session_id}",
            "/api/v2/modeling-sessions/{session_id}/objects",
            "/api/v2/modeling-sessions/{session_id}/objects/{object_id}",
            "/api/v2/modeling-sessions/{session_id}:preview",
            "/api/v2/modeling-sessions/{session_id}:publish",
            "/api/v2/modeling-sessions:import",
            "/api/v2/measurements",
            "/api/v2/models/{calibration_model_id}:export",
            "/api/v2/operations",
            "/api/v2/operations/{operation_id}",
            "/api/v2/operations/{operation_id}:cancel",
            "/api/v2/health",
            "/api/v2/openapi.json",
        )
    }
    return {
        "openapi": "3.1.0",
        "info": {"title": "V2 Measurement Engine", "version": "2.0.0"},
        "x-first-release-limitations": [
            "POST requests are not idempotent and are never retried automatically",
            "Operations and modeling sessions do not survive process restart",
            "Only one CPU operation runs at a time; concurrent mutation is unsupported",
            "Published assets are overwritten directly without atomic rollback",
            "Shared-memory frames have no lease, generation, checksum, or quarantine",
            "The engine does not perform platform tolerance decisions or fallback",
        ],
        "paths": paths,
        "components": {
            "schemas": {
                "MeasurementObject": {"oneOf": object_variants},
                "ImageDescriptor": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "uuid",
                        "camera_sn",
                        "type",
                        "height",
                        "width",
                        "channel",
                        "jpeg_size",
                        "bin_index",
                        "timestamp",
                    ],
                    "properties": {
                        "uuid": {"type": "string", "minLength": 1},
                        "camera_sn": {"type": "string", "minLength": 1},
                        "type": {"const": "jpg"},
                        "height": {"type": "integer"},
                        "width": {"type": "integer"},
                        "channel": {"const": 1},
                        "bin_index": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 29,
                        },
                        "jpeg_size": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 20 * 1024 * 1024,
                        },
                        "timestamp": {},
                    },
                },
                "EngineError": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "code",
                        "category",
                        "message",
                        "retryable",
                        "details",
                        "trace_id",
                    ],
                    "properties": {
                        "code": {"type": "string"},
                        "category": {
                            "enum": [
                                "request",
                                "image",
                                "model",
                                "measurement",
                                "timeout",
                                "cancelled",
                                "operation",
                                "internal",
                            ]
                        },
                        "message": {"type": "string"},
                        "retryable": {"type": "boolean"},
                        "details": {"type": "object"},
                        "trace_id": {"type": "string"},
                    },
                },
                "Operation": {
                    "type": "object",
                    "required": [
                        "contract_version",
                        "operation_id",
                        "kind",
                        "status",
                        "timeout_ms",
                    ],
                    "properties": {
                        "contract_version": {"const": "2.0"},
                        "operation_id": {"type": "string"},
                        "kind": {
                            "enum": ["preview", "publish", "measure", "import", "export"]
                        },
                        "status": {
                            "enum": [
                                "queued",
                                "running",
                                "succeeded",
                                "failed",
                                "cancelled",
                                "timed_out",
                            ]
                        },
                        "timeout_ms": {"type": "integer"},
                    },
                },
                "TimeoutLimits": {
                    "type": "object",
                    "properties": {
                        "measurement_ms": {"default": 10_000, "maximum": 60_000},
                        "preview_ms": {"default": 30_000, "maximum": 120_000},
                        "model_ms": {"default": 60_000, "maximum": 300_000},
                    },
                },
            }
        },
    }
