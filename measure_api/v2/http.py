"""Thin Flask adapter for the headless v2 engine application interface."""

from __future__ import annotations

import time
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
    """Create the production-only `/api/v2` HTTP adapter.

    Args:
        application: Headless engine application service.
        image_reader: Optional shared-memory image reader implementation.

    Returns:
        A configured Flask application exposing only the v2 contract.
    """
    app = Flask("measure-api-v2")
    reader = image_reader or SharedMemoryImageReader()

    @app.before_request
    def assign_trace_id() -> None:
        g.accepted_monotonic = time.monotonic()
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
        _reject_unknown_fields(data, {"image"})
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
        _reject_unknown_fields(
            data,
            {"image", "timeout_ms", "wait_ms", "client_request_id"},
        )
        image = None
        if "image" in data:
            image = reader.snapshot(_mapping(data["image"], "image"))
        operation = application.preview(
            session_id,
            image,
            timeout_ms=data.get("timeout_ms", 30_000),
            client_request_id=data.get("client_request_id"),
            accepted_monotonic=g.accepted_monotonic,
        )
        return _wait_response(application, operation, data.get("wait_ms", 0))

    @app.post("/api/v2/modeling-sessions/<session_id>:publish")
    def publish(session_id: str) -> Tuple[Response, int]:
        data = _json_body()
        _reject_unknown_fields(
            data,
            {"calibration_model_id", "timeout_ms", "wait_ms", "client_request_id"},
        )
        operation = application.publish(
            session_id,
            _string(data, "calibration_model_id"),
            timeout_ms=data.get("timeout_ms", 60_000),
            client_request_id=data.get("client_request_id"),
            accepted_monotonic=g.accepted_monotonic,
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
            accepted_monotonic=g.accepted_monotonic,
        )
        return _wait_response(
            application,
            operation,
            _query_int("wait_ms", 0),
        )

    @app.post("/api/v2/measurements")
    def measure() -> Tuple[Response, int]:
        data = _json_body()
        _reject_unknown_fields(
            data,
            {
                "calibration_model_id",
                "asset_ref",
                "image",
                "timeout_ms",
                "wait_ms",
                "client_request_id",
            },
        )
        snapshot = reader.snapshot(_mapping(data.get("image"), "image"))
        operation = application.measure(
            _string(data, "calibration_model_id"),
            _string(data, "asset_ref"),
            snapshot,
            timeout_ms=data.get("timeout_ms", 10_000),
            client_request_id=data.get("client_request_id"),
            accepted_monotonic=g.accepted_monotonic,
        )
        return _wait_response(application, operation, data.get("wait_ms", 0))

    @app.post("/api/v2/models/<calibration_model_id>:export")
    def export_model(calibration_model_id: str) -> Tuple[Response, int]:
        data = _json_body()
        _reject_unknown_fields(
            data,
            {"asset_ref", "timeout_ms", "wait_ms", "client_request_id"},
        )
        operation = application.export_model(
            calibration_model_id,
            _string(data, "asset_ref"),
            timeout_ms=data.get("timeout_ms", 60_000),
            client_request_id=data.get("client_request_id"),
            accepted_monotonic=g.accepted_monotonic,
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


def _reject_unknown_fields(data: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise EngineError(
            "request.field_unknown",
            "request",
            "Request contains unknown fields",
            details={"fields": sorted(unknown)},
        )


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


def _parameter_schema(name: str, default: Any, has_default: bool) -> Dict[str, Any]:
    if name in {"start", "end", "center"}:
        schema: Dict[str, Any] = {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 2,
            "maxItems": 2,
        }
    elif name in {"radius_min", "radius_max"}:
        schema = {"anyOf": [{"type": "number"}, {"type": "null"}]}
    elif name == "use_subpixel":
        schema = {"type": "boolean"}
    elif name in {"num_measures", "template_size"}:
        schema = {"type": "integer"}
    elif name == "transition":
        schema = {"type": "string", "enum": ["all", "positive", "negative"]}
    elif name == "interpolation":
        schema = {"type": "string", "enum": ["linear", "nearest", "cubic"]}
    elif name == "preprocessor_type":
        schema = {
            "type": "string",
            "enum": ["raw", "canny", "sobel", "clahe", "threshold"],
        }
    else:
        schema = {"type": "number"}
    if name in {
        "length1",
        "length2",
        "measure_length1",
        "measure_length2",
        "radius",
        "sigma",
        "template_size",
        "angle_step",
    }:
        schema["exclusiveMinimum"] = 0
    if name == "threshold":
        schema["minimum"] = 0
    if name == "num_measures":
        schema["minimum"] = 2
    if has_default:
        schema["default"] = default
    return schema


def _content(schema: Mapping[str, Any]) -> Dict[str, Any]:
    return {"application/json": {"schema": dict(schema)}}


def _responses(
    success_schema: Optional[Mapping[str, Any]] = None,
    *,
    created: bool = False,
    accepted: bool = False,
    empty: bool = False,
) -> Dict[str, Any]:
    code = "201" if created else "204" if empty else "200"
    response: Dict[str, Any] = {"description": "Success"}
    if success_schema is not None:
        response["content"] = _content(success_schema)
    responses = {code: response}
    if accepted:
        responses["202"] = {
            "description": "Operation accepted",
            "headers": {
                "Location": {
                    "required": True,
                    "schema": {"type": "string"},
                }
            },
            "content": _content({"$ref": "#/components/schemas/Operation"}),
        }
    responses["default"] = {
        "description": "Structured engine error",
        "content": _content({"$ref": "#/components/schemas/ErrorEnvelope"}),
    }
    return responses


def openapi_document() -> Dict[str, Any]:
    """Return the static production contract and its accepted limits.

    Returns:
        An OpenAPI-compatible JSON document for the v2 HTTP surface.
    """
    from measure_api.v2.application import _OBJECT_CONTRACTS

    object_create_variants = []
    object_response_variants = []
    object_result_variants = []
    parameter_variants = []
    observation_refs = {
        "point": "PointObservation",
        "line": "LineObservation",
        "circle": "CircleObservation",
        "distance": "ScalarObservation",
        "angle": "ScalarObservation",
    }
    for object_type, contract in _OBJECT_CONTRACTS.items():
        properties = {
            name: _parameter_schema(
                name,
                contract.defaults.get(name),
                name in contract.defaults,
            )
            for name in (*contract.required, *contract.defaults.keys())
        }
        if object_type == "FitCircle":
            properties["num_measures"]["minimum"] = 3
        parameters = {
            "type": "object",
            "additionalProperties": False,
            "required": list(contract.required),
            "properties": properties,
        }
        parameter_variants.append(parameters)
        required = ["object_type", "label", "parameters"]
        if contract.input_types:
            required.append("input_object_ids")
        variant: Dict[str, Any] = {
            "type": "object",
            "additionalProperties": False,
            "required": required,
            "properties": {
                "object_type": {"const": object_type},
                "label": {"type": "string", "minLength": 1},
                "parameters": parameters,
                "input_object_ids": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "minItems": len(contract.input_types),
                    "maxItems": len(contract.input_types),
                    "uniqueItems": True,
                    "default": [],
                },
            },
            "x-input-result-types": list(contract.input_types),
            "x-result-type": contract.result_type,
        }
        object_create_variants.append(variant)
        response_variant = {
            **variant,
            "required": ["object_id", *variant["required"]],
            "properties": {
                "object_id": {"type": "string", "minLength": 1, "readOnly": True},
                **variant["properties"],
            },
        }
        object_response_variants.append(response_variant)
        observation_ref = observation_refs[contract.result_type]
        object_result_variants.append(
            {
                "allOf": [
                    {"$ref": "#/components/schemas/MeasurementObjectResultBase"},
                    {
                        "type": "object",
                        "properties": {
                            "object_type": {"const": object_type},
                            "observation": {
                                "anyOf": [
                                    {"$ref": f"#/components/schemas/{observation_ref}"},
                                    {"type": "null"},
                                ]
                            },
                        },
                    },
                ]
            }
        )

    def schema_ref(name: str) -> Dict[str, str]:
        return {"$ref": f"#/components/schemas/{name}"}

    def json_body(name: str) -> Dict[str, Any]:
        return {
            "required": True,
            "content": _content(schema_ref(name)),
        }

    session_parameter = [{"$ref": "#/components/parameters/SessionId"}]
    operation_responses = _responses(schema_ref("Operation"), accepted=True)
    paths: Dict[str, Dict[str, Any]] = {
        "/api/v2/modeling-sessions": {
            "post": {
                "operationId": "createModelingSession",
                "requestBody": json_body("CreateSessionRequest"),
                "responses": _responses(schema_ref("ModelingSession"), created=True),
            }
        },
        "/api/v2/modeling-sessions/{session_id}": {
            "parameters": session_parameter,
            "get": {
                "operationId": "getModelingSession",
                "responses": _responses(schema_ref("ModelingSession")),
            },
            "delete": {
                "operationId": "deleteModelingSession",
                "responses": _responses(empty=True),
            },
        },
        "/api/v2/modeling-sessions/{session_id}/objects": {
            "parameters": session_parameter,
            "get": {
                "operationId": "listMeasurementObjects",
                "responses": _responses(schema_ref("MeasurementObjectList")),
            },
            "post": {
                "operationId": "createMeasurementObject",
                "requestBody": json_body("MeasurementObjectCreate"),
                "responses": _responses(schema_ref("MeasurementObject"), created=True),
            },
        },
        "/api/v2/modeling-sessions/{session_id}/objects/{object_id}": {
            "parameters": [
                *session_parameter,
                {"$ref": "#/components/parameters/ObjectId"},
            ],
            "get": {
                "operationId": "getMeasurementObject",
                "responses": _responses(schema_ref("MeasurementObject")),
            },
            "put": {
                "operationId": "updateMeasurementObject",
                "requestBody": json_body("MeasurementObjectUpdate"),
                "responses": _responses(schema_ref("MeasurementObject")),
            },
            "delete": {
                "operationId": "deleteMeasurementObject",
                "responses": _responses(empty=True),
            },
        },
        "/api/v2/modeling-sessions/{session_id}:preview": {
            "parameters": session_parameter,
            "post": {
                "operationId": "previewModelingSession",
                "requestBody": {
                    "required": False,
                    "content": _content(schema_ref("PreviewRequest")),
                },
                "responses": operation_responses,
            },
        },
        "/api/v2/modeling-sessions/{session_id}:publish": {
            "parameters": session_parameter,
            "post": {
                "operationId": "publishModelingSession",
                "requestBody": json_body("PublishRequest"),
                "responses": operation_responses,
            },
        },
        "/api/v2/modeling-sessions:import": {
            "post": {
                "operationId": "importModelPackage",
                "parameters": [
                    {"$ref": "#/components/parameters/ModelTimeout"},
                    {"$ref": "#/components/parameters/WaitMs"},
                    {"$ref": "#/components/parameters/ClientRequestId"},
                ],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/octet-stream": {
                            "schema": {"type": "string", "format": "binary"}
                        },
                        "multipart/form-data": {
                            "schema": {
                                "type": "object",
                                "required": ["model_package"],
                                "properties": {
                                    "model_package": {
                                        "type": "string",
                                        "format": "binary",
                                    }
                                },
                            }
                        },
                    },
                },
                "responses": operation_responses,
            }
        },
        "/api/v2/measurements": {
            "post": {
                "operationId": "measurePublishedModel",
                "requestBody": json_body("MeasurementRequest"),
                "responses": operation_responses,
            }
        },
        "/api/v2/models/{calibration_model_id}:export": {
            "parameters": [{"$ref": "#/components/parameters/CalibrationModelId"}],
            "post": {
                "operationId": "exportModelPackage",
                "requestBody": json_body("ExportRequest"),
                "responses": operation_responses,
            },
        },
        "/api/v2/operations": {
            "get": {
                "operationId": "listOperations",
                "parameters": [{"$ref": "#/components/parameters/ClientRequestId"}],
                "responses": _responses(schema_ref("OperationList")),
            }
        },
        "/api/v2/operations/{operation_id}": {
            "parameters": [{"$ref": "#/components/parameters/OperationId"}],
            "get": {
                "operationId": "getOperation",
                "responses": _responses(schema_ref("Operation")),
            },
        },
        "/api/v2/operations/{operation_id}:cancel": {
            "parameters": [{"$ref": "#/components/parameters/OperationId"}],
            "post": {
                "operationId": "cancelOperation",
                "responses": _responses(schema_ref("Operation")),
            },
        },
        "/api/v2/health": {
            "get": {
                "operationId": "health",
                "responses": _responses(schema_ref("Health")),
            }
        },
        "/api/v2/openapi.json": {
            "get": {
                "operationId": "openapi",
                "responses": _responses({"type": "object"}),
            }
        },
    }
    schemas: Dict[str, Any] = {
        "MeasurementObjectCreate": {
            "oneOf": object_create_variants,
            "discriminator": {"propertyName": "object_type"},
        },
        "MeasurementObject": {
            "oneOf": object_response_variants,
            "discriminator": {"propertyName": "object_type"},
        },
        "MeasurementObjectUpdate": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "label": {"type": "string", "minLength": 1},
                "parameters": {"anyOf": parameter_variants},
                "input_object_ids": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "uniqueItems": True,
                },
            },
        },
        "MeasurementObjectList": {
            "type": "object",
            "required": ["objects"],
            "properties": {
                "objects": {"type": "array", "items": schema_ref("MeasurementObject")}
            },
        },
        "MeasurementObjectResult": {
            "oneOf": object_result_variants,
            "discriminator": {"propertyName": "object_type"},
        },
        "MeasurementObjectResultBase": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "object_id",
                "object_type",
                "label",
                "valid",
                "observation",
                "quality",
                "error",
            ],
            "properties": {
                "object_id": {"type": "string"},
                "object_type": {"type": "string"},
                "label": {"type": "string"},
                "valid": {"type": "boolean"},
                "observation": {},
                "quality": {"type": "object"},
                "error": {
                    "anyOf": [schema_ref("EngineError"), {"type": "null"}]
                },
            },
        },
        "PointObservation": {
            "type": "object",
            "additionalProperties": False,
            "required": ["type", "row", "col"],
            "properties": {
                "type": {"const": "point"},
                "row": {"type": "number"},
                "col": {"type": "number"},
            },
        },
        "LineObservation": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "type",
                "a",
                "b",
                "c",
                "start_row",
                "start_col",
                "end_row",
                "end_col",
            ],
            "properties": {
                name: {"const": "line"} if name == "type" else {"type": "number"}
                for name in (
                    "type",
                    "a",
                    "b",
                    "c",
                    "start_row",
                    "start_col",
                    "end_row",
                    "end_col",
                )
            },
        },
        "CircleObservation": {
            "type": "object",
            "additionalProperties": False,
            "required": ["type", "center_row", "center_col", "radius"],
            "properties": {
                "type": {"const": "circle"},
                "center_row": {"type": "number"},
                "center_col": {"type": "number"},
                "radius": {"type": "number"},
            },
        },
        "ScalarObservation": {
            "type": "object",
            "additionalProperties": False,
            "required": ["type", "value", "unit"],
            "properties": {
                "type": {"const": "scalar"},
                "value": {"type": "number"},
                "unit": {"type": "string", "enum": ["pixel", "degree"]},
            },
        },
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
                "bin_index": {"type": "integer"},
                "jpeg_size": {"type": "integer"},
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
                    "type": "string",
                    "enum": [
                        "request",
                        "image",
                        "model",
                        "measurement",
                        "timeout",
                        "cancelled",
                        "operation",
                        "internal",
                    ],
                },
                "message": {"type": "string"},
                "retryable": {"type": "boolean"},
                "details": {"type": "object"},
                "trace_id": {"type": "string"},
            },
        },
        "ErrorEnvelope": {
            "type": "object",
            "additionalProperties": False,
            "required": ["error"],
            "properties": {"error": schema_ref("EngineError")},
        },
        "ModelingSession": {
            "type": "object",
            "additionalProperties": False,
            "required": ["session_id", "created_at", "expires_at", "object_count"],
            "properties": {
                "session_id": {"type": "string"},
                "created_at": {"type": "string", "format": "date-time"},
                "expires_at": {"type": "string", "format": "date-time"},
                "object_count": {"type": "integer", "minimum": 0},
            },
        },
        "CreateSessionRequest": {
            "type": "object",
            "additionalProperties": False,
            "required": ["image"],
            "properties": {"image": schema_ref("ImageDescriptor")},
        },
        "Operation": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "contract_version",
                "operation_id",
                "trace_id",
                "kind",
                "status",
                "client_request_id",
                "accepted_at",
                "started_at",
                "completed_at",
                "timeout_ms",
                "result",
                "error",
            ],
            "properties": {
                "contract_version": {"const": "2.0"},
                "operation_id": {"type": "string"},
                "trace_id": {"type": "string"},
                "kind": {
                    "type": "string",
                    "enum": ["preview", "publish", "measure", "import", "export"],
                },
                "status": {
                    "type": "string",
                    "enum": [
                        "queued",
                        "running",
                        "succeeded",
                        "failed",
                        "cancelled",
                        "timed_out",
                    ],
                },
                "client_request_id": {
                    "anyOf": [{"type": "string"}, {"type": "null"}]
                },
                "accepted_at": {"type": "string", "format": "date-time"},
                "started_at": {
                    "anyOf": [
                        {"type": "string", "format": "date-time"},
                        {"type": "null"},
                    ]
                },
                "completed_at": {
                    "anyOf": [
                        {"type": "string", "format": "date-time"},
                        {"type": "null"},
                    ]
                },
                "timeout_ms": {"type": "integer", "minimum": 1},
                "result": {
                    "anyOf": [schema_ref("OperationResult"), {"type": "null"}]
                },
                "error": {
                    "anyOf": [schema_ref("EngineError"), {"type": "null"}]
                },
            },
        },
        "OperationResult": {
            "oneOf": [
                schema_ref("PreviewResult"),
                schema_ref("PublishResult"),
                schema_ref("MeasurementResult"),
                schema_ref("ImportResult"),
                schema_ref("ExportResult"),
            ]
        },
        "PreviewResult": {
            "type": "object",
            "additionalProperties": False,
            "required": ["objects"],
            "properties": {
                "objects": {
                    "type": "array",
                    "items": schema_ref("MeasurementObjectResult"),
                }
            },
        },
        "PublishResult": {
            "type": "object",
            "additionalProperties": False,
            "required": ["calibration_model_id", "asset_ref"],
            "properties": {
                "calibration_model_id": {"type": "string"},
                "asset_ref": {"type": "string"},
            },
        },
        "MeasurementResult": {
            "type": "object",
            "additionalProperties": False,
            "required": ["calibration_model_id", "asset_ref", "objects"],
            "properties": {
                "calibration_model_id": {"type": "string"},
                "asset_ref": {"type": "string"},
                "objects": {
                    "type": "array",
                    "items": schema_ref("MeasurementObjectResult"),
                },
            },
        },
        "ImportResult": {
            "type": "object",
            "additionalProperties": False,
            "required": ["session_id"],
            "properties": {"session_id": {"type": "string"}},
        },
        "ExportResult": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "download_ref",
                "sha256",
                "size",
                "package_contract_version",
            ],
            "properties": {
                "download_ref": {"type": "string"},
                "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                "size": {"type": "integer", "minimum": 1},
                "package_contract_version": {"const": "2.0"},
            },
        },
        "OperationList": {
            "type": "object",
            "required": ["operations"],
            "properties": {
                "operations": {"type": "array", "items": schema_ref("Operation")}
            },
        },
        "PreviewRequest": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "image": schema_ref("ImageDescriptor"),
                "timeout_ms": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 120_000,
                    "default": 30_000,
                },
                "wait_ms": {"type": "integer", "minimum": 0, "default": 0},
                "client_request_id": {"type": "string"},
            },
        },
        "PublishRequest": {
            "type": "object",
            "additionalProperties": False,
            "required": ["calibration_model_id"],
            "properties": {
                "calibration_model_id": {"type": "string", "minLength": 1},
                "timeout_ms": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 300_000,
                    "default": 60_000,
                },
                "wait_ms": {"type": "integer", "minimum": 0, "default": 0},
                "client_request_id": {"type": "string"},
            },
        },
        "MeasurementRequest": {
            "type": "object",
            "additionalProperties": False,
            "required": ["calibration_model_id", "asset_ref", "image"],
            "properties": {
                "calibration_model_id": {"type": "string", "minLength": 1},
                "asset_ref": {"type": "string", "minLength": 1},
                "image": schema_ref("ImageDescriptor"),
                "timeout_ms": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 60_000,
                    "default": 10_000,
                },
                "wait_ms": {"type": "integer", "minimum": 0, "default": 0},
                "client_request_id": {"type": "string"},
            },
        },
        "ExportRequest": {
            "type": "object",
            "additionalProperties": False,
            "required": ["asset_ref"],
            "properties": {
                "asset_ref": {"type": "string", "minLength": 1},
                "timeout_ms": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 300_000,
                    "default": 60_000,
                },
                "wait_ms": {"type": "integer", "minimum": 0, "default": 0},
                "client_request_id": {"type": "string"},
            },
        },
        "Health": {
            "type": "object",
            "additionalProperties": False,
            "required": ["status"],
            "properties": {"status": {"const": "ok"}},
        },
        "TimeoutLimits": {
            "type": "object",
            "properties": {
                "measurement_ms": {
                    "type": "integer",
                    "default": 10_000,
                    "maximum": 60_000,
                },
                "preview_ms": {
                    "type": "integer",
                    "default": 30_000,
                    "maximum": 120_000,
                },
                "model_ms": {
                    "type": "integer",
                    "default": 60_000,
                    "maximum": 300_000,
                },
            },
        },
    }
    parameters = {
        name: {
            "name": wire_name,
            "in": location,
            "required": location == "path",
            "schema": schema,
        }
        for name, wire_name, location, schema in (
            ("SessionId", "session_id", "path", {"type": "string"}),
            ("ObjectId", "object_id", "path", {"type": "string"}),
            ("OperationId", "operation_id", "path", {"type": "string"}),
            (
                "CalibrationModelId",
                "calibration_model_id",
                "path",
                {"type": "string"},
            ),
            (
                "ModelTimeout",
                "timeout_ms",
                "query",
                {"type": "integer", "default": 60_000, "maximum": 300_000},
            ),
            ("WaitMs", "wait_ms", "query", {"type": "integer", "default": 0}),
            ("ClientRequestId", "client_request_id", "query", {"type": "string"}),
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
            "The engine trusts platform guarantees for slot and JPEG length ranges",
            "The engine does not perform platform tolerance decisions or fallback",
        ],
        "paths": paths,
        "components": {"schemas": schemas, "parameters": parameters},
    }
