"""Headless application interface for the v2 measurement engine."""

from __future__ import annotations

import hashlib
import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

SESSION_IDLE_TTL = timedelta(hours=8)


class EngineError(Exception):
    """Stable error returned by the application and HTTP interfaces."""

    def __init__(
        self,
        code: str,
        category: str,
        message: str,
        *,
        retryable: bool = False,
        details: Optional[Mapping[str, Any]] = None,
        trace_id: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.error: Dict[str, Any] = {
            "code": code,
            "category": category,
            "message": message,
            "retryable": retryable,
            "details": dict(details or {}),
            "trace_id": trace_id or uuid.uuid4().hex,
        }


@dataclass
class _ObjectDefinition:
    object_id: str
    object_type: str
    label: str
    parameters: Dict[str, Any]
    input_object_ids: List[str]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "object_id": self.object_id,
            "object_type": self.object_type,
            "label": self.label,
            "parameters": dict(self.parameters),
            "input_object_ids": list(self.input_object_ids),
        }


@dataclass
class _ModelingSession:
    session_id: str
    reference_image: np.ndarray
    created_at: datetime
    expires_at: datetime
    objects: List[_ObjectDefinition]


@dataclass(frozen=True)
class _ObjectContract:
    required: Sequence[str]
    defaults: Mapping[str, Any]
    input_types: Sequence[str]
    result_type: str


_OBJECT_CONTRACTS: Dict[str, _ObjectContract] = {
    "EdgePoint": _ObjectContract(
        ("row", "col", "angle", "length1", "length2"),
        {"sigma": 1.0, "threshold": 30.0, "interpolation": "linear"},
        (),
        "point",
    ),
    "EdgePair": _ObjectContract(
        ("row", "col", "angle", "length1", "length2"),
        {"sigma": 1.0, "threshold": 30.0, "interpolation": "linear"},
        (),
        "point",
    ),
    "FitLine": _ObjectContract(
        ("start", "end", "measure_length1", "measure_length2"),
        {
            "num_measures": 10,
            "sigma": 1.0,
            "threshold": 30.0,
            "transition": "all",
        },
        (),
        "line",
    ),
    "FitCircle": _ObjectContract(
        ("center", "radius", "measure_length1", "measure_length2"),
        {
            "radius_min": None,
            "radius_max": None,
            "num_measures": 12,
            "sigma": 1.0,
            "threshold": 30.0,
            "transition": "all",
            "start_phi": 0.0,
            "end_phi": 2 * math.pi,
        },
        (),
        "circle",
    ),
    "TemplateMatchPoint": _ObjectContract(
        ("row", "col"),
        {
            "template_size": 40,
            "preprocessor_type": "raw",
            "match_score_threshold": 0.5,
            "angle_range_half": 15.0,
            "angle_step": 1.0,
            "use_subpixel": True,
        },
        (),
        "point",
    ),
    "TwoPointsLine": _ObjectContract((), {}, ("point", "point"), "line"),
    "TwoPointsDistance": _ObjectContract((), {}, ("point", "point"), "distance"),
    "PointLineDistance": _ObjectContract((), {}, ("point", "line"), "distance"),
    "TwoLinesAngle": _ObjectContract((), {}, ("line", "line"), "angle"),
    "PointCircleDistance": _ObjectContract((), {}, ("point", "circle"), "distance"),
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class EngineApplication:
    """Own modeling sessions and expose the headless engine interface.

    Args:
        asset_root: Engine-owned persistent directory for published assets.
        now: Injectable UTC clock used for deterministic session expiry tests.
    """

    def __init__(
        self,
        asset_root: Path,
        *,
        now: Callable[[], datetime] = _utc_now,
        operations: Optional[Any] = None,
    ) -> None:
        self._asset_root = Path(asset_root)
        self._asset_root.mkdir(parents=True, exist_ok=True)
        self._now = now
        self._sessions: Dict[str, _ModelingSession] = {}
        if operations is None:
            from measure_api.v2.operations import OperationManager

            operations = OperationManager()
        self._operations = operations

    def create_modeling_session(self, reference_image: np.ndarray) -> Dict[str, Any]:
        """Create an in-memory modeling session from a private image snapshot."""
        image = self._private_gray_snapshot(reference_image)
        created_at = self._now()
        session_id = uuid.uuid4().hex
        session = _ModelingSession(
            session_id=session_id,
            reference_image=image,
            created_at=created_at,
            expires_at=created_at + SESSION_IDLE_TTL,
            objects=[],
        )
        self._sessions[session_id] = session
        return self._session_dict(session)

    def get_modeling_session(self, session_id: str) -> Dict[str, Any]:
        """Return session metadata and refresh its idle expiry."""
        session = self._session(session_id)
        self._touch(session)
        return self._session_dict(session)

    def delete_modeling_session(self, session_id: str) -> None:
        """Explicitly discard an unpublished modeling session."""
        self._session(session_id)
        del self._sessions[session_id]

    def create_object(self, session_id: str, request: Mapping[str, Any]) -> Dict[str, Any]:
        """Create a measurement object with a server-owned stable ID."""
        session = self._session(session_id)
        allowed_fields = {
            "object_type",
            "label",
            "parameters",
            "input_object_ids",
        }
        unknown_fields = set(request) - allowed_fields
        if unknown_fields:
            raise EngineError(
                "object.field_unknown",
                "request",
                "Measurement object contains unknown fields",
                details={"fields": sorted(unknown_fields)},
            )
        object_type = request.get("object_type")
        label = request.get("label")
        if not isinstance(object_type, str) or object_type not in _OBJECT_CONTRACTS:
            raise EngineError(
                "object.type_unsupported",
                "request",
                "Unsupported measurement object type",
                details={"object_type": object_type},
            )
        if not isinstance(label, str) or not label.strip():
            raise EngineError("object.label_required", "request", "Object label is required")
        if any(item.label == label for item in session.objects):
            raise EngineError("object.label_conflict", "request", "Object label already exists")

        parameters = self._parameters(object_type, request.get("parameters", {}))
        input_ids = self._input_ids(request.get("input_object_ids", []))
        self._validate_dependencies(session, object_type, input_ids)
        item = _ObjectDefinition(
            object_id=uuid.uuid4().hex,
            object_type=object_type,
            label=label,
            parameters=parameters,
            input_object_ids=input_ids,
        )
        session.objects.append(item)
        self._touch(session)
        return item.as_dict()

    def update_object(
        self,
        session_id: str,
        object_id: str,
        request: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Update mutable object fields while preserving object identity and type."""
        session = self._session(session_id)
        item = self._object(session, object_id)
        unknown = set(request) - {"label", "parameters", "input_object_ids"}
        if unknown:
            raise EngineError(
                "object.field_immutable",
                "request",
                "Object identity and type are immutable",
                details={"fields": sorted(unknown)},
            )
        if "label" in request:
            label = request["label"]
            if not isinstance(label, str) or not label.strip():
                raise EngineError("object.label_required", "request", "Object label is required")
            has_conflict = any(
                other.object_id != object_id and other.label == label
                for other in session.objects
            )
            if has_conflict:
                raise EngineError("object.label_conflict", "request", "Object label already exists")
            item.label = label
        if "parameters" in request:
            item.parameters = self._parameters(item.object_type, request["parameters"])
        if "input_object_ids" in request:
            input_ids = self._input_ids(request["input_object_ids"])
            self._validate_dependencies(session, item.object_type, input_ids)
            item.input_object_ids = input_ids
        self._touch(session)
        return item.as_dict()

    def list_objects(self, session_id: str) -> List[Dict[str, Any]]:
        """List modeling objects in creation order."""
        session = self._session(session_id)
        self._touch(session)
        return [item.as_dict() for item in session.objects]

    def get_object(self, session_id: str, object_id: str) -> Dict[str, Any]:
        """Return one modeling object by immutable ID."""
        session = self._session(session_id)
        item = self._object(session, object_id)
        self._touch(session)
        return item.as_dict()

    def delete_object(self, session_id: str, object_id: str) -> None:
        """Delete an unreferenced modeling object."""
        session = self._session(session_id)
        item = self._object(session, object_id)
        dependants = [
            candidate.object_id
            for candidate in session.objects
            if object_id in candidate.input_object_ids
        ]
        if dependants:
            raise EngineError(
                "object.dependency_conflict",
                "request",
                "Object is referenced by another measurement object",
                details={"dependent_object_ids": dependants},
            )
        session.objects.remove(item)
        self._touch(session)

    def preview(
        self,
        session_id: str,
        image: Optional[np.ndarray] = None,
        *,
        timeout_ms: int = 30_000,
        client_request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Preview all session objects as an engine operation."""
        timeout = self._timeout(timeout_ms, default=30_000, maximum=120_000)
        session = self._session(session_id)
        self._touch(session)
        snapshot = session.reference_image if image is None else self._private_gray_snapshot(image)
        return self._operations.submit(
            "preview",
            {
                "reference_image": session.reference_image.copy(),
                "image": snapshot.copy(),
                "objects": [item.as_dict() for item in session.objects],
            },
            timeout,
            client_request_id=client_request_id,
        )

    def publish(
        self,
        session_id: str,
        calibration_model_id: str,
        *,
        timeout_ms: int = 60_000,
        client_request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Directly overwrite one calibration model's current native asset."""
        timeout = self._timeout(timeout_ms, default=60_000, maximum=300_000)
        if not isinstance(calibration_model_id, str) or not calibration_model_id:
            raise EngineError(
                "model.identity_required",
                "request",
                "calibration_model_id is required",
            )
        session = self._session(session_id)
        if not session.objects:
            raise EngineError("model.empty", "model", "Model has no measurement objects")
        self._touch(session)
        asset_ref = self._asset_ref(calibration_model_id)

        def close_session(result: Dict[str, Any]) -> Dict[str, Any]:
            self._sessions.pop(session_id, None)
            return result

        return self._operations.submit(
            "publish",
            {
                "calibration_model_id": calibration_model_id,
                "asset_ref": asset_ref,
                "asset_dir": str(self._asset_dir(calibration_model_id)),
                "reference_image": session.reference_image.copy(),
                "objects": [item.as_dict() for item in session.objects],
            },
            timeout,
            client_request_id=client_request_id,
            on_success=close_session,
        )

    def measure(
        self,
        calibration_model_id: str,
        asset_ref: str,
        image: np.ndarray,
        *,
        timeout_ms: int = 10_000,
        client_request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Measure a private image snapshot with one published model."""
        timeout = self._timeout(timeout_ms, default=10_000, maximum=60_000)
        expected_ref = self._asset_ref(calibration_model_id)
        if asset_ref != expected_ref:
            raise EngineError(
                "model.asset_unavailable",
                "model",
                "Model asset is unavailable",
            )
        return self._operations.submit(
            "measure",
            {
                "calibration_model_id": calibration_model_id,
                "asset_ref": asset_ref,
                "asset_dir": str(self._asset_dir(calibration_model_id)),
                "image": self._private_gray_snapshot(image),
            },
            timeout,
            client_request_id=client_request_id,
        )

    def export_model(
        self,
        calibration_model_id: str,
        asset_ref: str,
        *,
        timeout_ms: int = 60_000,
        client_request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Export a published model as an immutable portable package."""
        timeout = self._timeout(timeout_ms, default=60_000, maximum=300_000)
        if asset_ref != self._asset_ref(calibration_model_id):
            raise EngineError("model.asset_unavailable", "model", "Model asset is unavailable")
        return self._operations.submit(
            "export",
            {
                "calibration_model_id": calibration_model_id,
                "asset_ref": asset_ref,
                "asset_dir": str(self._asset_dir(calibration_model_id)),
                "export_dir": str(self._asset_root / "exports"),
            },
            timeout,
            client_request_id=client_request_id,
        )

    def import_model(
        self,
        package: bytes,
        *,
        timeout_ms: int = 60_000,
        client_request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Import a portable package into a new editable modeling session."""
        timeout = self._timeout(timeout_ms, default=60_000, maximum=300_000)
        if not isinstance(package, bytes) or not package:
            raise EngineError("model.package_invalid", "model", "Model package is invalid")

        def create_session(result: Dict[str, Any]) -> Dict[str, Any]:
            try:
                session_data = self.create_modeling_session(result["reference_image"])
                session = self._sessions[session_data["session_id"]]
                object_ids = set()
                for definition in result["objects"]:
                    object_id = str(definition["object_id"])
                    if object_id in object_ids:
                        raise ValueError("duplicate object ID")
                    object_ids.add(object_id)
                    object_type = str(definition["object_type"])
                    item = _ObjectDefinition(
                        object_id=object_id,
                        object_type=object_type,
                        label=str(definition["label"]),
                        parameters=self._parameters(
                            object_type, definition["parameters"]
                        ),
                        input_object_ids=self._input_ids(
                            definition.get("input_object_ids", [])
                        ),
                    )
                    self._validate_dependencies(
                        session,
                        item.object_type,
                        item.input_object_ids,
                    )
                    session.objects.append(item)
                return {"session_id": session.session_id}
            except (EngineError, KeyError, TypeError, ValueError) as exc:
                if "session_data" in locals():
                    self._sessions.pop(session_data["session_id"], None)
                raise EngineError(
                    "model.package_invalid",
                    "model",
                    "Model package is invalid",
                    details={"diagnostic": str(exc)},
                ) from exc

        return self._operations.submit(
            "import",
            {"package": package},
            timeout,
            client_request_id=client_request_id,
            on_success=create_session,
        )

    def read_export(self, download_ref: str) -> bytes:
        """Resolve an engine-controlled export reference to package bytes."""
        prefix = "v2-export:"
        if not isinstance(download_ref, str) or not download_ref.startswith(prefix):
            raise EngineError("model.export_not_found", "model", "Model export not found")
        operation_id = download_ref[len(prefix) :]
        invalid_id = any(
            character not in "0123456789abcdef"
            for character in operation_id
        )
        if not operation_id or invalid_id:
            raise EngineError("model.export_not_found", "model", "Model export not found")
        path = self._asset_root / "exports" / f"{operation_id}.zip"
        try:
            return path.read_bytes()
        except OSError as exc:
            raise EngineError(
                "model.export_not_found",
                "model",
                "Model export not found",
            ) from exc

    def get_operation(self, operation_id: str) -> Dict[str, Any]:
        """Return one current-process engine operation."""
        return self._operations.get(operation_id)

    def list_operations(
        self, client_request_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List engine operations, optionally filtered by client request ID."""
        return self._operations.list(client_request_id)

    def wait_operation(self, operation_id: str, wait_ms: int) -> Dict[str, Any]:
        """Wait for an operation without coupling it to an HTTP connection."""
        return self._operations.wait(operation_id, wait_ms)

    def cancel_operation(self, operation_id: str) -> Dict[str, Any]:
        """Request cancellation of an engine operation."""
        return self._operations.cancel(operation_id)

    def close(self) -> None:
        """Release the operation manager and its CPU subprocess."""
        self._operations.close()

    def _session(self, session_id: str) -> _ModelingSession:
        session = self._sessions.get(session_id)
        if session is None:
            raise EngineError("session.not_found", "request", "Modeling session not found")
        if self._now() > session.expires_at:
            del self._sessions[session_id]
            raise EngineError("session.expired", "request", "Modeling session expired")
        return session

    @staticmethod
    def _timeout(value: Any, *, default: int, maximum: int) -> int:
        if value is None:
            return default
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise EngineError(
                "request.timeout_invalid",
                "request",
                "timeout_ms must be a positive integer",
            )
        if value > maximum:
            raise EngineError(
                "request.timeout_invalid",
                "request",
                "timeout_ms exceeds the operation limit",
                details={"maximum": maximum},
            )
        return value

    def _asset_dir(self, calibration_model_id: str) -> Path:
        key = hashlib.sha256(calibration_model_id.encode("utf-8")).hexdigest()
        return self._asset_root / "models" / key

    @staticmethod
    def _asset_ref(calibration_model_id: str) -> str:
        key = hashlib.sha256(calibration_model_id.encode("utf-8")).hexdigest()
        return f"v2-model:{key}"

    def _touch(self, session: _ModelingSession) -> None:
        session.expires_at = self._now() + SESSION_IDLE_TTL

    @staticmethod
    def _session_dict(session: _ModelingSession) -> Dict[str, Any]:
        return {
            "session_id": session.session_id,
            "created_at": session.created_at.isoformat(),
            "expires_at": session.expires_at.isoformat(),
            "object_count": len(session.objects),
        }

    @staticmethod
    def _object(session: _ModelingSession, object_id: str) -> _ObjectDefinition:
        for item in session.objects:
            if item.object_id == object_id:
                return item
        raise EngineError("object.not_found", "request", "Measurement object not found")

    @staticmethod
    def _private_gray_snapshot(image: np.ndarray) -> np.ndarray:
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise EngineError("image.invalid", "image", "Image is empty or invalid")
        if image.ndim == 2:
            gray = image
        elif image.ndim == 3 and image.shape[2] in (3, 4):
            import cv2

            code = cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
            gray = cv2.cvtColor(image, code)
        else:
            raise EngineError("image.invalid", "image", "Unsupported image shape")
        return np.ascontiguousarray(gray, dtype=np.uint8).copy()

    @staticmethod
    def _input_ids(value: Any) -> List[str]:
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise EngineError(
                "object.dependencies_invalid",
                "request",
                "input_object_ids must be an array of object IDs",
            )
        return list(value)

    @staticmethod
    def _parameters(object_type: str, value: Any) -> Dict[str, Any]:
        if not isinstance(value, Mapping):
            raise EngineError(
                "object.parameters_invalid",
                "request",
                "parameters must be an object",
            )
        contract = _OBJECT_CONTRACTS[object_type]
        allowed = set(contract.required) | set(contract.defaults)
        unknown = set(value) - allowed
        missing = set(contract.required) - set(value)
        if unknown or missing:
            raise EngineError(
                "object.parameters_invalid",
                "request",
                "Object parameters do not match the static contract",
                details={"missing": sorted(missing), "unknown": sorted(unknown)},
            )
        parameters = dict(contract.defaults)
        parameters.update(value)
        if not _all_finite(parameters):
            raise EngineError(
                "object.parameters_invalid",
                "request",
                "Object parameters must contain only finite numbers",
            )
        _validate_parameter_semantics(object_type, parameters)
        return parameters

    @staticmethod
    def _validate_dependencies(
        session: _ModelingSession,
        object_type: str,
        input_ids: Sequence[str],
    ) -> None:
        expected_types = _OBJECT_CONTRACTS[object_type].input_types
        if len(input_ids) != len(expected_types) or len(set(input_ids)) != len(input_ids):
            raise EngineError(
                "object.dependencies_invalid",
                "request",
                "Object dependencies do not match the static contract",
            )
        objects = {item.object_id: item for item in session.objects}
        for input_id, expected_type in zip(input_ids, expected_types):
            dependency = objects.get(input_id)
            if dependency is None:
                raise EngineError(
                    "object.dependency_not_found",
                    "request",
                    "Referenced measurement object does not exist",
                    details={"object_id": input_id},
                )
            result_type = _OBJECT_CONTRACTS[dependency.object_type].result_type
            if result_type != expected_type:
                raise EngineError(
                    "object.dependency_type_mismatch",
                    "request",
                    "Referenced object has an incompatible result type",
                    details={"object_id": input_id, "expected": expected_type},
                )


def _all_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float, np.integer, np.floating)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    return False


def _validate_parameter_semantics(object_type: str, parameters: Mapping[str, Any]) -> None:
    text_fields = {"transition", "interpolation", "preprocessor_type"}
    for name, value in parameters.items():
        if value is None or name in text_fields or name in {"start", "end", "center"}:
            continue
        if name == "use_subpixel":
            if not isinstance(value, bool):
                raise EngineError(
                    "object.parameters_invalid",
                    "request",
                    "use_subpixel must be a boolean",
                )
            continue
        if isinstance(value, bool) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise EngineError(
                "object.parameters_invalid",
                "request",
                f"{name} must be numeric",
            )
    for name in ("start", "end", "center"):
        if name in parameters:
            value = parameters[name]
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise EngineError(
                    "object.parameters_invalid",
                    "request",
                    f"{name} must contain exactly two coordinates",
                )
            if any(
                isinstance(item, bool)
                or not isinstance(item, (int, float, np.integer, np.floating))
                for item in value
            ):
                raise EngineError(
                    "object.parameters_invalid",
                    "request",
                    f"{name} coordinates must be numeric",
                )
    positive = {
        "length1",
        "length2",
        "measure_length1",
        "measure_length2",
        "radius",
        "sigma",
        "template_size",
        "angle_step",
    }
    for name in positive:
        if name in parameters and float(parameters[name]) <= 0:
            raise EngineError(
                "object.parameters_invalid",
                "request",
                f"{name} must be greater than zero",
            )
    if "threshold" in parameters and float(parameters["threshold"]) < 0:
        raise EngineError(
            "object.parameters_invalid",
            "request",
            "threshold cannot be negative",
        )
    if "num_measures" in parameters:
        count = parameters["num_measures"]
        minimum = 3 if object_type == "FitCircle" else 2
        if isinstance(count, bool) or not isinstance(count, int) or count < minimum:
            raise EngineError(
                "object.parameters_invalid",
                "request",
                f"num_measures must be an integer greater than or equal to {minimum}",
            )
    if "template_size" in parameters:
        template_size = parameters["template_size"]
        if isinstance(template_size, bool) or not isinstance(template_size, int):
            raise EngineError(
                "object.parameters_invalid",
                "request",
                "template_size must be an integer",
            )
    if parameters.get("transition") not in {None, "all", "positive", "negative"}:
        raise EngineError(
            "object.parameters_invalid",
            "request",
            "transition is unsupported",
        )
    if parameters.get("interpolation") not in {None, "linear", "nearest", "cubic"}:
        raise EngineError(
            "object.parameters_invalid",
            "request",
            "interpolation is unsupported",
        )
    if parameters.get("preprocessor_type") not in {
        None,
        "raw",
        "canny",
        "sobel",
        "clahe",
        "threshold",
    }:
        raise EngineError(
            "object.parameters_invalid",
            "request",
            "preprocessor_type is unsupported",
        )
    radius_min = parameters.get("radius_min")
    radius_max = parameters.get("radius_max")
    if (
        radius_min is not None
        and radius_max is not None
        and (float(radius_min) <= 0 or float(radius_max) <= float(radius_min))
    ):
        raise EngineError(
            "object.parameters_invalid",
            "request",
            "radius_min and radius_max are invalid",
        )
