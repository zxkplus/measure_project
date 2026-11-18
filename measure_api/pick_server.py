"""HTTP service for the three-axis gantry pick calibration.

Run with::

    python -m measure_api.pick_server

or::

    uvicorn --factory measure_api.pick_server:create_app --host 127.0.0.1 --port 8100

The service exposes three stateless business calls, a global debug switch and
read-only run artifact endpoints.  Calibration returns one 3x3 homogeneous
matrix per step; the caller stores both matrices and sends them back to
``/locate``.  The service never talks to a robot and never decodes an image:
target detection belongs to the caller (the v1 measurement API, the v2
``measurements.circles`` operation, or the equipment's own recipe), which
passes the sub-pixel centre in as ``(u, v)``.  Every request is JSON.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask
from starlette.exceptions import HTTPException as StarletteHTTPException

from measure.pick_api import (
    GantryPick,
    PickError,
    check_image_point,
    image_size_pair,
)
from measure.pick_artifacts import (
    RunRecorder,
    cleanup_runs,
    list_runs,
    read_manifest,
    validate_run_id,
    zip_run,
)
from measure.robot_calibration import (
    CalibrationError,
    CameraGripperOffset,
    ImageRobotCalibration,
    OffsetSample,
)

logger = logging.getLogger("measure_api.pick")

PREFIX = "/api/v1"


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass
class PickSettings:
    """Service paths and retention policy.

    All values come from environment variables; none are exposed as HTTP
    parameters, so the business requests stay minimal.
    """

    runs_dir: Path = Path("data/pick_runs")
    debug_default: bool = False
    annotated_max_px: int = 2000
    locate_keep_days: int = 2
    locate_keep_max: int = 500
    calibration_keep_days: int = 90
    calibration_keep_max: int = 100
    cors_origins: Tuple[str, ...] = ()

    @classmethod
    def from_env(cls) -> "PickSettings":
        """Build settings from ``PICK_*`` environment variables."""
        origins = tuple(
            item.strip()
            for item in os.environ.get("PICK_CORS_ORIGINS", "").split(",")
            if item.strip()
        )
        return cls(
            runs_dir=Path(os.environ.get("PICK_RUNS_DIR", "data/pick_runs")),
            debug_default=_env_bool("PICK_DEBUG", False),
            annotated_max_px=max(
                256, _env_int("PICK_ANNOTATED_MAX_PX", 2000)
            ),
            locate_keep_days=max(0, _env_int("PICK_LOCATE_KEEP_DAYS", 2)),
            locate_keep_max=max(0, _env_int("PICK_LOCATE_KEEP_MAX", 500)),
            calibration_keep_days=max(
                0, _env_int("PICK_CALIBRATION_KEEP_DAYS", 90)
            ),
            calibration_keep_max=max(
                0, _env_int("PICK_CALIBRATION_KEEP_MAX", 100)
            ),
            cors_origins=origins,
        )


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class PoseSampleModel(BaseModel):
    """One release/centred pose pair."""

    place_pose: List[float]
    center_pose: List[float]


class OffsetRequest(BaseModel):
    """Offset calibration request; ``model`` selects the fitted transform."""

    samples: List[PoseSampleModel] = Field(..., min_length=1)
    model: Literal["auto", "translation", "affine"] = "auto"


class VisionRequest(BaseModel):
    """Vision calibration request: detected centres plus flange poses."""

    image_size: List[float]
    image_points: List[List[float]]
    robot_points: List[List[float]]
    principal_point: Optional[List[float]] = None
    confidences: Optional[List[float]] = None


class LocateRequest(BaseModel):
    """Locate request: one detected centre plus the stored matrices."""

    image_size: List[float]
    image_uv: List[float]
    robot_pose: List[float]
    vision_matrix: List[List[float]]
    offset_matrix: Optional[List[List[float]]] = None
    radius_px: Optional[float] = None


class DebugRequest(BaseModel):
    """Global debug switch request."""

    enabled: bool


# ---------------------------------------------------------------------------
# Debug state
# ---------------------------------------------------------------------------


class DebugState:
    """A process-wide, in-memory debug switch."""

    def __init__(self, enabled: bool = False) -> None:
        self._enabled = bool(enabled)
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    def set(self, enabled: bool) -> bool:
        with self._lock:
            self._enabled = bool(enabled)
            logger.warning("pick debug switch -> %s", self._enabled)
            return self._enabled


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class PickService:
    """Stateless business facade plus optional debug artifact recording."""

    def __init__(self, settings: PickSettings) -> None:
        self.settings = settings
        self.settings.runs_dir.mkdir(parents=True, exist_ok=True)
        self.pick = GantryPick()
        self.debug = DebugState(settings.debug_default)
        self._last_cleanup = time.monotonic()
        self._cleanup()

    # -- retention -------------------------------------------------------

    def _cleanup(self) -> None:
        try:
            cleanup_runs(
                self.settings.runs_dir,
                keep_days_by_type={
                    "vision": self.settings.calibration_keep_days,
                    "offset": self.settings.calibration_keep_days,
                    "locate": self.settings.locate_keep_days,
                },
                keep_max_by_type={
                    "vision": self.settings.calibration_keep_max,
                    "offset": self.settings.calibration_keep_max,
                    "locate": self.settings.locate_keep_max,
                },
            )
        except Exception:  # noqa: BLE001 - cleanup is best effort
            logger.exception("pick run cleanup failed")
        self._last_cleanup = time.monotonic()

    def _maybe_cleanup(self) -> None:
        if time.monotonic() - self._last_cleanup < 60.0:
            return
        self._cleanup()

    # -- helpers ---------------------------------------------------------

    @staticmethod
    def _parse_xy(value: Sequence[float], name: str) -> Tuple[float, float]:
        try:
            array = np.asarray(value, dtype=float).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise PickError("bad_request", f"{name} must contain numbers", 400) from exc
        if array.size < 2 or not np.all(np.isfinite(array[:2])):
            raise PickError(
                "bad_request",
                f"{name} must contain two finite values",
                400,
            )
        return (float(array[0]), float(array[1]))

    def _parse_points(
        self, robot_points: Sequence[Sequence[float]]
    ) -> List[Tuple[float, float]]:
        if not isinstance(robot_points, (list, tuple)):
            raise PickError("bad_request", "robot_points must be an array", 400)
        return [
            self._parse_xy(value, f"robot_points[{index}]")
            for index, value in enumerate(robot_points)
        ]

    def _fail(
        self,
        recorder: Optional[RunRecorder],
        code: str,
        message: str,
        status: int,
        result: Dict[str, Any],
        *,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Publish a failed debug run (when enabled) and raise a stable error."""
        details: Dict[str, Any] = dict(extra or {})
        if recorder is not None:
            try:
                manifest = recorder.finish(
                    result,
                    status="failed",
                    error={"code": code, "message": message},
                )
                details["run_id"] = manifest["run_id"]
                self._maybe_cleanup()
            except Exception:  # noqa: BLE001 - keep the business error
                logger.exception("failed to publish debug run")
                recorder.abort()
        raise PickError(code, message, status, details=details)

    def _finish_success(
        self,
        recorder: Optional[RunRecorder],
        result: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if recorder is None:
            return None
        manifest = recorder.finish(result, status="ok")
        self._maybe_cleanup()
        return manifest

    # -- vision calibration ---------------------------------------------

    @staticmethod
    def _point_rows(
        calibration: ImageRobotCalibration,
        image_points: Sequence[Sequence[float]],
        robot_points: Sequence[Sequence[float]],
    ) -> List[Dict[str, Any]]:
        """Build the per-point table stored as ``points.json``."""
        rows: List[Dict[str, Any]] = []
        for index, sample in enumerate(calibration.samples):
            rows.append(
                {
                    "index": index,
                    "u": float(image_points[index][0]),
                    "v": float(image_points[index][1]),
                    "x": float(robot_points[index][0]),
                    "y": float(robot_points[index][1]),
                    "residual_px": float(sample.residual_px),
                    "residual_mm": float(sample.residual_mm),
                    "inlier": bool(sample.inlier),
                }
            )
        return rows

    def calibrate_vision(
        self, body: VisionRequest, *, debug: bool
    ) -> Dict[str, Any]:
        """Fit the 3x3 vision matrix from image points and flange poses."""
        size = image_size_pair(body.image_size)
        image_points = [
            check_image_point(point, size, f"image_points[{index}]")
            for index, point in enumerate(body.image_points)
        ]
        robot_points = [
            self._parse_xy(point, f"robot_points[{index}]")
            for index, point in enumerate(body.robot_points)
        ]
        if len(image_points) != len(robot_points):
            raise PickError(
                "point_count_mismatch",
                f"{len(image_points)} image points but "
                f"{len(robot_points)} robot points",
                400,
            )

        recorder: Optional[RunRecorder] = None
        if debug:
            recorder = RunRecorder(
                self.settings.runs_dir,
                "vision",
                {
                    "image_size": [size[0], size[1]],
                    "image_points": [[u, v] for u, v in image_points],
                    "robot_points": [[x, y] for x, y in robot_points],
                    "principal_point": body.principal_point,
                },
            )

        try:
            calibration = self.pick.calibrate_vision(
                image_points,
                robot_points,
                image_size=size,
                principal_point=body.principal_point,
                confidences=body.confidences,
            )
        except CalibrationError as exc:
            self._fail(
                recorder,
                "calibration_failed",
                str(exc),
                422,
                {"valid": False},
            )
            raise  # pragma: no cover - _fail always raises

        rows = self._point_rows(calibration, image_points, robot_points)
        if not calibration.valid:
            if recorder is not None:
                recorder.add_points(rows)
            self._fail(
                recorder,
                "calibration_invalid",
                f"rms {calibration.rms_px:.3f} px exceeds the quality gate",
                422,
                {"valid": False},
                extra={
                    "rms_px": float(calibration.rms_px),
                    "warnings": list(calibration.warnings),
                },
            )

        matrix = np.asarray(calibration.matrix, dtype=float)
        if recorder is not None:
            recorder.add_points(rows)
        principal = [float(value) for value in calibration.principal_point]
        debug_result: Dict[str, Any] = {
            "matrix": matrix.tolist(),
            "rms_px": float(calibration.rms_px),
            "rms_mm": float(calibration.rms_mm),
            "num_points": len(calibration.samples),
            "principal_point": principal,
            "image_size": [size[0], size[1]],
            "warnings": list(calibration.warnings),
            "samples": [sample.as_dict() for sample in calibration.samples],
        }
        manifest = self._finish_success(recorder, debug_result)
        response: Dict[str, Any] = {
            "ok": True,
            "matrix": matrix.tolist(),
            "rms_px": float(calibration.rms_px),
            "rms_mm": float(calibration.rms_mm),
            "principal_point": principal,
            "image_size": [size[0], size[1]],
            "warnings": list(calibration.warnings),
        }
        if manifest is not None:
            response["run_id"] = manifest["run_id"]
        return response

    # -- offset calibration ---------------------------------------------

    def calibrate_offset(
        self,
        body: OffsetRequest,
        *,
        debug: bool,
    ) -> Dict[str, Any]:
        """Return the fitted 3x3 offset transform and its diagnostics."""
        if not body.samples:
            raise PickError("bad_request", "samples must not be empty", 400)

        built: List[OffsetSample] = []
        raw: List[Dict[str, List[float]]] = []
        for index, sample in enumerate(body.samples):
            place = self._parse_xy(
                sample.place_pose, f"samples[{index}].place_pose"
            )
            centre = self._parse_xy(
                sample.center_pose, f"samples[{index}].center_pose"
            )
            built.append(OffsetSample(place_pose=place, center_pose=centre))
            raw.append(
                {
                    "place_pose": [place[0], place[1]],
                    "center_pose": [centre[0], centre[1]],
                }
            )

        recorder: Optional[RunRecorder] = None
        if debug:
            recorder = RunRecorder(
                self.settings.runs_dir,
                "offset",
                {"samples": raw, "model": body.model},
            )

        try:
            offset = CameraGripperOffset.from_samples(built, model=body.model)
        except CalibrationError as exc:
            self._fail(
                recorder,
                exc.code or "offset_invalid",
                str(exc),
                422,
                {"valid": False},
                extra={"warnings": [str(exc)]},
            )
            raise  # pragma: no cover - _fail always raises

        if not offset.valid:
            self._fail(
                recorder,
                "offset_invalid",
                "; ".join(offset.warnings) or "offset calibration is not valid",
                422,
                {"valid": False},
                extra={"warnings": list(offset.warnings)},
            )

        matrix = np.asarray(offset.matrix, dtype=float)
        diagnostics: Dict[str, Any] = {
            "matrix": matrix.tolist(),
            "offset_xy": [float(value) for value in offset.offset_xy],
            "std_xy": [float(value) for value in offset.std_xy],
            "num_samples": len(offset.samples),
            "warnings": list(offset.warnings),
            "model": offset.requested_model,
            "effective_model": offset.effective_model,
            "rms_mm": float(offset.rms_mm),
            "max_mm": float(offset.max_mm),
            "condition": float(offset.condition),
            "rotation_deg": float(offset.rotation_deg),
            "scale": float(offset.scale),
            "shear": float(offset.shear),
            "convention": "grasp_pose = matrix @ camera_centred_pose",
        }
        debug_result = dict(diagnostics)
        debug_result["samples"] = [sample.as_dict() for sample in offset.samples]
        manifest = self._finish_success(recorder, debug_result)
        response: Dict[str, Any] = {"ok": True, **diagnostics}
        if manifest is not None:
            response["run_id"] = manifest["run_id"]
        return response

    # -- locate ----------------------------------------------------------

    def locate(self, body: LocateRequest, *, debug: bool) -> Dict[str, Any]:
        """Return camera-centred and gripper XY for one detected point."""
        size = image_size_pair(body.image_size)
        pose = self._parse_xy(body.robot_pose, "robot_pose")
        point = check_image_point(body.image_uv, size, "image_uv")

        recorder: Optional[RunRecorder] = None
        if debug:
            recorder = RunRecorder(
                self.settings.runs_dir,
                "locate",
                {
                    "image_size": [size[0], size[1]],
                    "image_uv": [point[0], point[1]],
                    "robot_pose": [pose[0], pose[1]],
                    "vision_matrix": body.vision_matrix,
                    "offset_matrix": body.offset_matrix,
                    "radius_px": body.radius_px,
                },
            )

        try:
            location = self.pick.locate(
                point,
                pose,
                body.vision_matrix,
                body.offset_matrix,
                image_size=size,
                radius_px=body.radius_px,
            )
        except PickError as exc:
            self._fail(
                recorder,
                exc.code,
                exc.message,
                exc.status,
                {"valid": False, "reason": exc.code},
                extra=exc.details,
            )
            raise  # pragma: no cover - _fail always raises

        manifest = self._finish_success(recorder, location.as_dict())
        response: Dict[str, Any] = {
            "ok": True,
            "camera_center_xy": [
                float(location.camera_center_xy[0]),
                float(location.camera_center_xy[1]),
            ],
            "gripper_xy": None
            if location.gripper_xy is None
            else [
                float(location.gripper_xy[0]),
                float(location.gripper_xy[1]),
            ],
            "delta_xy": [
                float(location.delta_xy[0]),
                float(location.delta_xy[1]),
            ],
            "gripper_delta_xy": None
            if location.gripper_delta_xy is None
            else [
                float(location.gripper_delta_xy[0]),
                float(location.gripper_delta_xy[1]),
            ],
        }
        if manifest is not None:
            response["run_id"] = manifest["run_id"]
        return response


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _file_url(run_id: str, relative: str) -> str:
    """Build the public static URL for a run artifact."""
    return f"/files/{run_id}/{relative}"


def _error_payload(error: PickError) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "ok": False,
        "error": error.code,
        "message": error.message,
    }
    payload.update(error.details)
    return payload


def create_app(settings: Optional[PickSettings] = None) -> FastAPI:
    """Build the pick HTTP application."""
    settings = settings or PickSettings.from_env()
    service = PickService(settings)

    app = FastAPI(
        title="Gantry pick calibration API",
        version="2.0.0",
        docs_url=f"{PREFIX}/docs",
        openapi_url=f"{PREFIX}/openapi.json",
        redoc_url=None,
    )
    app.state.service = service
    app.state.settings = settings

    if settings.cors_origins:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.middleware("http")
    async def cache_headers(request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/files/"):
            response.headers["Cache-Control"] = (
                "public, max-age=86400, immutable"
            )
        return response

    @app.exception_handler(PickError)
    async def pick_error(request: Request, error: PickError) -> JSONResponse:
        return JSONResponse(_error_payload(error), status_code=error.status)

    @app.exception_handler(RequestValidationError)
    async def validation_error(
        request: Request, error: RequestValidationError
    ) -> JSONResponse:
        first = error.errors()[0] if error.errors() else {}
        field = ".".join(str(part) for part in first.get("loc", []))
        message = first.get("msg", "request validation failed")
        return JSONResponse(
            {
                "ok": False,
                "error": "bad_request",
                "message": f"{field}: {message}".strip(": "),
            },
            status_code=400,
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error(
        request: Request, error: StarletteHTTPException
    ) -> JSONResponse:
        return JSONResponse(
            {
                "ok": False,
                "error": "http_error",
                "message": str(error.detail),
            },
            status_code=error.status_code,
        )

    @app.exception_handler(Exception)
    async def unhandled_error(request: Request, error: Exception) -> JSONResponse:
        logger.exception("unhandled pick API error")
        return JSONResponse(
            {
                "ok": False,
                "error": "internal_error",
                "message": str(error),
            },
            status_code=500,
        )

    # -- debug -----------------------------------------------------------

    @app.get(f"{PREFIX}/debug")
    def get_debug() -> Dict[str, Any]:
        return {"ok": True, "enabled": service.debug.enabled}

    @app.put(f"{PREFIX}/debug")
    def set_debug(body: DebugRequest) -> Dict[str, Any]:
        return {"ok": True, "enabled": service.debug.set(body.enabled)}

    # -- calibration -----------------------------------------------------

    @app.post(f"{PREFIX}/calibrate/vision")
    def calibrate_vision(body: VisionRequest) -> Dict[str, Any]:
        return service.calibrate_vision(body, debug=service.debug.enabled)

    @app.post(f"{PREFIX}/calibrate/offset")
    def calibrate_offset(body: OffsetRequest) -> Dict[str, Any]:
        return service.calibrate_offset(body, debug=service.debug.enabled)

    # -- locate ----------------------------------------------------------

    @app.post(f"{PREFIX}/locate")
    def locate(body: LocateRequest) -> Dict[str, Any]:
        return service.locate(body, debug=service.debug.enabled)

    # -- runs ------------------------------------------------------------

    @app.get(f"{PREFIX}/runs")
    def runs(limit: int = Query(20, ge=1, le=100)) -> Dict[str, Any]:
        summaries = list_runs(settings.runs_dir, limit=limit)
        for item in summaries:
            run_id = item["run_id"]
            relative = item.get("overview") or item.get("annotated")
            item["thumbnail"] = (
                _file_url(run_id, relative) if relative else None
            )
            item["url"] = f"{PREFIX}/runs/{run_id}"
            item.pop("overview", None)
            item.pop("annotated", None)
        return {"ok": True, "runs": summaries}

    @app.get(f"{PREFIX}/runs/{{run_id}}")
    def run_manifest(run_id: str) -> Dict[str, Any]:
        if not validate_run_id(run_id):
            raise PickError("run_not_found", f"unknown run: {run_id}", 404)
        path = settings.runs_dir / run_id
        try:
            manifest = read_manifest(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise PickError("run_not_found", str(exc), 404) from exc
        return {"ok": True, "run": manifest}

    @app.get(f"{PREFIX}/runs/{{run_id}}/archive")
    def run_archive(run_id: str) -> FileResponse:
        if not validate_run_id(run_id):
            raise PickError("run_not_found", f"unknown run: {run_id}", 404)
        run_dir = settings.runs_dir / run_id
        if not run_dir.is_dir():
            raise PickError("run_not_found", f"unknown run: {run_id}", 404)
        try:
            archive = zip_run(run_dir)
        except (OSError, ValueError) as exc:
            raise PickError("archive_failed", str(exc), 500) from exc
        return FileResponse(
            archive,
            media_type="application/zip",
            filename=f"{run_id}.zip",
            background=BackgroundTask(os.unlink, str(archive)),
        )

    app.mount(
        "/files",
        StaticFiles(directory=str(settings.runs_dir)),
        name="files",
    )
    return app


def main() -> None:
    """Run the service with uvicorn."""
    import uvicorn

    host = os.environ.get("PICK_HOST", "127.0.0.1")
    port = _env_int("PICK_PORT", 8100)
    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":
    main()
