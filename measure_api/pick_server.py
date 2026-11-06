"""HTTP service for the three-axis gantry pick calibration.

Run with::

    python -m measure_api.pick_server

or::

    uvicorn --factory measure_api.pick_server:create_app --host 127.0.0.1 --port 8100

The service exposes three stateless business calls, a global debug switch and
read-only run artifact endpoints.  Calibration returns one 3x3 homogeneous
matrix per step; the caller stores both matrices and sends them back to
``/locate``.  The service never talks to a robot.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from fastapi import FastAPI, File, Form, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask
from starlette.exceptions import HTTPException as StarletteHTTPException

from measure.pick_api import (
    GantryPick,
    PickError,
    offset_homogeneous_matrix,
)
from measure.pick_artifacts import (
    RunRecorder,
    cleanup_runs,
    decode_image,
    list_runs,
    read_manifest,
    sha256_bytes,
    validate_run_id,
    zip_run,
)
from measure.robot_calibration import (
    CalibrationError,
    CameraGripperOffset,
    OffsetSample,
    fit_image_to_robot,
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
    """Offset calibration request."""

    samples: List[PoseSampleModel] = Field(..., min_length=1)


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

    def _record_vision_images(
        self,
        recorder: RunRecorder,
        images: Sequence[Tuple[str, bytes]],
        points: Sequence[Tuple[float, float]],
        decoded: Sequence[Optional[np.ndarray]],
        detections: Sequence[Optional[Any]],
        *,
        note: Optional[str] = None,
    ) -> None:
        for index, (filename, data) in enumerate(images):
            image = decoded[index] if index < len(decoded) else None
            detection = detections[index] if index < len(detections) else None
            recorder.add_image(
                index,
                filename,
                data,
                image,
                detection,
                robot_xy=points[index] if index < len(points) else None,
                note=note,
                error=note if image is None else None,
            )

    def _fail(
        self,
        recorder: Optional[RunRecorder],
        code: str,
        message: str,
        status: int,
        result: Dict[str, Any],
        *,
        artifact: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Publish a failed debug run (when enabled) and raise a stable error."""
        details: Dict[str, Any] = dict(extra or {})
        if recorder is not None:
            try:
                if artifact == "overview":
                    recorder.build_overview()
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

    def calibrate_vision(
        self,
        images: Sequence[Tuple[str, bytes]],
        robot_points: Sequence[Sequence[float]],
        *,
        debug: bool,
    ) -> Dict[str, Any]:
        """Return the fitted 3x3 vision matrix."""
        if len(images) < 4:
            raise PickError("bad_request", "at least 4 images are required", 400)
        if len(images) != len(robot_points):
            raise PickError(
                "image_count_mismatch",
                f"{len(images)} images but {len(robot_points)} robot points",
                400,
            )
        points = self._parse_points(robot_points)

        recorder: Optional[RunRecorder] = None
        if debug:
            recorder = RunRecorder(
                self.settings.runs_dir,
                "vision",
                {
                    "robot_points": [[x, y] for x, y in points],
                    "images": [
                        {
                            "filename": filename,
                            "sha256": sha256_bytes(data),
                            "bytes": len(data),
                        }
                        for filename, data in images
                    ],
                },
                annotated_max_px=self.settings.annotated_max_px,
            )

        decoded: List[Optional[np.ndarray]] = [
            decode_image(data) for _, data in images
        ]
        bad = [index for index, image in enumerate(decoded) if image is None]
        if bad:
            if recorder is not None:
                self._record_vision_images(
                    recorder, images, points, decoded, [None] * len(images),
                    note=f"aborted: image[{bad[0]}] is unreadable",
                )
            self._fail(
                recorder,
                "image_unreadable",
                f"image[{bad[0]}] cannot be decoded",
                422,
                {"valid": False},
                artifact="overview",
            )

        sizes = {
            (int(image.shape[1]), int(image.shape[0]))
            for image in decoded
            if image is not None
        }
        if len(sizes) != 1:
            if recorder is not None:
                self._record_vision_images(
                    recorder, images, points, decoded, [None] * len(images),
                    note="aborted: image sizes differ",
                )
            self._fail(
                recorder,
                "image_size_mismatch",
                f"calibration images have different sizes: {sorted(sizes)}",
                422,
                {"valid": False},
                artifact="overview",
            )
        width, height = next(iter(sizes))

        detections = [self.pick.locator.find(image) for image in decoded]
        invalid = [
            index
            for index, detection in enumerate(detections)
            if not detection.valid
        ]
        if invalid:
            if recorder is not None:
                self._record_vision_images(
                    recorder, images, points, decoded, detections,
                    note=f"target missing in image[{invalid[0]}]",
                )
            self._fail(
                recorder,
                "no_target",
                f"image[{invalid[0]}]: target not found "
                f"({detections[invalid[0]].reason or 'no_target'})",
                422,
                {"valid": False},
                artifact="overview",
                extra={"image_index": invalid[0]},
            )

        if recorder is not None:
            self._record_vision_images(
                recorder, images, points, decoded, detections
            )

        try:
            calibration = fit_image_to_robot(
                [detection.center for detection in detections],
                points,
                image_size=(width, height),
                confidences=[detection.confidence for detection in detections],
                metadata={"source": "pick_api"},
            )
        except CalibrationError as exc:
            self._fail(
                recorder,
                "calibration_failed",
                str(exc),
                422,
                {"valid": False},
                artifact="overview",
            )

        if not calibration.valid:
            if recorder is not None:
                for index, sample in enumerate(calibration.samples):
                    recorder.set_residual(index, sample.residual_px)
            self._fail(
                recorder,
                "calibration_invalid",
                f"rms {calibration.rms_px:.3f} px exceeds the quality gate",
                422,
                {"valid": False},
                artifact="overview",
                extra={
                    "rms_px": float(calibration.rms_px),
                    "warnings": list(calibration.warnings),
                },
            )

        matrix = np.asarray(calibration.matrix, dtype=float)
        if recorder is not None:
            for index, sample in enumerate(calibration.samples):
                recorder.set_residual(index, sample.residual_px)
            recorder.build_overview()

        debug_result: Dict[str, Any] = {
            "matrix": matrix.tolist(),
            "rms_px": float(calibration.rms_px),
            "rms_mm": float(calibration.rms_mm),
            "num_points": len(calibration.samples),
            "principal_point": [float(value) for value in calibration.principal_point],
            "image_size": None
            if calibration.image_size is None
            else [int(value) for value in calibration.image_size],
            "warnings": list(calibration.warnings),
        }
        manifest = self._finish_success(recorder, debug_result)
        response: Dict[str, Any] = {
            "ok": True,
            "matrix": matrix.tolist(),
        }
        if manifest is not None:
            response["run_id"] = manifest["run_id"]
        return response

    # -- offset calibration ---------------------------------------------

    def calibrate_offset(
        self,
        samples: Sequence[Mapping[str, Sequence[float]]],
        *,
        debug: bool,
    ) -> Dict[str, Any]:
        """Return the fitted 3x3 pure-translation offset matrix."""
        if not samples:
            raise PickError("bad_request", "samples must not be empty", 400)

        built: List[OffsetSample] = []
        raw: List[Dict[str, List[float]]] = []
        for index, sample in enumerate(samples):
            place = self._parse_xy(
                sample.get("place_pose", []), f"samples[{index}].place_pose"
            )
            centre = self._parse_xy(
                sample.get("center_pose", []), f"samples[{index}].center_pose"
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
                {"samples": raw},
                annotated_max_px=self.settings.annotated_max_px,
            )

        try:
            offset = CameraGripperOffset.from_samples(built)
        except CalibrationError as exc:
            self._fail(
                recorder,
                "offset_invalid",
                str(exc),
                422,
                {"valid": False},
                extra={"warnings": [str(exc)]},
            )

        if not offset.valid:
            self._fail(
                recorder,
                "offset_invalid",
                "; ".join(offset.warnings) or "offset calibration is not valid",
                422,
                {"valid": False},
                extra={"warnings": list(offset.warnings)},
            )

        matrix = offset_homogeneous_matrix(offset.offset_xy)
        debug_result = {
            "matrix": matrix.tolist(),
            "offset_xy": [float(value) for value in offset.offset_xy],
            "std_xy": [float(value) for value in offset.std_xy],
            "num_samples": len(offset.samples),
            "warnings": list(offset.warnings),
        }
        manifest = self._finish_success(recorder, debug_result)
        response: Dict[str, Any] = {
            "ok": True,
            "matrix": matrix.tolist(),
        }
        if manifest is not None:
            response["run_id"] = manifest["run_id"]
        return response

    # -- locate ----------------------------------------------------------

    def locate(
        self,
        image_bytes: bytes,
        filename: str,
        robot_pose: Sequence[float],
        vision_matrix: Sequence[Sequence[float]],
        offset_matrix: Optional[Sequence[Sequence[float]]],
        *,
        debug: bool,
    ) -> Dict[str, Any]:
        """Locate the target and return camera-centred and gripper XY."""
        pose = self._parse_xy(robot_pose, "robot_pose")
        image = decode_image(image_bytes)

        recorder: Optional[RunRecorder] = None
        if debug:
            recorder = RunRecorder(
                self.settings.runs_dir,
                "locate",
                {
                    "robot_pose": [pose[0], pose[1]],
                    "vision_matrix": vision_matrix,
                    "offset_matrix": offset_matrix,
                    "image": {
                        "filename": filename,
                        "sha256": sha256_bytes(image_bytes),
                        "bytes": len(image_bytes),
                    },
                },
                annotated_max_px=self.settings.annotated_max_px,
            )

        if image is None:
            if recorder is not None:
                recorder.add_image(
                    0, filename, image_bytes, None, None,
                    name="image", robot_xy=pose, note="image unreadable",
                    error="image_unreadable",
                )
            self._fail(
                recorder,
                "image_unreadable",
                "image cannot be decoded",
                422,
                {"valid": False},
                artifact="annotated",
            )

        try:
            location = self.pick.locate(
                image, pose, vision_matrix, offset_matrix
            )
        except PickError as exc:
            if recorder is not None:
                recorder.add_image(
                    0, filename, image_bytes, image, None,
                    name="image", robot_xy=pose, note=exc.message,
                    error=exc.code,
                )
                self._fail(
                    recorder,
                    exc.code,
                    exc.message,
                    exc.status,
                    {"valid": False, "reason": exc.code},
                    artifact="annotated",
                    extra=exc.details,
                )
            raise

        if not location.valid:
            if recorder is not None:
                recorder.add_image(
                    0, filename, image_bytes, image, location.detection,
                    name="image", robot_xy=pose,
                    note=location.reason, error=location.reason,
                )
            self._fail(
                recorder,
                location.reason,
                f"target not found ({location.reason})",
                422,
                location.as_dict(),
                artifact="annotated",
                extra={"confidence": float(location.confidence)},
            )

        if recorder is not None:
            height, width = image.shape[:2]
            extra_lines = [
                f"camera=({location.camera_center_xy[0]:.3f}, "
                f"{location.camera_center_xy[1]:.3f})",
                "grip=("
                + (
                    f"{location.gripper_xy[0]:.3f}, {location.gripper_xy[1]:.3f}"
                    if location.gripper_xy is not None
                    else "N/A"
                )
                + ")",
            ]
            recorder.add_image(
                0,
                filename,
                image_bytes,
                image,
                location.detection,
                name="image",
                robot_xy=pose,
                principal_point=((width - 1) * 0.5, (height - 1) * 0.5),
                extra_lines=extra_lines,
            )

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


def _read_upload(upload: UploadFile) -> bytes:
    """Read an uploaded file synchronously (the routes run in a thread pool)."""
    return upload.file.read()


def _filename(upload: UploadFile, fallback: str) -> str:
    name = upload.filename
    return name if name else fallback


def _parse_matrix_field(value: str, name: str) -> List[List[float]]:
    try:
        matrix = json.loads(value)
    except json.JSONDecodeError as exc:
        raise PickError(
            "bad_matrix", f"{name} is not valid JSON: {exc}", 400
        ) from exc
    if not isinstance(matrix, list):
        raise PickError("bad_matrix", f"{name} must be a 3x3 array", 400)
    return matrix


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


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
    def calibrate_vision(
        images: List[UploadFile] = File(...),
        robot_points: str = Form(...),
    ) -> Dict[str, Any]:
        try:
            points = json.loads(robot_points)
        except json.JSONDecodeError as exc:
            raise PickError(
                "bad_request", f"robot_points is not valid JSON: {exc}", 400
            ) from exc
        if not isinstance(points, list):
            raise PickError("bad_request", "robot_points must be an array", 400)
        payload = [
            (_filename(upload, f"image_{index}.jpg"), _read_upload(upload))
            for index, upload in enumerate(images)
        ]
        return service.calibrate_vision(
            payload, points, debug=service.debug.enabled
        )

    @app.post(f"{PREFIX}/calibrate/offset")
    def calibrate_offset(body: OffsetRequest) -> Dict[str, Any]:
        samples = [
            {
                "place_pose": item.place_pose,
                "center_pose": item.center_pose,
            }
            for item in body.samples
        ]
        return service.calibrate_offset(
            samples, debug=service.debug.enabled
        )

    # -- locate ----------------------------------------------------------

    @app.post(f"{PREFIX}/locate")
    def locate(
        image: UploadFile = File(...),
        robot_pose: str = Form(...),
        vision_matrix: str = Form(...),
        offset_matrix: Optional[str] = Form(None),
    ) -> Dict[str, Any]:
        try:
            pose = json.loads(robot_pose)
        except json.JSONDecodeError as exc:
            raise PickError(
                "bad_request", f"robot_pose is not valid JSON: {exc}", 400
            ) from exc
        if not isinstance(pose, list):
            raise PickError("bad_request", "robot_pose must be an array", 400)

        vision = _parse_matrix_field(vision_matrix, "vision_matrix")
        offset = (
            None
            if offset_matrix is None or offset_matrix.strip() in {"", "null"}
            else _parse_matrix_field(offset_matrix, "offset_matrix")
        )
        return service.locate(
            _read_upload(image),
            _filename(image, "image.jpg"),
            pose,
            vision,
            offset,
            debug=service.debug.enabled,
        )

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
