"""Plane, stereo and projection calculations with upstream-owned portable assets."""

from __future__ import annotations

import json

from typing import List, Literal, Optional, Union

import cv2
import numpy as np
from pydantic import Field, model_validator

from measure_api.v2.calibration_assets import (
    UNIT_MM, CalibrationApplicability, CalibrationAsset, CalibrationInput, CameraCalibration,
    ImageSize, PhysicalUnit, calibration_digest, check_applicability, load_calibration,
)
from measure_api.v2.errors import ServiceError
from measure_api.v2.families.calibrations import (
    BlobOptions, BoardGrid, CalibrationResult, CameraObservation, DotGridRequest, ObjectPoint,
    asset_result, detect_dot_grid,
)
from measure_api.v2.families.images import encode_png
from measure_api.v2.files import ArtifactWriter, image_metadata
from measure_api.v2.models import (
    CalculationRequest, CoordinateMapping, FileDescriptor, FileReference, ImageMetadata,
    OutputSpec, Point, StrictModel,
)
from measure_api.v2.registry import CalculationContext, Operation


def matrix3(value: List[List[float]], name: str) -> np.ndarray:
    """Validate invertible projective matrices before imported assets can be used.

    Args:
        value: Supplied value to normalize under the public schema and coordinate
            convention.
        name: Semantic matrix or operation name used in validation diagnostics.

    Returns:
        Finite, invertible 3-by-3 projective matrix as a floating-point array.
    """
    matrix = np.asarray(value, dtype=float)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or abs(np.linalg.det(matrix)) < 1e-12):
        raise ValueError(f"{name} must be an invertible finite 3 by 3 matrix")
    return matrix


def rotation3(value: List[List[float]]) -> np.ndarray:
    """Validate proper orthogonal rotation rather than arbitrary 3D linear maps.

    Args:
        value: Supplied value to normalize under the public schema and coordinate
            convention.

    Returns:
        Proper orthogonal 3-by-3 rotation with determinant one.
    """
    rotation = matrix3(value, "rotation")
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-7)):
        raise ValueError("rotation must be a proper orthogonal matrix")
    return rotation


class PlaneCanvas(StrictModel):
    image_size: ImageSize
    pixel_size: float = Field(gt=0)
    unit: PhysicalUnit
    frame: str = Field(default="plane_canvas", min_length=1)
    plane_to_canvas: List[List[float]]
    canvas_to_plane: List[List[float]]
    bounds: List[float] = Field(min_length=4, max_length=4)

    @model_validator(mode="after")
    def check_canvas(self) -> PlaneCanvas:
        forward = matrix3(self.plane_to_canvas, "plane_to_canvas")
        reverse = matrix3(self.canvas_to_plane, "canvas_to_plane")
        x0, y0, x1, y1 = self.bounds
        if x1 <= x0 or y1 <= y0:
            raise ValueError("canvas bounds must span a positive physical area")
        expected = np.array([[1 / self.pixel_size, 0, -x0 / self.pixel_size],
                             [0, 1 / self.pixel_size, -y0 / self.pixel_size], [0, 0, 1]])
        if not np.allclose(forward, expected) or not np.allclose(forward @ reverse, np.eye(3)):
            raise ValueError("canvas matrices must match declared bounds and pixel size")
        expected_size = (int(np.ceil((x1 - x0) / self.pixel_size - 1e-6)) + 1,
                         int(np.ceil((y1 - y0) / self.pixel_size - 1e-6)) + 1)
        if expected_size != (self.image_size.width, self.image_size.height):
            raise ValueError("canvas dimensions must match declared bounds and pixel size")
        return self


class PlaneView(StrictModel):
    camera: CameraCalibration
    camera_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    capture_id: str = Field(min_length=1)
    source: Optional[FileReference] = None
    camera_to_plane: List[List[float]]
    plane_to_camera: List[List[float]]
    board_to_camera_rotation: List[List[float]]
    board_to_camera_translation: List[float] = Field(min_length=3, max_length=3)
    mapping: CoordinateMapping

    @model_validator(mode="after")
    def check_geometry(self) -> PlaneView:
        forward = matrix3(self.camera_to_plane, "camera_to_plane")
        reverse = matrix3(self.plane_to_camera, "plane_to_camera")
        rotation = rotation3(self.board_to_camera_rotation)
        expected = np.array(self.camera.parameters.camera_matrix) @ np.column_stack((
            rotation[:, :2], self.board_to_camera_translation))
        scale = reverse.flat[np.argmax(np.abs(reverse))] / expected.flat[
            np.argmax(np.abs(reverse))]
        if not np.allclose(reverse, expected * scale, atol=1e-6):
            raise ValueError("plane mapping must agree with declared board-to-camera pose")
        if not np.allclose(forward @ reverse, np.eye(3), atol=1e-6):
            raise ValueError("camera_to_plane and plane_to_camera must be inverses")
        if self.camera_sha256 != calibration_digest(self.camera):
            raise ValueError("camera digest does not match the embedded supplied asset")
        if self.mapping.source_frame != self.camera.applicability.frame:
            raise ValueError("view mapping source frame does not match camera applicability")
        return self


class PlaneParameters(StrictModel):
    views: List[PlaneView] = Field(min_length=1, max_length=2)
    canvas: PlaneCanvas
    origin: Point = Field(default_factory=lambda: Point(x=0, y=0))
    physical_frame: str = Field(default="board_plane", min_length=1)
    provenance: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_views(self) -> PlaneParameters:
        ids = [view.camera.applicability.camera_id for view in self.views]
        if len(set(ids)) != len(ids):
            raise ValueError("Plane views must have distinct camera identities")
        if len({view.capture_id for view in self.views}) != 1:
            raise ValueError("Plane views must share one paired capture identity")
        if len({view.camera.applicability.setup_id for view in self.views}) != 1:
            raise ValueError("Plane views must share the imaging setup")
        for view in self.views:
            mapping = view.mapping
            if mapping.destination_frame != self.canvas.frame:
                raise ValueError("View mapping destination must match the shared canvas frame")
            expected = np.array(self.canvas.plane_to_canvas) @ np.array(view.camera_to_plane)
            nonlinear = bool(np.any(np.array(view.camera.parameters.distortion) != 0))
            if mapping.type != ("nonlinear" if nonlinear else "projective"):
                raise ValueError("View mapping type must reflect declared lens distortion")
            if nonlinear:
                if mapping.matrix is not None or mapping.inverse_matrix is not None:
                    raise ValueError("A nonlinear lens mapping cannot claim one projective matrix")
            elif (mapping.matrix is None or mapping.inverse_matrix is None
                  or not np.allclose(matrix3(mapping.matrix, "mapping"), expected)
                  or not np.allclose(matrix3(mapping.inverse_matrix, "inverse mapping"),
                                     np.linalg.inv(expected))):
                raise ValueError("View mapping matrix must agree with calibrated canvas geometry")
            details = mapping.details
            if (details.get("camera_sha256") != view.camera_sha256
                    or not np.allclose(matrix3(details.get("corrected_to_canvas"),
                                               "corrected_to_canvas"), expected)
                    or not np.allclose(matrix3(details.get("canvas_to_corrected"),
                                               "canvas_to_corrected"), np.linalg.inv(expected))):
                raise ValueError("View correction metadata must agree with calibrated canvas")
        return self


class PlaneCalibration(CalibrationAsset):
    kind: Literal["plane"] = "plane"
    parameters: PlaneParameters

    @model_validator(mode="after")
    def check_units(self) -> PlaneCalibration:
        if self.unit != self.parameters.canvas.unit:
            raise ValueError("Canvas physical unit must match calibration unit")
        if self.applicability != self.parameters.views[0].camera.applicability:
            raise ValueError("Primary applicability must match the first plane camera")
        return self


class PlaneResult(CalibrationResult):
    asset: Optional[PlaneCalibration] = None


CALIBRATION_ASSETS = {"plane": PlaneCalibration}


class PlaneViewInput(StrictModel):
    camera: CalibrationInput
    applicability: CalibrationApplicability
    capture_id: str = Field(min_length=1)
    observation: Optional[CameraObservation] = None
    image: Optional[FileReference] = None

    @model_validator(mode="after")
    def one_observation(self) -> PlaneViewInput:
        if (self.observation is None) == (self.image is None):
            raise ValueError("Provide exactly one accepted observation or board image")
        if self.observation and self.observation.view_id != self.capture_id:
            raise ValueError("observation.view_id must equal its paired capture_id")
        return self


class PlaneRequest(CalculationRequest):
    views: List[PlaneViewInput] = Field(min_length=1, max_length=2)
    unit: PhysicalUnit = "mm"
    pixel_size: float = Field(gt=0)
    destination_frame: str = Field(default="plane_canvas", min_length=1)
    grid: Optional[BoardGrid] = None
    spacing: Optional[float] = Field(default=None, gt=0)
    symmetric: bool = True
    refine: bool = True
    blob: BlobOptions = Field(default_factory=BlobOptions)
    output: Optional[OutputSpec] = None

    @model_validator(mode="after")
    def paired_views(self) -> PlaneRequest:
        if len({view.capture_id for view in self.views}) != 1:
            raise ValueError("Plane views must share the same capture_id")
        if len({view.applicability.camera_id for view in self.views}) != len(self.views):
            raise ValueError("Each plane view needs a distinct camera")
        if len({view.applicability.setup_id for view in self.views}) != 1:
            raise ValueError("Plane views must share the same setup")
        if any(view.image is not None for view in self.views) and (
                self.grid is None or self.spacing is None):
            raise ValueError("Image observations require a grid and physical spacing")
        return self


def resolve_view(
    view: PlaneViewInput, request: PlaneRequest, context: CalculationContext,
) -> tuple:
    """Resolve camera and accepted/image-derived observation with provenance.

    Args:
        view: One identified camera capture with supplied or image-derived board
            observations.
        request: Typed paired board views, canvas pixel size, physical unit and origin.
        context: Isolated operation identity, resource limits and verified shared-file
            access.

    Returns:
        Resolved camera asset and accepted board observation with pinned source provenance.
    """
    camera = load_calibration(view.camera, context)
    if not isinstance(camera, CameraCalibration):
        raise ServiceError("calibration.camera_required", "A pinhole camera asset is required")
    check_applicability(camera, view.applicability)
    if view.observation:
        observation = view.observation
        if observation.source:
            image, pinned = context.files.read_image(observation.source)
            if image.shape[:2] != (observation.image_size.height, observation.image_size.width):
                raise ServiceError("calibration.image_size_mismatch",
                                   "Observation source dimensions do not match declared image")
            observation = observation.model_copy(update={"source": pinned})
        return camera, observation
    detection = detect_dot_grid(DotGridRequest(
        image=view.image, grid=request.grid, spacing=request.spacing, unit=request.unit,
        symmetric=request.symmetric, refine=request.refine, blob=request.blob,
    ), context)
    if not detection.valid:
        raise ServiceError("calibration.board_not_found", "Board was not detected",
                           details={"capture_id": view.capture_id,
                                    "camera_id": view.applicability.camera_id})
    return camera, CameraObservation(
        view_id=view.capture_id, image_size=detection.image_size,
        image_points=detection.image_points, object_points=detection.object_points,
        ordering=detection.ordering, source=detection.source,
    )


def observation_arrays(observation: CameraObservation, camera: CameraCalibration) -> tuple:
    """Validate planar, uniquely ordered and in-image physical correspondences.

    Args:
        observation: Ordered physical board points and their observed image correspondences.
        camera: Applicable intrinsic matrix and lens-distortion calibration.

    Returns:
        Physical xyz and image xy arrays, in that order, after correspondence validation.
    """
    obj = np.array([[p.x, p.y, p.z] for p in observation.object_points], np.float64).reshape(-1, 3)
    img = np.array([[p.x, p.y] for p in observation.image_points], np.float64).reshape(-1, 2)
    size = camera.applicability.image_size
    if observation.image_size != size:
        raise ServiceError("calibration.image_size_mismatch", "Observation dimensions mismatch")
    if (len(obj) != len(img) or len(obj) < 6 or len(np.unique(obj, axis=0)) != len(obj)
            or len(np.unique(img, axis=0)) != len(img) or not np.allclose(obj[:, 2], 0)
            or np.linalg.matrix_rank(obj[:, :2] - obj[:, :2].mean(axis=0)) != 2
            or np.linalg.matrix_rank(img - img.mean(axis=0)) != 2):
        raise ServiceError("calibration.invalid_observation",
                           "Require six unique noncollinear ordered z=0 board correspondences")
    if np.any(img < 0) or np.any(img[:, 0] >= size.width) or np.any(img[:, 1] >= size.height):
        raise ServiceError("calibration.observation_outside", "Board point lies outside image")
    return obj, img


def pose(observation: CameraObservation, camera: CameraCalibration) -> tuple:
    """Solve board-to-camera pose and return per-point pixel residuals.

    Args:
        observation: Ordered physical board points and their observed image correspondences.
        camera: Applicable intrinsic matrix and lens-distortion calibration.

    Returns:
        Board-to-camera rotation, translation and per-point reprojection residuals.
    """
    obj, img = observation_arrays(observation, camera)
    k = np.array(camera.parameters.camera_matrix)
    d = np.array(camera.parameters.distortion)
    try:
        success, rv, tv = cv2.solvePnP(obj, img, k, d, flags=cv2.SOLVEPNP_ITERATIVE)
    except cv2.error as error:
        raise ServiceError("calibration.pose_failed", "Board pose could not be solved") from error
    if not success or not np.isfinite(rv).all() or not np.isfinite(tv).all():
        raise ServiceError("calibration.pose_failed", "Board pose could not be solved")
    rotation = cv2.Rodrigues(rv)[0]
    if np.any((rotation @ obj.T + tv)[2] <= 0):
        raise ServiceError("calibration.board_behind_camera", "Solved board is behind camera")
    projected = cv2.projectPoints(obj, rv, tv, k, d)[0].reshape(-1, 2)
    return rotation, tv.reshape(3), np.linalg.norm(projected - img, axis=1)


def perspective(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Transform xy samples and reject near-zero or sign-crossing homogeneous denominators.

    Args:
        points: Ordered sample coordinates in the transform or camera source frame.
        matrix: 3-by-3 forward transform in homogeneous xy coordinates.

    Returns:
        Projected xy points. Near-zero or mixed-sign denominators raise a plane-horizon
        error; this helper does not independently check all inputs/outputs for finiteness.
    """
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    denominator = homogeneous[:, 2]
    if (np.any(np.abs(denominator) < 1e-9) or np.any(denominator > 0)
            and np.any(denominator < 0)):
        raise ServiceError("calibration.plane_horizon", "Mapped region crosses the plane horizon")
    return homogeneous[:, :2] / denominator[:, None]


def make_plane_view(
    camera: CameraCalibration, observation: CameraObservation, unit: PhysicalUnit,
) -> tuple:
    rotation, translation, errors = pose(observation, camera)
    k = np.array(camera.parameters.camera_matrix)
    plane_to_camera = k @ np.column_stack((rotation[:, :2], translation))
    camera_to_plane = np.linalg.inv(plane_to_camera)
    if abs(camera_to_plane[2, 2]) > 1e-12:
        camera_to_plane /= camera_to_plane[2, 2]
    plane_to_camera = np.linalg.inv(camera_to_plane)
    view = PlaneView(
        camera=camera, camera_sha256=calibration_digest(camera), capture_id=observation.view_id,
        source=observation.source, camera_to_plane=camera_to_plane.tolist(),
        plane_to_camera=plane_to_camera.tolist(), board_to_camera_rotation=rotation.tolist(),
        board_to_camera_translation=translation.tolist(), mapping=CoordinateMapping(
            source_frame=camera.applicability.frame, destination_frame="plane_canvas",
            type="projective", unit="px",
        ),
    )
    return view, errors


def plane_parameters(
    views: List[PlaneView], pixel_size: float, unit: PhysicalUnit, frame: str,
    origin: Optional[Point] = None, provenance: Optional[dict] = None,
) -> PlaneParameters:
    """Compute corrected-view union canvas and explicitly composed source mappings.

    Args:
        views: Resolved paired plane views with board poses and lens-correction mappings.
        pixel_size: Physical length per pixel in the shared plane canvas.
        unit: Physical length unit used by the supplied calibration and board geometry.
        frame: Coordinate-frame identity for the returned image or mapping.
        origin: Previously established physical origin recorded in the returned parameters;
            this argument alone does not shift view or canvas coordinates.
        provenance: Caller and calculation provenance retained in the portable asset.

    Returns:
        Validated common canvas, view mappings, physical origin and provenance.
    """
    corners = []
    for view in views:
        size = view.camera.applicability.image_size
        samples = np.array([[0, 0], [size.width - 1, 0],
                            [size.width - 1, size.height - 1], [0, size.height - 1]], np.float64)
        corners.extend(perspective(samples, np.array(view.camera_to_plane)).tolist())
    bounds = np.array(corners)
    x0, y0 = bounds.min(axis=0)
    x1, y1 = bounds.max(axis=0)
    width = int(np.ceil((x1 - x0) / pixel_size - 1e-6)) + 1
    height = int(np.ceil((y1 - y0) / pixel_size - 1e-6)) + 1
    if width > 32768 or height > 32768:
        raise ServiceError("calibration.canvas_too_large", "Use a coarser physical pixel_size")
    affine = np.array([[1 / pixel_size, 0, -x0 / pixel_size],
                       [0, 1 / pixel_size, -y0 / pixel_size], [0, 0, 1]])
    canvas = PlaneCanvas(image_size=ImageSize(width=width, height=height), pixel_size=pixel_size,
                         unit=unit, frame=frame, plane_to_canvas=affine.tolist(),
                         canvas_to_plane=np.linalg.inv(affine).tolist(),
                         bounds=[float(x0), float(y0), float(x1), float(y1)])
    for view in views:
        forward = affine @ np.array(view.camera_to_plane)
        nonlinear = bool(np.any(np.array(view.camera.parameters.distortion) != 0))
        view.mapping = CoordinateMapping(
            source_frame=view.camera.applicability.frame, destination_frame=frame,
            type="nonlinear" if nonlinear else "projective",
            matrix=None if nonlinear else forward.tolist(),
            inverse_matrix=None if nonlinear else np.linalg.inv(forward).tolist(),
            details={"corrected_frame": f"{view.camera.applicability.camera_id}:pinhole",
                     "corrected_camera_matrix": view.camera.parameters.camera_matrix,
                     "distortion": view.camera.parameters.distortion,
                     "source_to_corrected": "undistortPoints(K,D,P=K)",
                     "corrected_to_canvas": forward.tolist(),
                     "canvas_to_corrected": np.linalg.inv(forward).tolist(),
                     "camera_sha256": view.camera_sha256,
                     "geometry_policy": "transform_sample_points_and_curves",
                     "physical_unit": unit},
        )
    return PlaneParameters(views=views, canvas=canvas, origin=origin or Point(x=0, y=0),
                           provenance=provenance or {})


def error_summary(errors: np.ndarray) -> dict:
    """Shared aggregate metrics retain all individual errors separately.

    Args:
        errors: Individual measured residuals used to compute aggregate quality metrics.

    Returns:
        Mean, RMS and maximum pixel residuals, plus the observed point count.
    """
    return {"mean_error_px": float(np.mean(errors)),
            "rms_error_px": float(np.sqrt(np.mean(np.square(errors)))),
            "max_error_px": float(np.max(errors)), "point_count": len(errors)}


def compute_plane(request: PlaneRequest, context: CalculationContext) -> CalibrationResult:
    """Establish a shared physical board canvas from explicitly paired camera views.

    Args:
        request: Typed paired board views, canvas pixel size, physical unit and origin.
        context: Isolated operation identity, resource limits and verified shared-file
            access.

    Returns:
        Validated asset, source identities, quality diagnostics and optional committed
        artifacts.
    """
    views, diagnostics, all_errors, model = [], [], [], None
    for source in request.views:
        try:
            camera, observation = resolve_view(source, request, context)
            objects, _ = observation_arrays(observation, camera)
            view, errors = make_plane_view(camera, observation, request.unit)
        except ServiceError as error:
            if error.code not in {"calibration.board_not_found", "calibration.pose_failed",
                                  "calibration.invalid_observation",
                                  "calibration.board_behind_camera"}:
                raise
            return PlaneResult(valid=False, diagnostics={
                "reason": error.code, "message": error.message,
                "camera_id": source.applicability.camera_id, "capture_id": source.capture_id})
        if model is not None and not np.array_equal(model, objects):
            raise ServiceError("calibration.board_mismatch", "Paired views require identical board "
                               "coordinates and ordering in the declared request unit")
        model = objects
        views.append(view)
        all_errors.extend(errors.tolist())
        diagnostics.append({"camera_id": source.applicability.camera_id,
                            "capture_id": source.capture_id, "residuals_px": errors.tolist(),
                            **error_summary(errors)})
    parameters = plane_parameters(views, request.pixel_size, request.unit,
                                 request.destination_frame)
    asset = PlaneCalibration(unit=request.unit, applicability=views[0].camera.applicability,
                             parameters=parameters)
    return PlaneResult(**asset_result(asset, context, request.output,
                       {"views": diagnostics, **error_summary(np.array(all_errors))}).model_dump())


OPERATIONS = [Operation(
    "/calibrations/plane-mappings", PlaneRequest, PlaneResult, compute_plane,
    "Compute paired board-plane mappings and physical canvas", "plane_stereo",
    {"asset": "calibration"},
)]


class StereoParameters(PlaneParameters):
    views: List[PlaneView] = Field(min_length=2, max_length=2)
    direction: Literal["camera_a_to_camera_b"] = "camera_a_to_camera_b"
    rotation: List[List[float]]
    translation: List[float] = Field(min_length=3, max_length=3)
    essential: List[List[float]]
    fundamental: List[List[float]]
    baseline: float = Field(gt=0)
    reprojection_error_px: float = Field(ge=0)

    @model_validator(mode="after")
    def check_stereo(self) -> StereoParameters:
        rotation = rotation3(self.rotation)
        translation = np.asarray(self.translation)
        if not np.isclose(np.linalg.norm(translation), self.baseline):
            raise ValueError("baseline must equal the translation magnitude")
        x, y, z = translation
        skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        essential = np.array(self.essential, dtype=float)
        fundamental = np.array(self.fundamental, dtype=float)
        ka = np.array(self.views[0].camera.parameters.camera_matrix)
        kb = np.array(self.views[1].camera.parameters.camera_matrix)
        if (essential.shape != (3, 3) or fundamental.shape != (3, 3)
                or not np.allclose(essential, skew @ rotation, atol=1e-6)
                or not np.allclose(fundamental, np.linalg.inv(kb).T @ essential
                                   @ np.linalg.inv(ka), atol=1e-6)):
            raise ValueError("Essential/fundamental matrices must match declared pose/intrinsics")
        a, b = self.views
        ra, rb = np.array(a.board_to_camera_rotation), np.array(b.board_to_camera_rotation)
        ta = np.array(a.board_to_camera_translation)
        tb = np.array(b.board_to_camera_translation)
        if (not np.allclose(rb, rotation @ ra, atol=1e-6)
                or not np.allclose(tb, rotation @ ta + translation, atol=1e-6)):
            raise ValueError("Shared plane poses must agree with the stereo relative pose")
        return self


class StereoCalibration(PlaneCalibration):
    kind: Literal["stereo"] = "stereo"
    parameters: StereoParameters


class StereoResult(CalibrationResult):
    asset: Optional[StereoCalibration] = None


CALIBRATION_ASSETS["stereo"] = StereoCalibration


class StereoViewInput(StrictModel):
    capture_id: str = Field(min_length=1)
    observation: Optional[CameraObservation] = None
    image: Optional[FileReference] = None

    @model_validator(mode="after")
    def check_source(self) -> StereoViewInput:
        if (self.observation is None) == (self.image is None):
            raise ValueError("Provide one accepted observation or board image")
        if self.observation and self.observation.view_id != self.capture_id:
            raise ValueError("observation.view_id must equal capture_id")
        return self


class StereoPair(StrictModel):
    capture_id: str = Field(min_length=1)
    a: StereoViewInput
    b: StereoViewInput

    @model_validator(mode="after")
    def paired(self) -> StereoPair:
        if self.capture_id != self.a.capture_id or self.capture_id != self.b.capture_id:
            raise ValueError("Both camera views must have the explicit same paired capture_id")
        return self


class StereoRequest(CalculationRequest):
    camera_a: CalibrationInput
    camera_b: CalibrationInput
    applicability_a: CalibrationApplicability
    applicability_b: CalibrationApplicability
    pairs: List[StereoPair] = Field(min_length=1)
    unit: PhysicalUnit = "mm"
    pixel_size: float = Field(gt=0)
    destination_frame: str = Field(default="plane_canvas", min_length=1)
    fix_intrinsics: bool = True
    grid: Optional[BoardGrid] = None
    spacing: Optional[float] = Field(default=None, gt=0)
    symmetric: bool = True
    refine: bool = True
    blob: BlobOptions = Field(default_factory=BlobOptions)
    output: Optional[OutputSpec] = None

    @model_validator(mode="after")
    def check_pairing(self) -> StereoRequest:
        identities = [pair.capture_id for pair in self.pairs]
        if len(set(identities)) != len(identities):
            raise ValueError("Paired capture identities must be unique")
        if self.applicability_a.camera_id == self.applicability_b.camera_id:
            raise ValueError("Stereo calibration requires distinct cameras")
        if self.applicability_a.setup_id != self.applicability_b.setup_id:
            raise ValueError("Stereo camera views must share a setup")
        if any(view.image is not None for pair in self.pairs for view in [pair.a, pair.b]):
            if self.grid is None or self.spacing is None:
                raise ValueError("Image pairs require grid and physical spacing")
        if (not self.fix_intrinsics
                and self.applicability_a.image_size != self.applicability_b.image_size):
            raise ValueError("Intrinsic refinement requires equal camera image sizes")
        return self


def pair_views(pair: StereoPair, request: StereoRequest) -> List[PlaneViewInput]:
    """Convert the paired source envelope to the shared observation resolver.

    Args:
        pair: Explicitly paired captures from the two calibrated cameras.
        request: Typed paired board captures, camera assets and fixed/refined intrinsic
            options.

    Returns:
        First and second camera view inputs retaining their shared capture identity.
    """
    return [PlaneViewInput(camera=source, applicability=applicability,
                           capture_id=pair.capture_id, observation=view.observation,
                           image=view.image) for source, applicability, view in [
                               (request.camera_a, request.applicability_a, pair.a),
                               (request.camera_b, request.applicability_b, pair.b)]]


def project_camera(points: np.ndarray, camera: CameraCalibration) -> np.ndarray:
    """Project already known 3D camera-frame coordinates with physical visibility.

    Args:
        points: Ordered sample coordinates in the transform or camera source frame.
        camera: Applicable intrinsic matrix and lens-distortion calibration.

    Returns:
        Observed image xy coordinates for visible camera-frame 3D points.
    """
    if len(points) == 0:
        return np.empty((0, 2))
    if np.any(points[:, 2] <= 0):
        raise ServiceError("geometry.behind_camera", "Projection requires positive camera depth")
    return cv2.projectPoints(points.astype(np.float64), np.zeros(3), np.zeros(3),
                             np.array(camera.parameters.camera_matrix),
                             np.array(camera.parameters.distortion))[0].reshape(-1, 2)


def validate_pair(
    a: CameraObservation, b: CameraObservation, camera_a: CameraCalibration,
    camera_b: CameraCalibration, rotation: np.ndarray, translation: np.ndarray,
) -> dict:
    """Validate both cross-camera directions, retaining each observed/projected dot.

    Args:
        a: Accepted ordered board observation from the first camera.
        b: Accepted ordered board observation from the second camera.
        camera_a: Intrinsic and distortion calibration for the first camera.
        camera_b: Intrinsic and distortion calibration for the second camera.
        rotation: Proper 3-by-3 rotation from the first camera to the second.
        translation: Three-component camera-to-camera displacement in the physical unit.

    Returns:
        Per-camera observed/projected residuals and shared pair validation metrics.
    """
    objects_a, pixels_a = observation_arrays(a, camera_a)
    objects_b, pixels_b = observation_arrays(b, camera_b)
    if not np.array_equal(objects_a, objects_b):
        raise ServiceError("calibration.board_mismatch", "Paired board ordering/coordinates differ")
    ra, ta, own_a = pose(a, camera_a)
    rb, tb, own_b = pose(b, camera_b)
    camera_points_a = objects_a @ ra.T + ta
    camera_points_b = objects_b @ rb.T + tb
    projected_b = project_camera(camera_points_a @ rotation.T + translation, camera_b)
    projected_a = project_camera((camera_points_b - translation) @ rotation, camera_a)
    errors_a = np.linalg.norm(projected_a - pixels_a, axis=1)
    errors_b = np.linalg.norm(projected_b - pixels_b, axis=1)
    return {"capture_id": a.view_id, "a_from_b_residuals_px": errors_a.tolist(),
            "b_from_a_residuals_px": errors_b.tolist(),
            "a_from_b_projected": projected_a.tolist(), "b_from_a_projected": projected_b.tolist(),
            "a_observed": pixels_a.tolist(), "b_observed": pixels_b.tolist(),
            "a_own_residuals_px": own_a.tolist(), "b_own_residuals_px": own_b.tolist(),
            "a_source": a.source.model_dump(mode="json") if a.source else None,
            "b_source": b.source.model_dump(mode="json") if b.source else None,
            **error_summary(np.concatenate((errors_a, errors_b)))}


def compute_stereo(request: StereoRequest, context: CalculationContext) -> CalibrationResult:
    """Calibrate fixed/refined intrinsics on independent, explicitly identified captures.

    Args:
        request: Typed paired board captures, camera assets and fixed/refined intrinsic
            options.
        context: Isolated operation identity, resource limits and verified shared-file
            access.

    Returns:
        Validated asset, source identities, quality diagnostics and optional committed
        artifacts.
    """
    accepted, rejected, seen = [], [], set()
    cameras = None
    for pair in request.pairs:
        try:
            resolved = [resolve_view(view, request, context) for view in pair_views(pair, request)]
            camera_a, a = resolved[0]
            camera_b, b = resolved[1]
            obj_a, img_a = observation_arrays(a, camera_a)
            obj_b, img_b = observation_arrays(b, camera_b)
            if not np.array_equal(obj_a, obj_b):
                raise ServiceError("calibration.board_mismatch", "Paired board coordinates differ")
            signature = (obj_a.tobytes(), img_a.tobytes(), img_b.tobytes())
            if signature in seen:
                raise ServiceError("calibration.duplicate_pair", "Repeated paired observation")
            seen.add(signature)
            accepted.append((a, b, obj_a, img_a, img_b))
            cameras = (camera_a, camera_b)
        except ServiceError as error:
            if error.code not in {"calibration.board_not_found", "calibration.invalid_observation",
                                  "calibration.observation_outside", "calibration.duplicate_pair",
                                  "calibration.board_mismatch"}:
                raise
            rejected.append({"capture_id": pair.capture_id, "code": error.code,
                             "reason": error.message})
    if len(accepted) < 3:
        return StereoResult(valid=False, diagnostics={
            "reason": "at_least_three_distinct_usable_pairs_required", "rejected_pairs": rejected,
            "accepted_pairs": [{"capture_id": item[0].view_id} for item in accepted]})
    ca, cb = cameras
    size = ca.applicability.image_size
    try:
        result = cv2.stereoCalibrate(
            [item[2].astype(np.float32) for item in accepted],
            [item[3].astype(np.float32) for item in accepted],
            [item[4].astype(np.float32) for item in accepted],
            np.array(ca.parameters.camera_matrix), np.array(ca.parameters.distortion),
            np.array(cb.parameters.camera_matrix), np.array(cb.parameters.distortion),
            (size.width, size.height),
            flags=(cv2.CALIB_FIX_INTRINSIC if request.fix_intrinsics
                   else cv2.CALIB_USE_INTRINSIC_GUESS),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-9),
        )
    except cv2.error as error:
        return StereoResult(valid=False, diagnostics={"reason": "stereo_solve_failed",
                                                          "message": str(error)})
    rms, ka, da, kb, db, rotation, translation, essential, fundamental = result[:9]
    if not all(np.isfinite(v).all() for v in result[:9]):
        return StereoResult(valid=False, diagnostics={"reason": "nonfinite_stereo_solution"})
    translation = translation.reshape(3)
    x, y, z = translation
    essential = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]]) @ rotation
    fundamental = np.linalg.inv(kb).T @ essential @ np.linalg.inv(ka)
    if np.linalg.norm(translation) < 1e-9:
        return StereoResult(valid=False, diagnostics={"reason": "zero_stereo_baseline"})
    original_digests = [calibration_digest(ca), calibration_digest(cb)]
    ca, cb = ca.model_copy(deep=True), cb.model_copy(deep=True)
    if not request.fix_intrinsics:
        for cam, k, d in [(ca, ka, da), (cb, kb, db)]:
            cam.parameters.camera_matrix = k.tolist()
            cam.parameters.distortion = d.reshape(-1).tolist()
    details = [validate_pair(a, b, ca, cb, rotation, translation) for a, b, *_ in accepted]
    first_a, first_b = accepted[0][:2]
    plane_a = make_plane_view(ca, first_a, request.unit)[0]
    # The B plane pose is constrained by the solved stereo transform, rather than two
    # independent poses which could disagree on the shared measurement plane.
    ra = np.array(plane_a.board_to_camera_rotation)
    ta = np.array(plane_a.board_to_camera_translation)
    plane_b = make_plane_view(cb, first_b, request.unit)[0]
    rb, tb = rotation @ ra, rotation @ ta + translation
    pb = np.array(cb.parameters.camera_matrix) @ np.column_stack((rb[:, :2], tb))
    plane_b.board_to_camera_rotation = rb.tolist()
    plane_b.board_to_camera_translation = tb.tolist()
    plane_b.plane_to_camera = pb.tolist()
    plane_b.camera_to_plane = np.linalg.inv(pb).tolist()
    parameters = plane_parameters(
        [plane_a, plane_b], request.pixel_size, request.unit, request.destination_frame,
        provenance={"capture_ids": [a.view_id for a, *_ in accepted],
                    "supplied_camera_sha256": original_digests,
                    "paired_sources": [{"capture_id": a.view_id,
                                        "a": a.source.model_dump(mode="json") if a.source else None,
                                        "b": b.source.model_dump(mode="json") if b.source else None}
                                       for a, b, *_ in accepted],
                    "fix_intrinsics": request.fix_intrinsics},
    )
    errors = np.array([e for item in details for key in ["a_from_b_residuals_px",
                                                      "b_from_a_residuals_px"] for e in item[key]])
    stereo = StereoParameters(**parameters.model_dump(), rotation=rotation.tolist(),
                              translation=translation.tolist(), essential=essential.tolist(),
                              fundamental=fundamental.tolist(), baseline=float(np.linalg.norm(
                                  translation)), reprojection_error_px=float(rms))
    return StereoResult(**asset_result(
        StereoCalibration(unit=request.unit, applicability=ca.applicability, parameters=stereo),
        context, request.output, {"accepted_pairs": details, "rejected_pairs": rejected,
                                 "stereo_solver_rms_px": float(rms), **error_summary(errors)},
    ).model_dump())


OPERATIONS.append(Operation(
    "/calibrations/stereo-rigs", StereoRequest, StereoResult, compute_stereo,
    "Calibrate relative camera pose from explicitly paired observations", "plane_stereo",
    {"asset": "calibration"},
))


class OriginRequest(CalculationRequest):
    calibration: CalibrationInput
    origin: Point
    unit: PhysicalUnit
    output: Optional[OutputSpec] = None


class PlaneStereoResult(CalibrationResult):
    asset: Optional[Union[PlaneCalibration, StereoCalibration]] = None


def adjust_origin(request: OriginRequest, context: CalculationContext) -> CalibrationResult:
    """Return a new physical frame whose zero is the supplied old-frame position.

    Args:
        request: Typed existing plane calibration and requested new physical origin.
        context: Isolated operation identity, resource limits and verified shared-file
            access.

    Returns:
        Validated asset, source identities, quality diagnostics and optional committed
        artifacts.
    """
    asset = load_calibration(request.calibration, context)
    if not isinstance(asset, PlaneCalibration):
        raise ServiceError("calibration.plane_required", "Physical origin needs a board plane")
    parent = calibration_digest(asset)
    asset = asset.model_copy(deep=True)
    factor = UNIT_MM[request.unit] / UNIT_MM[asset.unit]
    dx, dy = request.origin.x * factor, request.origin.y * factor
    shift = np.array([[1, 0, -dx], [0, 1, -dy], [0, 0, 1]])
    parameters = asset.parameters
    for view in parameters.views:
        h = shift @ np.array(view.camera_to_plane)
        view.camera_to_plane = h.tolist()
        view.plane_to_camera = np.linalg.inv(h).tolist()
        rotation = np.array(view.board_to_camera_rotation)
        translation = np.array(view.board_to_camera_translation) + rotation @ [dx, dy, 0]
        view.board_to_camera_translation = translation.tolist()
    parameters.origin = Point(x=parameters.origin.x + dx, y=parameters.origin.y + dy)
    recomputed = plane_parameters(parameters.views, parameters.canvas.pixel_size, asset.unit,
                                  parameters.canvas.frame, parameters.origin,
                                  {**parameters.provenance, "parent_sha256": parent,
                                   "origin_operation": "old_frame_position_becomes_new_zero"})
    if isinstance(asset, StereoCalibration):
        updated = parameters.model_dump()
        updated.update(recomputed.model_dump())
        asset.parameters = StereoParameters.model_validate(updated)
    else:
        asset.parameters = recomputed
    return PlaneStereoResult(**asset_result(
        asset, context, request.output, {"parent_sha256": parent, "origin_unit": asset.unit},
    ).model_dump())


def validate_board_observations(
    asset: CalibrationAsset, observations: list, unit: PhysicalUnit,
    context: CalculationContext,
) -> dict:
    """Validate known-board samples against existing supplied calibration geometry.

    Args:
        asset: Validated calibration value, including its declared setup and frame.
        observations: Known-board validation captures with explicit point ordering and
            identity.
        unit: Physical length unit used by the supplied calibration and board geometry.
        context: Isolated operation identity, resource limits and verified shared-file
            access.

    Returns:
        Residuals grouped by view or paired-camera capture, with aggregate mean, RMS,
        maximum pixel error and point count. No accepted/rejected capture lists are returned.
    """
    if not isinstance(asset, (PlaneCalibration, CameraCalibration)):
        raise ServiceError("calibration.board_validation_unsupported",
                           "Board reprojection requires camera, plane or stereo calibration")
    ratio = UNIT_MM[unit] / UNIT_MM[asset.unit]
    normalized = []
    for observation in observations:
        if observation.source:
            image, _ = context.files.read_image(observation.source)
            if image.shape[:2] != (observation.image_size.height, observation.image_size.width):
                raise ServiceError("calibration.image_size_mismatch",
                                   "Observation source dimensions differ from declared size")
        normalized.append(CameraObservation(
            **{**observation.model_dump(exclude={"camera_id"}),
               "object_points": [ObjectPoint(x=p.x * ratio, y=p.y * ratio, z=p.z * ratio)
                                 for p in observation.object_points]}))
    if isinstance(asset, StereoCalibration):
        cameras = [view.camera for view in asset.parameters.views]
        groups = {}
        for original, observation in zip(observations, normalized):
            if original.camera_id not in {cam.applicability.camera_id for cam in cameras}:
                raise ServiceError("calibration.unknown_camera", "Unknown observation camera")
            group = groups.setdefault(observation.view_id, {})
            if original.camera_id in group:
                raise ServiceError("calibration.duplicate_pair", "Repeated camera in capture")
            group[original.camera_id] = observation
        diagnostics = []
        for capture, group in groups.items():
            if len(group) != 2:
                raise ServiceError("calibration.unpaired_capture",
                                   "Stereo validation needs both cameras for each capture")
            diagnostics.append(validate_pair(group[cameras[0].applicability.camera_id],
                                             group[cameras[1].applicability.camera_id],
                                             cameras[0], cameras[1],
                                             np.array(asset.parameters.rotation),
                                             np.array(asset.parameters.translation)))
        errors = np.array([e for item in diagnostics for key in ["a_from_b_residuals_px",
                                                               "b_from_a_residuals_px"]
                           for e in item[key]])
        return {"pairs": diagnostics, **error_summary(errors)}
    details, errors_all = [], []
    for original, observation in zip(observations, normalized):
        if isinstance(asset, CameraCalibration):
            camera = asset
            if original.camera_id != camera.applicability.camera_id:
                raise ServiceError("calibration.unknown_camera", "Unknown camera observation")
            _, _, errors = pose(observation, camera)
        else:
            view = next((view for view in asset.parameters.views
                         if view.camera.applicability.camera_id == original.camera_id), None)
            if view is None:
                raise ServiceError("calibration.unknown_camera", "Unknown plane camera")
            obj, img = observation_arrays(observation, view.camera)
            camera_points = (obj @ np.array(view.board_to_camera_rotation).T
                             + np.array(view.board_to_camera_translation))
            projected = project_camera(camera_points, view.camera)
            errors = np.linalg.norm(projected - img, axis=1)
        errors_all.extend(errors.tolist())
        details.append({"camera_id": original.camera_id, "capture_id": observation.view_id,
                        "residuals_px": errors.tolist(), **error_summary(errors)})
    return {"views": details, **error_summary(np.array(errors_all))}


OPERATIONS.append(Operation(
    "/calibration-origin-adjustments", OriginRequest, PlaneStereoResult, adjust_origin,
    "Generate revised physical origin without mutating the supplied asset", "plane_stereo",
    {"asset": "calibration"},
))


class StereoTransformRequest(CalculationRequest):
    calibration: CalibrationInput
    source_camera: str = Field(min_length=1)
    destination_camera: str = Field(min_length=1)
    points: List[ObjectPoint] = Field(min_length=1)
    unit: PhysicalUnit
    target_unit: PhysicalUnit


class StereoTransformResult(StrictModel):
    points: List[ObjectPoint]
    unit: PhysicalUnit
    source_frame: str
    destination_frame: str
    direction: Literal["source_to_destination"] = "source_to_destination"
    matrix: List[List[float]]
    matrix_unit: PhysicalUnit
    calibration_sha256: str


def transform_stereo(
    request: StereoTransformRequest, context: CalculationContext,
) -> StereoTransformResult:
    """Map supplied known 3D coordinates between the two calibrated camera frames.

    Args:
        request: Typed known camera-frame 3D points and applicable stereo transform.
        context: Isolated operation identity, resource limits and verified shared-file
            access.

    Returns:
        Known 3D coordinates transformed between the calibrated camera frames.
    """
    asset = load_calibration(request.calibration, context)
    if not isinstance(asset, StereoCalibration):
        raise ServiceError("calibration.stereo_required", "A stereo relative pose is required")
    ids = [view.camera.applicability.camera_id for view in asset.parameters.views]
    if request.source_camera not in ids or request.destination_camera not in ids:
        raise ServiceError("calibration.unknown_camera", "Requested camera is outside the rig")
    rotation = np.array(asset.parameters.rotation)
    translation = np.array(asset.parameters.translation)
    if request.source_camera == request.destination_camera:
        rotation, translation = np.eye(3), np.zeros(3)
    elif request.source_camera == ids[1]:
        rotation, translation = rotation.T, -rotation.T @ translation
    points = np.array([[p.x, p.y, p.z] for p in request.points])
    points *= UNIT_MM[request.unit] / UNIT_MM[asset.unit]
    mapped = points @ rotation.T + translation
    mapped *= UNIT_MM[asset.unit] / UNIT_MM[request.target_unit]
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return StereoTransformResult(
        points=[ObjectPoint(x=x, y=y, z=z) for x, y, z in mapped], unit=request.target_unit,
        source_frame=f"camera:{request.source_camera}",
        destination_frame=f"camera:{request.destination_camera}", matrix=matrix.tolist(),
        matrix_unit=asset.unit, calibration_sha256=calibration_digest(asset),
    )


class ProjectionPose(StrictModel):
    """Optional explicit transform from a supplied object frame into camera 3D."""

    rotation: List[List[float]]
    translation: ObjectPoint
    unit: PhysicalUnit

    @model_validator(mode="after")
    def valid_rotation(self) -> ProjectionPose:
        rotation3(self.rotation)
        return self


class CameraProjectionRequest(CalculationRequest):
    calibration: CalibrationInput
    applicability: CalibrationApplicability
    points: List[ObjectPoint] = Field(min_length=1)
    unit: PhysicalUnit
    source_frame: str = Field(min_length=1)
    pose: Optional[ProjectionPose] = None


class CameraProjectionResult(StrictModel):
    points: List[Point]
    unit: Literal["px"] = "px"
    source_frame: str
    destination_frame: str
    direction: Literal["source_to_destination"] = "source_to_destination"
    inside_image: List[bool]
    calibration_sha256: str


def project_points(
    request: CameraProjectionRequest, context: CalculationContext,
) -> CameraProjectionResult:
    """Project known 3D points; no depth or arbitrary 3D reconstruction is inferred.

    Args:
        request: Typed known 3D camera points and applicable camera projection model.
        context: Isolated operation identity, resource limits and verified shared-file
            access.

    Returns:
        Projected image coordinates with per-point visibility and calibration identity.
    """
    asset = load_calibration(request.calibration, context)
    if not isinstance(asset, CameraCalibration):
        raise ServiceError("calibration.camera_required", "Projection needs camera intrinsics")
    check_applicability(asset, request.applicability)
    points = np.array([[p.x, p.y, p.z] for p in request.points])
    points *= UNIT_MM[request.unit] / UNIT_MM[asset.unit]
    if request.pose:
        rotation = np.array(request.pose.rotation)
        translation = request.pose.translation
        shift = np.array([translation.x, translation.y, translation.z])
        shift *= UNIT_MM[request.pose.unit] / UNIT_MM[asset.unit]
        points = points @ rotation.T + shift
    elif request.source_frame != f"camera:{asset.applicability.camera_id}":
        raise ServiceError("geometry.pose_required",
                           "An object frame needs an explicit camera pose")
    projected = project_camera(points, asset)
    size = asset.applicability.image_size
    return CameraProjectionResult(
        points=[Point(x=x, y=y) for x, y in projected], source_frame=request.source_frame,
        destination_frame=asset.applicability.frame,
        inside_image=[bool(0 <= x < size.width and 0 <= y < size.height) for x, y in projected],
        calibration_sha256=calibration_digest(asset),
    )


OPERATIONS.extend([
    Operation("/geometry/stereo-transforms", StereoTransformRequest, StereoTransformResult,
              transform_stereo, "Transform known 3D coordinates between camera frames",
              "plane_stereo", {"points": "physical_points"}),
    Operation("/geometry/camera-projections", CameraProjectionRequest, CameraProjectionResult,
              project_points, "Project known 3D coordinates into a supplied camera",
              "plane_stereo", {"points": "points"}),
])


class StitchImage(StrictModel):
    camera_id: str = Field(min_length=1)
    applicability: CalibrationApplicability
    capture_id: str = Field(min_length=1)
    image: FileReference

    @model_validator(mode="after")
    def camera_matches(self) -> StitchImage:
        if self.camera_id != self.applicability.camera_id:
            raise ValueError("camera_id must match declared image applicability")
        return self


class StitchRequest(CalculationRequest):
    calibration: CalibrationInput
    images: List[StitchImage] = Field(min_length=2, max_length=2)
    blend: Literal["average", "first", "last"] = "average"
    return_warped_views: bool = False
    output: OutputSpec

    @model_validator(mode="after")
    def paired(self) -> StitchRequest:
        if len({image.capture_id for image in self.images}) != 1:
            raise ValueError("Both images must share an explicit paired capture identity")
        if len({image.camera_id for image in self.images}) != 2:
            raise ValueError("Two distinct camera views are required")
        return self


class StitchResult(StrictModel):
    metadata: ImageMetadata
    sources: List[FileReference]
    mappings: List[CoordinateMapping]
    canvas: PlaneCanvas
    provenance: dict
    artifacts: List[FileDescriptor]
    manifest: FileDescriptor


def stitch_images(request: StitchRequest, context: CalculationContext) -> StitchResult:
    """Correct lens distortion and warp each paired view into the supplied physical plane.

    Args:
        request: Typed paired source images and their applicable shared-plane calibration.
        context: Isolated operation identity, resource limits and verified shared-file
            access.

    Returns:
        Committed canvas image, per-camera mappings and calibrated plane provenance.
    """
    asset = load_calibration(request.calibration, context)
    if not isinstance(asset, PlaneCalibration) or len(asset.parameters.views) != 2:
        raise ServiceError("calibration.two_view_plane_required", "Stitching needs two plane views")
    canvas = asset.parameters.canvas
    size = (canvas.image_size.width, canvas.image_size.height)
    if size[0] * size[1] > context.settings.max_image_pixels:
        raise ServiceError("image.too_large", "Plane canvas exceeds configured image pixel limit")
    warped, masks, sources, mappings, payloads = [], [], [], [], []
    shape = None
    for index, image_input in enumerate(request.images):
        view = next((view for view in asset.parameters.views
                     if view.camera.applicability.camera_id == image_input.camera_id), None)
        if view is None:
            raise ServiceError("calibration.unknown_camera",
                               "Image camera is outside supplied plane")
        check_applicability(view.camera, image_input.applicability)
        image, source = context.files.read_image(image_input.image)
        expected = view.camera.applicability.image_size
        if image.shape[:2] != (expected.height, expected.width):
            raise ServiceError("calibration.image_size_mismatch",
                               "Decoded image dimensions mismatch")
        if shape is not None and shape != image.ndim:
            raise ServiceError("image.incompatible_formats",
                               "Both views need the same pixel format")
        shape = image.ndim
        k = np.array(view.camera.parameters.camera_matrix)
        d = np.array(view.camera.parameters.distortion)
        mx, my = cv2.initUndistortRectifyMap(k, d, None, k,
                                          (expected.width, expected.height), cv2.CV_32FC1)
        corrected = cv2.remap(image, mx, my, cv2.INTER_LINEAR)
        corrected_mask = cv2.remap(np.full(image.shape[:2], 255, np.uint8), mx, my,
                                   cv2.INTER_NEAREST)
        matrix = np.array(canvas.plane_to_canvas) @ np.array(view.camera_to_plane)
        warped_image = cv2.warpPerspective(corrected, matrix, size, flags=cv2.INTER_LINEAR)
        mask = cv2.warpPerspective(corrected_mask, matrix, size,
                                   flags=cv2.INTER_NEAREST) > 0
        warped.append(warped_image)
        masks.append(mask)
        sources.append(source)
        mappings.append(view.mapping)
        payloads.append((f"mask-{index}.png", f"mask_{index}", "png",
                         encode_png(mask.astype(np.uint8) * 255)))
        if request.return_warped_views:
            payloads.append((f"view-{index}.png", f"warped_{index}", "png",
                             encode_png(warped_image)))
    coverage = np.sum(masks, axis=0).astype(np.uint8)
    result = np.zeros_like(warped[0])
    if request.blend == "average":
        sums = np.zeros_like(warped[0], np.float64)
        for image, mask in zip(warped, masks):
            sums += image * (mask[:, :, None] if image.ndim == 3 else mask)
        denominator = np.maximum(coverage, 1)
        if sums.ndim == 3:
            denominator = denominator[:, :, None]
        result = np.rint(sums / denominator).astype(np.uint8)
    else:
        ordered = range(2) if request.blend == "last" else reversed(range(2))
        for index in ordered:
            result[masks[index]] = warped[index][masks[index]]
    provenance = {"capture_id": request.images[0].capture_id,
                  "calibration_sha256": calibration_digest(asset),
                  "camera_ids": [image.camera_id for image in request.images],
                  "sources": [source.model_dump(mode="json") for source in sources],
                  "camera_sha256": [mapping.details["camera_sha256"] for mapping in mappings],
                  "blend": request.blend, "assumption": "all_measurement_points_on_board_plane"}
    payloads.extend([
        ("stitched.png", "stitched", "png", encode_png(result)),
        ("coverage.png", "coverage", "png", encode_png(coverage)),
        ("mappings.json", "mapping", "json", json.dumps({
            "schema_version": "1.0", "canvas": canvas.model_dump(mode="json"),
            "mappings": [mapping.model_dump(mode="json") for mapping in mappings],
            "provenance": provenance,
        }, sort_keys=True, allow_nan=False).encode("utf-8")),
    ])
    writer = ArtifactWriter(context.files, request.output, context.operation_id)
    artifacts, manifest = writer.commit(payloads)
    return StitchResult(metadata=image_metadata(result, canvas.frame), sources=sources,
                        mappings=mappings, canvas=canvas, provenance=provenance,
                        artifacts=artifacts, manifest=manifest)


OPERATIONS.append(Operation(
    "/image-stitches", StitchRequest, StitchResult, stitch_images,
    "Stitch paired camera images into a calibrated physical plane canvas", "plane_stereo",
    {"image": "image", "mappings": "transforms"},
))
