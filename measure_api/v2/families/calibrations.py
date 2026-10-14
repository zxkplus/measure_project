"""Scale/camera computations and portable upstream-owned asset utilities."""

from __future__ import annotations

import io
from typing import List, Literal, Optional, Union

import cv2
import numpy as np
from pydantic import Field, model_validator

from measure.measure_calibration import CameraCalibration as CoreCameraCalibration
from measure.pixel_calibration import CheckerboardScaleCalibration
from measure_api.v2.calibration_assets import (
    UNIT_MM, CameraCalibration, CameraParameters, CalibrationApplicability, CalibrationAsset,
    CalibrationInput, ImageSize, PhysicalUnit, ScalarCalibration, ScalarParameters,
    calibration_bytes,
    calibration_digest, check_applicability, load_calibration,
)
from measure_api.v2.errors import ServiceError
from measure_api.v2.families.images import encode_png
from measure_api.v2.files import ArtifactWriter, image_metadata
from measure_api.v2.models import (
    CalculationRequest, CoordinateMapping, FileDescriptor, FileReference, ImageMetadata,
    OutputSpec, Point, StrictModel,
)
from measure_api.v2.registry import CalculationContext, Operation


class ScaleRequest(CalculationRequest):
    corners: List[Point]
    square_size: float = Field(gt=0)
    unit: PhysicalUnit = "mm"
    applicability: CalibrationApplicability
    output: Optional[OutputSpec] = None


class CalibrationResult(StrictModel):
    valid: bool
    asset: Optional[Union[ScalarCalibration, CameraCalibration, CalibrationAsset]] = None
    sha256: Optional[str] = None
    size_bytes: Optional[int] = None
    diagnostics: dict = Field(default_factory=dict)
    artifacts: List[FileDescriptor] = Field(default_factory=list)
    manifest: Optional[FileDescriptor] = None


def compute_scale(request: ScaleRequest, context: CalculationContext) -> CalibrationResult:
    """Compute uniform scale from supplied or manually edited full-image corners.

    Args:
        request: Typed board corner observations, known square spacing, applicability and
            output destination.
        context: Isolated operation identity, resource limits and verified shared-file
            access.

    Returns:
        Validated asset, source identities, quality diagnostics and optional committed
        artifacts.
    """
    points = [(point.y, point.x) for point in request.corners]
    size = request.applicability.image_size
    if any(not (0 <= p.x < size.width and 0 <= p.y < size.height) for p in request.corners):
        raise ServiceError("calibration.observation_outside",
                           "Corner lies outside applicable image")
    coordinates = np.array(points, dtype=float).reshape(-1, 2)
    if (len(points) < 6 or len(set(points)) != len(points)
            or np.linalg.matrix_rank(coordinates - coordinates.mean(axis=0)) < 2):
        return CalibrationResult(valid=False, diagnostics={
            "reason": "at_least_six_unique_noncollinear_corners_required",
            "num_corners": len(points),
        })
    result = CheckerboardScaleCalibration().compute_scale(points, request.square_size)
    if not result.valid:
        return CalibrationResult(valid=False, diagnostics={"reason": result.error,
                                                          "num_corners": result.num_corners})
    asset = ScalarCalibration(unit=request.unit, applicability=request.applicability,
                              parameters=ScalarParameters(
                                  scale_per_px=result.scale_mm_per_px,
                                  square_size=request.square_size, spacing_px=result.spacing_px,
                              ))
    return asset_result(asset, context, request.output, {
        "num_corners": result.num_corners, "spacing_px": result.spacing_px,
        "method": "manually_supplied_corners",
    })


class LengthConversionRequest(CalculationRequest):
    calibration: CalibrationInput
    applicability: CalibrationApplicability
    length_px: float = Field(ge=0)
    target_unit: PhysicalUnit = "mm"


class LengthConversionResult(StrictModel):
    pixel_length: float
    physical_length: float
    unit: PhysicalUnit
    calibration_sha256: str
    source_frame: str


def convert_length(
    request: LengthConversionRequest, context: CalculationContext,
) -> LengthConversionResult:
    """Convert an explicit pixel length using a uniform physical calibration.

    Args:
        request: Typed pixel length and supplied scalar calibration with required
            applicability.
        context: Isolated operation identity, resource limits and verified shared-file
            access.

    Returns:
        Original pixel length and converted physical length in request.target_unit,
        with the source frame and calibration content identity.
    """
    asset = load_calibration(request.calibration, context)
    check_applicability(asset, request.applicability)
    if not isinstance(asset, ScalarCalibration):
        raise ServiceError("calibration.scale_required",
                           "Length conversion requires a scalar calibration; intrinsics alone "
                           "do not establish physical scale. For a plane map transform endpoints.")
    physical = (request.length_px * asset.parameters.scale_per_px
                * UNIT_MM[asset.unit] / UNIT_MM[request.target_unit])
    return LengthConversionResult(pixel_length=request.length_px, physical_length=physical,
                                  unit=request.target_unit,
                                  calibration_sha256=calibration_digest(asset),
                                  source_frame=asset.applicability.frame)


OPERATIONS = [
    Operation("/calibrations/scales", ScaleRequest, CalibrationResult, compute_scale,
              "Compute scalar calibration from supplied corners", "calibrations",
              {"asset": "calibration"}),
    Operation("/geometry/length-conversions", LengthConversionRequest, LengthConversionResult,
              convert_length, "Convert explicit pixel length using supplied scalar calibration",
              "calibrations", {"length": "physical_length"}),
]


class BoardGrid(StrictModel):
    """Number of ordered detected features, rather than number of squares."""

    columns: int = Field(ge=2, le=100)
    rows: int = Field(ge=2, le=100)


class DetectionROI(StrictModel):
    """Axis-aligned full-size rectangle in source pixel coordinates."""

    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(ge=5)
    height: int = Field(ge=5)


class CheckerboardRequest(CalculationRequest):
    image: FileReference
    grid: Optional[BoardGrid] = None
    roi: Optional[DetectionROI] = None
    quality_level: float = Field(default=0.05, gt=0, le=1)
    min_distance: float = Field(default=4, gt=0)
    max_corners: int = Field(default=2000, ge=6, le=10000)
    lattice_tolerance: float = Field(default=0.2, gt=0, lt=1)
    min_lattice_neighbors: int = Field(default=2, ge=2, le=4)


class CheckerboardResult(StrictModel):
    valid: bool
    corners: List[Point]
    ordering: Literal["detector_order"] = "detector_order"
    source: FileReference
    image_size: ImageSize
    diagnostics: dict


def detect_checkerboard(
    request: CheckerboardRequest, context: CalculationContext,
) -> CheckerboardResult:
    """Run public lattice refinement with source-coordinate ROI observations.

    Args:
        request: Typed source image, board grid, ROI and corner refinement options.
        context: Isolated operation identity, resource limits and verified shared-file
            access.

    Returns:
        Detector-order corners in source xy coordinates, verified source identity,
        actual image dimensions and corner/lattice diagnostics.
    """
    image, source = context.files.read_image(request.image)
    roi = request.roi
    if roi and (roi.x + roi.width > image.shape[1] or roi.y + roi.height > image.shape[0]):
        raise ServiceError("calibration.roi_outside", "ROI extends outside source image")
    detector = CheckerboardScaleCalibration(
        request.quality_level, request.min_distance, request.max_corners,
        request.lattice_tolerance, request.min_lattice_neighbors,
    )
    result = detector.calibrate(
        image, square_size_mm=1,
        board_grid=(request.grid.columns, request.grid.rows) if request.grid else None,
        roi=(roi.y, roi.x, roi.y + roi.height - 1, roi.x + roi.width - 1) if roi else None,
    )
    return CheckerboardResult(
        valid=result.valid, corners=[Point(x=x, y=y) for y, x in result.corners],
        source=source, image_size=ImageSize(width=image.shape[1], height=image.shape[0]),
        diagnostics={"method": result.method, "spacing_px": result.spacing_px,
                     "num_corners": result.num_corners, "reason": result.error,
                     "refinement": "subpixel_and_lattice",
                     "lattice_tolerance": request.lattice_tolerance},
    )


class ObjectPoint(StrictModel):
    """Physical board coordinate in the declared unit."""

    x: float
    y: float
    z: float = 0


class BlobOptions(StrictModel):
    """Public dark/light circle detection settings."""

    min_area: float = Field(default=5, gt=0)
    max_area: float = Field(default=10000, gt=0)
    min_circularity: float = Field(default=0.3, gt=0, le=1)
    min_convexity: float = Field(default=0.3, gt=0, le=1)
    color: Literal["dark", "light"] = "dark"

    @model_validator(mode="after")
    def check_area(self) -> BlobOptions:
        if self.max_area <= self.min_area:
            raise ValueError("max_area must be greater than min_area")
        return self


class DotGridRequest(CalculationRequest):
    image: FileReference
    grid: BoardGrid
    spacing: float = Field(gt=0)
    unit: PhysicalUnit = "mm"
    symmetric: bool = True
    refine: bool = True
    blob: BlobOptions = Field(default_factory=BlobOptions)


class DotGridResult(StrictModel):
    valid: bool
    source: FileReference
    image_size: ImageSize
    image_points: List[Point]
    object_points: List[ObjectPoint]
    ordering: Literal["row_major"] = "row_major"
    unit: PhysicalUnit
    diagnostics: dict


def board_points(grid: BoardGrid, spacing: float, symmetric: bool) -> List[ObjectPoint]:
    """Generate explicit row-major symmetric/asymmetric physical coordinates.

    Args:
        grid: Board row and column counts and symmetric/asymmetric pattern definition.
        spacing: Physical spacing between neighboring board points in the requested unit.
        symmetric: Whether rows share the same horizontal origin instead of staggered dots.

    Returns:
        Row-major physical board coordinates with the requested dot staggering.
    """
    return [ObjectPoint(x=(col if symmetric else 2 * col + row % 2) * spacing,
                        y=row * spacing) for row in range(grid.rows)
            for col in range(grid.columns)]


def detect_dot_grid(request: DotGridRequest, context: CalculationContext) -> DotGridResult:
    """Detect one image independently, preserving missing-grid domain outcomes.

    Args:
        request: Typed source image, dot-grid layout, blob filters and known physical
            spacing.
        context: Isolated operation identity, resource limits and verified shared-file
            access.

    Returns:
        Ordered dot centers, physical board correspondences and detection quality.
    """
    image, source = context.files.read_image(request.image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold, params.maxThreshold, params.thresholdStep = 1, 255, 5
    params.filterByArea = True
    params.minArea, params.maxArea = request.blob.min_area, request.blob.max_area
    params.filterByCircularity = True
    params.minCircularity = request.blob.min_circularity
    params.filterByConvexity = True
    params.minConvexity = request.blob.min_convexity
    params.filterByInertia = False
    params.filterByColor = True
    params.blobColor = 0 if request.blob.color == "dark" else 255
    flags = cv2.CALIB_CB_SYMMETRIC_GRID if request.symmetric else cv2.CALIB_CB_ASYMMETRIC_GRID
    found, centers = cv2.findCirclesGrid(
        gray, (request.grid.columns, request.grid.rows), flags=flags,
        blobDetector=cv2.SimpleBlobDetector_create(params),
    )
    if found and request.refine:
        cv2.cornerSubPix(gray, centers, (5, 5), (-1, -1),
                         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    points = [Point(x=float(p[0]), y=float(p[1])) for p in centers.reshape(-1, 2)] if found else []
    return DotGridResult(
        valid=bool(found), source=source, image_size=ImageSize(
            width=image.shape[1], height=image.shape[0]),
        image_points=points, object_points=board_points(request.grid, request.spacing,
                                                       request.symmetric),
        unit=request.unit, diagnostics={"reason": None if found else "grid_not_found",
                                       "refined": bool(found and request.refine),
                                       "symmetric": request.symmetric},
    )


OPERATIONS.extend([
    Operation("/calibration-features/checkerboard-corners", CheckerboardRequest,
              CheckerboardResult, detect_checkerboard, "Detect checkerboard lattice corners",
              "calibrations", {"corners": "points"}),
    Operation("/calibration-features/dot-grids", DotGridRequest, DotGridResult, detect_dot_grid,
              "Detect ordered dot-grid observations in one image", "calibrations",
              {"image_points": "points", "object_points": "physical_points"}),
])


class CameraObservation(StrictModel):
    """One accepted/manual ordered image-to-physical-board correspondence set."""

    view_id: str = Field(min_length=1)
    image_size: ImageSize
    image_points: List[Point]
    object_points: List[ObjectPoint]
    ordering: Literal["row_major", "explicit_correspondences"]
    source: Optional[FileReference] = None


class CameraImageView(StrictModel):
    """Upstream-owned image and stable per-view identity."""

    view_id: str = Field(min_length=1)
    image: FileReference


class CameraRequest(CalculationRequest):
    applicability: CalibrationApplicability
    unit: PhysicalUnit = "mm"
    observations: Optional[List[CameraObservation]] = None
    images: Optional[List[CameraImageView]] = None
    grid: Optional[BoardGrid] = None
    spacing: Optional[float] = Field(default=None, gt=0)
    symmetric: bool = True
    refine: bool = True
    blob: BlobOptions = Field(default_factory=BlobOptions)
    output: Optional[OutputSpec] = None

    @model_validator(mode="after")
    def check_sources(self) -> CameraRequest:
        if (self.observations is None) == (self.images is None):
            raise ValueError("Provide observations or images, exclusively")
        if self.images is not None and (self.grid is None or self.spacing is None):
            raise ValueError("Image calibration requires grid and positive physical spacing")
        views = self.observations if self.observations is not None else self.images
        if not views:
            raise ValueError("Provide at least one view")
        ids = [view.view_id for view in views]
        if len(set(ids)) != len(ids):
            raise ValueError("view_id must be unique")
        return self


def asset_result(
    asset: CalibrationAsset, context: CalculationContext,
    output: Optional[OutputSpec] = None, diagnostics: Optional[dict] = None,
) -> CalibrationResult:
    """Return a portable asset value and optionally an atomically committed JSON file.

    Args:
        asset: Validated calibration value, including its declared setup and frame.
        context: Isolated operation identity, resource limits and verified shared-file
            access.
        output: Optional upstream-owned destination for the jointly committed artifact set.
        diagnostics: Computation quality and provenance fields to include with the asset.

    Returns:
        Validated asset, source identities, quality diagnostics and optional committed
        artifacts.
    """
    result = CalibrationResult(valid=True, asset=asset, sha256=calibration_digest(asset),
                               size_bytes=len(calibration_bytes(asset)),
                               diagnostics=diagnostics or {})
    if output:
        result.artifacts, result.manifest = ArtifactWriter(
            context.files, output, context.operation_id,
        ).commit([("calibration.json", "calibration", "json", calibration_bytes(asset))])
    return result


def compute_camera(request: CameraRequest, context: CalculationContext) -> CalibrationResult:
    """Compute intrinsic/distortion parameters and expose per-view rejected observations.

    Args:
        request: Typed accepted board observations or image captures and intrinsic fitting
            options.
        context: Isolated operation identity, resource limits and verified shared-file
            access.

    Returns:
        Validated asset, source identities, quality diagnostics and optional committed
        artifacts.
    """
    rejected = []
    observations = list(request.observations or [])
    for view in request.images or []:
        detection = detect_dot_grid(DotGridRequest(
            image=view.image, grid=request.grid, spacing=request.spacing, unit=request.unit,
            symmetric=request.symmetric, refine=request.refine, blob=request.blob,
        ), context)
        if not detection.valid:
            rejected.append({"view_id": view.view_id, "reason": "grid_not_found",
                             "source": detection.source.model_dump(mode="json")})
        else:
            observations.append(CameraObservation(
                view_id=view.view_id, image_size=detection.image_size,
                image_points=detection.image_points, object_points=detection.object_points,
                ordering=detection.ordering, source=detection.source,
            ))
    accepted = []
    seen_observations = set()
    objects, pixels = [], []
    size = request.applicability.image_size
    for view in observations:
        obj = np.array([[p.x, p.y, p.z] for p in view.object_points], np.float32).reshape(-1, 3)
        img = np.array([[p.x, p.y] for p in view.image_points], np.float32).reshape(-1, 2)
        reason = None
        if view.image_size != size:
            reason = "image_size_mismatch"
        elif len(obj) != len(img) or len(obj) < 6:
            reason = "incomplete_correspondences"
        elif any(not (0 <= p.x < size.width and 0 <= p.y < size.height)
                 for p in view.image_points):
            reason = "image_point_outside"
        elif (len(np.unique(obj, axis=0)) != len(obj)
              or len(np.unique(img, axis=0)) != len(img)
              or np.linalg.matrix_rank(obj - obj.mean(axis=0), tol=1e-5) != 2
              or np.linalg.matrix_rank(img - img.mean(axis=0), tol=1e-5) != 2):
            reason = "degenerate_or_nonplanar_correspondences"
        elif not np.allclose(obj[:, 2], 0, atol=1e-6):
            reason = "board_plane_requires_z_zero"
        identity = (obj.tobytes(), img.tobytes())
        if reason is None and identity in seen_observations:
            reason = "duplicate_observation"
        if reason:
            rejected.append({"view_id": view.view_id, "reason": reason})
        else:
            seen_observations.add(identity)
            accepted.append(view)
            objects.append(obj)
            pixels.append(img)
    if len(accepted) < 3:
        return CalibrationResult(valid=False, diagnostics={
            "reason": "at_least_three_usable_views_required", "rejected_views": rejected,
            "accepted_views": [{"view_id": view.view_id} for view in accepted],
        })
    try:
        rms, matrix, distortion, rotations, translations = cv2.calibrateCamera(
            objects, pixels, (size.width, size.height), None, None,
        )
    except cv2.error as error:
        return CalibrationResult(valid=False, diagnostics={
            "reason": "intrinsic_solve_failed", "message": str(error),
            "rejected_views": rejected,
        })
    if (not np.isfinite(matrix).all() or not np.isfinite(distortion).all()
            or not np.isfinite(rms)):
        return CalibrationResult(valid=False, diagnostics={"reason": "nonfinite_solution",
                                                          "rejected_views": rejected})
    details = []
    all_errors = []
    for index, view in enumerate(accepted):
        projected, _ = cv2.projectPoints(objects[index], rotations[index], translations[index],
                                         matrix, distortion)
        errors = np.linalg.norm(pixels[index] - projected.reshape(-1, 2), axis=1)
        all_errors.extend(errors.tolist())
        details.append({"view_id": view.view_id, "residuals_px": errors.tolist(),
                        "mean_error_px": float(errors.mean()),
                        "rotation_vector": rotations[index].reshape(-1).tolist(),
                        "translation_vector": translations[index].reshape(-1).tolist(),
                        "translation_unit": request.unit,
                        "source": view.source.model_dump(mode="json") if view.source else None})
    asset = CameraCalibration(
        unit=request.unit, applicability=request.applicability,
        parameters=CameraParameters(camera_matrix=matrix.tolist(),
                                    distortion=distortion.reshape(-1).tolist(),
                                    reprojection_error_px=float(np.mean(all_errors))),
    )
    return asset_result(asset, context, request.output, {
        "accepted_views": details, "rejected_views": rejected, "rms_error_px": float(rms),
        "physical_scale_available": False,
    })


OPERATIONS.append(Operation(
    "/calibrations/cameras", CameraRequest, CalibrationResult, compute_camera,
    "Calibrate camera from images or supplied ordered observations", "calibrations",
    {"asset": "calibration"},
))


class CalibrationUtilityRequest(CalculationRequest):
    calibration: CalibrationInput


class BoardValidationObservation(CameraObservation):
    """Accepted board samples associated with one camera and paired view identity."""

    camera_id: str = Field(min_length=1)


class CalibrationValidationRequest(CalibrationUtilityRequest):
    applicability: Optional[CalibrationApplicability] = None
    observations: Optional[List[BoardValidationObservation]] = Field(default=None, min_length=1)
    unit: Optional[PhysicalUnit] = None
    max_error_px: Optional[float] = Field(default=None, ge=0)

    @model_validator(mode="after")
    def observation_unit(self) -> CalibrationValidationRequest:
        if self.observations is not None and self.unit is None:
            raise ValueError("Board observations require an explicit physical unit")
        if self.observations is None and self.max_error_px is not None:
            raise ValueError("Pixel error limits require board observations")
        return self


class CalibrationExportRequest(CalibrationUtilityRequest):
    output: OutputSpec


def inspect_calibration(
    request: CalibrationUtilityRequest, context: CalculationContext,
) -> CalibrationResult:
    """Inspect/import a supplied asset without registration or input mutation.

    Args:
        request: Typed inline or file-backed calibration to inspect or import.
        context: Isolated operation identity, resource limits and verified shared-file
            access.

    Returns:
        Validated asset, source identities, quality diagnostics and optional committed
        artifacts.
    """
    asset = load_calibration(request.calibration, context)
    return asset_result(asset, context, diagnostics={
        "physical_scale_available": isinstance(asset, ScalarCalibration),
        "kind": asset.kind, "applicability": asset.applicability.model_dump(),
    })


def validate_asset(
    request: CalibrationValidationRequest, context: CalculationContext,
) -> CalibrationResult:
    """Return actionable asset/applicability validation as a domain outcome.

    Args:
        request: Typed supplied calibration, required applicability and optional known-board
            captures.
        context: Isolated operation identity, resource limits and verified shared-file
            access.

    Returns:
        Validated asset, source identities, quality diagnostics and optional committed
        artifacts.
    """
    try:
        asset = load_calibration(request.calibration, context)
        if request.applicability:
            check_applicability(asset, request.applicability)
        if request.observations:
            from measure_api.v2.families.plane_stereo import validate_board_observations

            diagnostics = validate_board_observations(asset, request.observations,
                                                      request.unit, context)
            result = asset_result(asset, context, diagnostics=diagnostics)
            if request.max_error_px is not None:
                result.valid = diagnostics["max_error_px"] <= request.max_error_px
                result.diagnostics["requested_max_error_px"] = request.max_error_px
            return result
    except ServiceError as error:
        return CalibrationResult(valid=False, diagnostics={
            "code": error.code, "reason": error.message, "details": error.details,
        })
    return asset_result(asset, context)


def export_calibration(
    request: CalibrationExportRequest, context: CalculationContext,
) -> CalibrationResult:
    """Commit a canonical versioned asset into an upstream-owned isolated directory.

    Args:
        request: Typed supplied calibration and upstream-owned JSON output destination.
        context: Isolated operation identity, resource limits and verified shared-file
            access.

    Returns:
        Validated asset, source identities, quality diagnostics and optional committed
        artifacts.
    """
    return asset_result(load_calibration(request.calibration, context), context, request.output)


OPERATIONS.extend([
    Operation("/calibration-validations", CalibrationValidationRequest, CalibrationResult,
              validate_asset, "Validate supplied calibration and optional image applicability",
              "calibrations", {"asset": "calibration"}),
    Operation("/calibration-inspections", CalibrationUtilityRequest, CalibrationResult,
              inspect_calibration, "Inspect portable calibration without registration",
              "calibrations", {"asset": "calibration"}),
    Operation("/calibration-imports", CalibrationUtilityRequest, CalibrationResult,
              inspect_calibration, "Import portable calibration as a typed value",
              "calibrations", {"asset": "calibration"}),
    Operation("/calibration-exports", CalibrationExportRequest, CalibrationResult,
              export_calibration, "Export portable calibration with atomic manifest commit",
              "calibrations", {"asset": "calibration"}),
])


class UndistortionRequest(CalculationRequest):
    image: FileReference
    calibration: CalibrationInput
    applicability: CalibrationApplicability
    destination_frame: str = Field(default="undistorted", min_length=1)
    alpha: float = Field(default=0, ge=0, le=1)
    output: OutputSpec

    @model_validator(mode="after")
    def distinct_frame(self) -> UndistortionRequest:
        if self.destination_frame == self.applicability.frame:
            raise ValueError("Undistortion must create a distinct coordinate frame")
        return self


class UndistortionResult(StrictModel):
    metadata: ImageMetadata
    source: FileReference
    mapping: CoordinateMapping
    calibration_sha256: str
    artifacts: List[FileDescriptor]
    manifest: FileDescriptor


def undistort_image(
    request: UndistortionRequest, context: CalculationContext,
) -> UndistortionResult:
    """Correct an applicable image and commit the complete nonlinear inverse map.

    Args:
        request: Typed source image, applicable camera asset and corrected-image output
            destination.
        context: Isolated operation identity, resource limits and verified shared-file
            access.

    Returns:
        Corrected image, nonlinear inverse mapping and jointly committed artifacts.
    """
    asset = load_calibration(request.calibration, context)
    check_applicability(asset, request.applicability)
    if not isinstance(asset, CameraCalibration):
        raise ServiceError("calibration.camera_required", "Undistortion requires camera intrinsics")
    image, source = context.files.read_image(request.image)
    size = request.applicability.image_size
    if image.shape[:2] != (size.height, size.width):
        raise ServiceError("calibration.image_size_mismatch",
                           "Decoded image dimensions do not match camera calibration")
    parameters = asset.parameters
    camera = CoreCameraCalibration.from_dict({
        "grid_size": [2, 2], "circle_spacing_mm": 1,
        "K": parameters.camera_matrix, "D": parameters.distortion,
        "image_size": [size.height, size.width],
    })
    corrected = camera.undistort(image, alpha=request.alpha)
    new_matrix, valid_roi = cv2.getOptimalNewCameraMatrix(
        camera.K, camera.D, (size.width, size.height), request.alpha,
    )
    map_x, map_y = cv2.initUndistortRectifyMap(
        camera.K, camera.D, None, new_matrix, (size.width, size.height), cv2.CV_32FC1,
    )
    stream = io.BytesIO()
    np.savez_compressed(stream, source_x=map_x, source_y=map_y)
    writer = ArtifactWriter(context.files, request.output, context.operation_id)
    artifacts, manifest = writer.commit([
        ("undistorted.png", "undistorted", "png", encode_png(corrected)),
        ("mapping.npz", "mapping", "npz", stream.getvalue()),
    ])
    mapping = CoordinateMapping(
        source_frame=asset.applicability.frame, destination_frame=request.destination_frame,
        type="nonlinear", details={
            "model": "opencv_pinhole_distortion", "camera_matrix": parameters.camera_matrix,
            "distortion": parameters.distortion, "corrected_camera_matrix": new_matrix.tolist(),
            "source_to_destination": "undistortPoints(source,K,D,P=corrected_camera_matrix)",
            "inverse_map_direction": "destination_to_source",
            "inverse_map_artifact": artifacts[1].model_dump(mode="json"),
            "inverse_map_arrays": ["source_x", "source_y"],
            "valid_destination_roi": [int(value) for value in valid_roi],
            "calibration_sha256": calibration_digest(asset), "alpha": request.alpha,
        },
    )
    return UndistortionResult(
        metadata=image_metadata(corrected, request.destination_frame), source=source,
        mapping=mapping, calibration_sha256=calibration_digest(asset),
        artifacts=artifacts, manifest=manifest,
    )


OPERATIONS.append(Operation(
    "/image-undistortions", UndistortionRequest, UndistortionResult, undistort_image,
    "Correct image distortion with complete nonlinear frame mapping", "calibrations",
    {"image": "image", "mapping": "transform"},
))
