"""Stateless gantry pick operations hosted by the existing V2 compute service."""

from __future__ import annotations

import math
from typing import Annotated, List, Literal, Optional

from pydantic import Field, model_validator

from measure.pick_api import GantryPick, PickError
from measure.robot_calibration import CalibrationError, OffsetSample, fit_image_to_robot
from measure_api.v2.errors import ServiceError
from measure_api.v2.models import CalculationRequest, FileReference, StrictModel
from measure_api.v2.registry import CalculationContext, Operation


RobotPose = Annotated[List[float], Field(min_length=2, max_length=3)]
XY = Annotated[List[float], Field(min_length=2, max_length=2)]
MatrixRow = Annotated[List[float], Field(min_length=3, max_length=3)]
Matrix3x3 = Annotated[List[MatrixRow], Field(min_length=3, max_length=3)]


class GantryVisionRequest(CalculationRequest):
    """Same-size shared images paired with flange XY poses; optional Z is ignored."""

    images: List[FileReference] = Field(min_length=4)
    robot_points: List[RobotPose] = Field(min_length=4)

    @model_validator(mode="after")
    def check_counts(self) -> GantryVisionRequest:
        """Require one flange pose per image.

        Returns:
            This request after validating correspondence counts.
        """
        if len(self.images) != len(self.robot_points):
            raise ValueError("images and robot_points must have the same length")
        return self


class GantryVisionPointsRequest(CalculationRequest):
    """Original-image pixel centres paired with actual flange XY, without image detection."""

    image_points: List[XY] = Field(min_length=4)
    robot_points: List[XY] = Field(min_length=4)
    image_size: Annotated[List[int], Field(min_length=2, max_length=2)]
    calib_z: float

    @model_validator(mode="after")
    def check_points(self) -> GantryVisionPointsRequest:
        """Validate finite, in-frame correspondences and capture height."""
        if len(self.image_points) != len(self.robot_points):
            raise ValueError("image_points and robot_points must have the same length")
        if min(self.image_size) <= 1:
            raise ValueError("image_size must exceed one pixel in each dimension")
        values = [
            value
            for points in (self.image_points, self.robot_points)
            for point in points
            for value in point
        ] + [self.calib_z]
        if not all(math.isfinite(v) for v in values):
            raise ValueError("coordinates and calib_z must be finite")
        width, height = self.image_size
        if any(
            not (0 <= u <= width - 1 and 0 <= v <= height - 1)
            for u, v in self.image_points
        ):
            raise ValueError("image_points must lie in the original image")
        return self


class GantryVisionPointResidual(StrictModel):
    """Residual of one correspondence; robot distances use controller units."""

    residual_px: float
    residual_mm: float


class GantryVisionPointsResult(StrictModel):
    """Candidate affine and quality diagnostics; upstream explicitly applies it."""

    matrix: Matrix3x3
    valid: bool
    warnings: List[str]
    residuals: List[GantryVisionPointResidual]
    rms_px: float
    max_px: float
    rms_mm: float
    max_mm: float


def calibrate_vision_points(
    request: GantryVisionPointsRequest, context: CalculationContext,
) -> GantryVisionPointsResult:
    """Fit the shared affine solver using the supplied centres without redetection."""
    try:
        calibration = fit_image_to_robot(
            request.image_points, request.robot_points, image_size=request.image_size,
            calib_z=request.calib_z, robust=False,
        )
    except (CalibrationError, ValueError) as error:
        raise ServiceError("pick.calibration_failed", str(error)) from error
    return GantryVisionPointsResult(
        matrix=calibration.matrix.tolist(), valid=calibration.valid,
        warnings=calibration.warnings,
        residuals=[GantryVisionPointResidual(
            residual_px=sample.residual_px, residual_mm=sample.residual_mm,
        ) for sample in calibration.samples],
        rms_px=calibration.rms_px, max_px=calibration.max_px,
        rms_mm=calibration.rms_mm, max_mm=calibration.max_mm,
    )


class GantryOffsetSample(StrictModel):
    """Placed-target and camera-centred flange poses in the same controller unit."""

    place_pose: RobotPose
    center_pose: RobotPose


class GantryOffsetRequest(CalculationRequest):
    """Samples of the camera-to-gripper correction, fitted as selected."""

    samples: List[GantryOffsetSample] = Field(min_length=1)
    model: Literal["auto", "translation", "affine"] = "auto"


class GantryMatrixResult(StrictModel):
    """Upstream-owned row-major 3x3 vision affine or offset transform."""

    matrix: Matrix3x3


class GantryOffsetResult(GantryMatrixResult):
    """Offset transform plus the diagnostics that selected its model."""

    model: Literal["auto", "translation", "affine"]
    effective_model: Literal["translation", "affine"]
    rms_mm: float
    max_mm: float
    condition: float
    rotation_deg: float
    scale: float
    shear: float


class GantryLocateRequest(CalculationRequest):
    """Locate a circle using explicit matrices and the flange pose at capture time."""

    image: FileReference
    robot_pose: RobotPose
    vision_matrix: Matrix3x3
    offset_matrix: Optional[Matrix3x3] = None


class GantryLocateResult(StrictModel):
    """Flange XY poses that centre the camera and optionally the gripper on the target."""

    camera_center_xy: XY
    gripper_xy: Optional[XY] = None
    gripper_delta_xy: Optional[XY] = None


def _no_target_details(location) -> dict:
    """Describe a failed location; confidence moved onto the detection record."""
    details = {"reason": location.reason}
    detection = getattr(location, "detection", None)
    if detection is not None:
        details["confidence"] = detection.confidence
    return details


def calibrate_vision(
    request: GantryVisionRequest, context: CalculationContext,
) -> GantryMatrixResult:
    """Fit the existing vision affine from verified shared-file images.

    Args:
        request: Four or more image references and their matching flange poses.
        context: V2 operation identity, resource limits and verified shared-file access.

    Returns:
        The 3x3 vision matrix for upstream storage and subsequent location calls.
    """
    images = [context.files.read_image(reference)[0] for reference in request.images]
    try:
        # The core pick engine now takes detected points (caller detects); the
        # offline helper keeps server-side circle detection for this surface.
        calibration = GantryPick().calibrate_vision_from_images(
            images, [point[:2] for point in request.robot_points],
        )
    except CalibrationError as error:
        raise ServiceError("pick.calibration_failed", str(error)) from error
    if not calibration.valid:
        raise ServiceError(
            "pick.calibration_invalid", "Vision calibration failed its quality gate",
            details={"rms_px": calibration.rms_px, "warnings": calibration.warnings},
        )
    return GantryMatrixResult(matrix=calibration.matrix.tolist())


def calibrate_offset(
    request: GantryOffsetRequest, context: CalculationContext,
) -> GantryMatrixResult:
    """Fit the existing gripper correction without server-side recipe state.

    Args:
        request: Paired flange poses, the requested model and Z values that
            are ignored.
        context: V2 operation identity and resource limits.

    Returns:
        The selected 3x3 transform plus the diagnostics that selected it.
    """
    samples = [
        OffsetSample(place_pose=tuple(sample.place_pose[:2]),
                     center_pose=tuple(sample.center_pose[:2]))
        for sample in request.samples
    ]
    try:
        offset = GantryPick().calibrate_offset(samples, model=request.model)
    except CalibrationError as error:
        code = error.code or "offset_invalid"
        raise ServiceError("pick." + code, str(error)) from error
    if not offset.valid:
        raise ServiceError(
            "pick.offset_invalid", "Gripper offset calibration failed its quality gate",
            details={"warnings": offset.warnings},
        )
    return GantryOffsetResult(
        matrix=offset.matrix.tolist(),
        model=offset.requested_model,
        effective_model=offset.effective_model,
        rms_mm=float(offset.rms_mm),
        max_mm=float(offset.max_mm),
        condition=float(offset.condition),
        rotation_deg=float(offset.rotation_deg),
        scale=float(offset.scale),
        shear=float(offset.shear),
    )


def locate(request: GantryLocateRequest, context: CalculationContext) -> GantryLocateResult:
    """Calculate camera-centred and gripper flange XY using the shared pick engine.

    Args:
        request: Verified source image, current flange pose and explicit matrices.
        context: V2 operation identity, resource limits and verified shared-file access.

    Returns:
        Camera-centred XY and optional gripper XY in the controller's coordinate unit.
    """
    image, _ = context.files.read_image(request.image)
    try:
        location = GantryPick().locate_image(
            image, request.robot_pose[:2], request.vision_matrix, request.offset_matrix,
        )
    except PickError as error:
        raise ServiceError(
            "pick." + error.code, error.message, error.status, details=error.details,
        ) from error
    if not location.valid:
        raise ServiceError(
            "pick.no_target", "Target not found or insufficient circle confidence",
            details=_no_target_details(location),
        )
    return GantryLocateResult(
        camera_center_xy=list(location.camera_center_xy),
        gripper_xy=None if location.gripper_xy is None else list(location.gripper_xy),
        gripper_delta_xy=(
            None if location.gripper_delta_xy is None
            else list(location.gripper_delta_xy)
        ),
    )


OPERATIONS = [
    Operation("/calibrate/vision-points", GantryVisionPointsRequest, GantryVisionPointsResult,
              calibrate_vision_points, "Fit vision affine from taught pixel centres",
              "gantry_pick", {"matrix": "matrix"}),
    Operation("/calibrate/vision", GantryVisionRequest, GantryMatrixResult, calibrate_vision,
              "Compute gantry vision calibration from shared images", "gantry_pick",
              {"matrix": "matrix"}),
    Operation("/calibrate/offset", GantryOffsetRequest, GantryOffsetResult, calibrate_offset,
              "Compute gantry camera-to-gripper offset", "gantry_pick",
              {"matrix": "matrix"}),
    Operation("/locate", GantryLocateRequest, GantryLocateResult, locate,
              "Locate a circular target for camera or gripper centring", "gantry_pick",
              {"camera_center_xy": "robot_xy", "gripper_xy": "robot_xy"}),
]
