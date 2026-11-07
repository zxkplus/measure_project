"""Stateless gantry pick operations hosted by the existing V2 compute service."""

from __future__ import annotations

from typing import Annotated, List, Optional

from pydantic import Field, model_validator

from measure.pick_api import GantryPick, PickError, offset_homogeneous_matrix
from measure.robot_calibration import CalibrationError, OffsetSample
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


class GantryOffsetSample(StrictModel):
    """Placed-target and camera-centred flange poses in the same controller unit."""

    place_pose: RobotPose
    center_pose: RobotPose


class GantryOffsetRequest(CalculationRequest):
    """One or more samples of the additive camera-to-gripper XY correction."""

    samples: List[GantryOffsetSample] = Field(min_length=1)


class GantryMatrixResult(StrictModel):
    """Upstream-owned row-major 3x3 vision affine or pure-translation offset."""

    matrix: Matrix3x3


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
        calibration = GantryPick().calibrate_vision(
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
        request: Paired flange poses; trailing Z values are ignored.
        context: V2 operation identity and resource limits.

    Returns:
        A pure-translation 3x3 matrix containing the mean XY offset.
    """
    samples = [
        OffsetSample(place_pose=tuple(sample.place_pose[:2]),
                     center_pose=tuple(sample.center_pose[:2]))
        for sample in request.samples
    ]
    try:
        offset = GantryPick().calibrate_offset(samples)
    except CalibrationError as error:
        raise ServiceError("pick.offset_invalid", str(error)) from error
    if not offset.valid:
        raise ServiceError(
            "pick.offset_invalid", "Gripper offset calibration failed its quality gate",
            details={"warnings": offset.warnings},
        )
    return GantryMatrixResult(matrix=offset_homogeneous_matrix(offset.offset_xy).tolist())


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
        location = GantryPick().locate(
            image, request.robot_pose, request.vision_matrix, request.offset_matrix,
        )
    except PickError as error:
        raise ServiceError(
            "pick." + error.code, error.message, error.status, details=error.details,
        ) from error
    if not location.valid:
        raise ServiceError(
            "pick.no_target", "Target not found or insufficient circle confidence",
            details={"reason": location.reason, "confidence": location.confidence},
        )
    return GantryLocateResult(
        camera_center_xy=list(location.camera_center_xy),
        gripper_xy=None if location.gripper_xy is None else list(location.gripper_xy),
    )


OPERATIONS = [
    Operation("/calibrate/vision", GantryVisionRequest, GantryMatrixResult, calibrate_vision,
              "Compute gantry vision calibration from shared images", "gantry_pick",
              {"matrix": "matrix"}),
    Operation("/calibrate/offset", GantryOffsetRequest, GantryMatrixResult, calibrate_offset,
              "Compute gantry camera-to-gripper offset", "gantry_pick", {"matrix": "matrix"}),
    Operation("/locate", GantryLocateRequest, GantryLocateResult, locate,
              "Locate a circular target for camera or gripper centring", "gantry_pick",
              {"camera_center_xy": "robot_xy", "gripper_xy": "robot_xy"}),
]
