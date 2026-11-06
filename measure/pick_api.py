"""Gantry pick API: vision calibration, gripper offset and target location.

This module is the pure-computation core behind the HTTP pick service.  It
never imports a robot SDK; callers supply images and flange XY coordinates and
receive XY coordinates back.  Z is intentionally absent: the target platform is
a three-axis gantry whose Z axis is decoupled.

Coordinate conventions:

* Image points are ``(u, v) = (col, row)`` in pixels.
* Robot points are ``(X, Y)`` in the controller's native unit (normally mm).
* ``ImageRobotCalibration.robot_per_pixel`` maps image displacement to robot
  displacement; the full-affine intercept is diagnostic only.

The three public entry points are :meth:`GantryPick.calibrate_vision`,
:meth:`GantryPick.calibrate_offset` and :meth:`GantryPick.locate`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from measure.circle_locator import CircleDetection, CircleLocator
from measure.robot_calibration import (
    CameraGripperOffset,
    ImageRobotCalibration,
    OffsetSample,
    RobotPickRecipe,
    calibrate_image_to_robot,
)


class PickError(RuntimeError):
    """A stable pick API failure with an HTTP-friendly code.

    Attributes:
        code: Machine-readable error code, for example ``"vision_not_calibrated"``.
        message: Human-readable explanation.
        status: Suggested HTTP status code.
        details: Optional JSON-serialisable context merged into the response.
    """

    def __init__(
        self,
        code: str,
        message: str,
        status: int = 422,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.code = str(code)
        self.message = str(message)
        self.status = int(status)
        self.details: Dict[str, Any] = dict(details or {})


@dataclass(frozen=True)
class TargetLocation:
    """One target located relative to a known flange pose.

    Attributes:
        valid: ``True`` when the target was detected and the vision
            calibration is usable.  ``gripper_xy`` can still be ``None`` when
            the gripper offset has not been calibrated.
        reason: ``"ok"`` or the failure reason.
        robot_xy: Flange XY at capture time.
        image_uv: Detected target centre, or ``None`` when detection failed.
        radius_px: Detected radius in pixels, when available.
        confidence: Circle-locator confidence in ``[0, 1]``.
        camera_center_xy: Flange XY that puts the target at the image
            principal point, or ``None`` when the result is invalid.
        gripper_xy: Flange XY that puts the gripper over the target, or
            ``None`` when the result is invalid or the offset is missing.
        warnings: Non-fatal runtime warnings.
        detection: Raw :class:`CircleDetection` for annotation and replay;
            excluded from :meth:`as_dict`.
    """

    valid: bool
    reason: str
    robot_xy: Tuple[float, float]
    image_uv: Optional[Tuple[float, float]] = None
    radius_px: Optional[float] = None
    confidence: float = 0.0
    camera_center_xy: Optional[Tuple[float, float]] = None
    gripper_xy: Optional[Tuple[float, float]] = None
    warnings: Tuple[str, ...] = ()
    detection: Optional[CircleDetection] = field(
        default=None, repr=False, compare=False
    )

    @property
    def grasp_ready(self) -> bool:
        """Whether a gripper coordinate is available."""
        return self.gripper_xy is not None

    def as_dict(self) -> Dict[str, Any]:
        """Return the JSON-safe public result (never includes Z)."""

        def pair(value: Optional[Tuple[float, float]]) -> Optional[List[float]]:
            if value is None:
                return None
            return [float(value[0]), float(value[1])]

        return {
            "valid": bool(self.valid),
            "reason": self.reason,
            "robot_xy": pair(self.robot_xy),
            "image_uv": pair(self.image_uv),
            "radius_px": None if self.radius_px is None else float(self.radius_px),
            "confidence": float(self.confidence),
            "camera_center_xy": pair(self.camera_center_xy),
            "gripper_xy": pair(self.gripper_xy),
            "warnings": list(self.warnings),
        }


def _as_xy(value: Sequence[float], name: str = "robot_xy") -> Tuple[float, float]:
    """Parse at least two finite numbers, ignoring any trailing Z."""
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise PickError("bad_request", f"{name} must contain numbers", 400) from exc
    if array.size < 2:
        raise PickError("bad_request", f"{name} needs at least two values", 400)
    if not np.all(np.isfinite(array[:2])):
        raise PickError("bad_request", f"{name} contains non-finite values", 400)
    return (float(array[0]), float(array[1]))


def _image_shape(image: np.ndarray) -> Tuple[int, int]:
    """Validate an image array and return ``(width, height)``."""
    if not isinstance(image, np.ndarray):
        raise PickError("bad_request", "image must be a numpy array", 400)
    if image.ndim == 2:
        height, width = image.shape[:2]
    elif image.ndim == 3 and image.shape[2] == 3:
        height, width = image.shape[:2]
    else:
        raise PickError(
            "bad_request",
            "image must be 2D grayscale or 3-channel BGR",
            400,
        )
    if width <= 1 or height <= 1:
        raise PickError("bad_request", "image is too small", 400)
    return (int(width), int(height))


class GantryPick:
    """Camera-on-gantry calibration and target location.

    Typical use::

        pick = GantryPick()
        pick.calibrate_vision(images, robot_points)
        pick.calibrate_offset([OffsetSample((118, 89), (110, 85))])
        result = pick.locate(image, (114, 82))
        result.camera_center_xy
        result.gripper_xy

    Attributes:
        image_robot: Fitted image-to-robot calibration, or ``None``.
        gripper_offset: Fitted camera-to-gripper offset, or ``None``.
        locator: Circle locator used for every frame.
        metadata: Free-form metadata copied into the saved recipe.
    """

    def __init__(
        self,
        locator: Optional[CircleLocator] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.image_robot: Optional[ImageRobotCalibration] = None
        self.gripper_offset: Optional[CameraGripperOffset] = None
        self.locator: CircleLocator = locator or CircleLocator()
        self.metadata: Dict[str, Any] = dict(metadata or {})

    # -- state -----------------------------------------------------------

    @property
    def ready(self) -> bool:
        """Whether both calibrations are present and valid."""
        return bool(
            self.image_robot is not None
            and self.image_robot.valid
            and self.gripper_offset is not None
            and self.gripper_offset.valid
        )

    def set_vision(self, calibration: ImageRobotCalibration) -> None:
        """Install a fitted vision calibration (used by the HTTP service)."""
        if not isinstance(calibration, ImageRobotCalibration):
            raise TypeError("calibration must be an ImageRobotCalibration")
        self.image_robot = calibration

    def set_offset(self, offset: CameraGripperOffset) -> None:
        """Install a fitted gripper offset (used by the HTTP service)."""
        if not isinstance(offset, CameraGripperOffset):
            raise TypeError("offset must be a CameraGripperOffset")
        self.gripper_offset = offset

    # -- calibration -----------------------------------------------------

    def calibrate_vision(
        self,
        images: Sequence[np.ndarray],
        robot_points: Sequence[Sequence[float]],
        *,
        principal_point: Optional[Sequence[float]] = None,
        image_size: Optional[Sequence[int]] = None,
        calib_z: Optional[float] = None,
    ) -> ImageRobotCalibration:
        """Fit the image-to-robot affine from images and matching poses.

        Args:
            images: Four or more same-size grayscale/BGR images.  Each frame
                must contain the same circular target.
            robot_points: Matching flange poses ``(X, Y)`` or ``(X, Y, Z)``;
                only XY is fitted.
            principal_point: Camera principal point in pixels; defaults to
                the image centre when ``image_size`` is available.
            image_size: Optional expected ``(width, height)``.
            calib_z: Optional mean Z for diagnostics.

        Returns:
            The fitted calibration.  Check ``valid`` and ``warnings``.

        Raises:
            CalibrationError: On invalid input or a missing target.
        """
        calibration = calibrate_image_to_robot(
            images,
            robot_points,
            locator=self.locator,
            principal_point=principal_point,
            image_size=image_size,
            calib_z=calib_z,
            metadata=self.metadata or None,
        )
        self.image_robot = calibration
        return calibration

    def calibrate_offset(
        self,
        samples: Sequence[OffsetSample],
        *,
        center_tolerance_px: float = 5.0,
    ) -> CameraGripperOffset:
        """Aggregate release/centred pose pairs into a gripper offset.

        Args:
            samples: One or more :class:`OffsetSample` values.
            center_tolerance_px: Maximum accepted centring residual.

        Returns:
            The fitted offset.  Check ``valid`` and ``warnings``.

        Raises:
            CalibrationError: When *samples* is empty or contains bad values.
        """
        offset = CameraGripperOffset.from_samples(
            samples, center_tolerance_px=center_tolerance_px
        )
        self.gripper_offset = offset
        return offset

    # -- location --------------------------------------------------------

    def locate(
        self,
        image: np.ndarray,
        robot_xy: Sequence[float],
        *,
        strict: bool = False,
    ) -> TargetLocation:
        """Locate the target and return camera-centred and gripper XY.

        Args:
            image: Grayscale or BGR frame.
            robot_xy: Flange XY at capture time.  A trailing Z is ignored.
            strict: When ``True``, raise :class:`PickError` instead of
                returning an invalid result for detection failures.

        Returns:
            A :class:`TargetLocation`.  ``valid`` is false when the target
            was not detected or the image size does not match.  ``gripper_xy``
            is ``None`` when no valid offset is installed.

        Raises:
            PickError: On a bad pose/image, a missing calibration, or (with
                ``strict=True``) a detection failure.
        """
        pose = _as_xy(robot_xy)
        if self.image_robot is None:
            raise PickError(
                "vision_not_calibrated",
                "run /calibrate/vision first",
                409,
            )
        if not self.image_robot.valid:
            raise PickError(
                "vision_invalid",
                "the active vision calibration is not valid",
                409,
                details={"warnings": list(self.image_robot.warnings)},
            )

        size = _image_shape(image)
        if (
            self.image_robot.image_size is not None
            and size != tuple(self.image_robot.image_size)
        ):
            message = (
                f"image size {size} does not match calibration "
                f"{tuple(self.image_robot.image_size)}"
            )
            if strict:
                raise PickError("image_size_mismatch", message, 422)
            return TargetLocation(
                valid=False,
                reason="image_size_mismatch",
                robot_xy=pose,
            )

        detection = self.locator.find(image)
        if not detection.valid:
            if strict:
                raise PickError(
                    detection.reason or "no_target",
                    f"target not found ({detection.reason or 'no_target'})",
                    422,
                )
            return TargetLocation(
                valid=False,
                reason=detection.reason or "no_target",
                robot_xy=pose,
                image_uv=None,
                radius_px=_finite_or_none(detection.radius),
                confidence=float(detection.confidence),
                detection=detection,
            )

        centre = np.asarray(self.image_robot.principal_point, dtype=float)
        point = np.asarray(detection.center, dtype=float)
        delta = self.image_robot.robot_per_pixel @ (centre - point)
        camera_center = (float(pose[0] + delta[0]), float(pose[1] + delta[1]))

        gripper: Optional[Tuple[float, float]] = None
        warnings: List[str] = []
        if self.gripper_offset is None:
            warnings.append("offset_not_calibrated")
        elif not self.gripper_offset.valid:
            warnings.append("offset_invalid")
        else:
            gripper = (
                float(camera_center[0] + self.gripper_offset.offset_xy[0]),
                float(camera_center[1] + self.gripper_offset.offset_xy[1]),
            )

        return TargetLocation(
            valid=True,
            reason="ok",
            robot_xy=pose,
            image_uv=(float(point[0]), float(point[1])),
            radius_px=_finite_or_none(detection.radius),
            confidence=float(detection.confidence),
            camera_center_xy=camera_center,
            gripper_xy=gripper,
            warnings=tuple(warnings),
            detection=detection,
        )

    # -- recipe ----------------------------------------------------------

    def to_recipe(self) -> RobotPickRecipe:
        """Build the persistable recipe from the current state.

        Raises:
            PickError: When no vision calibration is installed.
        """
        if self.image_robot is None:
            raise PickError(
                "vision_not_calibrated",
                "cannot build a recipe without a vision calibration",
                409,
            )
        return RobotPickRecipe(
            self.image_robot,
            self.gripper_offset,
            self.locator,
            metadata=self.metadata,
        )

    @classmethod
    def from_recipe(cls, recipe: RobotPickRecipe) -> "GantryPick":
        """Restore a picker from a saved recipe."""
        pick = cls(locator=recipe.locator, metadata=recipe.metadata)
        pick.image_robot = recipe.image_robot
        pick.gripper_offset = recipe.gripper_offset
        return pick

    def save_recipe(self, filepath: str) -> None:
        """Write the current recipe as JSON."""
        self.to_recipe().save(filepath)

    @classmethod
    def load_recipe(cls, filepath: str) -> "GantryPick":
        """Read a recipe written by :meth:`save_recipe`."""
        return cls.from_recipe(RobotPickRecipe.load(filepath))

    def summary(self) -> Dict[str, Any]:
        """Return a compact diagnostic summary."""
        return {
            "ready": self.ready,
            "vision": None
            if self.image_robot is None
            else {
                "valid": bool(self.image_robot.valid),
                "rms_px": float(self.image_robot.rms_px),
                "warnings": list(self.image_robot.warnings),
            },
            "offset": None
            if self.gripper_offset is None
            else {
                "valid": bool(self.gripper_offset.valid),
                "offset_xy": [float(v) for v in self.gripper_offset.offset_xy],
                "std_xy": [float(v) for v in self.gripper_offset.std_xy],
                "warnings": list(self.gripper_offset.warnings),
            },
        }


def _finite_or_none(value: float) -> Optional[float]:
    """Return a finite float or ``None`` for NaNs and infinities."""
    number = float(value)
    return number if np.isfinite(number) else None


__all__ = [
    "GantryPick",
    "PickError",
    "TargetLocation",
]
