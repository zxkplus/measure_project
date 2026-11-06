"""Stateless gantry pick math: two 3x3 matrices in, coordinates out.

The HTTP service is deliberately thin.  Calibration returns one homogeneous
3x3 matrix per step; the calling software stores both matrices and sends them
back to :meth:`GantryPick.locate`.  Nothing is cached on the server.

Coordinate conventions:

* Image points are ``(u, v) = (col, row)`` in pixels.
* Robot points are ``(X, Y)`` in the controller unit (normally mm); Z is
  ignored by design.
* The vision matrix is ``[[A, b], [0, 1]]``.  ``locate`` uses only ``A`` and
  the image centre; ``b`` is target-position specific and is kept only for
  audit/storage.
* The offset matrix is a pure translation ``[[I, t], [0, 1]]``.
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
    calibrate_image_to_robot,
)


class PickError(RuntimeError):
    """A stable pick API failure with an HTTP-friendly code.

    Attributes:
        code: Machine-readable error code, for example ``"bad_matrix"``.
        message: Human-readable explanation.
        status: Suggested HTTP status code.
        details: Optional JSON-serialisable context.
    """

    def __init__(
        self,
        code: str,
        message: str,
        status: int = 400,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.code = str(code)
        self.message = str(message)
        self.status = int(status)
        self.details: Dict[str, Any] = dict(details or {})


@dataclass(frozen=True)
class TargetLocation:
    """One target located from a vision matrix and a current flange pose.

    Attributes:
        valid: ``True`` when the target was detected.
        reason: ``"ok"`` or the detection failure reason.
        robot_xy: Flange XY at capture time.
        image_uv: Detected target centre, or ``None`` on failure.
        radius_px: Detected radius in pixels, when available.
        confidence: Circle-locator confidence in ``[0, 1]``.
        camera_center_xy: Flange XY that puts the target at the image centre.
        gripper_xy: Flange XY that puts the gripper over the target; ``None``
            when no offset matrix was supplied.
        detection: Raw detection for annotation and replay; excluded from
            :meth:`as_dict`.
    """

    valid: bool
    reason: str
    robot_xy: Tuple[float, float]
    image_uv: Optional[Tuple[float, float]] = None
    radius_px: Optional[float] = None
    confidence: float = 0.0
    camera_center_xy: Optional[Tuple[float, float]] = None
    gripper_xy: Optional[Tuple[float, float]] = None
    detection: Optional[CircleDetection] = field(
        default=None, repr=False, compare=False
    )

    @property
    def grasp_ready(self) -> bool:
        """Whether a gripper coordinate is available."""
        return self.gripper_xy is not None

    def as_dict(self) -> Dict[str, Any]:
        """Return the JSON-safe debug result (never includes Z)."""

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


def _image_size(image: np.ndarray) -> Tuple[int, int]:
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


def _finite_matrix(value: Sequence[Sequence[float]], name: str) -> np.ndarray:
    """Parse a finite 3x3 matrix."""
    try:
        matrix = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise PickError("bad_matrix", f"{name} must contain numbers", 400) from exc
    if matrix.shape != (3, 3):
        raise PickError("bad_matrix", f"{name} must be 3x3", 400)
    if not np.all(np.isfinite(matrix)):
        raise PickError("bad_matrix", f"{name} contains non-finite values", 400)
    return matrix


def _vision_matrix(value: Sequence[Sequence[float]]) -> np.ndarray:
    """Validate the 3x3 vision matrix and return it as float64."""
    matrix = _finite_matrix(value, "vision_matrix")
    if not np.allclose(matrix[2], (0.0, 0.0, 1.0), atol=1e-6):
        raise PickError(
            "bad_matrix",
            "vision_matrix last row must be [0, 0, 1]",
            400,
        )
    if abs(float(np.linalg.det(matrix[:2, :2]))) < 1e-12:
        raise PickError(
            "bad_matrix",
            "vision_matrix upper-left 2x2 is singular",
            400,
        )
    return matrix


def _offset_matrix(value: Sequence[Sequence[float]]) -> Tuple[float, float]:
    """Validate a pure-translation offset matrix and return its XY."""
    matrix = _finite_matrix(value, "offset_matrix")
    if not np.allclose(matrix[2], (0.0, 0.0, 1.0), atol=1e-6):
        raise PickError(
            "bad_matrix",
            "offset_matrix last row must be [0, 0, 1]",
            400,
        )
    if not np.allclose(matrix[:2, :2], np.eye(2), atol=1e-6):
        raise PickError(
            "bad_matrix",
            "offset_matrix must be a pure translation",
            400,
        )
    return (float(matrix[0, 2]), float(matrix[1, 2]))


def offset_homogeneous_matrix(offset_xy: Sequence[float]) -> np.ndarray:
    """Return the 3x3 homogeneous matrix for a pure XY translation."""
    x, y = _as_xy(offset_xy, "offset_xy")
    return np.array(
        [[1.0, 0.0, x], [0.0, 1.0, y], [0.0, 0.0, 1.0]],
        dtype=float,
    )


class GantryPick:
    """Stateless gantry pick operations.

    Calibration methods return the existing rich engine objects
    (:class:`ImageRobotCalibration` / :class:`CameraGripperOffset`); the HTTP
    layer reduces them to one 3x3 matrix each.  ``locate`` only needs the two
    matrices, so it does not depend on any server-side recipe.
    """

    def __init__(self, locator: Optional[CircleLocator] = None) -> None:
        # The real gantry targets can be large (bottle mouths are >1000 px at
        # 5 MP), so the pick default is wider than CircleLocator's own default.
        self.locator = locator or CircleLocator(radius_range=(6.0, 3000.0))

    def calibrate_vision(
        self,
        images: Sequence[np.ndarray],
        robot_points: Sequence[Sequence[float]],
        *,
        principal_point: Optional[Sequence[float]] = None,
        image_size: Optional[Sequence[int]] = None,
        calib_z: Optional[float] = None,
    ) -> ImageRobotCalibration:
        """Fit the vision affine; the caller can read ``result.matrix``."""
        return calibrate_image_to_robot(
            images,
            robot_points,
            locator=self.locator,
            principal_point=principal_point,
            image_size=image_size,
            calib_z=calib_z,
        )

    def calibrate_offset(
        self,
        samples: Sequence[OffsetSample],
        *,
        center_tolerance_px: float = 5.0,
    ) -> CameraGripperOffset:
        """Fit the gripper offset; the caller can read ``result.offset_xy``."""
        return CameraGripperOffset.from_samples(
            samples, center_tolerance_px=center_tolerance_px
        )

    def locate(
        self,
        image: np.ndarray,
        robot_xy: Sequence[float],
        vision_matrix: Sequence[Sequence[float]],
        offset_matrix: Optional[Sequence[Sequence[float]]] = None,
    ) -> TargetLocation:
        """Locate the target from one image, one pose and two matrices.

        Args:
            image: Grayscale or BGR frame.
            robot_xy: Flange XY at capture time; trailing Z is ignored.
            vision_matrix: 3x3 homogeneous affine from ``/calibrate/vision``.
            offset_matrix: Optional 3x3 pure-translation matrix from
                ``/calibrate/offset``.  When omitted, ``gripper_xy`` is None.

        Returns:
            A :class:`TargetLocation`; check ``valid``.

        Raises:
            PickError: On a bad pose, image or matrix.
        """
        pose = _as_xy(robot_xy)
        matrix = _vision_matrix(vision_matrix)
        offset = None if offset_matrix is None else _offset_matrix(offset_matrix)
        width, height = _image_size(image)

        detection = self.locator.find(image)
        if not detection.valid:
            return TargetLocation(
                valid=False,
                reason=detection.reason or "no_target",
                robot_xy=pose,
                radius_px=_finite_or_none(detection.radius),
                confidence=float(detection.confidence),
                detection=detection,
            )

        centre = np.array(
            ((width - 1) * 0.5, (height - 1) * 0.5), dtype=float
        )
        point = np.asarray(detection.center, dtype=float)
        delta = matrix[:2, :2] @ (centre - point)
        camera = (float(pose[0] + delta[0]), float(pose[1] + delta[1]))
        gripper: Optional[Tuple[float, float]] = None
        if offset is not None:
            gripper = (
                float(camera[0] + offset[0]),
                float(camera[1] + offset[1]),
            )

        return TargetLocation(
            valid=True,
            reason="ok",
            robot_xy=pose,
            image_uv=(float(point[0]), float(point[1])),
            radius_px=_finite_or_none(detection.radius),
            confidence=float(detection.confidence),
            camera_center_xy=camera,
            gripper_xy=gripper,
            detection=detection,
        )


def _finite_or_none(value: float) -> Optional[float]:
    """Return a finite float or ``None`` for NaNs and infinities."""
    number = float(value)
    return number if np.isfinite(number) else None


__all__ = [
    "GantryPick",
    "PickError",
    "TargetLocation",
    "offset_homogeneous_matrix",
]
