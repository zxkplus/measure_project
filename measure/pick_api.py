"""Stateless gantry pick math: coordinates in, coordinates out.

The HTTP service is deliberately thin.  **Detection is not part of the
business path**: the caller locates the target with its own tool (the v1
measurement API, the v2 ``measurements.circles`` operation, or a recipe on the
equipment) and passes the sub-pixel centre in.  This module therefore works on
points only; the image-based helpers exist for offline replay and tests.

Coordinate conventions:

* Image points are ``(u, v) = (col, row)`` in pixels.
* Robot points are ``(X, Y)`` in the controller unit (normally mm); Z is
  ignored by design.
* The vision matrix is ``[[A, b], [0, 1]]``.  ``locate`` uses only ``A`` and
  the image centre; ``b`` is target-position specific and is kept only for
  audit/storage.
* The offset matrix is a pure translation ``[[I, t], [0, 1]]``.

Typical use::

    from measure.pick_api import GantryPick

    pick = GantryPick()
    calibration = pick.calibrate_vision(
        image_points=[[u, v], ...],
        robot_points=[[x, y], ...],
        image_size=(5472, 3648),
    )
    location = pick.locate(
        image_uv=(u, v),
        robot_xy=(x, y),
        vision_matrix=calibration.matrix,
        offset_matrix=offset.matrix,
        image_size=(5472, 3648),
    )
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
    fit_image_to_robot,
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
    """One target located from an image point, a pose and two matrices.

    Attributes:
        valid: ``True`` when the target position was usable.
        reason: ``"ok"`` or a detection failure reason (image helpers only).
        robot_xy: Flange XY at capture time.
        image_uv: Target centre in pixels, or ``None`` when unknown.
        camera_center_xy: Flange XY that puts the target at the image centre.
        gripper_xy: Flange XY that puts the gripper over the target; ``None``
            when no offset matrix was supplied.
        delta_xy: ``A @ (centre - uv)``, the relative move the controller
            needs.  Always ``(0, 0)`` when ``valid`` is ``False``.
        radius_px: Target radius echoed from the caller, if it supplied one.
        detection: Raw detection for annotation and replay of image-based
            runs; excluded from :meth:`as_dict`.
    """

    valid: bool
    reason: str
    robot_xy: Tuple[float, float]
    image_uv: Optional[Tuple[float, float]] = None
    camera_center_xy: Optional[Tuple[float, float]] = None
    gripper_xy: Optional[Tuple[float, float]] = None
    delta_xy: Tuple[float, float] = (0.0, 0.0)
    radius_px: Optional[float] = None
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
            "camera_center_xy": pair(self.camera_center_xy),
            "gripper_xy": pair(self.gripper_xy),
            "delta_xy": pair(self.delta_xy),
            "radius_px": None if self.radius_px is None else float(self.radius_px),
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


def image_size_pair(value: Sequence[int], name: str = "image_size") -> Tuple[int, int]:
    """Validate ``[width, height]`` and return it as two positive ints.

    Args:
        value: Two positive numbers, at least 2 in each direction.
        name: Field name used in the error message.

    Returns:
        ``(width, height)``.

    Raises:
        PickError: When the pair is missing, non-numeric or out of range.
    """
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise PickError("bad_request", f"{name} must contain numbers", 400) from exc
    if array.size != 2:
        raise PickError("bad_request", f"{name} must be [width, height]", 400)
    if not np.all(np.isfinite(array)) or np.any(array < 2) or np.any(array > 1e6):
        raise PickError(
            "bad_request",
            f"{name} must be two finite values in [2, 1e6]",
            400,
        )
    width, height = int(round(float(array[0]))), int(round(float(array[1])))
    if width < 2 or height < 2:
        raise PickError("bad_request", f"{name} is too small", 400)
    return (width, height)


def check_image_point(
    value: Sequence[float],
    image_size: Sequence[int],
    name: str = "image_uv",
) -> Tuple[float, float]:
    """Validate one sub-pixel image point against the frame size.

    The bounds check is what catches a caller that passes ``(row, col)``
    instead of ``(col, row)``: such a point usually lands outside the frame.

    Args:
        value: ``(u, v) = (col, row)`` in pixels.
        image_size: ``(width, height)`` of the frame.
        name: Field name used in the error message.

    Returns:
        The point as two floats.

    Raises:
        PickError: On missing, non-finite or out-of-image values.
    """
    point = _as_xy(value, name)
    width, height = image_size_pair(image_size)
    u, v = point
    if not (0.0 <= u <= width - 1.0 and 0.0 <= v <= height - 1.0):
        raise PickError(
            "point_out_of_image",
            f"{name} {list(point)} is outside the {width}x{height} frame",
            400,
            details={"image_size": [width, height], "point": [u, v]},
        )
    return point


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
    layer reduces them to one 3x3 matrix each.  ``locate`` only needs a point
    and the two matrices, so it depends on no image and no server-side recipe.

    The ``*_from_images`` / ``locate_image`` helpers keep the old
    detect-then-compute path for offline replay and tests.  They are not part
    of the HTTP contract.
    """

    def __init__(self, locator: Optional[CircleLocator] = None) -> None:
        # The real gantry targets can be large (bottle mouths are >1000 px at
        # 5 MP), so the pick default is wider than CircleLocator's own default.
        self.locator = locator or CircleLocator(radius_range=(6.0, 3000.0))

    # -- calibration -----------------------------------------------------

    def calibrate_vision(
        self,
        image_points: Sequence[Sequence[float]],
        robot_points: Sequence[Sequence[float]],
        *,
        image_size: Sequence[int],
        principal_point: Optional[Sequence[float]] = None,
        confidences: Optional[Sequence[float]] = None,
    ) -> ImageRobotCalibration:
        """Fit the vision affine from detected centres and flange poses.

        Args:
            image_points: ``(N, 2)`` target centres ``(u, v)`` in pixels.
            robot_points: ``(N, 2)`` or ``(N, 3)`` flange poses; Z is only
                summarised into ``calib_z``.
            image_size: ``(width, height)`` the points were measured on.
            principal_point: Camera centre in pixels; defaults to the image
                centre.
            confidences: Optional per-sample locator confidences.

        Returns:
            The fitted calibration; read ``matrix`` for the 3x3 result.

        Raises:
            PickError: On bad request data.
            CalibrationError: When the fit cannot be computed.
        """
        width, height = image_size_pair(image_size)
        for index, point in enumerate(image_points):
            check_image_point(point, (width, height), f"image_points[{index}]")
        if len(image_points) != len(robot_points):
            raise PickError(
                "point_count_mismatch",
                f"{len(image_points)} image points but "
                f"{len(robot_points)} robot points",
                400,
            )
        if len(image_points) < 4:
            raise PickError(
                "bad_request",
                "at least 4 point correspondences are required",
                400,
            )
        if confidences is not None and len(confidences) != len(image_points):
            raise PickError(
                "bad_request",
                f"confidences must match the number of points "
                f"({len(confidences)} != {len(image_points)})",
                400,
            )
        return fit_image_to_robot(
            image_points,
            robot_points,
            principal_point=principal_point,
            image_size=(width, height),
            confidences=confidences,
            metadata={"source": "pick_api"},
        )

    def calibrate_vision_from_images(
        self,
        images: Sequence[np.ndarray],
        robot_points: Sequence[Sequence[float]],
        **kwargs: Any,
    ) -> ImageRobotCalibration:
        """Locate a circle in every frame, then fit the affine (offline use)."""
        return calibrate_image_to_robot(
            images, robot_points, locator=self.locator, **kwargs
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

    # -- locate ----------------------------------------------------------

    def locate(
        self,
        image_uv: Sequence[float],
        robot_xy: Sequence[float],
        vision_matrix: Sequence[Sequence[float]],
        offset_matrix: Optional[Sequence[Sequence[float]]] = None,
        *,
        image_size: Sequence[int],
        radius_px: Optional[float] = None,
    ) -> TargetLocation:
        """Locate the target from one image point, one pose and two matrices.

        Args:
            image_uv: Target centre ``(u, v)`` from the caller's own detector.
            robot_xy: Flange XY at capture time; trailing Z is ignored.
            vision_matrix: 3x3 homogeneous affine from ``/calibrate/vision``.
            offset_matrix: Optional 3x3 pure-translation matrix from
                ``/calibrate/offset``.  When omitted, ``gripper_xy`` is None.
            image_size: ``(width, height)`` of the frame the point came from;
                it fixes the image centre the servo targets.
            radius_px: Optional target radius, echoed into the result.

        Returns:
            A :class:`TargetLocation`.

        Raises:
            PickError: On a bad pose, point, size or matrix.
        """
        pose = _as_xy(robot_xy)
        width, height = image_size_pair(image_size)
        point = check_image_point(image_uv, (width, height))
        matrix = _vision_matrix(vision_matrix)
        offset = None if offset_matrix is None else _offset_matrix(offset_matrix)

        centre = np.array(((width - 1) * 0.5, (height - 1) * 0.5), dtype=float)
        delta = matrix[:2, :2] @ (centre - np.asarray(point, dtype=float))
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
            image_uv=point,
            camera_center_xy=camera,
            gripper_xy=gripper,
            delta_xy=(float(delta[0]), float(delta[1])),
            radius_px=_finite_or_none(radius_px) if radius_px is not None else None,
        )

    def locate_image(
        self,
        image: np.ndarray,
        robot_xy: Sequence[float],
        vision_matrix: Sequence[Sequence[float]],
        offset_matrix: Optional[Sequence[Sequence[float]]] = None,
    ) -> TargetLocation:
        """Detect the target in a frame, then call :meth:`locate` (offline use)."""
        pose = _as_xy(robot_xy)
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
        detection = self.locator.find(image)
        if not detection.valid:
            return TargetLocation(
                valid=False,
                reason=detection.reason or "no_target",
                robot_xy=pose,
                radius_px=_finite_or_none(detection.radius),
                detection=detection,
            )
        return self.locate(
            detection.center,
            pose,
            vision_matrix,
            offset_matrix,
            image_size=(width, height),
            radius_px=detection.radius,
        )


def _finite_or_none(value: float) -> Optional[float]:
    """Return a finite float or ``None`` for NaNs and infinities."""
    number = float(value)
    return number if np.isfinite(number) else None


__all__ = [
    "GantryPick",
    "PickError",
    "TargetLocation",
    "check_image_point",
    "image_size_pair",
    "offset_homogeneous_matrix",
]
