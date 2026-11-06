"""
Camera-on-arm pick calibration: four-point image-to-robot transform and
camera-to-gripper offset.

Why the "four-point affine" is fitted as a *differential* transform
-------------------------------------------------------------------
The camera is mounted on the moving arm and the calibration target stays
fixed on the stage.  Each shot therefore yields one correspondence
``(robot pose r_i, target image point p_i)``.  If the four or more
correspondences are fitted as an absolute affine ``r = A p + b``, the
translation ``b`` belongs to the *specific physical target position* used
during calibration.  Evaluating that affine at the currently detected
``p`` returns the pose the arm is already at: the command is a no-op.
A numeric demonstration is in ``docs/robot-pick-calibration.md``.

The reusable part is the linear map between image *displacement* and
robot *displacement*::

    Δr = A · Δp          (A: 2x2, robot units per pixel)

``A`` is independent of where the target sits.  To put a target seen at
``p`` under the camera centre ``c``::

    Δr = A · (c - p)
    r_centre = r_current + Δr

With the optional camera-to-gripper offset ``o`` (``A_place - B_centre``)
the target can be grasped in a single move::

    grasp = r_current + A · (c - p) + o

This module implements both calibrations, their verification, and a
JSON recipe that stores every artefact.

Conventions
-----------
* ``image_uv`` is ``(u, v) = (col, row)`` in pixels, OpenCV order.
* ``robot_xy`` is ``(X, Y)`` in the controller's units (intended: mm).
* ``A`` maps *image displacement* to *robot displacement*; the fitted
  translation is stored as a diagnostic only (``translation_xy``).
* The principal point defaults to the image centre.  The absolute offset
  is insensitive to the principal point because the centring convention
  cancels between the offset sample and the grasp command, but the
  "target is at the image centre" verification criterion is not.
* A Cartesian XYZ gantry is assumed: the XY mapping does not change with
  Z.  Vision should run at a fixed camera height; changing camera Z with
  a perspective lens changes the mm/px scale.

Typical use (offline, no robot)::

    from measure import calibrate_image_to_robot

    result = calibrate_image_to_robot(images, robot_points)
    M = result.matrix            # 3x3 full affine (target-position specific)
    A = result.robot_per_pixel   # 2x2 reusable displacement map

Optional hardware capture (not needed for offline data)::

    from measure import capture_and_calibrate

    calib = capture_and_calibrate(robot, capture, positions=[...])
    offset = calibrate_gripper_offset(robot, capture, calib,
                                      place_pose=release_pose)
    recipe = RobotPickRecipe(calib, offset)
    recipe.save("pick_calibration.json")
"""

from __future__ import annotations

import itertools
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

import cv2
import numpy as np

from measure.circle_locator import CircleDetection, CircleLocator

# ---------------------------------------------------------------------------
# Schema / version constants
# ---------------------------------------------------------------------------

SCHEMA = "measure.robot_pick_calibration"
VERSION = 1
_IMAGE_SCHEMA = "measure.image_robot_calibration"
_OFFSET_SCHEMA = "measure.camera_gripper_offset"
_RECIPE_SCHEMA = "measure.robot_pick_recipe"

# Defaults
DEFAULT_MAX_RMS_PX = 3.0
DEFAULT_MAX_CONDITION = 50.0
DEFAULT_CENTER_TOLERANCE_PX = 5.0
DEFAULT_VERIFY_TOLERANCE_PX = 5.0


class CalibrationError(RuntimeError):
    """Raised when a calibration cannot be computed or is unusable."""


# ---------------------------------------------------------------------------
# Robot interface
# ---------------------------------------------------------------------------


@runtime_checkable
class RobotLink(Protocol):
    """Minimal blocking robot interface required by the calibration helpers.

    ``get_position`` must return at least ``(x, y)``; a third element is
    treated as Z.  ``move_to`` must block until the arm has settled (or
    the caller's ``settle_s`` must be long enough).  Implementations that
    do not use Z may ignore the third argument.
    """

    def get_position(self) -> Sequence[float]:
        """Return the current flange position ``(x, y[, z])``."""

    def move_to(self, x: float, y: float, z: Optional[float] = None) -> None:
        """Move the flange to ``(x, y, z)`` and block until finished."""


class CallableRobot:
    """Adapt plain callables to :class:`RobotLink`.

    ``move_to`` is always called as ``move_to(x, y, z)`` where ``z`` may be
    ``None``; the supplied callable must accept that signature (or a
    ``z=None`` default for ``(x, y)``-only controllers).
    """

    def __init__(
        self,
        get_position: Callable[[], Sequence[float]],
        move_to: Callable[..., None],
    ) -> None:
        self._get_position = get_position
        self._move_to = move_to

    def get_position(self) -> Sequence[float]:
        return self._get_position()

    def move_to(self, x: float, y: float, z: Optional[float] = None) -> None:
        self._move_to(x, y, z)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _emit(callback: Optional[Callable[[str, Dict[str, Any]], None]],
          kind: str, **payload: Any) -> None:
    if callback is not None:
        callback(kind, payload)


def _as_point(value: Sequence[float], name: str) -> np.ndarray:
    point = np.asarray(value, dtype=float).reshape(-1)
    if point.size < 2 or not np.all(np.isfinite(point[:2])):
        raise CalibrationError(f"{name} must be a finite (x, y) pair")
    return point


def _as_pose(value: Sequence[float], name: str) -> np.ndarray:
    pose = np.asarray(value, dtype=float).reshape(-1)
    if pose.size < 2 or not np.all(np.isfinite(pose)):
        raise CalibrationError(f"{name} must be a finite (x, y[, z]) pose")
    return pose


def _pixel_center(image_size: Tuple[int, int]) -> Tuple[float, float]:
    width, height = int(image_size[0]), int(image_size[1])
    return ((width - 1) * 0.5, (height - 1) * 0.5)


def _points_array(points: Sequence[Sequence[float]], name: str) -> np.ndarray:
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2:
        raise CalibrationError(f"{name} must be an (N, 2) array")
    if len(array) == 0:
        raise CalibrationError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise CalibrationError(f"{name} contains non-finite values")
    return array


def _read_pose(robot: RobotLink) -> np.ndarray:
    return _as_pose(robot.get_position(), "robot position")


def _move_robot(robot: RobotLink, pose: Sequence[float], settle_s: float) -> np.ndarray:
    target = _as_pose(pose, "target pose")
    if target.size >= 3:
        robot.move_to(float(target[0]), float(target[1]), float(target[2]))
    else:
        robot.move_to(float(target[0]), float(target[1]), None)
    if settle_s > 0:
        time.sleep(float(settle_s))
    return target


def _to_jsonable(value: Any) -> Any:
    """Recursively convert numpy containers/scalars for ``json.dump``."""
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _check_schema(data: Dict[str, Any], schema: str) -> None:
    if data.get("schema") != schema:
        raise CalibrationError(
            f"unexpected schema {data.get('schema')!r}; expected {schema!r}"
        )
    version = data.get("version", 0)
    try:
        version = int(version)
    except (TypeError, ValueError):
        raise CalibrationError("recipe version is not an integer") from None
    if version > VERSION:
        raise CalibrationError(
            f"recipe version {version} is newer than supported version {VERSION}"
        )


def _warn(warnings: List[str], message: str) -> None:
    if message not in warnings:
        warnings.append(message)


# ---------------------------------------------------------------------------
# Correspondence sample
# ---------------------------------------------------------------------------


@dataclass
class CalibrationSample:
    """One ``(robot pose, target image point)`` correspondence.

    Attributes:
        robot_xy: Flange position when the frame was captured.
        image_uv: Detected target centre ``(u, v)`` in pixels.
        confidence: Circle-locator confidence for this frame.
        residual_px: Absolute residual of this sample after the fit (px).
        residual_mm: Absolute residual converted to robot units.
        inlier: ``False`` when the robust fit down-weighted this sample.
    """

    robot_xy: Tuple[float, float]
    image_uv: Tuple[float, float]
    confidence: float = 1.0
    residual_px: float = 0.0
    residual_mm: float = 0.0
    inlier: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "robot_xy": [float(self.robot_xy[0]), float(self.robot_xy[1])],
            "image_uv": [float(self.image_uv[0]), float(self.image_uv[1])],
            "confidence": float(self.confidence),
            "residual_px": float(self.residual_px),
            "residual_mm": float(self.residual_mm),
            "inlier": bool(self.inlier),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CalibrationSample":
        robot = data.get("robot_xy", (0.0, 0.0))
        image = data.get("image_uv", (0.0, 0.0))
        return cls(
            robot_xy=(float(robot[0]), float(robot[1])),
            image_uv=(float(image[0]), float(image[1])),
            confidence=float(data.get("confidence", 1.0)),
            residual_px=float(data.get("residual_px", 0.0)),
            residual_mm=float(data.get("residual_mm", 0.0)),
            inlier=bool(data.get("inlier", True)),
        )


# ---------------------------------------------------------------------------
# Differential affine fit
# ---------------------------------------------------------------------------


def _solve_affine(
    robot_centered: np.ndarray,
    image_centered: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Solve ``robot_centered = image_centered @ A.T`` for ``A``.

    Returns:
        ``(a_transpose, a, rank)`` where ``a_transpose`` has shape (2, 2)
        and ``a`` is the robot-per-pixel matrix.
    """
    if weights is None:
        weights = np.ones(len(image_centered), dtype=float)
    sw = np.sqrt(np.asarray(weights, dtype=float))[:, None]
    solution, _, rank, _ = np.linalg.lstsq(
        image_centered * sw, robot_centered * sw, rcond=None
    )
    return solution, solution.T, int(rank)


def _full_design(image_points: np.ndarray) -> np.ndarray:
    """Design matrix ``[u, v, 1]`` for the absolute (nuisance-intercept) fit."""
    return np.column_stack([image_points, np.ones(len(image_points), dtype=float)])


def _residuals_px(
    robot: np.ndarray, design: np.ndarray, solution: np.ndarray
) -> np.ndarray:
    """Per-sample residual norm in pixels for a full-affine solution."""
    a = solution.T[:, :2]
    a_inv = np.linalg.inv(a)
    residual_robot = robot - design @ solution
    return np.linalg.norm(residual_robot @ a_inv.T, axis=1)


def _ransac_inliers(
    robot: np.ndarray, image: np.ndarray
) -> Optional[np.ndarray]:
    """Find an inlier mask with three-point exact affine models.

    The full affine has six unknowns, so three non-collinear correspondences
    determine it exactly.  With at least five samples, a gross outlier is
    rejected by the inlier count; with only four samples every three-point
    model has exactly three exact inliers and outliers cannot be detected,
    which is why RANSAC is only attempted for five or more samples.

    Returns:
        Boolean inlier mask, or ``None`` when RANSAC is not applicable.
    """
    n = len(robot)
    if n < 5:
        return None
    design = _full_design(image)
    span = max(
        float(np.ptp(image[:, 0])), float(np.ptp(image[:, 1])), 1.0
    )
    threshold = max(1.5, 0.01 * span)
    area_floor = 1e-6 * span * span
    best: Optional[Tuple[Tuple[int, float], np.ndarray]] = None

    for triple in itertools.combinations(range(n), 3):
        indices = list(triple)
        design_triple = design[indices]
        if abs(float(np.linalg.det(design_triple))) < area_floor:
            continue
        try:
            solution = np.linalg.solve(design_triple, robot[indices])
            if abs(float(np.linalg.det(solution.T[:, :2]))) < 1e-12:
                continue
            errors = _residuals_px(robot, design, solution)
        except np.linalg.LinAlgError:
            continue
        inliers = errors <= threshold
        if int(inliers.sum()) < 4:
            continue
        try:
            refit, _, rank, _ = np.linalg.lstsq(
                design[inliers], robot[inliers], rcond=None
            )
            if rank < 3 or abs(float(np.linalg.det(refit.T[:, :2]))) < 1e-12:
                continue
            refit_errors = _residuals_px(robot, design, refit)
        except np.linalg.LinAlgError:
            continue
        refit_inliers = refit_errors <= threshold
        if int(refit_inliers.sum()) < 4:
            continue
        score = (
            int(refit_inliers.sum()),
            -float(np.sum(refit_errors[refit_inliers] ** 2)),
        )
        if best is None or score > best[0]:
            best = (score, refit_inliers)
    return None if best is None else best[1]


def _robust_affine(
    robot_centered: np.ndarray,
    image_centered: np.ndarray,
    robust: bool,
    initial_inliers: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Least squares on a RANSAC inlier set.

    A soft (Huber) weight is not sufficient here: a gross outlier can sit
    hundreds of pixels away, and even a 1e-3 weight then pulls the fit by
    a pixel or two.  Calibration samples are few and detections are
    sub-pixel, so hard inlier selection is both simpler and more accurate.

    Args:
        robot_centered: Centered robot positions.
        image_centered: Centered image positions.
        robust: Honour *initial_inliers*.
        initial_inliers: Mask from :func:`_ransac_inliers`.

    Returns:
        ``(a_transpose, a, weights)`` with weights ``1`` for accepted samples
        and ``0`` for rejected samples.
    """
    n = len(robot_centered)
    weights = np.ones(n, dtype=float)
    if (
        not robust
        or n < 5
        or initial_inliers is None
        or int(np.sum(initial_inliers)) < 3
    ):
        a_t, a, _ = _solve_affine(robot_centered, image_centered)
        return a_t, a, weights

    inliers = np.asarray(initial_inliers, dtype=bool)
    weights = np.where(inliers, 1.0, 0.0)
    a_t, a, _ = _solve_affine(robot_centered, image_centered, weights)
    return a_t, a, weights


def _leave_one_out(
    robot: np.ndarray, image: np.ndarray
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Leave-one-out errors for the differential model.

    Returns:
        ``(rms_px, max_px, rms_mm, max_mm)`` or four ``None`` values when
        fewer than four correspondences are available.
    """
    n = len(robot)
    if n < 4:
        return None, None, None, None
    errors_px: List[float] = []
    errors_mm: List[float] = []
    for index in range(n):
        keep = np.ones(n, dtype=bool)
        keep[index] = False
        subset_robot = robot[keep]
        subset_image = image[keep]
        r_mean = subset_robot.mean(axis=0)
        p_mean = subset_image.mean(axis=0)
        r_centered = subset_robot - r_mean
        p_centered = subset_image - p_mean
        try:
            a_t, a, rank = _solve_affine(r_centered, p_centered)
            if rank < 2 or abs(np.linalg.det(a)) < 1e-15:
                continue
            j = np.linalg.inv(a)
        except (np.linalg.LinAlgError, ValueError):
            continue
        predicted_image = p_mean + (robot[index] - r_mean) @ j.T
        predicted_robot = r_mean + (image[index] - p_mean) @ a_t
        errors_px.append(float(np.linalg.norm(predicted_image - image[index])))
        errors_mm.append(float(np.linalg.norm(predicted_robot - robot[index])))
    if not errors_px:
        return None, None, None, None
    px = np.asarray(errors_px, dtype=float)
    mm = np.asarray(errors_mm, dtype=float)
    return (
        float(np.sqrt(np.mean(px ** 2))),
        float(np.max(px)),
        float(np.sqrt(np.mean(mm ** 2))),
        float(np.max(mm)),
    )


class ImageRobotCalibration:
    """Image-displacement to robot-displacement calibration.

    Public fields:
        robot_per_pixel: 2x2 matrix ``A``; ``Δrobot = A @ Δimage``.
        pixel_per_robot: 2x2 matrix ``A^-1``; ``Δimage = A^-1 @ Δrobot``.
        matrix: 3x3 homogeneous affine ``[[A, b], [0, 1]]`` (alias of
            ``matrix_3x3``); ``b`` is target-position specific.
        translation_xy: Nuisance intercept of the fitted full affine
            ``r = A p + b``.  It is tied to the calibration target's
            physical position and is never used by :meth:`servo_delta` or
            :meth:`center_pose`.
        principal_point: ``(u, v)`` used as the camera centre.
        image_size: ``(width, height)`` or ``None``.
        samples: Correspondence samples with per-point residuals.
        valid: ``True`` when every hard quality gate passes.
        warnings: Human-readable quality warnings.
        mm_per_px: Mean robot-units-per-pixel scale of ``A``.
        anisotropy: Ratio of the largest to the smallest singular value.
        rotation_deg: Closest rotation angle of ``A`` (polar decomposition);
            for a fronto-parallel camera it typically includes the 180°
            image-motion inversion (camera rotation + 180°).
        mirrored: ``True`` when ``det(A) < 0`` (axes mirrored).
        rms_px / max_px / rms_mm / max_mm: Fit residuals.
        inlier_rms_px: Residual RMS over robust inliers.
        loo_rms_px / loo_max_px / loo_rms_mm / loo_max_mm: Leave-one-out.
        condition: Condition number of the robot-pose design matrix.
        image_coverage: Convex-hull area of image points / image area.
        image_span_px: Larger of the image x/y point spans.
        robot_span_mm: Maximum pairwise robot-pose distance.
        calib_z: Mean Z at calibration time (``None`` when unavailable).
        created_at: ISO-8601 timestamp.
        metadata: Free-form JSON-serialisable metadata.
    """

    def __init__(self) -> None:
        self.robot_per_pixel: np.ndarray = np.eye(2, dtype=float)
        self.pixel_per_robot: np.ndarray = np.eye(2, dtype=float)
        self.translation_xy: Tuple[float, float] = (0.0, 0.0)
        self.principal_point: Tuple[float, float] = (0.0, 0.0)
        self.image_size: Optional[Tuple[int, int]] = None
        self.samples: List[CalibrationSample] = []
        self.valid: bool = False
        self.warnings: List[str] = []
        self.mm_per_px: float = 0.0
        self.anisotropy: float = 1.0
        self.rotation_deg: float = 0.0
        self.mirrored: bool = False
        self.rms_px: float = 0.0
        self.max_px: float = 0.0
        self.rms_mm: float = 0.0
        self.max_mm: float = 0.0
        self.inlier_rms_px: float = 0.0
        self.loo_rms_px: Optional[float] = None
        self.loo_max_px: Optional[float] = None
        self.loo_rms_mm: Optional[float] = None
        self.loo_max_mm: Optional[float] = None
        self.condition: float = 0.0
        self.image_coverage: Optional[float] = None
        self.image_span_px: float = 0.0
        self.robot_span_mm: float = 0.0
        self.calib_z: Optional[float] = None
        self.created_at: str = _now()
        self.metadata: Dict[str, Any] = {}

    # -- construction ----------------------------------------------------

    @classmethod
    def fit(
        cls,
        robot_points: Sequence[Sequence[float]],
        image_points: Sequence[Sequence[float]],
        *,
        principal_point: Optional[Sequence[float]] = None,
        image_size: Optional[Sequence[int]] = None,
        confidences: Optional[Sequence[float]] = None,
        robust: bool = True,
        max_rms_px: float = DEFAULT_MAX_RMS_PX,
        max_condition: float = DEFAULT_MAX_CONDITION,
        calib_z: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ImageRobotCalibration":
        """Fit the affine ``r = A p + b`` and retain the differential map ``A``.

        Only ``A`` is reusable across target positions; ``b`` is stored as a
        diagnostic intercept (``translation_xy``).

        Args:
            robot_points: ``(N, 2)`` flange positions.
            image_points: ``(N, 2)`` target image positions ``(u, v)``.
            principal_point: Camera centre in pixels; defaults to the image
                centre when *image_size* is supplied.
            image_size: ``(width, height)`` of the captured frames.
            confidences: Optional per-sample locator confidences.
            robust: Use Huber IRLS when at least five samples are available.
            max_rms_px: RMS residual above which the result is invalid.
            max_condition: Condition number above which the result is invalid.
            calib_z: Mean robot Z during calibration.
            metadata: Free-form metadata copied onto the result.

        Raises:
            CalibrationError: On invalid input, collinear robot motion, a
                singular mapping, or a missing principal point.

        Returns:
            The fitted calibration; check ``valid`` and ``warnings``.
        """
        robot = _points_array(robot_points, "robot_points")
        image = _points_array(image_points, "image_points")
        if len(robot) != len(image):
            raise CalibrationError("robot_points and image_points must be the same length")
        if len(robot) < 3:
            raise CalibrationError("at least 3 correspondences are required")

        if image_size is not None:
            image_size = (int(image_size[0]), int(image_size[1]))
            if image_size[0] <= 1 or image_size[1] <= 1:
                raise CalibrationError("image_size must be positive")
        if principal_point is None:
            if image_size is None:
                raise CalibrationError(
                    "principal_point or image_size is required"
                )
            principal = _pixel_center(image_size)
        else:
            principal = _as_point(principal_point, "principal_point")
        principal = (float(principal[0]), float(principal[1]))

        if confidences is None:
            confidences = [1.0] * len(robot)
        if len(confidences) != len(robot):
            raise CalibrationError("confidences must match the number of points")
        conf = np.asarray(confidences, dtype=float)
        if not np.all(np.isfinite(conf)):
            raise CalibrationError("confidences contain non-finite values")

        # With five or more samples, use a RANSAC inlier mask so a single
        # bad detection cannot bias the fit.  The means are then taken over
        # the inliers: the differential model has no intercept, so centering
        # on a contaminated mean would leave a constant offset on the good
        # samples.
        inlier_hint = _ransac_inliers(robot, image) if robust else None
        if inlier_hint is not None and np.any(inlier_hint):
            r_mean = robot[inlier_hint].mean(axis=0)
            p_mean = image[inlier_hint].mean(axis=0)
        else:
            r_mean = robot.mean(axis=0)
            p_mean = image.mean(axis=0)
        r_centered = robot - r_mean
        p_centered = image - p_mean

        singular = np.linalg.svd(r_centered, compute_uv=False)
        if len(singular) < 2 or singular[0] <= 1e-12:
            raise CalibrationError("robot poses do not span the XY plane")
        condition = float(singular[0] / max(singular[1], 1e-12))
        if singular[1] <= 1e-9 * max(1.0, singular[0]):
            raise CalibrationError("robot poses are collinear; move in both X and Y")

        a_t, a, weights = _robust_affine(
            r_centered, p_centered, robust, inlier_hint
        )
        if not np.all(np.isfinite(a)):
            raise CalibrationError("affine fit produced non-finite values")
        determinant = float(np.linalg.det(a))
        if abs(determinant) < 1e-12:
            raise CalibrationError("affine fit is singular; check the correspondences")
        try:
            j = np.linalg.inv(a)
        except np.linalg.LinAlgError:
            raise CalibrationError("affine fit is singular") from None

        residual_robot = r_centered - p_centered @ a_t
        residual_image = residual_robot @ j.T
        residual_px = np.linalg.norm(residual_image, axis=1)
        residual_mm = np.linalg.norm(residual_robot, axis=1)
        rms_px = float(np.sqrt(np.mean(residual_px ** 2)))
        max_px = float(np.max(residual_px))
        rms_mm = float(np.sqrt(np.mean(residual_mm ** 2)))
        max_mm = float(np.max(residual_mm))
        inlier_mask = weights >= 0.5
        inlier_rms_px = (
            float(np.sqrt(np.mean(residual_px[inlier_mask] ** 2)))
            if np.any(inlier_mask)
            else rms_px
        )

        singular_a = np.linalg.svd(a, compute_uv=False)
        mm_per_px = float(math.sqrt(abs(singular_a[0] * singular_a[1])))
        anisotropy = float(singular_a[0] / max(singular_a[1], 1e-12))
        u, _, v_t = np.linalg.svd(a)
        rotation = u @ v_t
        if np.linalg.det(rotation) < 0:
            rotation = u @ np.diag([1.0, -1.0]) @ v_t
        rotation_deg = float(math.degrees(math.atan2(rotation[1, 0], rotation[0, 0])))
        mirrored = bool(determinant < 0)

        span_x = float(np.ptp(image[:, 0]))
        span_y = float(np.ptp(image[:, 1]))
        image_span_px = max(span_x, span_y)
        robot_span = 0.0
        if len(robot) > 1:
            deltas = robot[:, None, :] - robot[None, :, :]
            robot_span = float(np.max(np.linalg.norm(deltas, axis=2)))
        image_coverage: Optional[float] = None
        if image_size is not None:
            hull = cv2.convexHull(image.astype(np.float32)).reshape(-1, 2)
            image_coverage = float(
                abs(cv2.contourArea(hull)) / float(image_size[0] * image_size[1])
            )

        inlier_indices = weights >= 0.5
        if np.any(~inlier_indices) and int(np.sum(inlier_indices)) >= 4:
            loo_rms_px, loo_max_px, loo_rms_mm, loo_max_mm = _leave_one_out(
                robot[inlier_indices], image[inlier_indices]
            )
        else:
            loo_rms_px, loo_max_px, loo_rms_mm, loo_max_mm = _leave_one_out(
                robot, image
            )

        warnings: List[str] = []
        if condition > 10.0:
            _warn(warnings, "ill_conditioned_motion")
        if anisotropy > 1.05:
            _warn(warnings, "anisotropic_scale")
        if mirrored:
            _warn(warnings, "mirrored_axes")
        if len(robot) == 3:
            _warn(warnings, "minimum_points")
        if np.any(~inlier_mask):
            _warn(warnings, "outlier_rejected")
        if inlier_rms_px > 0.5:
            _warn(warnings, "high_residual")
        if loo_rms_px is not None and loo_rms_px > 1.0:
            _warn(warnings, "poor_generalisation")
        if image_size is not None and image_span_px < 0.15 * min(image_size):
            _warn(warnings, "clustered_image_points")
        if image_coverage is not None and image_coverage < 0.10:
            _warn(warnings, "low_image_coverage")
        if mm_per_px < 1e-4 or mm_per_px > 100.0:
            _warn(warnings, "implausible_scale")

        valid = bool(
            np.isfinite(mm_per_px)
            and 1e-4 <= mm_per_px <= 100.0
            and condition <= max_condition
            and inlier_rms_px <= max_rms_px
            and (
                image_size is None
                or image_span_px >= 0.05 * min(image_size)
            )
        )

        translation = r_mean - a @ p_mean
        obj = cls()
        obj.robot_per_pixel = a
        obj.pixel_per_robot = j
        obj.translation_xy = (float(translation[0]), float(translation[1]))
        obj.principal_point = principal
        obj.image_size = image_size
        obj.samples = [
            CalibrationSample(
                robot_xy=(float(robot[i, 0]), float(robot[i, 1])),
                image_uv=(float(image[i, 0]), float(image[i, 1])),
                confidence=float(conf[i]),
                residual_px=float(residual_px[i]),
                residual_mm=float(residual_mm[i]),
                inlier=bool(inlier_mask[i]),
            )
            for i in range(len(robot))
        ]
        obj.valid = valid
        obj.warnings = warnings
        obj.mm_per_px = mm_per_px
        obj.anisotropy = anisotropy
        obj.rotation_deg = rotation_deg
        obj.mirrored = mirrored
        obj.rms_px = rms_px
        obj.max_px = max_px
        obj.rms_mm = rms_mm
        obj.max_mm = max_mm
        obj.inlier_rms_px = inlier_rms_px
        obj.loo_rms_px = loo_rms_px
        obj.loo_max_px = loo_max_px
        obj.loo_rms_mm = loo_rms_mm
        obj.loo_max_mm = loo_max_mm
        obj.condition = condition
        obj.image_coverage = image_coverage
        obj.image_span_px = image_span_px
        obj.robot_span_mm = robot_span
        obj.calib_z = float(calib_z) if calib_z is not None else None
        obj.metadata = dict(metadata or {})
        return obj

    # -- geometry --------------------------------------------------------

    def servo_delta(
        self,
        image_uv: Sequence[float],
        target_uv: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        """Robot displacement that moves *image_uv* to *target_uv*.

        Args:
            image_uv: Current target centre ``(u, v)`` in pixels.
            target_uv: Desired image position; defaults to the principal
                point (the camera centre).

        Returns:
            ``Δrobot`` as a length-2 array (robot units).
        """
        point = _as_point(image_uv, "image_uv")
        target = (
            np.asarray(self.principal_point, dtype=float)
            if target_uv is None
            else _as_point(target_uv, "target_uv")
        )
        return self.robot_per_pixel @ (target[:2] - point[:2])

    def center_pose(
        self,
        current_pose: Sequence[float],
        image_uv: Sequence[float],
        target_uv: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        """Return the pose that puts the detected target at *target_uv*.

        The pose's Z component is preserved.
        """
        pose = _as_pose(current_pose, "current_pose").copy()
        pose[:2] += self.servo_delta(image_uv, target_uv)
        return pose

    def image_delta(self, robot_delta: Sequence[float]) -> np.ndarray:
        """Return the image displacement caused by a robot displacement."""
        delta = _as_point(robot_delta, "robot_delta")[:2]
        return self.pixel_per_robot @ delta

    def robot_delta(self, image_delta: Sequence[float]) -> np.ndarray:
        """Return the robot displacement that produces an image displacement."""
        delta = _as_point(image_delta, "image_delta")[:2]
        return self.robot_per_pixel @ delta

    @property
    def matrix(self) -> np.ndarray:
        """The 3x3 homogeneous affine ``[[A, b], [0, 1]]``.

        ``b`` is tied to the calibration target position; use
        :attr:`robot_per_pixel` (or :meth:`servo_delta`) for commands.
        """
        return self.matrix_3x3

    @property
    def matrix_3x3(self) -> np.ndarray:
        """Homogeneous full affine ``[[A, b], [0, 1]]``.

        ``b`` is the intercept fitted on the calibration data and is tied
        to the physical position of the calibration target.  It is stored
        for inspection/reproduction only: :meth:`servo_delta` and
        :meth:`center_pose` use the differential map ``A`` only, because a
        new target has a different ``b``.
        """
        matrix = np.eye(3, dtype=float)
        matrix[:2, :2] = self.robot_per_pixel
        matrix[:2, 2] = self.translation_xy
        return matrix

    # -- serialisation ---------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "schema": _IMAGE_SCHEMA,
            "version": VERSION,
            "model": "differential_affine",
            "robot_per_pixel": self.robot_per_pixel.tolist(),
            "pixel_per_robot": self.pixel_per_robot.tolist(),
            "matrix_3x3": self.matrix_3x3.tolist(),
            "translation_xy": [float(self.translation_xy[0]),
                               float(self.translation_xy[1])],
            "translation_reusable": False,
            "principal_point": [float(self.principal_point[0]),
                                float(self.principal_point[1])],
            "image_size": None if self.image_size is None else [int(v) for v in self.image_size],
            "valid": bool(self.valid),
            "warnings": list(self.warnings),
            "mm_per_px": float(self.mm_per_px),
            "anisotropy": float(self.anisotropy),
            "rotation_deg": float(self.rotation_deg),
            "mirrored": bool(self.mirrored),
            "rms_px": float(self.rms_px),
            "max_px": float(self.max_px),
            "rms_mm": float(self.rms_mm),
            "max_mm": float(self.max_mm),
            "inlier_rms_px": float(self.inlier_rms_px),
            "loo_rms_px": self.loo_rms_px,
            "loo_max_px": self.loo_max_px,
            "loo_rms_mm": self.loo_rms_mm,
            "loo_max_mm": self.loo_max_mm,
            "condition": float(self.condition),
            "image_coverage": self.image_coverage,
            "image_span_px": float(self.image_span_px),
            "robot_span_mm": float(self.robot_span_mm),
            "calib_z": self.calib_z,
            "num_points": len(self.samples),
            "samples": [sample.as_dict() for sample in self.samples],
            "created_at": self.created_at,
            "metadata": _to_jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImageRobotCalibration":
        """Build a calibration from :meth:`to_dict` output."""
        _check_schema(data, _IMAGE_SCHEMA)
        model = data.get("model", "differential_affine")
        if model != "differential_affine":
            raise CalibrationError(f"unsupported calibration model {model!r}")
        matrix = np.asarray(data.get("robot_per_pixel"), dtype=float)
        if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
            raise CalibrationError("robot_per_pixel must be a finite 2x2 matrix")
        if abs(np.linalg.det(matrix)) < 1e-15:
            raise CalibrationError("robot_per_pixel is singular")
        # Always recompute the inverse from A: a hand-edited or stale
        # pixel_per_robot must never disagree with the forward map.
        try:
            pixel = np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            raise CalibrationError("robot_per_pixel is singular") from None
        principal = data.get("principal_point", (0.0, 0.0))
        image_size = data.get("image_size")
        obj = cls()
        translation = data.get("translation_xy", (0.0, 0.0))
        obj.robot_per_pixel = matrix
        obj.pixel_per_robot = pixel
        obj.translation_xy = (float(translation[0]), float(translation[1]))
        obj.principal_point = (float(principal[0]), float(principal[1]))
        obj.image_size = None if image_size is None else (int(image_size[0]),
                                                          int(image_size[1]))
        obj.valid = bool(data.get("valid", False))
        obj.warnings = [str(item) for item in data.get("warnings", [])]
        obj.mm_per_px = float(data.get("mm_per_px", 0.0))
        obj.anisotropy = float(data.get("anisotropy", 1.0))
        obj.rotation_deg = float(data.get("rotation_deg", 0.0))
        obj.mirrored = bool(data.get("mirrored", False))
        obj.rms_px = float(data.get("rms_px", 0.0))
        obj.max_px = float(data.get("max_px", 0.0))
        obj.rms_mm = float(data.get("rms_mm", 0.0))
        obj.max_mm = float(data.get("max_mm", 0.0))
        obj.inlier_rms_px = float(data.get("inlier_rms_px", 0.0))
        obj.loo_rms_px = data.get("loo_rms_px")
        obj.loo_max_px = data.get("loo_max_px")
        obj.loo_rms_mm = data.get("loo_rms_mm")
        obj.loo_max_mm = data.get("loo_max_mm")
        obj.condition = float(data.get("condition", 0.0))
        obj.image_coverage = data.get("image_coverage")
        obj.image_span_px = float(data.get("image_span_px", 0.0))
        obj.robot_span_mm = float(data.get("robot_span_mm", 0.0))
        obj.calib_z = data.get("calib_z")
        obj.created_at = str(data.get("created_at", _now()))
        obj.metadata = dict(data.get("metadata", {}))
        obj.samples = [
            CalibrationSample.from_dict(item) for item in data.get("samples", [])
        ]
        return obj

    def save(self, filepath: str) -> None:
        """Write the calibration as JSON."""
        with open(filepath, "w", encoding="utf-8") as handle:
            json.dump(_to_jsonable(self.to_dict()), handle, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "ImageRobotCalibration":
        """Read a calibration written by :meth:`save`."""
        with open(filepath, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise CalibrationError("calibration file must contain a JSON object")
        return cls.from_dict(data)

    def summary(self) -> Dict[str, Any]:
        """Return a compact diagnostic summary."""
        return {
            "valid": self.valid,
            "num_points": len(self.samples),
            "mm_per_px": self.mm_per_px,
            "rotation_deg": self.rotation_deg,
            "rms_px": self.rms_px,
            "max_px": self.max_px,
            "rms_mm": self.rms_mm,
            "loo_rms_px": self.loo_rms_px,
            "condition": self.condition,
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------------------
# Camera-to-gripper offset
# ---------------------------------------------------------------------------


@dataclass
class OffsetSample:
    """One release/centre pair used by the offset calibration.

    Attributes:
        place_pose: Flange XY at the moment the target was released
            (gripper centred over the target).
        center_pose: Flange XY after the camera was moved so the target
            is centred in the image.
        center_error_px: Residual distance from the principal point after
            the centring move.
        confidence: Locator confidence of the final detection.
        camera_pose: Flange XY where the target was first detected (before
            centring); kept for diagnostics and future joint fitting.
        image_uv: Detection at ``camera_pose``.
        z_place: Robot Z at release, when available.
        z_camera: Robot Z during the centring view, when available.
    """

    place_pose: Tuple[float, float]
    center_pose: Tuple[float, float]
    center_error_px: float = 0.0
    confidence: float = 0.0
    camera_pose: Optional[Tuple[float, float]] = None
    image_uv: Optional[Tuple[float, float]] = None
    z_place: Optional[float] = None
    z_camera: Optional[float] = None

    @property
    def offset_xy(self) -> Tuple[float, float]:
        """``place_pose - center_pose`` (add to a centred pose to grasp)."""
        return (
            float(self.place_pose[0] - self.center_pose[0]),
            float(self.place_pose[1] - self.center_pose[1]),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "place_pose": [float(self.place_pose[0]), float(self.place_pose[1])],
            "center_pose": [float(self.center_pose[0]), float(self.center_pose[1])],
            "offset_xy": list(self.offset_xy),
            "center_error_px": float(self.center_error_px),
            "confidence": float(self.confidence),
            "camera_pose": None if self.camera_pose is None else
            [float(self.camera_pose[0]), float(self.camera_pose[1])],
            "image_uv": None if self.image_uv is None else
            [float(self.image_uv[0]), float(self.image_uv[1])],
            "z_place": self.z_place,
            "z_camera": self.z_camera,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OffsetSample":
        place = data.get("place_pose", (0.0, 0.0))
        center = data.get("center_pose", (0.0, 0.0))
        camera_pose = data.get("camera_pose")
        image_uv = data.get("image_uv")
        return cls(
            place_pose=(float(place[0]), float(place[1])),
            center_pose=(float(center[0]), float(center[1])),
            center_error_px=float(data.get("center_error_px", 0.0)),
            confidence=float(data.get("confidence", 0.0)),
            camera_pose=None if camera_pose is None else
            (float(camera_pose[0]), float(camera_pose[1])),
            image_uv=None if image_uv is None else
            (float(image_uv[0]), float(image_uv[1])),
            z_place=data.get("z_place"),
            z_camera=data.get("z_camera"),
        )


class CameraGripperOffset:
    """Camera-to-gripper translation expressed as a grasp correction.

    ``offset_xy`` is the mean of ``place_pose - center_pose``.  Adding it
    to a camera-centred flange pose puts the gripper over the target::

        grasp_pose = camera_centred_pose + offset_xy

    Physically this is the camera centre relative to the gripper centre
    (``K - G``) under the ideal affine model, measured in the same
    robot-space convention.  The exact wording matters less than the sign:
    it is added to the camera-centred pose, never subtracted.
    """

    def __init__(self) -> None:
        self.offset_xy: Tuple[float, float] = (0.0, 0.0)
        self.samples: List[OffsetSample] = []
        self.std_xy: Tuple[float, float] = (0.0, 0.0)
        self.mean_center_error_px: float = 0.0
        self.max_center_error_px: float = 0.0
        self.center_tolerance_px: float = DEFAULT_CENTER_TOLERANCE_PX
        self.valid: bool = False
        self.warnings: List[str] = []
        self.created_at: str = _now()
        self.metadata: Dict[str, Any] = {}

    @classmethod
    def from_samples(
        cls,
        samples: Sequence[OffsetSample],
        *,
        center_tolerance_px: float = DEFAULT_CENTER_TOLERANCE_PX,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "CameraGripperOffset":
        """Aggregate offset samples into a reusable correction.

        Args:
            samples: One or more :class:`OffsetSample` values.
            center_tolerance_px: Maximum accepted centring residual.
            metadata: Free-form metadata.

        Raises:
            CalibrationError: When *samples* is empty or a pose is not finite.
        """
        if not samples:
            raise CalibrationError("at least one offset sample is required")
        for sample in samples:
            _as_point(sample.place_pose, "place_pose")
            _as_point(sample.center_pose, "center_pose")
        offsets = np.asarray([sample.offset_xy for sample in samples], dtype=float)
        center_errors = np.asarray(
            [sample.center_error_px for sample in samples], dtype=float
        )
        if not np.all(np.isfinite(offsets)):
            raise CalibrationError("offset samples contain non-finite values")
        if not np.all(np.isfinite(center_errors)):
            raise CalibrationError("offset samples contain non-finite centring errors")
        mean = offsets.mean(axis=0)
        if len(offsets) > 1:
            std = offsets.std(axis=0, ddof=1)
        else:
            std = np.zeros(2, dtype=float)

        warnings: List[str] = []
        mean_error = float(np.mean(center_errors))
        max_error = float(np.max(center_errors))
        if max_error > center_tolerance_px:
            _warn(warnings, "centring_error_high")
        if len(offsets) > 1 and float(np.max(std)) > 0.5:
            _warn(warnings, "offset_inconsistent")
        low_confidence = [s for s in samples if s.confidence and s.confidence < 0.5]
        if low_confidence:
            _warn(warnings, "low_confidence_detection")

        obj = cls()
        obj.offset_xy = (float(mean[0]), float(mean[1]))
        obj.samples = list(samples)
        obj.std_xy = (float(std[0]), float(std[1]))
        obj.mean_center_error_px = mean_error
        obj.max_center_error_px = max_error
        obj.center_tolerance_px = float(center_tolerance_px)
        obj.valid = bool(max_error <= center_tolerance_px and np.all(np.isfinite(mean)))
        obj.warnings = warnings
        obj.metadata = dict(metadata or {})
        return obj

    def apply(self, center_pose: Sequence[float]) -> np.ndarray:
        """Return the grasp pose for a camera-centred pose."""
        pose = _as_pose(center_pose, "center_pose").copy()
        pose[0] += self.offset_xy[0]
        pose[1] += self.offset_xy[1]
        return pose

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": _OFFSET_SCHEMA,
            "version": VERSION,
            "offset_xy": [float(self.offset_xy[0]), float(self.offset_xy[1])],
            "convention": "grasp_pose = camera_centred_pose + offset_xy",
            "std_xy": [float(self.std_xy[0]), float(self.std_xy[1])],
            "mean_center_error_px": float(self.mean_center_error_px),
            "max_center_error_px": float(self.max_center_error_px),
            "center_tolerance_px": float(self.center_tolerance_px),
            "valid": bool(self.valid),
            "warnings": list(self.warnings),
            "num_samples": len(self.samples),
            "samples": [sample.as_dict() for sample in self.samples],
            "created_at": self.created_at,
            "metadata": _to_jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CameraGripperOffset":
        _check_schema(data, _OFFSET_SCHEMA)
        offset = data.get("offset_xy")
        if offset is None or len(offset) != 2:
            raise CalibrationError("offset_xy must be a 2-element vector")
        obj = cls()
        obj.offset_xy = (float(offset[0]), float(offset[1]))
        std = data.get("std_xy", (0.0, 0.0))
        obj.std_xy = (float(std[0]), float(std[1]))
        obj.mean_center_error_px = float(data.get("mean_center_error_px", 0.0))
        obj.max_center_error_px = float(data.get("max_center_error_px", 0.0))
        obj.center_tolerance_px = float(
            data.get("center_tolerance_px", DEFAULT_CENTER_TOLERANCE_PX)
        )
        obj.valid = bool(data.get("valid", False))
        obj.warnings = [str(item) for item in data.get("warnings", [])]
        obj.created_at = str(data.get("created_at", _now()))
        obj.metadata = dict(data.get("metadata", {}))
        obj.samples = [OffsetSample.from_dict(item) for item in data.get("samples", [])]
        return obj

    def save(self, filepath: str) -> None:
        """Write the offset calibration as JSON."""
        with open(filepath, "w", encoding="utf-8") as handle:
            json.dump(_to_jsonable(self.to_dict()), handle, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "CameraGripperOffset":
        """Read an offset calibration written by :meth:`save`."""
        with open(filepath, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise CalibrationError("offset file must contain a JSON object")
        return cls.from_dict(data)

    def summary(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "offset_xy": [float(self.offset_xy[0]), float(self.offset_xy[1])],
            "std_xy": [float(self.std_xy[0]), float(self.std_xy[1])],
            "num_samples": len(self.samples),
            "mean_center_error_px": self.mean_center_error_px,
            "warnings": list(self.warnings),
        }


def combine_offsets(
    offsets: Sequence[CameraGripperOffset],
    *,
    center_tolerance_px: float = DEFAULT_CENTER_TOLERANCE_PX,
) -> CameraGripperOffset:
    """Merge several offset calibrations into one (mean + spread)."""
    samples: List[OffsetSample] = []
    for offset in offsets:
        samples.extend(offset.samples)
    return CameraGripperOffset.from_samples(
        samples, center_tolerance_px=center_tolerance_px
    )


# ---------------------------------------------------------------------------
# Recipe
# ---------------------------------------------------------------------------


class RobotPickRecipe:
    """Complete pick recipe: image calibration + gripper offset + locator.

    The command path uses only the differential map::

        grasp = r_current + A (c - p) + offset

    The stored full-affine intercept is diagnostic and is not used here.
    """

    def __init__(
        self,
        image_robot: ImageRobotCalibration,
        gripper_offset: Optional[CameraGripperOffset] = None,
        locator: Optional[CircleLocator] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not isinstance(image_robot, ImageRobotCalibration):
            raise TypeError("image_robot must be an ImageRobotCalibration")
        if gripper_offset is not None and not isinstance(gripper_offset, CameraGripperOffset):
            raise TypeError("gripper_offset must be a CameraGripperOffset")
        self.image_robot = image_robot
        self.gripper_offset = gripper_offset
        self.locator = locator or CircleLocator()
        self.metadata = dict(metadata or {})
        self.created_at = _now()

    # -- geometry --------------------------------------------------------

    def detect(self, image: np.ndarray, **kwargs: Any) -> CircleDetection:
        """Locate the target with the recipe's locator configuration."""
        return self.locator.find(image, **kwargs)

    def center_pose(self, current_pose: Sequence[float],
                    image_uv: Sequence[float]) -> np.ndarray:
        """Pose that puts the detected target at the principal point."""
        return self.image_robot.center_pose(current_pose, image_uv)

    def grasp_pose(
        self,
        current_pose: Sequence[float],
        image_uv: Sequence[float],
        z: Optional[float] = None,
    ) -> np.ndarray:
        """Pose that puts the gripper over the detected target.

        Args:
            current_pose: Flange pose at detection time.
            image_uv: Detected target centre.
            z: Optional grasp Z; when ``None`` the current Z is kept.

        Raises:
            CalibrationError: When the gripper offset is not calibrated.
        """
        if self.gripper_offset is None:
            raise CalibrationError("gripper offset has not been calibrated")
        center = self.image_robot.center_pose(current_pose, image_uv)
        pose = self.gripper_offset.apply(center)
        if z is not None:
            pose = pose.copy()
            if pose.size >= 3:
                pose[2] = float(z)
            else:
                pose = np.append(pose, float(z))
        return pose

    def pick_delta(self, image_uv: Sequence[float]) -> np.ndarray:
        """Single-move displacement from the current pose to the grasp pose.

        This is ``A·(c − p) + offset``: the user-facing one-shot command.
        """
        if self.gripper_offset is None:
            raise CalibrationError("gripper offset has not been calibrated")
        delta = self.image_robot.servo_delta(image_uv)
        return delta + np.asarray(self.gripper_offset.offset_xy, dtype=float)

    def expected_uv_at_grasp(self) -> Tuple[float, float]:
        """Image point where the target should appear at the grasp pose.

        Used to verify the offset end-to-end when the gripper does not
        occlude the target.
        """
        if self.gripper_offset is None:
            raise CalibrationError("gripper offset has not been calibrated")
        point = self.image_robot.pixel_per_robot @ np.asarray(
            self.gripper_offset.offset_xy, dtype=float
        )
        center = np.asarray(self.image_robot.principal_point, dtype=float)
        return (float(center[0] + point[0]), float(center[1] + point[1]))

    # -- validation ------------------------------------------------------

    def validate(self) -> Dict[str, Any]:
        """Return a combined validity report."""
        warnings = list(self.image_robot.warnings)
        if self.gripper_offset is None:
            warnings.append("offset_missing")
        else:
            warnings.extend(self.gripper_offset.warnings)
        valid = self.image_robot.valid and self.gripper_offset is not None \
            and self.gripper_offset.valid
        return {
            "valid": bool(valid),
            "warnings": list(dict.fromkeys(warnings)),
            "image_calibration_valid": bool(self.image_robot.valid),
            "offset_valid": bool(self.gripper_offset.valid)
            if self.gripper_offset is not None else False,
        }

    # -- serialisation ---------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": _RECIPE_SCHEMA,
            "version": VERSION,
            "created_at": self.created_at,
            "image_robot": self.image_robot.to_dict(),
            "gripper_offset": None if self.gripper_offset is None
            else self.gripper_offset.to_dict(),
            "locator": self.locator.to_dict(),
            "metadata": _to_jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RobotPickRecipe":
        _check_schema(data, _RECIPE_SCHEMA)
        image_data = data.get("image_robot")
        if not isinstance(image_data, dict):
            raise CalibrationError("recipe is missing image_robot")
        image_robot = ImageRobotCalibration.from_dict(image_data)
        offset_data = data.get("gripper_offset")
        offset = None if offset_data is None else CameraGripperOffset.from_dict(offset_data)
        locator = CircleLocator.from_dict(data.get("locator", {}))
        recipe = cls(
            image_robot,
            offset,
            locator,
            metadata=dict(data.get("metadata", {})),
        )
        recipe.created_at = str(data.get("created_at", recipe.created_at))
        return recipe

    def save(self, filepath: str) -> None:
        """Write the complete recipe as JSON."""
        with open(filepath, "w", encoding="utf-8") as handle:
            json.dump(_to_jsonable(self.to_dict()), handle, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "RobotPickRecipe":
        """Read a recipe written by :meth:`save`."""
        with open(filepath, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise CalibrationError("recipe file must contain a JSON object")
        return cls.from_dict(data)


# ---------------------------------------------------------------------------
# Driver helpers
# ---------------------------------------------------------------------------


def _detect_frame(
    capture: Callable[[], np.ndarray], locator: CircleLocator
) -> Tuple[np.ndarray, CircleDetection, Tuple[int, int]]:
    image = capture()
    if image is None or not isinstance(image, np.ndarray):
        raise CalibrationError("capture() must return a numpy image")
    detection = locator.find(image)
    size = (int(image.shape[1]), int(image.shape[0]))
    return image, detection, size


def _move_and_detect(
    robot: RobotLink,
    capture: Callable[[], np.ndarray],
    locator: CircleLocator,
    pose: Sequence[float],
    settle_s: float,
) -> Tuple[np.ndarray, CircleDetection, Tuple[int, int]]:
    _move_robot(robot, pose, settle_s)
    actual = _read_pose(robot)
    image, detection, size = _detect_frame(capture, locator)
    return actual, detection, size


def _servo_to_uv(
    robot: RobotLink,
    capture: Callable[[], np.ndarray],
    locator: CircleLocator,
    robot_per_pixel: np.ndarray,
    target_uv: Sequence[float],
    settle_s: float,
    *,
    max_iterations: int = 2,
    tolerance_px: float = 6.0,
    travel_limit_mm: Optional[float] = None,
    reference_pose: Optional[Sequence[float]] = None,
) -> Optional[Tuple[np.ndarray, CircleDetection]]:
    """Move the arm so the target reaches *target_uv*.

    Returns ``(pose, detection)`` of the final state, or ``None`` when the
    target is lost or a travel limit would be exceeded.  The final error is
    not enforced; callers decide whether it is acceptable.
    """
    pose = _read_pose(robot)
    detection = locator.find(capture())
    if not detection.valid:
        return None
    target = _as_point(target_uv, "target_uv")[:2]
    for _ in range(max_iterations + 1):
        error = target - np.asarray(detection.center, dtype=float)
        if float(np.linalg.norm(error)) <= tolerance_px:
            return pose, detection
        delta = np.asarray(robot_per_pixel, dtype=float) @ error
        new_pose = pose.copy()
        new_pose[:2] += delta
        if (
            travel_limit_mm is not None
            and reference_pose is not None
            and float(
                np.linalg.norm(new_pose[:2] - _as_pose(reference_pose, "reference")[:2])
            ) > travel_limit_mm
        ):
            return None
        _move_robot(robot, new_pose, settle_s)
        pose = _read_pose(robot)
        detection = locator.find(capture())
        if not detection.valid:
            return None
    return pose, detection


# ---------------------------------------------------------------------------
# Public calibration functions
# ---------------------------------------------------------------------------


def fit_image_to_robot(
    image_points: Sequence[Sequence[float]],
    robot_points: Sequence[Sequence[float]],
    *,
    principal_point: Optional[Sequence[float]] = None,
    image_size: Optional[Sequence[int]] = None,
    calib_z: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
    **fit_kwargs: Any,
) -> ImageRobotCalibration:
    """Fit the affine from detected image centres and robot poses.

    Use this when the circle centres are already known.  For raw images,
    use :func:`calibrate_image_to_robot`.

    Args:
        image_points: ``(N, 2)`` target centres ``(u, v)`` in pixels.
        robot_points: ``(N, 2)`` or ``(N, 3)`` flange poses; only XY is fitted.
        principal_point: Camera centre in pixels.
        image_size: ``(width, height)`` of the source frames.
        calib_z: Optional height; otherwise inferred by the caller/fit.
        metadata: Free-form metadata copied onto the result.
        **fit_kwargs: Forwarded to :meth:`ImageRobotCalibration.fit`
            (for example ``robust``, ``confidences`` or ``max_rms_px``).

    Returns:
        The fitted calibration; check ``valid`` and ``warnings``.
    """
    poses = [
        _as_pose(point, f"robot_points[{index}]")
        for index, point in enumerate(robot_points)
    ]
    if calib_z is None:
        z_values = [float(pose[2]) for pose in poses if pose.size >= 3]
        if z_values:
            calib_z = float(np.mean(z_values))
    return ImageRobotCalibration.fit(
        [pose[:2] for pose in poses],
        image_points,
        principal_point=principal_point,
        image_size=image_size,
        calib_z=calib_z,
        metadata=metadata,
        **fit_kwargs,
    )


def calibrate_image_to_robot(
    images: Sequence[np.ndarray],
    robot_points: Sequence[Sequence[float]],
    *,
    locator: Optional[CircleLocator] = None,
    principal_point: Optional[Sequence[float]] = None,
    image_size: Optional[Sequence[int]] = None,
    calib_z: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> ImageRobotCalibration:
    """Calibrate image-to-robot XY from images and matching robot poses.

    This is the offline, robot-free entry point.  Each image must contain
    one circular target; the locator returns its centre.  The function then
    fits ``r = A p + b``.  ``result.matrix`` is the 3x3 affine and
    ``result.robot_per_pixel`` is the reusable 2x2 displacement map.

    Args:
        images: Four or more grayscale/BGR images, all the same size.
        robot_points: Matching poses, each ``(x, y)`` or ``(x, y, z)``.
        locator: Circle locator; defaults to ``CircleLocator()``.
        principal_point: Camera centre in pixels; defaults to image centre.
        image_size: Optional expected ``(width, height)``; validated.
        calib_z: Optional height override; otherwise the mean Z of 3D poses.
        metadata: Extra JSON-serialisable metadata.

    Returns:
        The fitted calibration.  Soft quality issues set ``valid=False``
        and are listed in ``warnings``; hard input errors raise.

    Raises:
        CalibrationError: For fewer than four images, mismatched counts or
            sizes, a missing target, or a degenerate fit.
    """
    if isinstance(images, np.ndarray) and images.ndim == 2:
        raise CalibrationError(
            "images must be a sequence of images, not a single 2D image"
        )
    if len(images) != len(robot_points):
        raise CalibrationError("images and robot_points must be the same length")
    if len(images) < 4:
        raise CalibrationError("at least 4 images and 4 robot points are required")

    poses = [
        _as_pose(point, f"robot_points[{index}]")
        for index, point in enumerate(robot_points)
    ]
    frames: List[np.ndarray] = []
    expected_size: Optional[Tuple[int, int]] = None
    for index, image in enumerate(images):
        if not isinstance(image, np.ndarray):
            raise CalibrationError(f"images[{index}] must be a numpy array")
        if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] != 3):
            raise CalibrationError(
                f"images[{index}] must be 2D grayscale or 3-channel BGR"
            )
        height, width = int(image.shape[0]), int(image.shape[1])
        if expected_size is None:
            expected_size = (width, height)
        elif (width, height) != expected_size:
            raise CalibrationError(
                f"images[{index}] has size {(width, height)}; "
                f"expected {expected_size}"
            )
        frames.append(image)
    if image_size is not None:
        requested = (int(image_size[0]), int(image_size[1]))
        if requested != expected_size:
            raise CalibrationError(
                f"image_size {requested} does not match images {expected_size}"
            )

    locator = locator or CircleLocator()
    image_points: List[Tuple[float, float]] = []
    confidences: List[float] = []
    for index, image in enumerate(frames):
        detection = locator.find(image)
        if not detection.valid:
            raise CalibrationError(
                f"images[{index}]: target not found ({detection.reason})"
            )
        image_points.append(
            (float(detection.center[0]), float(detection.center[1]))
        )
        confidences.append(float(detection.confidence))

    z_values = [float(pose[2]) for pose in poses if pose.size >= 3]
    if calib_z is None and z_values:
        calib_z = float(np.mean(z_values))
    extra = dict(metadata or {})
    extra.setdefault("source", "images")
    extra.setdefault("num_images", len(frames))
    calibration = fit_image_to_robot(
        image_points,
        [pose[:2] for pose in poses],
        principal_point=principal_point,
        image_size=expected_size,
        calib_z=calib_z,
        metadata=extra,
        confidences=confidences,
    )
    if z_values and (max(z_values) - min(z_values)) > 1.0:
        _warn(calibration.warnings, "varying_calibration_z")
    return calibration


def capture_and_calibrate(
    robot: RobotLink,
    capture: Callable[[], np.ndarray],
    *,
    positions: Sequence[Sequence[float]],
    locator: Optional[CircleLocator] = None,
    principal_point: Optional[Sequence[float]] = None,
    settle_s: float = 0.15,
    on_event: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> ImageRobotCalibration:
    """Optional hardware driver: visit taught poses, capture, calibrate.

    Offline data should use :func:`calibrate_image_to_robot` instead.  This
    driver only visits explicit ``positions``; it never probes or searches
    for poses.

    Args:
        robot: Blocking robot interface.
        capture: Callable returning a fresh BGR/gray frame.
        positions: Taught poses, each ``(x, y)`` or ``(x, y, z)``; at least 4.
        locator: Circle locator; defaults to ``CircleLocator()``.
        principal_point: Camera centre in pixels; defaults to image centre.
        settle_s: Sleep after each move before capturing.
        on_event: Optional ``(kind, payload)`` callback (start/sample/fit).
        metadata: Extra metadata stored on the result.

    Returns:
        The fitted calibration.  Always inspect ``valid`` and ``warnings``
        before commanding a pick.

    Raises:
        CalibrationError: Fewer than four positions, a missing target at a
            taught pose, or a frame-size change.
    """
    if len(positions) < 4:
        raise CalibrationError("at least 4 taught positions are required")
    locator = locator or CircleLocator()
    start = _read_pose(robot)
    _, detection, image_size = _detect_frame(capture, locator)
    principal = (
        _pixel_center(image_size)
        if principal_point is None
        else (float(principal_point[0]), float(principal_point[1]))
    )
    _emit(on_event, "start", pose=start.tolist(),
          image_uv=list(detection.center) if detection.valid else None,
          confidence=detection.confidence, valid=detection.valid)

    samples: List[Tuple[np.ndarray, Tuple[float, float], float]] = []
    if detection.valid:
        samples.append((start.copy(), (float(detection.center[0]),
                                       float(detection.center[1])),
                        detection.confidence))
    default_z = float(start[2]) if start.size >= 3 else None
    for index, position in enumerate(positions):
        target = np.asarray(position, dtype=float).reshape(-1)
        if target.size < 2 or not np.all(np.isfinite(target[:2])):
            raise CalibrationError(
                f"positions[{index}] is not a finite (x, y[, z])"
            )
        if target.size < 3 and default_z is not None:
            target = np.append(target, default_z)
        actual, current, current_size = _move_and_detect(
            robot, capture, locator, target, settle_s
        )
        if current_size != image_size:
            raise CalibrationError("captured frames have different sizes")
        if not current.valid:
            raise CalibrationError(
                f"target not found at positions[{index}] ({current.reason})"
            )
        samples.append((actual.copy(), (float(current.center[0]),
                                        float(current.center[1])),
                        current.confidence))
        _emit(on_event, "sample", index=index, pose=actual.tolist(),
              image_uv=list(current.center), confidence=current.confidence)

    robot_points = np.asarray([item[0][:2] for item in samples], dtype=float)
    image_points = np.asarray([item[1] for item in samples], dtype=float)
    confidences = [float(item[2]) for item in samples]
    z_values = [float(item[0][2]) for item in samples if item[0].size >= 3]
    z_spread = (max(z_values) - min(z_values)) if z_values else 0.0
    calib_z = float(np.mean(z_values)) if z_values else None

    extra = dict(metadata or {})
    extra.setdefault("source", "capture_and_calibrate")
    extra.setdefault("z_spread_mm", float(z_spread))
    calibration = fit_image_to_robot(
        image_points,
        robot_points,
        principal_point=principal,
        image_size=image_size,
        calib_z=calib_z,
        metadata=extra,
        confidences=confidences,
    )
    if z_spread > 1.0:
        _warn(calibration.warnings, "varying_calibration_z")
    _emit(on_event, "fit", num_points=len(samples), valid=calibration.valid,
          warnings=list(calibration.warnings))
    return calibration


# ---------------------------------------------------------------------------
# Offset calibration driver
# ---------------------------------------------------------------------------


def calibrate_gripper_offset(
    robot: RobotLink,
    capture: Callable[[], np.ndarray],
    calibration: ImageRobotCalibration,
    *,
    place_pose: Optional[Sequence[float]] = None,
    locator: Optional[CircleLocator] = None,
    settle_s: float = 0.15,
    center_tolerance_px: float = DEFAULT_CENTER_TOLERANCE_PX,
    max_center_iterations: int = 2,
    require_valid_calibration: bool = True,
    on_event: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> CameraGripperOffset:
    """Camera-to-gripper offset calibration.

    Procedure (matches the manual workflow):

    1. The gripper has just released the target; call this function while
       the arm is still at that release pose (or pass ``place_pose``).
    2. Move the camera so the target is visible and call the function; it
       detects the target, moves the arm to put the target at the camera
       centre, and records that pose ``B``.
    3. ``offset = A_place - B``.  Add it to a camera-centred pose to put
       the gripper over the target.

    The target must already be visible in the current frame; the function
    does not search for it.

    Args:
        robot: Blocking robot interface.
        capture: Callable returning a fresh frame.
        calibration: A fitted :class:`ImageRobotCalibration`.
        place_pose: Release pose; defaults to the current pose (call while
            still at the release position).
        locator: Circle locator; defaults to ``CircleLocator()``.
        settle_s: Sleep after each move before capturing.
        center_tolerance_px: Accepted centring residual.
        max_center_iterations: Extra centring corrections.
        require_valid_calibration: Refuse to run on an invalid transform.
        on_event: Optional progress callback.

    Returns:
        The offset calibration; inspect ``valid`` and ``warnings``.

    Raises:
        CalibrationError: On an invalid transform, a missing target, or a
            lost target during centring.
    """
    if require_valid_calibration and not calibration.valid:
        raise CalibrationError(
            "image-to-robot calibration is not valid; run "
            "verify_image_to_robot() and fix the four-point calibration first"
        )
    locator = locator or CircleLocator()
    current = _read_pose(robot)
    release = current.copy() if place_pose is None else _as_pose(place_pose, "place_pose")
    _emit(on_event, "release", place_pose=release.tolist())

    image, first_detection, image_size = _detect_frame(capture, locator)
    if not first_detection.valid:
        raise CalibrationError(
            f"target not visible at the centring view ({first_detection.reason}); "
            "move the camera so the target is in the field of view"
        )
    if calibration.image_size is not None and image_size != calibration.image_size:
        raise CalibrationError(
            f"frame size {image_size} does not match calibration size "
            f"{calibration.image_size}"
        )
    target_uv = np.asarray(calibration.principal_point, dtype=float)
    _emit(on_event, "detect", pose=current.tolist(),
          image_uv=list(first_detection.center),
          confidence=first_detection.confidence)

    result = _servo_to_uv(
        robot, capture, locator, calibration.robot_per_pixel, target_uv,
        settle_s, max_iterations=max_center_iterations,
        tolerance_px=center_tolerance_px * 0.5,
    )
    if result is None:
        raise CalibrationError("target lost while centring the camera")
    center_pose, center_detection = result
    center_error = float(np.linalg.norm(
        np.asarray(center_detection.center, dtype=float) - target_uv
    ))
    _emit(on_event, "centred", pose=center_pose.tolist(),
          image_uv=list(center_detection.center), error_px=center_error)

    sample = OffsetSample(
        place_pose=(float(release[0]), float(release[1])),
        center_pose=(float(center_pose[0]), float(center_pose[1])),
        center_error_px=center_error,
        confidence=float(center_detection.confidence),
        camera_pose=(float(current[0]), float(current[1])),
        image_uv=(float(first_detection.center[0]), float(first_detection.center[1])),
        z_place=float(release[2]) if release.size >= 3 else None,
        z_camera=float(current[2]) if current.size >= 3 else None,
    )
    offset = CameraGripperOffset.from_samples(
        [sample], center_tolerance_px=center_tolerance_px
    )
    if (
        calibration.calib_z is not None
        and current.size >= 3
        and abs(float(current[2]) - float(calibration.calib_z)) > 1.0
    ):
        _warn(offset.warnings, "vision_z_differs_from_calibration")
    _emit(on_event, "offset", offset_xy=list(offset.offset_xy),
          valid=offset.valid, warnings=list(offset.warnings))
    return offset


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


@dataclass
class VerificationCheck:
    """One centring verification result."""

    target_uv: Tuple[float, float]
    residual_px: float
    residual_mm: float
    confidence: float
    valid: bool
    note: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "target_uv": [float(self.target_uv[0]), float(self.target_uv[1])],
            "residual_px": float(self.residual_px),
            "residual_mm": float(self.residual_mm),
            "confidence": float(self.confidence),
            "valid": bool(self.valid),
            "note": self.note,
        }


@dataclass
class VerificationReport:
    """Aggregated centring verification."""

    checks: List[VerificationCheck]
    tolerance_px: float
    passed: bool
    mean_error_px: float
    max_error_px: float
    mean_error_mm: float
    principal_point: Tuple[float, float]
    note: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "passed": bool(self.passed),
            "tolerance_px": float(self.tolerance_px),
            "mean_error_px": float(self.mean_error_px),
            "max_error_px": float(self.max_error_px),
            "mean_error_mm": float(self.mean_error_mm),
            "principal_point": [float(self.principal_point[0]),
                                float(self.principal_point[1])],
            "note": self.note,
            "checks": [check.as_dict() for check in self.checks],
        }


def verify_image_to_robot(
    robot: RobotLink,
    capture: Callable[[], np.ndarray],
    calibration: ImageRobotCalibration,
    *,
    locator: Optional[CircleLocator] = None,
    checks: int = 3,
    tolerance_px: float = DEFAULT_VERIFY_TOLERANCE_PX,
    settle_s: float = 0.15,
    on_event: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> VerificationReport:
    """Verify centring accuracy at several image positions.

    For each check the target is first placed at a chosen image point, then
    a single correction ``Δr = A(c − p)`` is applied and the residual from
    the principal point is measured.  This is exactly the user-facing
    "move to the computed pose, the target should be in the centre" test.

    A systematic mean residual that is stable across checks is normal when
    the principal point is not the image centre; the spread is the part
    that indicates a bad affine.

    Args:
        robot: Blocking robot interface.
        capture: Callable returning a fresh frame.
        calibration: Calibration to verify.
        locator: Circle locator; defaults to ``CircleLocator()``.
        checks: Number of positions (1-4).
        tolerance_px: Maximum accepted residual.
        settle_s: Sleep after each move.
        on_event: Optional progress callback.

    Returns:
        A :class:`VerificationReport`; inspect ``passed`` and per-check data.
    """
    locator = locator or CircleLocator()
    principal = np.asarray(calibration.principal_point, dtype=float)
    image_size = calibration.image_size
    if image_size is None:
        _, _, image_size = _detect_frame(capture, locator)
    width, height = image_size
    checks = max(1, min(4, int(checks)))
    candidates = [principal]
    if checks >= 2:
        candidates.append(np.array([0.22 * width, 0.78 * height]))
    if checks >= 3:
        candidates.append(np.array([0.78 * width, 0.22 * height]))
    if checks >= 4:
        candidates.append(np.array([0.78 * width, 0.78 * height]))

    results: List[VerificationCheck] = []
    for index, target in enumerate(candidates[:checks]):
        target = np.asarray(target, dtype=float)
        try:
            pose = _read_pose(robot)
            detection = locator.find(capture())
            if not detection.valid:
                raise CalibrationError(f"target not visible ({detection.reason})")
            if float(np.linalg.norm(target - np.asarray(detection.center))) > 1.0:
                move = calibration.servo_delta(detection.center, target)
                pose = pose.copy()
                pose[:2] += move
                _move_robot(robot, pose, settle_s)
                detection = locator.find(capture())
                if not detection.valid:
                    raise CalibrationError("target lost while moving to the check point")
            correction = calibration.servo_delta(detection.center)
            pose = _read_pose(robot).copy()
            pose[:2] += correction
            _move_robot(robot, pose, settle_s)
            final = locator.find(capture())
            if not final.valid:
                raise CalibrationError("target lost after the correction move")
            error = np.asarray(final.center, dtype=float) - principal
            error_px = float(np.linalg.norm(error))
            error_mm = float(np.linalg.norm(calibration.robot_per_pixel @ error))
            results.append(
                VerificationCheck(
                    target_uv=(float(principal[0]), float(principal[1])),
                    residual_px=error_px,
                    residual_mm=error_mm,
                    confidence=float(final.confidence),
                    valid=True,
                )
            )
            _emit(on_event, "check", index=index, residual_px=error_px,
                  residual_mm=error_mm, valid=True)
        except CalibrationError as exc:
            results.append(
                VerificationCheck(
                    target_uv=(float(principal[0]), float(principal[1])),
                    residual_px=float("inf"),
                    residual_mm=float("inf"),
                    confidence=0.0,
                    valid=False,
                    note=str(exc),
                )
            )
            _emit(on_event, "check", index=index, valid=False, note=str(exc))

    valid_checks = [item for item in results if item.valid]
    if valid_checks:
        errors = np.asarray([item.residual_px for item in valid_checks])
        mean_error = float(np.mean(errors))
        max_error = float(np.max(errors))
        mean_error_mm = float(np.mean([item.residual_mm for item in valid_checks]))
        spread = float(np.std(errors))
    else:
        mean_error = max_error = mean_error_mm = float("inf")
        spread = float("inf")
    passed = bool(valid_checks) and len(valid_checks) == len(results) \
        and max_error <= tolerance_px
    note = ""
    if valid_checks and mean_error > 1.0 and spread < 0.5:
        note = (
            "stable systematic offset; the principal point is probably not "
            "at the assumed image centre"
        )
    return VerificationReport(
        checks=results,
        tolerance_px=float(tolerance_px),
        passed=passed,
        mean_error_px=mean_error,
        max_error_px=max_error,
        mean_error_mm=mean_error_mm,
        principal_point=(float(principal[0]), float(principal[1])),
        note=note,
    )


def verify_pick(
    robot: RobotLink,
    capture: Callable[[], np.ndarray],
    recipe: RobotPickRecipe,
    *,
    locator: Optional[CircleLocator] = None,
    tolerance_px: float = DEFAULT_VERIFY_TOLERANCE_PX,
    settle_s: float = 0.15,
    check_grasp_view: bool = False,
    on_event: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """End-to-end check: centre the target, then move to the grasp pose.

    With ``check_grasp_view=True`` the target is re-detected at the grasp
    pose and compared with :meth:`RobotPickRecipe.expected_uv_at_grasp`.
    That check only works when the gripper does not occlude the target.

    Returns:
        A dictionary with ``passed``, the centring report and the planned
        grasp pose.  ``grasp_view`` is ``None`` when not requested.
    """
    if recipe.gripper_offset is None:
        raise CalibrationError("gripper offset has not been calibrated")
    locator = locator or CircleLocator()
    report = verify_image_to_robot(
        robot, capture, recipe.image_robot, locator=locator, checks=1,
        tolerance_px=tolerance_px, settle_s=settle_s, on_event=on_event,
    )
    detection = locator.find(capture())
    if not detection.valid:
        return {
            "passed": False,
            "note": "target not visible before the grasp move",
            "center_check": report.as_dict(),
            "grasp_view": None,
        }
    current = _read_pose(robot)
    grasp = recipe.grasp_pose(current, detection.center)
    _move_robot(robot, grasp, settle_s)
    _emit(on_event, "grasp", pose=grasp.tolist())
    grasp_view: Optional[Dict[str, Any]] = None
    if check_grasp_view:
        expected = np.asarray(recipe.expected_uv_at_grasp(), dtype=float)
        after = locator.find(capture())
        if after.valid:
            error_px = float(np.linalg.norm(np.asarray(after.center) - expected))
            error_mm = float(np.linalg.norm(
                recipe.image_robot.robot_per_pixel
                @ (np.asarray(after.center, dtype=float) - expected)
            ))
            grasp_view = {
                "valid": True,
                "expected_uv": [float(expected[0]), float(expected[1])],
                "detected_uv": [float(after.center[0]), float(after.center[1])],
                "residual_px": error_px,
                "residual_mm": error_mm,
                "confidence": float(after.confidence),
            }
            if error_px > tolerance_px:
                grasp_view["valid"] = False
        else:
            grasp_view = {
                "valid": False,
                "note": "target not visible at the grasp pose (gripper may occlude)",
            }
    passed = bool(report.passed)
    if grasp_view is not None and grasp_view.get("valid") is False:
        passed = False
    return {
        "passed": passed,
        "center_check": report.as_dict(),
        "grasp_pose": [float(value) for value in grasp],
        "grasp_view": grasp_view,
        "note": "",
    }
