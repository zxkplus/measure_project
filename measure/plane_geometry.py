"""Plane geometry: board poses, plane-to-pixel mappings, canvases and stitching.

All matrices use the column-vector convention ``p_destination ~ H @ p_source``.
A plane point is ``(X, Y, 0)`` in the board's physical unit; "corrected" pixels
are the pinhole plane after distortion removal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Tuple

import cv2
import numpy as np

from measure.computation_types import (
    Array,
    Mapping2D,
    Observation,
    Space,
    contract_error,
    invalid,
    make_mapping,
    observed,
    require_finite,
)
from measure.measure_calibration import (
    BoardObservation,
    CameraModel,
    Pose3D,
    validate_camera_model,
    validate_pose,
)

#: Physical unit factors relative to millimetres.
UNIT_MM = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "um": 0.001}


@dataclass(frozen=True)
class PlaneView:
    """One camera observing a physical plane under a known pose."""

    camera: CameraModel
    board_to_camera: Pose3D
    plane_space: Space
    plane_to_corrected: Mapping2D
    corrected_to_plane: Mapping2D
    image_size_wh: Tuple[int, int]


@dataclass(frozen=True)
class PlaneCanvas:
    """A quantised physical canvas covering every supplied view."""

    size_wh: Tuple[int, int]
    pixel_size: float
    origin_xy: Tuple[float, float]
    bounds_xyxy: Tuple[float, float, float, float]
    canvas_space: Space
    plane_space: Space


@dataclass(frozen=True)
class StitchOptions:
    """Blend policy for overlapping canvas coverage."""

    blend: str = "average"

    def __post_init__(self) -> None:
        if self.blend not in ("average", "first", "last"):
            raise contract_error("contract.invalid_parameter",
                                 f"unknown blend {self.blend!r}", "stitch.blend")


@dataclass(frozen=True)
class StitchResult:
    """Canvas pixels, per-view validity masks and per-pixel coverage."""

    pixels: Array
    valid_masks: Tuple[Array, ...]
    coverage: Array
    canvas: PlaneCanvas


@dataclass(frozen=True)
class ProjectionResult:
    """Projected pixel positions plus explicit in-image flags."""

    points_xy: Array
    inside_image: Array


def _homography(pose: Pose3D, camera: CameraModel) -> np.ndarray:
    """Return the plane-to-corrected-pixel homography for one pose."""
    rotation = np.asarray(pose.rotation, dtype=np.float64)
    translation = np.asarray(pose.translation, dtype=np.float64).reshape(3)
    matrix = np.asarray(camera.matrix, dtype=np.float64)
    return matrix @ np.column_stack([rotation[:, 0], rotation[:, 1], translation])


def _require_unit(unit: str) -> float:
    """Return the millimetre factor of a physical unit."""
    if unit not in UNIT_MM:
        raise contract_error("contract.invalid_parameter", f"unsupported unit {unit!r}", "unit")
    return UNIT_MM[unit]


def build_plane_view(camera: CameraModel, pose: Pose3D, *, plane_space: Space) -> PlaneView:
    """Build the plane/pixel mappings of one camera observing a plane.

    Args:
        camera: Validated pinhole camera.
        pose: Board-to-camera rigid pose.
        plane_space: Space of the physical plane.

    Returns:
        A view whose two mappings are exact inverses.
    """
    validate_camera_model(camera)
    validate_pose(pose)
    homography = _homography(pose, camera)
    plane_to_corrected = make_mapping(
        homography, source=plane_space, destination=camera.image_space, kind="projective")
    corrected_to_plane = make_mapping(
        plane_to_corrected.inverse_matrix, source=camera.image_space,
        destination=plane_space, kind="projective")
    return PlaneView(camera=camera, board_to_camera=pose, plane_space=plane_space,
                     plane_to_corrected=plane_to_corrected,
                     corrected_to_plane=corrected_to_plane,
                     image_size_wh=camera.image_size_wh)


def solve_plane_pose(view: BoardObservation, camera: CameraModel) -> Observation[Pose3D]:
    """Solve a board-to-camera pose from one ordered observation.

    Args:
        view: Board observation with image/object correspondences.
        camera: Calibrated camera.

    Returns:
        The solved pose, or ``solver_failed``.
    """
    validate_camera_model(camera)
    object_points = np.asarray(view.object_points_xyz, dtype=np.float64)
    image_points = np.asarray(view.image_points_xy, dtype=np.float64)
    solved, rvec, tvec = cv2.solvePnP(
        object_points, image_points, np.asarray(camera.matrix, dtype=np.float64),
        np.asarray(camera.distortion, dtype=np.float64),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not solved:
        return invalid(camera.image_space, "solver_failed")
    projected, _ = cv2.projectPoints(object_points, rvec, tvec,
                                     np.asarray(camera.matrix, dtype=np.float64),
                                     np.asarray(camera.distortion, dtype=np.float64))
    error = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
    rms = float(np.sqrt(np.mean(error ** 2)))
    pose = Pose3D(source_frame=view.object_space.frame,
                  destination_frame=camera.image_space.frame, unit=view.object_space.unit,
                  rotation=cv2.Rodrigues(rvec)[0],
                  translation=np.asarray(tvec, dtype=np.float64).reshape(3))
    return observed(camera.image_space, pose, {"rms_error_px": rms})


def _plane_bounds(views: Sequence[PlaneView]) -> Tuple[float, float, float, float]:
    """Return ``(x0, y0, x1, y1)`` covering every view's image footprint."""
    corners = []
    for view in views:
        width, height = view.image_size_wh
        pixels = np.array([[0.0, 0.0], [width - 1.0, 0.0],
                           [width - 1.0, height - 1.0], [0.0, height - 1.0]])
        corners.append(view.corrected_to_plane.apply(pixels))
    stacked = np.vstack(corners)
    return (float(stacked[:, 0].min()), float(stacked[:, 1].min()),
            float(stacked[:, 0].max()), float(stacked[:, 1].max()))


def build_plane_canvas(views: Sequence[PlaneView], *, pixel_size: float,
                       canvas_space: Space) -> Observation[PlaneCanvas]:
    """Build a quantised canvas covering every view.

    Args:
        views: Views to cover.
        pixel_size: Millimetres per canvas pixel.
        canvas_space: Space of the canvas.

    Returns:
        A valid canvas, or ``insufficient_views`` when none are supplied.
    """
    require_finite(pixel_size, "pixel_size")
    if pixel_size <= 0.0:
        raise contract_error("contract.invalid_parameter", "pixel_size must be positive",
                             "pixel_size")
    views = tuple(views)
    if not views:
        return invalid(canvas_space, "insufficient_views")
    x0, y0, x1, y1 = _plane_bounds(views)
    width = int(math.ceil((x1 - x0) / pixel_size)) + 1
    height = int(math.ceil((y1 - y0) / pixel_size)) + 1
    canvas = PlaneCanvas(
        size_wh=(width, height), pixel_size=float(pixel_size),
        origin_xy=(x0, y0), bounds_xyxy=(x0, y0, x1, y1),
        canvas_space=canvas_space, plane_space=views[0].plane_space,
    )
    return observed(canvas_space, canvas, {"view_count": len(views)})


def _canvas_from_plane(canvas: PlaneCanvas) -> np.ndarray:
    """Return the plane-to-canvas matrix."""
    origin_x, origin_y = canvas.origin_xy
    size = canvas.pixel_size
    return np.array([[1.0 / size, 0.0, -origin_x / size],
                     [0.0, 1.0 / size, -origin_y / size],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def stitch_plane_views(
    images: Sequence[Array], views: Sequence[PlaneView], canvas: PlaneCanvas, *,
    options: Optional[StitchOptions] = None,
) -> Observation[StitchResult]:
    """Warp every view onto the canvas and blend the overlapping coverage.

    Args:
        images: One image per view, in the same order.
        views: Plane views describing each camera.
        canvas: Target canvas.
        options: Blend policy.

    Returns:
        Blended canvas pixels, per-view masks and coverage counts.
    """
    options = options or StitchOptions()
    images = tuple(images)
    views = tuple(views)
    if len(images) != len(views):
        raise contract_error("contract.invalid_shape",
                             "one image per view is required", "images")
    width, height = canvas.size_wh
    canvas_from_plane = _canvas_from_plane(canvas)
    accumulator = np.zeros((height, width), dtype=np.float64)
    coverage = np.zeros((height, width), dtype=np.int32)
    masks = []
    chosen = np.zeros((height, width), dtype=np.float64)
    for image, view in zip(images, views):
        matrix = canvas_from_plane @ np.asarray(view.plane_to_corrected.inverse_matrix)
        warped = cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask = cv2.warpPerspective(np.ones_like(image, dtype=np.uint8), matrix,
                                   (width, height), flags=cv2.INTER_NEAREST,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
        masks.append(mask)
        accumulator += np.where(mask, warped.astype(np.float64), 0.0)
        if options.blend == "first":
            chosen = np.where(mask & (coverage == 0), warped.astype(np.float64), chosen)
        elif options.blend == "last":
            chosen = np.where(mask, warped.astype(np.float64), chosen)
        coverage += mask.astype(np.int32)
    if options.blend == "average":
        pixels = np.where(coverage > 0, accumulator / np.maximum(coverage, 1), 0.0)
    else:
        pixels = chosen
    result = StitchResult(pixels=pixels.astype(np.float64), valid_masks=tuple(masks),
                          coverage=coverage, canvas=canvas)
    return observed(canvas.canvas_space, result, {"view_count": len(views)})


def project_camera_points(points_xyz: Array, camera: CameraModel, *, source_frame: str,
                          unit: str) -> Observation[ProjectionResult]:
    """Project known 3-D camera-frame points into the image.

    Args:
        points_xyz: ``(N, 3)`` points in ``source_frame``.
        camera: Calibrated camera.
        source_frame: Frame of the points; must name this camera.
        unit: Physical unit of the points.

    Returns:
        Pixel positions and in-image flags, or ``behind_camera`` when any point
        is at or behind the principal plane.
    """
    factor = _require_unit(unit)
    frame = camera.image_space.frame
    if not (source_frame == frame or frame.startswith(source_frame)
            or source_frame.startswith(frame.split("/")[0])):
        raise contract_error("contract.space_mismatch",
                             "source_frame does not name this camera", "source_frame")
    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise contract_error("contract.invalid_shape", "points must be (N, 3)", "points")
    require_finite(points, "points")
    scaled = points * factor
    if bool((scaled[:, 2] <= 0.0).any()):
        return invalid(camera.image_space, "behind_camera")
    projected, _ = cv2.projectPoints(
        scaled, np.zeros(3), np.zeros(3), np.asarray(camera.matrix, dtype=np.float64),
        np.asarray(camera.distortion, dtype=np.float64),
    )
    pixels = projected.reshape(-1, 2)
    width, height = camera.image_size_wh
    inside = ((pixels[:, 0] >= 0.0) & (pixels[:, 0] <= width - 1.0)
              & (pixels[:, 1] >= 0.0) & (pixels[:, 1] <= height - 1.0))
    return observed(camera.image_space, ProjectionResult(points_xy=pixels, inside_image=inside))


def rebase_plane(views: Sequence[PlaneView], origin_xy: Tuple[float, float], *,
                 destination_plane: Space) -> Tuple[PlaneView, ...]:
    """Translate every view's plane origin without mutating the inputs.

    Args:
        views: Source views.
        origin_xy: New plane origin expressed in the old plane.
        destination_plane: Space of the rebased plane.

    Returns:
        New views whose mappings and poses are shifted accordingly.
    """
    require_finite(origin_xy, "origin_xy")
    shift = np.asarray([origin_xy[0], origin_xy[1], 0.0], dtype=np.float64)
    rebased = []
    for view in views:
        pose = Pose3D(
            source_frame=destination_plane.frame,
            destination_frame=view.board_to_camera.destination_frame,
            unit=view.board_to_camera.unit,
            rotation=np.array(view.board_to_camera.rotation, dtype=np.float64, copy=True),
            translation=np.asarray(view.board_to_camera.translation, dtype=np.float64) + shift,
        )
        rebased.append(build_plane_view(view.camera, pose, plane_space=destination_plane))
    return tuple(rebased)