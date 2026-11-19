"""
Geometric transform utilities for rotated box crop/straighten and coordinate mapping.

These are pure math functions with no GUI dependency — only numpy + OpenCV.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from measure.computation_types import (
    Mapping2D,
    RotatedBox,
    Space,
    contract_error,
    make_mapping,
    require_finite,
)

#: Output ``(width, height)`` in whole pixels.
TraceWh = Tuple[float, float]


def compute_rotated_box_corners(
    center: Tuple[float, float],
    size: Tuple[float, float],
    angle_deg: float,
) -> np.ndarray:
    """
    Compute the four corner points of a rotated rectangle.

    Args:
        center: (row, col) center of the box in image coordinates.
        size: (height, width) of the unrotated box.
        angle_deg: Rotation angle in degrees (counter-clockwise positive).

    Returns:
        (4, 2) array of corner coordinates (row, col), ordered
        top-left, top-right, bottom-right, bottom-left of the unrotated box.
    """
    cy, cx = center
    h, w = size
    half_h = h / 2.0
    half_w = w / 2.0

    # Corners in local (unrotated) coordinates, centered at origin
    corners_local = np.array(
        [
            [-half_h, -half_w],  # top-left
            [-half_h, half_w],  # top-right
            [half_h, half_w],  # bottom-right
            [half_h, -half_w],  # bottom-left
        ],
        dtype=np.float64,
    )

    # Rotation matrix (counter-clockwise)
    theta = np.deg2rad(angle_deg)
    cos_a = np.cos(theta)
    sin_a = np.sin(theta)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)

    # Rotate and translate
    corners_rotated = corners_local @ R.T
    corners_rotated[:, 0] += cy
    corners_rotated[:, 1] += cx

    return corners_rotated


def crop_and_straighten(
    image: np.ndarray,
    center: Tuple[float, float],
    size: Tuple[float, float],
    angle_deg: float,
    interpolation: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crop a rotated rectangular region from an image and straighten (deskew) it.

    The output is an upright image of size (size[0], size[1]) containing the
    rotated box content.

    Args:
        image: Input image (grayscale or BGR, uint8).
        center: (row, col) center of the rotated box in image coordinates.
        size: (height, width) of the straightened output.
        angle_deg: Rotation angle of the box in degrees (counter-clockwise positive).
        interpolation: OpenCV interpolation flag (default cv2.INTER_LINEAR).
        border_mode: OpenCV border mode (default cv2.BORDER_CONSTANT).
        border_value: Border fill value (default 0 = black).

    Returns:
        straightened: Upright image of shape (size[0], size[1], ...).
        M_inv: 3x3 matrix that maps **patch coordinates to image coordinates**
               (the inverse of the ``warpAffine`` matrix above).  Two mappings in
               this module read the same matrix differently, so be explicit:

               * ``map_point_via_affine(patch_point, M_inv[:2, :])`` — matches
                 this return value; it inverts internally.
               * ``cv2.warpAffine`` / ``AlignResult.M_inv`` callers that want
                 image -> patch must invert first.

               Verified against a taught box: patch centre maps to the box
               centre.
    """
    cy, cx = center
    h, w = size

    # Build the forward affine transform: maps straightened pixel coords
    # to original image coords.
    # In the straightened image, pixel (r, c) corresponds to a point at:
    #   local_row = r - h/2
    #   local_col = c - w/2
    # We then rotate by +angle_deg and translate to (cy, cx) in the original.
    theta = np.deg2rad(angle_deg)
    cos_a = np.cos(theta)
    sin_a = np.sin(theta)

    # Forward: straightened -> original
    # [col_orig]   [cos_a  -sin_a] [local_col]   [cx]
    # [row_orig] = [sin_a   cos_a] [local_row] + [cy]
    # which is: orig = R @ (straightened_pixel - size/2) + center
    M_forward = np.array(
        [
            [cos_a, -sin_a, cx - cos_a * (w / 2.0) + sin_a * (h / 2.0)],
            [sin_a, cos_a, cy - sin_a * (w / 2.0) - cos_a * (h / 2.0)],
        ],
        dtype=np.float64,
    )

    # Inverse: original -> straightened (used by warpAffine)
    M_inv_2x3 = cv2.invertAffineTransform(M_forward)

    # Build 3x3 inverse for point mapping
    M_inv_3x3 = np.eye(3, dtype=np.float64)
    M_inv_3x3[:2, :] = M_inv_2x3

    # Warp
    straightened = cv2.warpAffine(
        image,
        M_inv_2x3,
        (int(w), int(h)),
        flags=interpolation,
        borderMode=border_mode,
        borderValue=border_value,
    )

    return straightened, M_inv_3x3


def map_point_to_original(
    point_in_straightened: Tuple[float, float],
    M_inv: np.ndarray,
) -> Tuple[float, float]:
    """
    Map a point from straightened image coordinates back to original image coordinates.

    Args:
        point_in_straightened: (row, col) in the straightened image.
        M_inv: 3x3 inverse affine matrix from crop_and_straighten().

    Returns:
        (row, col) in the original image.
    """
    r, c = point_in_straightened
    pt = np.array([c, r, 1.0], dtype=np.float64)
    # M_inv maps original -> straightened, we need the inverse (forward)
    # Solve: M_inv @ [orig_c, orig_r, 1]^T = [c, r, 1]^T
    M_forward = np.linalg.inv(M_inv)
    result = M_forward @ pt
    return (result[1], result[0])


def build_forward_transform(
    center: Tuple[float, float],
    size: Tuple[float, float],
    angle_deg: float,
) -> np.ndarray:
    """
    Build the forward 3x3 affine transform: straightened -> original.

    Args:
        center: (row, col) center of the rotated box.
        size: (height, width) of the straightened output.
        angle_deg: Rotation angle in degrees (counter-clockwise positive).

    Returns:
        3x3 forward affine matrix.
    """
    cy, cx = center
    h, w = size

    theta = np.deg2rad(angle_deg)
    cos_a = np.cos(theta)
    sin_a = np.sin(theta)

    M_forward = np.eye(3, dtype=np.float64)
    M_forward[0, 0] = cos_a
    M_forward[0, 1] = -sin_a
    M_forward[0, 2] = cx - cos_a * (w / 2.0) + sin_a * (h / 2.0)
    M_forward[1, 0] = sin_a
    M_forward[1, 1] = cos_a
    M_forward[1, 2] = cy - sin_a * (w / 2.0) - cos_a * (h / 2.0)

    return M_forward


def build_inverse_transform(
    center: Tuple[float, float],
    size: Tuple[float, float],
    angle_deg: float,
) -> np.ndarray:
    """
    Build the inverse 3x3 affine transform: original -> straightened.

    Args:
        center: (row, col) center of the rotated box.
        size: (height, width) of the straightened output.
        angle_deg: Rotation angle in degrees (counter-clockwise positive).

    Returns:
        3x3 inverse affine matrix.
    """
    M_forward = build_forward_transform(center, size, angle_deg)
    return np.linalg.inv(M_forward)


def map_point_via_affine(
    point_in_patch: Tuple[float, float],
    M_inv: np.ndarray,
) -> Tuple[float, float]:
    """
    Map a point from straightened-patch coordinates back to original
    image coordinates using a 2x3 affine matrix.

    Args:
        point_in_patch: (row, col) in the straightened patch.
        M_inv: 2x3 affine matrix mapping **original → patch**
               (i.e. the matrix that was passed to ``cv2.warpAffine``).

    Returns:
        (row, col) in the original image.
    """
    r, c = point_in_patch
    M_forward = cv2.invertAffineTransform(M_inv)
    orig_col = (
        M_forward[0, 0] * c + M_forward[0, 1] * r + M_forward[0, 2]
    )
    orig_row = (
        M_forward[1, 0] * c + M_forward[1, 1] * r + M_forward[1, 2]
    )
    return (orig_row, orig_col)



# ===========================================================================
# Typed resampling contract
# ===========================================================================
#
# ``crop_and_straighten`` keeps its historical signature for existing callers;
# these entries expose the same geometry with an explicit mapping and a validity
# mask instead of silently trusting replicated border pixels.

INTERPOLATIONS = {"nearest", "linear", "cubic"}
BORDER_MODES = {"constant", "replicate", "reflect"}

_CV_INTERPOLATION = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
}
_CV_BORDER = {
    "constant": cv2.BORDER_CONSTANT,
    "replicate": cv2.BORDER_REPLICATE,
    "reflect": cv2.BORDER_REFLECT,
}


@dataclass(frozen=True)
class ResampleOptions:
    """Interpolation and border policy for one resampling operation."""

    interpolation: str = "linear"
    border_mode: str = "constant"
    border_value: int = 0

    def __post_init__(self) -> None:
        if self.interpolation not in INTERPOLATIONS:
            raise contract_error("contract.invalid_parameter",
                                 f"unknown interpolation {self.interpolation!r}",
                                 "resample.interpolation")
        if self.border_mode not in BORDER_MODES:
            raise contract_error("contract.invalid_parameter",
                                 f"unknown border mode {self.border_mode!r}",
                                 "resample.border_mode")
        if not isinstance(self.border_value, (int, np.integer)):
            raise contract_error("contract.invalid_parameter", "border_value must be an integer",
                                 "resample.border_value")


@dataclass(frozen=True)
class ResampleResult:
    """Resampled pixels, the source-to-patch mapping and the real-pixel mask."""

    pixels: np.ndarray
    mapping: Mapping2D
    valid_mask: np.ndarray


def _resample(image: np.ndarray, matrix: np.ndarray, size_wh: TraceWh,
              options: ResampleOptions, *, source: Space, destination: Space) -> ResampleResult:
    """Warp with a source-to-patch matrix and compute which output pixels are real."""
    width, height = int(size_wh[0]), int(size_wh[1])
    if width <= 0 or height <= 0:
        raise contract_error("contract.invalid_parameter", "output size must be positive",
                             "size_wh")
    mapping = make_mapping(matrix, source=source, destination=destination, kind="affine")
    pixels = cv2.warpAffine(
        image, np.asarray(matrix, dtype=np.float64)[:2, :], (width, height),
        flags=_CV_INTERPOLATION[options.interpolation],
        borderMode=_CV_BORDER[options.border_mode],
        borderValue=options.border_value,
    )
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float64),
                                 np.arange(height, dtype=np.float64))
    homogeneous = np.stack([grid_x, grid_y, np.ones_like(grid_x)], axis=-1)
    source_points = homogeneous @ np.asarray(mapping.inverse_matrix, dtype=np.float64).T
    source_points = source_points[..., :2] / source_points[..., 2:3]
    image_height, image_width = image.shape[:2]
    valid_mask = (
        (source_points[..., 0] >= -0.5) & (source_points[..., 0] <= image_width - 0.5)
        & (source_points[..., 1] >= -0.5) & (source_points[..., 1] <= image_height - 0.5)
    )
    return ResampleResult(pixels=pixels, mapping=mapping, valid_mask=valid_mask)


def rotated_patch_matrix(center: Tuple[float, float], size_wh: Tuple[float, float],
                         angle_rad: float, *, origin: str = "continuous") -> np.ndarray:
    """Return the source-to-patch matrix of a rotated rectangle.

    Args:
        center: Rectangle centre in source xy.
        size_wh: Full ``(width, height)`` in whole pixels.
        angle_rad: Clockwise rotation in radians.
        origin: ``continuous`` places the patch origin at ``size / 2`` (the
            sampling convention), ``pixel_center`` at ``(size - 1) / 2`` (the
            exact pixel-centre convention used by image cropping).

    Returns:
        A 3x3 invertible source-to-patch transform.
    """
    width, height = float(size_wh[0]), float(size_wh[1])
    if origin == "pixel_center":
        offset_x, offset_y = (width - 1.0) / 2.0, (height - 1.0) / 2.0
    elif origin == "continuous":
        offset_x, offset_y = width / 2.0, height / 2.0
    else:
        raise contract_error("contract.invalid_parameter",
                             f"unknown origin policy {origin!r}", "origin")
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    cx, cy = center
    return np.array([
        [cos_a, sin_a, offset_x - (cos_a * cx + sin_a * cy)],
        [-sin_a, cos_a, offset_y - (-sin_a * cx + cos_a * cy)],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def crop_region(
    image: np.ndarray, box: RotatedBox, *, source: Space, destination: Space,
    options: Optional[ResampleOptions] = None,
) -> ResampleResult:
    """Straighten one rotated box with the continuous ``size/2`` local origin.

    Args:
        image: ``uint8`` source image.
        box: Rotated box in ``source``; ``size_wh`` are the full output dimensions.
        source: Space of the input image.
        destination: Space of the produced patch.
        options: Interpolation and border policy.

    Returns:
        The straightened patch, its source-to-patch mapping and validity mask.
    """
    options = options or ResampleOptions()
    width, height = box.size_wh
    matrix = rotated_patch_matrix(box.center, box.size_wh, box.angle_rad)
    return _resample(image, matrix, (width, height), options,
                     source=source, destination=destination)


def warp_mapped(
    image: np.ndarray, mapping: Mapping2D, size_wh: TraceWh, *,
    options: Optional[ResampleOptions] = None,
) -> ResampleResult:
    """Warp an image through an existing mapping.

    Args:
        image: ``uint8`` image in ``mapping.source``.
        mapping: Explicit source-to-destination mapping.
        size_wh: Output ``(width, height)`` in whole pixels.
        options: Interpolation and border policy.

    Returns:
        The warped pixels, the same mapping and the validity mask of real pixels.
    """
    options = options or ResampleOptions()
    return _resample(image, mapping.matrix, size_wh, options,
                     source=mapping.source, destination=mapping.destination)


def legacy_roi_to_box(row: float, col: float, width: float, height: float,
                      angle_deg: float) -> RotatedBox:
    """Convert a legacy row/column ROI into a typed rotated box.

    Args:
        row: ROI centre row.
        col: ROI centre column.
        width: Full width in pixels.
        height: Full height in pixels.
        angle_deg: Rotation in degrees, clockwise positive.

    Returns:
        The equivalent :class:`RotatedBox` in xy coordinates.
    """
    require_finite((row, col, width, height, angle_deg), "legacy_roi")
    return RotatedBox(center=(float(col), float(row)),
                      size_wh=(float(width), float(height)),
                      angle_rad=math.radians(float(angle_deg)))
