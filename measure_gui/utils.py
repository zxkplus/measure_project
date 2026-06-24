"""
Geometric transform utilities for rotated box crop/straighten and coordinate mapping.

These are pure math functions with no GUI dependency — only numpy + OpenCV.
"""

from typing import Tuple

import cv2
import numpy as np


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
        M_inv: 3x3 inverse affine transform matrix that maps a point in the
               straightened image back to the original image coordinates.
               Use with map_point_to_original().
    """
    cy, cx = center
    h, w = size

    # Build the forward affine transform: maps straightened pixel coords
    # to original image coords.
    # In the straightened image, pixel (r, c) corresponds to a point at:
    #   local_row = r - h/2
    #   local_col = c - w/2
    # We then rotate by +angle_deg and translate to (cy, cx) in the original.
    print(f"angle_deg: {angle_deg}")
    theta = np.deg2rad(-angle_deg)
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
    cv2.imwrite("original.png", image)
    cv2.imwrite("straightened.png", straightened)


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


def cv2_to_pil(image: np.ndarray):
    """Convert an OpenCV BGR/grayscale image to PIL RGB Image."""
    from PIL import Image

    if len(image.shape) == 2:
        return Image.fromarray(image, mode="L").convert("RGB")
    else:
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def cv2_to_tk(image: np.ndarray):
    """Convert an OpenCV BGR/grayscale image to Tkinter PhotoImage."""
    from PIL import Image, ImageTk

    pil_img = cv2_to_pil(image)
    return ImageTk.PhotoImage(pil_img)
