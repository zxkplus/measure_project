"""
Coordinate-transform helpers shared by all measurement modules.

Extracts the rotated-rectangle corner calculation that was duplicated
in 7+ locations across ``measure1D``, ``measure2D``, and the GUI.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Tuple


def compute_rotated_rect_corners(
    center_col: float,
    center_row: float,
    half_length1: float,
    half_length2: float,
    angle_rad: float,
) -> np.ndarray:
    """Compute the four corner points of a rotated rectangle.

    The rectangle is centred at (*center_col*, *center_row*), has
    half‑extents *half_length1* along the primary direction and
    *half_length2* perpendicular to it, and is rotated by *angle_rad*
    radians (0 = right, π/2 = down — matching ``measure1D`` convention).

    Returns (4, 2) ``np.int32`` array ordered **top-left, top-right,
    bottom-right, bottom-left** (clockwise from the unrotated top-left).
    """
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    corners = np.array(
        [
            [center_col - half_length1 * cos_a + half_length2 * sin_a,
             center_row - half_length1 * sin_a - half_length2 * cos_a],
            [center_col + half_length1 * cos_a + half_length2 * sin_a,
             center_row + half_length1 * sin_a - half_length2 * cos_a],
            [center_col + half_length1 * cos_a - half_length2 * sin_a,
             center_row + half_length1 * sin_a + half_length2 * cos_a],
            [center_col - half_length1 * cos_a - half_length2 * sin_a,
             center_row - half_length1 * sin_a + half_length2 * cos_a],
        ],
        dtype=np.int32,
    )
    return corners


def draw_rotated_rect(
    img: np.ndarray,
    center_col: float,
    center_row: float,
    half_length1: float,
    half_length2: float,
    angle_rad: float,
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> None:
    """Draw a rotated rectangle outline directly on *img* (in-place)."""
    corners = compute_rotated_rect_corners(
        center_col, center_row, half_length1, half_length2, angle_rad
    )
    cv2.polylines(img, [corners], True, color, thickness)
