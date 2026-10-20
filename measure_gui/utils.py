"""GUI image helpers, plus re-exports of the shared geometry utilities.

The pure-math transform helpers (``crop_and_straighten``, ``map_point_via_affine``,
``compute_rotated_box_corners``, ...) now live in :mod:`measure.rectify` so the
API layers can use them without importing a GUI package.  They are re-exported
here for existing GUI callers.

Only :func:`cv2_to_pil` and :func:`cv2_to_tk` are genuinely GUI-specific; they
import PIL/Tkinter lazily so that importing this module stays cheap and headless.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from measure.rectify import (  # noqa: F401
    build_forward_transform,
    build_inverse_transform,
    compute_rotated_box_corners,
    crop_and_straighten,
    map_point_to_original,
    map_point_via_affine,
)

__all__ = [
    "compute_rotated_box_corners",
    "crop_and_straighten",
    "map_point_to_original",
    "build_forward_transform",
    "build_inverse_transform",
    "map_point_via_affine",
    "cv2_to_pil",
    "cv2_to_tk",
]


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