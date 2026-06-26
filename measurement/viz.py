"""
Visualisation helpers shared by all measurement modules.

Every function here replaces an inline pattern that was duplicated
across many call sites.  The goal is a single source of truth for:
  * grayscale → BGR conversion (to_bgr)
  * double-draw text for readability (draw_text_shadow)
  * legend panels (draw_legend)
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from measurement.constants import (
    BLACK,
    DEFAULT_FONT,
    DEFAULT_FONT_SCALE,
    DEFAULT_LINE_TYPE,
    GREEN,
)


# ---------------------------------------------------------------------------
# Image conversion
# ---------------------------------------------------------------------------


def to_bgr(image: np.ndarray) -> np.ndarray:
    """Return a **writable 3-channel BGR** copy of *image*.

    If *image* is 2‑D (grayscale) it is converted; otherwise a ``.copy()``
    is returned so the caller can safely draw on it without mutating the
    original.

    Replaces the inline pattern::

        if len(image.shape) == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()

    which appeared in ~30 places.
    """
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


def to_gray(image: np.ndarray) -> np.ndarray:
    """Return a single-channel grayscale version of *image*."""
    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


# ---------------------------------------------------------------------------
# Text / label drawing
# ---------------------------------------------------------------------------


def draw_text_shadow(
    img: np.ndarray,
    text: str,
    position: Tuple[int, int],
    color: Tuple[int, int, int] = GREEN,
    font_scale: float = DEFAULT_FONT_SCALE,
    thickness: int = 1,
    font: int = DEFAULT_FONT,
    shadow_color: Tuple[int, int, int] = BLACK,
    line_type: int = DEFAULT_LINE_TYPE,
) -> None:
    """Draw *text* with a dark outline so it is readable on any background.

    The text is drawn twice: first in *shadow_color* with a thicker stroke,
    then in *color* with normal thickness.  This is the "double-draw" pattern
    that was repeated in ~40 locations.
    """
    cv2.putText(
        img, text, position, font, font_scale,
        shadow_color, thickness + 2, line_type,
    )
    cv2.putText(
        img, text, position, font, font_scale,
        color, thickness, line_type,
    )


# ---------------------------------------------------------------------------
# Legend panel
# ---------------------------------------------------------------------------


def draw_legend(
    img: np.ndarray,
    entries: list[Tuple[str, Tuple[int, int, int]]],
    start_x: int = 10,
    start_y: int = 25,
    line_height: int = 18,
    panel_width: int = 220,
    font_scale: float = 0.4,
) -> None:
    """Draw a semi-transparent legend panel at the top-left of *img*.

    *entries* is a list of ``(label, bgr_color)`` tuples.
    """
    n = len(entries)
    x, y = start_x, start_y

    # Semi-transparent background
    overlay = img.copy()
    cv2.rectangle(
        overlay,
        (x - 4, y - 16),
        (x + panel_width, y + n * line_height + 4),
        (40, 40, 40),
        -1,
    )
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

    for label, color in entries:
        draw_text_shadow(
            img, label, (x, y),
            color=color, font_scale=font_scale, line_type=cv2.LINE_AA,
        )
        y += line_height
