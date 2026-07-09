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

from measure.constants import (
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


# ---------------------------------------------------------------------------
# Display-or-save helper
# ---------------------------------------------------------------------------

def display_or_save(
    image: np.ndarray,
    filename: str,
    subdir: str = "",
    window_name: str = "debug",
    wait_time: int = 1,
) -> None:
    """Save *image* (and optionally show it) based on global ``DISPLAY_MODE``.

    The image is always written to ``OUTPUT_DIR / subdir / filename`` when
    ``DISPLAY_MODE`` is ``'save'`` or ``'both'``.  When ``DISPLAY_MODE`` is
    ``'show'`` or ``'both'`` the image is also displayed via ``cv2.imshow``.

    This is the **single entry point** that replaces all ad-hoc
    ``cv2.imshow`` / ``cv2.waitKey`` / ``cv2.destroyAllWindows`` calls
    scattered across the test files.

    Parameters
    ----------
    image : np.ndarray
        Image to save / display.
    filename : str
        Output filename (e.g. ``"01_profile.png"``).
    subdir : str
        Optional subfolder inside ``OUTPUT_DIR`` (e.g. test name).
    window_name : str
        OpenCV window title (only used in 'show'/'both' modes).
    wait_time : int
        ``cv2.waitKey`` delay in ms.  Use ``-1`` to suppress the window
        even in ``'show'`` mode.
    """
    from measure.constants import DISPLAY_MODE, OUTPUT_DIR

    mode = DISPLAY_MODE

    # Always save when in 'save' or 'both' mode
    if mode in ("save", "both"):
        path = OUTPUT_DIR / subdir
        path.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path / filename), image)

    # Show when in 'show' or 'both' mode (and wait_time >= 0)
    if mode in ("show", "both") and wait_time >= 0:
        cv2.imshow(window_name, image)
        cv2.waitKey(wait_time)
        cv2.destroyAllWindows()
