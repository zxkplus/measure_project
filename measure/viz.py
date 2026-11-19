"""
Visualisation helpers shared by all measurement modules.

Every function here replaces an inline pattern that was duplicated
across many call sites.  The goal is a single source of truth for:
  * grayscale → BGR conversion (to_bgr)
  * double-draw text for readability (draw_text_shadow)
  * legend panels (draw_legend)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import cv2
import numpy as np

from measure.computation_types import (
    Observation,
    contract_error,
    require_space_match,
)

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


# ===========================================================================
# Typed raster rendering
# ===========================================================================
#
# Rendering never encodes, writes files or opens windows: every entry returns
# owned BGR pixels so adapters can choose how to publish them.

DEFAULT_PRIMITIVE_COLOR = (0, 220, 0)


@dataclass(frozen=True)
class RasterPrimitive:
    """One geometry observation plus its display policy."""

    geometry: "Observation[Any]"
    visible: bool = True
    color: Optional[Tuple[int, int, int]] = None


def _primitive_color(primitive: "RasterPrimitive") -> Tuple[int, int, int]:
    return tuple(primitive.color) if primitive.color else DEFAULT_PRIMITIVE_COLOR


def rasterize_geometry(image: np.ndarray, primitives, *, image_space) -> np.ndarray:
    """Draw typed geometry onto an owned BGR copy of the image.

    Args:
        image: Grayscale or BGR source image.
        primitives: Raster primitives to draw.
        image_space: Space of the image; every primitive must share it.

    Returns:
        A new BGR ``uint8`` image.

    Raises:
        ComputationContractError: When a primitive uses another frame or unit,
            even if it is invisible.
    """
    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3):
        raise contract_error("contract.invalid_shape", "image must be 2D or 3D", "image")
    canvas = np.array(to_bgr(image), dtype=np.uint8, copy=True)
    for primitive in primitives:
        require_space_match(primitive.geometry.space, image_space, "primitive.space")
    for primitive in primitives:
        if not primitive.visible or not primitive.geometry.valid:
            continue
        color = _primitive_color(primitive)
        value = primitive.geometry.value
        if isinstance(value, tuple) and len(value) == 2:
            center = (int(round(value[0])), int(round(value[1])))
            cv2.drawMarker(canvas, center, color, markerType=cv2.MARKER_CROSS,
                            markerSize=9, thickness=1)
        elif hasattr(value, "coefficients"):
            _draw_line_value(canvas, value, color)
        elif hasattr(value, "radius") and hasattr(value, "center"):
            cv2.circle(canvas, (int(round(value.center[0])), int(round(value.center[1]))),
                       max(1, int(round(value.radius))), color, 1)
        elif hasattr(value, "points_xy"):
            points = np.asarray(value.points_xy, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [points], bool(getattr(value, "closed", False)), color, 1)
    return canvas


def _draw_line_value(canvas: np.ndarray, value, color) -> None:
    """Draw a line value, using its endpoints when present."""
    if getattr(value, "start", None) is not None and getattr(value, "end", None) is not None:
        start = (int(round(value.start[0])), int(round(value.start[1])))
        end = (int(round(value.end[0])), int(round(value.end[1])))
        cv2.line(canvas, start, end, color, 1)
        return
    a, b, c = value.coefficients
    height, width = canvas.shape[:2]
    points = []
    if abs(b) > 1e-12:
        for x in (0, width - 1):
            points.append((x, int(round(-(a * x + c) / b))))
    if abs(a) > 1e-12:
        for y in (0, height - 1):
            points.append((int(round(-(b * y + c) / a)), y))
    if len(points) >= 2:
        cv2.line(canvas, points[-2], points[-1], color, 1)


def render_profile(profile, *, threshold_gray: float, size_wh: Tuple[int, int]) -> np.ndarray:
    """Render gray/smoothed/gradient curves as BGR pixels.

    Args:
        profile: Profile value carrying the three curves.
        threshold_gray: Threshold line drawn for reference.
        size_wh: Output ``(width, height)``.

    Returns:
        A new BGR ``uint8`` canvas.
    """
    width, height = int(size_wh[0]), int(size_wh[1])
    if width <= 0 or height <= 0:
        raise contract_error("contract.invalid_parameter", "size must be positive", "size_wh")
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    curves = (
        (np.asarray(profile.gray, dtype=np.float64), (255, 0, 0)),
        (np.asarray(profile.smoothed, dtype=np.float64), (0, 160, 0)),
        (np.asarray(profile.gradient, dtype=np.float64), (0, 0, 255)),
    )
    for values, color in curves:
        if values.size < 2:
            continue
        low, high = float(values.min()), float(values.max())
        span = high - low
        scaled = np.zeros_like(values) if span <= 1e-12 else (values - low) / span
        xs = np.linspace(0, width - 1, values.size).astype(np.int32)
        ys = (height - 1 - scaled * (height - 1)).astype(np.int32)
        points = np.column_stack([xs, ys]).reshape(-1, 1, 2)
        cv2.polylines(canvas, [points], False, color, 1)
    return canvas


def render_probes(image: np.ndarray, records, *, image_space) -> np.ndarray:
    """Overlay real probe records on an owned BGR copy.

    Args:
        image: Source image.
        records: Probe records with a ``spec`` describing each probe.
        image_space: Space of the image.

    Returns:
        A new BGR ``uint8`` image.
    """
    canvas = np.array(to_bgr(image), dtype=np.uint8, copy=True)
    for record in records:
        spec = record.spec
        center = (int(round(spec.center[0])), int(round(spec.center[1])))
        dx = math.cos(spec.angle_rad) * spec.search_half_length
        dy = math.sin(spec.angle_rad) * spec.search_half_length
        start = (int(round(center[0] - dx)), int(round(center[1] - dy)))
        end = (int(round(center[0] + dx)), int(round(center[1] + dy)))
        status = getattr(record, "status", "")
        color = (0, 200, 0) if status == "selected" else (0, 120, 255)
        cv2.line(canvas, start, end, color, 1)
        cv2.drawMarker(canvas, center, (0, 0, 255), markerType=cv2.MARKER_CROSS,
                       markerSize=5, thickness=1)
        selected = getattr(record, "selected", ()) or ()
        for edge in selected:
            point = (int(round(edge.point[0])), int(round(edge.point[1])))
            cv2.circle(canvas, point, 2, (255, 0, 255), -1)
    return canvas



