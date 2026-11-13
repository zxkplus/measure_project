"""Preprocessing preview helpers.

Pure computation: no GUI, no file I/O, no engine state.  The GUI renders what
these functions return, and nothing here mutates a workflow or a model asset.

Why this module exists
----------------------
Choosing a preprocessor is a visual decision, but the effect is invisible in the
GUI -- the template view shows the *raw* straightened patch and the editor never
shows the enhanced inspection image.  These helpers produce both sides so an
operator can compare them before committing a template.

Ordering is not interchangeable
-------------------------------
``TemplatePoint`` enhances the **cropped template patch**; ``.measure()``
enhances the **search window it is about to correlate** (the whole image when no
search region is supplied).  Both sides therefore follow "crop the region that
takes part in matching, then enhance" -- *not* "enhance the whole source image,
then crop".  For CLAHE the two orders differ by RMSE 16.59 on a real taught
asset, because tile geometry is relative to the array being enhanced.  Callers
must pass an already-cropped template patch here; this module never crops the
reference image on their behalf.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from measure.measure_template import Preprocessor, preprocessor_from_config

__all__ = [
    "PREPROCESSOR_TYPES",
    "PreviewColumn",
    "PreviewResult",
    "default_configs",
    "needs_normalization",
    "preview_matrix",
    "preview_pair",
    "to_display_uint8",
]

#: Built-in preprocessor types, in the order the GUI lists them.
PREPROCESSOR_TYPES: Tuple[str, ...] = ("raw", "canny", "sobel", "clahe", "threshold")


@dataclass
class PreviewResult:
    """One enhanced image, ready to display, plus the numbers worth reporting.

    Attributes:
        name: Preprocessor type, e.g. ``"clahe"``.
        image: uint8 display image (stretched only when the source exceeds the
            uint8 range -- see :func:`needs_normalization`).
        source: ``"template"`` or ``"inspection"``.
        stats: Human-facing metrics: ``shape``, ``source_dtype``,
            ``source_range``, ``mean``, ``std``, ``elapsed_ms``, ``normalized``.
    """

    name: str
    image: np.ndarray
    source: str
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PreviewColumn:
    """One preprocessor applied to both sides -- a column of the compare grid."""

    name: str
    config: Dict[str, Any]
    template: Optional[PreviewResult] = None
    inspection: Optional[PreviewResult] = None


def default_configs() -> List[Dict[str, Any]]:
    """Default serialized config per built-in type.

    These mirror the values the GUI has always handed to the engine, so the
    preview starts from the same settings a taught template would use.

    Returns:
        One fresh dict per entry in :data:`PREPROCESSOR_TYPES`.
    """
    return [
        {"type": "raw"},
        {"type": "canny", "t1": 50.0, "t2": 150.0},
        {"type": "sobel", "ksize": 3},
        {"type": "clahe", "clip_limit": 2.0, "tile_grid_size": [8, 8]},
        {"type": "threshold", "threshold": 128.0, "mode": "binary"},
    ]


def needs_normalization(image: np.ndarray) -> bool:
    """Whether :func:`to_display_uint8` will stretch the range instead of casting.

    Preprocessors differ in scale: ``raw`` and ``clahe`` stay on the 0-255
    intensity scale and are cast directly (so they remain visually comparable
    with the original), while ``sobel`` produces unbounded gradient magnitude
    (447 on a real asset) and has to be stretched to be visible at all.

    Args:
        image: Any numeric image array.

    Returns:
        True when the values fall outside uint8 and will be range-stretched.
    """
    arr = np.asarray(image)
    if arr.dtype == np.uint8 or arr.size == 0:
        return False
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return False
    return bool(finite.min() < 0.0 or finite.max() > 255.0)


def to_display_uint8(image: np.ndarray) -> np.ndarray:
    """Convert any preprocessor output into a uint8 image for display.

    uint8 input is passed through; float input on the intensity scale is
    clamped and rounded; anything wider is linearly stretched to 0-255.

    Args:
        image: Preprocessor output (uint8, or float32 for raw/clahe/sobel).

    Returns:
        A contiguous uint8 array of the same 2-D shape.
    """
    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)

    if arr.size == 0:
        return np.zeros(arr.shape[:2], dtype=np.uint8)

    if not needs_normalization(arr):
        return np.clip(np.rint(arr), 0.0, 255.0).astype(np.uint8)

    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape[:2], dtype=np.uint8)
    lo = float(finite.min())
    hi = float(finite.max())
    if hi <= lo:
        return np.zeros(arr.shape[:2], dtype=np.uint8)
    scaled = (arr.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(np.rint(scaled), 0.0, 255.0).astype(np.uint8)


def _describe(
    name: str,
    raw: np.ndarray,
    display: np.ndarray,
    source: str,
    elapsed_ms: float,
) -> PreviewResult:
    """Build a :class:`PreviewResult` with display statistics."""
    stats = {
        "shape": [int(display.shape[0]), int(display.shape[1])],
        "source_dtype": str(raw.dtype),
        "source_range": [round(float(raw.min()), 3), round(float(raw.max()), 3)],
        "mean": round(float(display.mean()), 2),
        "std": round(float(display.std()), 2),
        "elapsed_ms": round(elapsed_ms, 2),
        "normalized": needs_normalization(raw),
        # A span of zero means the preprocessor saturated the region (e.g. a
        # global threshold above the patch's darkest pixel). The caller should
        # say so rather than show a featureless white block.
        "uniform": bool(float(display.std()) == 0.0),
    }
    return PreviewResult(name=name, image=display, source=source, stats=stats)


def _crop_search_window(
    image: np.ndarray,
    search_region: Optional[Tuple[int, int, int, int]],
) -> np.ndarray:
    """Crop the region the matcher would enhance, clamped to the image bounds.

    Args:
        image: Full inspection image.
        search_region: ``(r1, r2, c1, c2)`` as accepted by
            :meth:`measure.measure_template.TemplatePoint.measure`, or None for
            the whole image.

    Returns:
        The window to enhance.
    """
    if search_region is None:
        return image
    r1, r2, c1, c2 = search_region
    r1 = max(0, int(r1))
    c1 = max(0, int(c1))
    r2 = min(image.shape[0], int(r2))
    c2 = min(image.shape[1], int(c2))
    return image[r1:r2, c1:c2]


def preview_pair(
    template_patch: Optional[np.ndarray],
    inspection_image: Optional[np.ndarray],
    preprocessor: Preprocessor,
    search_region: Optional[Tuple[int, int, int, int]] = None,
) -> Tuple[Optional[PreviewResult], Optional[PreviewResult]]:
    """Enhance both sides the way the matcher does.

    Args:
        template_patch: The **already cropped** straightened template patch
            (``alignment.template_image``). None when no template exists yet.
        inspection_image: Full inspection image. None when none is loaded.
        preprocessor: Instance to apply to both sides.
        search_region: Optional ``(r1, r2, c1, c2)`` search window; the engine
            passes none for the current alignment modes, which means the whole
            inspection image is enhanced.

    Returns:
        ``(template_result, inspection_result)``; either is None when its input
        is None or empty.
    """
    name = getattr(preprocessor, "name", type(preprocessor).__name__)

    template_result: Optional[PreviewResult] = None
    if template_patch is not None and np.asarray(template_patch).size:
        patch = np.asarray(template_patch)
        start = time.perf_counter()
        enhanced = preprocessor(patch)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        template_result = _describe(
            name, enhanced, to_display_uint8(enhanced), "template", elapsed_ms
        )

    inspection_result: Optional[PreviewResult] = None
    if inspection_image is not None and np.asarray(inspection_image).size:
        window = _crop_search_window(np.asarray(inspection_image), search_region)
        start = time.perf_counter()
        enhanced = preprocessor(window)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        inspection_result = _describe(
            name, enhanced, to_display_uint8(enhanced), "inspection", elapsed_ms
        )

    return template_result, inspection_result


def preview_matrix(
    template_patch: Optional[np.ndarray],
    inspection_image: Optional[np.ndarray],
    configs: Optional[Sequence[Dict[str, Any]]] = None,
    search_region: Optional[Tuple[int, int, int, int]] = None,
) -> List[PreviewColumn]:
    """Apply every requested preprocessor to both sides, for side-by-side compare.

    Args:
        template_patch: Cropped straightened template patch, or None.
        inspection_image: Full inspection image, or None.
        configs: Serialized preprocessor configs; defaults to
            :func:`default_configs`.
        search_region: Optional search window passed through to
            :func:`preview_pair`.

    Returns:
        One :class:`PreviewColumn` per config, in the given order.
    """
    columns: List[PreviewColumn] = []
    for config in (configs if configs is not None else default_configs()):
        preprocessor = preprocessor_from_config(dict(config))
        template_result, inspection_result = preview_pair(
            template_patch, inspection_image, preprocessor, search_region
        )
        columns.append(
            PreviewColumn(
                name=str(config.get("type", "unknown")),
                config=dict(config),
                template=template_result,
                inspection=inspection_result,
            )
        )
    return columns