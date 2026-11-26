"""
Template Matching 2D Point Measurement Module

Core principle:
1. Crop a square template centered at a user-clicked position on a reference image
2. Apply a configurable Preprocessor to both template and inspection images
3. Match using normalized cross-correlation (cv2.matchTemplate with TM_CCOEFF_NORMED)
4. Refine the match location to subpixel accuracy via quadratic interpolation
5. Compute Euclidean distance between two matched points

Usage:
    from measure_template import TemplatePoint, DistanceMeasure
    from measure_template import RawPreprocessor, CannyPreprocessor

    # Raw pixel matching (default)
    pt_a = TemplatePoint(ref, click_row=200, click_col=150, template_size=80)
    pt_b = TemplatePoint(ref, click_row=200, click_col=350, template_size=80)

    # Canny edge matching
    pt_a = TemplatePoint(ref, click_row=200, click_col=150, template_size=80,
                         preprocessor=CannyPreprocessor(50, 150))

    # Save for later use
    pt_a.save("template_A.npz")

    # ... later, on a new image ...
    pt_a = TemplatePoint.from_file("template_A.npz")
    pt_b = TemplatePoint.from_file("template_B.npz")

    dm = DistanceMeasure(pt_a, pt_b)
    result = dm.measure(inspection_img)
    print(f"Distance: {result['distance']:.3f} px")
    vis = dm.visualize(inspection_img)
"""

import base64
import os
import time
import logging
import json
import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any, List, Protocol
from measure.constants import EPS
from measure.template_search import keep_disjoint, peaks, refine_peak_xy
from measure.viz import to_bgr, to_gray, draw_text_shadow

logger = logging.getLogger(__name__)



class Preprocessor(Protocol):
    """
    Template matching preprocessing interface.

    Each Preprocessor transforms a raw grayscale image (uint8) into a
    representation suitable for cv2.matchTemplate(TM_CCOEFF_NORMED).

    Critical constraint: the same preprocessor instance is applied to both
    the template crop (at construction time) and the inspection image
    (at measure time), ensuring consistent feature representation.

    To create a custom preprocessor, implement __call__, name, serialize(),
    and deserialize(), then register it in _PREPROCESSOR_REGISTRY.
    """

    @property
    def name(self) -> str:
        """Human-readable name for display labels and debugging."""
        ...

    def serialize(self) -> Dict[str, Any]:
        """Serialize to a dict. Must be reconstructable via deserialize()."""
        ...

    @staticmethod
    def deserialize(data: Dict[str, Any]) -> 'Preprocessor':
        """Reconstruct a preprocessor from the dict returned by serialize()."""
        ...

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing to a grayscale image.

        Args:
            image: 2D grayscale image, dtype=uint8.

        Returns:
            2D processed image, any dtype suitable for matchTemplate.
        """
        ...


class RawPreprocessor:
    """No enhancement — passes raw pixel intensity as float32."""

    name = 'Raw'

    def serialize(self) -> Dict[str, Any]:
        return {'type': 'raw'}

    @staticmethod
    def deserialize(data: Dict[str, Any]) -> 'RawPreprocessor':
        return RawPreprocessor()

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return image.astype(np.float32)


class CannyPreprocessor:
    """Canny edge detection — produces a binary edge map (uint8: 0/255)."""

    def __init__(self, threshold1: float = 50.0, threshold2: float = 150.0):
        self.threshold1 = threshold1
        self.threshold2 = threshold2

    @property
    def name(self) -> str:
        return f'Canny(t1={self.threshold1:.0f}, t2={self.threshold2:.0f})'

    def serialize(self) -> Dict[str, Any]:
        return {'type': 'canny', 't1': self.threshold1, 't2': self.threshold2}

    @staticmethod
    def deserialize(data: Dict[str, Any]) -> 'CannyPreprocessor':
        return CannyPreprocessor(data['t1'], data['t2'])

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return cv2.Canny(image, self.threshold1, self.threshold2)


class SobelPreprocessor:
    """Sobel gradient magnitude — float32."""

    def __init__(self, kernel_size: int = 3):
        self.kernel_size = kernel_size

    @property
    def name(self) -> str:
        return f'Sobel(k={self.kernel_size})'

    def serialize(self) -> Dict[str, Any]:
        return {'type': 'sobel', 'ksize': self.kernel_size}

    @staticmethod
    def deserialize(data: Dict[str, Any]) -> 'SobelPreprocessor':
        return SobelPreprocessor(data['ksize'])

    def __call__(self, image: np.ndarray) -> np.ndarray:
        gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=self.kernel_size)
        gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=self.kernel_size)
        return np.sqrt(gx ** 2 + gy ** 2)


class CLAHEPreprocessor:
    """CLAHE contrast-limited adaptive histogram equalization — float32."""

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self._clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    @property
    def name(self) -> str:
        return f'CLAHE(cl={self.clip_limit:.1f})'

    def serialize(self) -> Dict[str, Any]:
        return {'type': 'clahe', 'clip_limit': self.clip_limit,
                'tile_grid_size': list(self.tile_grid_size)}

    @staticmethod
    def deserialize(data: Dict[str, Any]) -> 'CLAHEPreprocessor':
        return CLAHEPreprocessor(data['clip_limit'], tuple(data['tile_grid_size']))

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self._clahe.apply(image).astype(np.float32)


class ThresholdPreprocessor:
    """
    Global threshold binarization — uint8 (0/255).

    Uses cv2.threshold to split the image into foreground (255) and
    background (0) based on a fixed intensity threshold.

    Useful when the measurement target has consistent intensity contrast
    against the background (e.g., dark part on light surface).
    """

    def __init__(self, threshold: float = 128.0,
                 mode: str = 'binary'):
        """
        Args:
            threshold: Intensity threshold in [0, 255].
            mode: 'binary' (above→255, below→0) or
                  'binary_inv' (above→0, below→255).
        """
        self.threshold = threshold
        self.mode = mode
        if mode == 'binary_inv':
            self._cv_mode = cv2.THRESH_BINARY_INV
        else:
            self._cv_mode = cv2.THRESH_BINARY

    @property
    def name(self) -> str:
        inv = '_INV' if self.mode == 'binary_inv' else ''
        return f'Threshold(t={self.threshold:.0f}{inv})'

    def serialize(self) -> Dict[str, Any]:
        return {'type': 'threshold', 'threshold': self.threshold, 'mode': self.mode}

    @staticmethod
    def deserialize(data: Dict[str, Any]) -> 'ThresholdPreprocessor':
        return ThresholdPreprocessor(data['threshold'], data.get('mode', 'binary'))

    def __call__(self, image: np.ndarray) -> np.ndarray:
        _, binary = cv2.threshold(image, self.threshold, 255, self._cv_mode)
        return binary


# Registry: maps serialize()['type'] → Preprocessor class for deserialization.
# Users can register custom preprocessors:
#   _PREPROCESSOR_REGISTRY['my_type'] = MyPreprocessor
_PREPROCESSOR_REGISTRY: Dict[str, type] = {
    'raw': RawPreprocessor,
    'canny': CannyPreprocessor,
    'sobel': SobelPreprocessor,
    'clahe': CLAHEPreprocessor,
    'threshold': ThresholdPreprocessor,
}


_TEMPLATE_TYPE_REGISTRY: Dict[str, type] = {}  # populated after class definitions


def _deserialize_preprocessor(data: Dict[str, Any]) -> Preprocessor:
    """Reconstruct a preprocessor from its serialized dict."""
    ptype = data['type']
    cls = _PREPROCESSOR_REGISTRY.get(ptype)
    if cls is None:
        known = list(_PREPROCESSOR_REGISTRY.keys())
        raise ValueError(
            f"Unknown preprocessor type: '{ptype}'. "
            f"Known types: {known}. "
            f"Register a custom type via: _PREPROCESSOR_REGISTRY['{ptype}'] = YourClass"
        )
    return cls.deserialize(data)


def preprocessor_from_config(data: Dict[str, Any]) -> Preprocessor:
    """Public entry point for rebuilding a preprocessor from a config dict.

    Accepts exactly what :meth:`Preprocessor.serialize` produces (the same dict
    stored in a model manifest's ``alignment.preprocessor``), so callers such as
    the GUI preview can round-trip a configuration without reaching into the
    registry or the private deserializer.

    Args:
        data: Serialized preprocessor dict, e.g.
            ``{"type": "clahe", "clip_limit": 2.0, "tile_grid_size": [8, 8]}``.

    Returns:
        The reconstructed :class:`Preprocessor` instance.

    Raises:
        ValueError: When ``type`` is not registered.
    """
    return _deserialize_preprocessor(data)


def check_roi_covers_template(roi_xyxy: Tuple[float, float, float, float],
                              click_col: float, click_row: float,
                              template_size: float,
                              margin: float = 0.0) -> List[Dict[str, Any]]:
    """Check that a search ROI can reach the template's taught centre.

    A template match can only report centres in
    ``[roi_x1 + half, roi_x2 - half] x [roi_y1 + half, roi_y2 - half]``.
    When the taught click point lies outside that window, the correlation peak
    is pinned to the ROI border: the reported centre is biased by a fixed
    offset and the score is under-estimated, which typically shows up later as
    a constant distance deviation (e.g. every frame measuring 22 px short).

    Args:
        roi_xyxy: Search region ``(x1, y1, x2, y2)`` in image pixels, the same
            convention as the ``roi`` field of an algorithm element.
        click_col: Taught template centre column (x).
        click_row: Taught template centre row (y).
        template_size: Square template side length in pixels.
        margin: Extra margin in pixels required on every side (default 0.0,
            i.e. only require the template itself to fit).

    Returns:
        List of issue dicts, empty when the ROI can reach the taught centre.
        Each issue has ``side`` (``"left"``/``"right"``/``"top"``/``"bottom"``),
        ``required`` (the bound the ROI must satisfy), ``actual`` (current
        value) and a human-readable ``message``.
    """
    x1, y1, x2, y2 = (float(v) for v in roi_xyxy)
    half = float(template_size) / 2.0
    m = float(margin)
    checks = (
        ("left", x1, click_col - half - m, lambda a, r: a <= r),
        ("right", x2, click_col + half + m, lambda a, r: a >= r),
        ("top", y1, click_row - half - m, lambda a, r: a <= r),
        ("bottom", y2, click_row + half + m, lambda a, r: a >= r),
    )
    issues: List[Dict[str, Any]] = []
    for side, actual, required, ok in checks:
        if not ok(actual, required):
            issues.append({
                "side": side,
                "required": required,
                "actual": actual,
                "message": (
                    f"ROI {side} clips the template: "
                    f"roi_{side} must be <= {required:.1f} (actual {actual:.1f})"
                    if side in ("left", "top") else
                    f"ROI {side} clips the template: "
                    f"roi_{side} must be >= {required:.1f} (actual {actual:.1f})"
                ),
            })
    return issues


# =========================================================================
# TemplatePoint
# =========================================================================

#: How many coarse candidates the GUI's fine stage refines by default.
#: Refining every coarse peak is the accurate choice, but a full-turn search on
#: a symmetric part produces dozens of near-equal peaks and each costs a
#: full-resolution rescan. A cap of 8 was measured selecting a different
#: candidate (score 0.716 at a wrong position) on a real 360 degree project
#: whose coarse stage produced 28 neighbourhoods; 100 covers that set and
#: reproduced the uncapped answer exactly. Raise it further only with evidence,
#: and treat a cap below the coarse neighbourhood count as an accuracy risk.
DEFAULT_FINE_CANDIDATES = 100


def _fine_candidate_limit() -> int:
    """Return how many coarse candidates the fine stage may refine (0 = all).

    Mirrors ``SearchOptions.resolved_fine_candidates`` so the GUI path and the
    engine honour the same knob. Equal-looking coarse peaks can rank within a
    hundredth of each other, so refining them all is the accurate default; a
    deployment that needs the saving sets ``MEASURE_TEMPLATE_FINE_CANDIDATES``.
    """
    configured = os.environ.get("MEASURE_TEMPLATE_FINE_CANDIDATES", "").strip()
    if configured.isdigit() and int(configured) > 0:
        return int(configured)
    return 0


def _prefer_fine_result(coarse_result: Dict[str, Any],
                       fine_result: Dict[str, Any]) -> Dict[str, Any]:
    """Choose between a coarse pose and the refinement of it.

    A refinement that scores below 90% of the coarse candidate it came from is
    usually a false peak inside the neighbourhood, so the coarse pose is the
    safer report. Either way the returned result names the stage that produced
    its score, so two scores are never compared across stages unknowingly.

    Args:
        coarse_result: The coarse match that seeded the refinement.
        fine_result: The refinement, with coordinates already in the parent frame.

    Returns:
        ``fine_result`` marked ``score_stage="fine"``, or a copy of
        ``coarse_result`` marked ``score_stage="coarse_fallback"``.
    """
    coarse_valid = coarse_result.get('valid', False)
    coarse_score = coarse_result.get('match_score', 0.0)
    fine_used = (fine_result.get('valid', False) or coarse_valid) and (
        fine_result.get('match_score', 0.0) >= coarse_score * 0.9)
    if fine_used:
        fine_result['score_stage'] = 'fine'
        return fine_result
    fallback = dict(coarse_result)
    fallback['score_stage'] = 'coarse_fallback'
    return fallback


#: Old manifest/config spellings, mapped to the current parameter names.
#: Assets taught before the rename must keep loading with the value they were
#: saved with; reading only the new key would silently restore the default and
#: change how an existing model searches.
_LEGACY_PYRAMID_KEYS = {
    "decimate_exponent": "pyramid_decimate",
    "coarse_template_side": "pyramid_max_template_size",
}


def pyramid_setting(source: Dict[str, Any], key: str, default):
    """Read one coarse-search setting, accepting its pre-rename spelling.

    Args:
        source: Manifest or config mapping.
        key: Current parameter name.
        default: Value when neither the current nor the legacy key is present.

    Returns:
        The stored value, or ``default``.
    """
    if key in source:
        return source[key]
    legacy = _LEGACY_PYRAMID_KEYS.get(key)
    if legacy is not None and legacy in source:
        return source[legacy]
    return default


class TemplatePoint:
    """
    Template-matching point measurement object.

    Crops a square template centered at a user-clicked position on a reference
    image, applies a configurable Preprocessor, and matches against inspection
    images using normalized cross-correlation.

    Usage:
        # Raw pixels (default)
        pt = TemplatePoint(ref_image, click_row=200, click_col=300, template_size=80)

        # Canny edges
        pt = TemplatePoint(ref_image, click_row=200, click_col=300, template_size=80,
                           preprocessor=CannyPreprocessor(50, 150))

        pt.save("template_A.npz")
        pt2 = TemplatePoint.from_file("template_A.npz")
        result = pt2.measure(inspection_image)
        vis = pt2.visualize(inspection_image)
    """

    def __init__(self,
                 reference_image: np.ndarray,
                 click_row: float,
                 click_col: float,
                 template_size: int = 80,
                 preprocessor: Optional[Preprocessor] = None,
                 match_score_threshold: float = 0.5,
                 use_subpixel: bool = True,
                 rotation_invariant: bool = False,
                 angle_range: Tuple[float, float] = (-30.0, 30.0),
                 angle_step: float = 1.0,
                 scale_invariant: bool = False,
                 scale_range: Tuple[float, float] = (0.9, 1.1),
                 scale_step: float = 0.02,
                 coarse_fine: bool = True,
                 coarse_angle_step: float = 5.0,
                 coarse_scale_step: float = 0.1,
                 multi_target: bool = False,
                 max_matches: int = 0,
                 overlap: float = 0.3,
                 decimate_exponent: int = 0,
                 coarse_template_side: int = 128,
                 fine_candidates: int = DEFAULT_FINE_CANDIDATES,
                 metric: str = "use_polarity"):
        """
        Initialize a template point from a reference image.

        Crops a template_size × template_size square centered at (click_row, click_col)
        and applies the preprocessor to it.

        Args:
            reference_image: Grayscale reference image (uint8)
            click_row: User-clicked row position (y-coordinate in image)
            click_col: User-clicked column position (x-coordinate in image)
            template_size: Square template side length in pixels (default 80)
            preprocessor: Preprocessor instance (RawPreprocessor if None).
                          Applied identically to template and inspection images.
            match_score_threshold: Minimum NCC score for valid match (default 0.5)
            use_subpixel: Enable subpixel refinement of the correlation peak (default True)
            rotation_invariant: Enable multi-angle template matching.
            angle_range: (min, max) search range in degrees (default -30 to 30).
            angle_step: Step size in degrees for the fine angle grid (default 1.0).
            scale_invariant: Enable multi-scale template matching.
            scale_range: (min, max) scale factor range (default 0.9 to 1.1).
            scale_step: Step size for the fine scale grid (default 0.02).
            coarse_fine: Use two-stage search: coarse grid first, then refine
                         around top candidates (default True).
            coarse_angle_step: Step size for coarse angle search (default 5.0).
            coarse_scale_step: Step size for coarse scale search (default 0.1).
            multi_target: Enable multi-target detection. When True, finds ALL
                         instances of the template above match_score_threshold
                         instead of only the single best match (default False).
            max_matches: Maximum number of matches to return when
                        multi_target=True. 0 means unlimited (default 0).
            overlap: Maximum allowed IoU overlap between detected targets in [0, 1].
                     0 = no overlap at all (most aggressive NMS). Higher values
                     allow more overlap between detected boxes (default 0.3).
            decimate_exponent: Pyramid decimation level for coarse search.
                              0 = disabled. 1 = downsample 2x, 2 = 4x, etc.
                              (default 0, disabled).
            coarse_template_side: Maximum template side length (px) after
                                       decimation. Used to auto-compute the
                                       pyramid scale factor (default 400).
        """
        self.click_row = click_row
        self.click_col = click_col
        self.template_size = template_size
        self.preprocessor = preprocessor if preprocessor is not None else RawPreprocessor()
        self.match_score_threshold = match_score_threshold
        self.use_subpixel = use_subpixel
        self.rotation_invariant = rotation_invariant
        self.angle_range = angle_range
        self.angle_step = angle_step
        self.scale_invariant = scale_invariant
        self.scale_range = scale_range
        self.scale_step = scale_step
        self.coarse_fine = coarse_fine
        self.coarse_angle_step = coarse_angle_step
        self.coarse_scale_step = coarse_scale_step
        self.multi_target = multi_target
        self.max_matches = max_matches
        self.overlap = float(overlap)
        self.decimate_exponent = int(decimate_exponent)
        self.coarse_template_side = int(coarse_template_side)
        self.fine_candidates = int(fine_candidates)
        #: Halcon's Metric: require the same contrast, or accept an
        #: inverted part as well.
        self.metric = str(metric)

        # Crop the template region (clamped to image bounds)
        gray = self._to_gray(reference_image)
        self._actual_crop_bounds = self._compute_crop_bounds(gray.shape)
        r1, r2, c1, c2 = self._actual_crop_bounds
        crop = gray[r1:r2, c1:c2]

        # Guard: click position may be entirely outside the image
        if crop.size == 0:
            raise ValueError(
                f"Template crop is empty. Click position (row={click_row:.0f}, col={click_col:.0f}) "
                f"with template_size={template_size} is entirely outside the "
                f"image bounds (h={gray.shape[0]}, w={gray.shape[1]})."
            )

        # Compute the actual center of the crop in image coordinates
        # (may differ from click position if clamped at image boundary)
        self._crop_center_row = (r1 + r2) / 2.0
        self._crop_center_col = (c1 + c2) / 2.0
        self._crop_h, self._crop_w = crop.shape

        # Apply preprocessor to template crop
        self.edge_template = self.preprocessor(crop)

        # Pre-compute pyramid scale factor for coarse matching
        self.set_pyramid(self.decimate_exponent, self.coarse_template_side)

        # Canny-specific: warn if edge content is too sparse
        if isinstance(self.preprocessor, CannyPreprocessor):
            edge_ratio = np.count_nonzero(self.edge_template) / self.edge_template.size
            if edge_ratio < 0.001:
                print(f"Warning: Template has very few edges (edge pixel ratio={edge_ratio:.4f}). "
                      f"Matching may be unreliable. Consider selecting a point with more texture.")

        # Result storage
        self.result: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Pyramid (image pyramid) helpers
    # ------------------------------------------------------------------

    def set_pyramid(self, decimate: int, max_template_size: int) -> None:
        """Set pyramid decimation and recompute the coarse-search scale factor.

        ``decimate`` keeps its historical semantics: any non-zero value enables
        the pyramid coarse search and the level number itself is not used. The
        real downsampling ratio comes from ``max_template_size``.

        Args:
            decimate: Non-zero enables the pyramid coarse search.
            max_template_size: Longest template side allowed on the coarse
                level. Zero or negative disables the size cap, so the factor
                becomes 1.0 (no downsampling).
        """
        self.decimate_exponent = int(decimate)
        self.coarse_template_side = int(max_template_size)
        if self.decimate_exponent > 0 and self.coarse_template_side > 0:
            self._pyramid_scale = self._compute_pyramid_scale()
        else:
            self._pyramid_scale = 1.0

    def _compute_pyramid_scale(self) -> float:
        """
        Compute the scale factor for the pyramid coarse search.

        Mirrors ``measure.template_search`` so both paths agree: the factor is
        the more aggressive of the decimation level (``2**-decimate_exponent``)
        and the template side cap (``coarse_template_side / longest``),
        clamped so the shortest side keeps at least 6 px.

        Both terms matter for models with no fixed template size: the level
        bounds the ratio from above, while the cap scales it with the support,
        so a 2000 px support and a 200 px support end up with a comparable
        coarse-level cost.

        Returns:
            Scale factor in (0.0, 1.0]. 1.0 means no downsampling.
        """
        height = int(self._crop_h)
        width = int(self._crop_w)
        if height <= 0 or width <= 0:
            return 1.0
        level = int(self.decimate_exponent)
        if level <= 0:
            return 1.0
        factor = min(2.0 ** -level,
                     self.coarse_template_side / float(max(height, width)))
        shortest = max(1, min(height, width))
        return min(1.0, max(factor, min(1.0, 6.0 / shortest)))

    def measure(self,
                inspection_image: np.ndarray,
                search_region: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
        """
        Match the template against an inspection image.

        The same preprocessor used for the template is applied to the
        inspection image before matching.

        When rotation_invariant or scale_invariant is enabled, performs
        a multi-angle / multi-scale search to find the best match.

        Args:
            inspection_image: Grayscale inspection image (uint8)
            search_region: Optional search bounds as (r1, r2, c1, c2).
                           Restricts matching to this sub-region for speed
                           and to reduce false positives.

        Returns:
            Dict with keys:
                - 'matched_row': float, matched center row in inspection image
                - 'matched_col': float, matched center col in inspection image
                - 'dx': float, displacement in cols from reference click position
                - 'dy': float, displacement in rows from reference click position
                - 'match_score': float, NCC score in [-1, 1], higher = better
                - 'valid': bool, whether match_score >= match_score_threshold
                - 'int_peak_y': int, integer peak row in the heatmap
                - 'int_peak_x': int, integer peak col in the heatmap
                - 'best_rotation_deg': float, detected rotation in degrees (only
                  when rotation_invariant=True, else 0.0)
                - 'best_scale': float, detected scale factor (only when
                  scale_invariant=True, else 1.0)
        """
        gray = self._to_gray(inspection_image)

        # Prepare search image and apply preprocessor
        if search_region is not None:
            r1, r2, c1, c2 = search_region
            r1 = max(0, r1)
            r2 = min(gray.shape[0], r2)
            c1 = max(0, c1)
            c2 = min(gray.shape[1], c2)
            search_img = gray[r1:r2, c1:c2]
        else:
            r1, c1 = 0, 0
            search_img = gray

        t_pre = time.perf_counter()
        search_img = self.preprocessor(search_img)
        t_pre_end = time.perf_counter()

        # Validate image is large enough for template
        if search_img.shape[0] < self._crop_h or search_img.shape[1] < self._crop_w:
            raise ValueError(
                f"Inspection image size ({search_img.shape[0]}x{search_img.shape[1]}) "
                f"is smaller than template size ({self._crop_h}x{self._crop_w})"
            )

        # Dispatch: fast path vs multi-angle/scale vs multi-target
        t0 = time.perf_counter()
        # One dispatch. Every pose search -- translation only, angle, scale,
        # multi-target, decimated or not -- runs through the shared engine,
        # which owns the decimation and the coarse/fine stages. This used to
        # branch into a GUI pyramid shell that downsampled first, so a model
        # with the pyramid enabled was decimated twice on the way down.
        if not self.rotation_invariant and not self.scale_invariant:
            if self.multi_target:
                result = self._match_translation_multi(search_img, r1, c1)
            else:
                result = self._match_translation_only(search_img, r1, c1)
        else:
            if self.multi_target:
                result = self._match_multi_angle_scale_multi(search_img, r1, c1)
            else:
                result = self._match_multi_angle_scale(search_img, r1, c1)

        # Every path reports which stage produced the score. The two-stage
        # refinement already tags itself; everything else is a single pass over
        # the resolution it was asked for, so its score is the fine score.
        result.setdefault('score_stage', 'fine')

        # The reported validity must follow the reported score. Every stage
        # gates at its own threshold -- the coarse stage deliberately uses the
        # relaxed coarse_score_threshold -- so a result chosen by one stage can
        # carry a validity decided by another. Re-decide here, at the single
        # exit, against the caller's threshold.
        result['valid'] = bool(result.get('valid', False)) and (
            float(result.get('match_score', 0.0)) >= float(self.match_score_threshold))

        # Tag the result with ROI clipping diagnostics (peak pinned to the
        # search-region border) before returning it.
        self._attach_search_diagnostics(result, search_region, r1, c1, search_img)
        self.result = result

        # Detailed timing breakdown
        t_total = time.perf_counter() - t0
        logger.info("[TIMING] TemplatePoint.measure: preproc=%(pre)8.1fms  total=%(total)8.1fms  tmpl=%(tw)dx%(th)d  search=%(sw)dx%(sh)d  pyramid_scale=%(ps).3f",
                     {"pre": (t_pre_end - t_pre) * 1000,
                      "total": t_total * 1000,
                      "tw": self._crop_w, "th": self._crop_h,
                      "sw": search_img.shape[1], "sh": search_img.shape[0],
                      "ps": self._pyramid_scale})
        return self.result

    def _attach_search_diagnostics(self, result: Dict[str, Any],
                                   search_region,
                                   r1: int, c1: int,
                                   search_img: np.ndarray) -> None:
        """Attach 'search clipped' diagnostics to a match result (in place).

        A match is flagged as clipped when its reported centre sits on the
        boundary of the reachable centre window — meaning the template's true
        correlation peak lies outside the search region and the reported
        position/score is a boundary-constrained, lower-quality estimate.

        The flag is recomputed from the matched centre (not from the integer
        peak) so it also works for the subpixel path; the subpixel shift can
        move the centre inward slightly, hence the 0.5 px tolerance.
        """
        if search_region is None:
            result['diagnostics'] = {
                'search_clip': False,
                'clip_sides': [],
                'search_region': None,
            }
            return
        if not result.get('valid', False):
            # No reliable centre for invalid results: report no clipping claim.
            result['diagnostics'] = {
                'search_clip': False,
                'clip_sides': [],
                'search_region': None,
            }
            return

        h, w = search_img.shape[:2]
        r2, c2 = r1 + h, c1 + w
        half_r, half_c = self._crop_h / 2.0, self._crop_w / 2.0
        min_r, max_r = r1 + half_r, r2 - half_r
        min_c, max_c = c1 + half_c, c2 - half_c
        eps = 0.5  # px tolerance for the subpixel-refined centre

        def _sides(row: float, col: float) -> List[str]:
            sides = []
            if col <= min_c + eps:
                sides.append('left')
            if col >= max_c - eps:
                sides.append('right')
            if row <= min_r + eps:
                sides.append('top')
            if row >= max_r - eps:
                sides.append('bottom')
            return sides

        def _diag(row: float, col: float) -> Dict[str, Any]:
            sides = _sides(row, col)
            return {
                'search_clip': bool(sides),
                'clip_sides': sides,
                'reachable_center_range': {
                    'row': [min_r, max_r],
                    'col': [min_c, max_c],
                },
            }

        diag = _diag(float(result.get('matched_row', 0.0)),
                     float(result.get('matched_col', 0.0)))
        diag['search_region'] = [c1, r1, c2, r2]  # x1, y1, x2, y2
        result['diagnostics'] = diag

        # Multi-target mode: tag every match individually.
        for match in result.get('matches') or []:
            match['diagnostics'] = _diag(float(match.get('matched_row', 0.0)),
                                         float(match.get('matched_col', 0.0)))

    def _match_translation_only(self, search_img: np.ndarray,
                                 r1: int, c1: int) -> Dict[str, Any]:
        """Fast path: delegate the single-pose NCC peak search to the shared engine.

        The correlation peak, its subpixel refinement and the score threshold are
        owned by :func:`measure.template_search.search_support`; this method only
        converts the legacy row/column ROI offset into the returned dictionary.
        """
        from measure.computation_types import Space, make_mapping
        from measure.template_search import (
            PreparedSupport,
            SearchOptions,
            search_support,
        )

        space = Space("engine", "px")
        template = np.ascontiguousarray(self.edge_template, dtype=np.float32)
        prepared = PreparedSupport(
            pixels=template,
            valid_mask=np.ones(template.shape[:2], dtype=bool),
            anchor_support=(self._crop_w / 2.0, self._crop_h / 2.0),
            support_to_reference=make_mapping(
                np.eye(3, dtype=np.float64), source=space, destination=space,
                kind="similarity"),
            reference_space=space,
            support_space=space,
            mask=None,
        )
        options = SearchOptions(
            score_threshold=-1.0,
            max_matches=1,
            max_overlap=float(getattr(self, 'overlap', 0.3)),
            subpixel=self.use_subpixel,
            metric=str(getattr(self, 'metric', 'use_polarity')),
            coarse_fine=False,
            angle_min_rad=0.0,
            angle_max_rad=0.0,
            angle_step_rad=1.0,
            scale_min=1.0,
            scale_max=1.0,
            scale_step=1.0,
        )
        found = search_support(search_img, prepared, options, image_space=space)
        matches = found.value.matches if found.valid and found.value is not None else ()
        if not matches:
            return {
                'matched_row': float(r1),
                'matched_col': float(c1),
                'dx': 0.0,
                'dy': 0.0,
                'match_score': -1.0,
                'valid': False,
                'int_peak_y': 0,
                'int_peak_x': 0,
                'best_rotation_deg': 0.0,
                'best_scale': 1.0,
            }
        match = matches[0]
        matched_row = r1 + match.center[1]
        matched_col = c1 + match.center[0]
        peak_x, peak_y = match.peak_xy if match.peak_xy else (0, 0)
        return {
            'matched_row': matched_row,
            'matched_col': matched_col,
            'dx': matched_col - self.click_col,
            'dy': matched_row - self.click_row,
            'match_score': float(match.score),
            'valid': match.score >= self.match_score_threshold,
            'int_peak_y': peak_y,
            'int_peak_x': peak_x,
            'best_rotation_deg': 0.0,
            'best_scale': 1.0,
        }

    def _match_translation_multi(self, search_img: np.ndarray,
                                  r1: int, c1: int) -> Dict[str, Any]:
        """Multi-target translation-only matching, delegated to the shared engine.

        Finds every instance above the score threshold, suppresses overlapping
        detections with IoU, and returns the legacy dictionary shape.
        """
        matches = self._shared_matches(
            search_img, r1, c1, max_matches=int(getattr(self, 'max_matches', 0)))
        return self._build_multi_result(matches)

    def _shared_support(self):
        """Build the immutable support view of the current template."""
        from measure.computation_types import Space, make_mapping
        from measure.template_search import PreparedSupport

        space = Space("engine", "px")
        template = np.ascontiguousarray(self.edge_template, dtype=np.float32)
        return PreparedSupport(
            pixels=template,
            valid_mask=np.ones(template.shape[:2], dtype=bool),
            anchor_support=(self._crop_w / 2.0, self._crop_h / 2.0),
            support_to_reference=make_mapping(
                np.eye(3, dtype=np.float64), source=space, destination=space,
                kind="similarity"),
            reference_space=space,
            support_space=space,
            mask=None,
        )

    def _shared_matches(self, search_img: np.ndarray, r1: int, c1: int,
                        *, max_matches: int) -> List[Dict[str, Any]]:
        """Run the shared support search and convert matches to legacy dicts."""
        from measure.computation_types import Space
        from measure.template_search import SearchOptions, search_support

        space = Space("engine", "px")
        options = SearchOptions(
            score_threshold=float(self.match_score_threshold),
            max_matches=int(max_matches),
            max_overlap=float(getattr(self, 'overlap', 0.3)),
            subpixel=self.use_subpixel,
            metric=str(getattr(self, 'metric', 'use_polarity')),
            coarse_fine=False,
            angle_min_rad=0.0,
            angle_max_rad=0.0,
            angle_step_rad=1.0,
            scale_min=1.0,
            scale_max=1.0,
            scale_step=1.0,
        )
        found = search_support(search_img, self._shared_support(), options,
                               image_space=space)
        if not found.valid or found.value is None:
            return []
        matches = []
        for match in found.value.matches:
            matched_row = r1 + match.center[1]
            matched_col = c1 + match.center[0]
            peak_x, peak_y = match.peak_xy if match.peak_xy else (0, 0)
            matches.append({
                'matched_row': matched_row,
                'matched_col': matched_col,
                'dx': matched_col - self.click_col,
                'dy': matched_row - self.click_row,
                'match_score': float(match.score),
                'valid': True,
                'int_peak_y': peak_y,
                'int_peak_x': peak_x,
                'best_rotation_deg': 0.0,
                'best_scale': 1.0,
            })
        return matches

    def _match_multi_angle_scale(self, search_img: np.ndarray,
                                  r1: int, c1: int) -> Dict[str, Any]:
        """
        Multi-angle / multi-scale search, delegated to the shared engine.

        This used to enumerate (angle, scale) pairs with its own
        ``cv2.matchTemplate`` loop, its own template rotation and its own
        two-stage search, so the same model searched differently here and
        through the API. The engine owns all of that now; this method only
        adapts the GUI's attribute names and dict result.
        """
        found = self._shared_pose_matches(search_img, r1, c1, multi_target=False)
        if not found:
            return self._make_invalid_result()
        return self._build_multi_result(found)

    def _match_multi_angle_scale_multi(self, search_img: np.ndarray,
                                        r1: int, c1: int) -> Dict[str, Any]:
        """
        Multi-target multi-angle / multi-scale search, delegated to the engine.

        See ``_match_multi_angle_scale``: the engine owns the pose grid, the
        coarse/fine stages and the instance selection. The only GUI-specific
        part left is expressing the result as the legacy dict.
        """
        found = self._shared_pose_matches(search_img, r1, c1, multi_target=True)
        return self._build_multi_result(found)

    def _shared_pose_matches(self, search_img: np.ndarray, r1: int, c1: int,
                             *, multi_target: bool) -> List[Dict[str, Any]]:
        """Run the engine's search over the GUI's configured pose grid.

        Args:
            search_img: Preprocessed search image.
            r1: Row offset of ``search_img`` in the inspection image.
            c1: Column offset of ``search_img`` in the inspection image.
            multi_target: Whether every accepted instance is reported.

        Returns:
            Legacy match dicts, strongest first.
        """
        from measure.computation_types import Space
        from measure.template_search import SearchOptions, search_support

        space = Space("engine", "px")
        # GUI angles are counter-clockwise (cv2.getRotationMatrix2D convention);
        # the engine's are clockwise. The sign flips here, which also swaps the
        # bounds, so the engine is handed (-max, -min).
        angle_min, angle_max = -float(self.angle_range[1]), -float(self.angle_range[0])
        scale_min, scale_max = self.scale_range
        options = SearchOptions(
            score_threshold=float(self.match_score_threshold),
            max_matches=int(self.max_matches) if multi_target else 1,
            max_overlap=float(getattr(self, "overlap", 0.3)),
            subpixel=self.use_subpixel,
            metric=str(getattr(self, 'metric', 'use_polarity')),
            coarse_fine=bool(self.coarse_fine),
            coarse_angle_step_rad=float(np.deg2rad(self.coarse_angle_step)),
            coarse_scale_step=float(self.coarse_scale_step),
            angle_min_rad=float(np.deg2rad(angle_min)) if self.rotation_invariant else 0.0,
            angle_max_rad=float(np.deg2rad(angle_max)) if self.rotation_invariant else 0.0,
            angle_step_rad=float(np.deg2rad(self.angle_step)),
            scale_min=float(scale_min) if self.scale_invariant else 1.0,
            scale_max=float(scale_max) if self.scale_invariant else 1.0,
            scale_step=float(self.scale_step),
            decimate_exponent=int(self.decimate_exponent),
            coarse_template_side=int(self.coarse_template_side),
            fine_candidates=int(self.fine_candidates) or None,
            fine_search_radius_px=float(getattr(self, "fine_search_radius_px", 24.0)),
        )
        found = search_support(search_img, self._shared_support(), options,
                               image_space=space)
        if not found.valid or found.value is None:
            return []
        matches = []
        for match in found.value.matches:
            matched_row = r1 + match.center[1]
            matched_col = c1 + match.center[0]
            peak_x, peak_y = match.peak_xy if match.peak_xy else (0, 0)
            matches.append({
                'matched_row': matched_row,
                'matched_col': matched_col,
                'dx': matched_col - self.click_col,
                'dy': matched_row - self.click_row,
                'match_score': float(match.score),
                'valid': match.score >= self.match_score_threshold,
                'int_peak_y': int(peak_y),
                'int_peak_x': int(peak_x),
                'best_rotation_deg': float(-np.rad2deg(match.angle_rad)),
                'best_scale': float(match.scale),
            })
        return matches

    def _build_multi_result(self, matches: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Build the result dict for multi-target mode.

        The top-level fields contain the best match (for backward compatibility).
        The 'matches' key holds all matches sorted by score descending.
        """
        if matches:
            best = matches[0]
            result = dict(best)
            result['matches'] = matches
            result['num_matches'] = len(matches)
            return result
        result = self._make_invalid_result()
        result['matches'] = []
        result['num_matches'] = 0
        return result


    def _make_invalid_result(self) -> Dict[str, Any]:
        """Return an invalid result dict when no valid template could be generated."""
        return {
            'matched_row': self.click_row,
            'matched_col': self.click_col,
            'dx': 0.0,
            'dy': 0.0,
            'match_score': -1.0,
            'valid': False,
            'int_peak_y': 0,
            'int_peak_x': 0,
            'best_rotation_deg': 0.0,
            'best_scale': 1.0,
        }

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def visualize(self,
                  image: np.ndarray,
                  show_template_box: bool = True,
                  show_matched_point: bool = True,
                  show_labels: bool = True,
                  template_color: Tuple[int, int, int] = (0, 255, 0),
                  matched_color: Tuple[int, int, int] = (0, 0, 255),
                  line_thickness: int = 2,
                  point_radius: int = 6,
                  wait_time: int = -1) -> np.ndarray:
        """
        Visualize the template region and matched point(s) on an image.

        In multi-target mode, all detected matches are drawn with individual
        index labels and varying colours.

        Args:
            image: Input image (grayscale or BGR)
            show_template_box: Draw the template crop region
            show_matched_point: Draw the matched center crosshair(s)
            show_labels: Show text labels with coordinates and score
            template_color: Color for the template box (B, G, R)
            matched_color: Color for the matched point (B, G, R)
            line_thickness: Line thickness for drawings
            point_radius: Radius of the matched point marker
            wait_time: OpenCV waitKey time in ms (-1 = no display)

        Returns:
            Annotated BGR image (copy)
        """
        vis_img = self._to_bgr(image)

        if show_template_box and hasattr(self, '_actual_crop_bounds'):
            self._draw_template_box(vis_img, template_color, line_thickness)

        if show_matched_point and self.result is not None:
            matches = self.result.get('matches', None)
            if matches and len(matches) > 1:
                # Multi-target mode: draw each match with its own colour + index
                self._draw_multi_matches(vis_img, matches, point_radius,
                                         show_labels, line_thickness)
            else:
                # Single-target mode (backward compatible)
                self._draw_matched_point(vis_img, matched_color, point_radius,
                                         show_labels)

        self._draw_info(vis_img)

        if wait_time >= 0:
            cv2.imshow("Template Point", vis_img)
            cv2.waitKey(wait_time)
            if wait_time > 0:
                cv2.destroyWindow("Template Point")

        return vis_img

    def _draw_multi_matches(self, img: np.ndarray,
                            matches: List[Dict[str, Any]],
                            radius: int,
                            show_labels: bool,
                            thickness: int):
        """Draw all detected matches with distinct colours and index numbers."""
        # Colour palette for up to ~20 targets (cycles if more)
        palette = [
            (0, 0, 255),     # red
            (0, 255, 0),     # green
            (255, 0, 0),     # blue
            (0, 255, 255),   # yellow
            (255, 0, 255),   # magenta
            (255, 255, 0),   # cyan
            (128, 0, 255),   # orange
            (255, 128, 0),   # sky blue
            (0, 128, 255),   # orange-yellow
            (128, 255, 0),   # lime
        ]
        for i, m in enumerate(matches):
            color = palette[i % len(palette)]
            r, c = m['matched_row'], m['matched_col']
            score = m['match_score']
            angle = m.get('best_rotation_deg', 0.0)
            scale = m.get('best_scale', 1.0)

            # Crosshair
            cv2.line(img, (int(c) - radius, int(r)), (int(c) + radius, int(r)),
                     (0, 0, 0), thickness + 1)
            cv2.line(img, (int(c) - radius, int(r)), (int(c) + radius, int(r)),
                     color, thickness)
            cv2.line(img, (int(c), int(r) - radius), (int(c), int(r) + radius),
                     (0, 0, 0), thickness + 1)
            cv2.line(img, (int(c), int(r) - radius), (int(c), int(r) + radius),
                     color, thickness)

            # Index circle
            cv2.circle(img, (int(c), int(r)), radius + 4, (0, 0, 0), thickness + 1)
            cv2.circle(img, (int(c), int(r)), radius + 4, color, thickness)
            cv2.putText(img, str(i), (int(c) - 4, int(r) + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
            cv2.putText(img, str(i), (int(c) - 4, int(r) + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

            if show_labels:
                label = (f'#{i} ({c:.1f},{r:.1f}) '
                         f's:{score:.3f} a:{angle:.1f}deg sc:{scale:.2f}')
                cv2.putText(img, label, (int(c) + radius + 8, int(r) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2)
                cv2.putText(img, label, (int(c) + radius + 8, int(r) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    def _draw_template_box(self, img: np.ndarray,
                           color: Tuple[int, int, int],
                           thickness: int):
        """Draw the template crop region as a square centered on the click position."""
        half = self.template_size / 2.0
        r1 = int(self.click_row - half)
        r2 = int(self.click_row + half)
        c1 = int(self.click_col - half)
        c2 = int(self.click_col + half)

        # Double-draw for visibility (black outline + color)
        cv2.rectangle(img, (c1, r1), (c2, r2), (0, 0, 0), thickness + 2)
        cv2.rectangle(img, (c1, r1), (c2, r2), color, thickness)

        # Click position marker
        cv2.circle(img, (int(self.click_col), int(self.click_row)), 4, color, -1)

    def _draw_matched_point(self, img: np.ndarray,
                            color: Tuple[int, int, int],
                            radius: int,
                            show_labels: bool):
        """Draw the matched center as a crosshair."""
        r = self.result['matched_row']
        c = self.result['matched_col']

        # Crosshair
        cv2.line(img, (int(c) - radius, int(r)), (int(c) + radius, int(r)),
                 (0, 0, 0), 3)
        cv2.line(img, (int(c) - radius, int(r)), (int(c) + radius, int(r)),
                 color, 2)
        cv2.line(img, (int(c), int(r) - radius), (int(c), int(r) + radius),
                 (0, 0, 0), 3)
        cv2.line(img, (int(c), int(r) - radius), (int(c), int(r) + radius),
                 color, 2)

        if show_labels:
            score = self.result['match_score']
            label = f'({c:.1f}, {r:.1f}) score:{score:.3f}'
            cv2.putText(img, label, (int(c) + radius + 5, int(r) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
            cv2.putText(img, label, (int(c) + radius + 5, int(r) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    def _draw_info(self, img: np.ndarray):
        """Draw information text overlay."""
        y = 25
        mode_str = self.preprocessor.name if hasattr(self.preprocessor, 'name') else 'Custom'
        title = f'Template Point [{mode_str}]'
        draw_text_shadow(img, title, (10, y), color=(255, 255, 255), font_scale=0.6, thickness=1)

        y += 22
        size_text = f'Template: {self._crop_h}x{self._crop_w} px'
        draw_text_shadow(img, size_text, (10, y), color=(200, 200, 200), font_scale=0.45, thickness=1)

        if self.result is not None:
            y += 20
            num_matches = self.result.get('num_matches', 0)
            if num_matches > 1:
                count_text = f'Matches: {num_matches}'
                draw_text_shadow(img, count_text, (10, y), color=(200, 200, 200), font_scale=0.45, thickness=1)
                y += 20
            score_text = f'Score: {self.result["match_score"]:.4f}'
            draw_text_shadow(img, score_text, (10, y), color=(200, 200, 200), font_scale=0.45, thickness=1)

            y += 20
            valid_text = 'VALID' if self.result['valid'] else 'INVALID'
            vcolor = (0, 255, 0) if self.result['valid'] else (0, 0, 255)
            cv2.putText(img, valid_text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(img, valid_text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, vcolor, 1)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, filepath: str) -> None:
        """
        Serialize the template to a .npz file.

        Saves the processed template data and all configuration needed
        to reconstruct the TemplatePoint without the original reference image.

        Note: only preprocessors registered in _PREPROCESSOR_REGISTRY can be
        serialized. Custom unregistered preprocessors will raise ValueError.

        Args:
            filepath: Path to the .npz output file

        Raises:
            ValueError: If the preprocessor type is not in _PREPROCESSOR_REGISTRY.
        """
        pp_data = self.preprocessor.serialize()
        pp_type = pp_data.get('type', '')
        if pp_type not in _PREPROCESSOR_REGISTRY:
            raise ValueError(
                f"Cannot serialize preprocessor of type '{pp_type}'. "
                f"Register it via: _PREPROCESSOR_REGISTRY['{pp_type}'] = YourClass"
            )

        np.savez_compressed(
            filepath,
            edge_template=self.edge_template,
            click_row=self.click_row,
            click_col=self.click_col,
            template_size=self.template_size,
            preprocessor_json=json.dumps(pp_data),
            match_score_threshold=self.match_score_threshold,
            use_subpixel=self.use_subpixel,
            crop_center_row=self._crop_center_row,
            crop_center_col=self._crop_center_col,
            crop_h=self._crop_h,
            crop_w=self._crop_w,
            actual_crop_bounds=np.array(self._actual_crop_bounds, dtype=np.int32),
            rotation_invariant=self.rotation_invariant,
            angle_range=np.array(self.angle_range, dtype=np.float64),
            angle_step=self.angle_step,
            scale_invariant=self.scale_invariant,
            scale_range=np.array(self.scale_range, dtype=np.float64),
            scale_step=self.scale_step,
            coarse_fine=self.coarse_fine,
            coarse_angle_step=self.coarse_angle_step,
            coarse_scale_step=self.coarse_scale_step,
            multi_target=self.multi_target,
            max_matches=self.max_matches,
            overlap=self.overlap,
            decimate_exponent=self.decimate_exponent,
            coarse_template_side=self.coarse_template_side,
        )

    @classmethod
    def from_file(cls, filepath: str,
                  preprocessor: Optional[Preprocessor] = None) -> 'TemplatePoint':
        """
        Deserialize a TemplatePoint from a .npz file.

        Args:
            filepath: Path to the .npz file
            preprocessor: Optional preprocessor override. If provided, replaces
                          the preprocessor stored in the file.

        Returns:
            A fully initialized TemplatePoint ready for measure()
        """
        data = np.load(filepath, allow_pickle=False)
        # Bypass __init__ since we don't have a reference image
        obj = cls.__new__(cls)
        obj.edge_template = data['edge_template']
        obj.click_row = float(data['click_row'])
        obj.click_col = float(data['click_col'])
        obj.template_size = int(data['template_size'])
        obj.match_score_threshold = float(data['match_score_threshold'])
        obj.use_subpixel = bool(data['use_subpixel'])
        obj._crop_center_row = float(data['crop_center_row'])
        obj._crop_center_col = float(data['crop_center_col'])
        obj._crop_h = int(data['crop_h'])
        obj._crop_w = int(data['crop_w'])
        obj._actual_crop_bounds = tuple(data['actual_crop_bounds'].tolist())
        obj.result = None

        # Restore preprocessor
        if preprocessor is not None:
            obj.preprocessor = preprocessor
        elif 'preprocessor_json' in data:
            pp_data = json.loads(str(data['preprocessor_json']))
            obj.preprocessor = _deserialize_preprocessor(pp_data)
        elif 'preprocessor_data' in data:
            # Legacy: old format with dict stored as object array
            pp_data = data['preprocessor_data'].item()
            obj.preprocessor = _deserialize_preprocessor(pp_data)
        elif 'use_edges' in data:
            # Backward compat: old format with use_edges + canny_threshold*
            if bool(data['use_edges']):
                t1 = float(data.get('canny_threshold1', 50))
                t2 = float(data.get('canny_threshold2', 150))
                obj.preprocessor = CannyPreprocessor(t1, t2)
            else:
                obj.preprocessor = RawPreprocessor()
        else:
            obj.preprocessor = RawPreprocessor()

        # Restore rotation/scale invariant parameters (backward compatible)
        obj.rotation_invariant = bool(data.get('rotation_invariant', False))
        angle_range_arr = data.get('angle_range')
        if angle_range_arr is not None:
            obj.angle_range = tuple(float(x) for x in angle_range_arr)
        else:
            obj.angle_range = (-30.0, 30.0)
        obj.angle_step = float(data.get('angle_step', 1.0))
        obj.scale_invariant = bool(data.get('scale_invariant', False))
        scale_range_arr = data.get('scale_range')
        if scale_range_arr is not None:
            obj.scale_range = tuple(float(x) for x in scale_range_arr)
        else:
            obj.scale_range = (0.9, 1.1)
        obj.scale_step = float(data.get('scale_step', 0.02))
        obj.coarse_fine = bool(data.get('coarse_fine', True))
        obj.coarse_angle_step = float(data.get('coarse_angle_step', 5.0))
        obj.coarse_scale_step = float(data.get('coarse_scale_step', 0.1))
        obj.multi_target = bool(data.get('multi_target', False))
        obj.max_matches = int(data.get('max_matches', 0))
        obj.overlap = float(data.get('overlap', 0.3))
        obj.decimate_exponent = int(data.get('decimate_exponent', 0))
        obj.coarse_template_side = int(pyramid_setting(data, "coarse_template_side", 128))
        # from_dict bypasses __init__, so every attribute the constructor sets has
        # to be restored here; a missing one only shows up when measure() reads it.
        obj.fine_candidates = int(data.get("fine_candidates", DEFAULT_FINE_CANDIDATES))
        obj.metric = str(data.get("metric", "use_polarity"))
        obj.set_pyramid(obj.decimate_exponent, obj.coarse_template_side)

        return obj

    # ------------------------------------------------------------------
    # JSON Serialization (complementary to NPZ save/from_file)
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize TemplatePoint config and template data to a dict.

        The preprocessed template array is base64-encoded for JSON
        compatibility.  For large templates prefer the existing
        ``save()`` method which writes a binary .npz file.
        """
        pp_data = self.preprocessor.serialize()
        pp_type = pp_data.get("type", "")
        if pp_type not in _PREPROCESSOR_REGISTRY:
            raise ValueError(
                f"Cannot serialize preprocessor of type '{pp_type}'. "
                f"Register it via: _PREPROCESSOR_REGISTRY['{pp_type}'] = YourClass"
            )

        template_bytes = self.edge_template.tobytes()
        template_b64 = base64.b64encode(template_bytes).decode("ascii")

        return {
            "object_type": "TemplatePoint",
            "click_row": self.click_row,
            "click_col": self.click_col,
            "template_size": self.template_size,
            "preprocessor_data": pp_data,
            "match_score_threshold": self.match_score_threshold,
            "use_subpixel": self.use_subpixel,
            "crop_center_row": self._crop_center_row,
            "crop_center_col": self._crop_center_col,
            "crop_h": self._crop_h,
            "crop_w": self._crop_w,
            "actual_crop_bounds": list(self._actual_crop_bounds),
            "edge_template_b64": template_b64,
            "edge_template_dtype": str(self.edge_template.dtype),
            "rotation_invariant": self.rotation_invariant,
            "angle_range": list(self.angle_range),
            "angle_step": self.angle_step,
            "scale_invariant": self.scale_invariant,
            "scale_range": list(self.scale_range),
            "scale_step": self.scale_step,
            "coarse_fine": self.coarse_fine,
            "coarse_angle_step": self.coarse_angle_step,
            "coarse_scale_step": self.coarse_scale_step,
            "multi_target": self.multi_target,
            "max_matches": self.max_matches,
            "overlap": self.overlap,
            "decimate_exponent": self.decimate_exponent,
            "coarse_template_side": self.coarse_template_side,
            "fine_candidates": self.fine_candidates,
            "metric": self.metric,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any],
                  preprocessor: Optional["Preprocessor"] = None) -> "TemplatePoint":
        """Reconstruct a TemplatePoint from a dict.

        Uses ``cls.__new__`` to bypass ``__init__`` because no reference
        image is available.  The ``edge_template`` array is decoded from
        its base64 encoding.

        Args:
            data: Dict from ``to_dict()``.
            preprocessor: Optional override for the stored preprocessor.

        Returns:
            A fully initialized TemplatePoint ready for ``measure()``.
        """
        obj = cls.__new__(cls)

        obj.click_row = float(data["click_row"])
        obj.click_col = float(data["click_col"])
        obj.template_size = int(data["template_size"])
        obj.match_score_threshold = float(data["match_score_threshold"])
        obj.use_subpixel = bool(data["use_subpixel"])
        obj._crop_center_row = float(data["crop_center_row"])
        obj._crop_center_col = float(data["crop_center_col"])
        obj._crop_h = int(data["crop_h"])
        obj._crop_w = int(data["crop_w"])
        obj._actual_crop_bounds = tuple(data["actual_crop_bounds"])
        obj.result = None

        if preprocessor is not None:
            obj.preprocessor = preprocessor
        else:
            pp_data = data.get("preprocessor_data", {"type": "raw"})
            obj.preprocessor = _deserialize_preprocessor(pp_data)

        template_b64 = data.get("edge_template_b64", "")
        edge_dtype = np.dtype(data.get("edge_template_dtype", "float32"))
        if template_b64:
            template_bytes = base64.b64decode(template_b64)
            obj.edge_template = np.frombuffer(
                template_bytes, dtype=edge_dtype
            ).reshape(obj._crop_h, obj._crop_w)
        else:
            obj.edge_template = np.zeros(
                (obj._crop_h, obj._crop_w), dtype=edge_dtype
            )

        obj.rotation_invariant = bool(data.get("rotation_invariant", False))
        obj.angle_range = tuple(data.get("angle_range", (-30.0, 30.0)))
        obj.angle_step = float(data.get("angle_step", 1.0))
        obj.scale_invariant = bool(data.get("scale_invariant", False))
        obj.scale_range = tuple(data.get("scale_range", (0.9, 1.1)))
        obj.scale_step = float(data.get("scale_step", 0.02))
        obj.coarse_fine = bool(data.get("coarse_fine", True))
        obj.coarse_angle_step = float(data.get("coarse_angle_step", 5.0))
        obj.coarse_scale_step = float(data.get("coarse_scale_step", 0.1))
        obj.multi_target = bool(data.get("multi_target", False))
        obj.max_matches = int(data.get("max_matches", 0))
        obj.overlap = float(data.get("overlap", 0.3))
        obj.decimate_exponent = int(pyramid_setting(data, "decimate_exponent", 0))
        obj.coarse_template_side = int(pyramid_setting(data, "coarse_template_side", 128))
        # from_dict bypasses __init__, so every attribute the constructor sets has
        # to be restored here; a missing one only shows up when measure() reads it.
        obj.fine_candidates = int(data.get("fine_candidates", DEFAULT_FINE_CANDIDATES))
        obj.metric = str(data.get("metric", "use_polarity"))
        obj.set_pyramid(obj.decimate_exponent, obj.coarse_template_side)

        return obj

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale if it is BGR."""
        return to_gray(image)

    @staticmethod
    def _to_bgr(image: np.ndarray) -> np.ndarray:
        """Convert image to BGR if it is grayscale (returns a copy)."""
        return to_bgr(image)

    def _compute_crop_bounds(self, image_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """
        Compute the crop rectangle clamped to image bounds.

        Returns:
            (r1, r2, c1, c2): Row and column bounds of the crop
        """
        h, w = image_shape
        half = self.template_size / 2.0
        r1 = int(max(0, self.click_row - half))
        r2 = int(min(h, self.click_row + half))
        c1 = int(max(0, self.click_col - half))
        c2 = int(min(w, self.click_col + half))
        return (r1, r2, c1, c2)

    def _refine_subpixel_2d(self, heatmap: np.ndarray,
                             peak_y: int, peak_x: int) -> Tuple[float, float]:
        """
        Refine the correlation peak to subpixel accuracy.

        Delegates to the single subpixel primitive in ``measure.template_search``
        so this adapter cannot drift from the API layer's refinement.

        Args:
            heatmap: 2D correlation map from cv2.matchTemplate
            peak_y: Integer peak row index
            peak_x: Integer peak col index

        Returns:
            (subpixel_x, subpixel_y): Refined peak position in the heatmap
        """
        return refine_peak_xy(heatmap, peak_x, peak_y)


# =========================================================================
# DistanceMeasure
# =========================================================================

class DistanceMeasure:
    """
    Two-point distance measurement using template matching.

    Holds two TemplatePoint instances, matches both against an inspection image,
    and computes the Euclidean distance between their matched positions.

    Usage:
        dm = DistanceMeasure(point_a, point_b)
        result = dm.measure(inspection_image)
        print(f"Distance: {result['distance']:.3f} px")
        vis = dm.visualize(inspection_image)
    """

    def __init__(self, point_a: TemplatePoint, point_b: TemplatePoint):
        """
        Initialize a distance measure with two template points.

        Args:
            point_a: First TemplatePoint
            point_b: Second TemplatePoint
        """
        self.point_a = point_a
        self.point_b = point_b
        self.result: Optional[Dict[str, Any]] = None

    def measure(self,
                inspection_image: np.ndarray,
                search_region_a: Optional[Tuple[int, int, int, int]] = None,
                search_region_b: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
        """
        Match both templates against the inspection image and compute distance.

        Args:
            inspection_image: Grayscale inspection image (uint8)
            search_region_a: Optional search bounds for point A
            search_region_b: Optional search bounds for point B

        Returns:
            Dict with keys:
                - 'point_a': result dict from point_a.measure()
                - 'point_b': result dict from point_b.measure()
                - 'distance': float, Euclidean distance in pixels
                - 'valid': bool, True only when both points matched successfully
        """
        result_a = self.point_a.measure(inspection_image, search_region_a)
        result_b = self.point_b.measure(inspection_image, search_region_b)

        # Euclidean distance between matched centers
        d_row = result_b['matched_row'] - result_a['matched_row']
        d_col = result_b['matched_col'] - result_a['matched_col']
        distance = np.sqrt(d_row ** 2 + d_col ** 2)

        self.result = {
            'point_a': result_a,
            'point_b': result_b,
            'distance': float(distance),
            'valid': result_a['valid'] and result_b['valid'],
        }
        return self.result

    def visualize(self,
                  image: np.ndarray,
                  show_distance_line: bool = True,
                  distance_color: Tuple[int, int, int] = (255, 0, 255),
                  line_thickness: int = 2,
                  wait_time: int = -1,
                  **kwargs) -> np.ndarray:
        """
        Visualize both template points and the distance line between them.

        Args:
            image: Input image (grayscale or BGR)
            show_distance_line: Draw a line between the two matched points
            distance_color: Color for the distance line (B, G, R)
            line_thickness: Line thickness
            wait_time: OpenCV waitKey time in ms (-1 = no display)
            **kwargs: Forwarded to each TemplatePoint.visualize()
                      (e.g., show_matched_point, show_labels, template_color)

        Returns:
            Annotated BGR image (copy)
        """
        vis_img = self.point_a.visualize(image, wait_time=-1, **kwargs)
        vis_img = self.point_b.visualize(vis_img, wait_time=-1, **kwargs)

        if show_distance_line and self.result is not None:
            self._draw_distance_line(vis_img, distance_color, line_thickness)

        self._draw_distance_info(vis_img)

        if wait_time >= 0:
            cv2.imshow("Distance Measure", vis_img)
            cv2.waitKey(wait_time)
            if wait_time > 0:
                cv2.destroyWindow("Distance Measure")

        return vis_img

    def _draw_distance_line(self, img: np.ndarray,
                            color: Tuple[int, int, int],
                            thickness: int):
        """Draw a line between the two matched points with distance label."""
        r1 = self.result['point_a']['matched_row']
        c1 = self.result['point_a']['matched_col']
        r2 = self.result['point_b']['matched_row']
        c2 = self.result['point_b']['matched_col']

        # Double-draw for visibility
        cv2.line(img, (int(c1), int(r1)), (int(c2), int(r2)),
                 (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.line(img, (int(c1), int(r1)), (int(c2), int(r2)),
                 color, thickness, cv2.LINE_AA)

        # Distance label at midpoint
        mid_r = int((r1 + r2) / 2)
        mid_c = int((c1 + c2) / 2)
        dist_text = f'{self.result["distance"]:.2f} px'
        cv2.putText(img, dist_text, (mid_c - 30, mid_r - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(img, dist_text, (mid_c - 30, mid_r - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def _draw_distance_info(self, img: np.ndarray):
        """Draw distance measurement summary text."""
        y = img.shape[0] - 40

        if self.result is not None:
            valid_text = 'VALID' if self.result['valid'] else 'PARTIAL/INVALID'
            vcolor = (0, 255, 0) if self.result['valid'] else (0, 165, 255)
            cv2.putText(img, valid_text, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
            cv2.putText(img, valid_text, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, vcolor, 1)

    # ------------------------------------------------------------------
    # JSON Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize DistanceMeasure to a JSON-compatible dict."""
        return {
            "object_type": "DistanceMeasure",
            "version": 1,
            "point_a": self.point_a.to_dict(),
            "point_b": self.point_b.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DistanceMeasure":
        """Reconstruct a DistanceMeasure from a dict.

        Raises:
            ValueError: If version is unsupported or required keys missing.
        """
        version = data.get("version", 0)
        if version != 1:
            raise ValueError(
                f"Unsupported DistanceMeasure version: {version}. Expected 1."
            )
        point_a = TemplatePoint.from_dict(data["point_a"])
        point_b = TemplatePoint.from_dict(data["point_b"])
        return cls(point_a, point_b)

    def save(self, filepath: str) -> None:
        """Serialize the DistanceMeasure to a JSON file.

        Args:
            filepath: Path to the output .json file.
        """
        data = self.to_dict()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str) -> "DistanceMeasure":
        """Deserialize a DistanceMeasure from a JSON file.

        Args:
            filepath: Path to a .json file saved by DistanceMeasure.save().

        Returns:
            A fully reconstructed DistanceMeasure.

        Raises:
            FileNotFoundError: If filepath doesn't exist.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


# Populate the template type registry (must come after class definitions)
_TEMPLATE_TYPE_REGISTRY.update({
    "TemplatePoint": TemplatePoint,
    "DistanceMeasure": DistanceMeasure,
})
