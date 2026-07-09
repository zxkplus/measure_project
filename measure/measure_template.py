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
import json
import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any, List, Protocol
from measure.constants import EPS
from measure.viz import to_bgr, to_gray, draw_text_shadow


# =========================================================================
# Preprocessor Protocol & Built-in Implementations
# =========================================================================

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


# =========================================================================
# TemplatePoint
# =========================================================================

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
                 overlap: float = 0.3):
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

        # Canny-specific: warn if edge content is too sparse
        if isinstance(self.preprocessor, CannyPreprocessor):
            edge_ratio = np.count_nonzero(self.edge_template) / self.edge_template.size
            if edge_ratio < 0.001:
                print(f"Warning: Template has very few edges (edge pixel ratio={edge_ratio:.4f}). "
                      f"Matching may be unreliable. Consider selecting a point with more texture.")

        # Result storage
        self.result: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Core measurement
    # ------------------------------------------------------------------

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

        search_img = self.preprocessor(search_img)

        # Validate image is large enough for template
        if search_img.shape[0] < self._crop_h or search_img.shape[1] < self._crop_w:
            raise ValueError(
                f"Inspection image size ({search_img.shape[0]}x{search_img.shape[1]}) "
                f"is smaller than template size ({self._crop_h}x{self._crop_w})"
            )

        # Dispatch: fast path vs multi-angle/scale vs multi-target
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

        self.result = result
        return self.result

    def _match_translation_only(self, search_img: np.ndarray,
                                 r1: int, c1: int) -> Dict[str, Any]:
        """Fast path: single translation-only matchTemplate (existing logic)."""
        heatmap = cv2.matchTemplate(search_img, self.edge_template, cv2.TM_CCOEFF_NORMED)

        # Find integer peak
        _, max_val, _, max_loc = cv2.minMaxLoc(heatmap)
        int_peak_x = max_loc[0]
        int_peak_y = max_loc[1]

        # Subpixel refinement
        if self.use_subpixel:
            subpixel_x, subpixel_y = self._refine_subpixel_2d(heatmap, int_peak_y, int_peak_x)
        else:
            subpixel_x = float(int_peak_x)
            subpixel_y = float(int_peak_y)

        # Compute matched center in the full inspection image coordinates
        matched_row = r1 + subpixel_y + self._crop_h / 2.0
        matched_col = c1 + subpixel_x + self._crop_w / 2.0

        return {
            'matched_row': matched_row,
            'matched_col': matched_col,
            'dx': matched_col - self.click_col,
            'dy': matched_row - self.click_row,
            'match_score': float(max_val),
            'valid': max_val >= self.match_score_threshold,
            'int_peak_y': int_peak_y,
            'int_peak_x': int_peak_x,
            'best_rotation_deg': 0.0,
            'best_scale': 1.0,
        }

    def _match_translation_multi(self, search_img: np.ndarray,
                                  r1: int, c1: int) -> Dict[str, Any]:
        """
        Multi-target translation-only matching.

        Finds ALL local maxima in the NCC heatmap above the score threshold,
        applies spatial NMS, and returns a list of matches plus backward-
        compatible top-level best-match fields.
        """
        heatmap = cv2.matchTemplate(search_img, self.edge_template,
                                    cv2.TM_CCOEFF_NORMED)

        peaks = self._find_peaks(heatmap, self.match_score_threshold)

        # Build NMS input: (score, row, col, angle=0, scale=1, w, h)
        candidates = [(s, r, c, 0.0, 1.0, self._crop_w, self._crop_h)
                      for s, r, c in peaks]
        selected = self._spatial_nms(candidates, self.overlap, self.max_matches)

        matches = []
        for score, py, px, angle, scale, ww, wh in selected:
            if self.use_subpixel:
                spx, spy = self._refine_subpixel_2d(heatmap, py, px)
            else:
                spx, spy = float(px), float(py)
            matched_row = r1 + spy + wh / 2.0
            matched_col = c1 + spx + ww / 2.0
            matches.append({
                'matched_row': matched_row,
                'matched_col': matched_col,
                'dx': matched_col - self.click_col,
                'dy': matched_row - self.click_row,
                'match_score': float(score),
                'valid': True,
                'int_peak_y': py,
                'int_peak_x': px,
                'best_rotation_deg': 0.0,
                'best_scale': 1.0,
            })

        return self._build_multi_result(matches)

    def _match_multi_angle_scale_multi(self, search_img: np.ndarray,
                                        r1: int, c1: int) -> Dict[str, Any]:
        """
        Multi-target multi-angle / multi-scale matching.

        Enumerates (angle, scale) combinations, finds ALL local maxima in
        each heatmap, pools all candidates, applies global spatial NMS to
        deduplicate across angles/scales, then subpixel-refines each
        surviving match.
        """
        H, W = search_img.shape
        tmpl_h, tmpl_w = self.edge_template.shape

        # ---- 1. Build search grids (reuse _build_grid helper) ----
        def _build_grid(invariant, range_tuple, step):
            if not invariant:
                if range_tuple[0] == 0.0 and abs(range_tuple[1] - 1.0) < 1e-6:
                    return [0.0]
                avg = (range_tuple[0] + range_tuple[1]) / 2.0
                return [avg]
            vals = []
            v = range_tuple[0]
            eps = step * 0.01
            while v <= range_tuple[1] + eps:
                vals.append(v)
                v += step
            return vals

        if self.coarse_fine:
            # Coarse grid
            coarse_angles = _build_grid(self.rotation_invariant, self.angle_range,
                                        self.coarse_angle_step)
            coarse_scales = _build_grid(self.scale_invariant, self.scale_range,
                                        self.coarse_scale_step)

            # Step 1: Coarse search — collect all peaks from all heatmaps
            coarse_results = self._enumerate_angles_scales(
                search_img, coarse_angles, coarse_scales, W, H, tmpl_w, tmpl_h,
                multi_peak=True, score_threshold=self.match_score_threshold
            )
            if not coarse_results:
                return self._build_multi_result([])

            # Normalise candidates: (score, (py,px,hm), angle, scale, w, h)
            #                -> (score, py, px, angle, scale, w, h)
            coarse_flat = [(s, loc[0], loc[1], a, sc, ww, wh)
                           for s, loc, a, sc, ww, wh in coarse_results]

            # Step 2: NMS on coarse results to find distinct candidate regions
            coarse_nms = self._spatial_nms(coarse_flat, self.overlap, max_matches=0)
            # Take top-K for fine refinement (K=5 to cover sparse targets well)
            coarse_nms.sort(key=lambda x: x[0], reverse=True)
            K = min(5, len(coarse_nms))
            top_candidates = coarse_nms[:K]

            # Step 3: Fine search around each candidate's angle/scale
            fine_angles_set = set()
            fine_scales_set = set()
            for _, _, _, angle, scale, _, _ in top_candidates:
                fine_angles_set.add(angle)
                fine_scales_set.add(scale)
                if self.rotation_invariant:
                    for offset in [-1, 1]:
                        na = angle + offset * self.angle_step
                        if self.angle_range[0] - 1e-6 <= na <= self.angle_range[1] + 1e-6:
                            fine_angles_set.add(na)
                if self.scale_invariant:
                    for offset in [-1, 1]:
                        ns = scale + offset * self.scale_step
                        if self.scale_range[0] - 1e-6 <= ns <= self.scale_range[1] + 1e-6:
                            fine_scales_set.add(ns)

            fine_angles = sorted(fine_angles_set) if self.rotation_invariant else [0.0]
            fine_scales = sorted(fine_scales_set) if self.scale_invariant else [1.0]
        else:
            fine_angles = _build_grid(self.rotation_invariant, self.angle_range,
                                      self.angle_step)
            fine_scales = _build_grid(self.scale_invariant, self.scale_range,
                                      self.scale_step)

        # ---- 2. Fine enumeration (multi-peak) ----
        all_results = self._enumerate_angles_scales(
            search_img, fine_angles, fine_scales, W, H, tmpl_w, tmpl_h,
            multi_peak=True, score_threshold=self.match_score_threshold
        )
        if not all_results:
            return self._build_multi_result([])

        # Normalise: (score, (py,px,hm), angle, scale, w, h)
        #        -> (score, py, px, hm, angle, scale, w, h)
        all_flat = [(s, loc[0], loc[1], loc[2] if len(loc) > 2 else None,
                      a, sc, ww, wh)
                     for s, loc, a, sc, ww, wh in all_results]

        # ---- 3. Global spatial NMS ----
        selected = self._spatial_nms(all_flat, self.overlap, self.max_matches)

        # ---- 4. Subpixel refinement per selected match ----
        matches = []
        for score, py, px, heatmap, angle, scale, ww, wh in selected:
            # Position refinement
            if self.use_subpixel and heatmap is not None:
                spx, spy = self._refine_subpixel_2d(heatmap, py, px)
            else:
                spx, spy = float(px), float(py)

            # Angle refinement via 1D quadratic interpolation
            if self.rotation_invariant and len(fine_angles) >= 3:
                refined_angle = self._refine_1d_subpixel(
                    all_results, angle, self.rotation_invariant, self.scale_invariant,
                    index_dim=2, value_dim=0
                )
            else:
                refined_angle = angle

            # Scale refinement
            if self.scale_invariant and len(fine_scales) >= 3:
                refined_scale = self._refine_1d_subpixel(
                    all_results, scale, self.rotation_invariant, self.scale_invariant,
                    index_dim=3, value_dim=0
                )
            else:
                refined_scale = scale

            matched_row = r1 + spy + wh / 2.0
            matched_col = c1 + spx + ww / 2.0

            matches.append({
                'matched_row': matched_row,
                'matched_col': matched_col,
                'dx': matched_col - self.click_col,
                'dy': matched_row - self.click_row,
                'match_score': float(score),
                'valid': True,
                'int_peak_y': py,
                'int_peak_x': px,
                'best_rotation_deg': float(refined_angle),
                'best_scale': float(refined_scale),
            })

        return self._build_multi_result(matches)

    def _build_multi_result(self, matches: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Build the result dict for multi-target mode.

        The top-level fields contain the best match (for backward compatibility).
        The 'matches' key holds all matches sorted by score descending.
        """
        if matches:
            best = matches[0]
            return {
                'matched_row': best['matched_row'],
                'matched_col': best['matched_col'],
                'dx': best['dx'],
                'dy': best['dy'],
                'match_score': best['match_score'],
                'valid': True,
                'int_peak_y': best['int_peak_y'],
                'int_peak_x': best['int_peak_x'],
                'best_rotation_deg': best['best_rotation_deg'],
                'best_scale': best['best_scale'],
                'matches': matches,
                'num_matches': len(matches),
            }
        else:
            result = self._make_invalid_result()
            result['matches'] = []
            result['num_matches'] = 0
            return result

    def _match_multi_angle_scale(self, search_img: np.ndarray,
                                  r1: int, c1: int) -> Dict[str, Any]:
        """
        Multi-angle / multi-scale search via cv2.matchTemplate enumeration.

        Enumerates (angle, scale) combinations, finding the global-best
        correlation peak. Uses optional coarse-to-fine two-stage search.
        Subpixel refinement is applied to position, angle, and scale.
        """
        H, W = search_img.shape
        tmpl_h, tmpl_w = self.edge_template.shape

        # ---- 1. Build search grids ----
        def _build_grid(invariant, range_tuple, step):
            if not invariant:
                if range_tuple[0] == 0.0 and abs(range_tuple[1] - 1.0) < 1e-6:
                    return [0.0] if range_tuple[0] == 0.0 else [1.0]
                avg = (range_tuple[0] + range_tuple[1]) / 2.0
                return [avg]
            vals = []
            v = range_tuple[0]
            eps = step * 0.01
            while v <= range_tuple[1] + eps:
                vals.append(v)
                v += step
            return vals

        if self.coarse_fine:
            # Coarse grid
            coarse_angles = _build_grid(self.rotation_invariant, self.angle_range, self.coarse_angle_step)
            coarse_scales = _build_grid(self.scale_invariant, self.scale_range, self.coarse_scale_step)

            # Step 1: Coarse search — collect all results
            coarse_results = self._enumerate_angles_scales(
                search_img, coarse_angles, coarse_scales, W, H, tmpl_w, tmpl_h
            )
            if not coarse_results:
                return self._make_invalid_result()

            # Step 2: Select top-K candidates (K=3)
            coarse_results.sort(key=lambda x: x[0], reverse=True)
            K = min(3, len(coarse_results))
            top_candidates = coarse_results[:K]

            # Step 3: Fine search around each candidate
            fine_angles_set = set()
            fine_scales_set = set()
            for _, _, angle, scale, _, _ in top_candidates:
                # Add candidate point
                fine_angles_set.add(angle)
                fine_scales_set.add(scale)
                # Add neighbours at fine step
                if self.rotation_invariant:
                    for offset in [-1, 1]:
                        na = angle + offset * self.angle_step
                        if self.angle_range[0] - 1e-6 <= na <= self.angle_range[1] + 1e-6:
                            fine_angles_set.add(na)
                if self.scale_invariant:
                    for offset in [-1, 1]:
                        ns = scale + offset * self.scale_step
                        if self.scale_range[0] - 1e-6 <= ns <= self.scale_range[1] + 1e-6:
                            fine_scales_set.add(ns)

            fine_angles = sorted(fine_angles_set) if self.rotation_invariant else [0.0]
            fine_scales = sorted(fine_scales_set) if self.scale_invariant else [1.0]
        else:
            fine_angles = _build_grid(self.rotation_invariant, self.angle_range, self.angle_step)
            fine_scales = _build_grid(self.scale_invariant, self.scale_range, self.scale_step)

        # ---- 2. Fine enumeration ----
        all_results = self._enumerate_angles_scales(
            search_img, fine_angles, fine_scales, W, H, tmpl_w, tmpl_h
        )
        if not all_results:
            return self._make_invalid_result()

        best_score, best_loc, best_angle, best_scale, best_w, best_h = max(
            all_results, key=lambda x: x[0]
        )
        best_heatmap = best_loc[2] if len(best_loc) > 2 else None

        # ---- 3. Subpixel refinement ----
        # 3a. Position (re-run matchTemplate at best angle/scale if we don't have the heatmap)
        if best_heatmap is None and self.use_subpixel:
            warped = self._rotate_scale_template(best_angle, best_scale)
            if warped is not None:
                best_heatmap = cv2.matchTemplate(search_img, warped, cv2.TM_CCOEFF_NORMED)

        int_peak_y, int_peak_x = best_loc[0], best_loc[1]
        if self.use_subpixel and best_heatmap is not None:
            subpixel_x, subpixel_y = self._refine_subpixel_2d(
                best_heatmap, int_peak_y, int_peak_x
            )
        else:
            subpixel_x = float(int_peak_x)
            subpixel_y = float(int_peak_y)

        # 3b. Angle (1D quadratic interpolation across 3 grid points)
        if self.rotation_invariant and len(fine_angles) >= 3:
            refined_angle = self._refine_1d_subpixel(
                all_results, best_angle, self.rotation_invariant, self.scale_invariant,
                index_dim=2, value_dim=0
            )
        else:
            refined_angle = best_angle

        # 3c. Scale
        if self.scale_invariant and len(fine_scales) >= 3:
            refined_scale = self._refine_1d_subpixel(
                all_results, best_scale, self.rotation_invariant, self.scale_invariant,
                index_dim=3, value_dim=0
            )
        else:
            refined_scale = best_scale

        # ---- 4. Coordinate transform ----
        # The warped template dimensions affect the center offset
        matched_row = r1 + subpixel_y + best_h / 2.0
        matched_col = c1 + subpixel_x + best_w / 2.0

        return {
            'matched_row': matched_row,
            'matched_col': matched_col,
            'dx': matched_col - self.click_col,
            'dy': matched_row - self.click_row,
            'match_score': float(best_score),
            'valid': best_score >= self.match_score_threshold,
            'int_peak_y': int_peak_y,
            'int_peak_x': int_peak_x,
            'best_rotation_deg': float(refined_angle),
            'best_scale': float(refined_scale),
        }

    def _enumerate_angles_scales(self, search_img, angles, scales,
                                  W, H, tmpl_w, tmpl_h,
                                  multi_peak: bool = False,
                                  score_threshold: float = 0.0):
        """
        Enumerate all (angle, scale) combinations, calling matchTemplate
        for each valid combination.

        Args:
            multi_peak: If True, find ALL local maxima above score_threshold
                       for each heatmap (multi-target mode). If False, only
                       the single global peak (backward compatible).
            score_threshold: Minimum score for peaks when multi_peak=True.

        Returns:
            List of (score, (peak_y, peak_x, heatmap), angle, scale, warped_w, warped_h)
        """
        results = []
        for angle in angles:
            for scale in scales:
                warped = self._rotate_scale_template(angle, scale)
                if warped is None:
                    continue
                wh, ww = warped.shape
                if wh > H or ww > W:
                    continue
                heatmap = cv2.matchTemplate(search_img, warped, cv2.TM_CCOEFF_NORMED)

                if multi_peak:
                    peaks = self._find_peaks(heatmap, score_threshold)
                    for score, py, px in peaks:
                        results.append((score, (py, px, heatmap),
                                        angle, scale, ww, wh))
                else:
                    _, max_val, _, max_loc = cv2.minMaxLoc(heatmap)
                    results.append((float(max_val), (max_loc[1], max_loc[0], heatmap),
                                    angle, scale, ww, wh))
        return results

    def _rotate_scale_template(self, angle_deg: float, scale: float) -> Optional[np.ndarray]:
        """
        Generate a rotated and scaled version of the template.

        Rotation is applied around the template center. The output array
        is sized to fully contain the rotated+scaled template with no clipping.

        Returns:
            Warped template array, or None if the template is empty or degenerate.
        """
        h_t, w_t = self.edge_template.shape
        center = (w_t / 2.0, h_t / 2.0)

        # Build affine matrix: rotation around center + scale
        M = cv2.getRotationMatrix2D(center, angle_deg, scale)

        # Compute bounding box of the warped template
        corners = np.array([
            [0, 0], [w_t, 0], [w_t, h_t], [0, h_t]
        ], dtype=np.float32)
        transformed = cv2.transform(corners.reshape(1, -1, 2), M).reshape(-1, 2)
        min_x = np.floor(transformed[:, 0].min())
        max_x = np.ceil(transformed[:, 0].max())
        min_y = np.floor(transformed[:, 1].min())
        max_y = np.ceil(transformed[:, 1].max())
        new_w = int(max_x - min_x)
        new_h = int(max_y - min_y)

        if new_w <= 0 or new_h <= 0:
            return None

        # Adjust translation so all content fits in the output
        M[0, 2] -= min_x
        M[1, 2] -= min_y

        # Use template mean as border value to avoid introducing artificial
        # edges at rotated borders (which would bias NCC toward 0°).
        border_value = float(np.mean(self.edge_template))

        warped = cv2.warpAffine(self.edge_template, M, (new_w, new_h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=border_value)
        return warped

    @staticmethod
    def _find_peaks(heatmap: np.ndarray,
                    threshold: float) -> List[Tuple[float, int, int]]:
        """
        Find all local maxima in a correlation heatmap above a threshold.

        Uses 3×3 dilation to detect pixel-level local maxima: a pixel is a
        peak if it equals the dilated value (no larger neighbour) AND its
        score >= threshold.

        Args:
            heatmap: 2D correlation heatmap (e.g. from cv2.matchTemplate).
            threshold: Minimum score for a peak to be included.

        Returns:
            List of (score, row, col) tuples, sorted by score descending.
        """
        kernel = np.ones((3, 3), dtype=heatmap.dtype)
        dilated = cv2.dilate(heatmap, kernel)
        mask = (heatmap == dilated) & (heatmap >= threshold)
        peak_ys, peak_xs = np.where(mask)
        peaks = [(float(heatmap[py, px]), int(py), int(px))
                 for py, px in zip(peak_ys, peak_xs)]
        peaks.sort(key=lambda x: x[0], reverse=True)
        return peaks

    @staticmethod
    def _spatial_nms(candidates: List[tuple],
                     overlap: float,
                     max_matches: int = 0) -> List[tuple]:
        """
        IoU-based non-maximum suppression for template match candidates.

        Sorts candidates by score descending and greedily selects those
        whose axis-aligned bounding box IoU (Intersection over Union) with
        every already selected candidate is <= overlap.

        overlap=0.0 means no overlap allowed between targets (most aggressive).
        overlap=1.0 means any overlap is allowed (no NMS suppression).

        Candidates must be tuples where:
          index 1 = row (center y)
          index 2 = col (center x)
          index -2 = width (c[-2])
          index -1 = height (c[-1])

        Args:
            candidates: List of (score, row, col, ..., w, h) tuples.
            overlap: Maximum allowed IoU in [0, 1]. 0 = no overlap at all.
            max_matches: Maximum number of matches (0 = unlimited).

        Returns:
            Filtered list of candidates, sorted by score descending.
        """
        if overlap >= 1.0:
            # No NMS needed — just take top max_matches by score
            selected = sorted(candidates, key=lambda x: x[0], reverse=True)
            if max_matches > 0:
                selected = selected[:max_matches]
            return selected

        sorted_candidates = sorted(candidates, key=lambda x: x[0], reverse=True)
        selected: List[tuple] = []

        for c in sorted_candidates:
            row, col = c[1], c[2]
            w, h = c[-2], c[-1]  # width and height are always last two elements

            # Axis-aligned bounding box for candidate c
            x1 = col - w / 2.0
            y1 = row - h / 2.0
            x2 = col + w / 2.0
            y2 = row + h / 2.0
            area = w * h

            keep = True
            for s in selected:
                # Axis-aligned bounding box for selected s
                sx1 = s[2] - s[-2] / 2.0
                sy1 = s[1] - s[-1] / 2.0
                sx2 = s[2] + s[-2] / 2.0
                sy2 = s[1] + s[-1] / 2.0
                sarea = s[-2] * s[-1]

                # Intersection area
                ix1 = max(x1, sx1)
                iy1 = max(y1, sy1)
                ix2 = min(x2, sx2)
                iy2 = min(y2, sy2)
                iw = max(0.0, ix2 - ix1)
                ih = max(0.0, iy2 - iy1)
                inter_area = iw * ih

                if inter_area > 0:
                    iou = inter_area / (area + sarea - inter_area)
                    if iou > overlap:
                        keep = False
                        break

            if keep:
                selected.append(c)
                if max_matches > 0 and len(selected) >= max_matches:
                    break

        return selected

    @staticmethod
    def _refine_1d_subpixel(results, best_val, rotation_invariant, scale_invariant,
                             index_dim, value_dim):
        """
        Refine angle or scale to sub-step accuracy via 1D quadratic interpolation.

        Filters results to those matching best_val in the other dimension, then
        fits a parabola through the best value and its two neighbours.

        Args:
            results: List of (score, loc, angle, scale, w, h) tuples.
            best_val: The best angle (or scale) from the discrete search.
            rotation_invariant, scale_invariant: Flags.
            index_dim: 2 for angle, 3 for scale.
            value_dim: 0 for score.

        Returns:
            Refined angle or scale value (clamped to the neighbour range).
        """
        other_dim = 3 if index_dim == 2 else 2

        # Collect unique values of the target dimension and their best scores
        val_scores = {}  # {angle_or_scale: max_score}
        for r in results:
            v = r[index_dim]
            score = r[value_dim]
            if v not in val_scores or score > val_scores[v]:
                val_scores[v] = score

        sorted_vals = sorted(val_scores.keys())
        if len(sorted_vals) < 3:
            return best_val

        try:
            idx = sorted_vals.index(best_val)
        except ValueError:
            return best_val

        # Need at least one neighbour on each side
        if idx == 0 or idx == len(sorted_vals) - 1:
            return best_val

        x = np.array([sorted_vals[idx - 1], sorted_vals[idx], sorted_vals[idx + 1]],
                     dtype=np.float64)
        y = np.array([val_scores[xi] for xi in sorted_vals[idx - 1:idx + 2]],
                     dtype=np.float64)

        try:
            coeffs = np.polyfit(x, y, 2)
            a = coeffs[0]
            if abs(a) > 1e-12:
                refined = -coeffs[1] / (2.0 * a)
                refined = max(x[0], min(x[-1], refined))
                return float(refined)
        except Exception:
            pass
        return best_val

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
        Refine the correlation peak to subpixel accuracy using two independent
        1D quadratic fits along x and y through the peak.

        This mirrors the _refine_subpixel pattern from measure1D.py, extended
        to the correlation surface: one quadratic fit along the row direction
        through the peak, and one along the column direction.

        Args:
            heatmap: 2D correlation map from cv2.matchTemplate
            peak_y: Integer peak row index
            peak_x: Integer peak col index

        Returns:
            (subpixel_x, subpixel_y): Refined peak position in the heatmap
        """
        h, w = heatmap.shape

        # Refine along x (columns) — fit to heatmap[peak_y, peak_x-1 : peak_x+2]
        if 0 < peak_x < w - 1:
            x_vals = np.array([peak_x - 1, peak_x, peak_x + 1], dtype=np.float64)
            y_vals = heatmap[peak_y, peak_x - 1 : peak_x + 2].astype(np.float64)
            try:
                coeffs = np.polyfit(x_vals, y_vals, 2)
                a, b, c = coeffs
                if abs(a) > EPS:
                    subpixel_x = -b / (2.0 * a)
                else:
                    subpixel_x = float(peak_x)
                subpixel_x = max(x_vals[0], min(x_vals[-1], subpixel_x))
            except Exception:
                subpixel_x = float(peak_x)
        else:
            subpixel_x = float(peak_x)

        # Refine along y (rows) — fit to heatmap[peak_y-1 : peak_y+2, peak_x]
        if 0 < peak_y < h - 1:
            x_vals = np.array([peak_y - 1, peak_y, peak_y + 1], dtype=np.float64)
            y_vals = heatmap[peak_y - 1 : peak_y + 2, peak_x].astype(np.float64)
            try:
                coeffs = np.polyfit(x_vals, y_vals, 2)
                a, b, c = coeffs
                if abs(a) > EPS:
                    subpixel_y = -b / (2.0 * a)
                else:
                    subpixel_y = float(peak_y)
                subpixel_y = max(x_vals[0], min(x_vals[-1], subpixel_y))
            except Exception:
                subpixel_y = float(peak_y)
        else:
            subpixel_y = float(peak_y)

        return subpixel_x, subpixel_y


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
