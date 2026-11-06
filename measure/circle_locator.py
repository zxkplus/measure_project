"""
Robust sub-pixel circle locator for camera-on-arm pick calibration.

The locator answers one question: *where is the centre of the most
confident circular target in this image?*  It is used to feed the
image-to-robot calibration and must therefore fail closed: a low
confidence detection has to be rejected instead of silently producing a
centre a few pixels off.

Pipeline
--------
1. **Coarse candidates** — Otsu segmentation contours (one polarity or
   both) plus optional Hough circles.  Candidates are de-duplicated and
   capped so the expensive stage stays bounded.
2. **Radial edge refinement** — for each candidate, sample the intensity
   profile along ``num_rays`` rays around the candidate radius, locate the
   radial gradient peak per ray with a parabolic sub-pixel fit, then fit a
   circle to the resulting edge points (algebraic fit followed by
   geometric Gauss-Newton refinement, with MAD outlier rejection).  A
   second, narrower pass re-centres the search on the fitted circle.
3. **Confidence** — combines fit roundness, angular support, angular
   coverage and contrast-to-noise.  Hard gates reject non-circular blobs.

Conventions
-----------
* Image coordinates are ``(u, v) = (col, row)``, matching OpenCV.
* A detection is only usable when ``CircleDetection.valid`` is ``True``;
  ``confidence`` is a quality score in ``[0, 1]``.
* ``radius_range`` is expressed in pixels and filters both coarse and
  refined radii.

Typical use::

    from measure.circle_locator import CircleLocator

    locator = CircleLocator(radius_range=(10, 80))
    det = locator.find(image, roi=(100, 80, 540, 420))
    if det.valid:
        print(det.center, det.radius, det.confidence)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from measure.viz import to_bgr, to_gray

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TWO_PI = 2.0 * math.pi
# Laplacian kernel [[0, 1, 0], [1, -4, 1], [0, 1, 0]] has squared norm 20;
# dividing the MAD by sqrt(20) converts the response spread back to the
# additive pixel-noise sigma (Gaussian noise assumed).
_LAPLACIAN_NORM = math.sqrt(20.0)
_MAD_TO_SIGMA = 1.4826
# Minimum number of radial samples used by the sub-pixel parabola.
_MIN_RAY_SAMPLES = 40
_MAX_RAY_SAMPLES = 241
# Number of angular sectors used for the coverage statistic.
_COVERAGE_SECTORS = 24


def _clamp01(value: float) -> float:
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else float(value)


def _as_uint8(image: np.ndarray) -> np.ndarray:
    """Return an 8-bit single-channel image (normalised when necessary)."""
    gray = to_gray(image)
    if gray.dtype == np.uint8:
        return gray
    gray = np.asarray(gray, dtype=np.float32)
    low, high = float(gray.min()), float(gray.max())
    if high - low < 1e-9:
        return np.zeros(gray.shape, dtype=np.uint8)
    scaled = (gray - low) * (255.0 / (high - low))
    return scaled.astype(np.uint8)


def _resolve_roi(
    roi: Optional[Sequence[int]], width: int, height: int
) -> Tuple[int, int, int, int]:
    """Normalise ``(x1, y1, x2, y2)`` (inclusive) to the image bounds."""
    if roi is None:
        return 0, 0, width - 1, height - 1
    if len(roi) != 4:
        raise ValueError("roi must be (x1, y1, x2, y2)")
    x1, y1, x2, y2 = (int(round(float(v))) for v in roi)
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width - 1, x2)
    y2 = min(height - 1, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        raise ValueError("roi is degenerate or outside the image")
    return x1, y1, x2, y2


def estimate_noise_sigma(gray: np.ndarray) -> float:
    """Estimate additive pixel noise sigma from a Laplacian MAD."""
    data = np.asarray(gray, dtype=np.float32)
    if data.size < 16:
        return 0.0
    lap = cv2.Laplacian(data, cv2.CV_32F, ksize=1)
    median = float(np.median(lap))
    mad = float(np.median(np.abs(lap - median)))
    return _MAD_TO_SIGMA * mad / _LAPLACIAN_NORM


# ---------------------------------------------------------------------------
# Detection result
# ---------------------------------------------------------------------------


@dataclass
class CircleDetection:
    """One circle candidate and its quality diagnostics.

    Attributes:
        center: Sub-pixel centre ``(u, v) = (col, row)`` in full-image pixels.
        radius: Fitted radius in pixels.
        confidence: Quality score in ``[0, 1]``; higher is better.
        valid: ``True`` when every hard gate passed.
        reason: ``"ok"`` or the failed gate names joined by ``"+"``.
        method: Coarse source: ``"contour"`` or ``"hough"``.
        support: Fraction of sampled rays with an inlier edge point.
        coverage: Fraction of angular sectors containing an inlier.
        rms_error: RMS radial residual of the inlier edge points (px).
        max_error: Maximum radial residual of the inlier edge points (px).
        ellipticity: Fitted ellipse major/minor axis ratio (1 = circular).
        contrast: Inside/outside intensity difference in noise sigma units.
        inliers: Number of inlier edges used by the final fit.
        rays: Number of rays sampled.
        bbox: ``(x1, y1, x2, y2)`` pixel bounds of the fitted circle.
    """

    center: Tuple[float, float]
    radius: float
    confidence: float
    valid: bool
    reason: str = ""
    method: str = ""
    polarity: str = ""
    support: float = 0.0
    coverage: float = 0.0
    rms_error: float = 0.0
    max_error: float = 0.0
    ellipticity: float = 1.0
    contrast: float = 0.0
    inliers: int = 0
    rays: int = 0
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    edge_points: List[Tuple[float, float]] = field(default_factory=list)

    @property
    def center_rc(self) -> Tuple[float, float]:
        """Centre as ``(row, col)`` for code that uses row/col order."""
        return (float(self.center[1]), float(self.center[0]))

    def as_dict(self, include_points: bool = False) -> Dict[str, Any]:
        """Return a JSON-serialisable summary."""
        data: Dict[str, Any] = {
            "center_uv": [float(self.center[0]), float(self.center[1])],
            "radius": float(self.radius),
            "confidence": float(self.confidence),
            "valid": bool(self.valid),
            "reason": self.reason,
            "method": self.method,
            "polarity": self.polarity,
            "support": float(self.support),
            "coverage": float(self.coverage),
            "rms_error": float(self.rms_error),
            "max_error": float(self.max_error),
            "ellipticity": float(self.ellipticity),
            "contrast": float(self.contrast),
            "inliers": int(self.inliers),
            "rays": int(self.rays),
            "bbox": [int(v) for v in self.bbox],
        }
        if include_points:
            data["edge_points"] = [[float(u), float(v)] for u, v in self.edge_points]
        return data

    @classmethod
    def empty(cls, reason: str = "no_candidates") -> "CircleDetection":
        """Return an invalid placeholder detection."""
        return cls(
            center=(float("nan"), float("nan")),
            radius=0.0,
            confidence=0.0,
            valid=False,
            reason=reason,
        )


@dataclass
class _Candidate:
    """Internal coarse circle candidate."""

    cx: float
    cy: float
    radius: float
    score: float
    method: str
    polarity: str = "auto"


# ---------------------------------------------------------------------------
# Coarse candidate generation
# ---------------------------------------------------------------------------


def _contour_candidates(
    gray8: np.ndarray,
    ox: int,
    oy: int,
    radius_range: Tuple[float, float],
    polarity: str,
    max_candidates: int,
) -> List[_Candidate]:
    """Otsu threshold + external contours -> circle candidates."""
    rmin, rmax = radius_range
    flags = cv2.THRESH_BINARY_INV if polarity == "dark" else cv2.THRESH_BINARY
    threshold, mask = cv2.threshold(gray8, 0, 255, flags + cv2.THRESH_OTSU)
    if mask.mean() < 0.5:
        return []
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    candidates: List[_Candidate] = []
    min_area = 0.25 * math.pi * rmin * rmin
    for contour in contours:
        area = abs(float(cv2.contourArea(contour)))
        if area < min_area:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 1e-6:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        if not (rmin <= radius <= rmax):
            continue
        circularity = _clamp01(4.0 * math.pi * area / (perimeter * perimeter))
        fill = _clamp01(area / (math.pi * radius * radius + 1e-9))
        score = 0.5 * circularity + 0.5 * fill
        candidates.append(
            _Candidate(
                cx=float(cx) + ox,
                cy=float(cy) + oy,
                radius=float(radius),
                score=score,
                method="contour",
                polarity=polarity,
            )
        )
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:max_candidates]


def _hough_candidates(
    gray8: np.ndarray,
    ox: int,
    oy: int,
    radius_range: Tuple[float, float],
    param1: float,
    param2: float,
    max_candidates: int,
) -> List[_Candidate]:
    """Hough circle fallback, useful for low-contrast or partial targets."""
    rmin, rmax = radius_range
    min_radius = max(1, int(math.floor(rmin)))
    max_radius = max(min_radius + 1, int(math.ceil(rmax)))
    min_dist = max(8, int(round(rmin)))
    try:
        if hasattr(cv2, "HOUGH_GRADIENT_ALT"):
            # ALT is ~50x faster than HOUGH_GRADIENT here and returns far
            # fewer false circles on texture; param2 is the "perfectness"
            # threshold (0.75 is tolerant, 0.9 is strict).
            circles = cv2.HoughCircles(
                gray8,
                cv2.HOUGH_GRADIENT_ALT,
                dp=1.5,
                minDist=min_dist,
                param1=float(param1),
                param2=float(param2),
                minRadius=min_radius,
                maxRadius=max_radius,
            )
        else:  # pragma: no cover - OpenCV < 4.3
            circles = cv2.HoughCircles(
                gray8,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=min_dist,
                param1=float(param1),
                param2=25.0,
                minRadius=min_radius,
                maxRadius=max_radius,
            )
    except cv2.error:
        return []
    if circles is None:
        return []
    found: List[_Candidate] = []
    for cx, cy, radius in np.asarray(circles, dtype=np.float64).reshape(-1, 3):
        if not (rmin <= radius <= rmax):
            continue
        found.append(
            _Candidate(
                cx=float(cx) + ox,
                cy=float(cy) + oy,
                radius=float(radius),
                score=0.55,
                method="hough",
                polarity="auto",
            )
        )
    return found[:max_candidates]


def _same_circle(a: _Candidate, b: _Candidate) -> bool:
    """Return True when two coarse candidates almost certainly agree."""
    distance = math.hypot(a.cx - b.cx, a.cy - b.cy)
    smaller = min(a.radius, b.radius)
    larger = max(a.radius, b.radius)
    return distance < max(3.0, 0.35 * smaller) and (larger - smaller) < 0.40 * larger


def _dedupe(candidates: Sequence[_Candidate]) -> List[_Candidate]:
    """Keep the best-scoring candidate for each physical circle."""
    kept: List[_Candidate] = []
    for candidate in sorted(candidates, key=lambda c: c.score, reverse=True):
        if not any(_same_circle(candidate, other) for other in kept):
            kept.append(candidate)
    return kept


# ---------------------------------------------------------------------------
# Sub-pixel radial refinement
# ---------------------------------------------------------------------------


def _sample_profiles(
    gray32: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    angles: np.ndarray,
    span: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Sample radial intensity profiles around a circle.

    Args:
        gray32: Full-image float32 grayscale.
        cx, cy: Candidate centre.
        radius: Candidate radius in pixels.
        angles: ``(N,)`` ray angles in radians.
        span: Half-width of the radial search window in pixels.

    Returns:
        ``(profiles, radii, step)`` where ``profiles`` has shape
        ``(N, S)``, ``radii`` has shape ``(S,)`` and ``step`` is the
        radial spacing in pixels.
    """
    n_samples = int(2.0 * span / 0.5) + 1
    n_samples = max(_MIN_RAY_SAMPLES, min(_MAX_RAY_SAMPLES, n_samples))
    if n_samples % 2 == 0:
        n_samples += 1
    radii = np.linspace(radius - span, radius + span, n_samples, dtype=np.float32)
    step = float(radii[1] - radii[0])
    cos_a = np.cos(angles).astype(np.float32)[:, None]
    sin_a = np.sin(angles).astype(np.float32)[:, None]
    map_x = (cx + radii[None, :] * cos_a).astype(np.float32)
    map_y = (cy + radii[None, :] * sin_a).astype(np.float32)
    profiles = cv2.remap(
        gray32,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return np.asarray(profiles, dtype=np.float32), radii, step


def _peak_points(
    profiles: np.ndarray,
    radii: np.ndarray,
    step: float,
    angles: np.ndarray,
    cx: float,
    cy: float,
    sign: float,
    min_gradient: float,
    blur_sigma: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Locate one sub-pixel radial edge per ray.

    Args:
        profiles: ``(N, S)`` radial intensity profiles.
        radii: ``(S,)`` radius of each profile column.
        step: Radial spacing (pixels / column).
        angles: ``(N,)`` ray angles.
        cx, cy: Centre used to place the rays.
        sign: ``+1`` when intensity increases outward, ``-1`` otherwise.
        min_gradient: Minimum radial gradient magnitude (gray levels / px).
        blur_sigma: Gaussian sigma applied along the radial axis.

    Returns:
        ``(points_xy, ray_indices, peak_radii, peaks)`` for accepted rays.
    """
    # Smooth only along the radial axis; keep rays independent.
    smoothed = cv2.GaussianBlur(profiles, (1, 5), float(blur_sigma))
    gradient = np.gradient(smoothed, axis=1) / max(step, 1e-6)
    signed = gradient * float(sign)

    n_rays, n_samples = signed.shape
    peak_idx = np.argmax(signed, axis=1)
    rows = np.arange(n_rays)
    peaks = signed[rows, peak_idx]
    interior = (peak_idx > 0) & (peak_idx < n_samples - 1)
    accepted = interior & (peaks >= float(min_gradient))
    if not np.any(accepted):
        return (
            np.empty((0, 2), dtype=np.float64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )

    rows = rows[accepted]
    idx = peak_idx[accepted]
    y0 = signed[rows, idx - 1]
    y1 = signed[rows, idx]
    y2 = signed[rows, idx + 1]
    denominator = y0 - 2.0 * y1 + y2
    safe = np.abs(denominator) > 1e-9
    offset = np.zeros_like(y1)
    offset[safe] = 0.5 * (y0[safe] - y2[safe]) / denominator[safe]
    offset = np.clip(offset, -0.75, 0.75)
    radius_peak = radii[idx] + offset * step
    theta = angles[rows]
    points = np.column_stack(
        [cx + radius_peak * np.cos(theta), cy + radius_peak * np.sin(theta)]
    )
    return points, rows, radius_peak.astype(np.float64), y1.astype(np.float64)


def _fit_circle_algebraic(points: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """Kasa algebraic circle fit, translated to the point mean for conditioning."""
    if points.shape[0] < 3:
        return None
    mean = points.mean(axis=0)
    q = points - mean
    design = np.column_stack([2.0 * q[:, 0], 2.0 * q[:, 1], np.ones(len(q))])
    rhs = np.sum(q * q, axis=1)
    try:
        solution, _, rank, _ = np.linalg.lstsq(design, rhs, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if rank < 3:
        return None
    cx, cy = float(solution[0]), float(solution[1])
    radius_sq = float(solution[2]) + cx * cx + cy * cy
    if radius_sq <= 1e-9:
        return None
    return float(mean[0] + cx), float(mean[1] + cy), float(math.sqrt(radius_sq))


def _refine_circle_geometric(
    points: np.ndarray, center: Tuple[float, float], radius: float
) -> Tuple[float, float, float]:
    """Gauss-Newton minimisation of the geometric radial distance."""
    cx, cy, r = float(center[0]), float(center[1]), float(radius)
    for _ in range(8):
        dx = points[:, 0] - cx
        dy = points[:, 1] - cy
        distance = np.hypot(dx, dy)
        distance = np.maximum(distance, 1e-9)
        residual = distance - r
        jacobian = np.column_stack([-dx / distance, -dy / distance, -np.ones(len(points))])
        try:
            step, _, rank, _ = np.linalg.lstsq(jacobian, -residual, rcond=None)
        except np.linalg.LinAlgError:
            break
        if rank < 3:
            break
        cx += float(step[0])
        cy += float(step[1])
        r += float(step[2])
        if abs(step[0]) + abs(step[1]) + abs(step[2]) < 1e-4:
            break
    return cx, cy, r


def _robust_circle_fit(
    points: np.ndarray,
    radius_range: Tuple[float, float],
    max_iterations: int = 2,
) -> Optional[Tuple[float, float, float, np.ndarray]]:
    """Fit a circle with MAD-based outlier rejection.

    Returns:
        ``(cx, cy, radius, inlier_mask)`` or ``None`` if the fit failed.
    """
    rmin, rmax = radius_range
    mask = np.ones(len(points), dtype=bool)
    fit: Optional[Tuple[float, float, float]] = None
    for _ in range(max_iterations + 1):
        subset = points[mask]
        if len(subset) < 3:
            return None
        algebraic = _fit_circle_algebraic(subset)
        if algebraic is None:
            return None
        fit = _refine_circle_geometric(subset, algebraic[:2], algebraic[2])
        cx, cy, radius = fit
        if not (rmin * 0.5 <= radius <= rmax * 1.5):
            return None
        residuals = np.abs(np.hypot(points[:, 0] - cx, points[:, 1] - cy) - radius)
        median = float(np.median(residuals[mask]))
        mad = _MAD_TO_SIGMA * float(np.median(np.abs(residuals[mask] - median)))
        threshold = max(0.40, 3.0 * mad)
        updated = residuals <= threshold
        # Never discard more than half of the edge points in one pass.
        if updated.sum() < max(6, int(0.5 * len(points))):
            break
        if np.array_equal(updated, mask):
            mask = updated
            break
        mask = updated
    if fit is None:
        return None
    subset = points[mask]
    algebraic = _fit_circle_algebraic(subset)
    if algebraic is None:
        return None
    cx, cy, radius = _refine_circle_geometric(subset, algebraic[:2], algebraic[2])
    if not (rmin * 0.5 <= radius <= rmax * 1.5):
        return None
    return cx, cy, radius, mask


def _ring_median(gray32: np.ndarray, cx: float, cy: float, radius: float,
                 samples: int = 72) -> float:
    """Median intensity sampled on a circle of the given radius."""
    angles = np.linspace(0.0, _TWO_PI, samples, endpoint=False, dtype=np.float32)
    xs = (cx + radius * np.cos(angles)).reshape(1, -1).astype(np.float32)
    ys = (cy + radius * np.sin(angles)).reshape(1, -1).astype(np.float32)
    values = cv2.remap(
        gray32, xs, ys, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    return float(np.median(values))


def _fit_ellipticity(points: np.ndarray, support: float) -> float:
    """Return the ellipse major/minor ratio, or 1.0 when unreliable."""
    if len(points) < 10 or support < 0.40:
        return 1.0
    try:
        ellipse = cv2.fitEllipse(points.astype(np.float32).reshape(-1, 1, 2))
    except cv2.error:
        return 1.0
    axes = ellipse[1]
    minor = float(min(axes))
    major = float(max(axes))
    if minor <= 1e-6:
        return 1.0
    return float(np.clip(major / minor, 1.0, 3.0))


def _confidence(
    radius: float,
    support: float,
    coverage: float,
    rms_error: float,
    contrast: float,
    ellipticity: float,
) -> float:
    """Combine the quality components into one score in ``[0, 1]``."""
    q_round = _clamp01(1.0 - rms_error / (0.02 * radius + 0.35))
    q_support = _clamp01(support)
    q_coverage = _clamp01(coverage)
    q_contrast = 1.0 - math.exp(-max(contrast, 0.0) / 8.0)
    score = (
        (q_round ** 0.35)
        * (q_support ** 0.25)
        * (q_coverage ** 0.20)
        * (q_contrast ** 0.20)
    )
    if ellipticity > 1.05:
        score *= max(0.45, 1.0 / (1.0 + 2.0 * (ellipticity - 1.05)))
    return _clamp01(score)


# ---------------------------------------------------------------------------
# Locator
# ---------------------------------------------------------------------------


@dataclass
class CircleLocator:
    """Parameterised circle locator.

    The defaults target a matte dark disc (10-100 px radius) on a lighter
    stage.  All tolerances are per-detection; invalid detections are
    returned with ``valid=False`` and a ``reason`` instead of raising.

    Attributes:
        radius_range: Accepted radius interval in pixels.
        polarity: ``"dark"``, ``"light"`` or ``"auto"`` (try both).
        min_confidence: Hard gate: a score below this marks the detection
            invalid with reason ``low_confidence``.
        num_rays: Number of radial rays used for edge extraction.
        max_candidates: Maximum coarse candidates refined per frame.
        blur_sigma: Gaussian sigma along the radial intensity profile before
            differentiation; larger values are more noise tolerant.
        min_support: Minimum fraction of rays with an inlier edge.
        max_roundness: Maximum RMS radial residual divided by radius.
        min_contrast: Minimum inside/outside intensity difference in sigma.
        min_gradient: Minimum radial gradient (gray levels / px).
        hough_mode: ``"fallback"`` runs Hough only when Otsu contours yield
            no candidate, ``"always"`` augments every frame, ``"never"``
            disables it.  Hough is the expensive stage.
        hough_param1: Canny high threshold for Hough (lower = more sensitive).
        hough_param2: ``HOUGH_GRADIENT_ALT`` perfectness threshold.
    """

    radius_range: Tuple[float, float] = (6.0, 400.0)
    polarity: str = "auto"
    min_confidence: float = 0.35
    num_rays: int = 180
    max_candidates: int = 6
    blur_sigma: float = 1.0
    min_support: float = 0.45
    max_roundness: float = 0.05
    min_contrast: float = 3.0
    min_gradient: float = 2.0
    hough_mode: str = "fallback"
    hough_param1: float = 100.0
    hough_param2: float = 0.75

    # -- public API ------------------------------------------------------

    def find(
        self,
        image: np.ndarray,
        *,
        roi: Optional[Sequence[int]] = None,
        expected_center: Optional[Sequence[float]] = None,
        expected_radius: Optional[float] = None,
    ) -> CircleDetection:
        """Return the highest-confidence valid circle.

        Args:
            image: Grayscale or BGR image.
            roi: Optional ``(x1, y1, x2, y2)`` inclusive search region.
            expected_center: Optional prior ``(u, v)``; candidates farther
                than half the image diagonal are discarded.
            expected_radius: Optional prior radius in pixels; when supplied
                the coarse radius interval is narrowed to +/-40 %.

        Returns:
            The best valid detection, or the best refined candidate with
            ``valid=False``.  When nothing was refined, an invalid
            placeholder is returned.
        """
        detections = self.find_all(
            image,
            roi=roi,
            expected_center=expected_center,
            expected_radius=expected_radius,
        )
        for detection in detections:
            if detection.valid:
                return detection
        if detections:
            return detections[0]
        return CircleDetection.empty()

    def find_all(
        self,
        image: np.ndarray,
        *,
        roi: Optional[Sequence[int]] = None,
        expected_center: Optional[Sequence[float]] = None,
        expected_radius: Optional[float] = None,
    ) -> List[CircleDetection]:
        """Return every refined candidate, sorted by descending confidence."""
        gray8 = _as_uint8(image)
        height, width = gray8.shape[:2]
        x1, y1, x2, y2 = _resolve_roi(roi, width, height)
        crop = gray8[y1:y2 + 1, x1:x2 + 1]
        gray32 = gray8.astype(np.float32)
        noise = estimate_noise_sigma(crop)
        radius_range = self._effective_radius_range(expected_radius)

        candidates: List[_Candidate] = []
        for polarity in self._polarities():
            candidates.extend(
                _contour_candidates(
                    crop,
                    x1,
                    y1,
                    radius_range,
                    polarity,
                    max(4, self.max_candidates),
                )
            )
        run_hough = self.hough_mode == "always" or (
            self.hough_mode == "fallback" and not candidates
        )
        if self.hough_mode not in ("fallback", "always", "never"):
            raise ValueError("hough_mode must be 'fallback', 'always' or 'never'")
        if run_hough:
            candidates.extend(
                _hough_candidates(
                    crop,
                    x1,
                    y1,
                    radius_range,
                    self.hough_param1,
                    self.hough_param2,
                    max(4, self.max_candidates),
                )
            )
        if expected_center is not None:
            max_distance = 0.5 * math.hypot(width, height)
            candidates = [
                c for c in candidates
                if math.hypot(c.cx - expected_center[0], c.cy - expected_center[1])
                <= max_distance
            ]
        candidates = _dedupe(candidates)[: max(1, self.max_candidates)]

        detections: List[CircleDetection] = []
        for candidate in candidates:
            detection = self._refine(gray32, candidate, radius_range, noise)
            if detection is not None:
                detections.append(detection)
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def visualize(
        self,
        image: np.ndarray,
        detection: Optional[CircleDetection] = None,
        *,
        draw_edges: bool = False,
        wait_time: int = -1,
    ) -> np.ndarray:
        """Draw a detection on a writable BGR copy of *image*."""
        canvas = to_bgr(image)
        if detection is None or not np.isfinite(detection.center[0]):
            cv2.putText(
                canvas, "no circle", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 255), 2, cv2.LINE_AA,
            )
            return canvas
        cx, cy = int(round(detection.center[0])), int(round(detection.center[1]))
        radius = max(1, int(round(detection.radius)))
        color = (0, 200, 0) if detection.valid else (0, 0, 255)
        cv2.circle(canvas, (cx, cy), radius, color, 2, cv2.LINE_AA)
        cv2.drawMarker(canvas, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
        if draw_edges and detection.edge_points:
            for u, v in detection.edge_points:
                cv2.circle(canvas, (int(round(u)), int(round(v))), 1, (255, 0, 255), -1)
        text = (
            f"conf={detection.confidence:.2f} r={detection.radius:.2f}px "
            f"support={detection.support:.2f} rms={detection.rms_error:.2f}px "
            f"{detection.reason}"
        )
        cv2.putText(
            canvas, text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (0, 0, 0), 3, cv2.LINE_AA,
        )
        cv2.putText(
            canvas, text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (255, 255, 255), 1, cv2.LINE_AA,
        )
        if wait_time >= 0:
            cv2.imshow("circle", canvas)
            cv2.waitKey(wait_time)
            cv2.destroyAllWindows()
        return canvas

    # -- serialisation ---------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable configuration dictionary."""
        return {
            "radius_range": [float(self.radius_range[0]), float(self.radius_range[1])],
            "polarity": self.polarity,
            "min_confidence": float(self.min_confidence),
            "num_rays": int(self.num_rays),
            "max_candidates": int(self.max_candidates),
            "blur_sigma": float(self.blur_sigma),
            "min_support": float(self.min_support),
            "max_roundness": float(self.max_roundness),
            "min_contrast": float(self.min_contrast),
            "min_gradient": float(self.min_gradient),
            "hough_mode": self.hough_mode,
            "hough_param1": float(self.hough_param1),
            "hough_param2": float(self.hough_param2),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CircleLocator":
        """Build a locator from :meth:`to_dict` output (unknown keys ignored)."""
        radius_range = data.get("radius_range", (6.0, 400.0))
        return cls(
            radius_range=(float(radius_range[0]), float(radius_range[1])),
            polarity=str(data.get("polarity", "auto")),
            min_confidence=float(data.get("min_confidence", 0.35)),
            num_rays=int(data.get("num_rays", 180)),
            max_candidates=int(data.get("max_candidates", 6)),
            blur_sigma=float(data.get("blur_sigma", 1.0)),
            min_support=float(data.get("min_support", 0.45)),
            max_roundness=float(data.get("max_roundness", 0.08)),
            min_contrast=float(data.get("min_contrast", 3.0)),
            min_gradient=float(data.get("min_gradient", 2.0)),
            hough_mode=str(data.get("hough_mode", "fallback")),
            hough_param1=float(data.get("hough_param1", 100.0)),
            hough_param2=float(data.get("hough_param2", 0.75)),
        )

    # -- internals -------------------------------------------------------

    def _polarities(self) -> List[str]:
        if self.polarity == "auto":
            return ["dark", "light"]
        if self.polarity not in ("dark", "light"):
            raise ValueError("polarity must be 'dark', 'light' or 'auto'")
        return [self.polarity]

    def _effective_radius_range(
        self, expected_radius: Optional[float]
    ) -> Tuple[float, float]:
        rmin, rmax = (float(self.radius_range[0]), float(self.radius_range[1]))
        if rmin <= 0 or rmax <= rmin:
            raise ValueError("radius_range must be a positive increasing interval")
        if expected_radius is not None and expected_radius > 0:
            rmin = max(rmin, 0.6 * float(expected_radius))
            rmax = min(rmax, 1.4 * float(expected_radius))
            if rmax <= rmin:
                rmax = rmin + 1.0
        return rmin, rmax

    def _refine(
        self,
        gray32: np.ndarray,
        candidate: _Candidate,
        radius_range: Tuple[float, float],
        noise_sigma: float,
    ) -> Optional[CircleDetection]:
        """Refine one coarse candidate into a full detection."""
        rmin, rmax = radius_range
        if not (rmin <= candidate.radius <= rmax):
            return None
        n_rays = max(24, int(self.num_rays))
        angles = np.linspace(0.0, _TWO_PI, n_rays, endpoint=False)
        cx, cy, radius = candidate.cx, candidate.cy, candidate.radius

        inside = _ring_median(gray32, cx, cy, 0.50 * radius)
        outside = _ring_median(gray32, cx, cy, 1.45 * radius)
        sign = 1.0 if inside < outside else -1.0

        points = np.empty((0, 2), dtype=np.float64)
        ray_indices = np.empty(0, dtype=np.int64)
        final_mask = np.empty(0, dtype=bool)
        fit = None
        for pass_index in range(2):
            span = max(0.22 * radius, 3.0) if pass_index == 0 else max(0.12 * radius, 2.0)
            profiles, sample_radii, step = _sample_profiles(
                gray32, cx, cy, radius, angles, span
            )
            points, ray_indices, _, _ = _peak_points(
                profiles, sample_radii, step, angles, cx, cy, sign,
                self.min_gradient, self.blur_sigma,
            )
            if len(points) < max(12, int(0.30 * n_rays)):
                return None
            fit = _robust_circle_fit(points, radius_range)
            if fit is None:
                return None
            cx, cy, radius, final_mask = fit
            if not (rmin <= radius <= rmax):
                return None
            if math.hypot(cx - candidate.cx, cy - candidate.cy) > 0.5 * candidate.radius + 3.0:
                return None

        if fit is None or len(points) == 0:
            return None
        inlier_points = points[final_mask]
        inlier_rays = ray_indices[final_mask]
        if len(inlier_points) < 3:
            return None
        residuals = np.abs(
            np.hypot(inlier_points[:, 0] - cx, inlier_points[:, 1] - cy) - radius
        )
        rms_error = float(math.sqrt(float(np.mean(residuals ** 2))))
        max_error = float(np.max(residuals))
        support = float(len(inlier_points)) / float(n_rays)
        coverage = self._coverage(inlier_rays, n_rays)
        ellipticity = _fit_ellipticity(inlier_points, support)

        # Sample just inside/outside the fitted edge: this measures the edge
        # contrast itself and stays valid for a thin ring target as well.
        inside = _ring_median(gray32, cx, cy, 0.85 * radius)
        outside = _ring_median(gray32, cx, cy, 1.15 * radius)
        contrast = abs(inside - outside) / max(float(noise_sigma), 0.5)

        reasons: List[str] = []
        if support < self.min_support:
            reasons.append("low_support")
        if rms_error > max(0.6, self.max_roundness * radius):
            reasons.append("not_round")
        if contrast < self.min_contrast:
            reasons.append("low_contrast")
        confidence = _confidence(
            radius, support, coverage, rms_error, contrast, ellipticity
        )
        if confidence < self.min_confidence:
            reasons.append("low_confidence")
        height, width = gray32.shape[:2]
        bbox = (
            max(0, int(math.floor(cx - radius))),
            max(0, int(math.floor(cy - radius))),
            min(width - 1, int(math.ceil(cx + radius))),
            min(height - 1, int(math.ceil(cy + radius))),
        )
        return CircleDetection(
            center=(float(cx), float(cy)),
            radius=float(radius),
            confidence=confidence,
            valid=not reasons,
            reason="ok" if not reasons else "+".join(reasons),
            method=candidate.method,
            polarity=candidate.polarity,
            support=support,
            coverage=coverage,
            rms_error=rms_error,
            max_error=max_error,
            ellipticity=ellipticity,
            contrast=float(contrast),
            inliers=int(len(inlier_points)),
            rays=int(n_rays),
            bbox=bbox,
            edge_points=[(float(u), float(v)) for u, v in inlier_points],
        )

    @staticmethod
    def _coverage(ray_indices: np.ndarray, total_rays: int) -> float:
        """Fraction of angular sectors that received at least one inlier."""
        if total_rays <= 0 or len(ray_indices) == 0:
            return 0.0
        sectors = np.zeros(_COVERAGE_SECTORS, dtype=bool)
        bins = (ray_indices.astype(np.int64) * _COVERAGE_SECTORS) // max(1, total_rays)
        bins = np.clip(bins, 0, _COVERAGE_SECTORS - 1)
        sectors[bins] = True
        return float(sectors.mean())


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------


def find_circle_center(
    image: np.ndarray,
    *,
    radius_range: Tuple[float, float] = (6.0, 400.0),
    roi: Optional[Sequence[int]] = None,
    polarity: str = "auto",
    expected_center: Optional[Sequence[float]] = None,
    expected_radius: Optional[float] = None,
    **locator_kwargs: Any,
) -> CircleDetection:
    """Find the highest-confidence circle centre in one call.

    Args:
        image: Grayscale or BGR image.
        radius_range: Accepted radius interval in pixels.
        roi: Optional ``(x1, y1, x2, y2)`` inclusive search region.
        polarity: ``"dark"``, ``"light"`` or ``"auto"``.
        expected_center: Optional prior centre ``(u, v)``.
        expected_radius: Optional prior radius (narrows the search interval).
        **locator_kwargs: Extra :class:`CircleLocator` fields.

    Returns:
        The best :class:`CircleDetection`; check ``valid`` before use.
    """
    locator = CircleLocator(
        radius_range=radius_range,
        polarity=polarity,
        **locator_kwargs,
    )
    return locator.find(
        image,
        roi=roi,
        expected_center=expected_center,
        expected_radius=expected_radius,
    )
