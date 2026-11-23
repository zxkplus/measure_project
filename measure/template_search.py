"""Template preparation and search over an explicit, immutable support.

The support is a straightened crop plus an explicit valid-pixel mask and an
anchor expressed in support coordinates.  Searching never mutates the prepared
arrays: every variant is built from a private copy, and rotated supports whose
bounding box exceeds the search image are handled by padding instead of being
silently dropped (the historical D07 defect).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

from measure.computation_types import (
    Array,
    ComputationContractError,
    Mapping2D,
    Observation,
    RotatedBox,
    Space,
    XY,
    contract_error,
    invalid,
    make_mapping,
    observed,
    require_finite,
    require_finite_xy,
    require_xy_array,
)
from measure.preprocessing import RawSpec, preprocess_pixels
from measure.rectify import ResampleOptions, crop_region

#: Fraction of a support that must be inside the image for a variant to be usable.
MIN_VALID_FRACTION = 1.0


@dataclass(frozen=True)
class SupportSpec:
    """Teaching input: support box, reference anchor and optional valid mask."""

    box: RotatedBox
    anchor_reference: XY
    mask: Optional[Array] = None
    origin: str = "continuous"

    def __post_init__(self) -> None:
        if not isinstance(self.box, RotatedBox):
            raise contract_error("contract.invalid_parameter",
                                 "box must be a RotatedBox", "support.box")
        object.__setattr__(self, "anchor_reference",
                           require_finite_xy(self.anchor_reference, "support.anchor_reference"))
        if self.origin not in ("continuous", "pixel_center"):
            raise contract_error("contract.invalid_parameter",
                                 f"unknown origin {self.origin!r}", "support.origin")
        if self.mask is not None:
            mask = np.asarray(self.mask)
            height, width = int(self.box.size_wh[1]), int(self.box.size_wh[0])
            if mask.shape != (height, width):
                raise contract_error("contract.invalid_shape",
                                     "mask must match the integer support size",
                                     "support.mask")
            object.__setattr__(self, "mask", np.array(mask, dtype=bool, copy=True))


@dataclass(frozen=True)
class SearchOptions:
    """NCC localization policy; angles are clockwise radians, scales dimensionless."""

    score_threshold: float = 0.5
    max_matches: int = 1
    max_overlap: float = 0.3
    subpixel: bool = True
    coarse_fine: bool = True
    coarse_score_threshold: float = 0.3
    angle_min_rad: float = 0.0
    angle_max_rad: float = 0.0
    angle_step_rad: float = math.pi / 180.0
    scale_min: float = 1.0
    scale_max: float = 1.0
    scale_step: float = 0.02
    coarse_angle_step_rad: float = math.radians(5.0)
    coarse_scale_step: float = 0.1
    pyramid_levels: int = 0
    pyramid_max_template_size: int = 128
    fine_search_radius_px: float = 24.0
    fine_candidates: Optional[int] = None
    angle_list: Optional[Tuple[float, ...]] = None
    scale_list: Optional[Tuple[float, ...]] = None
    inplace_variants: bool = False

    def __post_init__(self) -> None:
        require_finite((self.score_threshold, self.coarse_score_threshold), "search.score")
        for value, name in ((self.score_threshold, "score_threshold"),
                            (self.coarse_score_threshold, "coarse_score_threshold")):
            if not (-1.0 <= value <= 1.0):
                raise contract_error("contract.invalid_parameter",
                                     f"{name} must be in [-1, 1]", f"search.{name}")
        if self.max_matches < 0:
            raise contract_error("contract.invalid_parameter", "max_matches must be >= 0",
                                 "search.max_matches")
        if self.angle_min_rad > self.angle_max_rad:
            raise contract_error("contract.invalid_parameter",
                                 "angle_min_rad must not exceed angle_max_rad",
                                 "search.angle_min_rad")
        if self.angle_step_rad <= 0.0:
            raise contract_error("contract.invalid_parameter", "angle_step_rad must be positive",
                                 "search.angle_step_rad")
        if self.scale_min <= 0.0 or self.scale_max < self.scale_min:
            raise contract_error("contract.invalid_parameter",
                                 "scale range must be positive and ordered", "search.scale_min")
        if self.scale_step <= 0.0:
            raise contract_error("contract.invalid_parameter", "scale_step must be positive",
                                 "search.scale_step")
        if self.fine_candidates is not None and self.fine_candidates < 1:
            raise contract_error("contract.invalid_parameter",
                                 "fine_candidates must be >= 1 or None",
                                 "search.fine_candidates")

    def resolved_fine_candidates(self) -> Optional[int]:
        """Return how many coarse neighbourhoods the fine stage may refine.

        ``None`` refines every neighbourhood the coarse stage kept, which is the
        accurate-and-stable default: equal-looking coarse peaks can rank within a
        hundredth of each other, so cutting the set can report a different — but
        equally valid — rotation than the full rescan would.

        The bound exists because a symmetric part searched over a full turn
        produces many near-equal peaks, and each costs a full-resolution rescan.
        ``MEASURE_TEMPLATE_FINE_CANDIDATES`` supplies a deployment-wide bound so
        a site can trade accuracy for cost without editing models.
        """
        if self.fine_candidates is not None:
            return int(self.fine_candidates)
        configured = os.environ.get("MEASURE_TEMPLATE_FINE_CANDIDATES", "").strip()
        if configured.isdigit() and int(configured) > 0:
            return int(configured)
        return None

    def angles(self) -> List[float]:
        """Return the requested clockwise angles in radians (inclusive)."""
        if self.angle_list is not None:
            return [float(value) for value in self.angle_list]
        if self.angle_max_rad - self.angle_min_rad <= 1e-15:
            return [float(self.angle_min_rad)]
        count = int(round((self.angle_max_rad - self.angle_min_rad) / self.angle_step_rad))
        return [self.angle_min_rad + index * self.angle_step_rad
                for index in range(count + 1)]

    def scales(self) -> List[float]:
        """Return the requested scales (inclusive)."""
        if self.scale_list is not None:
            return [float(value) for value in self.scale_list]
        if self.scale_max - self.scale_min <= 1e-15:
            return [float(self.scale_min)]
        count = int(round((self.scale_max - self.scale_min) / self.scale_step))
        return [self.scale_min + index * self.scale_step for index in range(count + 1)]


@dataclass(frozen=True)
class PreparedSupport:
    """Immutable support pixels, valid mask, anchor and reference mapping."""

    pixels: Array
    valid_mask: Array
    anchor_support: XY
    support_to_reference: Mapping2D
    reference_space: Space
    support_space: Space
    mask: Optional[Array] = None
    #: Pixel convention of ``pixels``; see :func:`_variant`.
    origin: str = "continuous"


@dataclass(frozen=True)
class TemplateMatch:
    """One located instance with its pose and support-to-image mapping."""

    center: XY
    anchor: XY
    angle_rad: float
    scale: float
    score: float
    support_to_image: Mapping2D
    peak_xy: Optional[Tuple[int, int]] = None
    size_wh: Tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True)
class MatchSet:
    """All selected instances of one search call."""

    matches: Tuple[TemplateMatch, ...]


@dataclass(frozen=True)
class RenderedVariant:
    """One rotated/scaled support, ready to correlate against a search image."""

    pixels: Array
    mask: Optional[Array]
    #: Matrix mapping support coordinates into variant coordinates.
    matrix: np.ndarray
    #: Support centre expressed in variant coordinates.
    center: XY
    size_wh: Tuple[int, int]


def _variant_geometry(shape: Tuple[int, ...], angle_rad: float, scale: float,
                      origin: str) -> Tuple[np.ndarray, np.ndarray,
                                            Tuple[int, int], Tuple[float, float]]:
    """Return the linear part, integer origin, output size and support centre.

    Args:
        shape: Support array shape; only the two leading dimensions are read.
        angle_rad: Clockwise pose angle in radians.
        scale: Dimensionless pose scale.
        origin: ``continuous`` or ``pixel_center``.

    Returns:
        ``(linear, origin_xy, size_wh, support_centre)`` in the requested
        convention.

    Raises:
        ComputationContractError: On an unknown origin policy.
    """
    height, width = int(shape[0]), int(shape[1])
    alpha = scale * math.cos(angle_rad)
    beta = scale * math.sin(angle_rad)
    linear = np.array([[alpha, -beta], [beta, alpha]], dtype=np.float64)
    if origin == "pixel_center":
        # Pixel-index geometry: corner 0 is the centre of pixel 0 and the far
        # corner is the centre of pixel size - 1, an inclusive box.
        center_x, center_y = (width - 1) / 2.0, (height - 1) / 2.0
        corners = np.array([[0.0, 0.0], [width - 1.0, 0.0],
                            [0.0, height - 1.0], [width - 1.0, height - 1.0]])
        inclusive = 1
    elif origin == "continuous":
        # Sampling geometry: pixel i covers [i, i + 1), so the support spans
        # [0, size] and its centre sits on size / 2.
        center_x, center_y = width / 2.0, height / 2.0
        corners = np.array([[0.0, 0.0], [width, 0.0],
                            [0.0, height], [width, height]])
        inclusive = 0
    else:
        raise contract_error("contract.invalid_parameter",
                             f"unknown variant origin {origin!r}", "origin")
    rotated = corners @ linear.T
    low = np.floor(rotated.min(axis=0))
    high = np.ceil(rotated.max(axis=0))
    size = (max(1, int(round(high[0] - low[0])) + inclusive),
            max(1, int(round(high[1] - low[1])) + inclusive))
    return linear, low, size, (center_x, center_y)


def variant_size(pixels: Array, angle_rad: float, scale: float, *,
                 origin: str = "continuous") -> Tuple[int, int]:
    """Return the output size one variant would occupy, without rendering it.

    Callers use this to reject poses that cannot fit their search window before
    paying for a warp.

    Args:
        pixels: Support pixels.
        angle_rad: Clockwise pose angle in radians.
        scale: Dimensionless pose scale.
        origin: ``continuous`` or ``pixel_center``.

    Returns:
        The ``(width, height)`` the variant would have.
    """
    return _variant_geometry(np.shape(pixels), angle_rad, scale, origin)[2]


def render_variant(pixels: Array, angle_rad: float, scale: float,
                   mask: Optional[Array] = None, *, inplace: bool = False,
                   origin: str = "continuous") -> RenderedVariant:
    """Render one rotated/scaled variant of a support.

    The variant is placed on the same pixel grid as the search image: the
    rotated footprint is snapped to an integer origin (``floor`` of the rotated
    minimum), so the correlation compares like-sampled pixels. Leaving the
    origin on the fractional rotated minimum instead misaligns the variant by
    up to half a pixel, which blurs the score peak and biases the recovered
    pose.

    Args:
        pixels: Support pixels.
        angle_rad: Clockwise pose angle in radians.
        scale: Dimensionless pose scale.
        mask: Optional support footprint mask.
        inplace: Render inside the original canvas instead of a tight box. The
            historical template search needs the template size preserved.
        origin: ``continuous`` treats pixel ``i`` as covering ``[i, i + 1)`` and
            centres the support on ``size / 2``; ``pixel_center`` treats pixel
            ``i`` as being point ``i`` and centres on ``(size - 1) / 2``.

    Returns:
        The variant pixels, its footprint mask (``None`` when empty), the
        support-to-variant matrix and the support centre in variant
        coordinates.

    Raises:
        ComputationContractError: On an unknown origin policy.
    """
    height, width = pixels.shape[:2]
    linear, low, size, (center_x, center_y) = _variant_geometry(
        pixels.shape, angle_rad, scale, origin)
    if inplace:
        # Rotate inside the original canvas, matching the historical template
        # search: the output keeps the template size and centre.
        out_width, out_height = width, height
        translation = np.array([center_x, center_y]) - linear @ np.array(
            [center_x, center_y])
    else:
        out_width, out_height = size
        # The fractional rotated minimum is snapped to a whole pixel so the
        # variant samples the same grid as the search image.
        translation = -low
    matrix = np.array([
        [linear[0, 0], linear[0, 1], translation[0]],
        [linear[1, 0], linear[1, 1], translation[1]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    center = matrix @ np.array([center_x, center_y, 1.0], dtype=np.float64)
    variant = cv2.warpAffine(pixels, matrix[:2, :], (out_width, out_height),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    footprint = mask if mask is not None else np.ones((height, width), dtype=bool)
    warped_mask = cv2.warpAffine(np.asarray(footprint, dtype=np.float32),
                                 matrix[:2, :], (out_width, out_height),
                                 flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    variant_mask = (warped_mask > 0.5).astype(np.float32)
    if variant_mask.sum() <= 0:
        variant_mask = None
    return RenderedVariant(pixels=variant, mask=variant_mask, matrix=matrix,
                           center=(float(center[0]), float(center[1])),
                           size_wh=(int(out_width), int(out_height)))


def _has_texture(pixels: Array) -> bool:
    """Return whether the support carries enough variation to correlate."""
    _, deviation = cv2.meanStdDev(pixels)
    return bool(float(deviation[0][0]) ** 2 > 1e-12)


def prepare_support(
    reference: Array, spec: SupportSpec, preprocessing: Any, *,
    reference_space: Space, support_space: Space,
) -> Observation[PreparedSupport]:
    """Straighten one support and pin its pixels, mask and anchor.

    Args:
        reference: ``uint8`` reference image.
        spec: Support box, reference anchor and optional mask.
        preprocessing: A preprocessing spec; colour-preserving raw is rejected.
        reference_space: Space of the reference image.
        support_space: Space of the produced support.

    Returns:
        A valid prepared support whose arrays are read-only, or an invalid
        observation with ``template_no_texture``.

    Raises:
        ComputationContractError: On unsupported input or a colour-preserving
            preset entering grayscale matching.
    """
    if isinstance(preprocessing, RawSpec) and preprocessing.channel_mode == "preserve":
        raise contract_error("contract.invalid_parameter",
                             "colour-preserving preprocessing cannot enter grayscale matching",
                             "preprocessing.channel_mode")
    processed = preprocess_pixels(reference, preprocessing)
    pixels = processed
    if pixels.dtype != np.uint8:
        pixels = np.clip(pixels, 0, 255).astype(np.uint8)
    crop = crop_region(pixels, spec.box, source=reference_space,
                       destination=support_space, origin=spec.origin)
    support_pixels = np.array(crop.pixels, dtype=np.float32, copy=True)
    if spec.mask is not None:
        valid_mask = np.array(spec.mask, dtype=bool, copy=True)
        support_pixels[~valid_mask] = 0.0
    else:
        valid_mask = np.ones(support_pixels.shape[:2], dtype=bool)
    if not _has_texture(support_pixels[valid_mask]):
        return invalid(support_space, "template_no_texture")
    # reference -> support is the mapping's forward direction
    anchor = crop.mapping.apply(np.asarray([spec.anchor_reference], dtype=np.float64))[0]
    support_pixels.setflags(write=False)
    valid_mask.setflags(write=False)
    prepared = PreparedSupport(
        pixels=support_pixels,
        valid_mask=valid_mask,
        anchor_support=(float(anchor[0]), float(anchor[1])),
        support_to_reference=crop.mapping,
        reference_space=reference_space,
        support_space=support_space,
        mask=None if spec.mask is None else np.array(spec.mask, dtype=bool, copy=True),
        origin=spec.origin,
    )
    return observed(support_space, prepared, {"support_shape": support_pixels.shape})


def has_texture(pixels: Array) -> bool:
    """Return whether pixels carry enough variation to correlate.

    Args:
        pixels: Support or search-window pixels.

    Returns:
        ``True`` when the standard deviation is non-zero beyond round-off.
    """
    return _has_texture(pixels)


def peak_argmax(scores: Array) -> Tuple[int, int, float]:
    """Return the strongest cell of a correlation map.

    Args:
        scores: Correlation score map.

    Returns:
        ``(x, y, value)`` of the maximum.
    """
    _, value, _, position = cv2.minMaxLoc(np.asarray(scores, dtype=np.float32))
    return int(position[0]), int(position[1]), float(value)


def rotated_box_overlap(center_a: XY, size_a: XY, angle_a: float,
                        center_b: XY, size_b: XY, angle_b: float) -> float:
    """Return the intersection over the smaller rotated box.

    Args:
        center_a: First box centre.
        size_a: First box ``(width, height)``.
        angle_a: First box rotation in degrees.
        center_b: Second box centre.
        size_b: Second box ``(width, height)``.
        angle_b: Second box rotation in degrees.

    Returns:
        Intersection area divided by the smaller box area, or ``0`` when the
        boxes do not intersect.
    """
    first = ((float(center_a[0]), float(center_a[1])),
             (float(size_a[0]), float(size_a[1])), float(angle_a))
    second = ((float(center_b[0]), float(center_b[1])),
              (float(size_b[0]), float(size_b[1])), float(angle_b))
    _, polygon = cv2.rotatedRectangleIntersection(first, second)
    if polygon is None:
        return 0.0
    area = float(cv2.contourArea(polygon))
    smaller = min(float(size_a[0]) * float(size_a[1]), float(size_b[0]) * float(size_b[1]))
    return area / smaller if smaller > 0.0 else 0.0


def decimate_matrix(factor: float) -> np.ndarray:
    """Return the full-frame to coarse-frame matrix of a decimation.

    Args:
        factor: Decimation factor in ``(0, 1]``.

    Returns:
        The 3x3 matrix mapping full-resolution pixel centres onto the coarse grid.
    """
    matrix = np.eye(3, dtype=np.float64)
    matrix[0, 0] = matrix[1, 1] = float(factor)
    matrix[0, 2] = matrix[1, 2] = (float(factor) - 1.0) / 2.0
    return matrix


def decimate_pixels(image: Array, factor: float, *, sigma: float
                    ) -> Tuple[Array, np.ndarray]:
    """Low-pass and decimate pixels onto a coarser grid.

    Args:
        image: Full-resolution ``float32`` pixels.
        factor: Decimation factor in ``(0, 1]``.
        sigma: Gaussian standard deviation applied before sampling.

    Returns:
        The decimated pixels and the 3x3 full-frame to coarse-frame matrix.
    """
    if not 0.0 < factor <= 1.0:
        raise contract_error("contract.invalid_parameter",
                             "factor must be in (0, 1]", "factor")
    height, width = image.shape[:2]
    affine = np.array([[factor, 0.0, (factor - 1.0) / 2.0],
                       [0.0, factor, (factor - 1.0) / 2.0]], dtype=np.float64)
    smoothed = cv2.GaussianBlur(image, (0, 0), float(sigma))
    coarse = cv2.warpAffine(smoothed, affine,
                            (max(1, int(round(width * factor))),
                             max(1, int(round(height * factor)))))
    matrix = np.eye(3, dtype=np.float64)
    matrix[:2, :] = affine
    return coarse, matrix


def correlate(search: Array, template: Array, mask: Optional[Array] = None) -> Array:
    """Return the normalised cross-correlation score map of one template.

    Args:
        search: Search-area pixels.
        template: Rendered variant pixels.
        mask: Optional variant footprint mask; ``None`` uses ordinary NCC.

    Returns:
        A score map with every non-finite value replaced by ``-1``, so masked
        correlation over a constant background cannot produce a false peak.
    """
    area = np.ascontiguousarray(search, dtype=np.float32)
    variant = np.ascontiguousarray(template, dtype=np.float32)
    if mask is None:
        scores = cv2.matchTemplate(area, variant, cv2.TM_CCOEFF_NORMED)
    else:
        scores = cv2.matchTemplate(area, variant, cv2.TM_CCOEFF_NORMED,
                                   mask=np.ascontiguousarray(mask, dtype=np.float32))
    return np.nan_to_num(scores, nan=-1.0, posinf=-1.0, neginf=-1.0)


def subpixel_offset(before: float, centre: float, after: float) -> float:
    """Refine one correlation peak with a local quadratic.

    The vertex sits at ``(before - after) / (2 * (before - 2c + after))``; the
    code has always used ``(before - after)`` over the inverted denominator,
    which is a sign convention carried by every recorded observation. This is
    the single subpixel primitive for every caller: the API layer and the
    measured-point adapter both report positions produced here.

    The offset is rejected (returns ``0.0``) when the parabola degenerates or
    the vertex leaves the centre pixel, so a flat or inverted neighbourhood can
    never shift a peak further than half a pixel.

    Callers pass score samples exactly as they read them (the map is
    ``float32``), and the neighbour terms are deliberately *not* widened to
    ``float64``: the denominator is a difference of nearby samples, so widening
    ``before``/``after`` changes it in the 8th significant digit and shifts the
    reported centre by a few micro-pixels. Every recorded golden position was
    produced with the narrow-precision arithmetic, so widening it would make
    earlier results unreproducible.

    Args:
        before: Score at the pixel before the peak along one axis.
        centre: Score at the peak.
        after: Score at the pixel after the peak along one axis.

    Returns:
        Offset from the integer peak in ``[-0.5, 0.5]``, or ``0.0``.
    """
    denom = float(before - 2 * centre + after)
    offset = float((before - after) / (2 * denom)) if abs(denom) > 1e-8 else 0.0
    return offset if np.isfinite(offset) and abs(offset) <= 0.5 else 0.0


def refine_peak_xy(scores: Array, x: int, y: int,
                   centre: Optional[float] = None) -> Tuple[float, float]:
    """Refine one correlation peak in both axes with :func:`subpixel_offset`.

    Args:
        scores: Correlation map.
        x: Integer peak column.
        y: Integer peak row.
        centre: Score at the peak as the peak finder reported it. Pass it: the
            neighbour samples are read from the map (``float32``) while the
            centre arrives as a ``float``, and that mixed width is part of the
            recorded convention (see :func:`subpixel_offset`). Omitting it
            falls back to the map's own sample.

    Returns:
        Refined ``(x, y)`` in correlation-map coordinates; an axis on the map
        border keeps its integer position.
    """
    height, width = scores.shape
    x, y = int(x), int(y)
    middle = float(scores[y, x]) if centre is None else float(centre)
    refined_x, refined_y = float(x), float(y)
    if 0 < x < width - 1:
        refined_x = float(x) + subpixel_offset(scores[y, x - 1], middle,
                                               scores[y, x + 1])
    if 0 < y < height - 1:
        refined_y = float(y) + subpixel_offset(scores[y - 1, x], middle,
                                               scores[y + 1, x])
    return refined_x, refined_y


def peaks(scores: Array, threshold: float, limit: int = 0,
          suppression_box: Optional[Tuple[float, float]] = None,
          overlap: float = 0.3) -> List[Tuple[float, float, float]]:
    """Return local maxima of a correlation map above a threshold.

    Without ``suppression_box`` this is a vectorised 3x3-dilation local-maximum
    pass: it reports every distinct lobe at once, at the cost of a single
    dilation, instead of millions of Python loop iterations. With it the scan is
    greedy instead: after a peak is accepted, a box of
    ``suppression_box * (1 - overlap)`` around it is blanked, so one instance
    cannot supply several nearby candidates.

    Both modes sort strongest first, so the caller sees a stable order at any
    ``limit``.

    Args:
        scores: Correlation map.
        threshold: Minimum accepted score.
        limit: Optional cap on candidates, keeping the highest scores; ``0``
            keeps every peak.
        suppression_box: Optional ``(width, height)`` of the matched variant;
            enables greedy suppression around each accepted peak.
        overlap: Fractional overlap still counted as the same instance, used
            only when ``suppression_box`` is given.

    Returns:
        ``(x, y, score)`` tuples, strongest first, at integer correlation-map
        cells in both modes. Subpixel refinement belongs to the caller
        (:func:`refine_peak_xy`), so this function never reports a refined
        coordinate that a caller might refine again.
    """
    finite = np.nan_to_num(scores, nan=-1.0, posinf=-1.0, neginf=-1.0)
    if suppression_box is None:
        dilated = cv2.dilate(finite, np.ones((3, 3), dtype=np.float32))
        mask = (finite >= threshold) & (finite >= dilated - 1e-12)
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            return []
        values = finite[ys, xs]
        order = np.argsort(-values)
        if limit and order.size > limit:
            order = order[:limit]
        return [(float(xs[i]), float(ys[i]), float(values[i])) for i in order]

    width, height = suppression_box
    half_w = max(1, int(width * (1 - overlap)))
    half_h = max(1, int(height * (1 - overlap)))
    work = finite.copy()
    found: List[Tuple[float, float, float]] = []
    cap = limit or 1000
    while len(found) < cap:
        x, y, score = peak_argmax(work)
        if score < threshold:
            break
        found.append((float(x), float(y), float(np.clip(score, -1, 1))))
        work[max(0, y - half_h):y + half_h + 1, max(0, x - half_w):x + half_w + 1] = -1.0
    return found


def keep_disjoint(overlap_of, candidates: list, max_overlap: float,
                  limit: int = 0, score_of=None) -> list:
    """Greedy overlap suppression over caller-owned candidate records.

    ``suppress`` applies this to :class:`TemplateMatch`; callers that carry
    their own candidate records (the API layer keeps a pose matrix per
    candidate) pass their own overlap and score accessors instead of
    re-implementing the loop, so the selection policy has exactly one
    implementation.

    Args:
        overlap_of: ``(candidate, kept_candidate) -> float`` in ``[0, 1]``.
        candidates: Candidate records, strongest first after sorting.
        max_overlap: Maximum allowed overlap between kept candidates.
        limit: Optional cap on kept candidates; ``0`` keeps every survivor.
        score_of: ``candidate -> float`` ranking key, higher is stronger. Only
            a tie-breaker for equal scores differs between callers, so every
            caller ranks through this one function.

    Returns:
        Kept candidates, strongest first.
    """
    rank = score_of if score_of is not None else (lambda item: item[0])
    ordered = sorted(candidates, key=rank, reverse=True)
    if max_overlap >= 1.0:
        return ordered[:limit] if limit else ordered
    kept: list = []
    for candidate in ordered:
        if all(overlap_of(candidate, other) <= max_overlap for other in kept):
            kept.append(candidate)
            if limit and len(kept) >= limit:
                break
    return kept


def match_overlap(a: TemplateMatch, b: TemplateMatch) -> float:
    """Return the overlap of two matched instances in their own poses.

    Each instance contributes its true rotated rectangle (``size_wh`` scaled by
    its own pose, rotated by its own angle), so two elongated supports matched
    a few degrees apart are not treated as overlapping just because their
    axis-aligned bounding boxes are. The measure is the intersection area over
    the *smaller* box area, which keeps suppression symmetric in score order and
    reaches 1.0 exactly when one instance sits inside the other.

    Args:
        a: First match.
        b: Second match.

    Returns:
        Overlap in ``[0, 1]``.
    """
    return rotated_box_overlap(
        a.center, a.size_wh, math.degrees(a.angle_rad),
        b.center, b.size_wh, math.degrees(b.angle_rad),
    )


def suppress(candidates: List[TemplateMatch], max_overlap: float,
             limit: int = 0) -> List[TemplateMatch]:
    """Greedy overlap suppression, strongest first.

    Uses :func:`match_overlap` (rotated geometry), the same primitive the API
    layer reports through ``max_overlap``, so a model cannot select a different
    set of instances depending on which caller ran it.

    Args:
        candidates: All located instances.
        max_overlap: Maximum allowed overlap between kept matches; ``1.0``
            disables suppression and ``0.0`` keeps only disjoint instances.
        limit: Optional cap on kept matches; ``0`` keeps every survivor.

    Returns:
        Kept matches sorted by score.
    """
    def overlap_of(candidate: TemplateMatch, other: TemplateMatch) -> float:
        return match_overlap(candidate, other)

    return keep_disjoint(overlap_of, candidates, max_overlap, limit,
                         score_of=lambda item: item.score)




def _coarse_grid(values: List[float], step: float) -> List[float]:
    """Reduce a fine grid to the coarse values spaced at least ``step`` apart."""
    if not values:
        return []
    coarse = [values[0]]
    for value in values[1:]:
        if abs(value - coarse[-1]) >= step - 1e-12:
            coarse.append(value)
    if coarse[-1] != values[-1]:
        coarse.append(values[-1])
    return coarse


def _two_stage(gray: Array, prepared: PreparedSupport, options: SearchOptions,
               image_space: Space, factor: float) -> Observation[MatchSet]:
    """Coarse stage locates neighbourhoods, then each is rescanned at full resolution.

    Two independent knobs decide what the coarse stage does, and they compose
    instead of excluding each other:

    * ``factor`` (from ``pyramid_levels`` and the template side cap) chooses the
      *resolution* the coarse scan runs at. At ``factor < 1`` both the support
      and the search image are low-passed and decimated, so each correlation is
      cheap and the coarse template side stays bounded however large the support
      is.
    * ``coarse_fine`` chooses the *pose density*: the coarse grid is the fine
      grid thinned to ``coarse_angle_step_rad`` / ``coarse_scale_step``, and each
      retained neighbourhood is then rescanned over the fine values near it.

    With ``coarse_fine`` off and ``factor`` 1 this is a single full-resolution
    pass, which is why the previous two functions could be replaced by it.

    Args:
        gray: ``float32`` inspection image.
        prepared: Immutable prepared support.
        options: Search policy carrying both grid resolutions.
        image_space: Space of the inspection image.
        factor: Decimation factor in ``(0, 1]``; ``1.0`` disables decimation.

    Returns:
        Matches located in the full-resolution image.
    """
    fine_angles = options.angles()
    fine_scales = options.scales()
    requested = len(fine_angles) * len(fine_scales)
    decimated = factor < 1.0 - 1e-12

    scan_gray, scan_support = gray, prepared
    if decimated:
        scan_gray, scan_support = _decimate_pair(gray, prepared, factor)

    # The coarse grid is thinned only when the caller asked for a two-pass pose
    # search; decimation alone does not thin it, because the coarse stage still
    # has to *find* the target before any neighbourhood exists.
    if options.coarse_fine and (len(fine_angles) > 1 or len(fine_scales) > 1):
        coarse_angles = _coarse_grid(fine_angles, options.coarse_angle_step_rad)
        coarse_scales = _coarse_grid(fine_scales, options.coarse_scale_step)
    else:
        coarse_angles, coarse_scales = fine_angles, fine_scales

    coarse = _scan_once(scan_gray, scan_support, SearchOptions(
        score_threshold=min(options.score_threshold, options.coarse_score_threshold),
        max_matches=0, max_overlap=options.max_overlap, subpixel=False,
        coarse_fine=False, angle_list=tuple(coarse_angles),
        scale_list=tuple(coarse_scales),
    ), image_space)
    diagnostics = {
        "requested_variant_count": requested,
        "coarse_poses": len(coarse_angles) * len(coarse_scales),
    }
    if decimated:
        diagnostics["pyramid_factor"] = float(factor)
    if not coarse.valid or coarse.value is None:
        diagnostics.update({"searched_variant_count": 0, "pruned_variant_count": 0,
                            "unavailable_variant_count": 0, "search_complete": True})
        return invalid(image_space, "no_matches", diagnostics)

    if not decimated and not options.coarse_fine:
        # One pass already covered the whole requested grid at full resolution,
        # so the coarse matches are final: rescanning them would repeat every
        # correlation for a subpixel refinement that pass did not ask for.
        diagnostics.update({
            "searched_variant_count": len(fine_angles) * len(fine_scales),
            "ncc_evaluations": len(fine_angles) * len(fine_scales),
            "pruned_variant_count": 0,
            "fine_neighborhoods": 0,
            "unavailable_variant_count": coarse.diagnostics.get(
                "unavailable_variant_count", 0),
            "search_complete": coarse.diagnostics.get("search_complete", True),
        })
        # The coarse pass ran with the relaxed coarse threshold and without
        # suppression, because it was meant to feed a refinement. Here it *is*
        # the result, so it must be filtered at the caller's threshold and
        # deduplicated exactly as the two-stage path does.
        accepted = [m for m in coarse.value.matches
                    if m.score >= options.score_threshold - 1e-12]
        selected = suppress(accepted, options.max_overlap)
        if options.max_matches:
            selected = selected[: options.max_matches]
        if not selected:
            return invalid(image_space, "no_matches", diagnostics)
        return observed(image_space, MatchSet(matches=tuple(selected)), diagnostics)

    # How many coarse neighbourhoods deserve a full-resolution rescan. This is
    # a cost/accuracy trade-off, not the caller's match count: one target can
    # produce several coarse peaks, and refining only one of them would make the
    # outcome depend on which peak happened to sort first. The default refines
    # every neighbourhood the coarse stage kept; fine_candidates (or the
    # MEASURE_TEMPLATE_FINE_CANDIDATES environment variable) bounds it for a
    # symmetric part where near-equal peaks each cost a full rescan.
    neighbourhoods = suppress(coarse.value.matches, options.max_overlap,
                              options.resolved_fine_candidates() or 0)

    refined: List[TemplateMatch] = []
    searched = 0
    for anchor in neighbourhoods:
        local_angles = ([a for a in fine_angles
                         if abs(a - anchor.angle_rad) <= options.coarse_angle_step_rad + 1e-12]
                        if options.coarse_fine else [anchor.angle_rad])
        local_scales = ([s for s in fine_scales
                         if abs(s - anchor.scale) <= options.coarse_scale_step + 1e-12]
                        if options.coarse_fine else [anchor.scale])
        if not local_angles or not local_scales:
            continue
        found = _rescan_neighbourhood(gray, prepared, options, anchor, factor,
                                      local_angles, local_scales, image_space)
        searched += len(local_angles) * len(local_scales)
        refined.extend(found)

    selected = suppress(refined, options.max_overlap)
    if options.max_matches:
        selected = selected[: options.max_matches]
    diagnostics.update({
        "searched_variant_count": searched,
        "ncc_evaluations": searched,
        "pruned_variant_count": len(coarse.value.matches),
        "fine_neighborhoods": len(neighbourhoods),
        "unavailable_variant_count": 0,
        "search_complete": True,
    })
    if not selected:
        return invalid(image_space, "no_matches", diagnostics)
    return observed(image_space, MatchSet(matches=tuple(selected)), diagnostics)


def _decimate_pair(gray: Array, prepared: PreparedSupport, factor: float
                   ) -> Tuple[Array, PreparedSupport]:
    """Low-pass and decimate the search image and the support by ``factor``.

    Args:
        gray: Full-resolution ``float32`` inspection image.
        prepared: Full-resolution prepared support.
        factor: Decimation factor in ``(0, 1)``.

    Returns:
        ``(small_gray, small_support)`` on the coarse grid.
    """
    support = np.asarray(prepared.pixels, dtype=np.float32)
    small_support = cv2.resize(support, None, fx=factor, fy=factor,
                               interpolation=cv2.INTER_AREA)
    small_gray = cv2.resize(gray, None, fx=factor, fy=factor,
                            interpolation=cv2.INTER_AREA)
    small = PreparedSupport(
        pixels=small_support,
        valid_mask=np.ones(small_support.shape[:2], dtype=bool),
        anchor_support=(prepared.anchor_support[0] * factor,
                        prepared.anchor_support[1] * factor),
        support_to_reference=prepared.support_to_reference,
        reference_space=prepared.reference_space,
        support_space=prepared.support_space,
        mask=None,
        origin=prepared.origin,
    )
    return small_gray, small


def _rescan_neighbourhood(gray: Array, prepared: PreparedSupport,
                          options: SearchOptions, anchor: TemplateMatch,
                          factor: float, angles: List[float], scales: List[float],
                          image_space: Space) -> List[TemplateMatch]:
    """Rescan one coarse neighbourhood at full resolution.

    The window is centred on the coarse match mapped back to full resolution and
    is sized from the support's own extent plus ``fine_search_radius_px``, so it
    never depends on the whole frame.

    Args:
        gray: Full-resolution ``float32`` inspection image.
        prepared: Full-resolution prepared support.
        options: Search policy (threshold, subpixel, overlap).
        anchor: The coarse neighbourhood to refine.
        factor: Decimation factor the anchor was found at; ``1.0`` without one.
        angles: Fine angles to evaluate in this neighbourhood.
        scales: Fine scales to evaluate in this neighbourhood.
        image_space: Space of the inspection image.

    Returns:
        Matches found inside the window, in full-resolution coordinates.
    """
    support_shape = np.asarray(prepared.pixels).shape[:2]
    height, width = gray.shape[:2]
    centre_x, centre_y = anchor.center[0] / factor, anchor.center[1] / factor
    radius = max(options.fine_search_radius_px, max(support_shape) * 0.75)
    left = int(max(0, centre_x - radius))
    top = int(max(0, centre_y - radius))
    right = int(min(width, centre_x + radius))
    bottom = int(min(height, centre_y + radius))
    window = gray[top:bottom, left:right]
    if window.size == 0:
        return []
    local = _scan_once(window, prepared, SearchOptions(
        score_threshold=options.score_threshold, max_matches=0,
        max_overlap=options.max_overlap, subpixel=options.subpixel,
        coarse_fine=False, angle_list=tuple(angles), scale_list=tuple(scales),
    ), image_space)
    if not local.valid or local.value is None:
        return []
    shifted: List[TemplateMatch] = []
    for found in local.value.matches:
        matrix = np.array(found.support_to_image.matrix, dtype=np.float64)
        matrix[0, 2] += left
        matrix[1, 2] += top
        shifted.append(TemplateMatch(
            center=(found.center[0] + left, found.center[1] + top),
            anchor=(found.anchor[0] + left, found.anchor[1] + top),
            angle_rad=found.angle_rad, scale=found.scale, score=found.score,
            support_to_image=make_mapping(matrix, source=prepared.support_space,
                                          destination=image_space, kind="similarity"),
            peak_xy=found.peak_xy, size_wh=found.size_wh,
        ))
    return shifted


def _pyramid_factor(prepared: PreparedSupport, options: SearchOptions) -> float:
    """Return the decimation factor implied by the pyramid options.

    ``pyramid_max_template_size`` is the only side-length cap: a decimation
    that would leave the support's longest side above it is not used. (A
    legacy ``max_coarse_template_side`` field claimed the same role but was
    never read; it has been removed rather than silently ignored.)
    """
    height, width = np.asarray(prepared.pixels).shape[:2]
    factor = min(2.0 ** -int(options.pyramid_levels),
                 options.pyramid_max_template_size / max(width, height))
    return min(1.0, max(factor, min(1.0, 6.0 / max(1, min(width, height)))))


def search_support(
    image: Array, prepared: PreparedSupport, options: SearchOptions, *,
    image_space: Space,
) -> Observation[MatchSet]:
    """Locate support instances in an image over the requested angle/scale grid.

    Args:
        image: ``uint8`` inspection image.
        prepared: Immutable prepared support.
        options: Search policy and variant grid.
        image_space: Space of the inspection image.

    Returns:
        Selected matches, or ``no_matches`` when nothing exceeds the threshold.
    """
    if not isinstance(prepared, PreparedSupport):
        raise contract_error("contract.invalid_parameter", "prepared support required",
                             "prepared")
    if image.dtype not in (np.uint8, np.float32):
        raise contract_error("contract.invalid_parameter",
                             "image must be uint8 or float32", "image")
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.array(gray, dtype=np.float32, copy=True)
    factor = _pyramid_factor(prepared, options)
    return _two_stage(gray, prepared, options, image_space, factor)


def _scan_once(gray: Array, prepared: PreparedSupport, options: SearchOptions,
               image_space: Space) -> Observation[MatchSet]:
    """Single-resolution search over the requested pose grid."""
    template = prepared.pixels
    support_height, support_width = template.shape[:2]
    angles = options.angles()
    scales = options.scales()
    requested = len(angles) * len(scales)
    searched = 0
    unavailable = 0
    candidates: List[TemplateMatch] = []

    # Every variant may be larger than the image; pad instead of dropping it.
    largest = int(math.ceil(max(support_width, support_height)
                            * max(scales) * math.sqrt(2.0))) + 2
    pad = max(0, largest - min(gray.shape[:2]))
    padded = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0.0)

    for angle in angles:
        for scale in scales:
            rendered = render_variant(
                np.asarray(template, dtype=np.float32), angle, scale,
                None if prepared.mask is None else prepared.mask,
                inplace=options.inplace_variants, origin=prepared.origin)
            variant = rendered.pixels
            variant_mask = rendered.mask
            variant_height, variant_width = variant.shape[:2]
            if variant_height > padded.shape[0] or variant_width > padded.shape[1]:
                unavailable += 1
                continue
            if variant_mask is not None and float(variant_mask.sum()) > len(variant_mask.flat) * 0.999:
                scores = cv2.matchTemplate(padded, variant, cv2.TM_CCOEFF_NORMED)
            elif variant_mask is not None:
                scores = cv2.matchTemplate(padded, variant, cv2.TM_CCORR_NORMED,
                                           mask=variant_mask)
            else:
                scores = cv2.matchTemplate(padded, variant, cv2.TM_CCOEFF_NORMED)
            scores = np.nan_to_num(scores, nan=-1.0, posinf=-1.0, neginf=-1.0)
            searched += 1
            if options.max_matches == 1:
                _, best, _, best_loc = cv2.minMaxLoc(scores)
                found = [(best_loc[0], best_loc[1], float(best))] if best >= options.score_threshold else []
            else:
                found = peaks(scores, options.score_threshold,
                              limit=8 * max(options.max_matches, 8))
            for x, y, score in found:
                peak_x, peak_y = (refine_peak_xy(scores, x, y) if options.subpixel
                                  else (float(x), float(y)))
                center_x = peak_x + rendered.center[0] - pad
                center_y = peak_y + rendered.center[1] - pad
                if not (0.0 <= center_x <= gray.shape[1] - 1
                        and 0.0 <= center_y <= gray.shape[0] - 1):
                    continue
                # Reuse the render matrix, shifted to this peak: the reported
                # pose is then exact by construction instead of re-derived.
                pose = rendered.matrix.copy()
                pose[0, 2] += peak_x - pad
                pose[1, 2] += peak_y - pad
                anchor = pose @ np.array([prepared.anchor_support[0],
                                          prepared.anchor_support[1], 1.0],
                                         dtype=np.float64)
                mapping = make_mapping(pose,
                                       source=prepared.support_space,
                                       destination=image_space, kind="similarity")
                candidates.append(TemplateMatch(
                    center=(float(center_x), float(center_y)),
                    anchor=(float(anchor[0]), float(anchor[1])),
                    angle_rad=float(angle), scale=float(scale), score=float(score),
                    support_to_image=mapping, peak_xy=(int(x), int(y)),
                    size_wh=(float(variant_width), float(variant_height)),
                ))
    diagnostics = {
        "requested_variant_count": requested,
        "searched_variant_count": searched,
        "ncc_evaluations": searched,
        "fine_neighborhoods": 0,
        "coarse_poses": requested,
        "unavailable_variant_count": unavailable,
        "pruned_variant_count": 0,
        "search_complete": searched + unavailable == requested,
    }
    if not candidates:
        return invalid(image_space, "no_matches", diagnostics)
    selected = suppress(candidates, options.max_overlap)
    if options.max_matches:
        selected = selected[: options.max_matches]
    return observed(image_space, MatchSet(matches=tuple(selected)), diagnostics)