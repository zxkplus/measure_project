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


#: The polarity policies Halcon's ``Metric`` offers.
METRIC_VALUES = ("use_polarity", "ignore_global_polarity")


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
    decimate_exponent: int = 0
    coarse_template_side: int = 128
    fine_search_radius_px: float = 24.0
    fine_candidates: Optional[int] = None
    angle_list: Optional[Tuple[float, ...]] = None
    scale_list: Optional[Tuple[float, ...]] = None
    inplace_variants: bool = False
    # --- Halcon-aligned controls -------------------------------------------
    #: ``"ncc"`` correlates preprocessed intensity; ``"shape"`` correlates edge
    #: gradient directions. Both share the pyramid cascade and the pruning.
    algorithm: str = "ncc"
    #: Halcon's Greediness: 0 rejects only candidates that provably cannot reach
    #: the threshold, 1 rejects as soon as the remaining levels cannot help.
    #: Our correlation cannot be inspected mid-sum, so the bound is applied
    #: between pyramid levels rather than between model points.
    greediness: float = 0.0
    #: When true the level count is derived from the support size and the two
    #: side limits; when false ``decimate_exponent`` is the level count and the
    #: historical single-jump behaviour is kept.
    auto_levels: bool = True
    #: Coarsest level at which a match must still be verified (Halcon's second
    #: ``NumLevels`` value). Levels above it only produce candidates.
    min_verify_level: int = 0
    #: Shape-only: edge points weaker than this are dropped from the model. A
    #: zero value selects a percentile of the support's gradient magnitude.
    min_contrast: float = 0.0
    #: Halcon's Metric: ``use_polarity`` requires the part to keep its contrast,
    #: ``ignore_global_polarity`` also accepts a globally inverted one. The
    #: latter is a score magnitude, so it accepts the photographic negative too
    #: and therefore trades some discrimination for that tolerance.
    metric: str = "use_polarity"

    @property
    def polarity_invariant(self) -> bool:
        """Return whether a globally inverted part should still match."""
        return self.metric == "ignore_global_polarity"

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
        if self.algorithm not in ("ncc", "shape"):
            raise contract_error("contract.invalid_parameter",
                                 f"unknown algorithm {self.algorithm!r}", "search.algorithm")
        if not (0.0 <= self.greediness <= 1.0):
            raise contract_error("contract.invalid_parameter",
                                 "greediness must be in [0, 1]", "search.greediness")
        if self.min_verify_level < 0:
            raise contract_error("contract.invalid_parameter",
                                 "min_verify_level must be >= 0",
                                 "search.min_verify_level")
        if self.min_contrast < 0.0:
            raise contract_error("contract.invalid_parameter",
                                 "min_contrast must be >= 0", "search.min_contrast")
        if self.metric not in METRIC_VALUES:
            raise contract_error("contract.invalid_parameter",
                                 f"unknown metric {self.metric!r}", "search.metric")

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


def correlate(search: Array, template: Array, mask: Optional[Array] = None, *,
              metric: str = "use_polarity") -> Array:
    """Return the normalised cross-correlation score map of one template.

    This is the single place the correlation score is computed; the scanning
    loops call it rather than reaching for ``cv2.matchTemplate`` themselves.
    Masked and unmasked variants use the same coefficient, so a mask changes
    which pixels count but not the scale the score is reported on.

    Args:
        search: Search-area pixels.
        template: Rendered variant pixels.
        mask: Optional variant footprint mask; ``None`` uses ordinary NCC.
        metric: ``use_polarity`` keeps the sign, ``ignore_global_polarity``
            takes the magnitude so a globally inverted part still scores.

    Returns:
        A score map with every non-finite value replaced by ``-1``, so masked
        correlation over a constant background cannot produce a false peak.
    """
    area = np.ascontiguousarray(search, dtype=np.float32)
    variant = np.ascontiguousarray(template, dtype=np.float32)
    if mask is None:
        scores = cv2.matchTemplate(area, variant, cv2.TM_CCOEFF_NORMED)
    else:
        # The mean-subtracted coefficient with a mask is the correct NCC: the
        # mean and the deviation are taken over the valid pixels only. The
        # masked path used to fall back to the plain correlation coefficient,
        # which is not normalised the same way, so a masked variant scored on a
        # different scale than an unmasked one.
        scores = cv2.matchTemplate(area, variant, cv2.TM_CCOEFF_NORMED,
                                   mask=np.ascontiguousarray(mask, dtype=np.float32))
    if metric == "ignore_global_polarity":
        # A globally inverted part correlates with the negated score, so the
        # magnitude accepts both. It must happen before the invalid sentinel
        # below, otherwise the -1 that marks a non-finite score becomes a 1 and
        # turns an unusable region into a perfect match.
        scores = np.abs(scores)
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




#: Two candidates whose scores are this close, while their centres are far
#: apart, mean the frame does not determine a unique answer. A periodic pattern
#: scores every repeat of the target almost equally well, and the reported
#: position is then one of several valid places rather than *the* place.
AMBIGUITY_SCORE_MARGIN = 0.05

#: How far apart two candidates count as separate places, as a fraction of the
#: support's longest side. Closer than this they are one target found twice.
AMBIGUITY_SEPARATION_RATIO = 0.5


def ambiguity_report(candidates: List[TemplateMatch], support_longest: float,
                     margin: float = AMBIGUITY_SCORE_MARGIN,
                     separation_ratio: float = AMBIGUITY_SEPARATION_RATIO) -> dict:
    """Describe whether more than one distinct place matches nearly as well.

    A repeated pattern makes every instance score alike, so the strongest match
    is one of several equally valid answers and its position carries no
    information about which instance the caller meant. Reporting that is the
    difference between "found it" and "found *a* it".

    Args:
        candidates: Deduplicated candidates, before the match count is applied.
        support_longest: Longest support side, in image pixels.
        margin: Score difference still counted as "nearly as well".
        separation_ratio: Minimum centre separation, as a fraction of
            ``support_longest``.

    Returns:
        Diagnostic fields; ``ambiguous`` is ``True`` only when a rival exists.
    """
    report: dict = {"ambiguous": False, "distinct_candidates": len(candidates)}
    if len(candidates) < 2:
        return report
    ordered = sorted(candidates, key=lambda match: match.score, reverse=True)
    best = ordered[0]
    separation = max(1.0, float(support_longest) * separation_ratio)
    rivals = []
    for other in ordered[1:]:
        if best.score - other.score > margin:
            continue
        distance = math.hypot(other.center[0] - best.center[0],
                              other.center[1] - best.center[1])
        if distance >= separation:
            rivals.append((other, distance))
    if not rivals:
        return report
    rival, distance = rivals[0]
    report.update({
        "ambiguous": True,
        "runner_up_score": float(rival.score),
        "runner_up_distance_px": float(distance),
        "rival_count": len(rivals),
    })
    return report


def _support_longest(prepared: PreparedSupport) -> float:
    """Return the longest support side in pixels."""
    return float(max(np.asarray(prepared.pixels).shape[:2]))


def _coarse_grid(values: List[float], step: float) -> List[float]:
    """Reduce a fine grid to the coarse values spaced at least ``step`` apart.

    The thinning is phased on the value closest to zero rather than on the first
    entry. Zero rotation is the cheapest pose to fit -- a support as wide as the
    frame fits only there -- and a grid whose phase happened to skip it left such
    a model with no usable pose at all, so the whole search failed for a reason
    that had nothing to do with the image.

    Args:
        values: Fine grid, ascending.
        step: Minimum spacing of the coarse values.

    Returns:
        The coarse subset, still ascending and still containing both ends.
    """
    if not values:
        return []
    anchor = min(range(len(values)), key=lambda index: abs(values[index]))
    coarse = [values[anchor]]
    for index in range(anchor - 1, -1, -1):
        if coarse[0] - values[index] >= step - 1e-12:
            coarse.insert(0, values[index])
    for index in range(anchor + 1, len(values)):
        if values[index] - coarse[-1] >= step - 1e-12:
            coarse.append(values[index])
    if coarse[0] != values[0]:
        coarse.insert(0, values[0])
    if coarse[-1] != values[-1]:
        coarse.append(values[-1])
    return coarse


def _two_stage(gray: Array, prepared: PreparedSupport, options: SearchOptions,
               image_space: Space, factor: float) -> Observation[MatchSet]:
    """Coarse stage locates neighbourhoods, then each is rescanned at full resolution.

    Two independent knobs decide what the coarse stage does, and they compose
    instead of excluding each other:

    * ``factor`` (from ``decimate_exponent`` and the template side cap) chooses the
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
        scale_list=tuple(coarse_scales), metric=options.metric,
    ), image_space)
    diagnostics = {
        "requested_variant_count": requested,
        "coarse_poses": len(coarse_angles) * len(coarse_scales),
        "algorithm": options.algorithm,
        "pyramid_levels": _resolve_levels(prepared, options),
        "search_strategy": "two_stage",
    }
    if decimated:
        diagnostics["pyramid_factor"] = float(factor)
    if not coarse.valid or coarse.value is None:
        diagnostics.update({"searched_variant_count": 0, "pruned_variant_count": 0,
                            "unavailable_variant_count": coarse.diagnostics.get(
                                "unavailable_variant_count", 0),
                            "search_complete": True})
        return invalid(image_space, coarse.reason or "no_matches", diagnostics)

    if not decimated and not options.coarse_fine:
        # One pass already covered the whole requested grid at full resolution,
        # so the coarse matches are final: rescanning them would repeat every
        # correlation for a subpixel refinement that pass did not ask for.
        diagnostics.update({
            "searched_variant_count": len(fine_angles) * len(fine_scales),
            "ncc_evaluations": len(fine_angles) * len(fine_scales),
            "pruned_variant_count": 0,
            "fine_neighborhoods": 0,
            "search_strategy": "single",
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
        distinct = suppress(accepted, options.max_overlap)
        diagnostics.update(ambiguity_report(distinct, _support_longest(prepared)))
        selected = distinct
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

    # Halcon's second NumLevels value: track the candidates down to this level
    # instead of all the way to full resolution. Each level skipped multiplies
    # the refinement's error by two, so this is an accuracy-for-time trade and
    # the default (0) keeps the historical full-resolution refinement.
    verify_level = min(max(0, int(options.min_verify_level)),
                       _levels_by_min_side(prepared))
    verify_factor = 2.0 ** -verify_level if verify_level else 1.0
    refine_gray, refine_support = gray, prepared
    if verify_level:
        refine_gray, refine_support = _decimate_pair(gray, prepared, verify_factor)
    diagnostics["verify_level"] = verify_level

    refined: List[TemplateMatch] = []
    searched = 0
    refine_stats: dict = {}
    for anchor in neighbourhoods:
        local_angles = ([a for a in fine_angles
                         if abs(a - anchor.angle_rad) <= options.coarse_angle_step_rad + 1e-12]
                        if options.coarse_fine else [anchor.angle_rad])
        local_scales = ([s for s in fine_scales
                         if abs(s - anchor.scale) <= options.coarse_scale_step + 1e-12]
                        if options.coarse_fine else [anchor.scale])
        if not local_angles or not local_scales:
            continue
        if verify_level:
            # The candidate lives in the coarse frame; move it into the verify
            # frame, refine there, then move the result back to full resolution.
            anchor_here = _rescale_match(anchor, verify_factor / factor,
                                         prepared, image_space)
            found = _rescan_neighbourhood(refine_gray, refine_support, options,
                                          anchor_here, 1.0,
                                          local_angles, local_scales, image_space,
                                          refine_stats)
            found = [_rescale_match(match, 1.0 / verify_factor, prepared, image_space)
                     for match in found]
        else:
            found = _rescan_neighbourhood(gray, prepared, options, anchor, factor,
                                          local_angles, local_scales, image_space,
                                          refine_stats)
        searched += len(local_angles) * len(local_scales)
        refined.extend(found)

    distinct = suppress(refined, options.max_overlap)
    diagnostics.update(ambiguity_report(distinct, _support_longest(prepared)))
    selected = distinct
    if options.max_matches:
        selected = selected[: options.max_matches]
    diagnostics.update({
        "searched_variant_count": searched,
        "ncc_evaluations": searched,
        "pruned_variant_count": len(coarse.value.matches),
        "fine_neighborhoods": len(neighbourhoods),
        "unavailable_variant_count": int(
            coarse.diagnostics.get("unavailable_variant_count", 0) or 0)
            + int(refine_stats.get("unavailable", 0)),
        "search_complete": True,
    })
    if not selected:
        return invalid(image_space, "no_matches", diagnostics)
    return observed(image_space, MatchSet(matches=tuple(selected)), diagnostics)


def _shape_pose_matrix(angle_rad: float, scale: float,
                       anchor_support: Tuple[float, float],
                       anchor_image: Tuple[float, float]) -> np.ndarray:
    """Return the support-to-image similarity matrix for one shape pose.

    The linear part uses the same convention as ``_variant_geometry``, so a
    shape pose and a correlated pose mean the same rotation. The translation is
    solved from the anchor rather than re-derived from a rendered box, because
    the shape model's box tightly bounds its points and therefore is not the
    variant box.

    Args:
        angle_rad: Clockwise pose angle in radians.
        scale: Dimensionless pose scale.
        anchor_support: Anchor position in support coordinates, ``(x, y)``.
        anchor_image: Where that anchor lands in the image, ``(x, y)``.

    Returns:
        A 3x3 homogeneous similarity matrix.
    """
    alpha = scale * math.cos(angle_rad)
    beta = scale * math.sin(angle_rad)
    anchor_x, anchor_y = anchor_support
    image_x, image_y = anchor_image
    return np.array([
        [alpha, -beta, image_x - (alpha * anchor_x - beta * anchor_y)],
        [beta, alpha, image_y - (beta * anchor_x + alpha * anchor_y)],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def _shape_search(gray: Array, prepared: PreparedSupport, options: SearchOptions,
                  image_space: Space) -> Observation[MatchSet]:
    """Locate a support by edge direction instead of intensity correlation.

    A coarse pass scores a decimated copy over the whole frame, the way Halcon's
    top pyramid level does, and each surviving peak is then refined at full
    resolution against the complete point set. The scoring cost is
    ``points * positions`` rather than ``image_area * support_area``, which is
    what makes a support that nearly fills the frame affordable.

    Args:
        gray: Full-resolution ``float32`` inspection image.
        prepared: Immutable prepared support.
        options: Search policy; only unit scale is supported for shape.
        image_space: Space of the inspection image.

    Returns:
        Selected matches, or ``no_matches`` when nothing exceeds the threshold.
    """
    from measure.shape_model import (build_shape_model, gradient_planes,
                                     rotate_model, score_at, score_map)

    fine_angles = options.angles()
    fine_scales = options.scales()
    diagnostics: dict = {
        "algorithm": "shape",
        "search_strategy": "shape_two_level",
        "requested_variant_count": len(fine_angles) * len(fine_scales),
        "ncc_evaluations": 0,
    }
    if len(fine_scales) > 1:
        # Multi-scale shape search is not implemented; correlation already
        # covers it, and silently ignoring the scale range would be a wrong
        # answer rather than a slower one.
        from dataclasses import replace

        fallback = _two_stage(gray, prepared, options, image_space,
                              _pyramid_factor(prepared, options))
        merged = dict(fallback.diagnostics)
        merged["shape_scale_fallback"] = "multi_scale_uses_correlation"
        merged["algorithm"] = "shape"
        return replace(fallback, diagnostics=merged)

    support_height, support_width = np.asarray(prepared.pixels).shape[:2]
    anchor_x, anchor_y = (float(prepared.anchor_support[0]),
                          float(prepared.anchor_support[1]))
    levels = max(1, _resolve_levels(prepared, options))
    factor = 2.0 ** -levels
    # Edge directions come from gradients taken after this step, so the
    # coarse pair must not be pre-blurred: smoothing first measurably
    # degrades the coarse peak that seeds every refinement.
    coarse_gray, coarse_support = _decimate_pair(gray, prepared, factor,
                                                 smooth=False)
    coarse_anchor = (anchor_x * factor, anchor_y * factor)
    coarse_obs = build_shape_model(
        np.asarray(coarse_support.pixels), prepared.support_space,
        anchor=(coarse_anchor[1], coarse_anchor[0]),
        min_contrast=options.min_contrast, level=levels)
    if not coarse_obs.valid:
        diagnostics["shape_points"] = 0
        return invalid(image_space, coarse_obs.reason or "template_no_texture",
                       diagnostics)
    coarse_model = coarse_obs.value
    diagnostics["shape_points"] = coarse_model.count
    diagnostics["coarse_level"] = levels

    coarse_gx, coarse_gy = gradient_planes(coarse_gray)
    coarse_limit = min(options.score_threshold, options.coarse_score_threshold)
    limit_per_angle = 8 * max(options.max_matches, 8)
    coarse_angles = _coarse_grid(fine_angles, options.coarse_angle_step_rad) or list(fine_angles)
    found: List[Tuple[float, float, float, float]] = []
    searched_angles = 0
    unavailable_angles = 0
    for angle in coarse_angles:
        posed = rotate_model(coarse_model, angle)
        height, width = posed.size_hw
        if height > coarse_gray.shape[0] or width > coarse_gray.shape[1]:
            # The rotated box leaves the frame: the support must fit whole, so
            # this angle simply cannot be scored.
            unavailable_angles += 1
            continue
        scores = score_map(coarse_gx, coarse_gy, posed,
                           polarity_invariant=options.polarity_invariant)
        searched_angles += 1
        # Record the anchor, not the box origin: the box is tight around the
        # points, so its origin depends on the angle's bounding box while the
        # anchor's image position does not.
        for x, y, score in peaks(scores, coarse_limit, limit=limit_per_angle):
            found.append((float(score), float(angle),
                          float(y) + posed.anchor_y, float(x) + posed.anchor_x))
    searched = searched_angles
    if not found:
        diagnostics.update({"searched_variant_count": searched, "coarse_peaks": 0,
                            "unavailable_variant_count": unavailable_angles,
                            "search_complete": True})
        reason = ("support_exceeds_search_area"
                  if searched_angles == 0 and unavailable_angles
                  else "no_matches")
        return invalid(image_space, reason, diagnostics)

    found.sort(reverse=True)
    budget = (limit_per_angle if options.max_matches == 0
              else max(1, options.max_matches) * PRUNE_FACTOR)
    diagnostics["coarse_peaks"] = len(found)

    full_obs = build_shape_model(
        np.asarray(prepared.pixels), prepared.support_space,
        mask=None if prepared.mask is None else prepared.mask,
        anchor=(anchor_y, anchor_x),
        min_contrast=options.min_contrast, level=0)
    if not full_obs.valid:
        return invalid(image_space, full_obs.reason or "template_no_texture",
                       diagnostics)
    full_model = full_obs.value
    image_gx, image_gy = gradient_planes(gray)

    # The coarse peak is only accurate to about one coarse pixel, so the
    # refinement window must cover that at full resolution, doubled for
    # slack, and never so wide that the per-position cost dominates.
    radius = max(4, min(24, int(round(1.0 / factor)) * 2 + 2))
    refined: List[TemplateMatch] = []
    placed_any = False
    for _score, coarse_angle, coarse_anchor_row, coarse_anchor_col in found[:budget]:
        anchor_row = coarse_anchor_row / factor
        anchor_col = coarse_anchor_col / factor
        posed = rotate_model(full_model, coarse_angle)
        base_row = int(round(anchor_row - posed.anchor_y))
        base_col = int(round(anchor_col - posed.anchor_x))
        rows: List[int] = []
        cols: List[int] = []
        for delta_row in range(-radius, radius + 1):
            for delta_col in range(-radius, radius + 1):
                row = base_row + delta_row
                col = base_col + delta_col
                if (row >= 0 and col >= 0
                        and row + posed.size_hw[0] <= gray.shape[0]
                        and col + posed.size_hw[1] <= gray.shape[1]):
                    rows.append(row)
                    cols.append(col)
        if not rows:
            continue
        placed_any = True
        scores = score_at(image_gx, image_gy, posed, rows, cols,
                          polarity_invariant=options.polarity_invariant)
        best = int(np.argmax(scores))
        best_anchor = (rows[best] + posed.anchor_y, cols[best] + posed.anchor_x)
        searched += 1

        # With the position settled, the angle neighbourhood is cheap: one
        # evaluation per angle instead of one per angle-position pair.
        local_angles = ([a for a in fine_angles
                         if abs(a - coarse_angle) <= options.coarse_angle_step_rad + 1e-12]
                        or [coarse_angle])
        best_angle = coarse_angle
        best_score = float(scores[best])
        best_row = rows[best]
        best_col = cols[best]
        for angle in local_angles:
            if angle == coarse_angle:
                continue
            candidate = rotate_model(full_model, angle)
            row = int(round(best_anchor[0] - candidate.anchor_y))
            col = int(round(best_anchor[1] - candidate.anchor_x))
            if not (row >= 0 and col >= 0
                    and row + candidate.size_hw[0] <= gray.shape[0]
                    and col + candidate.size_hw[1] <= gray.shape[1]):
                continue
            extra = score_at(image_gx, image_gy, candidate, [row], [col],
                             polarity_invariant=options.polarity_invariant)
            searched += 1
            if float(extra[0]) > best_score:
                best_score = float(extra[0])
                best_angle = angle
                best_row = row
                best_col = col
        if best_score < coarse_limit:
            continue
        peak_x, peak_y = float(best_col), float(best_row)
        if options.subpixel:
            peak_x, peak_y = _shape_subpixel(image_gx, image_gy, full_model,
                                             best_angle, best_row, best_col)
        anchor_image = (peak_y + rotate_model(full_model, best_angle).anchor_y,
                        peak_x + rotate_model(full_model, best_angle).anchor_x)
        matrix = _shape_pose_matrix(best_angle, 1.0, (anchor_x, anchor_y),
                                    (anchor_image[1], anchor_image[0]))
        if prepared.origin == "pixel_center":
            centre_x, centre_y = (support_width - 1) / 2.0, (support_height - 1) / 2.0
        else:
            centre_x, centre_y = support_width / 2.0, support_height / 2.0
        centre = matrix @ np.array([centre_x, centre_y, 1.0], dtype=np.float64)
        refined.append(TemplateMatch(
            center=(float(centre[0]), float(centre[1])),
            anchor=(float(anchor_image[1]), float(anchor_image[0])),
            angle_rad=float(best_angle), scale=1.0, score=float(best_score),
            support_to_image=make_mapping(matrix, source=prepared.support_space,
                                          destination=image_space, kind="similarity"),
            peak_xy=(int(round(peak_x)), int(round(peak_y))),
            size_wh=(float(support_width), float(support_height)),
        ))

    if not placed_any:
        # The rotated box fits nowhere in the frame: a shape score cannot be
        # formed at all. Correlation pads the frame up to the support instead,
        # so it still answers; say so rather than reporting "no match".
        from dataclasses import replace

        fallback = _two_stage(gray, prepared, options, image_space,
                              _pyramid_factor(prepared, options))
        merged = dict(fallback.diagnostics)
        merged["shape_support_too_large"] = "support_does_not_fit_uses_correlation"
        merged["algorithm"] = "shape"
        return replace(fallback, diagnostics=merged)

    # The refinement gated on the relaxed coarse threshold so a candidate could
    # be re-examined; the caller's threshold decides what is reported.
    accepted = [match for match in refined
                if match.score >= options.score_threshold - 1e-12]
    distinct = suppress(accepted, options.max_overlap)
    diagnostics.update(ambiguity_report(distinct, _support_longest(prepared)))
    selected = distinct
    if options.max_matches:
        selected = selected[: options.max_matches]
    diagnostics.update({
        "searched_variant_count": searched,
        "fine_positions": len(refined),
        "unavailable_variant_count": int(unavailable_angles),
        "search_complete": True,
    })
    if not selected:
        return invalid(image_space, "no_matches", diagnostics)
    return observed(image_space, MatchSet(matches=tuple(selected)), diagnostics)


def _shape_subpixel(gx: Array, gy: Array, model: Any, angle: float,
                    row: int, col: int) -> Tuple[float, float]:
    """Refine a shape peak to sub-pixel accuracy with the shared quadratic.

    Args:
        gx: Gradient ``x`` plane of the full-resolution image.
        gy: Gradient ``y`` plane of the full-resolution image.
        model: Full-resolution model.
        angle: Pose angle of the winning candidate.
        row: Integer peak row (model box top).
        col: Integer peak column (model box left).

    Returns:
        Refined ``(x, y)`` of the model box origin.
    """
    from measure.shape_model import rotate_model, score_at

    posed = rotate_model(model, angle)
    rows = [row - 1, row - 1, row - 1, row, row, row, row + 1, row + 1, row + 1]
    cols = [col - 1, col, col + 1, col - 1, col, col + 1, col - 1, col, col + 1]
    if rows[0] < 0 or cols[0] < 0:
        return float(col), float(row)
    if (rows[-1] + posed.size_hw[0] > gx.shape[0]
            or cols[-1] + posed.size_hw[1] > gx.shape[1]):
        return float(col), float(row)
    values = score_at(gx, gy, posed, rows, cols).reshape(3, 3)
    offset_x = subpixel_offset(float(values[1, 0]), float(values[1, 1]),
                               float(values[1, 2]))
    offset_y = subpixel_offset(float(values[0, 1]), float(values[1, 1]),
                               float(values[2, 1]))
    return float(col) + offset_x, float(row) + offset_y


def _decimate_pair(gray: Array, prepared: PreparedSupport, factor: float, *,
                   smooth: bool = True) -> Tuple[Array, PreparedSupport]:
    """Low-pass and decimate the search image and the support by ``factor``.

    Args:
        gray: Full-resolution ``float32`` inspection image.
        prepared: Full-resolution prepared support.
        factor: Decimation factor in ``(0, 1)``.
        smooth: Low-pass before sampling. Correlation needs it (INTER_AREA alone
            aliases a high-frequency support badly enough to lose the peak), but
            the edge-direction path derives gradients afterwards and loses coarse
            direction precision when the input is blurred first; its caller passes
            ``False`` and keeps the plain area average, which is its own 2x2 box.

    Returns:
        ``(small_gray, small_support)`` on the coarse grid.
    """
    support = np.asarray(prepared.pixels, dtype=np.float32)
    # Low-pass before sampling. Plain INTER_AREA aliases the phase of a
    # high-frequency support: the v2 acceptance corpus measured the correlation
    # peak fall from 0.65 to 0.27 without this pass, and coarse detection then
    # failed. The blur is applied here rather than through ``decimate_pixels``
    # because that helper also shifts the sampling grid by half a pixel, which is
    # correct for its pixel-centre callers but changes the phase this search has
    # always used -- the edge-direction path samples gradients and is far more
    # sensitive to that than correlation is.
    if smooth:
        sigma = 0.5 / float(factor)
        support = cv2.GaussianBlur(support, (0, 0), sigma)
        gray = cv2.GaussianBlur(gray, (0, 0), sigma)
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


def _fine_radius_px(prepared: PreparedSupport, options: SearchOptions) -> float:
    """Return the half-side of a full-resolution refinement window.

    The window must cover the support plus whatever the coarse stage could have
    been off by, but never more than the search area itself: past that point the
    "refinement" is a second full-frame correlation per candidate, which costs
    more than the coarse pass saved.

    Args:
        prepared: Immutable prepared support (full resolution).
        options: Search policy carrying the radius floor.

    Returns:
        Half-side in pixels, at least one.
    """
    support_shape = np.asarray(prepared.pixels).shape[:2]
    return float(max(options.fine_search_radius_px, max(support_shape) * 0.75))


def _rescale_match(match: TemplateMatch, scale: float,
                   prepared: PreparedSupport, image_space: Space) -> TemplateMatch:
    """Re-express a match in a frame scaled by ``scale``.

    Used to hand a coarse candidate to a finer or coarser pyramid level: scaling
    both the linear part and the translation moves the whole pose, while the
    rotation and the scale of the support are unchanged because a frame change
    is a similarity.

    Args:
        match: Match in the source frame.
        scale: Target-frame coordinates per source-frame coordinate.
        prepared: Support the match was produced from.
        image_space: Space of the image the source frame belongs to.

    Returns:
        The same match in the target frame.
    """
    matrix = np.array(match.support_to_image.matrix, dtype=np.float64)
    matrix[0:2, :] *= scale
    peak = None
    if match.peak_xy is not None:
        peak = (int(round(match.peak_xy[0] * scale)),
                int(round(match.peak_xy[1] * scale)))
    return TemplateMatch(
        center=(match.center[0] * scale, match.center[1] * scale),
        anchor=(match.anchor[0] * scale, match.anchor[1] * scale),
        angle_rad=match.angle_rad, scale=match.scale, score=match.score,
        support_to_image=make_mapping(matrix, source=prepared.support_space,
                                      destination=image_space, kind="similarity"),
        peak_xy=peak,
        size_wh=(match.size_wh[0] * scale, match.size_wh[1] * scale),
    )


def _rescan_neighbourhood(gray: Array, prepared: PreparedSupport,
                          options: SearchOptions, anchor: TemplateMatch,
                          factor: float, angles: List[float], scales: List[float],
                          image_space: Space,
                          stats: Optional[dict] = None) -> List[TemplateMatch]:
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
    # The window is clipped to the search area below, so it never needs an extra
    # cap. A cap is in fact harmful: anything below the support's own extent
    # leaves the support unable to fit the window, and a support larger than the
    # frame is no longer padded, so the refinement would find nothing at all.
    radius = _fine_radius_px(prepared, options)
    # The window must be able to hold the support whole, because the search no
    # longer pads. When the candidate sits near a border, clipping would shrink
    # the window below the support; slide it instead of shrinking it. The
    # support fits the frame (checked before scanning), so a window of at least
    # the support's size always exists.
    span_x = min(width, max(int(math.ceil(2.0 * radius)), int(support_shape[1])))
    span_y = min(height, max(int(math.ceil(2.0 * radius)), int(support_shape[0])))
    left = int(min(max(0, round(centre_x - span_x / 2.0)), width - span_x))
    top = int(min(max(0, round(centre_y - span_y / 2.0)), height - span_y))
    right = min(width, left + span_x)
    bottom = min(height, top + span_y)
    window = gray[top:bottom, left:right]
    if window.size == 0:
        return []
    local = _scan_once(window, prepared, SearchOptions(
        score_threshold=options.score_threshold, max_matches=0,
        max_overlap=options.max_overlap, subpixel=options.subpixel,
        coarse_fine=False, angle_list=tuple(angles), scale_list=tuple(scales),
        metric=options.metric,
    ), image_space)
    if stats is not None:
        stats["unavailable"] = stats.get("unavailable", 0) + int(
            local.diagnostics.get("unavailable_variant_count", 0) or 0)
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

    ``coarse_template_side`` is the only side-length cap: a decimation
    that would leave the support's longest side above it is not used. (A
    legacy ``max_coarse_template_side`` field claimed the same role but was
    never read; it has been removed rather than silently ignored.)
    """
    height, width = np.asarray(prepared.pixels).shape[:2]
    factor = min(2.0 ** -int(options.decimate_exponent),
                 options.coarse_template_side / max(width, height))
    return min(1.0, max(factor, min(1.0, 6.0 / max(1, min(width, height)))))


#: No pyramid level may shrink the support below this side. Measured failures
#: start around a 19 px long side, so the historical 6 px floor (which only
#: guarded against a degenerate image) is too permissive to build a cascade on.
MIN_LEVEL_SIDE = 12.0

#: Deepest pyramid the cascade will build. Halcon also caps the level count;
#: past about five levels a level adds cost without removing candidates.
MAX_LEVELS = 5

#: Each level keeps this multiple of the requested match count, so a target
#: that ranks slightly low on a coarse level still survives to the next one.
PRUNE_FACTOR = 4


def _levels_by_min_side(prepared: PreparedSupport,
                        min_side: float = MIN_LEVEL_SIDE) -> int:
    """Return how many halvings the support survives without falling apart.

    Args:
        prepared: Immutable prepared support (full resolution).
        min_side: Smallest acceptable support side, in pixels.

    Returns:
        The largest usable level count, at least zero.
    """
    shortest = float(min(np.asarray(prepared.pixels).shape[:2]))
    if shortest <= 0.0:
        return 0
    levels = 0
    side = shortest
    while levels < MAX_LEVELS and side / 2.0 >= min_side:
        side /= 2.0
        levels += 1
    return levels


def _resolve_levels(prepared: PreparedSupport, options: SearchOptions) -> int:
    """Return the pyramid depth the search should use.

    With ``auto_levels`` the depth is derived from the factor the pyramid
    options already imply, so a support that needs heavy decimation gets a deep
    cascade instead of one large jump. Without it, the caller's
    ``decimate_exponent`` is the depth verbatim, which reproduces the historical
    single-jump behaviour.

    Args:
        prepared: Immutable prepared support (full resolution).
        options: Search policy carrying both pyramid controls.

    Returns:
        Level count in ``[0, MAX_LEVELS]``; ``0`` means full resolution only.
    """
    if not options.auto_levels:
        return max(0, min(int(options.decimate_exponent), MAX_LEVELS))
    factor = _pyramid_factor(prepared, options)
    if factor >= 1.0 - 1e-12:
        return 0
    by_factor = int(round(math.log2(1.0 / factor)))
    return max(0, min(by_factor, _levels_by_min_side(prepared), MAX_LEVELS))


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
    # Halcon requires the model to lie completely inside the search area. Saying
    # so before scanning beats returning a "no match" the caller would read as
    # "the part is missing".
    support_height, support_width = np.asarray(prepared.pixels).shape[:2]
    if support_height > gray.shape[0] or support_width > gray.shape[1]:
        return invalid(image_space, "support_exceeds_search_area", {
            "support_hw": (int(support_height), int(support_width)),
            "search_hw": (int(gray.shape[0]), int(gray.shape[1])),
        })
    factor = _pyramid_factor(prepared, options)
    if options.algorithm == "shape":
        return _shape_search(gray, prepared, options, image_space)
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

    # The support must lie completely inside the search area, exactly as Halcon
    # requires: a variant larger than the frame is dropped rather than padded,
    # because a padded correlation scores against invented zeros and reports a
    # confidence the image never supported.
    for angle in angles:
        for scale in scales:
            rendered = render_variant(
                np.asarray(template, dtype=np.float32), angle, scale,
                None if prepared.mask is None else prepared.mask,
                inplace=options.inplace_variants, origin=prepared.origin)
            variant = rendered.pixels
            variant_mask = rendered.mask
            offset_x = offset_y = 0
            if variant_mask is not None:
                # A masked support is mostly empty: trim the variant to the
                # bounding box of its valid pixels. Correlation ignores masked
                # pixels anyway, so the score is unchanged, but the footprint
                # that has to fit the frame is the content rather than the box.
                # Without this a 45 degree masked support is dropped for a
                # bounding box far larger than anything it actually contains
                # (the historical D07 defect).
                rows, cols = np.nonzero(variant_mask)
                if rows.size:
                    top, left = int(rows.min()), int(cols.min())
                    bottom, right = int(rows.max()) + 1, int(cols.max()) + 1
                    if (top or left or bottom != variant.shape[0]
                            or right != variant.shape[1]):
                        variant = variant[top:bottom, left:right]
                        variant_mask = variant_mask[top:bottom, left:right]
                        offset_y, offset_x = top, left
            variant_height, variant_width = variant.shape[:2]
            if variant_height > gray.shape[0] or variant_width > gray.shape[1]:
                unavailable += 1
                continue
            scores = correlate(gray, variant, variant_mask, metric=options.metric)
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
                center_x = peak_x + rendered.center[0] - offset_x
                center_y = peak_y + rendered.center[1] - offset_y
                if not (0.0 <= center_x <= gray.shape[1] - 1
                        and 0.0 <= center_y <= gray.shape[0] - 1):
                    continue
                # Reuse the render matrix, shifted to this peak: the reported
                # pose is then exact by construction instead of re-derived.
                pose = rendered.matrix.copy()
                pose[0, 2] += peak_x - offset_x
                pose[1, 2] += peak_y - offset_y
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
        if unavailable and searched == 0:
            # Nothing could be evaluated at all: every pose was larger than the
            # search area. Saying "no match" would blame the image for a model
            # that can never fit.
            return invalid(image_space, "support_exceeds_search_area", diagnostics)
        return invalid(image_space, "no_matches", diagnostics)
    selected = suppress(candidates, options.max_overlap)
    if options.max_matches:
        selected = selected[: options.max_matches]
    return observed(image_space, MatchSet(matches=tuple(selected)), diagnostics)