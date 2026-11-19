"""Template preparation and search over an explicit, immutable support.

The support is a straightened crop plus an explicit valid-pixel mask and an
anchor expressed in support coordinates.  Searching never mutates the prepared
arrays: every variant is built from a private copy, and rotated supports whose
bounding box exceeds the search image are handled by padding instead of being
silently dropped (the historical D07 defect).
"""

from __future__ import annotations

import math
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

    def __post_init__(self) -> None:
        if not isinstance(self.box, RotatedBox):
            raise contract_error("contract.invalid_parameter",
                                 "box must be a RotatedBox", "support.box")
        object.__setattr__(self, "anchor_reference",
                           require_finite_xy(self.anchor_reference, "support.anchor_reference"))
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
    max_coarse_template_side: int = 400
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


def _variant(pixels: Array, angle_rad: float, scale: float,
             mask: Optional[Array] = None,
             inplace: bool = False) -> Tuple[Array, np.ndarray, Optional[Array]]:
    """Build one rotated/scaled variant and its support-to-variant matrix."""
    height, width = pixels.shape[:2]
    center_x, center_y = width / 2.0, height / 2.0
    alpha = scale * math.cos(angle_rad)
    beta = scale * math.sin(angle_rad)
    linear = np.array([[alpha, -beta], [beta, alpha]], dtype=np.float64)
    corners = np.array([[-center_x, -center_y], [center_x, -center_y],
                        [center_x, center_y], [-center_x, center_y]])
    rotated = corners @ linear.T
    min_xy = rotated.min(axis=0)
    max_xy = rotated.max(axis=0)
    if inplace:
        # Rotate inside the original canvas, matching the historical template
        # search: the output keeps the template size and centre.
        out_width, out_height = width, height
        translation = np.array([center_x, center_y]) - linear @ np.array(
            [center_x, center_y])
    else:
        out_width = max(1, int(round(max_xy[0] - min_xy[0])))
        out_height = max(1, int(round(max_xy[1] - min_xy[1])))
        # v = R (s - center) - min_xy  =>  translation = -R @ center - min_xy
        translation = -(linear @ np.array([center_x, center_y])) - min_xy
    matrix = np.array([
        [alpha, -beta, translation[0]],
        [beta, alpha, translation[1]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    variant = cv2.warpAffine(pixels, matrix[:2, :], (out_width, out_height),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    variant_mask = None
    if mask is not None:
        warped_mask = cv2.warpAffine(np.asarray(mask, dtype=np.float32),
                                     matrix[:2, :], (out_width, out_height),
                                     flags=cv2.INTER_NEAREST,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        variant_mask = (warped_mask > 0.5).astype(np.float32)
    return variant, matrix, variant_mask


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
    crop = crop_region(pixels, spec.box, source=reference_space, destination=support_space)
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
    )
    return observed(support_space, prepared, {"support_shape": support_pixels.shape})


def _refine_peak(scores: Array, x: int, y: int) -> Tuple[float, float]:
    """Refine one correlation peak with two independent 1-D quadratic fits.

    Args:
        scores: Correlation map.
        x: Integer peak column.
        y: Integer peak row.

    Returns:
        Refined ``(x, y)`` in correlation-map coordinates.
    """
    height, width = scores.shape
    refined_x, refined_y = float(x), float(y)
    if 0 < x < width - 1:
        values = scores[y, x - 1:x + 2].astype(np.float64)
        coeffs = np.polyfit(np.array([x - 1, x, x + 1], dtype=np.float64), values, 2)
        if abs(coeffs[0]) > 1e-12:
            refined_x = float(-coeffs[1] / (2.0 * coeffs[0]))
    if 0 < y < height - 1:
        values = scores[y - 1:y + 2, x].astype(np.float64)
        coeffs = np.polyfit(np.array([y - 1, y, y + 1], dtype=np.float64), values, 2)
        if abs(coeffs[0]) > 1e-12:
            refined_y = float(-coeffs[1] / (2.0 * coeffs[0]))
    return refined_x, refined_y


def _peaks(scores: Array, threshold: float, limit: int = 0) -> List[Tuple[int, int, float]]:
    """Return local maxima of a correlation map above a threshold.

    Uses a vectorised 3x3 dilation instead of a Python scan, so a full-frame
    heatmap costs the same as ``matchTemplate`` instead of millions of loop
    iterations.

    Args:
        scores: Correlation map.
        threshold: Minimum accepted score.
        limit: Optional cap on candidates, keeping the highest scores.

    Returns:
        ``(x, y, score)`` tuples, strongest first.
    """
    finite = np.nan_to_num(scores, nan=-1.0, posinf=-1.0, neginf=-1.0)
    dilated = cv2.dilate(finite, np.ones((3, 3), dtype=np.float32))
    mask = (finite >= threshold) & (finite >= dilated - 1e-12)
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return []
    values = finite[ys, xs]
    order = np.argsort(-values)
    if limit and order.size > limit:
        order = order[:limit]
    return [(int(xs[i]), int(ys[i]), float(values[i])) for i in order]


def _box_iou(a: TemplateMatch, b: TemplateMatch) -> float:
    """Return the axis-aligned IoU of two matched boxes."""
    ax1, ay1 = a.center[0] - a.size_wh[0] / 2.0, a.center[1] - a.size_wh[1] / 2.0
    ax2, ay2 = a.center[0] + a.size_wh[0] / 2.0, a.center[1] + a.size_wh[1] / 2.0
    bx1, by1 = b.center[0] - b.size_wh[0] / 2.0, b.center[1] - b.size_wh[1] / 2.0
    bx2, by2 = b.center[0] + b.size_wh[0] / 2.0, b.center[1] + b.size_wh[1] / 2.0
    inter_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = inter_width * inter_height
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def _suppress(candidates: List[TemplateMatch], max_overlap: float) -> List[TemplateMatch]:
    """Greedy IoU suppression, strongest first.

    Args:
        candidates: All located instances.
        max_overlap: Maximum allowed IoU between kept matches; ``1.0`` disables
            suppression and ``0.0`` keeps only non-overlapping boxes.

    Returns:
        Kept matches sorted by score.
    """
    ordered = sorted(candidates, key=lambda item: item.score, reverse=True)
    if max_overlap >= 1.0:
        return ordered
    kept: List[TemplateMatch] = []
    for candidate in ordered:
        if all(_box_iou(candidate, other) <= max_overlap for other in kept):
            kept.append(candidate)
    return kept


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
            variant, matrix, variant_mask = _variant(
                np.asarray(template, dtype=np.float32), angle, scale,
                None if prepared.mask is None else prepared.mask,
                inplace=options.inplace_variants)
            variant_height, variant_width = variant.shape[:2]
            if variant_height > padded.shape[0] or variant_width > padded.shape[1]:
                unavailable += 1
                continue
            if variant_mask is not None:
                scores = cv2.matchTemplate(padded, variant, cv2.TM_CCORR_NORMED,
                                           mask=variant_mask)
            else:
                scores = cv2.matchTemplate(padded, variant, cv2.TM_CCOEFF_NORMED)
            scores = np.nan_to_num(scores, nan=-1.0, posinf=-1.0, neginf=-1.0)
            searched += 1
            if options.max_matches == 1:
                _, best, _, best_loc = cv2.minMaxLoc(scores)
                peaks = [(best_loc[0], best_loc[1], float(best))] if best >= options.score_threshold else []
            else:
                peaks = _peaks(scores, options.score_threshold, limit=8 * max(options.max_matches, 8))
            for x, y, score in peaks:
                peak_x, peak_y = (_refine_peak(scores, x, y) if options.subpixel
                                  else (float(x), float(y)))
                center_x = peak_x + variant_width / 2.0 - pad
                center_y = peak_y + variant_height / 2.0 - pad
                if not (0.0 <= center_x <= gray.shape[1] - 1
                        and 0.0 <= center_y <= gray.shape[0] - 1):
                    continue
                alpha = scale * math.cos(angle)
                beta = scale * math.sin(angle)
                linear = np.array([[alpha, -beta], [beta, alpha]], dtype=np.float64)
                support_center = np.array([support_width / 2.0, support_height / 2.0])
                offset = linear @ (np.asarray(prepared.anchor_support) - support_center)
                anchor_x = center_x + offset[0]
                anchor_y = center_y + offset[1]
                support_to_image_matrix = np.array([
                    [alpha, -beta, center_x - (alpha * support_center[0]
                                               - beta * support_center[1])],
                    [beta, alpha, center_y - (beta * support_center[0]
                                              + alpha * support_center[1])],
                    [0.0, 0.0, 1.0],
                ], dtype=np.float64)
                mapping = make_mapping(support_to_image_matrix,
                                       source=prepared.support_space,
                                       destination=image_space, kind="similarity")
                candidates.append(TemplateMatch(
                    center=(float(center_x), float(center_y)),
                    anchor=(float(anchor_x), float(anchor_y)),
                    angle_rad=float(angle), scale=float(scale), score=float(score),
                    support_to_image=mapping, peak_xy=(int(x), int(y)),
                    size_wh=(float(variant_width), float(variant_height)),
                ))
    diagnostics = {
        "requested_variant_count": requested,
        "searched_variant_count": searched,
        "unavailable_variant_count": unavailable,
        "pruned_variant_count": 0,
        "search_complete": searched + unavailable == requested,
    }
    if not candidates:
        return invalid(image_space, "no_matches", diagnostics)
    selected = _suppress(candidates, options.max_overlap)
    if options.max_matches:
        selected = selected[: options.max_matches]
    return observed(image_space, MatchSet(matches=tuple(selected)), diagnostics)