"""Edge-direction models and scoring for shape-based matching.

A shape model stores only *edge points with unit gradient directions*, and
scores an image by the normalized dot product of those directions with the
image gradient at the corresponding places. Two consequences matter:

* The score is invariant to how bright the part is, because only directions
  enter the numerator.
* Every term is bounded by one, so ``score_so_far + (remaining points)`` is a
  valid upper bound on the final score. That is what allows a candidate to be
  abandoned after a fraction of its points.

Normalized cross-correlation, which the rest of this package uses, offers
neither: it scores a dense intensity template, the remaining terms are
unbounded, and the denominator couples the whole window. No safe partial-sum
bound exists, so the same early rejection cannot be built on it.

The cost models differ too. Correlation costs ``image_area * support_area`` (or
``O(A log A)`` through a DFT); this costs ``points * positions`` and does not
grow with the support area at all, which is what makes a support that nearly
fills the frame affordable.

Images are not resampled per pose: the gradient planes are computed once and
every angle only rotates the point list. Rotating a few thousand points is
cheap next to rendering a rotated support.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from measure.computation_types import (
    Array,
    Observation,
    Space,
    contract_error,
    invalid,
    observed,
)

#: Fraction of a support's pixels kept as edge points when no explicit contrast
#: floor is given. Measured on a real support: the top 10% of gradient magnitude
#: still describes the part, while a lower cut-off mostly adds flat-area noise.
DEFAULT_CONTRAST_PERCENTILE = 90.0

#: Points kept per model. Cost is proportional to this, and 2000 kept the score
#: well separated (0.83 for the true pose against 0.23 for a 200 px shift) at
#: about 0.2 ms per candidate position.
DEFAULT_MAX_POINTS = 2000

#: Below this many points a pose is not determined, so the model is refused.
MIN_POINTS = 4

#: Gradient magnitude below this is treated as flat and never becomes a point.
MIN_MAGNITUDE = 1e-6


@dataclass(frozen=True)
class ShapeModel:
    """One level of an edge-direction model, in support coordinates.

    Attributes:
        points_y: Row offset of every point inside the model box.
        points_x: Column offset of every point inside the model box.
        dir_y: Unit gradient ``dy`` at each point.
        dir_x: Unit gradient ``dx`` at each point.
        size_hw: Height and width of the model box.
        anchor_y: Anchor row offset inside the model box.
        anchor_x: Anchor column offset inside the model box.
        level: Pyramid level the model was built for; 0 is full resolution.
    """

    points_y: Array
    points_x: Array
    dir_y: Array
    dir_x: Array
    size_hw: Tuple[int, int]
    anchor_y: float
    anchor_x: float
    level: int

    @property
    def count(self) -> int:
        """Return how many points the model carries."""
        return int(self.points_y.size)


def gradient_planes(image: Array) -> Tuple[Array, Array]:
    """Return the ``float32`` gradient planes of one grayscale image.

    These are computed once per level and reused for every pose, which is what
    keeps a multi-angle search from re-deriving them per angle.

    Args:
        image: Grayscale ``uint8`` or ``float32`` image.

    Returns:
        ``(gradient_x, gradient_y)`` as ``float32`` arrays shaped like the image.

    Raises:
        ComputationContractError: When the image is not two-dimensional.
    """
    pixels = np.asarray(image)
    if pixels.ndim != 2:
        raise contract_error("contract.invalid_parameter",
                             "shape matching needs a grayscale image", "image")
    gray = np.ascontiguousarray(pixels, dtype=np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return gx, gy


def _contrast_floor(magnitude: Array, valid: Array, min_contrast: float) -> float:
    """Return the gradient magnitude above which a pixel becomes a point.

    Args:
        magnitude: Gradient magnitude plane.
        valid: Boolean mask of usable pixels.
        min_contrast: Explicit floor, or ``0.0`` to derive one.

    Returns:
        A non-negative floor.
    """
    if min_contrast > 0.0:
        return float(min_contrast)
    values = magnitude[valid]
    if values.size == 0:
        return float("inf")
    return float(np.percentile(values, DEFAULT_CONTRAST_PERCENTILE))


def _scatter(rows: Array, cols: Array, magnitude: Array, max_points: int
             ) -> Tuple[Array, Array]:
    """Select at most ``max_points`` points spread over the support.

    Taking simply the strongest points clusters them on the busiest edge and
    leaves the rest of the part unrepresented. Instead every point is assigned
    to a grid cell sized so the grid has about ``max_points`` cells, and the
    strongest point per cell is kept. Coverage stays even and the pass is linear.

    Args:
        rows: Candidate row offsets.
        cols: Candidate column offsets.
        magnitude: Gradient magnitude at each candidate.
        max_points: Upper bound on the returned count.

    Returns:
        The kept row and column offsets.
    """
    if rows.size <= max_points:
        return rows, cols
    height = int(rows.max()) + 1
    width = int(cols.max()) + 1
    cells = max(1, int(math.ceil(math.sqrt(max_points))))
    cell_h = max(1, height / cells)
    cell_w = max(1, width / cells)
    keys = ((rows / cell_h).astype(np.int64) * cells
            + (cols / cell_w).astype(np.int64))
    order = np.lexsort((-magnitude, keys))
    ordered_keys = keys[order]
    first = np.ones(ordered_keys.size, dtype=bool)
    first[1:] = ordered_keys[1:] != ordered_keys[:-1]
    picked = order[first][:max_points]
    return rows[picked], cols[picked]


def build_shape_model(pixels: Array, space: Space, *, mask: Optional[Array] = None,
                      min_contrast: float = 0.0,
                      max_points: int = DEFAULT_MAX_POINTS,
                      anchor: Optional[Tuple[float, float]] = None,
                      level: int = 0) -> Observation[ShapeModel]:
    """Extract edge points and gradient directions from one support.

    Args:
        pixels: Support pixels, grayscale.
        space: Space the support lives in; carried into the observation.
        mask: Optional boolean footprint; points outside it are ignored.
        min_contrast: Gradient magnitude floor, or ``0.0`` to derive one from
            the support's own magnitude distribution.
        max_points: Upper bound on the number of points kept.
        anchor: ``(row, column)`` of the anchor inside the support; defaults to
            the support centre.
        level: Pyramid level this support belongs to, recorded on the model.

    Returns:
        A valid model, or an invalid observation with ``template_no_texture``
        when the support has too little structure to describe.

    Raises:
        ComputationContractError: On a non-positive ``max_points``.
    """
    if max_points < MIN_POINTS:
        raise contract_error("contract.invalid_parameter",
                             f"max_points must be at least {MIN_POINTS}",
                             "shape.max_points")
    array = np.ascontiguousarray(pixels, dtype=np.float32)
    height, width = array.shape[:2]
    if mask is None:
        valid = np.ones((height, width), dtype=bool)
    else:
        valid = np.asarray(mask, dtype=bool)
        if valid.shape != (height, width):
            raise contract_error("contract.invalid_shape",
                                 "mask must match the support size", "shape.mask")
    gx, gy = gradient_planes(array)
    magnitude = np.hypot(gx, gy)
    floor = _contrast_floor(magnitude, valid, min_contrast)
    strong = valid & (magnitude > floor) & (magnitude > MIN_MAGNITUDE)
    rows, cols = np.nonzero(strong)
    if rows.size < MIN_POINTS:
        return invalid(space, "template_no_texture",
                       {"shape_points": int(rows.size)})
    rows, cols = _scatter(rows, cols, magnitude[rows, cols], int(max_points))
    if rows.size < MIN_POINTS:
        return invalid(space, "template_no_texture",
                       {"shape_points": int(rows.size)})
    dir_x = (gx[rows, cols] / magnitude[rows, cols]).astype(np.float32)
    dir_y = (gy[rows, cols] / magnitude[rows, cols]).astype(np.float32)
    if anchor is None:
        anchor_y, anchor_x = height / 2.0, width / 2.0
    else:
        anchor_y, anchor_x = float(anchor[0]), float(anchor[1])
    model = ShapeModel(
        points_y=np.ascontiguousarray(rows, dtype=np.float32),
        points_x=np.ascontiguousarray(cols, dtype=np.float32),
        dir_y=np.ascontiguousarray(dir_y),
        dir_x=np.ascontiguousarray(dir_x),
        size_hw=(int(height), int(width)),
        anchor_y=anchor_y,
        anchor_x=anchor_x,
        level=int(level),
    )
    return observed(space, model, {"shape_points": model.count})


def rotate_model(model: ShapeModel, angle_rad: float) -> ShapeModel:
    """Rotate a model's points and directions about its centre.

    The sign convention matches ``template_search.render_variant``: a positive
    angle turns the support the same way a rendered variant is turned, so a
    shape search and a correlation search over the same grid describe the same
    pose. No image is resampled; only the point list moves.

    Args:
        model: Model to rotate.
        angle_rad: Clockwise pose angle in radians.

    Returns:
        A new model whose box tightly bounds the rotated points, with the
        anchor moved to its rotated place.
    """
    height, width = model.size_hw
    centre_y, centre_x = height / 2.0, width / 2.0
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    dy = model.points_y - centre_y
    dx = model.points_x - centre_x
    # (x, y) -> (x cos + y sin, -x sin + y cos), the convention _variant_geometry
    # uses for rendered variants.
    rot_x = dx * cos_a + dy * sin_a
    rot_y = -dx * sin_a + dy * cos_a
    anchor_dy = model.anchor_y - centre_y
    anchor_dx = model.anchor_x - centre_x
    anchor_x = anchor_dx * cos_a + anchor_dy * sin_a
    anchor_y = -anchor_dx * sin_a + anchor_dy * cos_a
    low_x = float(min(rot_x.min(), anchor_x))
    low_y = float(min(rot_y.min(), anchor_y))
    high_x = float(max(rot_x.max(), anchor_x))
    high_y = float(max(rot_y.max(), anchor_y))
    left = math.floor(low_x)
    top = math.floor(low_y)
    out_width = int(math.ceil(high_x) - left) + 1
    out_height = int(math.ceil(high_y) - top) + 1
    return ShapeModel(
        points_y=np.ascontiguousarray(rot_y - top, dtype=np.float32),
        points_x=np.ascontiguousarray(rot_x - left, dtype=np.float32),
        dir_y=np.ascontiguousarray(-model.dir_x * sin_a + model.dir_y * cos_a),
        dir_x=np.ascontiguousarray(model.dir_x * cos_a + model.dir_y * sin_a),
        size_hw=(out_height, out_width),
        anchor_y=anchor_y - top,
        anchor_x=anchor_x - left,
        level=model.level,
    )


def dense_kernels(model: ShapeModel) -> Tuple[Array, Array]:
    """Return the sparse correlation kernels for one model pose.

    Args:
        model: A posed model (see ``rotate_model``).

    Returns:
        ``(kernel_x, kernel_y)``, both zero except at the model's points.
    """
    height, width = model.size_hw
    kernel_x = np.zeros((height, width), dtype=np.float32)
    kernel_y = np.zeros((height, width), dtype=np.float32)
    cols = np.rint(model.points_x).astype(np.int64)
    rows = np.rint(model.points_y).astype(np.int64)
    np.add.at(kernel_x, (rows, cols), model.dir_x)
    np.add.at(kernel_y, (rows, cols), model.dir_y)
    return kernel_x, kernel_y


def point_energy(gx: Array, gy: Array, model: ShapeModel) -> Array:
    """Return the summed gradient energy at the model's own points.

    The denominator must cover exactly the points the numerator covers. Using
    the whole window instead counts edges the model does not describe, which
    makes a correct match score far below one (measured 0.31 instead of 0.99 on
    a textured patch).

    Args:
        gx: Gradient ``x`` plane of the search image.
        gy: Gradient ``y`` plane of the search image.
        model: A posed model whose box fits inside the image.

    Returns:
        A ``float32`` map shaped like the position grid.
    """
    height, width = model.size_hw
    ones = np.zeros((height, width), dtype=np.float32)
    cols = np.rint(model.points_x).astype(np.int64)
    rows = np.rint(model.points_y).astype(np.int64)
    ones[rows, cols] = 1.0
    image_x = np.ascontiguousarray(gx, dtype=np.float32)
    image_y = np.ascontiguousarray(gy, dtype=np.float32)
    energy_x = cv2.matchTemplate(image_x * image_x, ones, cv2.TM_CCORR)
    energy_y = cv2.matchTemplate(image_y * image_y, ones, cv2.TM_CCORR)
    return (energy_x + energy_y).astype(np.float32)


def score_map(gx: Array, gy: Array, model: ShapeModel,
              energy: Optional[Array] = None) -> Array:
    """Return the shape score of every window position.

    Args:
        gx: Gradient ``x`` plane of the search image.
        gy: Gradient ``y`` plane of the search image.
        model: A posed model whose box fits inside the image.
        energy: Optional precomputed ``point_energy`` for this model.

    Returns:
        A ``float32`` map of scores in ``[-1, 1]``, shaped like the position
        grid (``H - h + 1`` by ``W - w + 1``).
    """
    kernel_x, kernel_y = dense_kernels(model)
    image_x = np.ascontiguousarray(gx, dtype=np.float32)
    image_y = np.ascontiguousarray(gy, dtype=np.float32)
    # TM_CCORR is a plain dot product, so a signed sparse kernel is valid here;
    # the normalized variants would reject the negative directions.
    numerator = (cv2.matchTemplate(image_x, kernel_x, cv2.TM_CCORR)
                 + cv2.matchTemplate(image_y, kernel_y, cv2.TM_CCORR))
    if energy is None:
        energy = point_energy(gx, gy, model)
    denominator = np.sqrt(np.maximum(energy, 1e-6)) * math.sqrt(max(model.count, 1))
    scores = numerator / denominator
    return np.clip(scores, -1.0, 1.0).astype(np.float32)


def score_at(gx: Array, gy: Array, model: ShapeModel,
             rows: Sequence[int], cols: Sequence[int]) -> Array:
    """Score a few explicit window positions without building a full map.

    Used once the coarse pass has narrowed the search: gathering a few thousand
    points per position costs a fraction of a millisecond, whereas another dense
    map would repeat the whole-image work.

    Args:
        gx: Gradient ``x`` plane of the search image.
        gy: Gradient ``y`` plane of the search image.
        model: A posed model.
        rows: Window top offsets, one per position.
        cols: Window left offsets, one per position.

    Returns:
        A ``float32`` vector of scores in ``[-1, 1]``.
    """
    image_x = np.asarray(gx, dtype=np.float32)
    image_y = np.asarray(gy, dtype=np.float32)
    point_rows = np.rint(model.points_y).astype(np.int64)
    point_cols = np.rint(model.points_x).astype(np.int64)
    norm = math.sqrt(max(model.count, 1))
    scores = np.empty(len(rows), dtype=np.float32)
    for index in range(len(rows)):
        sample_y = image_y[rows[index] + point_rows, cols[index] + point_cols]
        sample_x = image_x[rows[index] + point_rows, cols[index] + point_cols]
        numerator = float((model.dir_y * sample_y + model.dir_x * sample_x).sum())
        energy = float((sample_y * sample_y + sample_x * sample_x).sum())
        scores[index] = float(np.clip(numerator / (math.sqrt(max(energy, 1e-6)) * norm),
                                      -1.0, 1.0))
    return scores


def greedy_bound(score_so_far: float, used_points: int, total_points: int,
                 greediness: float) -> float:
    """Return the score below which a candidate cannot still pay off.

    Halcon interpolates between two bounds. With ``greediness`` 0 the bound is
    the safe one: every remaining point can contribute at most one, so
    ``score - (1 - used/total)`` can never reject a real match. With
    ``greediness`` 1 it is the optimistic one, ``score * used/total``, which
    rejects far more but can miss a weak target.

    Args:
        score_so_far: Score computed from the points evaluated so far.
        used_points: Number of points already summed.
        total_points: Number of points in the model.
        greediness: Trade-off in ``[0, 1]``; 0 is safe, 1 is fastest.

    Returns:
        The floor a candidate's current score is compared against.
    """
    total = max(1, int(total_points))
    used = min(max(0, int(used_points)), total)
    fraction = used / float(total)
    safe = score_so_far - (1.0 - fraction)
    optimistic = score_so_far * fraction
    return safe + greediness * (optimistic - safe)
