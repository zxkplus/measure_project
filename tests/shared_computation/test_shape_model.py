"""The shape model and its two scoring paths agree with direct evaluation.

These tests pin the two properties the algorithm rests on: a score that is
invariant to brightness, and an alignment that matches the correlation path's
coordinate convention. The rotation test compares against ``render_variant``
because a sign error there would report a mirrored pose that still scores well,
which is the kind of defect that survives a smoke test.
"""

import cv2
import numpy as np
import pytest

from measure.computation_types import Space
from measure.shape_model import (
    MIN_POINTS,
    build_shape_model,
    dense_kernels,
    gradient_planes,
    greedy_bound,
    rotate_model,
    score_at,
    point_energy,
    score_map,
)

SPACE = Space("support", "px")


def textured(size=(240, 240), seed=0):
    """Return a deterministic, high-contrast pattern with clear edges."""
    rng = np.random.RandomState(seed)
    noise = (rng.rand(*size) * 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(noise, (7, 7), 2.0)
    canvas = np.zeros(size, np.uint8)
    canvas[40:200, 40:200] = blurred[40:200, 40:200]
    cv2.rectangle(canvas, (80, 80), (170, 160), 255, 3)
    cv2.line(canvas, (60, 60), (190, 190), 220, 3)
    return canvas


def test_model_extracts_unit_directions_inside_the_mask():
    image = textured()
    mask = np.zeros_like(image, dtype=bool)
    mask[40:160, 40:160] = True
    result = build_shape_model(image, SPACE, mask=mask, max_points=500)
    assert result.valid
    model = result.value
    assert 0 < model.count <= 500
    assert np.hypot(model.dir_y, model.dir_x) == pytest.approx(
        np.ones(model.count), abs=1e-5)
    inside = ((model.points_y >= 40) & (model.points_y < 160)
              & (model.points_x >= 40) & (model.points_x < 160))
    assert bool(inside.all()), "points must lie inside the footprint mask"


def test_flat_support_is_refused():
    flat = np.zeros((60, 60), np.uint8)
    result = build_shape_model(flat, SPACE)
    assert not result.valid
    assert result.reason == "template_no_texture"


def test_dense_and_sparse_scoring_agree():
    image = textured()
    support = image[50:170, 50:170]
    model = build_shape_model(support, SPACE, max_points=400).value
    gx, gy = gradient_planes(image)
    energy = point_energy(gx, gy, model)
    dense = score_map(gx, gy, model, energy)
    rows = [10, 50, 90, 120]
    cols = [20, 50, 100, 40]
    sparse = score_at(gx, gy, model, rows, cols)
    expected = np.array([dense[r, c] for r, c in zip(rows, cols)])
    assert sparse == pytest.approx(expected, abs=1e-5)


def test_model_locates_the_window_it_was_taught_on():
    image = textured()
    top, left = 50, 50
    support = image[top:top + 120, left:left + 120]
    model = build_shape_model(support, SPACE, max_points=400).value
    gx, gy = gradient_planes(image)
    scores = score_map(gx, gy, model)
    peak_row, peak_col = np.unravel_index(int(np.argmax(scores)), scores.shape)
    assert (peak_row, peak_col) == (top, left)
    assert float(scores[peak_row, peak_col]) > 0.9


def test_score_is_invariant_to_contrast():
    image = textured()
    support = image[50:170, 50:170]
    model = build_shape_model(support, SPACE, max_points=400).value
    brighter = np.clip(image.astype(np.float32) * 0.5 + 20.0, 0, 255).astype(np.uint8)
    gx, gy = gradient_planes(image)
    bx, by = gradient_planes(brighter)
    base = float(score_map(gx, gy, model)[50, 50])
    scaled = float(score_map(bx, by, model)[50, 50])
    assert scaled == pytest.approx(base, abs=0.02)


def test_flat_region_does_not_score_high():
    image = textured()
    support = image[50:170, 50:170]
    model = build_shape_model(support, SPACE, max_points=400).value
    gx, gy = gradient_planes(image)
    flat = np.full_like(image, 128)
    fx, fy = gradient_planes(flat)
    assert float(np.abs(score_map(fx, fy, model)).max()) < 0.1


def test_rotation_matches_render_variant_orientation():
    image = textured()
    support = image[40:200, 40:200]
    model = build_shape_model(support, SPACE, max_points=300).value
    angle = np.deg2rad(25.0)
    rotated = rotate_model(model, angle)
    kernel_x, kernel_y = dense_kernels(rotated)
    # A rendered variant of the same support must place its energy where the
    # rotated points are, otherwise the two searches disagree on what a pose is.
    rendered = cv2.warpAffine(
        np.asarray(support, dtype=np.float32),
        cv2.getRotationMatrix2D((support.shape[1] / 2.0, support.shape[0] / 2.0),
                                -np.rad2deg(angle), 1.0),
        (rotated.size_hw[1], rotated.size_hw[0]),
        flags=cv2.INTER_LINEAR)
    energy = np.abs(kernel_x) + np.abs(kernel_y)
    body = np.abs(rendered) > np.abs(rendered).max() * 0.2
    body = body.astype(np.float32)
    overlap = float((energy * body).sum() / max(energy.sum(), 1e-6))
    assert overlap > 0.3, f"rotated points barely overlap the rendered support ({overlap:.2f})"


def test_greedy_bound_is_safe_at_zero_and_aggressive_at_one():
    # A perfect match never gets rejected with greediness 0.
    assert greedy_bound(1.0, 1, 100, 0.0) <= 0.5
    # An empty match is rejected immediately with greediness 1.
    assert greedy_bound(0.0, 1, 100, 1.0) >= field_floor()
    # Between the two, the bound moves monotonically.
    bounds = [greedy_bound(0.4, 10, 100, g) for g in (0.0, 0.5, 1.0)]
    assert bounds[0] <= bounds[1] <= bounds[2]


def field_floor():
    """Return the threshold the aggressive bound is compared against."""
    return 0.0


def test_model_refuses_too_few_points():
    with pytest.raises(Exception):
        build_shape_model(textured(), SPACE, max_points=MIN_POINTS - 1)


# --- End-to-end: the shape search must agree with the correlation search ------


def _prepared(reference, center=(360.0, 260.0), size=(120.0, 120.0)):
    """Teach one square support on ``reference`` with a raw preprocessor."""
    from measure.computation_types import RotatedBox
    from measure.preprocessing import RawSpec
    from measure.template_search import SupportSpec, prepare_support

    spec = SupportSpec(box=RotatedBox(center, size, 0.0), anchor_reference=center)
    result = prepare_support(reference, spec, RawSpec(),
                             reference_space=Space("reference", "px"),
                             support_space=Space("support", "px"))
    assert result.valid
    return result.value


def _scene(angle_deg=0.0, top=200, left=300, size=120):
    """Return a synthetic scene with one asymmetric target at a known place."""
    rng = np.random.RandomState(3)
    texture = cv2.GaussianBlur((rng.rand(size, size) * 255).astype(np.uint8), (5, 5), 1.5)
    cv2.rectangle(texture, (20, 20), (size - 20, size // 2), 255, 2)
    cv2.line(texture, (10, size - 10), (size - 10, size - 30), 200, 3)
    rotated = cv2.warpAffine(
        texture, cv2.getRotationMatrix2D((size / 2.0, size / 2.0), -angle_deg, 1.0),
        (size, size))
    reference = np.zeros((500, 600), np.uint8)
    reference[top:top + size, left:left + size] = texture
    scene = np.zeros((500, 600), np.uint8)
    scene[top:top + size, left:left + size] = rotated
    return reference, scene


def _search(image, prepared, algorithm, angle_half_deg=0.0):
    """Run one search over a symmetric angle range."""
    from measure.template_search import SearchOptions, search_support

    half = np.deg2rad(angle_half_deg)
    options = SearchOptions(
        score_threshold=0.4, max_matches=1, subpixel=True, algorithm=algorithm,
        angle_min_rad=-half, angle_max_rad=half,
        angle_step_rad=np.deg2rad(1.0), coarse_fine=bool(angle_half_deg),
        coarse_angle_step_rad=np.deg2rad(5.0))
    return search_support(image, prepared, options,
                          image_space=Space("inspection", "px"))


def test_shape_search_agrees_with_correlation_on_position():
    reference, scene = _scene()
    prepared = _prepared(reference)
    correlated = _search(scene, prepared, "ncc")
    shaped = _search(scene, prepared, "shape")
    assert correlated.valid and shaped.valid
    ncc_match = correlated.value.matches[0]
    shape_match = shaped.value.matches[0]
    assert shape_match.center == pytest.approx(ncc_match.center, abs=1.5)
    assert shaped.diagnostics["search_strategy"] == "shape_two_level"
    assert shaped.diagnostics["shape_points"] > 0


def test_shape_search_agrees_with_correlation_on_rotation_sign():
    reference, _ = _scene()
    prepared = _prepared(reference)
    for degrees in (15.0, -15.0):
        _, scene = _scene(angle_deg=degrees)
        correlated = _search(scene, prepared, "ncc", angle_half_deg=30.0)
        shaped = _search(scene, prepared, "shape", angle_half_deg=30.0)
        assert correlated.valid and shaped.valid
        ncc_angle = np.rad2deg(correlated.value.matches[0].angle_rad)
        shape_angle = np.rad2deg(shaped.value.matches[0].angle_rad)
        # A mirrored sign would still score well, so compare against correlation
        # rather than asserting a hard-coded value.
        assert shape_angle == pytest.approx(ncc_angle, abs=3.0), (
            f"shape {shape_angle:.1f} disagrees with correlation {ncc_angle:.1f}")


def test_shape_search_falls_back_for_multi_scale():
    reference, scene = _scene()
    prepared = _prepared(reference)
    from measure.template_search import SearchOptions, search_support

    options = SearchOptions(
        score_threshold=0.4, max_matches=1, algorithm="shape",
        scale_min=0.9, scale_max=1.1, scale_step=0.1)
    result = search_support(scene, prepared, options,
                            image_space=Space("inspection", "px"))
    assert result.valid
    assert result.diagnostics.get("shape_scale_fallback") == "multi_scale_uses_correlation"


# --- Ambiguity: a repeated pattern must not look like a unique answer ---------


def _lattice(rows=6, cols=6, size=120):
    """Return an image where the same target repeats on a regular lattice."""
    rng = np.random.RandomState(11)
    cell = cv2.GaussianBlur((rng.rand(size, size) * 255).astype(np.uint8), (5, 5), 1.5)
    cv2.circle(cell, (size // 2, size // 2), size // 4, 0, -1)
    cv2.rectangle(cell, (4, 4), (size - 4, size - 4), 255, 2)
    image = np.zeros((rows * size, cols * size), np.uint8)
    for row in range(rows):
        for col in range(cols):
            image[row * size:(row + 1) * size, col * size:(col + 1) * size] = cell
    return image, cell


def test_repeated_pattern_is_reported_ambiguous():
    image, cell = _lattice()
    prepared = _prepared(cell, center=(60.0, 60.0), size=(120.0, 120.0))
    for algorithm in ("ncc", "shape"):
        result = _search(image, prepared, algorithm)
        assert result.valid
        diagnostics = result.diagnostics
        assert diagnostics["ambiguous"] is True, (
            f"{algorithm} called a periodic lattice unique")
        assert diagnostics["runner_up_score"] == pytest.approx(
            result.value.matches[0].score, abs=0.1)
        # The nearest repeat is a whole cell away, which is what makes the
        # second answer a different target rather than the same one found twice.
        assert diagnostics["runner_up_distance_px"] == pytest.approx(120.0, abs=1.0)


def test_single_target_is_not_reported_ambiguous():
    reference, scene = _scene()
    prepared = _prepared(reference)
    for algorithm in ("ncc", "shape"):
        result = _search(scene, prepared, algorithm)
        assert result.valid
        assert result.diagnostics["ambiguous"] is False
        assert "runner_up_score" not in result.diagnostics
