"""Template behavior is observed through prepared supports and complete search calls."""

import cv2
import numpy as np
import pytest


def prepare(api, space, reference, *, anchor=(80.0, 70.0), mask=None):
    spec = api("template_search", "SupportSpec")(
        box=api("computation_types", "RotatedBox")((80.0, 70.0), (40.0, 40.0), 0.0),
        anchor_reference=anchor, mask=mask,
    )
    return api("template_search", "prepare_support")(
        reference, spec, api("preprocessing", "RawSpec")(),
        reference_space=space("reference"), support_space=space("support"),
    )


def stamp(reference, *, center, angle_deg=0.0, scale=1.0, size=(240, 180)):
    """Render an independently specified pose to make a known search target."""
    matrix = cv2.getRotationMatrix2D((80.0, 70.0), -angle_deg, scale)
    matrix[:, 2] += np.array(center) - [80.0, 70.0]
    return cv2.warpAffine(reference, matrix, size, flags=cv2.INTER_LINEAR)


def test_preparation_keeps_anchor_distinct_from_support_center(api, space, textured_image):
    result = prepare(api, space, textured_image, anchor=(75.0, 66.0))
    assert result.valid and result.value.anchor_support == pytest.approx((15.0, 16.0))
    assert result.value.pixels.shape == (40, 40)
    assert not result.value.pixels.flags.writeable and not result.value.valid_mask.flags.writeable


def test_uniform_template_is_invalid_instead_of_correlating_everywhere(api, space):
    result = prepare(api, space, np.full((160, 200), 127, np.uint8))
    assert not result.valid and result.value is None and result.reason == "template_no_texture"


def test_color_preserving_raw_cannot_silently_enter_grayscale_matching(api, space, textured_image):
    spec = api("template_search", "SupportSpec")(
        box=api("computation_types", "RotatedBox")((80.0, 70.0), (40.0, 40.0), 0.0),
        anchor_reference=(80.0, 70.0),
    )
    with pytest.raises(api("computation_types", "ComputationContractError")):
        api("template_search", "prepare_support")(
            textured_image, spec, api("preprocessing", "RawSpec")(channel_mode="preserve"),
            reference_space=space("reference"), support_space=space("support"),
        )


def test_translation_returns_feature_anchor_and_explicit_support_mapping(
    api, space, textured_image,
):
    prepared = prepare(api, space, textured_image, anchor=(75.0, 66.0)).value
    image = stamp(textured_image, center=(120.0, 90.0))
    result = api("template_search", "search_support")(
        image, prepared, api("template_search", "SearchOptions")(
            score_threshold=0.95, coarse_fine=False, subpixel=False,
        ), image_space=space("inspection"),
    )
    assert result.valid and len(result.value.matches) == 1
    match = result.value.matches[0]
    assert match.center == pytest.approx((120.0, 90.0), abs=0.1)
    assert match.anchor == pytest.approx((115.0, 86.0), abs=0.1)
    np.testing.assert_allclose(match.support_to_image.matrix @ [15.0, 16.0, 1.0],
                               [115.0, 86.0, 1.0], atol=0.1)


def test_max_matches_zero_keeps_two_independent_instances(api, space, textured_image):
    prepared = prepare(api, space, textured_image).value
    image = np.maximum(stamp(textured_image, center=(55.0, 60.0)),
                       stamp(textured_image, center=(170.0, 120.0)))
    result = api("template_search", "search_support")(
        image, prepared, api("template_search", "SearchOptions")(
            max_matches=0, score_threshold=0.95, coarse_fine=False, subpixel=False,
        ), image_space=space("inspection"),
    )
    assert result.valid and len(result.value.matches) == 2
    centers = sorted(match.center for match in result.value.matches)
    np.testing.assert_allclose(centers, [[55.0, 60.0], [170.0, 120.0]], atol=0.2)


def test_no_match_has_no_stale_match_payload(api, space, textured_image):
    prepared = prepare(api, space, textured_image).value
    search = api("template_search", "search_support")
    options = api("template_search", "SearchOptions")(score_threshold=0.95, coarse_fine=False)
    assert search(textured_image, prepared, options, image_space=space("inspection")).valid
    result = search(
        np.zeros_like(textured_image), prepared, options, image_space=space("inspection"),
    )
    assert not result.valid and result.value is None and result.reason == "no_matches"


def test_search_keeps_shared_prepared_arrays_unchanged(api, space, textured_image):
    prepared = prepare(api, space, textured_image).value
    pixels, mask = prepared.pixels.copy(), prepared.valid_mask.copy()
    options = api("template_search", "SearchOptions")(coarse_fine=False)
    search = api("template_search", "search_support")
    first = search(textured_image, prepared, options, image_space=space("inspection"))
    second = search(textured_image, prepared, options, image_space=space("inspection"))
    assert first.valid and second.valid
    assert first.value.matches[0].anchor == pytest.approx(second.value.matches[0].anchor)
    np.testing.assert_array_equal(prepared.pixels, pixels)
    np.testing.assert_array_equal(prepared.valid_mask, mask)
    assert not prepared.pixels.flags.writeable


def test_d07_large_rotated_bounding_box_does_not_discard_valid_masked_support(api, space):
    reference = np.zeros((150, 150), np.uint8)
    rng = np.random.default_rng(7)
    reference[55:95, 55:95] = rng.integers(20, 240, (40, 40), dtype=np.uint8)
    mask = np.zeros((84, 84), bool)
    mask[22:62, 22:62] = True
    spec = api("template_search", "SupportSpec")(
        box=api("computation_types", "RotatedBox")((75.0, 75.0), (84.0, 84.0), 0.0),
        anchor_reference=(75.0, 75.0), mask=mask,
    )
    prepared = api("template_search", "prepare_support")(
        reference, spec, api("preprocessing", "RawSpec")(),
        reference_space=space("reference"), support_space=space("support"),
    ).value
    matrix = cv2.getRotationMatrix2D((75.0, 75.0), -45.0, 1.0)
    matrix[:, 2] += [-29.0, -29.0]
    image = cv2.warpAffine(reference, matrix, (93, 93))
    result = api("template_search", "search_support")(
        image, prepared, api("template_search", "SearchOptions")(
            angle_min_rad=np.pi / 4, angle_max_rad=np.pi / 4,
            score_threshold=0.8, coarse_fine=False, subpixel=False,
        ), image_space=space("inspection"),
    )
    assert result.valid
    assert result.value.matches[0].anchor == pytest.approx((46.0, 46.0), abs=1.0)
    assert result.diagnostics["unavailable_variant_count"] == 0


def test_d08_two_targets_keep_their_own_angle_and_scale_during_refinement(
    api, space, textured_image,
):
    prepared = prepare(api, space, textured_image).value
    image = np.maximum(stamp(textured_image, center=(55.0, 60.0), angle_deg=-10.0, scale=0.9),
                       stamp(textured_image, center=(170.0, 120.0), angle_deg=15.0, scale=1.1))
    result = api("template_search", "search_support")(
        image, prepared, api("template_search", "SearchOptions")(
            max_matches=2, score_threshold=0.6, coarse_fine=False,
            angle_min_rad=np.deg2rad(-15), angle_max_rad=np.deg2rad(20),
            angle_step_rad=np.deg2rad(5), scale_min=0.85, scale_max=1.15, scale_step=0.05,
        ), image_space=space("inspection"),
    )
    assert result.valid and len(result.value.matches) == 2
    left, right = sorted(result.value.matches, key=lambda match: match.anchor[0])
    assert left.anchor == pytest.approx((55.0, 60.0), abs=1.0)
    assert right.anchor == pytest.approx((170.0, 120.0), abs=1.0)
    assert left.angle_rad == pytest.approx(np.deg2rad(-10), abs=np.deg2rad(2))
    assert right.angle_rad == pytest.approx(np.deg2rad(15), abs=np.deg2rad(2))
    assert left.scale == pytest.approx(0.9, abs=0.03)
    assert right.scale == pytest.approx(1.1, abs=0.03)


def test_search_diagnostics_account_for_every_requested_variant(api, space, textured_image):
    prepared = prepare(api, space, textured_image).value
    result = api("template_search", "search_support")(
        textured_image, prepared, api("template_search", "SearchOptions")(
            coarse_fine=False, angle_min_rad=0, angle_max_rad=np.deg2rad(2),
            angle_step_rad=np.deg2rad(1),
        ), image_space=space("inspection"),
    )
    diagnostics = result.diagnostics
    assert diagnostics["requested_variant_count"] == 3
    assert diagnostics["searched_variant_count"] == 3
    assert diagnostics["unavailable_variant_count"] == diagnostics["pruned_variant_count"] == 0
    assert diagnostics["search_complete"] is True


def test_clockwise_rotation_search_recovers_the_stamped_pose(api, space, textured_image):
    prepared = prepare(api, space, textured_image).value
    image = stamp(textured_image, center=(120.0, 90.0), angle_deg=12.0)
    result = api("template_search", "search_support")(
        image, prepared, api("template_search", "SearchOptions")(
            score_threshold=0.6, coarse_fine=False, subpixel=True,
            angle_min_rad=np.deg2rad(8.0), angle_max_rad=np.deg2rad(16.0),
            angle_step_rad=np.deg2rad(1.0),
        ), image_space=space("inspection"),
    )
    assert result.valid
    match = result.value.matches[0]
    # One grid step of slack: the shared variant uses the continuous size/2
    # origin, while v2 rotates the pixel-corner box (a half-pixel difference
    # that can shift the best grid cell by one step).
    assert np.rad2deg(match.angle_rad) == pytest.approx(12.0, abs=1.5)
    assert match.scale == pytest.approx(1.0, abs=1e-9)
    assert match.center == pytest.approx((120.0, 90.0), abs=1.0)


def test_scale_search_recovers_the_stamped_scale(api, space, textured_image):
    prepared = prepare(api, space, textured_image).value
    image = stamp(textured_image, center=(120.0, 90.0), scale=1.08)
    result = api("template_search", "search_support")(
        image, prepared, api("template_search", "SearchOptions")(
            score_threshold=0.6, coarse_fine=False, subpixel=True,
            scale_min=1.02, scale_max=1.14, scale_step=0.02,
        ), image_space=space("inspection"),
    )
    assert result.valid
    match = result.value.matches[0]
    assert match.scale == pytest.approx(1.08, abs=0.021)
    assert match.center == pytest.approx((120.0, 90.0), abs=1.0)


def test_coarse_fine_search_matches_the_single_pass_result(api, space, textured_image):
    prepared = prepare(api, space, textured_image).value
    image = stamp(textured_image, center=(120.0, 90.0), angle_deg=12.0)
    options = dict(score_threshold=0.6, subpixel=True,
                   angle_min_rad=np.deg2rad(8.0), angle_max_rad=np.deg2rad(16.0),
                   angle_step_rad=np.deg2rad(1.0),
                   coarse_angle_step_rad=np.deg2rad(4.0))
    fine = api("template_search", "search_support")(
        image, prepared,
        api("template_search", "SearchOptions")(coarse_fine=False, **options),
        image_space=space("inspection"),
    )
    coarse = api("template_search", "search_support")(
        image, prepared,
        api("template_search", "SearchOptions")(coarse_fine=True, **options),
        image_space=space("inspection"),
    )
    assert coarse.valid and fine.valid
    for left, right in zip(coarse.value.matches, fine.value.matches):
        assert left.center == pytest.approx(right.center, abs=1.0)
        assert left.angle_rad == pytest.approx(right.angle_rad, abs=np.deg2rad(1.0))
        assert left.score == pytest.approx(right.score, abs=0.05)


def test_pyramid_search_refines_a_large_support_at_full_resolution(api, space):
    rng = np.random.RandomState(3)
    reference = cv2.GaussianBlur((rng.rand(700, 700) * 255).astype(np.uint8), (7, 7), 2.0)
    prepared = api("template_search", "prepare_support")(
        reference,
        api("template_search", "SupportSpec")(
            box=api("computation_types", "RotatedBox")((350.0, 350.0), (600.0, 600.0), 0.0),
            anchor_reference=(350.0, 350.0),
        ),
        api("preprocessing", "RawSpec")(),
        reference_space=space("reference"), support_space=space("support"),
    ).value
    image = np.zeros((1000, 1000), np.uint8)
    image[200:800, 200:800] = reference[50:650, 50:650]
    result = api("template_search", "search_support")(
        image, prepared, api("template_search", "SearchOptions")(
            score_threshold=0.6, pyramid_max_template_size=400,
        ), image_space=space("inspection"),
    )
    assert result.valid
    assert result.diagnostics["pyramid_factor"] < 1.0
    match = result.value.matches[0]
    assert match.center == pytest.approx((500.0, 500.0), abs=0.6)
    assert match.score > 0.99


def test_variant_rendering_snaps_the_footprint_onto_the_pixel_grid(api):
    pixels = np.zeros((30, 40), dtype=np.float32)
    rendered = api("template_search", "render_variant")(
        pixels, np.deg2rad(23.0), 1.0, origin="pixel_center")
    corners = np.array([[0.0, 0.0, 1.0], [39.0, 0.0, 1.0],
                        [0.0, 29.0, 1.0], [39.0, 29.0, 1.0]])
    mapped = corners @ rendered.matrix.T
    width, height = rendered.size_wh
    # Anchoring on the fractional minimum instead of floor(min) would let the
    # footprint run past the last whole pixel, i.e. sample off the search grid.
    for axis, extent in ((0, width), (1, height)):
        assert 0.0 <= mapped[:, axis].min() < 1.0
        assert mapped[:, axis].max() <= extent - 1 + 1e-9
    assert rendered.pixels.shape == (height, width)
    assert api("template_search", "variant_size")(
        pixels, np.deg2rad(23.0), 1.0, origin="pixel_center") == rendered.size_wh


def test_unrotated_variant_keeps_each_pixel_convention(api):
    pixels = np.arange(200, dtype=np.float32).reshape(10, 20)
    pixel_center = api("template_search", "render_variant")(
        pixels, 0.0, 1.0, origin="pixel_center")
    continuous = api("template_search", "render_variant")(
        pixels, 0.0, 1.0, origin="continuous")
    np.testing.assert_allclose(pixel_center.matrix, np.eye(3))
    np.testing.assert_allclose(continuous.matrix, np.eye(3))
    np.testing.assert_allclose(pixel_center.pixels, pixels)
    np.testing.assert_allclose(continuous.pixels, pixels)
    assert pixel_center.center == pytest.approx((9.5, 4.5))
    assert continuous.center == pytest.approx((10.0, 5.0))
    assert pixel_center.size_wh == continuous.size_wh == (20, 10)


def test_unknown_variant_origin_is_rejected(api):
    with pytest.raises(Exception) as error:
        api("template_search", "render_variant")(
            np.zeros((8, 8), dtype=np.float32), 0.0, 1.0, origin="corner")
    assert "origin" in str(error.value)


def test_decimation_and_pose_thinning_compose(api, space, textured_image):
    """``pyramid_levels`` and ``coarse_fine`` are independent knobs, not exclusive.

    They used to short-circuit each other: a decimated search forced the coarse
    grid to stay at full density, so enabling both cost more than enabling
    neither. Each knob must now change its own quantity and leave the recovered
    pose unchanged.
    """
    image = stamp(textured_image, center=(120.0, 90.0))
    prepared = prepare(api, space, textured_image).value
    options = api("template_search", "SearchOptions")

    def search(**overrides):
        return api("template_search", "search_support")(
            image, prepared, options(score_threshold=0.6, angle_step_rad=np.deg2rad(2.0),
                                     angle_min_rad=np.deg2rad(-8.0),
                                     angle_max_rad=np.deg2rad(8.0), **overrides),
            image_space=space("inspection"),
        )

    plain = search(coarse_fine=False, pyramid_levels=0)
    decimated = search(coarse_fine=False, pyramid_levels=1)
    thinned = search(coarse_fine=True, pyramid_levels=0)
    both = search(coarse_fine=True, pyramid_levels=1)
    for result in (plain, decimated, thinned, both):
        assert result.valid, result.reason
        assert result.value.matches[0].center == pytest.approx((120.0, 90.0), abs=0.2)

    # Decimation alone records a factor; pose thinning alone does not.
    assert plain.diagnostics.get("pyramid_factor") is None
    assert decimated.diagnostics["pyramid_factor"] == pytest.approx(0.5)
    assert both.diagnostics["pyramid_factor"] == pytest.approx(0.5)

    # Thinning cuts the coarse pose count; decimation does not.
    assert thinned.diagnostics["coarse_poses"] < plain.diagnostics["coarse_poses"]
    assert decimated.diagnostics["coarse_poses"] == plain.diagnostics["coarse_poses"]

    # Composing them pays the coarse cost of the thinned grid, not the full one.
    assert both.diagnostics["coarse_poses"] == thinned.diagnostics["coarse_poses"]
    assert both.diagnostics["searched_variant_count"] <= \
        thinned.diagnostics["searched_variant_count"] + 4


def test_single_pass_reports_no_neighbourhoods(api, space, textured_image):
    """With neither knob set the search is one full-resolution grid pass."""
    image = stamp(textured_image, center=(120.0, 90.0))
    prepared = prepare(api, space, textured_image).value
    result = api("template_search", "search_support")(
        image, prepared, api("template_search", "SearchOptions")(
            score_threshold=0.6, coarse_fine=False, pyramid_levels=0,
        ), image_space=space("inspection"),
    )
    assert result.valid
    assert result.diagnostics["fine_neighborhoods"] == 0
    assert result.diagnostics["pruned_variant_count"] == 0
    assert "pyramid_factor" not in result.diagnostics


def test_fine_candidates_refines_every_neighbourhood_by_default(api, space, textured_image):
    """The default refines all coarse neighbourhoods; the knob can bound it."""
    image = stamp(textured_image, center=(120.0, 90.0))
    prepared = prepare(api, space, textured_image).value
    options = api("template_search", "SearchOptions")

    def search(**overrides):
        return api("template_search", "search_support")(
            image, prepared, options(score_threshold=0.3, coarse_fine=True,
                                     pyramid_levels=1, angle_step_rad=np.deg2rad(5.0),
                                     angle_min_rad=np.deg2rad(-30.0),
                                     angle_max_rad=np.deg2rad(30.0), **overrides),
            image_space=space("inspection"),
        )

    unbounded = search()
    bounded = search(fine_candidates=1)
    assert unbounded.valid and bounded.valid
    assert options(fine_candidates=None).resolved_fine_candidates() is None
    assert options(fine_candidates=3).resolved_fine_candidates() == 3
    # A finite bound can only reduce the refined set, never enlarge it.
    assert bounded.diagnostics["fine_neighborhoods"] <= \
        unbounded.diagnostics["fine_neighborhoods"]
    assert max(1, bounded.diagnostics["fine_neighborhoods"]) == 1


def test_fine_candidates_environment_variable_bounds_the_search(api, space, textured_image,
                                                                monkeypatch):
    """A site can bound the refined set without editing models."""
    image = stamp(textured_image, center=(120.0, 90.0))
    prepared = prepare(api, space, textured_image).value
    monkeypatch.setenv("MEASURE_TEMPLATE_FINE_CANDIDATES", "2")
    options = api("template_search", "SearchOptions")(fine_candidates=None)
    assert options.resolved_fine_candidates() == 2
    result = api("template_search", "search_support")(
        image, prepared, api("template_search", "SearchOptions")(
            score_threshold=0.3, coarse_fine=True, pyramid_levels=1,
            angle_step_rad=np.deg2rad(5.0), angle_min_rad=np.deg2rad(-30.0),
            angle_max_rad=np.deg2rad(30.0),
        ), image_space=space("inspection"),
    )
    assert result.diagnostics["fine_neighborhoods"] <= 2
