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
