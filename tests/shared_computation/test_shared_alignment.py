"""Correspondence estimates and control correction obey declared transform directions."""

import numpy as np
import pytest


def test_similarity_estimates_known_scale_rotation_and_translation(api, space):
    module = api("alignment")
    result = api("alignment", "estimate_similarity")(
        np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]]),
        np.array([[3.0, 4.0], [3.0, 24.0], [-17.0, 4.0]]),
        source=space("reference"), destination=space("observed"),
        options=module.SimilarityOptions(),
    )
    assert result.valid and result.value.scale == pytest.approx(2.0)
    assert result.value.angle_rad == pytest.approx(np.pi / 2)
    np.testing.assert_allclose(result.value.mapping.matrix,
                               [[0.0, -2.0, 3.0], [2.0, 0.0, 4.0], [0.0, 0.0, 1.0]],
                               atol=1e-10)


def test_d04_repeated_source_points_cannot_define_a_similarity(api, space):
    module = api("alignment")
    result = api("alignment", "estimate_similarity")(
        np.array([[1.0, 2.0], [1.0, 2.0]]), np.array([[3.0, 4.0], [5.0, 6.0]]),
        source=space("reference"), destination=space("observed"),
        options=module.SimilarityOptions(),
    )
    assert not result.valid and result.value is None
    assert result.reason == "degenerate_correspondences"


def test_single_correspondence_requires_explicit_translation_policy(api, space):
    module = api("alignment")
    result = api("alignment", "estimate_similarity")(
        np.array([[1.0, 2.0]]), np.array([[4.0, 6.0]]), source=space("reference"),
        destination=space("observed"), options=module.SimilarityOptions(),
    )
    assert not result.valid and result.value is None


def test_explicit_single_point_policy_returns_translation_without_inventing_scale(api, space):
    module = api("alignment")
    result = api("alignment", "estimate_similarity")(
        np.array([[1.0, 2.0]]), np.array([[4.0, 6.0]]), source=space("reference"),
        destination=space("observed"),
        options=module.SimilarityOptions(allow_single_translation=True, minimum_inliers=1),
    )
    assert result.valid and result.value.scale == 1.0 and result.value.angle_rad == 0.0
    np.testing.assert_allclose(result.value.mapping.matrix,
                               [[1.0, 0.0, 3.0], [0.0, 1.0, 4.0], [0.0, 0.0, 1.0]])


def test_ransac_rejects_an_outlier_without_losing_input_correspondence_identity(api, space):
    module = api("alignment")
    result = api("alignment", "estimate_similarity")(
        np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0], [20.0, 15.0]]),
        np.array([[3.0, 4.0], [23.0, 4.0], [3.0, 24.0], [23.0, 24.0], [1000.0, 1000.0]]),
        source=space("reference"), destination=space("observed"),
        options=module.SimilarityOptions(method="ransac", ransac_threshold=0.5,
                                         max_residual=0.5, minimum_inliers=4),
    )
    assert result.valid and result.value.scale == pytest.approx(2.0)
    np.testing.assert_array_equal(result.value.inliers, [True, True, True, True, False])
    assert len(result.value.residuals) == 5 and result.value.rms_residual < 1e-9


def test_ransac_does_not_change_numpy_global_random_state(api, space):
    estimate = api("alignment", "estimate_similarity")
    options = api("alignment", "SimilarityOptions")(method="ransac", random_seed=7)
    source, destination = space("reference"), space("observed")
    before = np.random.get_state()
    result = estimate(
        np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]]),
        np.array([[3.0, 4.0], [13.0, 4.0], [3.0, 14.0], [13.0, 14.0]]),
        source=source, destination=destination, options=options,
    )
    after = np.random.get_state()
    assert result.valid
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])


def test_d03_control_correction_maps_observed_points_back_to_reference(api, space):
    module = api("alignment")
    initial = api("computation_types", "make_mapping")(
        np.eye(3), source=space("source"), destination=space("observed-patch"),
        kind="similarity",
    )
    result = api("alignment", "correct_alignment")(
        initial, np.array([[10.0, 10.0], [30.0, 10.0], [10.0, 30.0]]),
        np.array([[13.0, 12.0], [33.0, 12.0], [13.0, 32.0]]),
        reference_patch=space("reference-patch"), control_ids=("a", "b", "c"),
        options=module.AlignmentOptions(solver=module.SimilarityOptions()),
    )
    assert result.valid
    np.testing.assert_allclose(result.value.matrix @ [13.0, 12.0, 1.0],
                               [10.0, 10.0, 1.0], atol=1e-9)
    assert result.value.destination == space("reference-patch")


@pytest.mark.parametrize("policy,valid", [("fail", False), ("rigid_fallback", True)])
def test_insufficient_controls_follow_explicit_fallback_policy(api, space, policy, valid):
    module = api("alignment")
    initial = api("computation_types", "make_mapping")(
        np.eye(3), source=space("source"), destination=space("observed-patch"),
        kind="similarity",
    )
    result = api("alignment", "correct_alignment")(
        initial, np.array([[10.0, 10.0], [30.0, 10.0]]),
        np.array([[13.0, 12.0], [33.0, 12.0]]), reference_patch=space("reference-patch"),
        control_ids=("a", "b"), options=module.AlignmentOptions(
            solver=module.SimilarityOptions(), minimum_controls=3, failure_policy=policy,
        ),
    )
    assert result.valid is valid
    if valid:
        assert result.diagnostics["fallback_used"] is True
        np.testing.assert_allclose(result.value.matrix, np.eye(3))
    else:
        assert result.value is None
