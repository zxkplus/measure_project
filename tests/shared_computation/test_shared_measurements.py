"""Profiles, complete pairs and image metrology are tested against synthetic features."""

import numpy as np
import pytest


def test_profile_reports_source_xy_coordinates_without_half_pixel_shift(api, space, ramp_image):
    probe = api("measure1D", "ProbeSpec")(
        center=(50.0, 40.0), angle_rad=0.0, search_half_length=20.0, average_half_width=4.0,
    )
    result = api("measure1D", "sample_profile")(ramp_image, probe, space=space(), sigma=0.0)
    assert result.valid
    np.testing.assert_allclose(result.value.gray, np.arange(30.0, 70.0))
    np.testing.assert_allclose(result.value.smoothed, np.arange(30.0, 70.0))
    np.testing.assert_allclose(result.value.gradient, np.ones(40))
    np.testing.assert_allclose(result.value.sample_points_xy[[0, -1]], [[30.0, 40.0], [69.0, 40.0]])
    assert result.value.roi_pixels.shape == (8, 40)


def test_noninteger_probe_half_width_is_not_replaced_with_integer_roi_half_height(api, space):
    probe = api("measure1D", "ProbeSpec")(
        center=(50.0, 40.0), angle_rad=0.0, search_half_length=20.0, average_half_width=3.7,
    )
    result = api("measure1D", "sample_profile")(
        np.zeros((80, 100), np.uint8), probe, space=space(), sigma=0.0,
    )
    assert result.valid and result.value.roi_pixels.shape == (7, 40)
    np.testing.assert_allclose(result.value.sample_points_xy[:, 1], 40.0)


@pytest.mark.parametrize("selection,expected", [("first", [9.5]), ("last", [69.5]),
                                                ("all", [9.5, 29.5, 49.5, 69.5])])
def test_positive_edge_selection_follows_probe_direction(
    api, space, band_image, selection, expected,
):
    probe = api("measure1D", "ProbeSpec")(
        center=(50.0, 40.0), angle_rad=0.0, search_half_length=48.0, average_half_width=4.0,
    )
    options = api("measure1D", "EdgeOptions")(
        sigma=1.0, threshold=30.0, polarity="positive", selection=selection,
    )
    result = api("measure1D", "detect_edges")(band_image, probe, options, space=space())
    assert result.valid
    np.testing.assert_allclose([edge.point[0] for edge in result.value.edges], expected, atol=0.1)
    assert all(edge.amplitude > 0 for edge in result.value.edges)
    assert len(result.value.distances) == len(expected) - 1


def test_fraction_threshold_and_gray_threshold_select_identical_edges(api, space, band_image):
    probe = api("measure1D", "ProbeSpec")(
        center=(50.0, 40.0), angle_rad=0.0, search_half_length=48.0, average_half_width=4.0,
    )
    detect = api("measure1D", "detect_edges")
    options = api("measure1D", "EdgeOptions")
    gray = detect(band_image, probe, options(threshold=25.5), space=space())
    fraction = detect(band_image, probe, options(threshold=0.1, threshold_unit="fraction"),
                      space=space())
    assert gray.valid and fraction.valid
    np.testing.assert_allclose([e.point for e in gray.value.edges],
                               [e.point for e in fraction.value.edges])
    assert fraction.diagnostics["threshold_gray"] == pytest.approx(25.5)


@pytest.mark.parametrize("polarity,centers", [
    ("positive", [14.5, 34.5, 54.5, 74.5]),
    ("negative", [24.5, 44.5, 64.5]),
    ("all", [14.5, 34.5, 54.5, 74.5]),
])
def test_complete_pairs_apply_polarity_after_detecting_both_signs(
    api, space, band_image, polarity, centers,
):
    probe = api("measure1D", "ProbeSpec")(
        center=(50.0, 40.0), angle_rad=0.0, search_half_length=48.0, average_half_width=4.0,
    )
    options = api("measure1D", "PairOptions")(polarity=polarity)
    result = api("measure1D", "detect_edge_pairs")(band_image, probe, options, space=space())
    assert result.valid
    np.testing.assert_allclose([pair.center[0] for pair in result.value.pairs], centers, atol=0.1)
    np.testing.assert_allclose([pair.width for pair in result.value.pairs], 10.0, atol=0.1)


@pytest.mark.parametrize("selection,center", [("first", 14.5), ("last", 74.5)])
def test_pair_selection_does_not_filter_to_a_single_unpairable_edge(
    api, space, band_image, selection, center,
):
    probe = api("measure1D", "ProbeSpec")(
        center=(50.0, 40.0), angle_rad=0.0, search_half_length=48.0, average_half_width=4.0,
    )
    result = api("measure1D", "detect_edge_pairs")(
        band_image, probe, api("measure1D", "PairOptions")(selection=selection), space=space(),
    )
    assert result.valid and len(result.value.pairs) == 1
    assert result.value.pairs[0].center[0] == pytest.approx(center, abs=0.1)
    assert len(result.value.center_distances) == 0


def test_endpoints_mode_preserves_first_and_last_edge_identity(api, space, band_image):
    probe = api("measure1D", "ProbeSpec")(
        center=(50.0, 40.0), angle_rad=0.0, search_half_length=48.0, average_half_width=4.0,
    )
    result = api("measure1D", "detect_edge_pairs")(
        band_image, probe, api("measure1D", "PairOptions")(selection="endpoints"), space=space(),
    )
    pair = result.value.pairs[0]
    assert pair.edge_a.point[0] == pytest.approx(9.5, abs=0.1)
    assert pair.edge_b.point[0] == pytest.approx(79.5, abs=0.1)
    assert pair.width == pytest.approx(70.0, abs=0.1)
    assert result.diagnostics["pairing_mode"] == "endpoints"


def test_blank_image_is_an_invalid_observation_with_a_real_diagnostic_profile(api, space):
    probe = api("measure1D", "ProbeSpec")(
        center=(50.0, 40.0), angle_rad=0.0, search_half_length=20.0, average_half_width=4.0,
    )
    result = api("measure1D", "detect_edges")(
        np.zeros((80, 100), np.uint8), probe, api("measure1D", "EdgeOptions")(), space=space(),
    )
    assert not result.valid and result.value is None and result.reason == "no_edges"
    np.testing.assert_array_equal(result.diagnostics["profile"].gray, np.zeros(40))


def test_rejected_outside_probe_is_not_silently_replicated(api, space):
    probe = api("measure1D", "ProbeSpec")(
        center=(1.0, 40.0), angle_rad=0.0, search_half_length=20.0, average_half_width=4.0,
        border_policy="reject",
    )
    result = api("measure1D", "sample_profile")(
        np.zeros((80, 100), np.uint8), probe, space=space(),
    )
    assert not result.valid and result.reason == "probe_outside_image" and result.value is None


def test_image_line_fit_exposes_actual_probe_depth_and_averaging_width(api, space):
    image = np.zeros((120, 180), np.uint8)
    image[60:, :] = 255
    result = api("measure2D", "probe_and_fit_line")(
        image, api("measure2D", "LineSpec")((30.0, 60.0), (150.0, 60.0), probe_count=7),
        api("measure2D", "ProbeFitOptions")(search_half_length=12, average_half_width=3),
        space=space(),
    )
    assert result.valid
    assert result.value.coefficients == pytest.approx((0.0, 1.0, -59.5), abs=0.15)
    fit = result.diagnostics["fit"]
    assert fit.requested_probe_count == fit.successful_probe_count == fit.sample_count == 7
    for record in result.diagnostics["probes"]:
        assert record.spec.search_half_length == 12
        assert record.spec.average_half_width == 3


def test_image_circle_fit_uses_radial_search_and_reports_selected_probe_identity(api, space):
    yy, xx = np.indices((180, 200))
    image = np.where((xx - 100) ** 2 + (yy - 90) ** 2 <= 40 ** 2, 255, 0).astype(np.uint8)
    result = api("measure2D", "probe_and_fit_circle")(
        image, api("measure2D", "CircleSpec")(
            center=(100.0, 90.0), radius=40.0, probe_count=16, radius_limits=(37.0, 43.0),
        ), api("measure2D", "ProbeFitOptions")(search_half_length=8, average_half_width=2),
        space=space(),
    )
    assert result.valid and result.value.center == pytest.approx((100.0, 90.0), abs=0.5)
    assert result.value.radius == pytest.approx(40.0, abs=0.75)
    fit = result.diagnostics["fit"]
    np.testing.assert_array_equal(fit.sample_probe_indices, np.arange(16))
    assert fit.successful_probe_count == 16


def test_full_circle_does_not_repeat_its_first_probe_at_two_pi(api, space):
    result = api("measure2D", "probe_and_fit_circle")(
        np.zeros((180, 200), np.uint8),
        api("measure2D", "CircleSpec")(center=(100.0, 90.0), radius=40.0, probe_count=4),
        api("measure2D", "ProbeFitOptions")(search_half_length=8, average_half_width=2),
        space=space(),
    )
    records = result.diagnostics["probes"]
    np.testing.assert_allclose([record.spec.center for record in records],
                               [[140.0, 90.0], [100.0, 130.0], [60.0, 90.0], [100.0, 50.0]])


def test_failed_probe_fit_keeps_requested_count_without_fabricating_geometry(api, space):
    result = api("measure2D", "probe_and_fit_line")(
        np.zeros((120, 180), np.uint8),
        api("measure2D", "LineSpec")((30.0, 60.0), (150.0, 60.0), probe_count=7),
        api("measure2D", "ProbeFitOptions")(), space=space(),
    )
    assert not result.valid and result.value is None and result.reason == "insufficient_samples"
    fit = result.diagnostics["fit"]
    assert fit.requested_probe_count == 7 and fit.successful_probe_count == 0
    assert fit.residuals is None and fit.rms_error is None
