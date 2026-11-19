"""Known geometric truths constrain fits and composition, including D01 and D10."""

import numpy as np
import pytest


def test_d01_fitted_horizontal_line_is_three_pixels_from_point(api, space, observed):
    line = api("geometric_fits", "fit_line_observation")(
        np.array([[0.0, 10.0], [20.0, 10.0]]), space=space(),
    )
    result = api("geometry_ops", "point_line_distance")(observed((5.0, 13.0)), line)
    assert result.valid and result.value == pytest.approx(3.0)


def test_two_point_line_uses_xy_coefficients_and_true_endpoints(api, observed):
    result = api("geometry_ops", "line_through_points")(
        observed((2.0, 10.0)), observed((8.0, 10.0)),
    )
    assert result.valid
    assert result.value.coefficients == pytest.approx((0.0, 1.0, -10.0))
    assert {result.value.start, result.value.end} == {(2.0, 10.0), (8.0, 10.0)}


def test_coincident_points_do_not_produce_a_zero_line(api, observed):
    result = api("geometry_ops", "line_through_points")(observed((2.0, 7.0)),
                                                               observed((2.0, 7.0)))
    assert not result.valid and result.value is None and result.reason == "coincident_points"


def test_point_distance_uses_the_three_four_five_triangle(api, observed):
    result = api("geometry_ops", "point_distance")(observed((2.0, 7.0)), observed((5.0, 11.0)))
    assert result.valid and result.value == pytest.approx(5.0)


@pytest.mark.parametrize("other_frame,other_unit", [("other-camera", "px"),
                                                   ("camera:left/source", "mm")])
def test_even_invalid_points_cannot_hide_a_space_mismatch(
    api, observed, other_frame, other_unit,
):
    with pytest.raises(api("computation_types", "ComputationContractError")):
        api("geometry_ops", "point_distance")(
            observed(None, reason="no_edges"),
            observed(None, frame=other_frame, unit=other_unit, reason="no_edges"),
        )


def test_invalid_input_never_becomes_a_measured_zero_distance(api, observed):
    result = api("geometry_ops", "point_distance")(observed((2.0, 7.0)),
                                                     observed(None, reason="no_edges"))
    assert not result.valid and result.value is None and result.reason == "invalid_input"


def test_circle_center_conversion_preserves_the_physical_space(api, observed):
    circle = observed(api("computation_types", "CircleValue")((2.0, 7.0), 5.0), unit="mm")
    result = api("geometry_ops", "circle_center")(circle)
    assert result.valid and result.value == (2.0, 7.0)
    assert result.space == circle.space


def test_circle_center_conversion_preserves_failure_reason(api, observed):
    result = api("geometry_ops", "circle_center")(observed(None, reason="insufficient_samples"))
    assert not result.valid and result.value is None and result.reason == "insufficient_samples"


@pytest.mark.parametrize("point,expected", [((3.0, 4.0), 0.0), ((0.0, 0.0), 5.0),
                                          ((6.0, 8.0), 5.0)])
def test_point_circle_distance_means_distance_to_circumference(api, observed, point, expected):
    circle = observed(api("computation_types", "CircleValue")((0.0, 0.0), 5.0))
    result = api("geometry_ops", "point_circle_distance")(observed(point), circle)
    assert result.valid and result.value == pytest.approx(expected)


@pytest.mark.parametrize("coefficients,expected", [((1.0, 0.0, -2.0), np.pi / 2),
                                                  ((0.0, 1.0, -8.0), 0.0)])
def test_line_angle_is_smaller_unsigned_angle_in_radians(api, observed, coefficients, expected):
    line_type = api("computation_types", "LineValue")
    result = api("geometry_ops", "line_angle")(
        observed(line_type((0.0, 1.0, -10.0))), observed(line_type(coefficients)),
    )
    assert result.valid and result.value == pytest.approx(expected)
    assert result.space.unit == "rad"


def test_d10_sampled_diameter_spread_is_not_an_ellipse_axis(api, observed):
    circle = observed(api("computation_types", "CircleValue")((0.0, 0.0), 5.0))
    result = api("geometry_ops", "sampled_diameter_spread")(
        circle, observed(np.array([[3.0, 0.0], [0.0, 5.0], [-4.0, 0.0]])),
    )
    assert result.valid and result.value == pytest.approx(4.0)


def test_spread_with_one_sample_is_unknown_rather_than_zero(api, observed):
    circle = observed(api("computation_types", "CircleValue")((0.0, 0.0), 5.0))
    result = api("geometry_ops", "sampled_diameter_spread")(
        circle, observed(np.array([[3.0, 0.0]])),
    )
    assert not result.valid and result.value is None


def test_line_fit_exposes_actual_signed_residuals(api, space):
    result = api("geometric_fits", "fit_line_observation")(
        np.array([[0.0, 9.0], [0.0, 11.0], [20.0, 9.0], [20.0, 11.0]]), space=space(),
    )
    fit = result.diagnostics["fit"]
    assert result.valid and result.value.coefficients == pytest.approx((0.0, 1.0, -10.0))
    np.testing.assert_allclose(fit.residuals, [-1.0, 1.0, -1.0, 1.0])
    assert fit.rms_error == pytest.approx(1.0)
    assert fit.requested_probe_count is None and fit.successful_probe_count is None


def test_circle_fit_recovers_known_center_radius_and_zero_residual(api, space):
    result = api("geometric_fits", "fit_circle_observation")(
        np.array([[8.0, 4.0], [3.0, 9.0], [-2.0, 4.0], [3.0, -1.0]]), space=space(),
    )
    assert result.valid
    assert result.value.center == pytest.approx((3.0, 4.0))
    assert result.value.radius == pytest.approx(5.0)
    assert result.diagnostics["fit"].rms_error == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("name,points,reason", [
    ("fit_line_observation", [[1.0, 1.0]], "insufficient_samples"),
    ("fit_line_observation", [[1.0, 1.0], [1.0, 1.0]], "coincident_points"),
    ("fit_circle_observation", [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], "collinear_points"),
])
def test_degenerate_sample_sets_are_domain_failures(api, space, name, points, reason):
    result = api("geometric_fits", name)(np.array(points), space=space())
    assert not result.valid and result.value is None and result.reason == reason


def test_nonfinite_fit_samples_are_contract_errors(api, space):
    with pytest.raises(api("computation_types", "ComputationContractError")):
        api("geometric_fits", "fit_line_observation")(
            np.array([[0.0, 1.0], [float("nan"), 2.0]]), space=space(),
        )
