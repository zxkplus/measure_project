"""Product acceptance and measured quality remain independent from observation validity."""

import numpy as np
import pytest


@pytest.mark.parametrize("value,status", [(9.0, "fail"), (10.0, "pass"),
                                          (20.0, "pass"), (21.0, "fail")])
def test_inclusive_range_keeps_valid_measurements_even_when_product_fails(
    api, observed, value, status,
):
    result = api("evaluation", "evaluate_range")(
        observed(value), api("evaluation", "LimitsSpec")(unit="px", minimum=10.0, maximum=20.0),
    )
    assert result.measurement.valid and result.measurement.value == value
    assert result.evaluation_status == status


@pytest.mark.parametrize("policy,status", [("not_evaluated", "not_evaluated"),
                                          ("fail", "fail"), ("ignore", "not_evaluated")])
def test_invalid_range_never_substitutes_zero(api, observed, policy, status):
    result = api("evaluation", "evaluate_range")(
        observed(None, reason="no_edges"),
        api("evaluation", "LimitsSpec")(unit="px", maximum=1.0, invalid_policy=policy),
    )
    assert not result.measurement.valid and result.measurement.value is None
    assert result.evaluation_status == status and result.valid_count == 0


def test_error_policy_raises_a_numeric_contract_error(api, observed):
    with pytest.raises(api("computation_types", "ComputationContractError")) as caught:
        api("evaluation", "evaluate_range")(
            observed(None, reason="no_edges"),
            api("evaluation", "LimitsSpec")(unit="px", maximum=1.0, invalid_policy="error"),
        )
    assert caught.value.code == "evaluation.invalid_input"


def test_diameter_variation_compares_calibrated_lengths_across_frames(api, observed):
    result = api("evaluation", "diameter_variation")(
        (observed(12.0, frame="left", unit="mm"), observed(15.0, frame="right", unit="mm")),
        api("evaluation", "LimitsSpec")(unit="mm", maximum=3.0),
    )
    assert result.measurement.valid and result.measurement.value == pytest.approx(3.0)
    assert result.measurement.space.frame == "aggregate" and result.evaluation_status == "pass"


def test_ignore_policy_excludes_only_invalid_values_and_reports_counts(api, observed):
    result = api("evaluation", "diameter_variation")(
        (observed(10.0), observed(None, reason="no_edges"), observed(16.0)),
        api("evaluation", "LimitsSpec")(unit="px", maximum=5.0, invalid_policy="ignore"),
    )
    assert result.measurement.valid and result.measurement.value == pytest.approx(6.0)
    assert result.evaluation_status == "fail"
    assert result.input_count == 3 and result.valid_count == 2


@pytest.mark.parametrize("policy,status", [("not_evaluated", "not_evaluated"), ("fail", "fail")])
def test_aggregation_does_not_silently_ignore_invalid_values(api, observed, policy, status):
    result = api("evaluation", "diameter_variation")(
        (observed(10.0), observed(None, reason="no_edges"), observed(16.0)),
        api("evaluation", "LimitsSpec")(unit="px", maximum=9.0, invalid_policy=policy),
    )
    assert not result.measurement.valid and result.measurement.value is None
    assert result.evaluation_status == status


def test_ignoring_invalid_values_still_requires_two_diameters(api, observed):
    result = api("evaluation", "diameter_variation")(
        (observed(10.0), observed(None, reason="no_edges")),
        api("evaluation", "LimitsSpec")(unit="px", maximum=9.0, invalid_policy="ignore"),
    )
    assert not result.measurement.valid and result.evaluation_status == "not_evaluated"


def test_centerline_deviation_means_horizontal_range_rather_than_fitted_axis_error(
    api, observed, space,
):
    result = api("evaluation", "centerline_deviation")(
        (observed((1.0, 0.0)), observed((4.0, 100.0)), observed((2.0, 200.0))),
        api("evaluation", "LimitsSpec")(unit="px", maximum=3.0), space=space(),
    )
    assert result.measurement.valid and result.measurement.value == pytest.approx(3.0)
    assert result.evaluation_status == "pass"


def test_centerline_deviation_cannot_mix_camera_frames(api, observed, space):
    with pytest.raises(api("computation_types", "ComputationContractError")):
        api("evaluation", "centerline_deviation")(
            (observed((1.0, 0.0)), observed((4.0, 100.0), frame="other-camera")),
            api("evaluation", "LimitsSpec")(unit="px", maximum=3.0), space=space(),
        )


@pytest.mark.parametrize("function", ["diameter_variation", "centerline_deviation"])
def test_empty_aggregation_is_not_a_zero_measurement(api, space, function):
    kwargs = {"space": space()} if function == "centerline_deviation" else {}
    result = api("evaluation", function)(
        (), api("evaluation", "LimitsSpec")(unit="px", maximum=3.0), **kwargs,
    )
    assert not result.measurement.valid and result.measurement.value is None
    assert result.input_count == result.valid_count == 0


def test_limits_units_are_checked_before_computing_a_verdict(api, observed):
    with pytest.raises(api("computation_types", "ComputationContractError")):
        api("evaluation", "evaluate_range")(
            observed(10.0, unit="px"), api("evaluation", "LimitsSpec")(unit="mm", maximum=3.0),
        )


@pytest.mark.parametrize("minimum,maximum", [(None, None), (3.0, 2.0), (None, float("nan"))])
def test_limits_must_define_a_finite_nonempty_interval(api, minimum, maximum):
    with pytest.raises(api("computation_types", "ComputationContractError")):
        api("evaluation", "LimitsSpec")(unit="px", minimum=minimum, maximum=maximum)


def test_d09_unknown_probe_count_remains_unknown_in_quality(api, space):
    fitted = api("geometric_fits", "fit_line_observation")(
        np.array([[0.0, 9.0], [0.0, 11.0], [20.0, 9.0], [20.0, 11.0]]), space=space(),
    )
    quality = api("evaluation", "summarize_quality")(fit=fitted.diagnostics["fit"])
    assert quality.sample_count == 4 and quality.rms_error == pytest.approx(1.0)
    assert quality.requested_probe_count is None and quality.coverage_ratio is None


def test_d09_coverage_counts_successful_probes_rather_than_multiple_edges(api, space):
    edge_type = api("measure1D", "EdgeValue")
    probe_type = api("measure1D", "ProbeSpec")
    record_type = api("measure2D", "ProbeRecord")
    edges = (edge_type(point=(1.0, 0.0), amplitude=10.0, profile_position=1.0),
             edge_type(point=(2.0, 0.0), amplitude=-20.0, profile_position=2.0))
    records = (
        record_type(index=0, spec=probe_type((0.0, 0.0), 0.0, 5.0, 2.0), status="selected",
                    candidates=edges, selected=edges, reason=None),
        record_type(index=1, spec=probe_type((0.0, 2.0), 0.0, 5.0, 2.0), status="no_edge",
                    candidates=(), selected=(), reason="no_edges"),
    )
    quality = api("evaluation", "summarize_quality")(
        probes=records, amplitudes=np.array([10.0, -20.0]),
    )
    assert quality.requested_probe_count == 2 and quality.successful_probe_count == 1
    assert quality.sample_count == 2 and quality.coverage_ratio == pytest.approx(0.5)
    assert quality.amplitude_mean == pytest.approx(-5.0)
    assert quality.amplitude_abs_min == pytest.approx(10.0)


def test_quality_with_no_measurement_information_does_not_claim_zero_error(api):
    result = api("evaluation", "summarize_quality")()
    assert result.rms_error is None and result.coverage_ratio is None
    assert result.sample_count is None
