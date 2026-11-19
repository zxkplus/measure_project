"""Shared values reject contradictory observations and ambiguous mappings."""

import numpy as np
import pytest


@pytest.mark.parametrize("frame,unit", [("", "px"), ("source", "")])
def test_space_requires_explicit_identity(api, frame, unit):
    error = api("computation_types", "ComputationContractError")
    with pytest.raises(error):
        api("computation_types", "Space")(frame=frame, unit=unit)


@pytest.mark.parametrize("valid,value,reason", [
    (True, None, None), (True, (1.0, 2.0), "no_edges"),
    (False, (1.0, 2.0), "no_edges"), (False, None, None),
])
def test_observation_cannot_disagree_with_its_payload(api, space, valid, value, reason):
    error = api("computation_types", "ComputationContractError")
    with pytest.raises(error):
        api("computation_types", "Observation")(
            space=space(), valid=valid, value=value, reason=reason,
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), (1.0, float("nan"))])
def test_valid_observation_requires_finite_numeric_payload(api, observed, value):
    with pytest.raises(api("computation_types", "ComputationContractError")):
        observed(value)


def test_zero_is_a_valid_measurement(observed):
    result = observed(0.0)
    assert result.valid and result.value == 0.0 and result.reason is None


def test_line_normalization_keeps_endpoints_and_xy_equation_consistent(api):
    line = api("computation_types", "LineValue")(
        coefficients=(0.0, -2.0, 20.0), start=(2.0, 10.0), end=(8.0, 10.0),
    )
    assert line.coefficients == pytest.approx((0.0, 1.0, -10.0))
    assert {line.start, line.end} == {(2.0, 10.0), (8.0, 10.0)}


@pytest.mark.parametrize("coefficients", [(0.0, 0.0, 1.0), (float("nan"), 1.0, 0.0)])
def test_degenerate_line_is_a_contract_error(api, coefficients):
    with pytest.raises(api("computation_types", "ComputationContractError")):
        api("computation_types", "LineValue")(coefficients=coefficients)


@pytest.mark.parametrize("radius", [0.0, -1.0, float("inf")])
def test_circle_requires_a_finite_positive_radius(api, radius):
    with pytest.raises(api("computation_types", "ComputationContractError")):
        api("computation_types", "CircleValue")(center=(2.0, 3.0), radius=radius)


def test_mapping_direction_and_inverse_are_not_caller_ambiguities(api, space):
    mapping = api("computation_types", "make_mapping")(
        np.array([[2.0, 0.0, 3.0], [0.0, 2.0, -4.0], [0.0, 0.0, 1.0]]),
        source=space("patch"), destination=space("source"), kind="similarity",
    )
    np.testing.assert_allclose(mapping.matrix @ [1.0, 2.0, 1.0], [5.0, 0.0, 1.0])
    np.testing.assert_allclose(mapping.inverse_matrix @ [5.0, 0.0, 1.0], [1.0, 2.0, 1.0])


@pytest.mark.parametrize("matrix,kind", [
    (np.zeros((3, 3)), "projective"),
    (np.diag([2.0, 3.0, 1.0]), "similarity"),
    (np.diag([-1.0, 1.0, 1.0]), "similarity"),
    (np.full((3, 3), np.nan), "affine"),
    (np.eye(2), "affine"),
])
def test_mapping_rejects_singularity_nonfinite_and_false_model_claims(api, space, matrix, kind):
    with pytest.raises(api("computation_types", "ComputationContractError")):
        api("computation_types", "make_mapping")(
            matrix, source=space("patch"), destination=space("source"), kind=kind,
        )
