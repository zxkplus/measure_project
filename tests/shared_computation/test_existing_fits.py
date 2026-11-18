"""The retained fit entry points keep their legacy row/column output fields."""

import numpy as np
import pytest

from measure.geometric_fits import fit_circle_points, fit_line_points


def test_existing_line_fit_keeps_xy_input_and_row_column_endpoints():
    result = fit_line_points(np.array([[0.0, 10.0], [20.0, 10.0]]))
    assert result["point"] == pytest.approx((10.0, 10.0))
    endpoints = sorted([result["start"], result["end"]])
    np.testing.assert_allclose(endpoints, [[10.0, -2.0], [10.0, 22.0]])
    assert result["num_points"] == 2 and result["rms_error"] == pytest.approx(0.0)


def test_existing_circle_fit_keeps_legacy_center_and_real_residuals():
    result = fit_circle_points(np.array([[8.0, 4.0], [3.0, 9.0], [-2.0, 4.0], [3.0, -1.0]]))
    assert result["center"] == pytest.approx((4.0, 3.0))
    assert result["radius"] == pytest.approx(5.0)
    assert result["num_points"] == 4
    np.testing.assert_allclose(result["residuals"], np.zeros(4), atol=1e-12)
    assert result["rms_error"] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("fit,points,reason", [
    (fit_line_points, [[1.0, 2.0], [1.0, 2.0]], "coincident_points"),
    (fit_circle_points, [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], "collinear_points"),
])
def test_existing_fits_keep_their_value_error_failure_contract(fit, points, reason):
    with pytest.raises(ValueError, match=reason):
        fit(np.array(points))
