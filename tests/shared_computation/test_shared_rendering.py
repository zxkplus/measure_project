"""Rendering consumes public observed geometry and returns independent BGR pixels."""

import numpy as np
import pytest


def test_rasterization_does_not_modify_source_or_decimal_geometry(api, space, observed):
    image = np.zeros((40, 60), np.uint8)
    point = observed((20.25, 12.75))
    primitive = api("viz", "RasterPrimitive")(geometry=point)
    result = api("viz", "rasterize_geometry")(image, (primitive,), image_space=space())
    assert result.shape == (40, 60, 3) and result.dtype == np.uint8
    assert result.any() and not image.any()
    assert point.value == (20.25, 12.75) and not np.shares_memory(image, result)


def test_invisible_or_invalid_primitives_do_not_invent_locations(api, space, observed):
    primitive = api("viz", "RasterPrimitive")
    values = (primitive(geometry=observed((20.0, 12.0)), visible=False),
              primitive(geometry=observed(None, reason="no_edges")))
    result = api("viz", "rasterize_geometry")(
        np.zeros((40, 60), np.uint8), values, image_space=space(),
    )
    assert not result.any()


def test_invisible_geometry_cannot_hide_wrong_display_space(api, space, observed):
    primitive = api("viz", "RasterPrimitive")(
        geometry=observed((2.0, 3.0), frame="physical", unit="mm"), visible=False,
    )
    with pytest.raises(api("computation_types", "ComputationContractError")):
        api("viz", "rasterize_geometry")(
            np.zeros((40, 60), np.uint8), (primitive,), image_space=space(),
        )


def test_profile_rendering_returns_pixels_rather_than_encoded_bytes(api, space, ramp_image):
    probe = api("measure1D", "ProbeSpec")(
        center=(50.0, 40.0), angle_rad=0.0, search_half_length=20.0, average_half_width=4.0,
    )
    profile = api("measure1D", "sample_profile")(
        ramp_image, probe, space=space(), sigma=0.0,
    ).value
    result = api("viz", "render_profile")(profile, threshold_gray=30.0, size_wh=(480, 240))
    assert isinstance(result, np.ndarray) and result.shape == (240, 480, 3)
    assert result.dtype == np.uint8 and np.any(result != 255)


def test_probe_rendering_uses_real_measurement_records(api, space):
    image = np.zeros((120, 180), np.uint8)
    image[60:] = 255
    fitted = api("measure2D", "probe_and_fit_line")(
        image, api("measure2D", "LineSpec")((30.0, 60.0), (150.0, 60.0), probe_count=5),
        api("measure2D", "ProbeFitOptions")(), space=space(),
    )
    original = image.copy()
    result = api("viz", "render_probes")(
        image, fitted.diagnostics["probes"], image_space=space(),
    )
    assert result.shape == (120, 180, 3) and not np.shares_memory(result, image)
    assert np.any(result[:, :, 0] != result[:, :, 1])
    np.testing.assert_array_equal(image, original)
