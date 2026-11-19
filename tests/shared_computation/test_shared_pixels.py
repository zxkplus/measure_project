"""Pixel operations preserve channel rules, continuous crop origins and input ownership."""

import numpy as np
import pytest


def test_raw_grayscale_uses_bgr_luma_without_changing_input(api):
    image = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], np.uint8)
    original = image.copy()
    result = api("preprocessing", "preprocess_pixels")(
        image, api("preprocessing", "RawSpec")(channel_mode="gray"),
    )
    assert result.dtype == np.float32 and result.shape == (1, 3)
    np.testing.assert_array_equal(result, [[29.0, 150.0, 76.0]])
    np.testing.assert_array_equal(image, original)


def test_raw_preserve_keeps_color_and_returns_owned_float_pixels(api):
    image = np.array([[[10, 20, 30]]], np.uint8)
    result = api("preprocessing", "preprocess_pixels")(
        image, api("preprocessing", "RawSpec")(channel_mode="preserve"),
    )
    assert result.dtype == np.float32 and result.shape == (1, 1, 3)
    np.testing.assert_array_equal(result, [[[10.0, 20.0, 30.0]]])
    assert not np.shares_memory(result, image)


@pytest.mark.parametrize("mode,expected", [
    ("binary", [0, 0, 255]), ("binary_inv", [255, 255, 0]),
    ("truncate", [50, 100, 100]), ("tozero", [0, 0, 150]),
    ("tozero_inv", [50, 100, 0]),
])
def test_threshold_modes_include_the_threshold_boundary(api, mode, expected):
    result = api("preprocessing", "preprocess_pixels")(
        np.array([[50, 100, 150]], np.uint8),
        api("preprocessing", "ThresholdSpec")(threshold=100, mode=mode),
    )
    assert result.dtype == np.uint8
    np.testing.assert_array_equal(result, [expected])


def test_sobel_keeps_gradient_above_255_in_float32(api):
    image = np.zeros((9, 9), np.uint8)
    image[:, 4:] = 255
    result = api("preprocessing", "preprocess_pixels")(
        image, api("preprocessing", "SobelSpec")(kernel_size=3),
    )
    assert result.dtype == np.float32 and result[4, 3] == pytest.approx(1020.0)


def test_canny_returns_binary_uint8_pixels(api):
    image = np.zeros((20, 20), np.uint8)
    image[:, 10:] = 255
    result = api("preprocessing", "preprocess_pixels")(image, api("preprocessing", "CannySpec")())
    assert result.dtype == np.uint8 and set(np.unique(result)) == {0, 255}


def test_clahe_preserves_dimensions_without_modifying_source(api):
    image = np.arange(256, dtype=np.uint8).reshape(16, 16)
    original = image.copy()
    result = api("preprocessing", "preprocess_pixels")(image, api("preprocessing", "ClaheSpec")())
    assert result.shape == (16, 16) and result.dtype == np.float32
    assert np.isfinite(result).all() and not np.shares_memory(result, image)
    np.testing.assert_array_equal(image, original)


def test_crop_uses_continuous_size_over_two_origin(api, space, ramp_image):
    result = api("rectify", "crop_region")(
        ramp_image, api("computation_types", "RotatedBox")((50.0, 40.0), (10.0, 8.0), 0.0),
        source=space(), destination=space("patch"), options=api("rectify", "ResampleOptions")(),
    )
    assert result.pixels.shape == (8, 10)
    np.testing.assert_allclose(result.mapping.matrix @ [50.0, 40.0, 1.0], [5.0, 4.0, 1.0])
    np.testing.assert_allclose(result.mapping.inverse_matrix @ [0.0, 0.0, 1.0],
                               [45.0, 36.0, 1.0])
    np.testing.assert_array_equal(result.pixels[0], np.arange(45, 55))


def test_replicated_border_pixels_do_not_become_real_valid_pixels(api, space):
    mapping = api("computation_types", "make_mapping")(
        np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        source=space(), destination=space("patch"), kind="similarity",
    )
    result = api("rectify", "warp_mapped")(
        np.full((4, 4), 77, np.uint8), mapping, (6, 4),
        options=api("rectify", "ResampleOptions")(border_mode="replicate"),
    )
    np.testing.assert_array_equal(result.pixels, np.full((4, 6), 77))
    np.testing.assert_array_equal(result.valid_mask[0], [False, False, True, True, True, True])


def test_legacy_roi_conversion_swaps_row_col_once_and_converts_degrees(api):
    result = api("rectify", "legacy_roi_to_box")(
        row=12.0, col=30.0, width=20.0, height=10.0, angle_deg=90.0,
    )
    assert result.center == (30.0, 12.0) and result.size_wh == (20.0, 10.0)
    assert result.angle_rad == pytest.approx(np.pi / 2)
