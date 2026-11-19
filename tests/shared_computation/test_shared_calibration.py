"""Calibration solves recover independently specified physical camera parameters."""

from dataclasses import replace

import cv2
import numpy as np
import pytest

from measure.pixel_calibration import CheckerboardScaleCalibration


def test_existing_scalar_calibration_preserves_row_column_input_contract():
    corners = [(float(row), float(col)) for row in (10, 20) for col in (10, 20, 30)]
    result = CheckerboardScaleCalibration().compute_scale(corners, square_size_mm=5.0)
    assert result.valid and result.scale_mm_per_px == pytest.approx(0.5)


def test_scalar_scale_validator_accepts_consistent_physical_spacing(api):
    assert api("pixel_calibration", "validate_scalar_scale")(
        0.5, square_size=5.0, spacing_px=10.0,
    ) is None


@pytest.mark.parametrize("scale,square,spacing", [
    (0.0, None, None), (-1.0, None, None), (float("nan"), None, None),
    (0.5, 5.0, None), (0.5, 5.0, 20.0),
])
def test_scalar_scale_validator_rejects_nonphysical_or_inconsistent_parameters(
    api, scale, square, spacing,
):
    with pytest.raises(api("computation_types", "ComputationContractError")):
        api("pixel_calibration", "validate_scalar_scale")(
            scale, square_size=square, spacing_px=spacing,
        )


def test_camera_validator_accepts_a_finite_pinhole_model(api, camera_model):
    assert api("measure_calibration", "validate_camera_model")(camera_model()) is None


def test_camera_validator_rejects_zero_focal_length(api, camera_model):
    camera = camera_model()
    bad_matrix = camera.matrix.copy()
    bad_matrix[0, 0] = 0.0
    with pytest.raises(api("computation_types", "ComputationContractError")):
        api("measure_calibration", "validate_camera_model")(replace(camera, matrix=bad_matrix))


def test_pose_validator_rejects_reflection_instead_of_calling_it_rotation(api):
    with pytest.raises(api("computation_types", "ComputationContractError")):
        pose = api("measure_calibration", "Pose3D")(
            source_frame="board", destination_frame="camera:left", unit="mm",
            rotation=np.diag([-1.0, 1.0, 1.0]), translation=np.array([0.0, 0.0, 100.0]),
        )
        api("measure_calibration", "validate_pose")(pose)


def test_pose_validator_accepts_a_proper_rotation(api):
    pose = api("measure_calibration", "Pose3D")(
        source_frame="board", destination_frame="camera:left", unit="mm",
        rotation=np.eye(3), translation=np.array([0.0, 0.0, 100.0]),
    )
    assert api("measure_calibration", "validate_pose")(pose) is None


def test_dot_grid_detection_preserves_ordered_physical_correspondences(api, space):
    image = np.full((180, 240), 255, np.uint8)
    for row in range(5):
        for col in range(6):
            cv2.circle(image, (60 + 20 * col, 40 + 20 * row), 5, 0, -1)
    result = api("measure_calibration", "detect_dot_grid")(
        image, api("measure_calibration", "BoardSpec")(
            columns=6, rows=5, spacing=4.0, unit="mm", symmetric=True,
        ), api("measure_calibration", "DotDetectionOptions")(), image_space=space(),
        object_frame="board", view_id="capture-1",
    )
    assert result.valid and result.value.image_points_xy.shape == (30, 2)
    np.testing.assert_allclose(result.value.object_points_xyz[[0, 1, 6]],
                               [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    assert result.value.ordering == "row_major"


def test_blank_dot_grid_is_a_domain_failure(api, space):
    result = api("measure_calibration", "detect_dot_grid")(
        np.full((180, 240), 255, np.uint8), api("measure_calibration", "BoardSpec")(
            columns=6, rows=5, spacing=4.0, unit="mm",
        ), api("measure_calibration", "DotDetectionOptions")(),
        image_space=space(), object_frame="board", view_id="blank",
    )
    assert not result.valid and result.value is None and result.reason == "grid_not_found"


def test_intrinsics_recover_known_focal_lengths_from_multiple_board_poses(
    api, camera_model, board_views,
):
    camera = camera_model()
    result = api("measure_calibration", "fit_camera_intrinsics")(
        board_views(camera), camera_id=camera.camera_id, image_space=camera.image_space,
        image_size_wh=camera.image_size_wh,
        options=api("measure_calibration", "IntrinsicOptions")(),
    )
    assert result.valid
    np.testing.assert_allclose(result.value.camera.matrix,
                               [[320.0, 0.0, 160.0], [0.0, 300.0, 120.0], [0.0, 0.0, 1.0]],
                               atol=0.05)
    assert result.value.rms_error_px < 0.001 and len(result.value.poses) == 5
    assert result.value.accepted_view_ids == tuple("capture-" + str(i) for i in range(5))


def test_intrinsics_do_not_count_copies_as_independent_views(api, camera_model, board_views):
    camera = camera_model()
    view = board_views(camera)[0]
    result = api("measure_calibration", "fit_camera_intrinsics")(
        (view, replace(view, view_id="duplicate-1"), replace(view, view_id="duplicate-2")),
        camera_id=camera.camera_id, image_space=camera.image_space,
        image_size_wh=camera.image_size_wh,
        options=api("measure_calibration", "IntrinsicOptions")(),
    )
    assert not result.valid and result.value is None and result.reason == "insufficient_views"


def test_stereo_recovers_a_to_b_baseline_without_mutating_supplied_intrinsics(
    api, camera_model, board_views,
):
    left, right = camera_model("left"), camera_model("right")
    before_left, before_right = left.matrix.copy(), right.matrix.copy()
    pairs = tuple(api("measure_calibration", "StereoObservationPair")(
        capture_id=a.view_id, a=a, b=b,
    ) for a, b in zip(board_views(left), board_views(right, camera_shift_x=-10.0)))
    result = api("measure_calibration", "fit_stereo_rig")(
        pairs, left, right, unit="mm", options=api("measure_calibration", "StereoOptions")(),
    )
    assert result.valid and result.value.baseline == pytest.approx(10.0, abs=0.005)
    np.testing.assert_allclose(result.value.a_to_b.translation, [-10.0, 0.0, 0.0], atol=0.005)
    np.testing.assert_allclose(result.value.a_to_b.rotation, np.eye(3), atol=1e-4)
    np.testing.assert_allclose(result.value.essential,
                               [[0.0, 0.0, 0.0], [0.0, 0.0, 10.0], [0.0, -10.0, 0.0]],
                               atol=0.005)
    np.testing.assert_array_equal(left.matrix, before_left)
    np.testing.assert_array_equal(right.matrix, before_right)


def test_stereo_requires_three_distinct_pairs(api, camera_model, board_views):
    left, right = camera_model("left"), camera_model("right")
    pair = api("measure_calibration", "StereoObservationPair")(
        capture_id="capture-0", a=board_views(left)[0],
        b=board_views(right, camera_shift_x=-10.0)[0],
    )
    result = api("measure_calibration", "fit_stereo_rig")(
        (pair,), left, right, unit="mm", options=api("measure_calibration", "StereoOptions")(),
    )
    assert not result.valid and result.value is None


def test_asymmetric_board_points_stagger_the_odd_rows(api):
    board = api("measure_calibration", "BoardSpec")(
        columns=3, rows=2, spacing=2.0, symmetric=False)
    points = api("measure_calibration", "board_object_points")(board)
    # OpenCV's asymmetric grid offsets every odd row by one spacing, so the
    # physical x of a column is 2 * column + row % 2.
    np.testing.assert_allclose(points[:, 0], [0.0, 4.0, 8.0, 2.0, 6.0, 10.0])
    np.testing.assert_allclose(points[:, 1], [0.0, 0.0, 0.0, 2.0, 2.0, 2.0])


def test_symmetric_board_points_share_one_column_origin(api):
    board = api("measure_calibration", "BoardSpec")(
        columns=3, rows=2, spacing=2.0, symmetric=True)
    points = api("measure_calibration", "board_object_points")(board)
    np.testing.assert_allclose(points[:, 0], [0.0, 2.0, 4.0, 0.0, 2.0, 4.0])


def test_light_dots_on_a_dark_board_are_detected(api, space):
    image = np.zeros((180, 240), np.uint8)
    for row in range(4):
        for col in range(5):
            cv2.circle(image, (60 + 30 * col, 40 + 30 * row), 7, 255, -1)
    result = api("measure_calibration", "detect_dot_grid")(
        image, api("measure_calibration", "BoardSpec")(columns=5, rows=4, spacing=3.0),
        api("measure_calibration", "DotDetectionOptions")(color="light", refine=True),
        image_space=space(), object_frame="board", view_id="light",
    )
    assert result.valid, result.reason
    assert result.value.image_points_xy.shape == (20, 2)
    assert result.diagnostics["refined"] is True
    assert result.value.image_points_xy[6] == pytest.approx([60 + 30, 40 + 30], abs=0.2)


def test_undistortion_returns_a_complete_inverse_map(api, camera_model):
    camera = camera_model()
    distorted = api("measure_calibration", "CameraModel")(
        camera_id="wide", image_space=camera.image_space,
        image_size_wh=camera.image_size_wh,
        matrix=np.array([[260.0, 0.0, 160.0], [0.0, 250.0, 120.0], [0.0, 0.0, 1.0]]),
        distortion=np.array([-0.28, 0.09, 0.0, 0.0, 0.0]),
    )
    rng = np.random.RandomState(5)
    image = (rng.rand(240, 320, 3) * 255).astype(np.uint8)
    result = api("measure_calibration", "undistort_image")(image, distorted, alpha=0.0)
    assert result.pixels.shape == image.shape and result.pixels.dtype == image.dtype
    assert not np.array_equal(result.pixels, image)
    assert result.camera_matrix.shape == (3, 3)
    assert result.inverse_map_x.shape == (240, 320)
    assert result.inverse_map_y.shape == (240, 320)
    assert 0 <= result.valid_roi_xywh[0] < 320 and 0 <= result.valid_roi_xywh[1] < 240


def test_undistortion_of_a_distortion_free_camera_is_the_identity(api, camera_model):
    camera = camera_model()
    image = np.zeros((240, 320), np.uint8)
    result = api("measure_calibration", "undistort_image")(image, camera, alpha=0.0)
    # No distortion means the inverse map is the identity grid, so a wrong
    # direction or a transposed map cannot pass.
    np.testing.assert_allclose(result.inverse_map_x,
                               np.tile(np.arange(320.0), (240, 1)), atol=1e-3)
    np.testing.assert_allclose(result.inverse_map_y,
                               np.tile(np.arange(240.0)[:, None], (1, 320)), atol=1e-3)
    np.testing.assert_allclose(result.camera_matrix, camera.matrix, atol=1e-9)


def test_undistortion_rejects_a_size_mismatch(api, camera_model):
    camera = camera_model()
    with pytest.raises(Exception) as error:
        api("measure_calibration", "undistort_image")(
            np.zeros((100, 100), np.uint8), camera)
    assert "image size" in str(error.value)
