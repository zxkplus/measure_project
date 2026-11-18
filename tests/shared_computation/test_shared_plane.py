"""Plane geometry is checked using known poses, black valid pixels and literal projections."""

import numpy as np
import pytest


def plane_view(api, space, camera):
    pose = api("measure_calibration", "Pose3D")(
        source_frame="board", destination_frame="camera:" + camera.camera_id, unit="mm",
        rotation=np.eye(3), translation=np.array([0.0, 0.0, 100.0]),
    )
    return api("plane_geometry", "build_plane_view")(
        camera, pose, plane_space=space("board", "mm"),
    )


def test_plane_pose_recovers_known_board_to_camera_translation(api, camera_model, board_views):
    camera = camera_model()
    result = api("plane_geometry", "solve_plane_pose")(board_views(camera)[0], camera)
    assert result.valid
    np.testing.assert_allclose(result.value.translation, [-10.0, -8.0, 100.0], atol=1e-4)
    np.testing.assert_allclose(result.value.rotation, np.eye(3), atol=1e-5)
    assert result.diagnostics["rms_error_px"] < 1e-4


def test_plane_mapping_uses_corrected_pixels_and_declared_pose(api, space, camera_model):
    view = plane_view(api, space, camera_model())
    point = view.plane_to_corrected.matrix @ [0.0, 0.0, 1.0]
    np.testing.assert_allclose(point[:2] / point[2], [160.0, 120.0])
    point = view.corrected_to_plane.matrix @ [192.0, 150.0, 1.0]
    np.testing.assert_allclose(point[:2] / point[2], [10.0, 10.0])


def test_plane_canvas_includes_both_boundary_pixel_centers(api, space, camera_model):
    view = plane_view(api, space, camera_model())
    result = api("plane_geometry", "build_plane_canvas")(
        (view,), pixel_size=0.5, canvas_space=space("canvas"),
    )
    assert result.valid and result.value.size_wh == (201, 161)
    np.testing.assert_allclose(result.value.bounds_xyxy,
                               [-50.0, -40.0, 49.6875, 39.666666666666664])


def test_projection_returns_known_pixel_positions_without_inventing_depth(api, camera_model):
    result = api("plane_geometry", "project_camera_points")(
        np.array([[0.0, 0.0, 100.0], [10.0, 10.0, 100.0]]), camera_model(),
        source_frame="camera:left", unit="mm",
    )
    assert result.valid
    np.testing.assert_allclose(result.value.points_xy, [[160.0, 120.0], [192.0, 150.0]])
    np.testing.assert_array_equal(result.value.inside_image, [True, True])


def test_projection_marks_outside_pixels_without_invalidating_known_geometry(api, camera_model):
    result = api("plane_geometry", "project_camera_points")(
        np.array([[100.0, 0.0, 100.0]]), camera_model(), source_frame="camera:left", unit="mm",
    )
    assert result.valid
    np.testing.assert_allclose(result.value.points_xy, [[480.0, 120.0]])
    np.testing.assert_array_equal(result.value.inside_image, [False])


def test_projection_rejects_entire_set_if_one_point_is_behind_camera(api, camera_model):
    result = api("plane_geometry", "project_camera_points")(
        np.array([[0.0, 0.0, 100.0], [0.0, 0.0, -10.0]]), camera_model(),
        source_frame="camera:left", unit="mm",
    )
    assert not result.valid and result.value is None and result.reason == "behind_camera"


def test_projection_converts_explicit_physical_units_once(api, camera_model):
    result = api("plane_geometry", "project_camera_points")(
        np.array([[1.0, 1.0, 10.0]]), camera_model(), source_frame="camera:left", unit="cm",
    )
    assert result.valid
    np.testing.assert_allclose(result.value.points_xy, [[192.0, 150.0]])


def test_object_frame_projection_requires_an_explicit_camera_pose(api, camera_model):
    with pytest.raises(api("computation_types", "ComputationContractError")):
        api("plane_geometry", "project_camera_points")(
            np.array([[1.0, 1.0, 10.0]]), camera_model(), source_frame="workpiece", unit="mm",
        )


@pytest.mark.parametrize("blend,expected", [("average", 50), ("first", 0), ("last", 100)])
def test_stitching_treats_real_black_pixels_as_valid_coverage(
    api, space, camera_model, blend, expected,
):
    views = (plane_view(api, space, camera_model("left")),
             plane_view(api, space, camera_model("right")))
    canvas = api("plane_geometry", "build_plane_canvas")(
        views, pixel_size=1.0, canvas_space=space("canvas"),
    ).value
    first, second = np.zeros((240, 320), np.uint8), np.full((240, 320), 100, np.uint8)
    result = api("plane_geometry", "stitch_plane_views")(
        (first, second), views, canvas, options=api("plane_geometry", "StitchOptions")(blend=blend),
    )
    assert result.valid and result.value.pixels[40, 50] == expected
    assert result.value.coverage[40, 50] == 2
    assert result.value.valid_masks[0][40, 50]
    assert not np.shares_memory(result.value.pixels, first)


def test_rebasing_plane_moves_the_geometry_not_just_origin_metadata(api, space, camera_model):
    view = plane_view(api, space, camera_model())
    result = api("plane_geometry", "rebase_plane")(
        (view,), (10.0, 20.0), destination_plane=space("new-board", "mm"),
    )[0]
    np.testing.assert_allclose(result.board_to_camera.translation, [10.0, 20.0, 100.0])
    projected = result.plane_to_corrected.matrix @ [0.0, 0.0, 1.0]
    np.testing.assert_allclose(projected[:2] / projected[2], [192.0, 180.0])
    np.testing.assert_array_equal(view.board_to_camera.translation, [0.0, 0.0, 100.0])
