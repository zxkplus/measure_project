"""Tests for the stateless gantry pick matrix API."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from measure import GantryPick, PickError
from measure.pick_api import offset_homogeneous_matrix
from measure.robot_calibration import CalibrationError

WIDTH = 640
HEIGHT = 480
IMAGE_CENTRE = np.array([(WIDTH - 1) * 0.5, (HEIGHT - 1) * 0.5])
ROBOT_PER_PIXEL = np.array([[0.05, 0.0], [0.0, 0.05]])
ROBOT_ORIGIN = np.array([100.0, 200.0])
CORNER_UV = [(100.0, 100.0), (540.0, 100.0), (100.0, 380.0), (540.0, 380.0)]


def _image_for(centre, radius=45):
    image = np.full((HEIGHT, WIDTH), 210, dtype=np.uint8)
    cv2.circle(
        image,
        (int(round(centre[0])), int(round(centre[1]))),
        radius,
        40,
        -1,
    )
    image = cv2.GaussianBlur(image, (0, 0), 1.5)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def _robot_point(uv):
    return (ROBOT_PER_PIXEL @ (np.asarray(uv) - IMAGE_CENTRE)) + ROBOT_ORIGIN


def _calibration():
    images = [_image_for(uv) for uv in CORNER_UV]
    points = [_robot_point(uv).tolist() for uv in CORNER_UV]
    pick = GantryPick()
    calibration = pick.calibrate_vision(images, points)
    assert calibration.valid
    assert calibration.matrix.shape == (3, 3)
    return pick, calibration


def test_vision_matrix_and_camera_centre():
    pick, calibration = _calibration()
    matrix = calibration.matrix
    uv = (300.0, 200.0)
    robot_xy = (90.0, 180.0)

    location = pick.locate(_image_for(uv), robot_xy, matrix)

    assert location.valid
    assert location.reason == "ok"
    assert location.image_uv == pytest.approx(uv, abs=1e-3)
    assert location.gripper_xy is None

    expected = np.asarray(robot_xy) + matrix[:2, :2] @ (
        IMAGE_CENTRE - np.asarray(location.image_uv)
    )
    assert location.camera_center_xy == pytest.approx(
        (expected[0], expected[1]), abs=1e-9
    )


def test_offset_matrix_adds_to_camera_centre():
    pick, calibration = _calibration()
    offset = offset_homogeneous_matrix((8.0, 4.0))

    location = pick.locate(
        _image_for((300.0, 200.0)),
        (90.0, 180.0),
        calibration.matrix,
        offset,
    )

    assert location.valid
    assert location.grasp_ready
    assert location.gripper_xy == pytest.approx(
        (
            location.camera_center_xy[0] + 8.0,
            location.camera_center_xy[1] + 4.0,
        ),
        abs=1e-9,
    )


def test_translation_column_is_ignored_by_locate():
    pick, calibration = _calibration()
    changed = calibration.matrix.copy()
    changed[0, 2] += 1000.0
    changed[1, 2] -= 500.0

    original = pick.locate(
        _image_for((300.0, 200.0)), (90.0, 180.0), calibration.matrix
    )
    moved = pick.locate(_image_for((300.0, 200.0)), (90.0, 180.0), changed)

    assert moved.camera_center_xy == pytest.approx(original.camera_center_xy)


def test_locate_accepts_three_values_and_ignores_z():
    pick, calibration = _calibration()
    two = pick.locate(
        _image_for((300.0, 200.0)), (90.0, 180.0), calibration.matrix
    )
    three = pick.locate(
        _image_for((300.0, 200.0)), (90.0, 180.0, 55.0), calibration.matrix
    )

    assert three.valid
    assert three.camera_center_xy == pytest.approx(two.camera_center_xy)


def test_missing_target_returns_invalid():
    pick, calibration = _calibration()
    blank = np.full((HEIGHT, WIDTH, 3), 210, dtype=np.uint8)

    location = pick.locate(blank, (90.0, 180.0), calibration.matrix)

    assert not location.valid
    assert location.reason
    assert location.camera_center_xy is None
    assert location.gripper_xy is None
    assert location.detection is not None


def test_bad_vision_matrix_is_rejected():
    pick, calibration = _calibration()
    bad = calibration.matrix.copy()
    bad[:2, :2] = 0.0

    with pytest.raises(PickError) as error:
        pick.locate(_image_for((300.0, 200.0)), (90.0, 180.0), bad)
    assert error.value.code == "bad_matrix"

    with pytest.raises(PickError):
        pick.locate(
            _image_for((300.0, 200.0)), (90.0, 180.0), [[1, 2], [3, 4]]
        )


def test_bad_offset_matrix_is_rejected():
    pick, calibration = _calibration()
    rotated = np.array(
        [[0.0, -1.0, 8.0], [1.0, 0.0, 4.0], [0.0, 0.0, 1.0]]
    )

    with pytest.raises(PickError) as error:
        pick.locate(
            _image_for((300.0, 200.0)),
            (90.0, 180.0),
            calibration.matrix,
            rotated,
        )
    assert error.value.code == "bad_matrix"


def test_offset_matrix_helper_layout():
    matrix = offset_homogeneous_matrix((8.0, 4.0))
    assert matrix.shape == (3, 3)
    assert matrix.tolist() == [
        [1.0, 0.0, 8.0],
        [0.0, 1.0, 4.0],
        [0.0, 0.0, 1.0],
    ]


def test_vision_requires_four_images():
    pick = GantryPick()
    images = [_image_for(uv) for uv in CORNER_UV[:3]]
    points = [_robot_point(uv).tolist() for uv in CORNER_UV[:3]]
    with pytest.raises(CalibrationError):
        pick.calibrate_vision(images, points)
