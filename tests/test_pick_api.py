"""Tests for the pure gantry pick API."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from measure import GantryPick, OffsetSample, PickError
from measure.robot_calibration import CalibrationError, RobotPickRecipe

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
    return (ROBOT_PER_PIXEL @ (np.asarray(uv, dtype=float) - IMAGE_CENTRE)) + ROBOT_ORIGIN


def _calibrated_pick():
    images = [_image_for(uv) for uv in CORNER_UV]
    points = [_robot_point(uv).tolist() for uv in CORNER_UV]
    pick = GantryPick()
    calibration = pick.calibrate_vision(images, points)
    assert calibration.valid
    assert calibration.rms_px < 1e-6
    return pick


def test_vision_calibration_and_camera_centre():
    pick = _calibrated_pick()
    uv = (300.0, 200.0)
    robot_xy = (90.0, 180.0)

    location = pick.locate(_image_for(uv), robot_xy)

    assert location.valid
    assert location.reason == "ok"
    assert location.image_uv == pytest.approx(uv, abs=1e-3)
    assert location.gripper_xy is None
    assert location.warnings == ("offset_not_calibrated",)

    expected = np.asarray(robot_xy) + pick.image_robot.robot_per_pixel @ (
        np.asarray(pick.image_robot.principal_point)
        - np.asarray(location.image_uv)
    )
    assert location.camera_center_xy == pytest.approx(
        (expected[0], expected[1]), abs=1e-9
    )


def test_offset_adds_to_camera_centre():
    pick = _calibrated_pick()
    pick.calibrate_offset([OffsetSample(place_pose=(108.0, 184.0),
                                       center_pose=(100.0, 180.0))])

    location = pick.locate(_image_for((300.0, 200.0)), (90.0, 180.0))

    assert location.valid
    assert location.grasp_ready
    assert location.gripper_xy == pytest.approx(
        (
            location.camera_center_xy[0] + 8.0,
            location.camera_center_xy[1] + 4.0,
        ),
        abs=1e-6,
    )


def test_missing_vision_raises():
    with pytest.raises(PickError) as error:
        GantryPick().locate(_image_for((300.0, 200.0)), (0.0, 0.0))
    assert error.value.code == "vision_not_calibrated"
    assert error.value.status == 409


def test_missing_target_returns_invalid():
    pick = _calibrated_pick()
    blank = np.full((HEIGHT, WIDTH, 3), 210, dtype=np.uint8)

    location = pick.locate(blank, (90.0, 180.0))

    assert not location.valid
    assert location.reason
    assert location.camera_center_xy is None
    assert location.gripper_xy is None
    assert location.detection is not None


def test_image_size_mismatch_returns_invalid():
    pick = _calibrated_pick()
    smaller = cv2.resize(_image_for((300.0, 200.0)), (320, 240))

    location = pick.locate(smaller, (90.0, 180.0))

    assert not location.valid
    assert location.reason == "image_size_mismatch"


def test_locate_accepts_three_values_and_ignores_z():
    pick = _calibrated_pick()
    two = pick.locate(_image_for((300.0, 200.0)), (90.0, 180.0))
    three = pick.locate(_image_for((300.0, 200.0)), (90.0, 180.0, 55.0))

    assert three.valid
    assert three.camera_center_xy == pytest.approx(two.camera_center_xy)


def test_recipe_roundtrip(tmp_path):
    pick = _calibrated_pick()
    pick.calibrate_offset([OffsetSample(place_pose=(108.0, 184.0),
                                       center_pose=(100.0, 180.0))])
    path = tmp_path / "recipe.json"
    pick.save_recipe(str(path))

    restored = GantryPick.load_recipe(str(path))
    assert restored.ready
    original = pick.locate(_image_for((300.0, 200.0)), (90.0, 180.0))
    replayed = restored.locate(_image_for((300.0, 200.0)), (90.0, 180.0))
    assert replayed.camera_center_xy == pytest.approx(original.camera_center_xy)
    assert replayed.gripper_xy == pytest.approx(original.gripper_xy)


def test_recipe_is_json_serialisable(tmp_path):
    pick = _calibrated_pick()
    path = tmp_path / "recipe.json"
    pick.save_recipe(str(path))

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema"] == "measure.robot_pick_recipe"
    assert data["image_robot"]["robot_per_pixel"]
    assert data["gripper_offset"] is None

    recipe = RobotPickRecipe.from_dict(data)
    assert recipe.image_robot.valid


def test_vision_requires_four_images():
    pick = GantryPick()
    images = [_image_for(uv) for uv in CORNER_UV[:3]]
    points = [_robot_point(uv).tolist() for uv in CORNER_UV[:3]]
    with pytest.raises(CalibrationError):
        pick.calibrate_vision(images, points)
