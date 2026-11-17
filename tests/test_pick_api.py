"""Tests for the stateless gantry pick matrix API (coordinates only)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from measure import GantryPick, PickError
from measure.pick_api import (
    check_image_point,
    image_size_pair,
    offset_homogeneous_matrix,
)
from measure.robot_calibration import CalibrationError, OffsetSample

WIDTH = 640
HEIGHT = 480
IMAGE_CENTRE = np.array([(WIDTH - 1) * 0.5, (HEIGHT - 1) * 0.5])
ROBOT_PER_PIXEL = np.array([[0.05, 0.0], [0.0, 0.05]])
ROBOT_ORIGIN = np.array([100.0, 200.0])
CORNER_UV = [(100.0, 100.0), (540.0, 100.0), (100.0, 380.0), (540.0, 380.0)]

# One real calibration set, copied from the 2026-11-16 debug package.  It needs
# no image: this is what the coordinate-only endpoint receives.
REAL_IMAGE_SIZE = (5472, 3648)
REAL_IMAGE_POINTS = [
    (2458.040224337917, 2279.4969844857173),
    (3206.1075863362894, 1334.4446890724007),
    (3308.557276553789, 2183.0556801280945),
    (2356.3462968677204, 1431.227404097654),
]
REAL_ROBOT_POINTS = [
    (10.928000450134277, 321.6470031738281),
    (20.928000450134277, 331.6470031738281),
    (10.928000450134277, 331.6470031738281),
    (20.928000450134277, 321.6470031738281),
]
REAL_MATRIX = np.array(
    [
        [-0.0013213626423343333, -0.011627365736710232, 40.681749989616],
        [0.011604128501496913, -0.0013960386325849925, 296.3037814102695],
        [0.0, 0.0, 1.0],
    ]
)


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
    points = [_robot_point(uv).tolist() for uv in CORNER_UV]
    pick = GantryPick()
    calibration = pick.calibrate_vision(
        CORNER_UV, points, image_size=(WIDTH, HEIGHT)
    )
    assert calibration.valid
    assert calibration.matrix.shape == (3, 3)
    return pick, calibration


# ---------------------------------------------------------------------------
# Vision calibration
# ---------------------------------------------------------------------------


def test_vision_matrix_and_camera_centre():
    pick, calibration = _calibration()
    matrix = calibration.matrix
    uv = (300.0, 200.0)
    robot_xy = (90.0, 180.0)

    location = pick.locate(
        uv, robot_xy, matrix, image_size=(WIDTH, HEIGHT)
    )

    assert location.valid
    assert location.reason == "ok"
    assert location.image_uv == pytest.approx(uv, abs=1e-9)
    assert location.gripper_xy is None

    expected = np.asarray(robot_xy) + matrix[:2, :2] @ (
        IMAGE_CENTRE - np.asarray(uv)
    )
    assert location.camera_center_xy == pytest.approx(
        (expected[0], expected[1]), abs=1e-9
    )
    assert location.delta_xy == pytest.approx((expected[0] - 90.0, expected[1] - 180.0))


def test_real_calibration_set_reproduces_matrix():
    """Regression: the 2026-11-16 package must reproduce bit-for-bit."""
    calibration = GantryPick().calibrate_vision(
        REAL_IMAGE_POINTS,
        REAL_ROBOT_POINTS,
        image_size=REAL_IMAGE_SIZE,
    )
    assert calibration.valid
    assert calibration.rms_px == pytest.approx(0.2073, abs=5e-4)
    assert np.allclose(calibration.matrix, REAL_MATRIX, atol=1e-12, rtol=0.0)


def test_real_locate_matches_recorded_coordinates():
    location = GantryPick().locate(
        (2836.937, 1806.012),
        (15.928, 326.647),
        REAL_MATRIX,
        [[1.0, 0.0, 123.67973454793294],
         [0.0, 1.0, 1.4771626790364583],
         [0.0, 0.0, 1.0]],
        image_size=REAL_IMAGE_SIZE,
    )
    assert location.valid
    assert location.camera_center_xy == pytest.approx(
        (15.8587, 325.4455), abs=1e-4
    )
    assert location.gripper_xy == pytest.approx(
        (139.5384, 326.9227), abs=1e-4
    )


def test_vision_requires_four_points():
    pick = GantryPick()
    points = [_robot_point(uv).tolist() for uv in CORNER_UV[:3]]
    with pytest.raises(PickError):
        pick.calibrate_vision(
            CORNER_UV[:3], points, image_size=(WIDTH, HEIGHT)
        )


def test_vision_requires_matching_counts():
    pick = GantryPick()
    points = [_robot_point(uv).tolist() for uv in CORNER_UV[:3]]
    with pytest.raises(PickError) as error:
        pick.calibrate_vision(
            CORNER_UV, points, image_size=(WIDTH, HEIGHT)
        )
    assert error.value.code == "point_count_mismatch"


def test_vision_rejects_point_outside_frame():
    pick = GantryPick()
    points = [_robot_point(uv).tolist() for uv in CORNER_UV]
    swapped = [(v, u) for u, v in CORNER_UV]  # row/col swap
    with pytest.raises(PickError) as error:
        pick.calibrate_vision(swapped, points, image_size=(WIDTH, HEIGHT))
    assert error.value.code == "point_out_of_image"


# ---------------------------------------------------------------------------
# Offset
# ---------------------------------------------------------------------------


def test_offset_matrix_adds_to_camera_centre():
    pick, calibration = _calibration()
    offset = offset_homogeneous_matrix((8.0, 4.0))

    location = pick.locate(
        (300.0, 200.0),
        (90.0, 180.0),
        calibration.matrix,
        offset,
        image_size=(WIDTH, HEIGHT),
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


def test_offset_matrix_helper_layout():
    matrix = offset_homogeneous_matrix((8.0, 4.0))
    assert matrix.shape == (3, 3)
    assert matrix.tolist() == [
        [1.0, 0.0, 8.0],
        [0.0, 1.0, 4.0],
        [0.0, 0.0, 1.0],
    ]


def test_inconsistent_offset_samples_are_invalid():
    samples = [
        OffsetSample(place_pose=(20.0, 20.0), center_pose=(20.0, 20.0)),
        OffsetSample(place_pose=(139.0, 327.0), center_pose=(15.8, 326.2)),
    ]
    offset = GantryPick().calibrate_offset(samples)
    assert not offset.valid
    assert "offset_inconsistent" in offset.warnings


def test_consistent_offset_samples_stay_valid():
    samples = [
        OffsetSample(place_pose=(139.2, 327.1), center_pose=(15.8, 326.1)),
        OffsetSample(place_pose=(139.4, 327.3), center_pose=(15.9, 326.3)),
    ]
    offset = GantryPick().calibrate_offset(samples)
    assert offset.valid
    assert offset.offset_xy == pytest.approx((123.45, 1.0), abs=1e-9)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_translation_column_is_ignored_by_locate():
    pick, calibration = _calibration()
    changed = calibration.matrix.copy()
    changed[0, 2] += 1000.0
    changed[1, 2] -= 500.0

    original = pick.locate(
        (300.0, 200.0), (90.0, 180.0), calibration.matrix,
        image_size=(WIDTH, HEIGHT),
    )
    moved = pick.locate(
        (300.0, 200.0), (90.0, 180.0), changed, image_size=(WIDTH, HEIGHT)
    )

    assert moved.camera_center_xy == pytest.approx(original.camera_center_xy)


def test_locate_accepts_three_values_and_ignores_z():
    pick, calibration = _calibration()
    two = pick.locate(
        (300.0, 200.0), (90.0, 180.0), calibration.matrix,
        image_size=(WIDTH, HEIGHT),
    )
    three = pick.locate(
        (300.0, 200.0), (90.0, 180.0, 55.0), calibration.matrix,
        image_size=(WIDTH, HEIGHT),
    )

    assert three.valid
    assert three.camera_center_xy == pytest.approx(two.camera_center_xy)


def test_bad_vision_matrix_is_rejected():
    pick, calibration = _calibration()
    bad = calibration.matrix.copy()
    bad[:2, :2] = 0.0

    with pytest.raises(PickError) as error:
        pick.locate(
            (300.0, 200.0), (90.0, 180.0), bad, image_size=(WIDTH, HEIGHT)
        )
    assert error.value.code == "bad_matrix"

    with pytest.raises(PickError):
        pick.locate(
            (300.0, 200.0), (90.0, 180.0), [[1, 2], [3, 4]],
            image_size=(WIDTH, HEIGHT),
        )


def test_bad_offset_matrix_is_rejected():
    pick, calibration = _calibration()
    rotated = np.array(
        [[0.0, -1.0, 8.0], [1.0, 0.0, 4.0], [0.0, 0.0, 1.0]]
    )

    with pytest.raises(PickError) as error:
        pick.locate(
            (300.0, 200.0),
            (90.0, 180.0),
            calibration.matrix,
            rotated,
            image_size=(WIDTH, HEIGHT),
        )
    assert error.value.code == "bad_matrix"


def test_helpers_validate_size_and_point():
    assert image_size_pair([640, 480]) == (640, 480)
    with pytest.raises(PickError):
        image_size_pair([1, 480])
    with pytest.raises(PickError):
        image_size_pair([640])
    assert check_image_point((0.0, 0.0), (640, 480)) == (0.0, 0.0)
    with pytest.raises(PickError) as error:
        check_image_point((700.0, 0.0), (640, 480))
    assert error.value.code == "point_out_of_image"


# ---------------------------------------------------------------------------
# Library helpers that keep the old detect-then-compute path (offline only)
# ---------------------------------------------------------------------------


def test_image_helpers_still_locate_and_calibrate():
    pick = GantryPick()
    images = [_image_for(uv) for uv in CORNER_UV]
    points = [_robot_point(uv).tolist() for uv in CORNER_UV]
    calibration = pick.calibrate_vision_from_images(images, points)
    assert calibration.valid

    location = pick.locate_image(_image_for((300.0, 200.0)), (90.0, 180.0),
                                 calibration.matrix)
    assert location.valid
    assert location.image_uv == pytest.approx((300.0, 200.0), abs=1e-2)


def test_missing_target_returns_invalid():
    pick, calibration = _calibration()
    blank = np.full((HEIGHT, WIDTH, 3), 210, dtype=np.uint8)

    location = pick.locate_image(blank, (90.0, 180.0), calibration.matrix)

    assert not location.valid
    assert location.reason
    assert location.camera_center_xy is None
    assert location.gripper_xy is None
    assert location.detection is not None


def test_calibration_error_still_raised_for_degenerate_poses():
    pick = GantryPick()
    collinear = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (30.0, 0.0)]
    with pytest.raises(CalibrationError):
        pick.calibrate_vision(
            CORNER_UV, collinear, image_size=(WIDTH, HEIGHT)
        )
