"""
Tests for the camera-on-arm pick calibration.

Covers the differential affine fit (including gross-outlier rejection and
the four-point no-op pitfall), the camera-to-gripper offset, recipe
persistence, verification, and a synthetic closed-loop arm where the true
camera model is known.
"""

from __future__ import annotations

import json
import math

import cv2
import numpy as np
import pytest

from measure import (
    CalibrationError,
    CameraGripperOffset,
    CircleLocator,
    ImageRobotCalibration,
    OffsetSample,
    RobotPickRecipe,
    calibrate_gripper_offset,
    calibrate_image_to_robot,
    capture_and_calibrate,
    combine_offsets,
    fit_image_to_robot,
    verify_image_to_robot,
    verify_pick,
)
from measure.robot_calibration import CallableRobot


# ---------------------------------------------------------------------------
# Synthetic arm with a known camera model
# ---------------------------------------------------------------------------


class SyntheticArm:
    """A 2D gantry whose camera renders one fixed circular target.

    The world-to-image model is ``p = c_pp + H^-1 (W - pose - K)`` where
    ``H`` is the mm-per-pixel matrix (scale + rotation) and ``K`` is the
    camera offset from the flange.  The reusable differential affine is
    therefore ``A = -H``.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        mm_per_px: float = 0.05,
        rotation_deg: float = 20.0,
        camera_offset=(10.0, 5.0),
        gripper_offset=(2.0, 1.0),
        principal_point=None,
        target_world=(120.0, 90.0),
        radius_mm: float = 2.5,
        noise: float = 1.0,
        blur: float = 0.8,
        seed: int = 0,
        start_pose=(114.0, 82.0, 50.0),
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.mm_per_px = float(mm_per_px)
        self.rotation_deg = float(rotation_deg)
        theta = math.radians(rotation_deg)
        self.h = mm_per_px * np.array(
            [[math.cos(theta), -math.sin(theta)],
             [math.sin(theta), math.cos(theta)]],
            dtype=float,
        )
        self.camera_offset = np.asarray(camera_offset, dtype=float)
        self.gripper_offset = np.asarray(gripper_offset, dtype=float)
        self.principal_point = (
            np.array([(width - 1) * 0.5, (height - 1) * 0.5], dtype=float)
            if principal_point is None
            else np.asarray(principal_point, dtype=float)
        )
        self.target_world = np.asarray(target_world, dtype=float)
        self.radius_mm = float(radius_mm)
        self.noise = float(noise)
        self.blur = float(blur)
        self.seed = int(seed)
        self.pose = np.asarray(start_pose, dtype=float)
        self._frame = 0

    # -- ground truth ----------------------------------------------------

    @property
    def true_affine(self) -> np.ndarray:
        return -self.h

    def true_center_pose(self, target=None) -> np.ndarray:
        world = self.target_world if target is None else np.asarray(target)
        return world - self.camera_offset

    def true_grasp_pose(self, target=None) -> np.ndarray:
        world = self.target_world if target is None else np.asarray(target)
        return world - self.gripper_offset

    def image_of(self, world=None, pose=None) -> np.ndarray:
        world = self.target_world if world is None else np.asarray(world, dtype=float)
        pose = self.pose if pose is None else np.asarray(pose, dtype=float)
        camera = pose[:2] + self.camera_offset
        return self.principal_point + np.linalg.inv(self.h) @ (world - camera)

    # -- rendering / robot interface -------------------------------------

    def render(self) -> np.ndarray:
        self._frame += 1
        centre = self.image_of()
        radius = self.radius_mm / self.mm_per_px
        yy, xx = np.mgrid[0:self.height, 0:self.width]
        distance = np.hypot(xx - centre[0], yy - centre[1])
        alpha = np.clip(0.5 + (radius - distance), 0.0, 1.0)
        image = 200.0 + (40.0 - 200.0) * alpha
        if self.blur > 0:
            image = cv2.GaussianBlur(image, (0, 0), self.blur)
        rng = np.random.default_rng(self.seed + self._frame)
        image = image + rng.normal(0, self.noise, image.shape)
        return np.clip(image, 0, 255).astype(np.uint8)

    def capture(self) -> np.ndarray:
        return self.render()

    def get_position(self):
        return self.pose.copy()

    def move_to(self, x: float, y: float, z=None) -> None:
        self.pose[0] = float(x)
        self.pose[1] = float(y)
        if z is not None:
            self.pose[2] = float(z)


def make_arm(**kwargs) -> SyntheticArm:
    return SyntheticArm(**kwargs)


def make_locator() -> CircleLocator:
    return CircleLocator(radius_range=(25, 90), polarity="dark", min_confidence=0.3)


def make_robot(arm: SyntheticArm) -> CallableRobot:
    return CallableRobot(arm.get_position, arm.move_to)


def corner_poses(arm: SyntheticArm, margin: float = 0.20) -> list:
    """Four taught poses that place the target near the image corners."""
    centre = arm.true_center_pose()
    centre_px = arm.principal_point
    width, height = float(arm.width), float(arm.height)
    corners = [
        (margin * width, margin * height),
        ((1.0 - margin) * width, margin * height),
        ((1.0 - margin) * width, (1.0 - margin) * height),
        (margin * width, (1.0 - margin) * height),
    ]
    return [
        np.append(centre + arm.true_affine @ (np.asarray(q) - centre_px), 50.0)
        for q in corners
    ]


def render_at(arm: SyntheticArm, poses) -> list:
    """Render one frame per pose without leaving the arm moved."""
    saved = arm.pose.copy()
    images = []
    try:
        for pose in poses:
            arm.pose = np.asarray(pose, dtype=float)
            images.append(arm.capture())
    finally:
        arm.pose = saved
    return images


def pure_calibration(arm: SyntheticArm, locator, poses=None):
    """Offline calibration: images at the given poses, no robot motion."""
    poses = corner_poses(arm) if poses is None else list(poses)
    images = render_at(arm, poses)
    return calibrate_image_to_robot(images, poses, locator=locator)


# ---------------------------------------------------------------------------
# Differential affine fit
# ---------------------------------------------------------------------------


class TestImageRobotCalibration:
    def test_servo_delta_sign_and_center_pose(self):
        affine = np.array([[-0.1, 0.0], [0.0, -0.1]])
        image = np.array(
            [[100, 100], [500, 100], [500, 400], [100, 400]], dtype=float
        )
        robot = image @ affine.T + np.array([100.0, 50.0])
        calibration = ImageRobotCalibration.fit(
            robot, image, principal_point=(320, 240), image_size=(640, 480)
        )
        assert calibration.valid
        # A target right of centre needs the camera to move right (+X).
        assert np.allclose(calibration.servo_delta((420, 240)), [10.0, 0.0])
        pose = calibration.center_pose([100.0, 50.0, 7.0], (420, 240))
        assert np.allclose(pose, [110.0, 50.0, 7.0])
        # Image displacement and robot displacement are inverse maps.
        assert np.allclose(calibration.image_delta([10.0, 0.0]), [-100.0, 0.0])

    def test_fit_recovers_known_map(self):
        affine = np.array([[-0.05, 0.01], [0.008, -0.052]])
        offset = np.array([100.0, 50.0])
        rng = np.random.default_rng(3)
        image = rng.uniform([100, 100], [500, 400], (7, 2))
        robot = image @ affine.T + offset
        calibration = ImageRobotCalibration.fit(
            robot, image, principal_point=(320, 240), image_size=(640, 480)
        )
        assert calibration.valid
        assert np.allclose(calibration.robot_per_pixel, affine, atol=1e-10)
        assert np.allclose(
            calibration.pixel_per_robot, np.linalg.inv(affine), atol=1e-10
        )
        assert calibration.rms_px < 1e-9
        assert np.allclose(
            calibration.translation_xy,
            robot.mean(axis=0) - affine @ image.mean(axis=0),
            atol=1e-9,
        )
        design = np.column_stack([image, np.ones(len(image))])
        predicted = design @ calibration.matrix_3x3.T
        assert np.allclose(predicted[:, :2], robot, atol=1e-8)
        assert calibration.mm_per_px == pytest.approx(
            math.sqrt(abs(np.linalg.det(affine)))
        )

    def test_fit_needs_three_points(self):
        with pytest.raises(CalibrationError):
            ImageRobotCalibration.fit(
                [[0, 0], [1, 1]],
                [[0, 0], [1, 0]],
                principal_point=(10, 10),
                image_size=(20, 20),
            )

    def test_fit_rejects_collinear_robot_motion(self):
        image = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        robot = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=float)
        with pytest.raises(CalibrationError):
            ImageRobotCalibration.fit(
                robot, image, principal_point=(5, 5), image_size=(20, 20)
            )

    def test_fit_rejects_collinear_image_points(self):
        image = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=float)
        robot = np.array([[0, 0], [10, 0], [0, 10], [10, 10]], dtype=float)
        with pytest.raises(CalibrationError):
            ImageRobotCalibration.fit(
                robot, image, principal_point=(5, 5), image_size=(20, 20)
            )

    def test_clustered_points_are_invalid(self):
        affine = np.array([[-0.05, 0.0], [0.0, -0.05]])
        image = np.array([[300, 200], [303, 200], [300, 203], [303, 203]],
                         dtype=float)
        robot = image @ affine.T + np.array([100.0, 50.0])
        calibration = ImageRobotCalibration.fit(
            robot, image, principal_point=(320, 240), image_size=(640, 480)
        )
        assert not calibration.valid
        assert "clustered_image_points" in calibration.warnings

    def test_robust_fit_rejects_gross_outlier(self):
        affine = np.array([[-0.05, 0.01], [0.008, -0.052]])
        offset = np.array([100.0, 50.0])
        rng = np.random.default_rng(0)
        image = rng.uniform([100, 100], [500, 400], (6, 2))
        robot = image @ affine.T + offset
        robot[-1] += 40.0
        calibration = ImageRobotCalibration.fit(
            robot, image, principal_point=(320, 240), image_size=(640, 480)
        )
        assert calibration.valid
        assert "outlier_rejected" in calibration.warnings
        assert np.allclose(calibration.robot_per_pixel, affine, atol=1e-9)
        assert calibration.inlier_rms_px < 0.1
        assert [sample.inlier for sample in calibration.samples] == [
            True, True, True, True, True, False
        ]

    def test_leave_one_out_is_reported(self):
        affine = np.array([[-0.05, 0.0], [0.0, -0.05]])
        offset = np.array([100.0, 50.0])
        rng = np.random.default_rng(4)
        image = rng.uniform([100, 100], [500, 400], (6, 2))
        image = image + rng.normal(0, 0.2, image.shape)
        robot = image @ affine.T + offset
        calibration = ImageRobotCalibration.fit(
            robot, image, principal_point=(320, 240), image_size=(640, 480)
        )
        assert calibration.loo_rms_px is not None
        assert calibration.loo_rms_px < 1.0
        assert calibration.loo_rms_mm <= calibration.loo_rms_px * 0.06

    def test_default_principal_point_is_image_center(self):
        affine = np.array([[-0.05, 0.0], [0.0, -0.05]])
        image = np.array([[100, 100], [500, 100], [500, 400], [100, 400]],
                         dtype=float)
        robot = image @ affine.T + np.array([10.0, 20.0])
        calibration = ImageRobotCalibration.fit(
            robot, image, image_size=(640, 480)
        )
        assert calibration.principal_point == (319.5, 239.5)

    def test_serialization_roundtrip(self, tmp_path):
        affine = np.array([[-0.05, 0.01], [0.008, -0.052]])
        rng = np.random.default_rng(8)
        image = rng.uniform([100, 100], [500, 400], (6, 2))
        robot = image @ affine.T + np.array([100.0, 50.0])
        calibration = ImageRobotCalibration.fit(
            robot, image, principal_point=(320, 240), image_size=(640, 480),
            calib_z=12.5,
        )
        path = tmp_path / "calibration.json"
        calibration.save(str(path))
        restored = ImageRobotCalibration.load(str(path))
        assert np.allclose(restored.robot_per_pixel, calibration.robot_per_pixel)
        assert np.allclose(restored.translation_xy, calibration.translation_xy)
        assert restored.principal_point == calibration.principal_point
        assert restored.image_size == calibration.image_size
        assert restored.calib_z == pytest.approx(12.5)
        assert len(restored.samples) == len(calibration.samples)

    def test_rejects_wrong_schema_and_future_version(self, tmp_path):
        with pytest.raises(CalibrationError):
            ImageRobotCalibration.from_dict({"schema": "wrong", "version": 1})
        with pytest.raises(CalibrationError):
            ImageRobotCalibration.from_dict(
                {"schema": "measure.image_robot_calibration", "version": 99}
            )


# ---------------------------------------------------------------------------
# Camera-to-gripper offset
# ---------------------------------------------------------------------------


class TestCameraGripperOffset:
    def test_offset_mean_and_apply(self):
        samples = [
            OffsetSample(place_pose=(110, 90), center_pose=(100, 80)),
            OffsetSample(place_pose=(120, 95), center_pose=(110, 85)),
        ]
        offset = CameraGripperOffset.from_samples(samples)
        assert offset.valid
        assert np.allclose(offset.offset_xy, [10.0, 10.0])
        assert np.allclose(offset.std_xy, [0.0, 0.0])
        assert np.allclose(offset.apply([0.0, 0.0, 7.0]), [10.0, 10.0, 7.0])
        assert offset.samples[0].offset_xy == (10.0, 10.0)

    def test_large_centring_error_is_invalid(self):
        sample = OffsetSample(
            place_pose=(110, 90), center_pose=(100, 80), center_error_px=20.0
        )
        offset = CameraGripperOffset.from_samples([sample], center_tolerance_px=5.0)
        assert not offset.valid
        assert "centring_error_high" in offset.warnings

    def test_combine_offsets(self):
        first = CameraGripperOffset.from_samples(
            [OffsetSample(place_pose=(110, 90), center_pose=(100, 80))]
        )
        second = CameraGripperOffset.from_samples(
            [OffsetSample(place_pose=(112, 92), center_pose=(100, 80))]
        )
        combined = combine_offsets([first, second])
        assert combined.offset_xy == pytest.approx((11.0, 11.0))
        assert len(combined.samples) == 2

    def test_empty_samples_raise(self):
        with pytest.raises(CalibrationError):
            CameraGripperOffset.from_samples([])

    def test_serialization_roundtrip(self, tmp_path):
        offset = CameraGripperOffset.from_samples(
            [OffsetSample(place_pose=(110, 90), center_pose=(100, 80),
                          center_error_px=0.4)]
        )
        path = tmp_path / "offset.json"
        offset.save(str(path))
        restored = CameraGripperOffset.load(str(path))
        assert restored.offset_xy == offset.offset_xy
        assert restored.valid == offset.valid
        assert restored.samples[0].place_pose == (110.0, 90.0)


# ---------------------------------------------------------------------------
# Recipe geometry
# ---------------------------------------------------------------------------


class TestRobotPickRecipe:
    @staticmethod
    def _recipe():
        affine = np.array([[-0.1, 0.0], [0.0, -0.1]])
        image = np.array([[100, 100], [500, 100], [500, 400], [100, 400]],
                         dtype=float)
        robot = image @ affine.T + np.array([100.0, 50.0])
        calibration = ImageRobotCalibration.fit(
            robot, image, principal_point=(320, 240), image_size=(640, 480)
        )
        offset = CameraGripperOffset.from_samples(
            [OffsetSample(place_pose=(110, 90), center_pose=(100, 80))]
        )
        return RobotPickRecipe(calibration, offset)

    def test_single_move_grasp_formula(self):
        recipe = self._recipe()
        current = [100.0, 50.0, 7.0]
        detection = (420.0, 240.0)
        center = recipe.center_pose(current, detection)
        assert np.allclose(center, [110.0, 50.0, 7.0])
        grasp = recipe.grasp_pose(current, detection)
        assert np.allclose(grasp, [120.0, 60.0, 7.0])
        assert np.allclose(
            np.asarray(current[:2]) + recipe.pick_delta(detection),
            [120.0, 60.0],
        )
        expected = recipe.expected_uv_at_grasp()
        # J = A^-1 = -10 I; offset (10, 10) -> (-100, -100) px from centre.
        assert expected == pytest.approx((220.0, 140.0))
        assert np.allclose(recipe.grasp_pose(current, detection, z=3.0),
                           [120.0, 60.0, 3.0])

    def test_missing_offset_raises_and_fails_validation(self):
        calibration = self._recipe().image_robot
        recipe = RobotPickRecipe(calibration)
        with pytest.raises(CalibrationError):
            recipe.grasp_pose([0, 0], (10, 10))
        with pytest.raises(CalibrationError):
            recipe.expected_uv_at_grasp()
        report = recipe.validate()
        assert not report["valid"]
        assert "offset_missing" in report["warnings"]

    def test_recipe_roundtrip(self, tmp_path):
        recipe = self._recipe()
        path = tmp_path / "recipe.json"
        recipe.save(str(path))
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["schema"] == "measure.robot_pick_recipe"
        restored = RobotPickRecipe.load(str(path))
        assert np.allclose(
            restored.image_robot.robot_per_pixel,
            recipe.image_robot.robot_per_pixel,
        )
        assert restored.gripper_offset.offset_xy == recipe.gripper_offset.offset_xy
        assert np.allclose(restored.pick_delta((420, 240)),
                           recipe.pick_delta((420, 240)))
        assert restored.validate()["valid"]

    def test_recipe_rejects_wrong_schema(self):
        with pytest.raises(CalibrationError):
            RobotPickRecipe.from_dict({"schema": "other", "version": 1})


# ---------------------------------------------------------------------------
# Offline pure API
# ---------------------------------------------------------------------------


class TestPureImageCalibration:
    def test_pure_four_point_calibration(self):
        arm = make_arm()
        locator = make_locator()
        poses = corner_poses(arm)
        calibration = pure_calibration(arm, locator, poses)

        assert calibration.valid, calibration.warnings
        assert len(calibration.samples) == 4
        assert calibration.matrix.shape == (3, 3)
        assert np.allclose(calibration.matrix, calibration.matrix_3x3)
        assert calibration.robot_per_pixel.shape == (2, 2)
        assert np.allclose(
            calibration.robot_per_pixel, arm.true_affine, atol=2e-3
        )
        assert calibration.mm_per_px == pytest.approx(arm.mm_per_px, rel=0.02)
        assert calibration.calib_z == pytest.approx(50.0)
        assert calibration.rms_px < 0.3

        arm.pose = np.array([104.0, 82.0, 50.0])
        report = verify_image_to_robot(
            make_robot(arm), arm.capture, calibration, locator=locator,
            settle_s=0.0, checks=3,
        )
        assert report.passed, report.as_dict()

    def test_pure_seven_point_calibration(self):
        arm = make_arm()
        locator = make_locator()
        centre = arm.true_center_pose()
        centre_px = arm.principal_point
        width, height = float(arm.width), float(arm.height)
        targets = [
            (0.20 * width, 0.20 * height),
            (0.50 * width, 0.20 * height),
            (0.80 * width, 0.20 * height),
            (0.20 * width, 0.50 * height),
            (0.80 * width, 0.50 * height),
            (0.20 * width, 0.80 * height),
            (0.80 * width, 0.80 * height),
        ]
        poses = [
            np.append(centre + arm.true_affine @ (np.asarray(q) - centre_px), 50.0)
            for q in targets
        ]
        calibration = pure_calibration(arm, locator, poses)

        assert calibration.valid, calibration.warnings
        assert len(calibration.samples) == 7
        assert np.allclose(
            calibration.robot_per_pixel, arm.true_affine, atol=2e-3
        )

    def test_pure_accepts_two_dimensional_robot_points(self):
        arm = make_arm()
        locator = make_locator()
        poses = corner_poses(arm)
        images = render_at(arm, poses)
        calibration = calibrate_image_to_robot(
            images, [pose[:2] for pose in poses], locator=locator
        )

        assert calibration.valid, calibration.warnings
        assert calibration.calib_z is None
        assert np.allclose(
            calibration.robot_per_pixel, arm.true_affine, atol=2e-3
        )

    def test_pure_rejects_bad_image(self):
        arm = make_arm()
        locator = make_locator()
        poses = corner_poses(arm)
        images = render_at(arm, poses)
        images[2] = np.zeros_like(images[2])
        with pytest.raises(CalibrationError, match=r"images\[2\]"):
            calibrate_image_to_robot(images, poses, locator=locator)

    def test_pure_rejects_mismatch(self):
        arm = make_arm()
        locator = make_locator()
        poses = corner_poses(arm)
        images = render_at(arm, poses)
        with pytest.raises(CalibrationError, match="same length"):
            calibrate_image_to_robot(images, poses[:3], locator=locator)

    def test_pure_requires_four_images(self):
        arm = make_arm()
        locator = make_locator()
        poses = corner_poses(arm)
        images = render_at(arm, poses)
        with pytest.raises(CalibrationError, match="at least 4"):
            calibrate_image_to_robot(images[:3], poses[:3], locator=locator)

    def test_pure_rejects_mixed_sizes(self):
        arm = make_arm()
        locator = make_locator()
        poses = corner_poses(arm)
        images = render_at(arm, poses)
        images[1] = np.zeros((arm.height // 2, arm.width // 2), dtype=np.uint8)
        with pytest.raises(CalibrationError, match="size"):
            calibrate_image_to_robot(images, poses, locator=locator)

    def test_pure_rejects_single_image_array(self):
        arm = make_arm()
        poses = corner_poses(arm)
        single = np.zeros((arm.height, arm.width), dtype=np.uint8)
        with pytest.raises(CalibrationError, match="sequence"):
            calibrate_image_to_robot(single, poses, locator=make_locator())

    def test_fit_image_to_robot_wrapper(self):
        arm = make_arm()
        poses = corner_poses(arm)
        points = [arm.image_of(pose=pose) for pose in poses]
        calibration = fit_image_to_robot(
            points, poses, principal_point=arm.principal_point,
            image_size=(arm.width, arm.height),
        )

        assert calibration.valid, calibration.warnings
        assert calibration.calib_z == pytest.approx(50.0)
        assert np.allclose(
            calibration.robot_per_pixel, arm.true_affine, atol=2e-3
        )
        direct = ImageRobotCalibration.fit(
            [pose[:2] for pose in poses], points,
            principal_point=arm.principal_point,
            image_size=(arm.width, arm.height),
        )
        assert np.allclose(
            calibration.robot_per_pixel, direct.robot_per_pixel
        )

    def test_matrix_is_three_by_three(self):
        arm = make_arm()
        calibration = pure_calibration(arm, make_locator())
        assert calibration.matrix.shape == (3, 3)
        assert np.allclose(calibration.matrix, calibration.matrix_3x3)
        assert np.allclose(
            calibration.matrix[:2, :2], calibration.robot_per_pixel
        )
        assert np.allclose(
            calibration.matrix[:2, 2], calibration.translation_xy
        )


# ---------------------------------------------------------------------------
# Synthetic closed-loop arm
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_explicit_positions_calibration(self):
        arm = make_arm()
        locator = make_locator()
        center = arm.true_center_pose()
        positions = [
            [center[0], center[1], 50.0],
            [center[0] - 8.0, center[1], 50.0],
            [center[0], center[1] - 8.0, 50.0],
            [center[0] - 8.0, center[1] - 8.0, 50.0],
        ]
        calibration = capture_and_calibrate(
            make_robot(arm), arm.capture, positions=positions,
            locator=locator, settle_s=0.0,
        )
        assert calibration.valid, calibration.warnings
        assert len(calibration.samples) == 5  # start frame + 4 taught poses
        # Z is the height axis: recorded, but never part of the XY fit.
        assert calibration.calib_z == pytest.approx(50.0)
        assert np.allclose(
            calibration.robot_per_pixel, arm.true_affine, atol=2e-3
        )

    def test_four_taught_corners_without_start_target(self):
        """The exact on-site workflow: four corner shots, no start frame.

        The arm starts far from the stage (target not visible); only the four
        taught poses contribute.  This is the clarified four-point procedure.
        """
        arm = make_arm()
        locator = make_locator()
        arm.pose = np.array([300.0, 300.0, 50.0])
        centre = arm.true_center_pose()
        centre_px = arm.principal_point
        width, height = float(arm.width), float(arm.height)
        corners = [
            (0.20 * width, 0.20 * height),
            (0.80 * width, 0.20 * height),
            (0.80 * width, 0.80 * height),
            (0.20 * width, 0.80 * height),
        ]
        positions = [
            np.append(centre + arm.true_affine @ (np.asarray(q) - centre_px), 50.0)
            for q in corners
        ]
        calibration = capture_and_calibrate(
            make_robot(arm), arm.capture, positions=positions,
            locator=locator, settle_s=0.0,
        )
        assert calibration.valid, calibration.warnings
        assert len(calibration.samples) == 4
        assert np.allclose(
            calibration.robot_per_pixel, arm.true_affine, atol=2e-3
        )

    def test_gripper_offset_end_to_end(self):
        arm = make_arm()
        locator = make_locator()
        calibration = pure_calibration(arm, locator)
        assert calibration.valid
        arm.pose = np.array([114.0, 82.0, 50.0])
        release = np.append(arm.true_grasp_pose(), 10.0)
        offset = calibrate_gripper_offset(
            make_robot(arm), arm.capture, calibration, place_pose=release,
            locator=locator, settle_s=0.0,
        )
        assert offset.valid, offset.warnings
        assert np.allclose(
            offset.offset_xy,
            arm.camera_offset - arm.gripper_offset,
            atol=0.02,
        )
        assert offset.mean_center_error_px < 1.0

    def test_verify_pick_end_to_end(self):
        arm = make_arm()
        locator = make_locator()
        calibration = pure_calibration(arm, locator)
        arm.pose = np.array([114.0, 82.0, 50.0])
        release = np.append(arm.true_grasp_pose(), 10.0)
        offset = calibrate_gripper_offset(
            make_robot(arm), arm.capture, calibration, place_pose=release,
            locator=locator, settle_s=0.0,
        )
        recipe = RobotPickRecipe(calibration, offset, locator)

        arm.target_world = np.array([130.0, 70.0])
        arm.pose = np.array([120.0, 65.0, 50.0])
        result = verify_pick(
            make_robot(arm), arm.capture, recipe, locator=locator,
            settle_s=0.0, check_grasp_view=True,
        )
        assert result["passed"], result
        assert result["grasp_view"]["valid"]
        assert result["grasp_view"]["residual_px"] < 1.0
        assert np.allclose(
            result["grasp_pose"][:2], arm.true_grasp_pose(), atol=0.1
        )

    def test_principal_point_offset_is_absorbed(self):
        """A non-centred principal point cancels in the relative pipeline.

        The commanded centring point and the verification criterion are the
        same pixel, so the servo drives the target there exactly; the offset
        absorbs the physical principal-point bias.  The grasp stays accurate.
        """
        arm = make_arm(
            principal_point=(320.0 + 6.0, 240.0 - 4.0),
            start_pose=(114.0, 82.0, 50.0),
        )
        locator = make_locator()
        calibration = pure_calibration(arm, locator)
        assert calibration.valid
        arm.pose = np.array([114.0, 82.0, 50.0])
        release = np.append(arm.true_grasp_pose(), 10.0)
        offset = calibrate_gripper_offset(
            make_robot(arm), arm.capture, calibration, place_pose=release,
            locator=locator, settle_s=0.0,
        )
        recipe = RobotPickRecipe(calibration, offset, locator)

        report = verify_image_to_robot(
            make_robot(arm), arm.capture, calibration, locator=locator,
            settle_s=0.0, checks=2,
        )
        assert report.passed
        assert report.max_error_px < 0.6

        arm.target_world = np.array([130.0, 70.0])
        arm.pose = np.array([120.0, 65.0, 50.0])
        detection = locator.find(arm.capture())
        assert detection.valid
        grasp = recipe.grasp_pose(arm.pose, detection.center)
        assert np.allclose(grasp[:2], arm.true_grasp_pose(), atol=0.1)

        # The expected target position at the grasp pose is still exact.
        expected = np.asarray(recipe.expected_uv_at_grasp())
        arm.pose = np.append(grasp[:2], 50.0)
        after = locator.find(arm.capture())
        assert after.valid
        assert np.linalg.norm(np.asarray(after.center) - expected) < 1.0

    def test_absolute_affine_translation_is_target_specific(self):
        arm = make_arm()
        poses = np.array(
            [[110.0, 85.0], [102.0, 85.0], [110.0, 77.0], [102.0, 77.0]],
            dtype=float,
        )
        images = np.array(
            [arm.image_of(pose=np.append(pose, 50.0)) for pose in poses]
        )
        calibration = ImageRobotCalibration.fit(
            poses, images, principal_point=arm.principal_point,
            image_size=(arm.width, arm.height),
        )
        # A naive absolute affine fitted to (image, pose) reproduces the
        # pose each frame was taken at -- moving there is a no-op.
        design = np.column_stack([images, np.ones(len(images))])
        absolute, _, _, _ = np.linalg.lstsq(design, poses, rcond=None)
        naive = design @ absolute
        assert np.allclose(naive, poses, atol=1e-8)
        assert np.max(np.linalg.norm(naive - poses, axis=1)) < 1e-8

        # The reusable command is the differential correction, which moves
        # the camera onto the target.
        center_pose = arm.true_center_pose()
        for pose, image_point in zip(poses, images):
            corrected = pose + calibration.servo_delta(image_point)
            assert np.allclose(corrected, center_pose, atol=0.02)
        gap = np.max(np.linalg.norm(naive - center_pose, axis=1))
        assert gap > 10.0

    def test_verification_detects_bad_scale(self):
        arm = make_arm()
        locator = make_locator()
        calibration = pure_calibration(arm, locator)
        bad = ImageRobotCalibration.from_dict(calibration.to_dict())
        bad.robot_per_pixel = calibration.robot_per_pixel * 1.5
        bad.pixel_per_robot = np.linalg.inv(bad.robot_per_pixel)
        arm.pose = np.array([104.0, 82.0, 50.0])
        report = verify_image_to_robot(
            make_robot(arm), arm.capture, bad, locator=locator,
            settle_s=0.0, checks=1,
        )
        assert not report.passed
        assert report.max_error_px > 5.0

    def test_blank_image_raises(self):
        arm = make_arm()
        locator = make_locator()
        blank = np.zeros((arm.height, arm.width), dtype=np.uint8)
        poses = corner_poses(arm)
        with pytest.raises(CalibrationError):
            calibrate_image_to_robot([blank] * 4, poses, locator=locator)

    def test_offset_without_visible_target_raises(self):
        arm = make_arm()
        locator = make_locator()
        calibration = pure_calibration(arm, locator)
        blank = np.zeros((arm.height, arm.width), dtype=np.uint8)
        with pytest.raises(CalibrationError):
            calibrate_gripper_offset(
                make_robot(arm), lambda: blank.copy(), calibration, settle_s=0.0
            )
