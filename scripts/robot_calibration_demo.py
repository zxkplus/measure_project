#!/usr/bin/env python3
"""Runnable synthetic demo for the camera-on-arm pick calibration.

No hardware is needed: a virtual gantry moves a camera over a fixed
circular target, so the true affine map, camera offset and gripper offset
are known.  The demo runs the two calibration functions, the verification,
the one-move pick formula, and writes the recipe under
``output/robot_calibration_demo/``.

Run with::

    python scripts/robot_calibration_demo.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from measure import (
    CircleLocator,
    RobotPickRecipe,
    calibrate_gripper_offset,
    calibrate_image_to_robot,
    verify_image_to_robot,
    verify_pick,
)
from measure.robot_calibration import CallableRobot

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "robot_calibration_demo"


# ---------------------------------------------------------------------------
# Virtual gantry + camera
# ---------------------------------------------------------------------------


class VirtualArm:
    """A 2D gantry whose camera renders one fixed circular target."""

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        mm_per_px: float = 0.05,
        rotation_deg: float = 20.0,
        camera_offset=(10.0, 5.0),
        gripper_offset=(2.0, 1.0),
        target_world=(120.0, 90.0),
        radius_mm: float = 2.5,
        start_pose=(114.0, 82.0, 50.0),
    ) -> None:
        self.width = width
        self.height = height
        self.mm_per_px = mm_per_px
        theta = math.radians(rotation_deg)
        self.h = mm_per_px * np.array(
            [[math.cos(theta), -math.sin(theta)],
             [math.sin(theta), math.cos(theta)]]
        )
        self.camera_offset = np.asarray(camera_offset, dtype=float)
        self.gripper_offset = np.asarray(gripper_offset, dtype=float)
        self.principal_point = np.array([(width - 1) * 0.5, (height - 1) * 0.5])
        self.target_world = np.asarray(target_world, dtype=float)
        self.radius_mm = radius_mm
        self.pose = np.asarray(start_pose, dtype=float)
        self.frame = 0

    @property
    def true_affine(self) -> np.ndarray:
        return -self.h

    def true_center_pose(self):
        return self.target_world - self.camera_offset

    def true_grasp_pose(self):
        return self.target_world - self.gripper_offset

    def image_of(self, world=None, pose=None) -> np.ndarray:
        world = self.target_world if world is None else np.asarray(world, dtype=float)
        pose = self.pose if pose is None else np.asarray(pose, dtype=float)
        camera = pose[:2] + self.camera_offset
        return self.principal_point + np.linalg.inv(self.h) @ (world - camera)

    def render(self) -> np.ndarray:
        self.frame += 1
        centre = self.image_of()
        radius = self.radius_mm / self.mm_per_px
        yy, xx = np.mgrid[0:self.height, 0:self.width]
        distance = np.hypot(xx - centre[0], yy - centre[1])
        alpha = np.clip(0.5 + (radius - distance), 0.0, 1.0)
        image = 200.0 + (40.0 - 200.0) * alpha
        image = cv2.GaussianBlur(image, (0, 0), 0.8)
        rng = np.random.default_rng(1000 + self.frame)
        image = image + rng.normal(0, 1.0, image.shape)
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


def corner_poses(arm: VirtualArm, margin: float = 0.20) -> list:
    """Four robot poses that place the target near the image corners."""
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


def render_at(arm: VirtualArm, poses) -> list:
    """Render one frame per pose; restore the arm afterwards."""
    saved = arm.pose.copy()
    images = []
    try:
        for pose in poses:
            arm.pose = np.asarray(pose, dtype=float)
            images.append(arm.capture())
    finally:
        arm.pose = saved
    return images


def section(title: str) -> None:
    print()
    print(f"== {title} " + "=" * max(0, 68 - len(title)))


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    arm = VirtualArm()
    robot = CallableRobot(arm.get_position, arm.move_to)
    locator = CircleLocator(radius_range=(25, 90), polarity="dark")

    section("four-point calibration (offline images + 4 corner points)")
    poses = corner_poses(arm)
    images = render_at(arm, poses)
    calibration = calibrate_image_to_robot(images, poses, locator=locator)
    print(f"valid              : {calibration.valid}")
    print(f"samples            : {len(calibration.samples)}")
    print(f"mm/px              : {calibration.mm_per_px:.5f} "
          f"(true {arm.mm_per_px:.5f})")
    print(f"rotation (map)     : {calibration.rotation_deg:.2f} deg")
    print(f"fit residual       : {calibration.rms_px:.4f} px "
          f"({calibration.rms_mm:.4f} mm)")
    print(f"leave-one-out      : {calibration.loo_rms_px:.4f} px")
    print(f"image coverage     : {calibration.image_coverage:.3f}")
    print(f"warnings           : {calibration.warnings}")
    print("robot_per_pixel    :")
    print(np.array2string(calibration.robot_per_pixel, precision=6))
    print("matrix (3x3)       :")
    print(np.array2string(calibration.matrix, precision=6))
    print("true A = -H        :")
    print(np.array2string(arm.true_affine, precision=6))

    # Show numerically why the absolute affine translation is not reusable.
    section("why the absolute affine translation is not the command")
    images = np.array([sample.image_uv for sample in calibration.samples])
    poses = np.array([sample.robot_xy for sample in calibration.samples])
    design = np.column_stack([images, np.ones(len(images))])
    absolute, *_ = np.linalg.lstsq(design, poses, rcond=None)
    index = 0
    naive = design[index] @ absolute
    current = poses[index]
    correction = calibration.servo_delta(images[index])
    print(f"sample p           : {np.round(images[index], 2)}")
    print(f"pose r             : {np.round(current, 2)} mm")
    print(f"absolute affine(p) : {np.round(naive, 2)} mm  "
          f"(distance {np.linalg.norm(naive - current):.3f} mm -> no-op)")
    print(f"A(c - p)           : {np.round(correction, 2)} mm  "
          f"(distance {np.linalg.norm(correction):.1f} mm)")
    print("The reusable calibration is A; the fitted translation b belongs")
    print("to the calibration target's physical position.")

    section("four-point verification")
    arm.pose = np.array([104.0, 82.0, 50.0])
    report = verify_image_to_robot(
        robot, arm.capture, calibration, locator=locator, settle_s=0.0, checks=3
    )
    print(f"passed             : {report.passed}")
    print(f"mean / max error   : {report.mean_error_px:.3f} / "
          f"{report.max_error_px:.3f} px")
    print(f"mean error (mm)    : {report.mean_error_mm:.4f} mm")
    print(f"note               : {report.note or '-'}")

    section("camera-to-gripper offset")
    arm.pose = np.array([114.0, 82.0, 50.0])
    release = np.append(arm.true_grasp_pose(), 10.0)
    offset = calibrate_gripper_offset(
        robot, arm.capture, calibration, place_pose=release,
        locator=locator, settle_s=0.0,
    )
    print(f"offset             : {np.round(offset.offset_xy, 4)} mm")
    print(f"true K - G         : {np.round(arm.camera_offset - arm.gripper_offset, 4)} mm")
    print(f"centring error     : {offset.mean_center_error_px:.3f} px")
    print(f"valid              : {offset.valid}")
    print(f"warnings           : {offset.warnings}")

    recipe = RobotPickRecipe(calibration, offset, locator)
    section("one-move pick on a new target")
    arm.target_world = np.array([130.0, 70.0])
    arm.pose = np.array([120.0, 65.0, 50.0])
    detection = locator.find(arm.capture())
    print(f"target detection   : valid={detection.valid} "
          f"conf={detection.confidence:.3f} "
          f"centre={np.round(detection.center, 2)}")
    grasp = recipe.grasp_pose(arm.pose, detection.center)
    truth = arm.true_grasp_pose()
    print(f"commanded grasp    : {np.round(grasp, 3)} mm")
    print(f"true grasp         : {np.round(truth, 3)} mm")
    print(f"error              : {np.linalg.norm(grasp[:2] - truth):.4f} mm")
    print(f"expected uv@grasp  : {np.round(recipe.expected_uv_at_grasp(), 2)}")

    result = verify_pick(
        robot, arm.capture, recipe, locator=locator, settle_s=0.0,
        check_grasp_view=True,
    )
    print(f"verify_pick passed : {result['passed']}")
    if result.get("grasp_view"):
        view = result["grasp_view"]
        if view.get("valid"):
            print(f"grasp-view residual : {view['residual_px']:.3f} px")
        else:
            print(f"grasp-view          : {view.get('note')}")

    # Annotated detection image.
    arm.pose = np.array([114.0, 82.0, 50.0])
    arm.target_world = np.array([120.0, 90.0])
    annotated = locator.visualize(arm.capture(), locator.find(arm.capture()),
                                  draw_edges=True)
    cv2.imwrite(str(OUTPUT_DIR / "target_detection.png"), annotated)

    section("artifacts")
    calibration_path = OUTPUT_DIR / "image_robot_calibration.json"
    recipe_path = OUTPUT_DIR / "pick_recipe.json"
    calibration.save(str(calibration_path))
    recipe.save(str(recipe_path))
    print(f"calibration : {calibration_path}")
    print(f"recipe      : {recipe_path}")
    print(f"detection   : {OUTPUT_DIR / 'target_detection.png'}")
    summary = {
        "calibration_valid": calibration.valid,
        "offset_valid": offset.valid,
        "recipe_valid": recipe.validate()["valid"],
        "rms_px": calibration.rms_px,
        "grasp_error_mm": float(np.linalg.norm(grasp[:2] - truth)),
        "verify_pick_passed": bool(result["passed"]),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
