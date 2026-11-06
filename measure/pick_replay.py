"""Offline replay for a gantry pick debug run.

Usage::

    python -m measure.pick_replay data/pick_runs/20261106-150834-vision-a1b2c3

The run directory is self-contained: original images, request inputs, the
recipe snapshot and the original result.  Replay re-runs detection and
calibration, compares the new result with ``result.json`` and exits non-zero on
a mismatch.  No HTTP service and no robot are required.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from measure.circle_locator import CircleLocator
from measure.pick_api import GantryPick
from measure.pick_artifacts import read_manifest
from measure.robot_calibration import (
    CameraGripperOffset,
    OffsetSample,
    RobotPickRecipe,
    fit_image_to_robot,
)


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return data


def _load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read image: {path}")
    return image


def _compare(
    label: str,
    expected: Any,
    actual: Any,
    *,
    tolerance: float = 1e-6,
) -> bool:
    if expected is None and actual is None:
        print(f"PASS {label}: both None")
        return True
    if expected is None or actual is None:
        print(f"FAIL {label}: expected={expected!r} actual={actual!r}")
        return False
    expected_array = np.asarray(expected, dtype=float).reshape(-1)
    actual_array = np.asarray(actual, dtype=float).reshape(-1)
    if expected_array.shape != actual_array.shape:
        print(f"FAIL {label}: shape {expected_array.shape} != {actual_array.shape}")
        return False
    difference = float(np.max(np.abs(expected_array - actual_array)))
    if difference <= tolerance:
        print(f"PASS {label}: max |diff|={difference:.3g}")
        return True
    print(f"FAIL {label}: max |diff|={difference:.6g}")
    return False


def _replay_vision(run_dir: Path, inputs: Dict[str, Any], result: Dict[str, Any],
                   recipe: Optional[Dict[str, Any]]) -> bool:
    points = inputs.get("robot_points")
    if not isinstance(points, list) or len(points) < 4:
        print("FAIL vision: inputs.json has no robot_points")
        return False
    manifest = read_manifest(run_dir)
    images = manifest.get("images") or []
    if len(images) != len(points):
        print("FAIL vision: image count does not match robot points")
        return False
    locator = CircleLocator()
    if recipe and isinstance(recipe.get("locator"), dict):
        locator = CircleLocator.from_dict(recipe["locator"])
    centres: List[Tuple[float, float]] = []
    confidences: List[float] = []
    for entry in images:
        image = _load_image(run_dir / entry["original"])
        detection = locator.find(image)
        print(
            f"  {entry['original']}: uv=({detection.center[0]:.3f}, "
            f"{detection.center[1]:.3f}) conf={detection.confidence:.3f} "
            f"valid={detection.valid}"
        )
        if not detection.valid:
            print(f"FAIL vision: target not found in {entry['original']}")
            return False
        centres.append((float(detection.center[0]), float(detection.center[1])))
        confidences.append(float(detection.confidence))
    first = _load_image(run_dir / images[0]["original"])
    height, width = first.shape[:2]
    calibration = fit_image_to_robot(
        centres,
        [point[:2] for point in points],
        image_size=(width, height),
        confidences=confidences,
    )
    print(f"  replay: valid={calibration.valid} rms_px={calibration.rms_px:.6f}")
    ok = True
    ok &= _compare("vision.rms_px", result.get("rms_px"), calibration.rms_px, tolerance=1e-5)
    expected_matrix = result.get("robot_per_pixel")
    if expected_matrix is not None:
        ok &= _compare(
            "vision.robot_per_pixel",
            expected_matrix,
            calibration.robot_per_pixel,
            tolerance=1e-8,
        )
    return bool(ok)


def _replay_offset(inputs: Dict[str, Any], result: Dict[str, Any]) -> bool:
    samples = inputs.get("samples")
    if not isinstance(samples, list) or not samples:
        print("FAIL offset: inputs.json has no samples")
        return False
    built: List[OffsetSample] = []
    for item in samples:
        place = item.get("place_pose") or []
        centre = item.get("center_pose") or []
        if len(place) < 2 or len(centre) < 2:
            print("FAIL offset: sample poses need at least two values")
            return False
        built.append(
            OffsetSample(
                place_pose=(float(place[0]), float(place[1])),
                center_pose=(float(centre[0]), float(centre[1])),
            )
        )
    offset = CameraGripperOffset.from_samples(built)
    print(f"  replay: offset={offset.offset_xy} std={offset.std_xy}")
    return _compare("offset.offset_xy", result.get("offset_xy"), offset.offset_xy)


def _replay_locate(run_dir: Path, inputs: Dict[str, Any], result: Dict[str, Any],
                   recipe: Optional[Dict[str, Any]]) -> bool:
    if not recipe:
        print("FAIL locate: recipe_snapshot.json is missing")
        return False
    pick = GantryPick.from_recipe(RobotPickRecipe.from_dict(recipe))
    image_info = inputs.get("image") or {}
    original = image_info.get("original") or "images/00_original"
    image_path = run_dir / original
    if not image_path.is_file():
        # Older runs may only record the filename; use the manifest entry.
        manifest = read_manifest(run_dir)
        entry = (manifest.get("images") or [{}])[0]
        image_path = run_dir / entry.get("original", original)
    image = _load_image(image_path)
    pose = inputs.get("robot_pose") or []
    if len(pose) < 2:
        print("FAIL locate: inputs.json has no robot_pose")
        return False
    location = pick.locate(image, (float(pose[0]), float(pose[1])))
    print(
        f"  replay: valid={location.valid} camera={location.camera_center_xy} "
        f"gripper={location.gripper_xy}"
    )
    ok = bool(location.valid == bool(result.get("valid", True)))
    ok &= _compare(
        "locate.camera_center_xy",
        result.get("camera_center_xy"),
        location.camera_center_xy,
        tolerance=1e-5,
    )
    ok &= _compare(
        "locate.gripper_xy",
        result.get("gripper_xy"),
        location.gripper_xy,
        tolerance=1e-5,
    )
    return bool(ok)


def replay(run_dir: Path) -> bool:
    """Replay one run directory and print a PASS/FAIL report."""
    run_dir = Path(run_dir)
    manifest = read_manifest(run_dir)
    inputs = _read_json(run_dir / manifest.get("inputs", "inputs.json"))
    result = _read_json(run_dir / manifest.get("result", "result.json"))
    recipe_path = manifest.get("recipe_snapshot")
    recipe = _read_json(run_dir / recipe_path) if recipe_path else None
    run_type = manifest.get("type")
    print(f"Replaying {run_dir.name} ({run_type}, status={manifest.get('status')})")
    if run_type == "vision":
        return _replay_vision(run_dir, inputs, result, recipe)
    if run_type == "offset":
        return _replay_offset(inputs, result)
    if run_type == "locate":
        return _replay_locate(run_dir, inputs, result, recipe)
    print(f"FAIL: unknown run type {run_type!r}")
    return False


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Replay a gantry pick debug run and compare it with result.json."
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Run directory, for example data/pick_runs/20261106-150834-vision-a1b2c3",
    )
    args = parser.parse_args(argv)
    try:
        passed = replay(args.run_dir)
    except Exception as exc:  # noqa: BLE001 - CLI should report, not traceback
        print(f"FAIL: {exc}")
        return 2
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
