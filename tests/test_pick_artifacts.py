"""Tests for debug run artifacts and annotation."""

from __future__ import annotations

import json
import zipfile
import cv2
import numpy as np

from measure.circle_locator import CircleDetection, CircleLocator
from measure.pick_artifacts import (
    RunRecorder,
    list_runs,
    read_manifest,
    render_annotation,
    validate_run_id,
    zip_run,
)


def _image_and_bytes(centre=(300, 200), radius=45):
    image = np.full((480, 640), 210, dtype=np.uint8)
    cv2.circle(image, centre, radius, 40, -1)
    image = cv2.GaussianBlur(image, (0, 0), 1.5)
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), buffer.tobytes()


def test_run_recorder_layout_and_manifest(tmp_path):
    image, data = _image_and_bytes()
    detection = CircleLocator().find(image)
    assert detection.valid

    recorder = RunRecorder(
        tmp_path,
        "vision",
        {"robot_points": [[1.0, 2.0]]},
        annotated_max_px=800,
    )
    entry = recorder.add_image(
        0,
        "corner.png",
        data,
        image,
        detection,
        robot_xy=(1.0, 2.0),
    )
    recorder.set_residual(0, 0.02)
    recorder.build_overview()
    manifest = recorder.finish({"valid": True, "rms_px": 0.02})

    run_dir = tmp_path / manifest["run_id"]
    assert validate_run_id(manifest["run_id"])
    assert (run_dir / "manifest.json").is_file()
    assert (run_dir / "inputs.json").is_file()
    assert (run_dir / "result.json").is_file()
    assert (run_dir / "README.txt").is_file()
    assert (run_dir / "overview.jpg").is_file()
    assert (run_dir / "images" / "00_original.png").is_file()
    assert (run_dir / "images" / "00_annotated.jpg").is_file()
    assert entry["annotated"] == "images/00_annotated.jpg"
    assert entry["residual_px"] == 0.02
    assert manifest["status"] == "ok"
    assert manifest["type"] == "vision"
    assert manifest["images"][0]["sha256"]

    annotated = cv2.imread(str(run_dir / entry["annotated"]))
    assert annotated is not None
    assert max(annotated.shape[:2]) <= 800

    reread = read_manifest(run_dir)
    assert reread["run_id"] == manifest["run_id"]
    summaries = list_runs(tmp_path)
    assert summaries[0]["run_id"] == manifest["run_id"]
    assert summaries[0]["overview"] == "overview.jpg"


def test_run_recorder_failed_status(tmp_path):
    image, data = _image_and_bytes()
    recorder = RunRecorder(tmp_path, "locate", {"robot_pose": [1.0, 2.0]})
    recorder.add_image(
        0,
        "shot.png",
        data,
        image,
        None,
        name="image",
        robot_xy=(1.0, 2.0),
        note="no target",
        error="no_target",
    )
    manifest = recorder.finish(
        {"valid": False, "reason": "no_target"},
        status="failed",
        error={"code": "no_target", "message": "target not found"},
    )
    run_dir = tmp_path / manifest["run_id"]
    assert manifest["status"] == "failed"
    assert manifest["error"]["code"] == "no_target"
    assert (run_dir / "images" / "image_annotated.jpg").is_file()
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    assert result["valid"] is False


def test_annotation_handles_invalid_detection():
    image, _ = _image_and_bytes()
    annotated = render_annotation(
        image,
        CircleDetection.empty("no_candidates"),
        index=2,
        robot_xy=(1.0, 2.0),
        max_px=640,
    )
    assert annotated.shape[2] == 3
    assert max(annotated.shape[:2]) <= 640


def test_zip_run_contains_manifest(tmp_path):
    image, data = _image_and_bytes()
    recorder = RunRecorder(tmp_path, "offset", {"samples": []})
    recorder.finish({"offset_xy": [8.0, 4.0]})
    archive = zip_run(tmp_path / recorder.run_id)
    try:
        with zipfile.ZipFile(archive) as bundle:
            names = set(bundle.namelist())
    finally:
        archive.unlink()
    assert "manifest.json" in names
    assert "inputs.json" in names
    assert "result.json" in names
