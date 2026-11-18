"""Replay of archived offset runs, including the affine model."""

from __future__ import annotations

import numpy as np

from measure.pick_artifacts import RunRecorder
from measure.pick_replay import replay
from measure.robot_calibration import CameraGripperOffset, OffsetSample

CENTRES = np.array([[0.0, 0.0], [200.0, 0.0], [0.0, 150.0], [200.0, 150.0],
                    [100.0, 0.0], [0.0, 75.0], [200.0, 75.0], [100.0, 150.0]])


def _samples(centres, places) -> list:
    return [
        {"center_pose": list(map(float, centre)),
         "place_pose": list(map(float, place))}
        for centre, place in zip(centres, places)
    ]


def test_affine_offset_run_replays(tmp_path):
    theta = np.radians(1.5)
    linear = 1.01 * np.array([[np.cos(theta), -np.sin(theta)],
                              [np.sin(theta), np.cos(theta)]])
    shift = np.array([8.0, 4.0])
    places = (np.column_stack([CENTRES, np.ones(len(CENTRES))]) @
              np.column_stack([linear, shift]).T)
    samples = _samples(CENTRES, places)

    offset = CameraGripperOffset.from_samples(
        [OffsetSample(place_pose=tuple(item["place_pose"]),
                      center_pose=tuple(item["center_pose"])) for item in samples])
    assert offset.effective_model == "affine"

    recorder = RunRecorder(tmp_path, "offset", {"samples": samples, "model": "auto"})
    recorder.finish({"matrix": offset.matrix.tolist(),
                     "offset_xy": list(offset.offset_xy)})

    assert replay(tmp_path / recorder.run_id)


def test_legacy_translation_run_still_replays(tmp_path):
    places = CENTRES + np.array([8.0, 4.0])
    samples = _samples(CENTRES, places)

    # A pre-matrix run: inputs carry no model, the result is the pure translation.
    recorder = RunRecorder(tmp_path, "offset", {"samples": samples})
    recorder.finish({"matrix": [[1.0, 0.0, 8.0],
                                [0.0, 1.0, 4.0],
                                [0.0, 0.0, 1.0]]})

    assert replay(tmp_path / recorder.run_id)