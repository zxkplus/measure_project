"""Multi-point camera-to-gripper offset: translation, affine and their gates."""

from __future__ import annotations

import numpy as np
import pytest

from measure.robot_calibration import (
    CalibrationError,
    CameraGripperOffset,
    OffsetSample,
    combine_offsets,
)


def _samples(centres, places) -> list:
    return [
        OffsetSample(place_pose=tuple(float(value) for value in place),
                     center_pose=tuple(float(value) for value in centre))
        for centre, place in zip(centres, places)
    ]


def _affine(angle_deg: float, scale: float, shift) -> np.ndarray:
    theta = np.radians(angle_deg)
    matrix = np.eye(3)
    matrix[:2, :2] = scale * np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    matrix[:2, 2] = shift
    return matrix


def _apply(matrix: np.ndarray, centres: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([centres, np.ones(len(centres))])
    return (homogeneous @ matrix.T)[:, :2]


TRANSLATION_CENTRES = np.array(
    [[0.0, 0.0], [200.0, 0.0], [0.0, 150.0], [200.0, 150.0],
     [100.0, 0.0], [0.0, 75.0], [200.0, 75.0], [100.0, 150.0]]
)


# ---------------------------------------------------------------------------
# Translation stays translation
# ---------------------------------------------------------------------------


def test_single_sample_reproduces_the_historical_offset():
    offset = CameraGripperOffset.from_samples(
        _samples([[110.0, 85.0]], [[118.0, 89.0]]))

    assert offset.effective_model == "translation"
    assert offset.offset_xy == pytest.approx((8.0, 4.0), abs=1e-12)
    assert np.array_equal(offset.matrix, np.array(
        [[1.0, 0.0, 8.0], [0.0, 1.0, 4.0], [0.0, 0.0, 1.0]]))


def test_translation_only_data_ignores_the_sample_count():
    centres = TRANSLATION_CENTRES
    places = centres + np.array([8.0, 4.0])
    one = CameraGripperOffset.from_samples(_samples(centres[:1], places[:1]))
    many = CameraGripperOffset.from_samples(_samples(centres, places))

    assert str(many.effective_model) == "translation"
    assert np.array_equal(one.matrix, many.matrix)


def test_noisy_translation_data_stays_translation():
    rng = np.random.default_rng(7)
    centres = TRANSLATION_CENTRES
    places = centres + np.array([8.0, 4.0]) + rng.normal(0.0, 0.05, centres.shape)

    offset = CameraGripperOffset.from_samples(_samples(centres, places))

    assert offset.effective_model == "translation"
    assert np.array_equal(offset.matrix[:2, :2], np.eye(2))
    assert offset.offset_xy == pytest.approx(
        tuple(np.mean(places - centres, axis=0)), abs=1e-12)
    assert offset.valid


def test_explicit_translation_model_never_returns_an_affine():
    matrix = _affine(3.0, 1.0, (8.0, 4.0))
    places = _apply(matrix, TRANSLATION_CENTRES)

    offset = CameraGripperOffset.from_samples(
        _samples(TRANSLATION_CENTRES, places), model="translation")

    assert offset.effective_model == "translation"
    assert np.array_equal(offset.matrix[:2, :2], np.eye(2))
    assert not offset.valid
    assert "offset_inconsistent" in offset.warnings


# ---------------------------------------------------------------------------
# A real affine relation is recovered
# ---------------------------------------------------------------------------


def test_affine_relation_is_recovered():
    matrix = _affine(1.5, 1.01, (8.0, 4.0))
    places = _apply(matrix, TRANSLATION_CENTRES)

    offset = CameraGripperOffset.from_samples(_samples(TRANSLATION_CENTRES, places))

    assert offset.effective_model == "affine"
    assert offset.valid
    assert offset.rotation_deg == pytest.approx(1.5, abs=1e-6)
    assert offset.scale == pytest.approx(1.01, abs=1e-9)
    assert offset.shear == pytest.approx(0.0, abs=1e-9)
    assert offset.offset_xy == pytest.approx((8.0, 4.0), abs=1e-9)
    assert offset.rms_mm <= 1e-9
    assert np.allclose(offset.matrix, matrix, atol=1e-9)


def test_affine_survives_small_noise():
    rng = np.random.default_rng(11)
    matrix = _affine(-2.0, 0.995, (-12.0, 6.0))
    centres = TRANSLATION_CENTRES + np.array([400.0, 250.0])
    places = _apply(matrix, centres) + rng.normal(0.0, 0.05, centres.shape)

    offset = CameraGripperOffset.from_samples(_samples(centres, places))

    assert offset.effective_model == "affine"
    assert offset.rotation_deg == pytest.approx(-2.0, abs=0.2)
    assert offset.scale == pytest.approx(0.995, abs=0.002)
    assert offset.offset_xy == pytest.approx((-12.0, 6.0), abs=0.1)
    assert offset.rms_mm < 0.2


def test_four_samples_are_accepted_with_a_redundancy_warning():
    centres = TRANSLATION_CENTRES[:4]
    places = _apply(_affine(2.0, 1.0, (8.0, 4.0)), centres)

    offset = CameraGripperOffset.from_samples(_samples(centres, places))

    assert offset.effective_model == "affine"
    assert "offset_low_redundancy" in offset.warnings


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


def test_affine_needs_four_samples():
    centres = TRANSLATION_CENTRES[:3]
    places = centres + np.array([8.0, 4.0])
    with pytest.raises(CalibrationError):
        CameraGripperOffset.from_samples(_samples(centres, places), model="affine")


def test_collinear_samples_cannot_identify_an_affine():
    centres = np.array([[0.0, 0.0], [50.0, 50.0], [100.0, 100.0], [150.0, 150.0]])
    places = centres + np.array([8.0, 4.0])
    with pytest.raises(CalibrationError):
        CameraGripperOffset.from_samples(_samples(centres, places), model="affine")


def test_mirrored_relation_is_rejected():
    centres = TRANSLATION_CENTRES
    matrix = np.eye(3)
    matrix[0, 0] = -1.0
    places = _apply(matrix, centres)
    with pytest.raises(CalibrationError):
        CameraGripperOffset.from_samples(_samples(centres, places), model="affine")


def test_implausible_rotation_is_rejected():
    centres = TRANSLATION_CENTRES
    places = _apply(_affine(30.0, 1.0, (8.0, 4.0)), centres)
    with pytest.raises(CalibrationError):
        CameraGripperOffset.from_samples(_samples(centres, places), model="affine")


def test_unknown_model_is_rejected():
    with pytest.raises(CalibrationError):
        CameraGripperOffset.from_samples(
            _samples([[110.0, 85.0]], [[118.0, 89.0]]), model="rigid")


# ---------------------------------------------------------------------------
# Application, serialisation and combination
# ---------------------------------------------------------------------------


def test_apply_uses_the_matrix():
    matrix = _affine(5.0, 1.0, (8.0, 4.0))
    offset = CameraGripperOffset.from_samples(
        _samples(TRANSLATION_CENTRES, _apply(matrix, TRANSLATION_CENTRES)))
    assert offset.effective_model == "affine"

    pose = offset.apply((10.0, 20.0, -3.0))

    expected = offset.matrix @ np.array([10.0, 20.0, 1.0])
    assert pose[:2] == pytest.approx(expected[:2], abs=1e-12)
    # A translation-only application would differ, so this pins the matrix path.
    assert pose[:2] != pytest.approx((10.0 + offset.offset_xy[0],
                                      20.0 + offset.offset_xy[1]), abs=1e-6)
    assert pose[2] == pytest.approx(-3.0)


def test_serialisation_roundtrip_keeps_the_affine():
    matrix = _affine(1.0, 1.005, (8.0, 4.0))
    offset = CameraGripperOffset.from_samples(
        _samples(TRANSLATION_CENTRES, _apply(matrix, TRANSLATION_CENTRES)))

    restored = CameraGripperOffset.from_dict(offset.to_dict())

    assert restored.effective_model == "affine"
    assert np.allclose(restored.matrix, offset.matrix, atol=1e-12)
    assert restored.requested_model == offset.requested_model


def test_legacy_payload_reads_as_translation():
    payload = {
        "schema": "measure.camera_gripper_offset",
        "version": 1,
        "offset_xy": [8.0, 4.0],
        "valid": True,
        "samples": [],
    }

    offset = CameraGripperOffset.from_dict(payload)

    assert offset.effective_model == "translation"
    assert offset.is_translation
    assert offset.apply((100.0, 200.0))[:2] == pytest.approx((108.0, 204.0))


def test_combine_offsets_pools_samples():
    first = CameraGripperOffset.from_samples(
        _samples(TRANSLATION_CENTRES[:4], TRANSLATION_CENTRES[:4] + [8.0, 4.0]))
    second = CameraGripperOffset.from_samples(
        _samples(TRANSLATION_CENTRES[4:], TRANSLATION_CENTRES[4:] + [8.0, 4.0]))

    combined = combine_offsets([first, second])

    assert combined.effective_model == "translation"
    assert combined.offset_xy == pytest.approx((8.0, 4.0), abs=1e-12)
    assert len(combined.samples) == len(TRANSLATION_CENTRES)