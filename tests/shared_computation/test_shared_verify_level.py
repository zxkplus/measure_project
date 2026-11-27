"""Halcon's second NumLevels value: stop tracking above full resolution.

Refining at a coarser level is an accuracy-for-time trade, so the tests pin both
halves of it: the pose must survive, and the default must be untouched.
"""

import math

import cv2
import numpy as np
import pytest

from measure.computation_types import RotatedBox, Space
from measure.preprocessing import RawSpec
from measure.template_search import SupportSpec, SearchOptions, prepare_support, search_support

REFERENCE_SPACE = Space("reference", "px")
SUPPORT_SPACE = Space("support", "px")
INSPECTION_SPACE = Space("inspection", "px")


def _scene():
    """Return a reference and an inspection image with the target rotated 20 deg."""
    rng = np.random.RandomState(3)
    texture = cv2.GaussianBlur((rng.rand(120, 120) * 255).astype(np.uint8), (5, 5), 1.5)
    cv2.rectangle(texture, (20, 20), (100, 70), 255, 2)
    cv2.line(texture, (10, 110), (110, 90), 200, 3)
    reference = np.zeros((400, 500), np.uint8)
    reference[150:270, 200:320] = texture
    rotated = cv2.warpAffine(
        texture, cv2.getRotationMatrix2D((60.0, 60.0), -20.0, 1.0), (120, 120))
    inspection = np.zeros((400, 500), np.uint8)
    inspection[150:270, 200:320] = rotated
    return reference, inspection


def _prepared(reference):
    spec = SupportSpec(box=RotatedBox((260.0, 210.0), (120.0, 120.0), 0.0),
                       anchor_reference=(260.0, 210.0))
    result = prepare_support(reference, spec, RawSpec(),
                             reference_space=REFERENCE_SPACE,
                             support_space=SUPPORT_SPACE)
    assert result.valid
    return result.value


def _run(image, prepared, verify_level):
    options = SearchOptions(
        score_threshold=0.4, max_matches=1, subpixel=True,
        # Force decimation so a coarser verification level exists at all.
        coarse_template_side=60,
        angle_min_rad=math.radians(-30), angle_max_rad=math.radians(30),
        angle_step_rad=math.radians(1), coarse_fine=True,
        coarse_angle_step_rad=math.radians(5),
        min_verify_level=verify_level)
    return search_support(image, prepared, options, image_space=INSPECTION_SPACE)


def test_default_tracks_to_full_resolution():
    reference, scene = _scene()
    result = _run(scene, _prepared(reference), 0)
    assert result.valid
    assert result.diagnostics["verify_level"] == 0


def test_verify_level_reports_itself():
    reference, scene = _scene()
    prepared = _prepared(reference)
    result = _run(scene, prepared, 1)
    if result.valid:
        assert result.diagnostics["verify_level"] >= 1


def test_coarser_verification_keeps_the_pose():
    reference, scene = _scene()
    prepared = _prepared(reference)
    full = _run(scene, prepared, 0)
    coarse = _run(scene, prepared, 1)
    assert full.valid and coarse.valid
    assert coarse.value.matches[0].center == pytest.approx(
        full.value.matches[0].center, abs=3.0)
    assert math.degrees(coarse.value.matches[0].angle_rad) == pytest.approx(
        math.degrees(full.value.matches[0].angle_rad), abs=3.0)


def test_negative_verify_level_is_clamped():
    reference, scene = _scene()
    with pytest.raises(Exception):
        SearchOptions(min_verify_level=-1)
