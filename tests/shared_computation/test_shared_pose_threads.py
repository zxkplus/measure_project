"""Concurrent pose evaluation must not change a single recorded value.

Pose variants differ only by the support's rotation and share the search window,
so the engine may evaluate them in parallel. The merge is in pose order, which is
what makes the result independent of scheduling; these tests pin that rather than
trusting it.
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
    """Return a reference and an inspection image holding one rotated target."""
    rng = np.random.RandomState(11)
    texture = cv2.GaussianBlur((rng.rand(100, 100) * 255).astype(np.uint8), (5, 5), 1.5)
    cv2.circle(texture, (50, 50), 25, 0, -1)
    cv2.rectangle(texture, (5, 5), (95, 95), 255, 2)
    reference = np.zeros((360, 480), np.uint8)
    reference[140:240, 190:290] = texture
    rotated = cv2.warpAffine(
        texture, cv2.getRotationMatrix2D((50.0, 50.0), -25.0, 1.0), (100, 100))
    inspection = np.zeros((360, 480), np.uint8)
    inspection[140:240, 190:290] = rotated
    return reference, inspection


def _prepared(reference):
    spec = SupportSpec(box=RotatedBox((240.0, 190.0), (100.0, 100.0), 0.0),
                       anchor_reference=(240.0, 190.0))
    result = prepare_support(reference, spec, RawSpec(),
                             reference_space=REFERENCE_SPACE,
                             support_space=SUPPORT_SPACE)
    assert result.valid
    return result.value


def _run(image, prepared, threads):
    options = SearchOptions(
        score_threshold=0.4, max_matches=1, subpixel=True,
        angle_min_rad=math.radians(-45), angle_max_rad=math.radians(45),
        angle_step_rad=math.radians(1), coarse_fine=False,
        search_threads=threads)
    return search_support(image, prepared, options, image_space=INSPECTION_SPACE)


def test_serial_and_parallel_searches_agree_exactly():
    reference, scene = _scene()
    prepared = _prepared(reference)
    serial = _run(scene, prepared, 1)
    parallel = _run(scene, prepared, 8)
    assert serial.valid and parallel.valid
    left = serial.value.matches[0]
    right = parallel.value.matches[0]
    assert left.center == right.center
    assert left.anchor == right.anchor
    assert left.angle_rad == right.angle_rad
    assert left.score == right.score
    assert (serial.diagnostics["searched_variant_count"]
            == parallel.diagnostics["searched_variant_count"])


def test_thread_count_never_changes_the_match_count():
    reference, scene = _scene()
    prepared = _prepared(reference)
    counts = {threads: len(_run(scene, prepared, threads).value.matches)
              for threads in (1, 2, 4, 16)}
    assert len(set(counts.values())) == 1, counts


def test_zero_means_let_the_engine_decide():
    """0 is the automatic setting, not "no threads"."""
    from measure.template_search import search_threads

    reference, scene = _scene()
    prepared = _prepared(reference)
    automatic = _run(scene, prepared, 0)
    explicit = _run(scene, prepared, search_threads())
    assert automatic.value.matches[0].center == pytest.approx(
        explicit.value.matches[0].center, abs=1e-9)
