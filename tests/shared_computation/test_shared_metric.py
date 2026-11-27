"""Halcon's Metric: contrast-polarity tolerance, and its two traps.

`ignore_global_polarity` turns the score into a magnitude, which makes a
photographic negative of the part match too. Two things are easy to get wrong:
the magnitude must be taken before the invalid-score sentinel (otherwise -1
becomes a perfect 1), and every stage of a search must carry the policy (a
refinement that forgets it silently undoes the coarse stage's tolerance).
"""

import math

import cv2
import numpy as np
import pytest

from measure.computation_types import RotatedBox, Space
from measure.preprocessing import RawSpec
from measure.template_search import (
    METRIC_VALUES,
    SearchOptions,
    SupportSpec,
    prepare_support,
    search_support,
)

REFERENCE_SPACE = Space("reference", "px")
SUPPORT_SPACE = Space("support", "px")
INSPECTION_SPACE = Space("inspection", "px")


def _target(size=100):
    """Return an asymmetric, high-contrast patch."""
    rng = np.random.RandomState(5)
    texture = cv2.GaussianBlur((rng.rand(size, size) * 255).astype(np.uint8), (5, 5), 1.5)
    cv2.circle(texture, (size // 2, size // 2), size // 4, 0, -1)
    cv2.rectangle(texture, (5, 5), (size - 5, size - 5), 255, 2)
    return texture


def _scene(invert=False, top=150, left=200, size=100):
    reference = np.zeros((400, 500), np.uint8)
    reference[top:top + size, left:left + size] = _target(size)
    scene = np.full((400, 500), 0, np.uint8) if not invert else np.full((400, 500), 255, np.uint8)
    patch = _target(size) if not invert else 255 - _target(size)
    scene[top:top + size, left:left + size] = patch
    return reference, scene


def _prepared(reference, center=(250.0, 200.0), size=(100.0, 100.0)):
    spec = SupportSpec(box=RotatedBox(center, size, 0.0), anchor_reference=center)
    result = prepare_support(reference, spec, RawSpec(),
                             reference_space=REFERENCE_SPACE,
                             support_space=SUPPORT_SPACE)
    assert result.valid
    return result.value


def _search(image, prepared, algorithm, metric):
    options = SearchOptions(
        score_threshold=0.5, max_matches=1, subpixel=True,
        algorithm=algorithm, metric=metric,
        angle_min_rad=0.0, angle_max_rad=0.0, angle_step_rad=math.radians(1.0))
    return search_support(image, prepared, options, image_space=INSPECTION_SPACE)


@pytest.mark.parametrize("algorithm", ["ncc", "shape"])
def test_inverted_part_needs_ignore_global_polarity(algorithm):
    reference, _ = _scene()
    _, inverted = _scene(invert=True)
    prepared = _prepared(reference)

    strict = _search(inverted, prepared, algorithm, "use_polarity")
    assert not strict.valid, "an inverted part must not match under use_polarity"

    tolerant = _search(inverted, prepared, algorithm, "ignore_global_polarity")
    assert tolerant.valid
    match = tolerant.value.matches[0]
    assert match.center == pytest.approx((250.0, 200.0), abs=1.5)
    assert match.score > 0.9


@pytest.mark.parametrize("algorithm", ["ncc", "shape"])
def test_upright_part_still_matches_with_polarity_tolerance(algorithm):
    reference, scene = _scene()
    prepared = _prepared(reference)
    result = _search(scene, prepared, algorithm, "ignore_global_polarity")
    assert result.valid
    assert result.value.matches[0].center == pytest.approx((250.0, 200.0), abs=1.5)


def test_use_polarity_is_the_default():
    assert SearchOptions().metric == "use_polarity"
    assert SearchOptions().polarity_invariant is False
    assert SearchOptions(metric="ignore_global_polarity").polarity_invariant is True


def test_metric_values_match_halcon():
    assert METRIC_VALUES == ("use_polarity", "ignore_global_polarity")


def test_unknown_metric_is_rejected():
    with pytest.raises(Exception):
        SearchOptions(metric="ignore_polarity")


def test_magnitude_is_taken_before_the_invalid_score_sentinel():
    # A constant search area makes every correlation non-finite, which the score
    # map marks with -1. Taking the magnitude afterwards would turn that -1 into
    # a perfect 1 and report a match in an image with no edges at all.
    reference, _ = _scene()
    prepared = _prepared(reference)
    blank = np.zeros((400, 500), np.uint8)
    for algorithm in ("ncc", "shape"):
        result = _search(blank, prepared, algorithm, "ignore_global_polarity")
        assert not result.valid, f"{algorithm} matched a featureless image"
