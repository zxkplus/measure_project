"""Teaching-time fit check: the template must be placeable in the search area.

The engine no longer pads, so a pose whose footprint leaves the frame is
skipped. A support as wide as the frame therefore matches only at zero rotation,
and that is invisible until a part arrives rotated. These tests pin the check
that is supposed to catch it before the model is saved.
"""

import math

import pytest

from measure.measure_template import check_support_fits_search_area


def test_small_support_fits_at_any_angle():
    issues = check_support_fits_search_area(
        (5472, 3648), (314, 314), (math.radians(-180), math.radians(180)))
    assert issues == []


def test_support_larger_than_the_area_is_reported():
    issues = check_support_fits_search_area((400, 500), (600, 600))
    kinds = {issue["kind"] for issue in issues}
    assert kinds == {"support_too_large"}
    assert {issue["limb"] for issue in issues} == {"height", "width"}


def test_support_as_wide_as_the_frame_fails_only_when_rotated():
    fit = (5472, 3648)
    support = (5138, 3648)
    assert check_support_fits_search_area(fit, support, (0.0, 0.0)) == []
    rotated = check_support_fits_search_area(
        fit, support, (math.radians(-15), math.radians(15)))
    assert rotated, "a frame-wide support cannot rotate"
    assert all(issue["kind"] == "rotation_too_large" for issue in rotated)


def test_the_envelope_uses_the_critical_angle_not_just_the_extremes():
    # A 100x10 support needs 100.5 px of height at 5.7 degrees -- slightly more
    # than it needs at either end of a 90 degree range -- so an area that fits
    # both extremes can still be too short.
    support = (100, 10)
    fits_extremes_only = check_support_fits_search_area(
        (100, 101), support, (0.0, math.radians(90)))
    assert fits_extremes_only, "the critical angle's footprint must be checked"
    assert all(issue["kind"] == "rotation_too_large" for issue in fits_extremes_only)
    roomy = check_support_fits_search_area(
        (102, 102), support, (0.0, math.radians(90)))
    assert roomy == []


def test_margin_reduces_the_available_area():
    assert check_support_fits_search_area((100, 100), (90, 90)) == []
    assert check_support_fits_search_area((100, 100), (90, 90), margin=6.0)


def test_negative_margin_is_ignored():
    assert check_support_fits_search_area((100, 100), (90, 90), margin=-50.0) == []


def test_issue_carries_a_readable_message():
    issues = check_support_fits_search_area((400, 500), (600, 600))
    assert issues
    for issue in issues:
        assert set(issue) >= {"kind", "limb", "required", "actual", "message"}
        assert issue["message"]
        assert issue["actual"] > issue["required"]
