"""Pyramid configuration tests for ``TemplatePoint``.

``set_pyramid`` is the single place that recomputes ``_pyramid_scale``; these
tests pin the shared semantics so the GUI and the engine cannot drift: the
ratio is the more aggressive of the decimation level (``2**-decimate``) and the
template side cap (``pyramid_max_template_size / longest``), with the shortest
side clamped to at least 6 px.
"""

import numpy as np
import pytest

from measure.measure_template import TemplatePoint


@pytest.fixture()
def reference_image():
    rng = np.random.default_rng(7)
    return (rng.random((200, 240)) * 255).astype(np.uint8)


def make_point(image, decimate=0, max_size=400):
    return TemplatePoint(
        image, 100, 120, template_size=80,
        pyramid_decimate=decimate, pyramid_max_template_size=max_size,
    )


def test_disabled_pyramid_keeps_full_resolution(reference_image):
    point = make_point(reference_image, decimate=0, max_size=40)
    assert point.pyramid_decimate == 0
    assert point._pyramid_scale == 1.0


def test_enabled_pyramid_uses_size_cap(reference_image):
    point = make_point(reference_image, decimate=1, max_size=40)
    assert point._pyramid_scale == pytest.approx(40 / 80)


def test_level_number_bounds_the_ratio(reference_image):
    """The level caps the ratio even when the size cap would allow less."""
    two_x = make_point(reference_image, decimate=1, max_size=400)
    eight_x = make_point(reference_image, decimate=3, max_size=400)
    assert two_x._pyramid_scale == pytest.approx(0.5)
    assert eight_x._pyramid_scale == pytest.approx(0.125)


def test_size_cap_wins_when_it_is_more_aggressive(reference_image):
    point = make_point(reference_image, decimate=1, max_size=20)
    assert point._pyramid_scale == pytest.approx(20 / 80)


def test_cap_above_template_does_not_disable_the_level(reference_image):
    """A generous cap must not switch the pyramid off; the level still applies.

    This is the case that matters for models with no fixed template size: a
    cap that exceeds the support leaves the level as the only bound.
    """
    point = make_point(reference_image, decimate=2, max_size=400)
    assert point._pyramid_scale == pytest.approx(0.25)


def test_shortest_side_keeps_six_pixels(reference_image):
    point = make_point(reference_image, decimate=8, max_size=400)
    assert point._pyramid_scale * 80 >= 6 - 1e-9


def test_matches_the_engine_formula(reference_image):
    """The GUI path and the shared engine must agree for the same inputs."""
    from measure.computation_types import Space, make_mapping
    from measure.template_search import PreparedSupport, SearchOptions, _pyramid_factor

    space = Space("support", "px")
    for side in (80, 200, 400, 1200):
        pixels = np.ones((side, side), np.float32)
        prepared = PreparedSupport(
            pixels=pixels, valid_mask=np.ones(pixels.shape, bool),
            anchor_support=(side / 2, side / 2),
            support_to_reference=make_mapping(
                np.eye(3), source=space, destination=space, kind="similarity"),
            reference_space=space, support_space=space, mask=None,
        )
        for decimate in (1, 2, 3):
            for cap in (40, 128, 400):
                point = make_point(reference_image, decimate=decimate, max_size=cap)
                point._crop_h = point._crop_w = side
                expected = _pyramid_factor(
                    prepared, SearchOptions(pyramid_levels=decimate,
                                            pyramid_max_template_size=cap))
                assert point._compute_pyramid_scale() == pytest.approx(expected)


def test_set_pyramid_updates_scale_in_place(reference_image):
    point = make_point(reference_image, decimate=0, max_size=400)
    point.set_pyramid(1, 40)
    assert point.pyramid_decimate == 1
    assert point._pyramid_scale == pytest.approx(0.5)
    point.set_pyramid(0, 40)
    assert point._pyramid_scale == 1.0


def test_save_roundtrip_keeps_pyramid_scale(reference_image, tmp_path):
    point = make_point(reference_image, decimate=1, max_size=40)
    path = tmp_path / "template.npz"
    point.save(str(path))

    restored = TemplatePoint.from_file(str(path))
    assert restored.pyramid_decimate == 1
    assert restored.pyramid_max_template_size == 40
    assert restored._pyramid_scale == pytest.approx(point._pyramid_scale)


def test_deserialize_uses_same_formula(reference_image, tmp_path):
    point = make_point(reference_image, decimate=0, max_size=40)
    payload = point.to_dict() if hasattr(point, "to_dict") else None
    if payload is None:
        pytest.skip("TemplatePoint has no dict serialization")
    payload["pyramid_decimate"] = 1
    payload["pyramid_max_template_size"] = 40
    restored = TemplatePoint.from_dict(payload)
    assert restored._pyramid_scale == pytest.approx(0.5)
