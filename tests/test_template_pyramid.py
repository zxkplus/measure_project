"""Pyramid configuration tests for ``TemplatePoint``.

``set_pyramid`` is the single place that recomputes ``_pyramid_scale``; these
tests pin the historical semantics (any non-zero ``decimate`` enables the
coarse pyramid, the ratio comes from ``pyramid_max_template_size``) so the GUI
and the model loaders can share it.
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


def test_level_number_does_not_change_the_ratio(reference_image):
    two_x = make_point(reference_image, decimate=1, max_size=40)
    eight_x = make_point(reference_image, decimate=3, max_size=40)
    assert two_x._pyramid_scale == eight_x._pyramid_scale


def test_cap_above_template_means_no_downsampling(reference_image):
    point = make_point(reference_image, decimate=2, max_size=400)
    assert point._pyramid_scale == 1.0


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
