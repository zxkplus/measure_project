"""Tests for :mod:`measure.preview` — the preprocessing preview layer.

These are pure-function tests: no GUI, no assets on disk.  They pin the two
properties the GUI depends on:

* display normalisation is predictable per preprocessor output scale, and
* enhancement is applied to the *already cropped* region, never to the full
  source image and then cropped (the two orders differ materially for CLAHE).
"""

from __future__ import annotations

import numpy as np
import pytest

from measure.measure_template import (
    CLAHEPreprocessor,
    RawPreprocessor,
    ThresholdPreprocessor,
    preprocessor_from_config,
)
from measure.preview import (
    PREPROCESSOR_TYPES,
    default_configs,
    needs_normalization,
    preview_matrix,
    preview_pair,
    to_display_uint8,
)


@pytest.fixture()
def texture() -> np.ndarray:
    """A deterministic 120x140 grayscale image with structure."""
    rng = np.random.default_rng(20261113)
    base = np.linspace(40, 210, 140, dtype=np.float32)[None, :]
    noise = rng.normal(0.0, 6.0, (120, 140)).astype(np.float32)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


class TestDisplayConversion:
    def test_uint8_passes_through(self):
        image = np.arange(12, dtype=np.uint8).reshape(3, 4)
        out = to_display_uint8(image)
        assert out.dtype == np.uint8
        assert np.array_equal(out, image)
        assert not needs_normalization(image)

    def test_intensity_float_is_cast_not_stretched(self):
        image = np.linspace(4.0, 255.0, 12, dtype=np.float32).reshape(3, 4)
        out = to_display_uint8(image)
        assert out.dtype == np.uint8
        assert out.min() == 4
        assert out.max() == 255
        assert not needs_normalization(image)

    def test_wide_float_is_stretched(self):
        image = np.linspace(0.0, 447.0, 12, dtype=np.float32).reshape(3, 4)
        out = to_display_uint8(image)
        assert out.dtype == np.uint8
        assert out.min() == 0
        assert out.max() == 255
        assert needs_normalization(image)

    def test_negative_values_are_stretched(self):
        image = np.linspace(-5.0, 200.0, 12, dtype=np.float32).reshape(3, 4)
        assert needs_normalization(image)
        out = to_display_uint8(image)
        assert out.min() == 0 and out.max() == 255

    def test_constant_image_does_not_divide_by_zero(self):
        image = np.full((4, 4), 300.0, dtype=np.float32)  # needs stretching, but flat
        out = to_display_uint8(image)
        assert out.dtype == np.uint8
        assert np.array_equal(out, np.zeros((4, 4), dtype=np.uint8))


class TestPreviewPair:
    def test_template_is_enhanced_after_cropping(self, texture):
        patch = texture[10:50, 20:80].copy()
        clahe = CLAHEPreprocessor(2.0, (8, 8))

        template_result, _ = preview_pair(patch, None, clahe)

        expected = to_display_uint8(clahe(patch))
        assert np.array_equal(template_result.image, expected)
        assert template_result.source == "template"
        assert template_result.stats["shape"] == [40, 60]

    def test_crop_first_differs_from_enhance_first(self, texture):
        """Guards the ordering rule: the two orders are not interchangeable."""
        patch = texture[10:50, 20:80].copy()
        clahe = CLAHEPreprocessor(2.0, (8, 8))

        crop_first, _ = preview_pair(patch, None, clahe)
        enhance_first = to_display_uint8(clahe(texture)[10:50, 20:80])

        assert not np.array_equal(crop_first.image, enhance_first)
        assert float(np.abs(crop_first.image.astype(int) - enhance_first.astype(int)).max()) > 2

    def test_missing_inputs_yield_none(self, texture):
        template_result, inspection_result = preview_pair(None, None, RawPreprocessor())
        assert template_result is None
        assert inspection_result is None

    def test_search_region_limits_the_enhanced_window(self, texture):
        clahe = CLAHEPreprocessor(2.0, (8, 8))
        _, inspection_result = preview_pair(
            None, texture, clahe, search_region=(0, 40, 0, 60)
        )
        assert inspection_result.stats["shape"] == [40, 60]

    def test_threshold_reports_uniform_output(self):
        patch = np.full((20, 20), 200, dtype=np.uint8)
        _, _ = preview_pair(patch, None, ThresholdPreprocessor(128.0))
        template_result, _ = preview_pair(patch, None, ThresholdPreprocessor(128.0))
        assert template_result.stats["uniform"] is True
        assert template_result.image.min() == template_result.image.max() == 255

    def test_stats_describe_the_displayed_image(self, texture):
        patch = texture[0:30, 0:30].copy()
        template_result, _ = preview_pair(patch, None, RawPreprocessor())
        stats = template_result.stats
        assert stats["mean"] == pytest.approx(float(template_result.image.mean()), abs=0.01)
        assert stats["std"] == pytest.approx(float(template_result.image.std()), abs=0.01)
        # RawPreprocessor emits float32; the display copy is uint8.
        assert stats["source_dtype"] == "float32"
        assert template_result.image.dtype == np.uint8
        assert stats["elapsed_ms"] >= 0.0


class TestPreviewMatrix:
    def test_one_column_per_builtin_type(self, texture):
        patch = texture[10:50, 20:80].copy()
        columns = preview_matrix(patch, texture)
        assert [c.name for c in columns] == list(PREPROCESSOR_TYPES)
        for column in columns:
            assert column.template is not None
            assert column.inspection is not None
            assert column.config["type"] == column.name

    def test_explicit_configs_are_honoured(self, texture):
        patch = texture[10:50, 20:80].copy()
        columns = preview_matrix(
            patch, texture, configs=[{"type": "clahe", "clip_limit": 4.0,
                                     "tile_grid_size": [4, 4]}]
        )
        assert len(columns) == 1
        assert columns[0].config["clip_limit"] == 4.0


class TestDefaultConfigs:
    def test_every_default_config_round_trips(self):
        configs = default_configs()
        assert [c["type"] for c in configs] == list(PREPROCESSOR_TYPES)
        for config in configs:
            rebuilt = preprocessor_from_config(dict(config))
            assert rebuilt.serialize()["type"] == config["type"]

    def test_defaults_are_fresh_copies(self):
        first = default_configs()
        first[0]["type"] = "mutated"
        assert default_configs()[0]["type"] == "raw"

    def test_clahe_defaults_match_the_engine(self):
        config = next(c for c in default_configs() if c["type"] == "clahe")
        assert config["clip_limit"] == 2.0
        assert list(config["tile_grid_size"]) == [8, 8]