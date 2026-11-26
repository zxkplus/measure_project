"""Backend selection and NCC parity tests for the optional tm_ref path."""

from __future__ import annotations

import numpy as np
import pytest

from measure import template_backend


def test_legacy_backend_is_the_default(monkeypatch):
    monkeypatch.delenv("MEASURE_TEMPLATE_BACKEND", raising=False)
    assert template_backend.selected_backend() == "legacy"
    assert template_backend.resolved_backend() == "legacy"


def test_unknown_backend_falls_back_to_legacy(monkeypatch):
    monkeypatch.setenv("MEASURE_TEMPLATE_BACKEND", "not-a-backend")
    assert template_backend.selected_backend() == "legacy"


def test_tm_ref_correlate_matches_legacy(monkeypatch):
    pytest.importorskip("tm_ref")
    rng = np.random.default_rng(31)
    search = rng.integers(0, 255, (90, 120), dtype=np.uint8)
    template = rng.integers(0, 255, (23, 29), dtype=np.uint8)
    mask = rng.random(template.shape) > 0.2

    monkeypatch.setenv("MEASURE_TEMPLATE_BACKEND", "legacy")
    expected = template_backend.correlate(search, template, mask)

    monkeypatch.setenv("MEASURE_TEMPLATE_BACKEND", "tm_ref")
    actual = template_backend.correlate(search, template, mask)

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_tm_ref_build_info_reports_availability(monkeypatch):
    pytest.importorskip("tm_ref")
    monkeypatch.setenv("MEASURE_TEMPLATE_BACKEND", "auto")
    info = template_backend.build_info()
    assert info["selected"] == "auto"
    assert info["resolved"] == "tm_ref"
    assert info["tm_ref_available"] is True
    assert info["tm_ref_version"]