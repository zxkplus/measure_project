"""Golden regression for fine-search locality on a long support.

``tests/data/template-precision/tooth`` holds a four-times decimated crop of the
2026-11-09 tooth-outer capture (frame-05-best.png) and its portable template.
The full image measures 1975x487 support against a 4096x3072 frame; here it is
1975/4 x 487/4 against 4096/4 x 3072/4 (1024x768). The decimated template was
rebuilt with ``/template-builds`` from the decimated image, so this fixture is
fully synthetic and contains no external asset dependency.

Any fine-search change must keep the located center, anchor and both mappings
within ``1e-6`` px of the values recorded here.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from measure_api.v2.app import create_app
from measure_api.v2.families.templates import TemplateSearch, _scan, _variant
from measure_api.v2.models import ServiceSettings

DATA = Path(__file__).parent / "data" / "template-precision" / "tooth"
GOLDEN = json.loads((DATA / "golden.json").read_text(encoding="utf-8"))
TEMPLATE_NAME = "template.npz"
IMAGE_NAME = "frame-sc.png"
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 768
SEARCH = {
    "score_threshold": 0.7, "max_matches": 1, "overlap": 0.3, "subpixel": True,
    "angle_min": -10, "angle_max": 10, "angle_step": 2,
    "scale_min": 1, "scale_max": 1, "scale_step": 0.1,
    "coarse_fine": False, "coarse_score_threshold": 0.5,
    "coarse_angle_step": 5, "coarse_scale_step": 0.1,
    "pyramid_levels": 2, "pyramid_max_template_size": 64,
}


def ref(path: str) -> dict:
    return {"root": "exchange", "path": path}


@pytest.fixture(autouse=True)
def _fixture_files(tmp_path):
    """Stage the checked-in decimated frame and portable template as exchange inputs."""
    shutil.copy(DATA / IMAGE_NAME, tmp_path / IMAGE_NAME)
    shutil.copy(DATA / TEMPLATE_NAME, tmp_path / TEMPLATE_NAME)


@pytest.fixture
def client(tmp_path):
    settings = ServiceSettings(shared_roots={"exchange": tmp_path}, max_workers=1)
    with TestClient(create_app(settings)) as value:
        yield value


def locate(client, override: dict) -> dict:
    search = dict(SEARCH, **override)
    result = client.post("/api/v2/template-matches", json={
        "image": ref(IMAGE_NAME), "template": ref(TEMPLATE_NAME), "search": search,
    }).json()["result"]
    assert result["valid"], result
    return result


@pytest.mark.parametrize("policy", ["whole", "neighborhood"])
def test_long_support_localization_matches_golden(client, policy):
    result = locate(client, {"fine_search_policy": policy})
    match = result["matches"][0]
    assert len(result["matches"]) == 1
    assert match["center"] == pytest.approx(GOLDEN["center"], abs=1e-6)
    assert match["anchor"] == pytest.approx(GOLDEN["anchor"], abs=1e-6)
    assert match["support_mapping"]["matrix"][0][2] == pytest.approx(
        GOLDEN["support_dx"], abs=1e-6)
    assert match["support_mapping"]["matrix"][1][2] == pytest.approx(
        GOLDEN["support_dy"], abs=1e-6)
    assert match["angle_degrees"] == pytest.approx(GOLDEN["angle"], abs=1e-9)
    assert result["diagnostics"]["pyramid_scale"] == pytest.approx(
        GOLDEN["pyramid_scale"], abs=1e-12)


def test_neighborhood_policy_shrinks_every_fine_window(client):
    result = locate(client, {"fine_search_policy": "neighborhood"})
    fine = result["diagnostics"]["fine_search"]
    assert fine["policy"] == "neighborhood"
    assert fine["windows"]
    # Every window is the coarse center +/- the half diagonal radius, clipped to the search area.
    half = fine["neighborhood_radius"]
    for window in fine["windows"]:
        assert window["width"] <= min(2 * half + 1, IMAGE_WIDTH)
        assert window["height"] <= min(2 * half + 1, IMAGE_HEIGHT)
    # The legacy policy rescans the whole search area; the neighborhood policy must be smaller.
    whole = locate(client, {"fine_search_policy": "whole"})["diagnostics"]["fine_search"]
    assert whole["windows"][0]["width"] == IMAGE_WIDTH
    assert all(window["width"] <= 2 * fine["neighborhood_radius"] + 1
               for window in fine["windows"])
    # The radius follows the support half diagonal, not the image; every window is a strict subset.
    assert 2 * fine["neighborhood_radius"] + 1 < IMAGE_WIDTH
    assert all(window["height"] <= 2 * fine["neighborhood_radius"] + 1
               for window in fine["windows"])


def test_whole_policy_keeps_legacy_diagnostics(client):
    result = locate(client, {"fine_search_policy": "whole"})
    fine = result["diagnostics"]["fine_search"]
    assert fine["policy"] == "whole"
    assert fine["neighborhood_radius"] is None


def test_declared_scale_reaches_template_side():
    """Guard the golden fixture: 1975x487 decimated by four."""
    with np.load(DATA / TEMPLATE_NAME) as archive:
        pixels = archive["pixels"]
    assert pixels.shape == tuple(GOLDEN["support_shape"])

def _synthetic(client, tmp_path):
    """Teach a small support and place it twice, far apart, in a wide image."""
    rng = np.random.default_rng(7)
    patch = rng.integers(10, 240, (16, 24), dtype=np.uint8)
    reference = np.zeros((80, 120), np.uint8)
    reference[20:36, 30:54] = patch
    cv2.imwrite(str(tmp_path / "ref.png"), reference)
    asset = client.post("/api/v2/template-builds", json={
        "image": ref("ref.png"),
        "roi": {"center": {"x": 41.5, "y": 27.5}, "width": 24, "height": 16},
        "output": {"directory": ref("templates")},
    }).json()["result"]["template"]
    image = np.zeros((120, 400), np.uint8)
    image[40:56, 40:64] = patch
    image[40:56, 300:324] = patch
    cv2.imwrite(str(tmp_path / "inspection.png"), image)
    return asset


def test_rotation_variant_larger_than_window_is_reported_not_clipped():
    """A pose variant that cannot fit its window is skipped, never clipped into a peak."""
    rng = np.random.default_rng(7)
    patch = rng.integers(10, 240, (16, 24), dtype=np.uint8)
    search = rng.integers(0, 255, (120, 400)).astype(np.float32)
    options = TemplateSearch(angle_min=-40, angle_max=40, angle_step=80,
                             fine_search_expand_px=0)
    assert _variant(patch, 0.0, 1.0, (400, 120))[3] == (24, 16)
    assert _variant(patch, 40.0, 1.0, (400, 120))[3] == (29, 28)
    skipped = []
    # A 26 px window cannot hold the 29x28 rotated variant.
    found = _scan(search[:26, :26], patch, options, 40.0, 1.0, (0, 0), skipped=skipped)
    assert found == []
    assert skipped and skipped[0]["required"] == [29, 28]
    wider = []
    _scan(search[:31, :31], patch, options, 40.0, 1.0, (0, 0), skipped=wider)
    assert wider == []
