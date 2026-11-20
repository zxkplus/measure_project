"""Search-cost controls, process caches and counters stay numerically neutral."""

import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

from measure_api.v2.app import create_app
from measure_api.v2.families.templates import (
    TemplateSearch, default_fine_candidates, search_threads,
)
from measure_api.v2.files import SharedFiles
from measure_api.v2.models import ServiceSettings


def ref(path):
    return {"root": "exchange", "path": path}


@pytest.fixture
def client(tmp_path):
    settings = ServiceSettings(shared_roots={"exchange": tmp_path}, max_workers=1)
    with TestClient(create_app(settings)) as value:
        yield value


def teach(client, tmp_path, patch, reference=None):
    reference = reference if reference is not None else 120
    image = np.zeros((reference, reference), np.uint8)
    image[30:30 + patch.shape[0], 40:40 + patch.shape[1]] = patch
    import cv2
    cv2.imwrite(str(tmp_path / "reference.png"), image)
    result = client.post("/api/v2/template-builds", json={
        "image": ref("reference.png"),
        "roi": {"center": {"x": 40 + (patch.shape[1] - 1) / 2,
                           "y": 30 + (patch.shape[0] - 1) / 2},
                "width": patch.shape[1], "height": patch.shape[0]},
        "preprocessing": {"type": "raw"},
        "output": {"directory": ref("templates")},
    })
    assert result.status_code == 200, result.text
    return result.json()["result"]["artifacts"][0]["file"]["path"]


def match(client, template, search, image_name="inspection.png"):
    response = client.post("/api/v2/template-matches", json={
        "template": ref(template), "image": ref(image_name), "search": search,
    })
    assert response.status_code == 200, response.text
    return response.json()


def write_inspection(tmp_path, patches, shape=(160, 200)):
    import cv2
    image = np.zeros(shape, np.uint8)
    for row, column, patch in patches:
        image[row:row + patch.shape[0], column:column + patch.shape[1]] = patch
    cv2.imwrite(str(tmp_path / "inspection.png"), image)


def test_fine_candidates_default_is_unbounded_and_opt_in(monkeypatch):
    monkeypatch.delenv("MEASURE_V2_TEMPLATE_FINE_CANDIDATES", raising=False)
    assert default_fine_candidates() is None
    assert TemplateSearch(max_matches=1).resolved_fine_candidates() is None
    monkeypatch.setenv("MEASURE_V2_TEMPLATE_FINE_CANDIDATES", "2")
    assert default_fine_candidates() == 2
    assert TemplateSearch(max_matches=5).resolved_fine_candidates() == 2
    monkeypatch.setenv("MEASURE_V2_TEMPLATE_FINE_CANDIDATES", "all")
    assert default_fine_candidates() is None
    assert TemplateSearch(max_matches=0).resolved_fine_candidates() is None
    assert TemplateSearch(max_matches=2, fine_candidates=1).resolved_fine_candidates() == 1


def test_search_threads_is_bounded(monkeypatch):
    # An explicit value is honoured exactly, bounded to 1..16, whatever machine runs this.
    for value, expected in (("1", 1), ("4", 4), ("99", 16), ("0", 1)):
        monkeypatch.setenv("MEASURE_V2_SEARCH_THREADS", value)
        assert search_threads() == expected
    monkeypatch.delenv("MEASURE_V2_SEARCH_THREADS", raising=False)
    default = search_threads()
    assert 1 <= default <= 16
    assert default == min(16, os.cpu_count())
    # An unusable value is treated as unconfigured, not as "single thread".
    monkeypatch.setenv("MEASURE_V2_SEARCH_THREADS", "all")
    assert search_threads() == default


def test_repeated_targets_bounded_refinement_matches_full_refinement(client, tmp_path):
    rng = np.random.default_rng(5)
    patch = rng.integers(20, 230, (16, 24), np.uint8)
    template = teach(client, tmp_path, patch)
    write_inspection(tmp_path, [
        (20, 20, patch), (100, 60, patch), (30, 150, patch),
    ])
    search = {"score_threshold": 0.7, "max_matches": 1, "pyramid_levels": 0}
    bounded = match(client, template, dict(search, fine_candidates=1))
    full = match(client, template, dict(search, fine_candidates=1000))
    assert bounded["result"]["valid"] and full["result"]["valid"]
    assert bounded["result"]["matches"] == full["result"]["matches"]


def test_search_plan_and_counters_describe_the_work(client, tmp_path):
    rng = np.random.default_rng(11)
    patch = rng.integers(20, 230, (16, 24), np.uint8)
    template = teach(client, tmp_path, patch)
    write_inspection(tmp_path, [(20, 20, patch), (100, 60, patch)])
    first = match(client, template, {"score_threshold": 0.7, "max_matches": 1,
                                     "pyramid_levels": 1, "pyramid_max_template_size": 8})
    plan = first["result"]["diagnostics"]["search_plan"]
    assert plan["resample"] in {"blur", "none"}
    assert plan["coarse_poses"] >= 1
    assert plan["neighborhoods_available"] >= plan["neighborhoods_refined"] >= 1
    assert plan["fine_candidates"] is None
    counters = first["diagnostics"]["timing"]["counters"]
    assert counters["ncc_evaluations"] >= 1
    assert counters["fine_neighborhoods"] >= 1
    assert counters["template_cache_miss"] == 1
    assert counters.get("template_cache_hit", 0) == 0
    second = match(client, template, {"score_threshold": 0.7, "max_matches": 1,
                                      "pyramid_levels": 1, "pyramid_max_template_size": 8})
    assert second["diagnostics"]["timing"]["counters"]["template_cache_hit"] == 1


def test_threaded_pose_grid_reproduces_single_thread_result(tmp_path, monkeypatch):
    rng = np.random.default_rng(23)
    patch = rng.integers(20, 230, (18, 26), np.uint8)
    results = []
    for threads in ("1", "4", None):
        if threads is None:
            monkeypatch.delenv("MEASURE_V2_SEARCH_THREADS", raising=False)
        else:
            monkeypatch.setenv("MEASURE_V2_SEARCH_THREADS", threads)
        settings = ServiceSettings(shared_roots={"exchange": tmp_path}, max_workers=1)
        with TestClient(create_app(settings)) as client:
            template = teach(client, tmp_path, patch, reference=140)
            write_inspection(tmp_path, [(25, 30, patch), (95, 120, patch)], shape=(180, 220))
            response = match(client, template, {
                "score_threshold": 0.6, "max_matches": 2, "pyramid_levels": 1,
                "pyramid_max_template_size": 12, "angle_min": -4, "angle_max": 4,
                "angle_step": 1,
            })
            results.append(response["result"]["matches"])
    assert len(results[0]) == len(results[1])
    for left, right in zip(results[0], results[1]):
        assert left["center"] == pytest.approx(right["center"], abs=1e-9)
        assert left["score"] == pytest.approx(right["score"], abs=1e-9)
        assert left["angle_degrees"] == pytest.approx(right["angle_degrees"], abs=1e-9)


def test_decoded_cache_rejects_a_corrupt_blob(tmp_path):
    file_contract = SharedFiles(ServiceSettings(
        shared_roots={"exchange": tmp_path},
        cache_max_bytes=64 * 1024 * 1024, cache_max_entries=8,
        state_directory=tmp_path / "state",
    ))
    directory = tmp_path / "state" / "decoded-cache"
    directory.mkdir(parents=True)
    path = directory / ("a" * 64 + ".cache")
    image = np.arange(64, dtype=np.uint8).reshape(8, 8)
    file_contract._write_cached(directory, path, image)
    loaded = SharedFiles._read_cached(path)
    assert loaded is not None and np.array_equal(loaded, image)
    payload = bytearray(path.read_bytes())
    payload[40] ^= 0xFF
    path.write_bytes(bytes(payload))
    assert SharedFiles._read_cached(path) is None
    path.write_bytes(b"broken")
    assert SharedFiles._read_cached(path) is None