"""Two-stage (ROI -> patch) capability of the v2 engine.

The v2 engine now shares :mod:`measure.multi_target_workflow` with the GUI and
the v1 API, which gives it the two-stage workflow:

  1. an oriented ROI is located on the whole image;
  2. measurement objects are defined in the straightened patch it produces;
  3. one result is returned **per detected target** (top-k).

These tests cover the capability itself, plus the contract details that make it
safe: an ROI is required before objects exist, object ids survive a
publish/measure round-trip, and the flat response shape is preserved.
"""

from __future__ import annotations

import numpy as np
import pytest

from measure_api.v2.application import EngineApplication, EngineError

# ROI box used by most tests
ROI_CENTER = (150.0, 180.0)
ROI_SIZE = (220.0, 260.0)
# With angle 0 the patch origin is (40, 50): patch (r, c) -> image (40+r, 50+c)
PATCH_ORIGIN = (40.0, 50.0)
EDGE_COL = 160            # hard vertical edge in the reference image
EDGE_PATCH_COL = EDGE_COL - PATCH_ORIGIN[1]


def _reference(seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = rng.integers(30, 80, (300, 360), dtype=np.uint8)
    image[:, EDGE_COL:] = 220
    return image


def _inspection(placements, shape=(520, 620)) -> np.ndarray:
    """Inspection image with copies of the ROI content pasted at placements."""
    rng = np.random.default_rng(9)
    image = rng.integers(30, 80, shape, dtype=np.uint8)
    ref = _reference()
    r1, c1 = int(PATCH_ORIGIN[0]), int(PATCH_ORIGIN[1])
    patch = ref[r1:r1 + int(ROI_SIZE[0]), c1:c1 + int(ROI_SIZE[1])]
    for (row, col) in placements:
        image[row:row + patch.shape[0], col:col + patch.shape[1]] = patch
    return image


def _roi(**overrides):
    roi = {
        "center": list(ROI_CENTER),
        "size": list(ROI_SIZE),
        "angle_deg": 0.0,
        "match_score_threshold": 0.3,
    }
    roi.update(overrides)
    return roi


def _edge(label: str, row: float) -> dict:
    return {
        "object_type": "EdgePoint",
        "label": label,
        "parameters": {
            "row": row,
            "col": EDGE_PATCH_COL,
            "angle": 0.0,
            "length1": 60.0,
            "length2": 4.0,
        },
    }


def _run(application, operation, timeout=30_000):
    return application.wait_operation(operation["operation_id"], timeout)


# ---------------------------------------------------------------------------
# ROI resource
# ---------------------------------------------------------------------------


class TestRoiResource:
    def test_session_starts_without_roi(self, tmp_path):
        application = EngineApplication(tmp_path)
        try:
            session = application.create_modeling_session(_reference())
            assert session["roi"] is None
            assert application.get_modeling_roi(session["session_id"]) is None
        finally:
            application.close()

    def test_set_and_get_roi(self, tmp_path):
        application = EngineApplication(tmp_path)
        try:
            session = application.create_modeling_session(_reference())
            sid = session["session_id"]
            stored = application.set_modeling_roi(sid, _roi())

            assert stored["center"] == list(ROI_CENTER)
            assert stored["size"] == list(ROI_SIZE)
            assert stored["alignment_mode"] == "single_box"
            assert stored["roi_state"]["box_center"] == list(ROI_CENTER)
            assert application.get_modeling_roi(sid) == stored
            assert application.get_modeling_session(sid)["roi"] == stored
        finally:
            application.close()

    def test_objects_require_an_roi_first(self, tmp_path):
        application = EngineApplication(tmp_path)
        try:
            session = application.create_modeling_session(_reference())
            with pytest.raises(EngineError) as excinfo:
                application.create_object(session["session_id"], _edge("e", 60.0))
            assert excinfo.value.error["code"] == "session.roi_required"
        finally:
            application.close()

    @pytest.mark.parametrize(
        "roi, field",
        [
            ({"center": [1.0]}, "center"),
            ({"size": [0.0, 10.0]}, "size"),
            ({"angle_step": 0.0}, "angle_step"),
            ({"alignment_mode": "sideways"}, "alignment_mode"),
            ({"nonsense": 1}, "field_unknown"),
        ],
    )
    def test_invalid_roi_is_rejected(self, tmp_path, roi, field):
        application = EngineApplication(tmp_path)
        try:
            session = application.create_modeling_session(_reference())
            payload = _roi()
            payload.update(roi)
            with pytest.raises(EngineError) as excinfo:
                application.set_modeling_roi(session["session_id"], payload)
            assert "roi" in excinfo.value.error["code"] or field == "field_unknown"
        finally:
            application.close()


# ---------------------------------------------------------------------------
# Two-stage measurement
# ---------------------------------------------------------------------------


class TestTwoStageMeasurement:
    def test_patch_coordinates_map_back_to_image(self, tmp_path):
        """A tool defined in patch space reports image-space coordinates."""
        application = EngineApplication(tmp_path)
        try:
            session = application.create_modeling_session(_reference())
            sid = session["session_id"]
            application.set_modeling_roi(sid, _roi())
            application.create_object(sid, _edge("e1", 60.0))

            completed = _run(application, application.preview(sid))
            assert completed["status"] == "succeeded"
            observation = completed["result"]["objects"][0]["observation"]
            # patch row 60 -> image row 100; the edge sits at image col 160
            assert observation["row"] == pytest.approx(100.0, abs=1.0)
            assert observation["col"] == pytest.approx(EDGE_COL, abs=1.0)
        finally:
            application.close()

    def test_composed_object_resolves_in_patch_space(self, tmp_path):
        application = EngineApplication(tmp_path)
        try:
            session = application.create_modeling_session(_reference())
            sid = session["session_id"]
            application.set_modeling_roi(sid, _roi())
            first = application.create_object(sid, _edge("e1", 60.0))
            second = application.create_object(sid, _edge("e2", 140.0))
            application.create_object(
                sid,
                {
                    "object_type": "TwoPointsDistance",
                    "label": "gap",
                    "parameters": {},
                    "input_object_ids": [first["object_id"], second["object_id"]],
                },
            )

            completed = _run(application, application.preview(sid))
            by_label = {o["label"]: o for o in completed["result"]["objects"]}
            assert by_label["gap"]["valid"]
            assert by_label["gap"]["observation"]["value"] == pytest.approx(80.0, abs=1.0)
        finally:
            application.close()

    def test_topk_returns_one_instance_per_target(self, tmp_path):
        application = EngineApplication(tmp_path)
        try:
            session = application.create_modeling_session(_reference())
            sid = session["session_id"]
            application.set_modeling_roi(sid, _roi())
            application.create_object(sid, _edge("e1", 60.0))

            placements = [(int(PATCH_ORIGIN[0]), int(PATCH_ORIGIN[1])),
                          (int(PATCH_ORIGIN[0]) + 200, int(PATCH_ORIGIN[1]) + 300)]
            completed = _run(application, application.preview(sid, _inspection(placements)))

            result = completed["result"]
            assert len(result["instances"]) == 2
            assert [i["index"] for i in result["instances"]] == [1, 2]
            for instance in result["instances"]:
                assert set(instance) >= {
                    "index", "score", "rotation_deg", "scale",
                    "center_row", "center_col", "valid", "measurements",
                }
                assert [m["label"] for m in instance["measurements"]] == ["e1"]
            # the flat view mirrors the primary instance
            assert len(result["objects"]) == 1
            assert result["objects"][0]["label"] == "e1"
        finally:
            application.close()

    def test_no_detection_yields_empty_instances(self, tmp_path):
        application = EngineApplication(tmp_path)
        try:
            session = application.create_modeling_session(_reference())
            sid = session["session_id"]
            application.set_modeling_roi(sid, _roi())
            application.create_object(sid, _edge("e1", 60.0))

            blank = np.zeros((520, 620), dtype=np.uint8)
            completed = _run(application, application.preview(sid, blank))
            assert completed["status"] == "succeeded"
            assert completed["result"]["instances"] == []
            assert completed["result"]["objects"][0]["valid"] is False
        finally:
            application.close()


# ---------------------------------------------------------------------------
# Publish / measure round-trip
# ---------------------------------------------------------------------------


class TestPublishMeasure:
    def test_object_ids_survive_publish_and_measure(self, tmp_path):
        application = EngineApplication(tmp_path)
        try:
            session = application.create_modeling_session(_reference())
            sid = session["session_id"]
            application.set_modeling_roi(sid, _roi())
            created = application.create_object(sid, _edge("e1", 60.0))

            published = _run(application, application.publish(sid, "two-stage"))
            assert published["status"] == "succeeded"

            measured = _run(
                application,
                application.measure(
                    "two-stage", published["result"]["asset_ref"], _reference()
                ),
            )
            assert measured["status"] == "succeeded"
            assert measured["result"]["objects"][0]["object_id"] == created["object_id"]

            # the ROI is part of the asset, so the patch frame is restored
            observation = measured["result"]["objects"][0]["observation"]
            assert observation["row"] == pytest.approx(100.0, abs=1.0)
            assert observation["col"] == pytest.approx(EDGE_COL, abs=1.0)
        finally:
            application.close()

    def test_asset_uses_the_shared_manifest_format(self, tmp_path):
        import json

        application = EngineApplication(tmp_path)
        try:
            session = application.create_modeling_session(_reference())
            sid = session["session_id"]
            application.set_modeling_roi(sid, _roi())
            application.create_object(sid, _edge("e1", 60.0))
            published = _run(application, application.publish(sid, "manifest-model"))

            asset_dir = application._asset_dir("manifest-model")
            assert {p.name for p in asset_dir.iterdir()} >= {
                "manifest.json", "reference.npy", "workflow.npz",
            }
            manifest = json.loads((asset_dir / "manifest.json").read_text())
            assert manifest["contract_version"] == "3.0"
            assert manifest["calibration_model_id"] == "manifest-model"
            assert manifest["alignment"]["roi_state"]["box_center"] == list(ROI_CENTER)
            # the ROI definition is recoverable from the asset
            assert manifest["alignment"]["box_size"] == list(ROI_SIZE)
            assert manifest["objects"][0]["label"] == "e1"
        finally:
            application.close()

    def test_republish_after_import_keeps_the_patch_frame(self, tmp_path):
        application = EngineApplication(tmp_path)
        try:
            session = application.create_modeling_session(_reference())
            sid = session["session_id"]
            application.set_modeling_roi(sid, _roi())
            application.create_object(sid, _edge("e1", 60.0))
            published = _run(application, application.publish(sid, "origin"))
            exported = _run(
                application,
                application.export_model("origin", published["result"]["asset_ref"]),
            )
            package = application.read_export(exported["result"]["download_ref"])

            imported = _run(application, application.import_model(package))
            imported_sid = imported["result"]["session_id"]
            # the imported session is editable and already has the ROI
            assert application.get_modeling_roi(imported_sid) is not None

            republished = _run(application, application.publish(imported_sid, "imported"))
            measured = _run(
                application,
                application.measure(
                    "imported", republished["result"]["asset_ref"], _reference()
                ),
            )
            assert measured["status"] == "succeeded"
            observation = measured["result"]["objects"][0]["observation"]
            assert observation["row"] == pytest.approx(100.0, abs=1.0)
        finally:
            application.close()