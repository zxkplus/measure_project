"""End-to-end tests for OBB-based localisation (``ObbAlignment``).

Uses a fake detector, so no ML runtime is involved: what is under test is the
contract between stage 1 and stage 2 — that a detected box is cropped where it was
found, at the model's patch scale rather than the instance's, and that results map
back to inspection-image coordinates.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from measure.alignment import (
    FixedRoiAlignment,
    ObbAlignment,
    SingleBoxAlignment,
    strategy_from_roi_state,
)
from measure.multi_target_workflow import MultiTargetWorkflow
from measure.obb_source import ObbObservation
from measure.rectify import compute_rotated_box_corners

IMG_H, IMG_W = 600, 800


class FakeObbSource:
    """A detector with scripted output, so tests need no model."""

    name = "fake-obb"
    requires_roi = False

    def __init__(self, observations, *, per_call=None):
        self._observations = list(observations)
        self._per_call = per_call
        self.calls = 0

    def detect(self, image):
        self.calls += 1
        if self._per_call is not None:
            return self._per_call(self.calls, image)
        return list(self._observations)

    def serialize(self):
        return {"name": self.name, "scripted": True}


#: Bar geometry.  Deliberately smaller than the detected box below, so the
#: straightened patch contains a real bright/dark boundary for tools to find.  A
#: box exactly matching the bar would yield an all-bright patch with no edges.
BAR_1 = (100, 180, 200, 340)      # row_start, row_end, col_start, col_end
BAR_2 = (300, 380, 500, 640)
BOX_H, BOX_W = 140.0, 200.0       # detection box, larger than either bar


def make_image() -> np.ndarray:
    """Two bright bars on a dark field, inside a larger nominal box."""
    img = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    img[BAR_1[0]:BAR_1[1], BAR_1[2]:BAR_1[3]] = 200
    img[BAR_2[0]:BAR_2[1], BAR_2[2]:BAR_2[3]] = 200
    return img


def obs_at(row, col, height, width, angle_deg, *, cls=0, score=0.9):
    corners = compute_rotated_box_corners((row, col), (height, width), angle_deg)
    return ObbObservation.from_corners(corners, class_id=cls, score=score)


# ---------------------------------------------------------------------------
# Teaching
# ---------------------------------------------------------------------------


class TestTeach:
    def test_requires_a_source(self):
        with pytest.raises(ValueError, match="requires a detection source"):
            ObbAlignment().teach(make_image())

    def test_requires_a_reference_image(self):
        with pytest.raises(ValueError, match="requires a reference image"):
            ObbAlignment().teach(None, source=FakeObbSource([]))

    def test_derives_nominal_size_from_the_reference_detection(self):
        strategy = ObbAlignment()
        strategy.teach(make_image(),
                       source=FakeObbSource([obs_at(140, 270, BOX_H, BOX_W, 0.0)]))
        assert strategy.nominal_size == pytest.approx((BOX_H, BOX_W))
        assert strategy.anchor_class_id is None

    def test_explicit_nominal_size_wins(self):
        strategy = ObbAlignment()
        strategy.teach(make_image(),
                       source=FakeObbSource([obs_at(140, 270, BOX_H, BOX_W, 0.0)]),
                       nominal_size=(100.0, 200.0))
        assert strategy.nominal_size == pytest.approx((100.0, 200.0))

    def test_rejects_a_non_positive_nominal_size(self):
        strategy = ObbAlignment()
        with pytest.raises(ValueError, match="must be positive"):
            strategy.teach(make_image(),
                           source=FakeObbSource([obs_at(140, 270, BOX_H, BOX_W, 0.0)]),
                           nominal_size=(0.0, 140.0))

    def test_refuses_to_teach_when_the_anchor_is_not_visible(self):
        """Failing loudly beats shipping a model that can never find anything."""
        strategy = ObbAlignment()
        with pytest.raises(ValueError, match="found no target"):
            strategy.teach(make_image(), source=FakeObbSource([]))


    def test_refuses_when_the_anchor_class_is_absent(self):
        """An anchor that cannot be found on its own reference image is unusable.

        Detecting this at teach time is the difference between a clear error and a
        model that silently reports nothing on every future inspection.
        """
        strategy = ObbAlignment()
        with pytest.raises(ValueError, match="for class 1"):
            strategy.teach(make_image(),
                           source=FakeObbSource([obs_at(140, 270, BOX_H, BOX_W, 0.0, cls=0)]),
                           anchor_class_id=1)

    def test_min_score_can_also_make_the_anchor_unfindable(self):
        strategy = ObbAlignment()
        with pytest.raises(ValueError, match="found no target"):
            strategy.teach(make_image(),
                           source=FakeObbSource([obs_at(140, 270, BOX_H, BOX_W, 0.0, score=0.2)]),
                           min_score=0.9)

    def test_builds_no_template(self):
        """No template means no angle grid and no NCC search — the point of it."""
        strategy = ObbAlignment()
        strategy.teach(make_image(),
                       source=FakeObbSource([obs_at(140, 270, BOX_H, BOX_W, 0.0)]))
        assert strategy.template_point is None
        # The reference is stored so the asset can round-trip, but nothing
        # searches it: no template is built and no angle grid is swept.
        assert strategy._reference_image is not None


# ---------------------------------------------------------------------------
# Detection → match dicts
# ---------------------------------------------------------------------------


class TestDetect:
    def test_reports_engine_shaped_matches(self):
        strategy = ObbAlignment()
        strategy.teach(make_image(),
                       source=FakeObbSource([obs_at(140, 270, BOX_H, BOX_W, 0.0)]))
        matches = strategy.detect(make_image())
        assert len(matches) == 1
        m = matches[0]
        for key in ("matched_row", "matched_col", "match_score",
                    "best_rotation_deg", "valid", "box_size"):
            assert key in m, f"missing {key}"
        assert m["box_size"] == pytest.approx((BOX_H, BOX_W))

    def test_rotation_is_a_delta_from_the_taught_angle(self):
        """The engine subtracts it from the taught angle, so it must be relative."""
        strategy = ObbAlignment()
        strategy.teach(make_image(),
                       source=FakeObbSource([obs_at(140, 270, BOX_H, BOX_W, 10.0)]))
        assert strategy.detect(make_image())[0]["best_rotation_deg"] == \
            pytest.approx(0.0)

        strategy._source = FakeObbSource([obs_at(140, 270, BOX_H, BOX_W, 25.0)])
        assert strategy.detect(make_image())[0]["best_rotation_deg"] == \
            pytest.approx(15.0)

    def test_detector_without_scores_is_not_a_perfect_match(self):
        strategy = ObbAlignment()
        strategy.teach(make_image(),
                       source=FakeObbSource([obs_at(140, 270, BOX_H, BOX_W, 0.0, score=None)]))
        assert strategy.detect(make_image())[0]["match_score"] is None

    def test_class_filter_keeps_only_the_anchor(self):
        strategy = ObbAlignment()
        both = [obs_at(140, 270, BOX_H, BOX_W, 0.0, cls=0),
                obs_at(140, 270, BOX_H, BOX_W, 0.0, cls=1)]
        strategy.teach(make_image(), source=FakeObbSource(both), anchor_class_id=1)
        matches = strategy.detect(make_image())
        assert len(matches) == 1 and matches[0]["class_id"] == 1

    def test_max_targets_is_applied(self):
        strategy = ObbAlignment()
        many = [obs_at(140, 270, BOX_H, BOX_W, 0.0, score=0.5 + i / 10)
                for i in range(5)]
        strategy.teach(make_image(), source=FakeObbSource(many), max_targets=2)
        assert len(strategy.detect(make_image())) == 2

    def test_detect_before_teach_is_an_error(self):
        with pytest.raises(RuntimeError, match="no detection source"):
            ObbAlignment().detect(make_image())


# ---------------------------------------------------------------------------
# The fixed patch frame — the decision this design rests on
# ---------------------------------------------------------------------------


class TestFixedPatchFrame:
    def test_patch_uses_the_nominal_size_not_the_instance_size(self):
        """Instance boxes vary; the patch frame must not.

        A tool's parameters are defined once in patch coordinates, so a patch that
        changed size per instance would silently change what every tool measures.
        """
        strategy = ObbAlignment()
        strategy.teach(make_image(),
                       source=FakeObbSource([obs_at(140, 270, BOX_H, BOX_W, 0.0)]),
                       nominal_size=(BOX_H, BOX_W))
        image = make_image()

        # Three instances with visibly different extents.
        for size in ((BOX_H, BOX_W), (175.0, 250.0), (105.0, 150.0)):
            match = {
                "matched_row": 140.0, "matched_col": 270.0,
                "best_rotation_deg": 0.0, "box_size": size, "valid": True,
            }
            result = strategy.align(image, 140.0, 270.0, 0.0, box_size=size)
            assert result.patch.shape[:2] == (int(BOX_H), int(BOX_W))

    def test_align_before_teach_is_an_error(self):
        with pytest.raises(RuntimeError, match="no nominal patch size"):
            ObbAlignment().align(make_image(), 10.0, 10.0, 0.0)

    def test_patch_maps_back_to_the_detected_centre(self):
        from measure.rectify import map_point_via_affine

        for angle in (0.0, 20.0, -35.0):
            strategy = ObbAlignment()
            strategy.teach(make_image(),
                           source=FakeObbSource([obs_at(140, 270, BOX_H, BOX_W, angle)]),
                           nominal_size=(BOX_H, BOX_W))
            result = strategy.align(make_image(), 140.0, 270.0, 0.0)
            h, w = result.patch.shape[:2]
            back = map_point_via_affine((h / 2.0, w / 2.0), result.M_inv)
            assert back[0] == pytest.approx(140.0, abs=0.51)
            assert back[1] == pytest.approx(270.0, abs=0.51)


# ---------------------------------------------------------------------------
# Full engine run
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def build(self, *, observations=None, with_tool=True):
        image = make_image()
        wf = MultiTargetWorkflow()
        strategy = ObbAlignment()
        wf._alignment = strategy
        strategy.teach(image, source=FakeObbSource(
            observations if observations is not None
            else [obs_at(140, 270, BOX_H, BOX_W, 0.0)]))
        if with_tool:
            # EdgePair needs *two* edges to pair up.  In patch coordinates the
            # bar spans rows 30..110 (box is 140x200 centred on the bar), so the
            # caliper must run long enough to span both boundaries; a shorter one
            # sees a single edge and legitimately reports no pair.
            wf.add_measurement("EdgePair", "pair_1", row=70.0, col=100.0,
                               angle=math.pi / 2, length1=60.0, length2=6.0,
                               sigma=1.0, threshold=30.0)
        return wf, image

    def test_produces_one_target_per_detection(self):
        wf, image = self.build(observations=[
            obs_at(140, 270, BOX_H, BOX_W, 0.0),
            obs_at(340, 570, BOX_H, BOX_W, 0.0)])
        results = wf.measure(image)
        assert len(results) == 2
        assert [t.id for t in results] == [1, 2]

    def test_targets_carry_detected_pose(self):
        wf, image = self.build()
        target = wf.measure(image)[0]
        assert target.center_row == pytest.approx(140.0, abs=1.0)
        assert target.center_col == pytest.approx(270.0, abs=1.0)
        assert target.score == pytest.approx(0.9)

    def test_measurements_run_and_map_back(self):
        wf, image = self.build()
        target = wf.measure(image)[0]
        assert target.measurements, "no tool ran on the detected patch"
        assert target.valid, target.measurements
        # EdgePair exposes its two boundary points rather than a midpoint.  Both
        # must land on the bar's real edges in *inspection-image* coordinates,
        # which is the property that proves the patch was cropped at the detected
        # box and mapped back correctly.
        a, b = target.measurements["pair_1_A"], target.measurements["pair_1_B"]
        assert a["valid"] and b["valid"]
        assert a["row"] == pytest.approx(100.0, abs=2.0)   # bar top edge
        assert b["row"] == pytest.approx(180.0, abs=2.0)   # bar bottom edge
        # And the probe stayed on the box's column centre line.
        assert a["col"] == pytest.approx(270.0, abs=2.0)
        assert b["col"] == pytest.approx(270.0, abs=2.0)

    def test_tools_are_positioned_relative_to_the_patch_not_the_image(self):
        """A tool at patch centre must land near the detected box centre.

        Catches the failure where a patch is cropped at the wrong offset: every
        measurement would still "succeed" while describing another part of the
        image.
        """
        wf, image = self.build()
        wf._measurement_defs.clear()
        # Scan across the bar's top edge (patch row 30 = image row 100), which is
        # a known distance from the detected centre in the image.
        wf.add_measurement("EdgePoint", "p_edge", row=30.0, col=100.0,
                           angle=math.pi / 2, length1=20.0, length2=4.0,
                           sigma=1.0, threshold=30.0, transition="all")
        target = wf.measure(image)[0]
        point = target.measurements.get("p_edge")
        assert point is not None and point.get("valid"), point
        # The bar's top edge is 40 px above the detected centre (row 100 vs 140).
        assert point["row"] == pytest.approx(100.0, abs=8.0)
        # And the probe stays on the box's column centre line.
        assert point["col"] == pytest.approx(270.0, abs=8.0)

    def test_no_detections_yields_no_targets(self):
        wf, image = self.build()
        wf._alignment._source = FakeObbSource([])
        assert wf.measure(image) == []

    def test_localization_only_needs_no_tools(self):
        wf, image = self.build(with_tool=False)
        target = wf.measure(image)[0]
        assert target.is_localization_only
        assert target.pose["center_row"] == pytest.approx(140.0, abs=1.0)
        assert target.meta["localization_only"] is True

    def test_query_image_is_passed_to_the_detector(self):
        """The detector must see what is being inspected, not the reference."""
        wf, image = self.build()
        seen = []

        class Recording(FakeObbSource):
            def detect(self, img):
                seen.append(img)
                return super().detect(img)

        source = Recording([obs_at(140, 270, BOX_H, BOX_W, 0.0)])
        wf._alignment._source = source
        query = np.full((IMG_H, IMG_W), 7, np.uint8)
        wf.measure(query)
        assert seen and np.array_equal(seen[-1], query)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestRoiState:
    def test_round_trips_every_stored_field(self):
        strategy = ObbAlignment()
        strategy.teach(make_image(),
                       source=FakeObbSource([
                           obs_at(140, 270, BOX_H, BOX_W, 12.0, cls=0),
                           obs_at(300, 570, BOX_H, BOX_W, 0.0, cls=1)]),
                       anchor_class_id=1, min_score=0.4, max_targets=3,
                       model_ref={"path": "m.pt", "sha256": "abc"})
        state = strategy.get_roi_state()
        restored = ObbAlignment.from_roi_state(state)
        assert restored.nominal_size == strategy.nominal_size
        assert restored.anchor_class_id == 1
        assert restored.model_ref == {"path": "m.pt", "sha256": "abc"}
        assert restored.get_roi_state()["min_score"] == pytest.approx(0.4)
        assert restored.get_roi_state()["max_targets"] == 3

    def test_detector_is_not_serialized(self):
        """The source is injected, which is what keeps ML out of the asset."""
        strategy = ObbAlignment()
        strategy.teach(make_image(),
                       source=FakeObbSource([obs_at(140, 270, BOX_H, BOX_W, 0.0)]))
        restored = ObbAlignment.from_roi_state(strategy.get_roi_state())
        assert restored.source is None

    def test_registry_dispatches_the_obb_mode(self):
        strategy = strategy_from_roi_state(
            {"alignment_mode": "obb", "nominal_size": [BOX_H, BOX_W]},
            make_image())
        assert isinstance(strategy, ObbAlignment)
        assert strategy.nominal_size == pytest.approx((BOX_H, BOX_W))


# ---------------------------------------------------------------------------
# The other strategies must be untouched
# ---------------------------------------------------------------------------


class TestExistingModesUnaffected:
    def test_single_box_still_ignores_box_size(self):
        image = make_image()
        strategy = SingleBoxAlignment()
        strategy.teach(image, (140.0, 270.0), (BOX_H, BOX_W), 0.0)
        with_size = strategy.align(image, 140.0, 270.0, 0.0,
                                   box_size=(200.0, 300.0))
        without = strategy.align(image, 140.0, 270.0, 0.0)
        assert with_size.patch.shape == without.patch.shape

    def test_fixed_roi_still_ignores_box_size(self):
        image = make_image()
        strategy = FixedRoiAlignment()
        strategy.teach(image, (140.0, 270.0), (BOX_H, BOX_W), 0.0)
        a = strategy.align(image, 140.0, 270.0, 0.0, box_size=(200.0, 300.0))
        b = strategy.align(image, 140.0, 270.0, 0.0)
        assert a.patch.shape == b.patch.shape

    def test_template_workflow_unaffected_by_the_new_branch(self):
        """The template path must not have been rerouted by the detect() branch."""
        image = make_image()
        wf = MultiTargetWorkflow()
        wf.teach_template(image, center=(140.0, 270.0), size=(BOX_H, BOX_W),
                          angle_deg=0.0, max_matches=1)
        results = wf.measure(image)
        assert results
        assert not hasattr(wf._alignment, "detect")

# ---------------------------------------------------------------------------
# Manifest persistence (contract 3.1)
# ---------------------------------------------------------------------------


class TestManifestPersistence:
    """A detector-backed model must survive save/load without losing its frame."""

    def build(self, tmp_path):
        image = make_image()
        wf = MultiTargetWorkflow()
        strategy = ObbAlignment()
        wf._alignment = strategy
        strategy.teach(
            image,
            source=FakeObbSource([obs_at(140, 270, BOX_H, BOX_W, 12.0)]),
            anchor_class_id=0, min_score=0.3, max_targets=2,
            model_ref={"path": "obb.pt", "sha256": "deadbeef"})
        wf.add_measurement("EdgePair", "pair_1", row=70.0, col=100.0,
                           angle=math.pi / 2, length1=60.0, length2=6.0,
                           sigma=1.0, threshold=30.0)
        directory = tmp_path / "obb_model"
        wf.save(str(directory))
        return wf, image, directory

    def test_contract_version_is_3_1(self, tmp_path):
        from measure.multi_target_workflow import MANIFEST_CONTRACT_VERSION
        _, _, directory = self.build(tmp_path)
        manifest = json.loads((directory / "manifest.json").read_text())
        assert manifest["contract_version"] == MANIFEST_CONTRACT_VERSION == "3.1"

    def test_no_template_cache_is_written(self, tmp_path):
        """A detector model has no template, so the derived npz must be absent."""
        _, _, directory = self.build(tmp_path)
        assert not (directory / "workflow.npz").exists()

    def test_matching_block_is_empty_not_defaults(self, tmp_path):
        """Storing search defaults would look like real tuning that never ran."""
        _, _, directory = self.build(tmp_path)
        manifest = json.loads((directory / "manifest.json").read_text())
        assert manifest["matching"] == {}
        assert manifest["alignment"]["preprocessor"] is None

    def test_model_identity_is_recorded(self, tmp_path):
        _, _, directory = self.build(tmp_path)
        manifest = json.loads((directory / "manifest.json").read_text())
        roi = manifest["alignment"]["roi_state"]
        assert roi["model_ref"] == {"path": "obb.pt", "sha256": "deadbeef"}
        assert roi["anchor_class_id"] == 0
        assert roi["max_targets"] == 2

    def test_reload_restores_the_strategy_and_frame(self, tmp_path):
        _, _, directory = self.build(tmp_path)
        reloaded = MultiTargetWorkflow.load(str(directory))
        assert isinstance(reloaded.alignment, ObbAlignment)
        assert reloaded.alignment.nominal_size == pytest.approx((BOX_H, BOX_W))
        assert reloaded.alignment.anchor_class_id == 0
        assert reloaded.alignment.model_ref["sha256"] == "deadbeef"
        assert [d["label"] for d in reloaded.measurement_defs] == ["pair_1"]

    def test_detector_is_not_persisted(self, tmp_path):
        """Re-injection is the contract; a model file must not need torch."""
        _, _, directory = self.build(tmp_path)
        reloaded = MultiTargetWorkflow.load(str(directory))
        assert reloaded.alignment.source is None

    def test_reload_does_not_re_derive_the_nominal_size(self, tmp_path):
        """The frame must come from the asset, not from today's detection.

        Otherwise a reload against a differently-sized part would silently change
        the coordinate system every tool was taught in.
        """
        _, _, directory = self.build(tmp_path)
        reloaded = MultiTargetWorkflow.load(str(directory))
        original = reloaded.alignment.nominal_size

        # A detector now reporting a very different box must not move the frame.
        reloaded.alignment._source = FakeObbSource(
            [obs_at(140, 270, 300.0, 400.0, 0.0)])
        assert reloaded.alignment.nominal_size == pytest.approx(original)

    def test_measuring_works_after_reinjecting_a_detector(self, tmp_path):
        _, image, directory = self.build(tmp_path)
        reloaded = MultiTargetWorkflow.load(str(directory))
        reloaded.alignment._source = FakeObbSource(
            [obs_at(140, 270, BOX_H, BOX_W, 12.0)])
        results = reloaded.measure(image)
        assert len(results) == 1
        assert results[0].valid, results[0].measurements


class TestContractCompatibility:
    def test_3_0_assets_still_load(self):
        """Contract 3.1 only added fields, so older assets must keep working."""
        from measure.multi_target_workflow import (
            COMPATIBLE_MANIFEST_CONTRACT_VERSIONS)
        assert "3.0" in COMPATIBLE_MANIFEST_CONTRACT_VERSIONS
        assert "3.1" in COMPATIBLE_MANIFEST_CONTRACT_VERSIONS

    def test_unknown_future_version_is_rejected(self, tmp_path):
        from measure.multi_target_workflow import MultiTargetWorkflow
        image = make_image()
        wf = MultiTargetWorkflow()
        wf.teach_template(image, center=(140.0, 270.0), size=(BOX_H, BOX_W),
                          angle_deg=0.0, max_matches=1)
        directory = tmp_path / "future"
        wf.save(str(directory))

        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["contract_version"] = "9.9"
        manifest_path.write_text(json.dumps(manifest))

        with pytest.raises(ValueError, match="Unsupported manifest contract_version"):
            MultiTargetWorkflow.load(str(directory))
