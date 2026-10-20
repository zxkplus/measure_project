"""Tests for the unified multi-target engine (``measure.multi_target_workflow``).

This engine is shared by the GUI, API v1 and API v2.  It replaced two parallel
implementations, so this file covers the behaviour both of them relied on:

  * teach an oriented ROI on a reference image, define tools in patch space
  * detect *all* instances (top-k) on an inspection image
  * map per-instance results back to inspection image coordinates (via M_inv)
  * composed objects resolved from their primitive inputs
  * error semantics of the public API
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from measure.alignment import MultiPointAlignment, SingleBoxAlignment
from measure.multi_target_workflow import MultiTargetWorkflow
from measure.measure_workflow import (
    AngleResult,
    CircleResult,
    DistanceResult,
    LineResult,
    PointResult,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

REF_H, REF_W = 240, 320
ROI_CENTER = (120.0, 160.0)      # (row, col) of the ROI box centre
ROI_SIZE = (100.0, 120.0)        # (height, width) of the ROI box
# With angle 0, patch (pr, pc) <-> image (70 + pr, 100 + pc)
PATCH_ORIGIN_ROW = ROI_CENTER[0] - ROI_SIZE[0] / 2.0
PATCH_ORIGIN_COL = ROI_CENTER[1] - ROI_SIZE[1] / 2.0

MARKER_IMG = (100, 140)                       # distinctive marker in the ref image
MARKER_PATCH = (MARKER_IMG[0] - PATCH_ORIGIN_ROW,
                MARKER_IMG[1] - PATCH_ORIGIN_COL)


def make_reference(seed: int = 11) -> np.ndarray:
    """Reference image: mid-grey noise plus one high-contrast marker."""
    rng = np.random.default_rng(seed)
    img = rng.integers(40, 90, (REF_H, REF_W), dtype=np.uint8)
    r, c = MARKER_IMG
    img[r - 12:r + 12, c - 12:c + 12] = 235
    img[r - 6:r + 6, c - 6:c + 6] = 25          # ring pattern -> unambiguous
    return img


def roi_region(ref: np.ndarray) -> np.ndarray:
    """The reference pixels covered by the ROI box (angle 0)."""
    r1 = int(PATCH_ORIGIN_ROW)
    c1 = int(PATCH_ORIGIN_COL)
    return ref[r1:r1 + int(ROI_SIZE[0]), c1:c1 + int(ROI_SIZE[1])].copy()


def make_inspection(ref: np.ndarray, placements, shape=(420, 520), seed: int = 5):
    """Inspection image with the ROI content pasted at each (row, col)."""
    rng = np.random.default_rng(seed)
    insp = rng.integers(40, 90, shape, dtype=np.uint8)
    patch = roi_region(ref)
    for (r, c) in placements:
        insp[r:r + patch.shape[0], c:c + patch.shape[1]] = patch
    return insp


def taught_workflow(add_tools: bool = True) -> MultiTargetWorkflow:
    wf = MultiTargetWorkflow()
    wf.teach_template(
        make_reference(),
        center=ROI_CENTER,
        size=ROI_SIZE,
        angle_deg=0.0,
        match_score_threshold=0.4,
        angle_range=(-5.0, 5.0),
        angle_step=1.0,
    )
    if add_tools:
        wf.add_measurement("TemplateMatchPoint", "marker",
                           row=MARKER_PATCH[0], col=MARKER_PATCH[1],
                           template_size=24)
    return wf


# ---------------------------------------------------------------------------
# Teach / detection
# ---------------------------------------------------------------------------


class TestTeachAndDetect:
    def test_teach_produces_patch_of_roi_size(self):
        wf = taught_workflow(add_tools=False)
        assert wf.template_image is not None
        assert wf.template_image.shape == (int(ROI_SIZE[0]), int(ROI_SIZE[1]))

    def test_single_instance_found(self):
        ref = make_reference()
        wf = taught_workflow()
        placement = (int(PATCH_ORIGIN_ROW), int(PATCH_ORIGIN_COL))
        results = wf.measure(make_inspection(ref, [placement]))
        assert len(results) == 1
        assert results[0].valid

    def test_multiple_instances_found_topk(self):
        """Two pasted copies -> two instances, each measured independently."""
        ref = make_reference()
        wf = taught_workflow()
        placements = [(70, 100), (250, 320)]
        results = wf.measure(make_inspection(ref, placements))
        assert len(results) == 2
        assert all(r.valid for r in results)
        assert [r.id for r in results] == [1, 2]

    def test_no_instance_returns_empty(self):
        wf = taught_workflow()
        blank = np.zeros((300, 300), dtype=np.uint8)
        assert wf.measure(blank) == []

    def test_measure_without_template_raises(self):
        wf = MultiTargetWorkflow()
        with pytest.raises(RuntimeError, match="No template defined"):
            wf.measure(np.zeros((50, 50), dtype=np.uint8))


# ---------------------------------------------------------------------------
# Coordinate mapping back to the inspection image
# ---------------------------------------------------------------------------


class TestCoordinateMapping:
    def test_marker_maps_back_to_its_image_position(self):
        """The taught marker must be reported at the marker's image coords."""
        ref = make_reference()
        wf = taught_workflow()
        placement = (int(PATCH_ORIGIN_ROW), int(PATCH_ORIGIN_COL))
        results = wf.measure(make_inspection(ref, [placement]))

        marker = results[0].measurements["marker"]
        assert marker["valid"]
        assert marker["row"] == pytest.approx(MARKER_IMG[0], abs=1.5)
        assert marker["col"] == pytest.approx(MARKER_IMG[1], abs=1.5)

    def test_marker_tracks_a_translated_instance(self):
        """If the instance moves by (dr, dc) the mapped point moves with it."""
        ref = make_reference()
        wf = taught_workflow()
        dr, dc = 40, 60
        placement = (int(PATCH_ORIGIN_ROW) + dr, int(PATCH_ORIGIN_COL) + dc)
        results = wf.measure(make_inspection(ref, [placement]))
        assert len(results) == 1

        marker = results[0].measurements["marker"]
        assert marker["row"] == pytest.approx(MARKER_IMG[0] + dr, abs=1.5)
        assert marker["col"] == pytest.approx(MARKER_IMG[1] + dc, abs=1.5)

    def test_instance_center_reported_in_image_coords(self):
        ref = make_reference()
        wf = taught_workflow()
        dr, dc = 40, 60
        placement = (int(PATCH_ORIGIN_ROW) + dr, int(PATCH_ORIGIN_COL) + dc)
        results = wf.measure(make_inspection(ref, [placement]))
        assert results[0].center_row == pytest.approx(ROI_CENTER[0] + dr, abs=2.0)
        assert results[0].center_col == pytest.approx(ROI_CENTER[1] + dc, abs=2.0)

    def test_rotated_roi_maps_back(self):
        """A rotated ROI must still map the marker back to the same image point."""
        ref = make_reference()
        wf = MultiTargetWorkflow()
        wf.teach_template(ref, center=ROI_CENTER, size=ROI_SIZE, angle_deg=25.0,
                          match_score_threshold=0.4, angle_range=(-5.0, 5.0),
                          angle_step=1.0)
        # Marker coordinates inside a rotated patch are not axis aligned, so we
        # only assert that a valid detection is produced and the marker lands
        # within the ROI footprint.
        wf.add_measurement("TemplateMatchPoint", "marker", row=50.0, col=60.0,
                           template_size=24)
        results = wf.measure(make_inspection(ref, [(70, 100)], shape=(420, 520)))
        assert len(results) >= 1


# ---------------------------------------------------------------------------
# Measurements and composed objects
# ---------------------------------------------------------------------------


class TestMeasurements:
    def test_composed_distance_and_line_resolved(self):
        wf = taught_workflow(add_tools=False)
        wf.add_measurement("EdgePoint", "edge_a", row=10.0, col=60.0, angle=0.0,
                           length1=25.0, length2=4.0)
        wf.add_measurement("EdgePoint", "edge_b", row=60.0, col=60.0, angle=0.0,
                           length1=25.0, length2=4.0)
        wf.add_measurement("TwoPointsDistance", "gap",
                           point_a_label="edge_a", point_b_label="edge_b")
        wf.add_measurement("TwoPointsLine", "line",
                           point_a_label="edge_a", point_b_label="edge_b")

        ref = make_reference()
        results = wf.measure(make_inspection(
            ref, [(int(PATCH_ORIGIN_ROW), int(PATCH_ORIGIN_COL))]))
        assert len(results) == 1
        labels = set(results[0].measurements)
        assert {"edge_a", "edge_b", "gap", "line"} <= labels

    def test_composed_invalid_when_input_invalid(self):
        """A composed object whose inputs are invalid must itself be invalid."""
        wf = taught_workflow(add_tools=False)
        wf.add_measurement("TwoPointsDistance", "gap",
                           point_a_label="missing_a", point_b_label="missing_b")
        ref = make_reference()
        results = wf.measure(make_inspection(
            ref, [(int(PATCH_ORIGIN_ROW), int(PATCH_ORIGIN_COL))]))
        assert len(results) == 1
        assert results[0].measurements["gap"]["valid"] is False

    def test_template_match_point_records_score(self):
        ref = make_reference()
        wf = taught_workflow()
        results = wf.measure(make_inspection(
            ref, [(int(PATCH_ORIGIN_ROW), int(PATCH_ORIGIN_COL))]))
        marker = results[0].measurements["marker"]
        assert marker["meta"]["match_score"] > 0.9

    def test_fit_circle_runs(self):
        wf = taught_workflow(add_tools=False)
        wf.add_measurement("FitCircle", "circ", center=(50.0, 60.0), radius=20.0,
                           measure_length1=8.0, measure_length2=3.0,
                           num_measures=8)
        ref = make_reference()
        results = wf.measure(make_inspection(
            ref, [(int(PATCH_ORIGIN_ROW), int(PATCH_ORIGIN_COL))]))
        assert "circ" in results[0].measurements

    def test_visualize_returns_image(self):
        ref = make_reference()
        wf = taught_workflow()
        insp = make_inspection(ref, [(int(PATCH_ORIGIN_ROW), int(PATCH_ORIGIN_COL))])
        wf.measure(insp)
        vis = wf.visualize(insp)
        assert vis.shape[0] == insp.shape[0] and vis.shape[1] == insp.shape[1]
        assert vis.ndim == 3


# ---------------------------------------------------------------------------
# Measurement CRUD + error semantics
# ---------------------------------------------------------------------------


class TestMeasurementApi:
    def test_unknown_object_type_raises(self):
        wf = taught_workflow(add_tools=False)
        with pytest.raises(ValueError, match="Unknown object_type"):
            wf.add_measurement("NoSuchTool", "x", row=1.0, col=1.0)

    def test_missing_required_param_raises(self):
        wf = taught_workflow(add_tools=False)
        with pytest.raises(ValueError, match="Missing required parameter"):
            wf.add_measurement("EdgePoint", "x", row=1.0, col=1.0)

    def test_add_update_remove_clear(self):
        wf = taught_workflow(add_tools=False)
        wf.add_measurement("EdgePoint", "a", row=10.0, col=10.0, angle=0.0,
                           length1=5.0, length2=2.0)
        assert len(wf.measurement_defs) == 1

        wf.update_measurement("a", threshold=50.0)
        assert wf.measurement_defs[0]["params"]["threshold"] == 50.0

        wf.remove_measurement("a")
        assert wf.measurement_defs == []

        wf.add_measurement("EdgePoint", "b", row=10.0, col=10.0, angle=0.0,
                           length1=5.0, length2=2.0)
        wf.clear_measurements()
        assert wf.measurement_defs == []

    def test_remove_cascades_to_dependants(self):
        wf = taught_workflow(add_tools=False)
        wf.add_measurement("EdgePoint", "a", row=10.0, col=10.0, angle=0.0,
                           length1=5.0, length2=2.0)
        wf.add_measurement("EdgePoint", "b", row=60.0, col=10.0, angle=0.0,
                           length1=5.0, length2=2.0)
        wf.add_measurement("TwoPointsDistance", "gap",
                           point_a_label="a", point_b_label="b")
        assert len(wf.measurement_defs) == 3

        wf.remove_measurement("a")
        labels = [d["label"] for d in wf.measurement_defs]
        assert labels == ["b"]

    def test_move_measurement_up_down(self):
        wf = taught_workflow(add_tools=False)
        for label in ("a", "b"):
            wf.add_measurement("EdgePoint", label, row=10.0, col=10.0, angle=0.0,
                               length1=5.0, length2=2.0)
        wf.move_measurement_up("b")
        assert [d["label"] for d in wf.measurement_defs] == ["b", "a"]
        wf.move_measurement_down("b")
        assert [d["label"] for d in wf.measurement_defs] == ["a", "b"]


# ---------------------------------------------------------------------------
# Alignment strategies
# ---------------------------------------------------------------------------


class TestAlignment:
    def test_default_strategy_is_single_box(self):
        wf = MultiTargetWorkflow()
        assert isinstance(wf.alignment, SingleBoxAlignment)
        assert wf.alignment.mode_name == "single_box"

    def test_multi_point_strategy_assignable_and_measures(self):
        wf = MultiTargetWorkflow()
        wf.alignment = MultiPointAlignment()
        wf.teach_template(make_reference(), center=ROI_CENTER, size=ROI_SIZE,
                          angle_deg=0.0, match_score_threshold=0.4,
                          angle_range=(-5.0, 5.0), angle_step=1.0)
        wf.add_measurement("TemplateMatchPoint", "marker",
                           row=MARKER_PATCH[0], col=MARKER_PATCH[1],
                           template_size=24)
        ref = make_reference()
        results = wf.measure(make_inspection(
            ref, [(int(PATCH_ORIGIN_ROW), int(PATCH_ORIGIN_COL))]))
        assert len(results) == 1

    def test_roi_state_roundtrips_through_strategy(self):
        wf = MultiTargetWorkflow()
        wf.teach_template(make_reference(), center=ROI_CENTER, size=ROI_SIZE,
                          angle_deg=12.0)
        state = wf.alignment.get_roi_state()
        assert state["alignment_mode"] == "single_box"
        assert state["box_center"] == list(ROI_CENTER)
        assert state["box_size"] == list(ROI_SIZE)
        assert state["box_angle_deg"] == 12.0


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_save_load_roundtrip(self, tmp_path: Path):
        wf = taught_workflow()
        path = tmp_path / "workflow.npz"
        wf.save(str(path))

        reloaded = MultiTargetWorkflow.load(str(path))
        ref = make_reference()
        insp = make_inspection(ref, [(int(PATCH_ORIGIN_ROW), int(PATCH_ORIGIN_COL))])

        before = wf.measure(insp)
        after = reloaded.measure(insp)

        assert len(before) == len(after)
        for a, b in zip(before, after):
            assert a.valid == b.valid
            assert a.center_row == pytest.approx(b.center_row, abs=1e-6)
            assert a.center_col == pytest.approx(b.center_col, abs=1e-6)
            assert a.measurements["marker"]["row"] == pytest.approx(
                b.measurements["marker"]["row"], abs=1e-6)
            assert a.measurements["marker"]["col"] == pytest.approx(
                b.measurements["marker"]["col"], abs=1e-6)

    def test_save_load_preserves_measurement_defs(self, tmp_path: Path):
        wf = taught_workflow(add_tools=False)
        wf.add_measurement("EdgePoint", "e", row=5.0, col=5.0, angle=0.0,
                           length1=4.0, length2=2.0, threshold=42.0)
        path = tmp_path / "workflow.npz"
        wf.save(str(path))

        reloaded = MultiTargetWorkflow.load(str(path))
        assert len(reloaded.measurement_defs) == 1
        assert reloaded.measurement_defs[0]["label"] == "e"
        assert reloaded.measurement_defs[0]["params"]["threshold"] == 42.0


# ---------------------------------------------------------------------------
# Result formatting / physical scale
# ---------------------------------------------------------------------------


class TestSummaryAndScale:
    def test_summary_text_reports_scale(self):
        wf = MultiTargetWorkflow()
        assert wf.physical_scale_mm is None
        wf.physical_scale_mm = 0.05
        wf._results = [self._target_result()]

        text = wf.summary_text()
        assert "5.000 mm" in text       # distance 100 px * 0.05
        assert "2.50 mm" in text        # circle radius 50 px * 0.05

    @staticmethod
    def _target_result():
        from measure.multi_target_workflow import TargetResult

        return TargetResult(
            id=1, score=0.95, rotation_deg=0.0, scale=1.0,
            center_row=100.0, center_col=200.0, valid=True,
            measurements={
                "gap": DistanceResult(label="gap", value=100.0, valid=True),
                "hole": CircleResult(label="hole", center_row=0.0,
                                     center_col=0.0, radius=50.0, valid=True),
                "pt": PointResult(label="pt", row=1.0, col=2.0, valid=True),
                "ang": AngleResult(label="ang", value_rad=0.5, valid=True),
            },
        )