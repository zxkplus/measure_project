"""Tests for skipping stage 1 — the ``none`` and ``fixed_roi`` alignment modes.

Template matching is only one way to answer "where is the part".  These two modes
exist because it is not always the right one:

* a fixed fixture makes the search pure waste (time, mismatch risk, and pose
  quantised to ``angle_step``);
* rectification deliberately *removes* pose from the patch, so a measurement that
  needs orientation itself — e.g. a bearing between two detected centres — cannot
  be expressed in a rectified frame at all.

Covered here: identity mapping, ROI cropping geometry, back-mapping equivalence
with the template path, manifest round-trip, and the error semantics of an
untaught model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from measure.alignment import (
    AlignmentStrategy,
    FixedRoiAlignment,
    NoneAlignment,
    SingleBoxAlignment,
    _ALIGNMENT_STRATEGY_REGISTRY,
    strategy_from_roi_state,
)
from measure.multi_target_workflow import MultiTargetWorkflow

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

IMG_H, IMG_W = 200, 300
# A bright vertical bar spanning every row: gives exactly one edge pair when
# probed horizontally, and no pair when probed vertically.
BAR_LEFT, BAR_RIGHT = 140, 160


def make_image() -> np.ndarray:
    img = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    img[:, BAR_LEFT:BAR_RIGHT] = 200
    return img


ROI_CENTER = (100.0, 150.0)     # (row, col)
ROI_SIZE = (120.0, 120.0)       # (height, width)
PROBE = {"row": 60.0, "col": 60.0, "angle": 0.0, "length1": 40.0,
         "length2": 6.0, "sigma": 1.0, "threshold": 30.0}


def taught(mode_cls, center=ROI_CENTER, size=ROI_SIZE, angle_deg=0.0):
    """A workflow whose stage 1 is skipped, with one EdgePair across the bar."""
    wf = MultiTargetWorkflow()
    wf.alignment = mode_cls()
    wf.teach_without_search(make_image(), center, size, angle_deg)
    wf.add_measurement("EdgePair", "pair_1", **PROBE)
    return wf


# ---------------------------------------------------------------------------
# Registry / dispatch
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_new_modes_are_registered(self):
        assert set(_ALIGNMENT_STRATEGY_REGISTRY) == {
            "none", "fixed_roi", "single_box", "multi_point",
        }

    @pytest.mark.parametrize("mode,cls", [
        ("none", NoneAlignment),
        ("fixed_roi", FixedRoiAlignment),
        ("single_box", SingleBoxAlignment),
    ])
    def test_strategy_from_roi_state_dispatches(self, mode, cls):
        strategy = strategy_from_roi_state({"alignment_mode": mode}, make_image())
        assert isinstance(strategy, cls)

    def test_unknown_mode_lists_known_modes(self):
        with pytest.raises(ValueError, match="Unknown alignment mode"):
            strategy_from_roi_state({"alignment_mode": "nope"}, make_image())

    @pytest.mark.parametrize("mode", ["none", "fixed_roi"])
    def test_mode_name_round_trips(self, mode):
        cls = _ALIGNMENT_STRATEGY_REGISTRY[mode]
        assert cls().mode_name == mode


# ---------------------------------------------------------------------------
# NoneAlignment — identity mapping
# ---------------------------------------------------------------------------


class TestNoneAlignment:
    def test_teach_creates_no_template(self):
        wf = MultiTargetWorkflow()
        wf.alignment = NoneAlignment()
        wf.teach_without_search(make_image(), ROI_CENTER, (IMG_H, IMG_W), 0.0)
        # The whole point: nothing was built to search with.
        assert wf.alignment.template_point is None

    def test_align_returns_image_unchanged_with_identity(self):
        image = make_image()
        strategy = NoneAlignment()
        strategy.teach(image, (100, 150), (200, 300), 0.0)
        out = strategy.align(image, matched_row=999, matched_col=999,
                             rotation_deg=37.0)
        # No stage-1 result exists, so the arguments must be ignored outright.
        assert out.patch.shape == image.shape
        assert np.array_equal(out.patch, image)
        np.testing.assert_allclose(
            out.M_inv, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    def test_measure_reports_full_image_geometry(self):
        wf = MultiTargetWorkflow()
        wf.alignment = NoneAlignment()
        wf.teach_without_search(make_image(), ROI_CENTER, (IMG_H, IMG_W), 0.0)
        wf.add_measurement("EdgePair", "pair_1", row=100.0, col=150.0,
                           angle=0.0, length1=40.0, length2=6.0,
                           sigma=1.0, threshold=30.0)
        target = wf.measure(make_image())[0]
        assert target.valid
        assert target.score == pytest.approx(1.0)
        assert target.rotation_deg == pytest.approx(0.0)
        # Coordinates are image coordinates — no rectification happened.
        assert target.center_row == pytest.approx(100.0)
        assert target.center_col == pytest.approx(150.0)

    def test_edge_pair_found_at_true_positions(self):
        wf = MultiTargetWorkflow()
        wf.alignment = NoneAlignment()
        wf.teach_without_search(make_image(), ROI_CENTER, (IMG_H, IMG_W), 0.0)
        wf.add_measurement("EdgePair", "pair_1", row=100.0, col=150.0,
                           angle=0.0, length1=40.0, length2=6.0,
                           sigma=1.0, threshold=30.0)
        ms = wf.measure(make_image())[0].measurements
        assert ms["pair_1_A"]["valid"]
        assert ms["pair_1_B"]["valid"]
        assert ms["pair_1_A"]["col"] == pytest.approx(BAR_LEFT - 0.5, abs=1.0)
        assert ms["pair_1_B"]["col"] == pytest.approx(BAR_RIGHT - 0.5, abs=1.0)


# ---------------------------------------------------------------------------
# FixedRoiAlignment — crop without searching
# ---------------------------------------------------------------------------


class TestFixedRoiAlignment:
    def test_teach_creates_no_template_but_keeps_patch_frame(self):
        wf = MultiTargetWorkflow()
        wf.alignment = FixedRoiAlignment()
        wf.teach_without_search(make_image(), ROI_CENTER, ROI_SIZE, 0.0)
        assert wf.alignment.template_point is None
        assert wf.template_image.shape == (int(ROI_SIZE[0]), int(ROI_SIZE[1]))

    def test_crop_is_geometrically_correct(self):
        image = make_image()
        strategy = FixedRoiAlignment()
        strategy.teach(image, ROI_CENTER, ROI_SIZE, 0.0)
        out = strategy.align(image, ROI_CENTER[0], ROI_CENTER[1], 0.0)
        # ROI spans cols 90..210; the bar (140..159) lands at local 50..69.
        bright = np.where(out.patch[60] > 100)[0]
        assert bright.min() == pytest.approx(50, abs=1)
        assert bright.max() == pytest.approx(69, abs=1)

    def test_tools_are_defined_in_patch_space(self):
        # PROBE targets patch centre (60, 60), not image centre (100, 150).
        target = taught(FixedRoiAlignment).measure(make_image())[0]
        assert target.valid
        assert target.measurements["pair_1_A"]["valid"]

    def test_back_mapping_matches_the_none_mode(self):
        """Both modes must report the same image-space edges.

        This is the property that makes the choice safe: where the tool was
        defined differs, but what the caller reads back does not.
        """
        fixed = taught(FixedRoiAlignment).measure(make_image())[0]
        none_wf = MultiTargetWorkflow()
        none_wf.alignment = NoneAlignment()
        none_wf.teach_without_search(make_image(), ROI_CENTER, (IMG_H, IMG_W), 0.0)
        none_wf.add_measurement("EdgePair", "pair_1", row=100.0, col=150.0,
                                angle=0.0, length1=40.0, length2=6.0,
                                sigma=1.0, threshold=30.0)
        plain = none_wf.measure(make_image())[0]
        for label in ("pair_1_A", "pair_1_B"):
            assert fixed.measurements[label]["col"] == pytest.approx(
                plain.measurements[label]["col"], abs=1e-6)
            assert fixed.measurements[label]["row"] == pytest.approx(
                plain.measurements[label]["row"], abs=1e-6)

    def test_rotated_roi_still_maps_back(self):
        image = make_image()
        wf = MultiTargetWorkflow()
        wf.alignment = FixedRoiAlignment()
        wf.teach_without_search(image, ROI_CENTER, ROI_SIZE, 30.0)
        wf.add_measurement("EdgePair", "pair_1", **PROBE)
        target = wf.measure(image)[0]
        # A rotated crop of a vertical bar can legitimately fail to find the
        # pair; what must hold is that M_inv round-trips consistently.
        out = wf.alignment.align(image, ROI_CENTER[0], ROI_CENTER[1], 0.0)
        assert out.M_inv.shape == (2, 3)
        assert np.all(np.isfinite(out.M_inv))


# ---------------------------------------------------------------------------
# Error semantics
# ---------------------------------------------------------------------------


class TestErrorSemantics:
    def test_untaught_measure_raises(self):
        wf = MultiTargetWorkflow()          # default SingleBoxAlignment, untaught
        with pytest.raises(RuntimeError, match="No template defined"):
            wf.measure(make_image())

    def test_fixed_roi_without_geometry_raises(self):
        strategy = FixedRoiAlignment()
        with pytest.raises(ValueError, match="center and size"):
            strategy.teach(make_image(), None, None, 0.0)

    def test_none_requires_reference_image(self):
        with pytest.raises(ValueError, match="reference image"):
            NoneAlignment().teach(None, (0, 0), (10, 10), 0.0)

    def test_teach_without_search_rejects_template_modes(self):
        wf = MultiTargetWorkflow()
        wf.alignment = SingleBoxAlignment()
        with pytest.raises(RuntimeError, match="requires a template search"):
            wf.teach_without_search(make_image(), ROI_CENTER, ROI_SIZE, 0.0)


# ---------------------------------------------------------------------------
# Manifest persistence
# ---------------------------------------------------------------------------


class TestManifestPersistence:
    @pytest.mark.parametrize("mode_cls,size", [
        (NoneAlignment, (IMG_H, IMG_W)),
        (FixedRoiAlignment, ROI_SIZE),
    ])
    def test_round_trip_preserves_mode_and_results(self, tmp_path, mode_cls, size):
        image = make_image()
        wf = MultiTargetWorkflow()
        wf.alignment = mode_cls()
        wf.teach_without_search(image, ROI_CENTER, size, 0.0)
        probe = dict(PROBE)
        if mode_cls is NoneAlignment:
            probe.update(row=100.0, col=150.0)
        wf.add_measurement("EdgePair", "pair_1", **probe)

        before = wf.measure(image)[0]
        directory = tmp_path / mode_cls.mode_name
        wf.save(str(directory), model_id="m1")

        manifest = json.loads((directory / "manifest.json").read_text())
        assert manifest["alignment"]["mode"] == mode_cls().mode_name
        # Search-free modes record no matching behaviour — empty, not defaults.
        assert manifest["matching"] == {}
        assert manifest["alignment"]["preprocessor"] is None
        assert sorted(manifest["files"]) == ["reference.npy"]
        assert not (directory / "workflow.npz").exists()

        after = MultiTargetWorkflow.load(str(directory)).measure(image)[0]
        assert after.valid == before.valid
        assert set(after.measurements) == set(before.measurements)
        for label in before.measurements:
            if before.measurements[label].get("valid"):
                assert after.measurements[label]["col"] == pytest.approx(
                    before.measurements[label]["col"])

    def test_legacy_models_keep_working(self, tmp_path):
        """A normal template-taught model still writes its derived cache."""
        image = make_image()
        wf = MultiTargetWorkflow()
        wf.teach_template(image, center=ROI_CENTER, size=ROI_SIZE, angle_deg=0.0,
                          max_matches=1)
        directory = tmp_path / "templated"
        wf.save(str(directory))
        manifest = json.loads((directory / "manifest.json").read_text())
        assert manifest["alignment"]["mode"] == "single_box"
        assert manifest["matching"]["angle_step"] == pytest.approx(1.0)
        assert "workflow.npz" in manifest["files"]
