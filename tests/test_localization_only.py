"""Tests for localisation-only models — stage 1 without stage 2.

A model may legitimately define no measurement tools: some tasks only want to
know *where* the part is and *how it is rotated*.  These tests pin that as a
supported configuration rather than an accident of empty collections, covering

  * ``is_taught`` as the single "may I save / execute" predicate
  * ``TargetResult.pose`` staying consistent with its source fields
  * ``valid`` semantics when there is nothing to measure
  * manifest round-trip for both search-free modes
  * the GUI path that previously refused such models (needs a display)
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from measure.alignment import (
    FixedRoiAlignment,
    NoneAlignment,
    SingleBoxAlignment,
)
from measure.multi_target_workflow import MultiTargetWorkflow, TargetResult

IMG_H, IMG_W = 200, 300
BAR_LEFT, BAR_RIGHT = 140, 160
ROI_CENTER = (100.0, 150.0)
ROI_SIZE = (120.0, 120.0)


def make_image() -> np.ndarray:
    img = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    img[:, BAR_LEFT:BAR_RIGHT] = 200
    return img


PROBE = {"row": 60.0, "col": 60.0, "angle": 0.0, "length1": 40.0,
         "length2": 6.0, "sigma": 1.0, "threshold": 30.0}


def search_free(cls, *, with_tool: bool = False):
    wf = MultiTargetWorkflow()
    wf.alignment = cls()
    size = (IMG_H, IMG_W) if cls is NoneAlignment else ROI_SIZE
    wf.teach_without_search(make_image(), ROI_CENTER, size, 0.0)
    if with_tool:
        wf.add_measurement("EdgePair", "pair_1", **PROBE)
    return wf


# ---------------------------------------------------------------------------
# is_taught
# ---------------------------------------------------------------------------


class TestIsTaught:
    def test_fresh_workflow_is_not_taught(self):
        assert MultiTargetWorkflow().is_taught is False

    @pytest.mark.parametrize("cls", [NoneAlignment, FixedRoiAlignment])
    def test_search_free_modes_count_as_taught(self, cls):
        # The old predicate was "has a template", which is false here — that is
        # precisely why GUI save/execute refused these models.
        wf = search_free(cls)
        assert wf.is_taught is True
        assert wf._alignment.template_point is None

    def test_template_mode_counts_as_taught(self):
        wf = MultiTargetWorkflow()
        wf.teach_template(make_image(), center=ROI_CENTER, size=ROI_SIZE,
                          angle_deg=0.0, max_matches=1)
        assert wf.is_taught is True

    def test_save_refuses_an_untaught_model(self, tmp_path):
        with pytest.raises(RuntimeError, match="Nothing taught yet"):
            MultiTargetWorkflow().save(str(tmp_path))


# ---------------------------------------------------------------------------
# pose view
# ---------------------------------------------------------------------------


class TestPoseView:
    def build(self, cls=NoneAlignment):
        wf = search_free(cls)
        return wf.measure(make_image())[0]

    def test_pose_mirrors_the_source_fields(self):
        t = self.build()
        assert t.pose == {
            "center_row": t.center_row,
            "center_col": t.center_col,
            "angle_deg": t.rotation_deg,
            "scale": t.scale,
            "score": t.score,
        }

    def test_pose_is_a_view_not_a_copy_of_state(self):
        """Mutating the dataclass must change ``pose`` too."""
        t = self.build()
        t.center_col += 5.0
        assert t.pose["center_col"] == t.center_col

    def test_localization_only_true_when_no_tools(self):
        assert self.build().is_localization_only is True

    def test_localization_only_false_with_tools(self):
        t = search_free(FixedRoiAlignment, with_tool=True).measure(make_image())[0]
        assert t.is_localization_only is False
        assert t.measurements

    def test_meta_flags_localization_only(self):
        assert self.build().meta["localization_only"] is True
        t = search_free(FixedRoiAlignment, with_tool=True).measure(make_image())[0]
        assert t.meta["localization_only"] is False


class TestToDictBackwardCompatibility:
    """Existing keys are contractual; only additions are allowed."""

    LEGACY_KEYS = {"id", "score", "rotation_deg", "scale", "center_row",
                   "center_col", "valid", "measurements", "meta"}

    def test_legacy_keys_unchanged_and_new_keys_present(self):
        d = search_free(NoneAlignment).measure(make_image())[0].to_dict()
        assert self.LEGACY_KEYS <= set(d)
        assert {"pose", "is_localization_only"} <= set(d)

    def test_pose_agrees_with_flat_keys_in_dict(self):
        d = search_free(NoneAlignment).measure(make_image())[0].to_dict()
        assert d["pose"]["center_row"] == d["center_row"]
        assert d["pose"]["angle_deg"] == d["rotation_deg"]

    def test_serializable_to_json(self):
        d = search_free(NoneAlignment).measure(make_image())[0].to_dict()
        # meta holds M_inv (ndarray); JSON dump must not be asked to encode it.
        printable = {k: v for k, v in d.items() if k != "meta"}
        json.dumps(printable)


# ---------------------------------------------------------------------------
# valid semantics with nothing to measure
# ---------------------------------------------------------------------------


class TestValidSemantics:
    def test_matched_and_no_tools_is_valid(self):
        t = search_free(NoneAlignment).measure(make_image())[0]
        assert t.valid is True
        assert t.measurements == {}

    def test_no_search_means_no_score_not_a_perfect_one(self):
        """A fabricated 1.0 made every image look perfectly located.

        ``none`` mode runs no matcher, so it has no correlation value to
        report; score stays None and JSON-serialises as null.
        """
        import math
        t = search_free(NoneAlignment).measure(make_image())[0]
        assert t.score is None
        assert t.pose["score"] is None
        assert not math.isnan(t.score) if t.score is not None else True
        # Must survive a strict JSON dump (NaN would raise here).
        json.dumps({k: v for k, v in t.to_dict().items() if k != "meta"},
                   allow_nan=False)

    def test_unrelated_image_still_returns_the_fixed_frame(self):
        """Documenting the honest consequence of skipping stage 1.

        With no locator there is nothing to reject a wrong image: the model
        returns its stored ROI centre regardless.  Callers that need
        presence-detection must keep a locator (template or OBB); this mode
        only removes the *measurement* stage, not the *detection* stage.
        """
        wf = search_free(NoneAlignment)
        blank = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        t = wf.measure(blank)[0]
        assert t.center_col == pytest.approx(ROI_CENTER[1])
        assert t.score is None          # and never claims confidence


# ---------------------------------------------------------------------------
# Manifest round-trip
# ---------------------------------------------------------------------------


class TestManifestRoundTrip:
    @pytest.mark.parametrize("cls,size", [
        (NoneAlignment, (IMG_H, IMG_W)),
        (FixedRoiAlignment, ROI_SIZE),
    ])
    def test_localization_only_asset_survives_save_load(self, tmp_path, cls, size):
        image = make_image()
        wf = MultiTargetWorkflow()
        wf.alignment = cls()
        wf.teach_without_search(image, ROI_CENTER, size, 0.0)
        directory = tmp_path / cls.mode_name
        wf.save(str(directory))

        manifest = json.loads((directory / "manifest.json").read_text())
        assert manifest["alignment"]["mode"] == cls.mode_name
        assert manifest["objects"] == []
        assert manifest["matching"] == {}

        reloaded = MultiTargetWorkflow.load(str(directory))
        assert reloaded.is_taught
        before = wf.measure(image)[0]
        after = reloaded.measure(image)[0]
        for key in before.pose:
            assert after.pose[key] == pytest.approx(before.pose[key])
        assert after.is_localization_only is True


# ---------------------------------------------------------------------------
# GUI path (needs a real display; skipped otherwise)
# ---------------------------------------------------------------------------

_tk = pytest.importorskip("tkinter")


def _display_available() -> bool:
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        return False
    try:
        root = _tk.Tk()
    except Exception:
        return False
    root.destroy()
    return True


gui_only = pytest.mark.skipif(
    not _display_available(), reason="requires an X/Wayland display")


@gui_only
class TestGuiLoadPath:
    """Regression guard for the three GUI refusals this work removed.

    Loading a search-free project used to fail on a missing ``workflow.npz``,
    then came back as ``single_box`` because restoring the ROI fired the
    interactive confirm callback, and finally could not be executed at all.
    """

    def make_project(self, root_dir: Path) -> str:
        from measure.multi_target_workflow import MultiTargetWorkflow as MW
        image = make_image()
        wf = MW()
        wf.alignment = FixedRoiAlignment()
        wf.teach_without_search(image, ROI_CENTER, ROI_SIZE, 0.0)
        wf.add_measurement("EdgePair", "pair_1", **PROBE)
        proj = root_dir / "proj"
        proj.mkdir()
        wf.save(str(proj))
        json.dump({"version": 4, "project_name": "proj",
                   "view": {"roi_confirmed": True}},
                  (proj / "project.json").open("w"))
        return str(proj)

    def test_load_preserves_mode_and_executes(self, tmp_path, monkeypatch):
        import tkinter.messagebox as mb
        from measure_gui.app import MeasureApp
        from measure_gui.project_manager import ProjectManager

        popups = []
        monkeypatch.setattr(mb, "showwarning",
                            lambda *a, **k: popups.append(a))
        monkeypatch.setattr(mb, "showerror", lambda *a, **k: popups.append(a))
        monkeypatch.setattr(ProjectManager, "RECENT_FILE",
                            str(tmp_path / "recent.json"))

        proj = self.make_project(tmp_path)
        root = _tk.Tk()
        root.withdraw()
        app = MeasureApp(root)
        try:
            ProjectManager.load_project(proj, app)
            assert app._workflow.alignment.mode_name == "fixed_roi"
            assert len(app._workflow.measurement_defs) == 1
            app._inspection_image = make_image()
            before = len(popups)
            app._execute()
            assert len(popups) == before, f"execute blocked: {popups[before:]}"
            targets = app._workflow.results
            assert len(targets) == 1
            assert targets[0].measurements, "tools lost during restore"
        finally:
            root.destroy()

    def test_restore_does_not_reteach(self, tmp_path, monkeypatch):
        """The interactive confirm callback must stay silent while loading."""
        import tkinter.messagebox as mb
        from measure_gui.app import MeasureApp
        from measure_gui.project_manager import ProjectManager

        monkeypatch.setattr(mb, "showwarning", lambda *a, **k: None)
        monkeypatch.setattr(mb, "showerror", lambda *a, **k: None)
        monkeypatch.setattr(ProjectManager, "RECENT_FILE",
                            str(tmp_path / "recent.json"))

        proj = self.make_project(tmp_path)
        calls = []
        original = MultiTargetWorkflow.teach_template
        monkeypatch.setattr(
            MultiTargetWorkflow, "teach_template",
            lambda self, *a, **k: (calls.append(1), original(self, *a, **k))[1])

        root = _tk.Tk()
        root.withdraw()
        app = MeasureApp(root)
        try:
            ProjectManager.load_project(proj, app)
            assert calls == [], "load rebuilt the template from combobox state"
            # ...but a genuine user confirmation must still teach.
            app._on_roi_confirmed(100.0, 150.0, 120.0, 120.0, 0.0)
            assert calls, "interactive confirm was suppressed permanently"
        finally:
            root.destroy()
