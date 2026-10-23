"""Project-directory health, missing-file handling, and GUI/API interchange.

Three entry points reach the same directory — the GUI, the v1 API, and the engine
— and they used to disagree about what is required.  The v1 API skipped a missing
model silently and left a ``None`` workflow that failed later for unrelated-looking
reasons; the GUI refused to open an intact model because a view-layout file was
gone.  These tests pin the agreed behaviour:

* a missing **model** file fails loudly, naming the file;
* a missing **state or cache** file is tolerated (the GUI warns, the API is quiet);
* ``reference.npy`` is rebuilt from its portable PNG copy when that exists;
* each tool can open the other's project, in both directions.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from measure.asset_health import (
    FORMAT_API,
    FORMAT_DAMAGED,
    FORMAT_GUI,
    FORMAT_LEGACY,
    FORMAT_MODEL_ONLY,
    FORMAT_UNKNOWN,
    describe_missing,
    inspect_asset,
    message_for,
)
from measure.multi_target_workflow import MultiTargetWorkflow

IMG_H, IMG_W = 200, 300


def make_workflow(with_tool: bool = True) -> MultiTargetWorkflow:
    image = np.zeros((IMG_H, IMG_W), np.uint8)
    image[:, 140:160] = 200
    wf = MultiTargetWorkflow()
    wf.teach_template(image, center=(100.0, 150.0), size=(120.0, 120.0),
                      angle_deg=0.0, max_matches=1)
    if with_tool:
        wf.add_measurement("EdgePair", "pair_1", row=60.0, col=60.0, angle=0.0,
                           length1=40.0, length2=6.0, sigma=1.0, threshold=30.0)
    return wf


@pytest.fixture
def reference_image() -> np.ndarray:
    image = np.zeros((IMG_H, IMG_W), np.uint8)
    image[:, 140:160] = 200
    return image


def write_gui_project(directory, reference_image):
    """A project as the GUI saves it: model asset + project.json + PNG copies."""
    import cv2

    wf = make_workflow()
    wf.save(str(directory))
    (directory / "project.json").write_text(
        json.dumps({"version": 4, "project_name": directory.name}))
    cv2.imwrite(str(directory / "reference.png"), reference_image)
    return wf


def write_api_project(directory, reference_image):
    """A project as the v1 API saves it: same model asset, config.json state."""
    import cv2

    wf = make_workflow()
    wf.save(str(directory))
    (directory / "config.json").write_text(json.dumps(
        {"version": 2, "project_name": directory.name,
         "measurement_defs": [{"object_type": "EdgePair", "label": "pair_1"}]}))
    cv2.imwrite(str(directory / "reference.png"), reference_image)
    return wf


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


class TestFormatDetection:
    def test_gui_project(self, tmp_path, reference_image):
        write_gui_project(tmp_path, reference_image)
        health = inspect_asset(str(tmp_path))
        assert health.format == FORMAT_GUI
        assert health.contract_version is not None
        assert health.loadable

    def test_api_project(self, tmp_path, reference_image):
        write_api_project(tmp_path, reference_image)
        health = inspect_asset(str(tmp_path))
        assert health.format == FORMAT_API
        assert health.loadable

    def test_bare_model_asset(self, tmp_path):
        make_workflow().save(str(tmp_path))
        health = inspect_asset(str(tmp_path))
        assert health.format == FORMAT_MODEL_ONLY
        assert health.loadable
        # Both state files are simply absent; nothing was lost.
        assert set(health.missing_optional) >= {"project.json", "config.json"}

    def test_accepts_a_file_inside_the_directory(self, tmp_path, reference_image):
        write_gui_project(tmp_path, reference_image)
        for name in ("manifest.json", "workflow.npz", "project.json"):
            assert inspect_asset(str(tmp_path / name)).format == FORMAT_GUI

    def test_legacy_single_file_asset(self, tmp_path):
        """A pre-manifest archive is recognised so the migration script can be named.

        The distinction from a modern project whose manifest was deleted is made by
        reading the archive: only a legacy one carries ``workflow_meta``.
        """
        np.savez(str(tmp_path / "workflow.npz"),
                 workflow_meta=np.array(["{}"], dtype=object))
        health = inspect_asset(str(tmp_path))
        assert health.format == FORMAT_LEGACY
        assert "迁移" in message_for(health)

    def test_modern_cache_without_manifest_is_damage_not_legacy(self, tmp_path):
        """Migrating a deleted manifest would turn damage into a plausible model."""
        make_workflow().save(str(tmp_path))
        (tmp_path / "manifest.json").unlink()
        health = inspect_asset(str(tmp_path))
        assert health.format == FORMAT_DAMAGED
        assert not health.loadable
        assert "备份" in message_for(health)

    def test_unreadable_cache_is_not_called_legacy(self, tmp_path):
        (tmp_path / "workflow.npz").write_bytes(b"not really an npz")
        assert inspect_asset(str(tmp_path)).format == FORMAT_DAMAGED

    def test_unknown_directory(self, tmp_path):
        (tmp_path / "notes.txt").write_text("hello")
        health = inspect_asset(str(tmp_path))
        assert health.format == FORMAT_UNKNOWN
        assert not health.loadable

    def test_nonexistent_path_is_not_reported_as_its_parent(self, tmp_path):
        """Naming the parent would send the user to look in the wrong place."""
        missing = tmp_path / "typo"
        health = inspect_asset(str(missing))
        assert not health.exists
        assert not health.loadable
        assert str(missing) in health.summary()


# ---------------------------------------------------------------------------
# Missing files
# ---------------------------------------------------------------------------


class TestMissingFiles:
    def test_missing_manifest_blocks(self, tmp_path, reference_image):
        write_gui_project(tmp_path, reference_image)
        (tmp_path / "manifest.json").unlink()
        health = inspect_asset(str(tmp_path))
        # Without the manifest the directory is indistinguishable from a legacy
        # project, which is its own (actionable) diagnosis.
        assert not health.loadable
        assert "迁移" in message_for(health)

    def test_missing_reference_without_png_blocks(self, tmp_path):
        make_workflow().save(str(tmp_path))
        (tmp_path / "reference.npy").unlink()
        health = inspect_asset(str(tmp_path))
        assert not health.loadable
        assert "reference.npy" in health.missing_required

    def test_missing_reference_with_png_is_recoverable(self, tmp_path,
                                                       reference_image):
        write_gui_project(tmp_path, reference_image)
        (tmp_path / "reference.npy").unlink()
        health = inspect_asset(str(tmp_path))
        assert health.loadable, "a rebuildable file must not block loading"
        assert health.requires_reference_rebuild
        assert "reference.npy" not in health.missing_required

    def test_derived_cache_absence_never_blocks(self, tmp_path, reference_image):
        write_gui_project(tmp_path, reference_image)
        for name in ("workflow.npz", "template.npz"):
            path = tmp_path / name
            if path.exists():
                path.unlink()
        health = inspect_asset(str(tmp_path))
        assert health.loadable

    def test_state_file_absence_never_blocks(self, tmp_path, reference_image):
        write_gui_project(tmp_path, reference_image)
        (tmp_path / "project.json").unlink()
        health = inspect_asset(str(tmp_path))
        # Falls back to reporting the model asset, which is loadable.
        assert health.loadable

    def test_unsupported_contract_blocks(self, tmp_path):
        make_workflow().save(str(tmp_path))
        manifest_path = tmp_path / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["contract_version"] = "9.9"
        manifest_path.write_text(json.dumps(manifest))
        health = inspect_asset(str(tmp_path))
        assert not health.loadable
        assert "9.9" in message_for(health)

    def test_messages_separate_required_from_optional(self):
        required = describe_missing(["a.npy"], [])
        assert "无法打开" in required and "a.npy" in required
        optional = describe_missing([], ["b.npz"])
        assert "可以打开" in optional and "b.npz" in optional


# ---------------------------------------------------------------------------
# reference.npy rebuild
# ---------------------------------------------------------------------------


class TestReferenceRebuild:
    def test_rebuild_from_png_restores_a_working_model(self, tmp_path,
                                                       reference_image):
        import cv2

        wf = make_workflow()
        wf.save(str(tmp_path))
        cv2.imwrite(str(tmp_path / "reference.png"), reference_image)
        (tmp_path / "reference.npy").unlink()

        reloaded = MultiTargetWorkflow.load(str(tmp_path))
        results = reloaded.measure(np.load(str(tmp_path / "reference.npy")))
        assert len(results) == 1

    def test_rebuild_updates_the_recorded_digest(self, tmp_path, reference_image):
        """Leaving a stale sha256 behind would make the next load fail."""
        import cv2

        make_workflow().save(str(tmp_path))
        cv2.imwrite(str(tmp_path / "reference.png"), reference_image)
        (tmp_path / "reference.npy").unlink()
        MultiTargetWorkflow.load(str(tmp_path))

        manifest = json.loads((tmp_path / "manifest.json").read_text())
        # The digest must describe the file that is actually on disk now.
        from measure.multi_target_workflow import _sha256_file
        assert manifest["reference_image"]["sha256"] == \
            _sha256_file(str(tmp_path / "reference.npy"))
        # And a second load must therefore succeed.
        MultiTargetWorkflow.load(str(tmp_path))

    def test_rebuild_refuses_a_png_of_the_wrong_size(self, tmp_path,
                                                     reference_image):
        """A different-sized PNG is a different image.

        Accepting it would rebuild the asset around mismatched pixels and
        mis-register every taught tool, so this must fail rather than "helpfully"
        produce a plausible-looking model.
        """
        import cv2

        make_workflow().save(str(tmp_path))
        cv2.imwrite(str(tmp_path / "reference.png"),
                    np.zeros((50, 50), np.uint8))
        (tmp_path / "reference.npy").unlink()

        with pytest.raises(ValueError, match="does not match the manifest"):
            MultiTargetWorkflow.load(str(tmp_path))


# ---------------------------------------------------------------------------
# GUI behaviour
# ---------------------------------------------------------------------------


class TestGuiLoadBehaviour:
    def _load(self, directory, monkeypatch, tmp_path):
        import tkinter as tk
        import tkinter.messagebox as mb
        from measure_gui.app import MeasureApp
        from measure_gui.project_manager import ProjectManager

        monkeypatch.setattr(mb, "showwarning", lambda *a, **k: None)
        monkeypatch.setattr(mb, "showerror", lambda *a, **k: None)
        monkeypatch.setattr(mb, "showinfo", lambda *a, **k: None)
        monkeypatch.setattr(ProjectManager, "RECENT_FILE",
                            str(tmp_path / "recent.json"))
        root = tk.Tk()
        root.withdraw()
        app = MeasureApp(root)
        try:
            ProjectManager.load_project(str(directory), app)
            return app
        finally:
            root.destroy()

    def test_missing_model_file_raises_with_the_file_named(self, tmp_path,
                                                           reference_image,
                                                           monkeypatch):
        write_gui_project(tmp_path, reference_image)
        (tmp_path / "manifest.json").unlink()
        (tmp_path / "reference.png").unlink()
        with pytest.raises(FileNotFoundError) as excinfo:
            self._load(tmp_path, monkeypatch, tmp_path)
        assert "无法打开" in str(excinfo.value)

    def test_missing_state_file_still_opens(self, tmp_path, reference_image,
                                            monkeypatch):
        """project.json holds view state only; its loss must not block the model."""
        write_gui_project(tmp_path, reference_image)
        (tmp_path / "project.json").unlink()
        app = self._load(tmp_path, monkeypatch, tmp_path)
        assert app._workflow is not None

    def test_missing_cache_still_opens(self, tmp_path, reference_image,
                                       monkeypatch):
        write_gui_project(tmp_path, reference_image)
        (tmp_path / "workflow.npz").unlink()
        app = self._load(tmp_path, monkeypatch, tmp_path)
        assert app._workflow is not None

    def test_rebuild_is_reported_not_silent(self, tmp_path, reference_image,
                                            monkeypatch):
        """Repairing a file on open should be visible, not a hidden write."""
        import cv2

        write_gui_project(tmp_path, reference_image)
        cv2.imwrite(str(tmp_path / "reference.png"), reference_image)
        (tmp_path / "reference.npy").unlink()
        app = self._load(tmp_path, monkeypatch, tmp_path)
        assert app._workflow is not None
        assert "重建" in app._status_text.get()


# ---------------------------------------------------------------------------
# GUI <-> API interchange
# ---------------------------------------------------------------------------


class TestInterchange:
    """Either tool must open the other's project, both reading and writing."""

    def test_engine_opens_both_layouts(self, tmp_path, reference_image):
        gui_dir = tmp_path / "gui"
        api_dir = tmp_path / "api"
        gui_dir.mkdir()
        api_dir.mkdir()
        write_gui_project(gui_dir, reference_image)
        write_api_project(api_dir, reference_image)

        for directory in (gui_dir, api_dir):
            wf = MultiTargetWorkflow.load(str(directory))
            assert wf.measurement_defs, f"{directory} lost its tools"

    def test_gui_opens_an_api_project(self, tmp_path, reference_image,
                                      monkeypatch):
        api_dir = tmp_path / "api"
        api_dir.mkdir()
        write_api_project(api_dir, reference_image)

        import tkinter as tk
        import tkinter.messagebox as mb
        from measure_gui.app import MeasureApp
        from measure_gui.project_manager import ProjectManager

        monkeypatch.setattr(mb, "showwarning", lambda *a, **k: None)
        monkeypatch.setattr(mb, "showerror", lambda *a, **k: None)
        monkeypatch.setattr(mb, "showinfo", lambda *a, **k: None)
        monkeypatch.setattr(ProjectManager, "RECENT_FILE",
                            str(tmp_path / "recent.json"))
        root = tk.Tk()
        root.withdraw()
        app = MeasureApp(root)
        try:
            ProjectManager.load_project(str(api_dir), app)
            assert app._workflow is not None
            # The API's state file is read as a fallback for ordering/layout.
            assert ProjectManager._read_project_state(str(api_dir)) is not None
        finally:
            root.destroy()

    def test_api_opens_a_gui_project(self, tmp_path, reference_image):
        from measure_api.project import MeasureProject

        gui_dir = tmp_path / "gui"
        gui_dir.mkdir()
        write_gui_project(gui_dir, reference_image)

        project = MeasureProject(str(gui_dir))
        info = project.load()
        assert info["num_measurements"] >= 0
        assert project._workflow is not None

    def test_api_save_writes_the_shared_model_asset(self, tmp_path,
                                                    reference_image):
        """Saving in the API must produce what the GUI and engine read."""
        from measure_api.project import MeasureProject

        project = MeasureProject(str(tmp_path / "saved"))
        project._workflow = make_workflow()
        project._reference_image = reference_image
        project.save()

        saved = tmp_path / "saved"
        assert (saved / "manifest.json").is_file()
        assert (saved / "reference.npy").is_file()
        assert (saved / "config.json").is_file()
        # And the engine agrees it is a healthy asset.
        assert inspect_asset(str(saved)).loadable

    def test_api_cache_name_does_not_matter(self, tmp_path, reference_image):
        """template.npz vs workflow.npz is a naming difference, not a format one."""
        for cache_name in ("template.npz", "workflow.npz"):
            directory = tmp_path / cache_name.replace(".", "_")
            directory.mkdir()
            make_workflow().save(str(directory))
            if cache_name != "workflow.npz":
                (directory / "workflow.npz").rename(directory / cache_name)
            # The manifest is the asset; load must work either way.
            assert inspect_asset(str(directory)).loadable
            assert MultiTargetWorkflow.load(str(directory)).measurement_defs


class TestApiMissingFileReporting:
    def test_missing_model_raises_instead_of_silently_skipping(self, tmp_path,
                                                             reference_image):
        """The old code left _workflow as None and failed later, elsewhere."""
        from measure_api.project import MeasureProject

        directory = tmp_path / "broken"
        directory.mkdir()
        write_api_project(directory, reference_image)
        (directory / "manifest.json").unlink()
        (directory / "reference.png").unlink()

        with pytest.raises(FileNotFoundError):
            MeasureProject(str(directory)).load()

    def test_empty_project_loads_without_a_model(self, tmp_path):
        """Saving before teaching is normal; reopening must not be an error."""
        from measure_api.project import MeasureProject

        project = MeasureProject(str(tmp_path / "fresh"))
        project.save()
        reloaded = MeasureProject(str(tmp_path / "fresh"))
        info = reloaded.load()
        assert info["num_measurements"] == 0
        assert reloaded._workflow is None

    def test_missing_cache_is_tolerated_silently(self, tmp_path,
                                                 reference_image):
        """The API reports via return values and must not fail over a cache."""
        from measure_api.project import MeasureProject

        directory = tmp_path / "nocache"
        directory.mkdir()
        write_gui_project(directory, reference_image)
        (directory / "workflow.npz").unlink()

        project = MeasureProject(str(directory))
        project.load()
        assert project._workflow is not None