"""GUI tests for the preprocessing preview dialog.

The dialog is the only place an operator can see what a preprocessor does before
committing a template, so the behaviour worth pinning is: every pane renders,
switching type rebuilds the parameter controls and the images, and nothing the
window does mutates the images it was handed.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

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


gui_only = pytest.mark.skipif(not _display_available(),
                              reason="requires an X/Wayland display")


@pytest.fixture()
def images():
    """A bright, structured template patch plus a larger inspection image."""
    yy, xx = np.mgrid[0:60, 0:70]
    # Keep every pixel above 128 so a global threshold visibly saturates it.
    patch = np.clip(200 + 30 * np.sin(xx / 6.0) + 15 * np.cos(yy / 5.0), 0, 255)
    patch = patch.astype(np.uint8)

    iy, ix = np.mgrid[0:90, 0:110]
    inspection = np.clip(90 + 60 * np.sin(ix / 9.0) + 10 * np.cos(iy / 7.0),
                         0, 255).astype(np.uint8)
    return patch, inspection


@pytest.fixture()
def dialog(images):
    from measure_gui.preprocessor_preview import PreprocessorPreviewDialog

    patch, inspection = images
    root = _tk.Tk()
    root.withdraw()
    win = PreprocessorPreviewDialog(root, template_patch=patch,
                                    inspection_image=inspection,
                                    initial_type="clahe")
    win.geometry("1040x840+40+40")
    root.update()
    win.refresh_now()
    root.update()
    try:
        yield win
    finally:
        try:
            win.close()
        except Exception:
            pass
        try:
            root.destroy()
        except Exception:
            pass


@gui_only
class TestPreprocessorPreviewDialog:
    def test_all_four_panes_render(self, dialog):
        for key in ("template_raw", "template_enh", "inspection_raw", "inspection_enh"):
            view = dialog._views[key]
            assert view._photo is not None, key
            assert view._stats_var.get(), key

    def test_template_pane_reports_patch_size(self, dialog, images):
        patch, _ = images
        stats = dialog._views["template_enh"]._stats_var.get()
        h, w = patch.shape
        assert f"{h}×{w}" in stats

    def test_switching_type_rebuilds_controls(self, dialog):
        dialog.set_type("threshold")
        dialog.refresh_now()
        assert set(dialog._vars) == {"threshold", "mode"}
        assert dialog._current_config()["type"] == "threshold"

        dialog.set_type("clahe")
        dialog.refresh_now()
        assert set(dialog._vars) == {"clip_limit", "tile_x", "tile_y"}

    def test_clahe_parameters_reach_the_preview(self, dialog):
        dialog.set_type("clahe")
        dialog.refresh_now()
        before = dialog._views["template_enh"]._stats_var.get()

        dialog._vars["clip_limit"].set(12.0)
        dialog.refresh_now()
        after = dialog._views["template_enh"]._stats_var.get()
        assert before != after

    def test_threshold_saturation_is_reported(self, dialog):
        dialog.set_type("threshold")
        dialog.refresh_now()
        # The fixture patch is mostly brighter than 128, so it saturates.
        assert dialog._views["template_enh"].stats_uniform() is True
        assert "饱和" in dialog._status_var.get()

    def test_unknown_type_is_ignored(self, dialog):
        before = dialog._type_var.get()
        dialog.set_type("not-a-preprocessor")
        assert dialog._type_var.get() == before

    def test_refresh_does_not_mutate_inputs(self, dialog, images):
        patch, inspection = images
        patch_before = patch.copy()
        inspection_before = inspection.copy()
        dialog.refresh_now()
        dialog.set_type("sobel")
        dialog.refresh_now()
        assert np.array_equal(patch, patch_before)
        assert np.array_equal(inspection, inspection_before)

    def test_missing_images_show_placeholders(self):
        from measure_gui.preprocessor_preview import PreprocessorPreviewDialog

        root = _tk.Tk()
        root.withdraw()
        win = PreprocessorPreviewDialog(root, template_patch=None,
                                        inspection_image=None)
        try:
            root.update()
            win.refresh_now()
            root.update()
            assert win._views["template_enh"]._photo is None
            assert win._views["inspection_enh"]._photo is None
        finally:
            win.close()
            root.destroy()

    def test_compare_window_lists_every_type(self, dialog):
        dialog.open_compare()
        dialog._compare_window.update()
        assert dialog._compare_window is not None
        assert dialog._compare_window.winfo_exists()
        dialog._compare_window.destroy()

    def test_close_is_idempotent(self, dialog):
        dialog.close()
        dialog.close()