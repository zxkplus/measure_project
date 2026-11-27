"""GUI pyramid controls and ROI-coverage validation.

The pyramid combobox used to offer "2x / 4x / 8x" while the engine only treats
the value as an on/off switch, so three labels produced one behaviour. These
tests pin the replacement (switch + size cap) and the ROI validation helper that
catches a search ROI clipping the taught template.
"""

from __future__ import annotations

import os

import pytest

from measure_gui.app import template_roi_issues

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


# ----------------------------------------------------------------------
# ROI coverage helper (pure, no Tk)
# ----------------------------------------------------------------------

def _roi_state(center_row, center_col, height, width, angle_deg=0.0):
    return {
        "center_row": center_row,
        "center_col": center_col,
        "height": height,
        "width": width,
        "angle_deg": angle_deg,
    }


def test_roi_covering_the_template_reports_no_issue():
    state = _roi_state(150.0, 200.0, 300.0, 400.0)
    assert template_roi_issues(state, click_col=200.0, click_row=150.0,
                               template_size=80.0) == []


def test_roi_clipping_the_left_side_is_reported():
    # 120 px wide box, click 40 px left of centre, template needs 40 px either
    # side -> only 20 px of room remains on the left.
    state = _roi_state(150.0, 200.0, 300.0, 120.0)
    issues = template_roi_issues(state, click_col=160.0, click_row=150.0,
                                 template_size=80.0)
    sides = {issue["side"] for issue in issues}
    assert sides == {"left"}
    assert issues[0]["available"] == pytest.approx(20.0)
    assert issues[0]["required"] == pytest.approx(40.0)


def test_missing_roi_state_is_not_an_error():
    assert template_roi_issues({}, click_col=0.0, click_row=0.0, template_size=80.0) == []


def test_box_equal_to_the_template_is_legal():
    # Mirrors the 007 localization model: a 309.55 px box rotated 47.25 deg with
    # a 309 px template. The box *is* the patch, so equal size is legal; only a
    # box that cannot contain the template is a problem.
    state = _roi_state(1392.96, 1298.99, 309.55, 309.55, angle_deg=47.2457)
    assert template_roi_issues(state, click_col=1298.99, click_row=1392.96,
                               template_size=309.0) == []


def test_rotated_box_is_checked_in_its_own_frame():
    # A click 145 px below the centre of a 100 x 300 box:
    #   unrotated -> it eats the vertical room (5 px left, template needs 8)
    #   rotated 90 deg -> that offset becomes horizontal (right, 95 px short)
    # An axis-aligned bounding box would hide the second case entirely.
    unrotated = _roi_state(100.0, 100.0, 300.0, 100.0, angle_deg=0.0)
    rotated = _roi_state(100.0, 100.0, 300.0, 100.0, angle_deg=90.0)

    bottom = template_roi_issues(unrotated, click_col=100.0, click_row=245.0,
                                 template_size=16.0)
    right = template_roi_issues(rotated, click_col=100.0, click_row=245.0,
                                template_size=16.0)
    assert {issue["side"] for issue in bottom} == {"bottom"}
    assert {issue["side"] for issue in right} == {"right"}


# ----------------------------------------------------------------------
# Tool panel pyramid controls
# ----------------------------------------------------------------------

@pytest.fixture()
def panel():
    from measure_gui.tool_panel import ToolPanel

    root = _tk.Tk()
    root.withdraw()
    widget = ToolPanel(root)
    widget.pack()
    root.update()
    yield widget
    root.destroy()


@gui_only
def test_default_pyramid_is_exponent_one_with_a_128px_cap(panel):
    """The default matches the engine: exponent 1 and a 128 px coarse cap.

    The exponent is a number rather than a flag because taught models use 2 and
    3; showing them as a tick would understate what they run.
    """
    assert panel.pyramid_config() == (1, 128)


@gui_only
def test_disabling_the_pyramid_maps_to_decimate_zero(panel):
    panel.set_pyramid_config(0, 400)
    assert panel.pyramid_config() == (0, 400)
    assert "未降采样" in panel._pyramid_factor_label.cget("text")


@gui_only
def test_factor_label_shows_the_real_ratio(panel):
    panel.set_pyramid_config(1, 400)
    panel.update_pyramid_factor(800)
    assert "0.50×" in panel._pyramid_factor_label.cget("text")


@gui_only
def test_invalid_size_falls_back_to_the_engine_default(panel):
    panel._pyramid_size_var.set("not-a-number")
    assert panel.pyramid_config() == (1, 128)


@gui_only
def test_invalid_exponent_falls_back_to_enabled(panel):
    """A blank exponent must not silently disable the pyramid."""
    panel._pyramid_decimate_var.set("")
    decimate, size = panel.pyramid_config()
    assert decimate == 1
    assert size == 128


@gui_only
def test_exponent_is_a_number_not_a_flag(panel):
    """A model taught at decimate 3 must round-trip as 3, not as 1."""
    panel.set_pyramid_config(3, 96)
    assert panel.pyramid_config() == (3, 96)
    assert panel._pyramid_decimate_var.get() == "3"


@gui_only
def test_change_callback_receives_both_values(panel):
    seen = []
    panel.on_pyramid_changed = lambda decimate, size: seen.append((decimate, size))
    panel.set_pyramid_config(1, 250)
    panel._fire_pyramid()
    assert seen == [(1, 250)]


@gui_only
def test_restore_keeps_legacy_callers_working(panel):
    """A caller that omits the pyramid fields keeps the panel defaults."""
    panel.set_matching_params("raw", 0.5, 15.0, 1, overlap=0.3, angle_step_deg=1.0)
    assert panel.pyramid_config() == (1, 128)


# ----------------------------------------------------------------------
# Halcon-aligned controls: Metric and the verification level
# ----------------------------------------------------------------------

@gui_only
def test_metric_defaults_to_use_polarity(panel):
    """The strict policy is the default; tolerance is opt-in."""
    assert panel.metric() == "use_polarity"


@gui_only
def test_metric_round_trips(panel):
    panel.set_metric("ignore_global_polarity")
    assert panel.metric() == "ignore_global_polarity"


@gui_only
def test_unknown_metric_falls_back_instead_of_being_stored(panel):
    panel.set_metric("ignore_polarity")
    assert panel.metric() == "use_polarity"


@gui_only
def test_verify_level_defaults_to_full_resolution(panel):
    """0 keeps the historical behaviour, so nothing changes unless asked."""
    assert panel.verify_level() == 0


@gui_only
def test_verify_level_round_trips_and_clamps(panel):
    panel.set_verify_level(2)
    assert panel.verify_level() == 2
    panel.set_verify_level(-5)
    assert panel.verify_level() == 0


@gui_only
def test_algorithm_defaults_to_correlation(panel):
    """Shape is opt-in; an untouched panel must keep running NCC."""
    assert panel.algorithm() == "ncc"


@gui_only
def test_algorithm_round_trips(panel):
    panel.set_algorithm("shape")
    assert panel.algorithm() == "shape"


@gui_only
def test_unknown_algorithm_falls_back(panel):
    panel.set_algorithm("surf")
    assert panel.algorithm() == "ncc"
