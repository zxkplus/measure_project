"""Every measurement type must reach the results table.

A real session reported a composed angle that appeared in the summary text but not
in the per-target table.  The cause was a ``dict.get(key, default)`` trap: angle
payloads carry ``value: None`` alongside ``value_deg``, and ``get`` only falls back
when the key is *absent*, so formatting ``None`` raised TypeError and aborted the
row loop — leaving the earlier rows on screen and silently dropping the rest.
"""

from __future__ import annotations

import os

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


def _payload(kind: str) -> dict:
    """A result dict shaped like the engine's, one per measurement type."""
    if kind == "angle":
        # The critical shape: value present but None, real number in value_deg.
        return {"type": "angle", "label": "a", "valid": True,
                "value": None, "value_deg": 37.54}
    if kind == "distance":
        return {"type": "distance", "label": "d", "valid": True,
                "value": 12.5, "value_deg": None}
    if kind == "circle":
        return {"type": "circle", "label": "c", "valid": True,
                "center_row": 10.0, "center_col": 20.0, "radius": 5.0}
    if kind == "line":
        return {"type": "line", "label": "l", "valid": True,
                "start_row": 1.0, "start_col": 2.0,
                "end_row": 3.0, "end_col": 4.0}
    return {"type": "point", "label": "p", "valid": True,
            "row": 5.0, "col": 6.0, "meta": {"match_score": 0.9}}


@gui_only
class TestMeasurementTableRendering:
    def _panel(self):
        from measure_gui.result_panel import ResultPanel
        root = _tk.Tk()
        root.withdraw()
        return root, ResultPanel(root)

    @pytest.mark.parametrize("kind", ["angle", "distance", "circle", "line", "point"])
    def test_each_type_renders_a_value(self, kind):
        root, panel = self._panel()
        try:
            text = panel._format_measurement_value(_payload(kind), kind, True)
            assert text and text != "-", f"{kind} produced no value"
            assert "格式化失败" not in text
        finally:
            root.destroy()

    def test_angle_uses_value_deg_not_the_null_value_slot(self):
        """``value`` is None for angles; ``value_deg`` holds the number."""
        root, panel = self._panel()
        try:
            text = panel._format_measurement_value(_payload("angle"), "angle", True)
            assert text == "37.540°"
        finally:
            root.destroy()

    def test_invalid_rows_render_a_dash(self):
        root, panel = self._panel()
        try:
            assert panel._format_measurement_value(
                _payload("angle"), "angle", False) == "-"
        finally:
            root.destroy()

    def test_unknown_type_does_not_raise(self):
        root, panel = self._panel()
        try:
            assert panel._format_measurement_value({}, "mystery", True) == "-"
        finally:
            root.destroy()

    def test_one_bad_row_cannot_hide_the_others(self):
        """The regression: an unformattable value used to abort the whole loop.

        Simulated by a payload that raises inside the formatter; the row must still
        appear (marked) and the following rows must still be rendered.
        """
        from measure.multi_target_workflow import TargetResult
        root, panel = self._panel()
        try:
            target = TargetResult(
                id=1, score=0.9, rotation_deg=0.0, scale=1.0,
                center_row=10.0, center_col=20.0, valid=True,
                measurements={
                    "ok_1": _payload("line"),
                    # 'angle' with a non-numeric value_deg forces the formatter to
                    # raise, standing in for any future shape it cannot handle.
                    "broken": {"type": "angle", "label": "broken", "valid": True,
                               "value": None, "value_deg": "not a number"},
                    "ok_2": _payload("distance"),
                },
            )
            panel.set_results([target], summary="")
            tree = panel._target_tree
            tree.selection_set(tree.get_children()[0])
            panel._on_target_select(None)

            rows = [panel._meas_tree.item(r, "values") for r in
                    panel._meas_tree.get_children()]
            labels = [row[0] for row in rows]
            # All three rows present, in order: a formatting failure costs the value,
            # never the row.
            assert labels == ["ok_1", "broken", "ok_2"], labels
            broken = next(r for r in rows if r[0] == "broken")
            assert "格式化失败" in broken[2]
            assert next(r for r in rows if r[0] == "ok_2")[2] != "-"
        finally:
            root.destroy()


class TestNoGetDefaultTrap:
    """Guard the specific expression that caused the bug."""

    def test_result_panel_does_not_get_value_with_a_bare_default(self):
        """``.get('value', fallback)`` cannot fall back when the value is None.

        Checked by reading the source, because the failure is silent in the happy
        path — it only appears for payloads that carry ``value: None``.
        """
        path = os.path.join(os.path.dirname(__file__), "..", "measure_gui",
                            "result_panel.py")
        source = open(path, encoding="utf-8").read()
        assert "result.get('value', " not in source, \
            "use _first_number(...) so a None value falls through to value_deg"
        assert 'result.get("value", ' not in source, \
            "use _first_number(...) so a None value falls through to value_deg"
