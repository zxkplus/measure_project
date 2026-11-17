"""Tests for the ``selection`` parameter of circle fitting.

``selection`` mirrors the v2 ``measurements.circles`` contract
(:class:`measure_api.v2.families.measurements.ImageFitRequest`): every radial
probe reports its edges outermost-first, and

* ``nearest`` (default) keeps the edge closest to the expected radius,
* ``first`` keeps the outermost edge,
* ``last`` keeps the innermost edge,
* ``all`` keeps every edge inside ``[radius_min, radius_max]``.

The GUI exposes the same choice in the "拟合圆参数 (Fit Circle)" dialog, so these
tests also pin that the value travels from the dialog params through the object
registry into the engine (and survives serialization).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from measure.measure2D import CircleMeasureObject
from measure.measure_workflow import FitCircleObject
from measure.object_specs import defaults_for, get_spec

CENTER = (200.0, 200.0)
INNER_EDGE = 56.0
OUTER_EDGE = 76.0


def _two_ring_image(size: int = 400) -> np.ndarray:
    """Black image with two bright bands: edges at 40/56 and 76/90 px radius."""
    image = np.zeros((size, size), np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    radius = np.hypot(yy - CENTER[0], xx - CENTER[1])
    image[(radius >= 40) & (radius <= INNER_EDGE)] = 255
    image[(radius >= OUTER_EDGE) & (radius <= 90)] = 255
    return image


def _circle(radius: float = 60.0, selection: str = "nearest") -> CircleMeasureObject:
    """Circle probe spanning both bands, keeping only the 56 px and 76 px edges."""
    return CircleMeasureObject(
        center=CENTER,
        radius=radius,
        radius_min=50.0,
        radius_max=88.0,
        measure_length1=24.0,
        measure_length2=5.0,
        num_measures=16,
        sigma=1.0,
        threshold=30.0,
        transition="all",
        selection=selection,
    )


class TestCircleSelection:
    """Which radial edge becomes a fit sample."""

    @pytest.mark.parametrize(
        "expected_radius, selection, fitted_radius",
        [
            (60.0, "nearest", INNER_EDGE),  # 56 is 4 px away, 76 is 16 px away
            (70.0, "nearest", OUTER_EDGE),  # 76 is 6 px away, 56 is 14 px away
            (60.0, "first", OUTER_EDGE),    # outermost edge wins
            (60.0, "last", INNER_EDGE),     # innermost edge wins
        ],
    )
    def test_selection_picks_the_expected_edge(self, expected_radius, selection,
                                               fitted_radius):
        obj = _circle(radius=expected_radius, selection=selection)
        result = obj.measure(_two_ring_image())

        assert result is not None, f"{selection} produced no fit"
        assert result["radius"] == pytest.approx(fitted_radius, abs=2.0)
        # One sample per probe: the two bands must not be mixed together.
        assert result["num_points"] == obj.num_measures

    def test_default_selection_is_nearest(self):
        assert _circle().selection == "nearest"
        result = _circle(radius=60.0).measure(_two_ring_image())
        assert result["radius"] == pytest.approx(INNER_EDGE, abs=2.0)

    def test_all_keeps_every_edge_in_the_radius_window(self):
        obj = _circle(radius=60.0, selection="all")
        result = obj.measure(_two_ring_image())

        assert result is not None
        assert result["num_points"] == 2 * obj.num_measures
        # The fitted circle must stay between the two measured bands.
        assert INNER_EDGE - 2.0 <= result["radius"] <= OUTER_EDGE + 2.0

    def test_unsupported_selection_is_rejected(self):
        with pytest.raises(ValueError, match="selection"):
            _circle(selection="closest")

    def test_selection_survives_serialization(self):
        obj = _circle(selection="first")
        data = obj.to_dict()
        assert data["selection"] == "first"
        assert CircleMeasureObject.from_dict(data).selection == "first"

    def test_legacy_dict_without_selection_defaults_to_nearest(self):
        data = _circle().to_dict()
        data.pop("selection")
        assert CircleMeasureObject.from_dict(data).selection == "nearest"


class TestFitCircleObjectSelection:
    """The workflow wrapper must forward ``selection`` to the engine."""

    def test_measure_uses_selected_edge(self):
        obj = FitCircleObject(
            "circle", center=CENTER, radius=60.0,
            measure_length1=24.0, measure_length2=5.0,
            radius_min=50.0, radius_max=88.0,
            num_measures=16, transition="all", selection="first",
        )
        result = obj.measure(_two_ring_image())

        assert result.valid
        assert result.radius == pytest.approx(OUTER_EDGE, abs=2.0)
        assert obj._cached_obj.selection == "first"

    def test_to_dict_from_dict_roundtrip(self):
        obj = FitCircleObject(
            "circle", center=CENTER, radius=60.0,
            measure_length1=24.0, measure_length2=5.0,
            num_measures=16, selection="last",
        )
        data = obj.to_dict()
        assert data["selection"] == "last"
        assert FitCircleObject.from_dict("circle", data).selection == "last"

    def test_default_is_nearest(self):
        obj = FitCircleObject(
            "circle", center=CENTER, radius=60.0,
            measure_length1=24.0, measure_length2=5.0,
        )
        assert obj.selection == "nearest"


class TestObjectSpecSelection:
    """The registry is what the GUI dialog and the API builders read."""

    def test_default_is_nearest(self):
        assert defaults_for("FitCircle")["selection"] == "nearest"

    def test_factory_forwards_selection(self):
        spec = get_spec("FitCircle")
        obj = spec.factory(
            "circle", center=CENTER, radius=60.0,
            measure_length1=24.0, measure_length2=5.0,
            selection="last",
        )
        assert obj.selection == "last"

    def test_parameter_key_is_advertised(self):
        assert "selection" in get_spec("FitCircle").param_keys


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


@pytest.mark.skipif(not _display_available(), reason="requires an X/Wayland display")
class TestFitCircleDialog:
    """The GUI dialog must expose the choice as a read-only dropdown."""

    @pytest.fixture()
    def dialog(self):
        from measure_gui.dialogs import FitCircleDialog

        root = _tk.Tk()
        root.withdraw()
        dlg = FitCircleDialog(root, params={"selection": "first"})
        root.update()
        try:
            yield root, dlg
        finally:
            try:
                dlg.destroy()
            except Exception:
                pass
            try:
                root.destroy()
            except Exception:
                pass

    def _selection_combo(self, dlg):
        import tkinter.ttk as ttk

        for child in dlg.winfo_children():
            for widget in child.winfo_children():
                if (isinstance(widget, ttk.Combobox)
                        and widget.cget("textvariable") == str(dlg._vars["selection"])):
                    return widget
        raise AssertionError("selection combobox not found")

    def test_dropdown_lists_all_selections(self, dialog):
        _root, dlg = dialog
        combo = self._selection_combo(dlg)
        assert tuple(combo.cget("values")) == ("nearest", "first", "last", "all")
        assert str(combo.cget("state")) == "readonly"

    def test_existing_value_is_preselected(self, dialog):
        _root, dlg = dialog
        assert dlg._vars["selection"].get() == "first"

    def test_default_is_nearest(self):
        from measure_gui.dialogs import FitCircleDialog

        root = _tk.Tk()
        root.withdraw()
        dlg = FitCircleDialog(root, params={"num_measures": 12})
        root.update()
        try:
            assert dlg._vars["selection"].get() == "nearest"
        finally:
            dlg.destroy()
            root.destroy()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
