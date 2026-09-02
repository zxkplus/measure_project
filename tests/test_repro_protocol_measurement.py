"""
Unit tests for the protocol measurement reproducer (synthetic images only).

The real 8 MB debug images are NOT part of the repo — these tests validate
the pure geometry/grey-crossing logic on small generated images:

  * Liang-Barsky clip, inclusive-threshold crossings (grey == 185 counts as
    dark — verified against real segment endpoints),
  * point (direction filter + nearest selection) / segment (outer pair) /
    circle (radial band + fit),
  * object-level res combination rules and half-away rounding.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.repro_protocol_measurement import (
    CONFIG,
    _clip_line_to_rect,
    _round_val,
    compute_res,
    detect_circle,
    detect_point,
    detect_segment,
    edge_crossings,
)


def make_cfg(**over) -> dict:
    c = dict(CONFIG)
    c.update(over)
    return c


# ---------------------------------------------------------------------------
# Clipping
# ---------------------------------------------------------------------------


class TestClipLineToRect:
    def test_fully_inside(self):
        assert _clip_line_to_rect((10, 10, 200, 10), (0, 0, 300, 300)) == (0.0, 1.0)

    def test_partially_clipped(self):
        lo, hi = _clip_line_to_rect((-50, 25, 300, 25), (0, 0, 200, 50))
        assert lo is not None
        # kept param interval covers x in [0, 200] of a 350-long segment
        assert abs(lo - 50 / 350) < 1e-9
        assert abs(hi - 250 / 350) < 1e-9

    def test_miss(self):
        assert _clip_line_to_rect((10, 10, 20, 20), (100, 100, 200, 200)) is None


# ---------------------------------------------------------------------------
# Grey crossings (inclusive threshold semantics)
# ---------------------------------------------------------------------------


class TestEdgeCrossings:
    def _row_image(self, values: list) -> np.ndarray:
        img = np.full((40, len(values)), 128, dtype=np.uint8)
        img[20, :] = values
        return img

    def test_bright_dark_bright_region(self):
        # row: bright 200, dark 30 from x=100..199, bright 200 again
        row = np.full(300, 200, dtype=np.uint8)
        row[100:200] = 30
        img = self._row_image(row.tolist())
        cr = edge_crossings((0, 20, 299, 20), (0, 0, 299, 39), 185, img)
        assert len(cr) == 2
        c0, c1 = cr
        assert c0["dir"] == "down" and abs(c0["x"] - 100.0) <= 1.0
        assert c1["dir"] == "up" and abs(c1["x"] - 200.0) <= 1.0

    def test_grey_equal_threshold_counts_as_dark(self):
        # pixel exactly 185 must be treated as dark (<= 185) and create a
        # crossing: 200 -> 185 flips state.
        row = np.full(40, 128, dtype=np.uint8)
        row[10] = 200
        row[11] = 185
        row[12] = 180
        img = self._row_image(row.tolist())
        cr = edge_crossings((0, 20, 39, 20), (0, 0, 39, 39), 185, img)
        downs = [c for c in cr if c["dir"] == "down"]
        assert any(abs(c["x"] - 11.0) < 1.0 for c in downs)

    def test_roi_restricts_crossings(self):
        row = np.full(300, 200, dtype=np.uint8)
        row[100:200] = 30
        img = self._row_image(row.tolist())
        cr = edge_crossings((0, 20, 299, 20), (120, 0, 170, 39), 185, img)
        assert len(cr) == 0  # dark region entirely inside clipped-out roi

    def test_vertical_line(self):
        img = np.full((300, 40), 200, dtype=np.uint8)
        img[100:200, 20] = 30
        cr = edge_crossings((20, 0, 20, 299), (0, 0, 39, 299), 185, img)
        assert len(cr) == 2
        assert cr[0]["dir"] == "down" and abs(cr[0]["y"] - 100.0) <= 1.0
        assert cr[1]["dir"] == "up" and abs(cr[1]["y"] - 200.0) <= 1.0


# ---------------------------------------------------------------------------
# Point / segment detection
# ---------------------------------------------------------------------------


class TestPointSegment:
    def _img(self, dark_x0=0, dark_x1=299):
        # horizontal dark band y 60..120 in bright 200 background; darkness
        # only spans x in [dark_x0, dark_x1)
        img = np.full((200, 300), 200, dtype=np.uint8)
        img[60:121, dark_x0:dark_x1] = 30
        return img

    def test_point_nearest_down(self):
        img = self._img()
        elem = {
            "type": "point", "coordinates": [150, 50],
            "roi": [0, 0, 299, 199], "line": [150, 0, 150, 199],
            "threshold": 185, "isInv": True,
        }
        out = detect_point(elem, img, make_cfg())
        x, y = out["coordinates"]
        assert abs(x - 150) <= 1.0
        assert abs(y - 60.0) <= 2.0  # upper edge (down crossing, nearest)

    def test_point_isinv_flips_side(self):
        img = self._img()
        # isInv False -> look for 'up' crossing -> lower edge y=120
        elem = {
            "type": "point", "coordinates": [150, 150],
            "roi": [0, 0, 299, 199], "line": [150, 0, 150, 199],
            "threshold": 185, "isInv": False,
        }
        out = detect_point(elem, img, make_cfg())
        assert out["coordinates"] is not None
        _, y = out["coordinates"]
        assert abs(y - 120.0) <= 2.0

    def test_segment_outer_pair(self):
        # dark band only from x=100..199 -> crossings at x=100/200
        img = self._img(dark_x0=100, dark_x1=200)
        elem = {
            "type": "segment", "coordinates": [0, 90, 299, 90],
            "roi": [0, 0, 299, 199], "line": [0, 90, 299, 90],
            "threshold": 185, "isInv": True,
        }
        out = detect_segment(elem, img, make_cfg())
        x0, y0, x1, y1 = out["coordinates"]
        assert abs(x0 - 100.0) <= 2.0 and abs(x1 - 200.0) <= 2.0
        assert y0 == y1 == 90


class TestCircle:
    def test_dark_disc_fit(self):
        img = np.full((240, 240), 220, dtype=np.uint8)
        cv2 = pytest.importorskip("cv2")
        cv2.circle(img, (120, 120), 40, 30, -1)  # dark disc radius 40
        elem = {
            "type": "circle", "coordinates": [120, 120, 40],
            "roi": [0, 0, 239, 239], "line": [], "threshold": 185, "isInv": True,
        }
        out = detect_circle(elem, img, make_cfg())
        cx, cy, r = out["coordinates"]
        assert abs(cx - 120) <= 2 and abs(cy - 120) <= 2
        assert abs(r - 40) <= 2


# ---------------------------------------------------------------------------
# Result combination + rounding
# ---------------------------------------------------------------------------


class TestResRules:
    def test_height(self):
        elems = [{"type": "point", "coords_rounded": [100, 329]}]
        assert compute_res("height", elems, 5472) == pytest.approx(5143)

    def test_distance_single_segment(self):
        elems = [{"type": "segment", "coords_rounded": [718, 542, 2806, 542]}]
        assert compute_res("distance", elems, 5472) == pytest.approx(2088)

    def test_distance_two_points(self):
        elems = [
            {"type": "point", "coords_rounded": [578, 1550]},
            {"type": "point", "coords_rounded": [2087, 1550]},
        ]
        assert compute_res("distance", elems, 5472) == pytest.approx(1509)

    def test_distance_point_segment(self):
        elems = [
            {"type": "point", "coords_rounded": [1743, 329]},
            {"type": "segment", "coords_rounded": [687, 743, 2845, 743]},
        ]
        assert compute_res("distance", elems, 5472) == pytest.approx(414)

    def test_diameter_circle(self):
        elems = [{"type": "circle", "coords_rounded": [3358, 1529, 732]}]
        assert compute_res("diameter", elems, 3648) == pytest.approx(732)

    def test_missing_coords_yields_zero(self):
        elems = [{"type": "point", "coords_rounded": None}]
        assert compute_res("distance", elems, 5472) == 0.0


class TestRounding:
    def test_half_away(self):
        assert _round_val(328.5, "half_away") == 329
        assert _round_val(327.4, "half_away") == 327
        assert _round_val(2806.43, "half_away") == 2806
        assert _round_val(-1.5, "half_away") == -2
