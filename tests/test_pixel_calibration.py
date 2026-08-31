"""
Tests for the checkerboard pixel-scale calibration module.

Covers the lattice-spacing estimator on synthetic checkerboards: full board,
partial board (only a crop visible), rotated board, failure cases, manual
corner-editing recomputation, persistence, and the conversion helpers.
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np
import pytest

from measure import CheckerboardScaleCalibration, CalibrationResult
from measure.pixel_calibration import format_length, to_physical


# ---------------------------------------------------------------------------
# Synthetic checkerboard helpers
# ---------------------------------------------------------------------------


def make_checkerboard(
    squares_cols: int,
    squares_rows: int,
    square_px: int,
    value: int = 200,
) -> np.ndarray:
    """Build a clean synthetic checkerboard (grayscale uint8).

    Args:
        squares_cols, squares_rows: number of squares per axis.
        square_px: side length of one square in pixels.
        value: brightness of the bright squares (dark squares are 0).
    """
    img = np.zeros((squares_rows * square_px, squares_cols * square_px),
                   dtype=np.uint8)
    for r in range(squares_rows):
        for c in range(squares_cols):
            if (r + c) % 2 == 0:
                img[r * square_px:(r + 1) * square_px,
                    c * square_px:(c + 1) * square_px] = value
    # Slight blur so corners are a bit realistic (avoids perfect steps).
    return cv2.GaussianBlur(img, (3, 3), 0)


def assert_scale_ok(result: CalibrationResult, square_px: int,
                    square_mm: float, rel_tol: float = 0.01) -> None:
    """Assert a valid result whose scale matches square_mm / square_px."""
    assert result.valid, f"expected valid calibration, got error={result.error}"
    expected = square_mm / square_px
    assert result.scale_mm_per_px > 0
    assert abs(result.scale_mm_per_px - expected) / expected < rel_tol, (
        f"scale {result.scale_mm_per_px:.6f} vs expected {expected:.6f} "
        f"(spacing {result.spacing_px:.3f}px, corners {result.num_corners})"
    )


# ---------------------------------------------------------------------------
# Detection / estimation
# ---------------------------------------------------------------------------


class TestCheckerboardScaleCalibration:
    def test_full_board_scale(self, test_output_dir):
        square_px, square_mm = 50, 5.0
        img = make_checkerboard(10, 10, square_px)
        result = CheckerboardScaleCalibration().calibrate(img, square_mm)
        assert_scale_ok(result, square_px, square_mm)
        assert result.num_corners >= 6
        assert result.method == "lattice"
        # Overlay must be produced and be a BGR image.
        assert result.overlay_image is not None
        assert result.overlay_image.shape[2] == 3

    def test_partial_board_scale(self, test_output_dir):
        """Only a crop of a larger board is visible — scale must survive."""
        square_px, square_mm = 40, 10.0
        img = make_checkerboard(16, 16, square_px)
        # Keep only a 8x8-square sub-region in the middle.
        y0, y1 = 4 * square_px, 12 * square_px
        x0, x1 = 4 * square_px, 12 * square_px
        crop = img[y0:y1, x0:x1]
        assert crop.shape[0] == 8 * square_px
        result = CheckerboardScaleCalibration().calibrate(crop, square_mm)
        assert_scale_ok(result, square_px, square_mm)

    def test_rotated_board_scale(self, test_output_dir):
        square_px, square_mm = 45, 2.0
        img = make_checkerboard(10, 10, square_px)
        rows, cols = img.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 30.0, 1.0)
        rotated = cv2.warpAffine(img, M, (cols, rows),
                                 flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=128)
        result = CheckerboardScaleCalibration().calibrate(rotated, square_mm)
        # Slightly looser tolerance due to interpolation.
        assert_scale_ok(result, square_px, square_mm, rel_tol=0.02)

    def test_blank_image_invalid(self):
        result = CheckerboardScaleCalibration().calibrate(
            np.zeros((200, 200), dtype=np.uint8), 5.0)
        assert not result.valid
        assert result.error  # human-readable message present

    def test_smooth_gradient_invalid(self):
        grad = np.tile(np.linspace(0, 255, 200, dtype=np.uint8), (200, 1))
        result = CheckerboardScaleCalibration().calibrate(grad, 5.0)
        assert not result.valid

    def test_invalid_square_size_zero(self):
        img = make_checkerboard(10, 10, 50)
        with pytest.raises(ValueError):
            CheckerboardScaleCalibration().calibrate(img, 0.0)

    def test_invalid_square_size_negative(self):
        img = make_checkerboard(10, 10, 50)
        with pytest.raises(ValueError):
            CheckerboardScaleCalibration().calibrate(img, -1.0)

    def test_invalid_square_size_non_numeric(self):
        img = make_checkerboard(10, 10, 50)
        with pytest.raises(ValueError):
            CheckerboardScaleCalibration().calibrate(img, "abc")  # type: ignore


class TestManualEditRecompute:
    """After auto-detection, the GUI lets the user edit corners; the scale
    must be recomputed correctly from the edited set."""

    def _detect(self):
        square_px, square_mm = 50, 5.0
        img = make_checkerboard(10, 10, square_px)
        calib = CheckerboardScaleCalibration()
        result = calib.calibrate(img, square_mm)
        assert result.valid
        return calib, result, square_px, square_mm

    def test_remove_some_corners(self):
        calib, result, square_px, square_mm = self._detect()
        pts = list(result.corners)
        removed = pts[::5]  # remove every 5th
        kept = [p for i, p in enumerate(pts) if i % 5 != 0]
        assert len(kept) < len(pts)
        r2 = calib.compute_scale(kept, square_mm)
        assert_scale_ok(r2, square_px, square_mm)
        assert r2.num_corners == len(kept)

    def test_add_a_corner(self):
        calib, result, square_px, square_mm = self._detect()
        pts = list(result.corners)
        # Add a corner at a plausible lattice position (in the middle).
        row = min(p[0] for p in pts) + square_px * 2.0
        col = min(p[1] for p in pts) + square_px * 2.0
        edited = pts + [(row, col)]
        r2 = calib.compute_scale(edited, square_mm)
        assert_scale_ok(r2, square_px, square_mm)

    def test_too_few_corners_invalid(self):
        calib, result, square_px, square_mm = self._detect()
        pts = list(result.corners)
        too_few = pts[:5]
        r2 = calib.compute_scale(too_few, square_mm)
        assert not r2.valid
        assert r2.error


class TestPersistence:
    def test_save_load_roundtrip(self, test_output_dir):
        img = make_checkerboard(10, 10, 50)
        calib = CheckerboardScaleCalibration()
        result = calib.calibrate(img, 5.0)
        assert result.valid

        path = os.path.join(str(test_output_dir), "calib.json")
        calib.save(path)

        loaded = CheckerboardScaleCalibration.load(path)
        assert loaded.valid
        assert loaded.scale_mm_per_px == pytest.approx(
            calib.scale_mm_per_px)
        assert loaded.square_size_mm == pytest.approx(5.0)
        assert loaded.num_corners == calib.num_corners

    def test_load_missing_file(self):
        with pytest.raises(FileNotFoundError):
            CheckerboardScaleCalibration.load("/nonexistent/calib.json")


class TestHelpers:
    def test_to_physical(self):
        assert to_physical(100.0, 0.05) == pytest.approx(5.0)
        assert to_physical(0.0, 0.05) == 0.0

    def test_format_length(self):
        s = format_length(12.345, 246.9, decimals=3)
        assert "12.345 mm" in s
        assert "246.900 px" in s


class TestGuiFormat:
    """GUI summary/table formatting must show mm+px when calibrated and keep
    the old px-only output when uncalibrated."""

    def _make_distance(self, value=100.0):
        from measure.measure_workflow import DistanceResult
        return DistanceResult(label="gap", value=value, valid=True)

    def _make_circle(self, radius=50.0):
        from measure.measure_workflow import CircleResult
        return CircleResult(label="hole", center_row=0.0, center_col=0.0,
                            radius=radius, valid=True)

    def _make_line(self, p0=(0.0, 0.0), p1=(30.0, 40.0)):
        from measure.measure_workflow import LineResult
        return LineResult(label="edge", a=0.0, b=0.0, c=0.0,
                          start_row=p0[0], start_col=p0[1],
                          end_row=p1[0], end_col=p1[1], valid=True)

    def test_distance_no_scale_keeps_px(self):
        from measure_gui.multi_target import _format_geometric_result
        s = _format_geometric_result("gap", self._make_distance())
        assert "100.000 px" in s
        assert "mm" not in s

    def test_distance_with_scale_shows_both(self):
        from measure_gui.multi_target import _format_geometric_result
        s = _format_geometric_result("gap", self._make_distance(), scale_mm=0.05)
        assert "5.000 mm" in s
        assert "100.000 px" in s

    def test_circle_radius(self):
        from measure_gui.multi_target import _format_geometric_result
        r = self._make_circle(50.0)
        assert "50.00px" in _format_geometric_result("hole", r)
        assert "5.00 mm" in _format_geometric_result("hole", r, scale_mm=0.1)

    def test_line_length(self):
        from measure_gui.multi_target import _format_geometric_result
        line = self._make_line()  # length = 50 px
        s = _format_geometric_result("edge", line, scale_mm=0.02)
        assert "1.000 mm" in s
        assert "50.000 px" in s
        s0 = _format_geometric_result("edge", line)
        assert "50.000 px" in s0

    def test_point_and_angle_unchanged(self):
        from measure.measure_workflow import AngleResult, PointResult
        from measure_gui.multi_target import _format_geometric_result
        pt = PointResult(label="p", row=10.0, col=20.0, valid=True)
        ang = AngleResult(label="a", value_rad=0.5, valid=True)
        assert "row=10.00, col=20.00" in _format_geometric_result("p", pt, scale_mm=0.05)
        assert "28.65°" in _format_geometric_result("a", ang, scale_mm=0.05)

    def test_result_dict_with_scale(self):
        from measure_gui.multi_target import _format_result_dict
        d = {"type": "distance", "label": "gap", "valid": True, "value": 200.0}
        assert "2.000 mm" in _format_result_dict("gap", d, scale_mm=0.01)
        assert "200.000 px" in _format_result_dict("gap", d, scale_mm=0.01)
