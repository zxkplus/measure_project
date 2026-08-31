"""
Pixel ↔ physical scale calibration from a (possibly partial) checkerboard.

This module is intentionally lightweight: it only estimates the uniform
pixel-to-physical conversion ratio (mm / pixel) that is valid under a
telecentric (orthographic) lens.  It performs **no** distortion correction
and computes no intrinsics — the user supplies a single checkerboard image
(the full board may be *partially* visible) and the physical square side
length; the ratio is derived from the average adjacent-corner spacing.

Key idea for partial boards:
  ``cv2.findChessboardCorners`` requires the FULL grid to be visible and
  fails on partial views.  Instead we detect corners (Shi-Tomasi), then
  estimate the lattice spacing from the *nearest-neighbour distance
  distribution* of the detected corners.  Because the scale is uniform
  (telecentric), any detected sub-lattice yields the same ratio, so a
  partial board is enough.

Usage:
    from measure import CheckerboardScaleCalibration

    calib = CheckerboardScaleCalibration()
    result = calib.calibrate(board_image, square_size_mm=5.0)
    if result.valid:
        print(result.scale_mm_per_px)          # mm / pixel
        pts = result.corners                    # detected corners (row, col)
        # ... user edits pts (add / delete / drag) ...
        result2 = calib.compute_scale(pts, square_size_mm=5.0)   # re-estimate

    calib.save("calib.json")                    # persist
    calib2 = CheckerboardScaleCalibration.load("calib.json")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from measure.viz import to_bgr

# Minimum corner count required for a meaningful spacing estimate.
MIN_CORNERS = 6
# Nearest neighbours considered per corner (4 orthogonal neighbours in a
# chessboard lattice).
_NEIGHBOUR_K = 4
# Outlier filter: keep nearest-neighbour distances within
# [OUTLIER_FACTOR * median, 1 / OUTLIER_FACTOR * median].
_OUTLIER_FACTOR = 0.5
# Minimum ROI side length (px) considered valid for detection.
_MIN_ROI_SIDE = 5


def _normalize_roi(
    roi: Tuple[int, int, int, int],
    image_shape: Tuple[int, ...],
) -> Tuple[int, int, int, int]:
    """Clamp + validate an ROI box to the image bounds.

    Args:
        roi: (row0, col0, row1, col1) in full-image coords (inclusive).
        image_shape: (h, w) of the image.

    Returns:
        Normalized (row0, col0, row1, col1) with row0 <= row1, col0 <= col1
        and both inside the image.

    Raises:
        ValueError: If the ROI is degenerate (too small to detect on).
    """
    h, w = image_shape[:2]
    r0, c0, r1, c1 = (int(v) for v in roi)
    r0, r1 = min(r0, r1), max(r0, r1)
    c0, c1 = min(c0, c1), max(c0, c1)
    r0, c0 = max(0, r0), max(0, c0)
    r1, c1 = min(h - 1, r1), min(w - 1, c1)
    if r1 - r0 + 1 < _MIN_ROI_SIDE or c1 - c0 + 1 < _MIN_ROI_SIDE:
        raise ValueError(
            f"拉框区域太小（{r1 - r0 + 1}×{c1 - c0 + 1} px），"
            f"请框选更大的棋盘格区域。"
        )
    return r0, c0, r1, c1


@dataclass
class CalibrationResult:
    """Outcome of a checkerboard pixel-scale calibration."""

    valid: bool
    scale_mm_per_px: float = 0.0
    spacing_px: float = 0.0
    num_corners: int = 0
    method: str = "lattice"
    corners: List[Tuple[float, float]] = field(default_factory=list)
    overlay_image: Optional[np.ndarray] = None
    error: str = ""


class CheckerboardScaleCalibration:
    """Estimate the mm/px scale from a (partially visible) checkerboard.

    Lifecycle:  __init__ → calibrate(image, square_size_mm) / compute_scale(...)
                → save() / load()
    """

    def __init__(self) -> None:
        self.valid: bool = False
        self.scale_mm_per_px: float = 0.0
        self.spacing_px: float = 0.0
        self.num_corners: int = 0
        self.method: str = "lattice"
        self.square_size_mm: float = 0.0
        self.error: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calibrate(
        self,
        image: np.ndarray,
        square_size_mm: float,
        board_grid: Optional[Tuple[int, int]] = None,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> CalibrationResult:
        """Detect checkerboard corners and compute the mm/px scale.

        Args:
            image: Board image (grayscale uint8 or BGR).
            square_size_mm: Physical side length of one square (mm), > 0.
            board_grid: Optional (cols, rows) inner-corner grid size, e.g.
                (9, 6).  When given, ``cv2.findChessboardCornersSB`` is tried
                first (more accurate when the full grid is visible); falls
                back to the lattice method otherwise.  The lattice method is
                always used as the primary path because it tolerates partial
                boards.
            roi: Optional detection region ``(row0, col0, row1, col1)`` in
                full-image coordinates (inclusive bounds).  When given, corner
                detection runs only inside this box — use it to exclude
                background texture outside the checkerboard.  Detected corners
                are reported back in full-image coordinates.

        Returns:
            CalibrationResult with detected corners and overlay image.
        """
        self._validate_square_size(square_size_mm)
        self.square_size_mm = float(square_size_mm)

        if roi is not None:
            r0, c0, r1, c1 = _normalize_roi(roi, image.shape)
            sub = image[r0:r1 + 1, c0:c1 + 1]
            corners, method = self._detect_corners(sub, board_grid=board_grid)
            # Offset crop-local corners back to full-image coordinates.
            corners = [(r + r0, c + c0) for r, c in corners]
        else:
            corners, method = self._detect_corners(image, board_grid=board_grid)

        result = self.compute_scale(corners, square_size_mm)
        result.method = method
        result.corners = list(corners)
        result.overlay_image = self._draw_corners(image, corners)
        self._store(result)
        return result

    def compute_scale(
        self,
        corners: List[Tuple[float, float]],
        square_size_mm: float,
    ) -> CalibrationResult:
        """Estimate the mm/px scale from an explicit corner list.

        This is the entry point used by the GUI after the user manually
        edits (adds / deletes / drags) detected corners, so the ratio can be
        recomputed live without re-running detection.

        Args:
            corners: List of (row, col) corner points in pixel coords.
            square_size_mm: Physical side length of one square (mm), > 0.

        Returns:
            CalibrationResult (``corners`` and ``overlay_image`` left as-is;
            the GUI sets them when needed).
        """
        self._validate_square_size(square_size_mm)
        self.square_size_mm = float(square_size_mm)

        spacing = self._estimate_spacing(corners)
        n = len(corners)

        if spacing is None or spacing <= 0:
            error = (
                f"角点过少或间距无法估计（当前 {n} 个角点，至少需要 "
                f"{MIN_CORNERS} 个）。请增大棋盘格在画面中的占比，"
                f"或手动添加角点。"
            )
            result = CalibrationResult(
                valid=False,
                num_corners=n,
                method=self.method,
                corners=list(corners),
                error=error,
            )
            self._store(result)
            return result

        scale = self.square_size_mm / spacing
        result = CalibrationResult(
            valid=True,
            scale_mm_per_px=scale,
            spacing_px=spacing,
            num_corners=n,
            method=self.method,
            corners=list(corners),
        )
        self._store(result)
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filepath: str) -> None:
        """Serialize calibration state (no corner list / overlay) to JSON."""
        data: Dict[str, Any] = {
            "valid": self.valid,
            "scale_mm_per_px": self.scale_mm_per_px,
            "spacing_px": self.spacing_px,
            "num_corners": self.num_corners,
            "method": self.method,
            "square_size_mm": self.square_size_mm,
            "error": self.error,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "CheckerboardScaleCalibration":
        """Restore calibration state from a JSON file written by save()."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        calib = cls()
        calib.valid = bool(data.get("valid", False))
        calib.scale_mm_per_px = float(data.get("scale_mm_per_px", 0.0))
        calib.spacing_px = float(data.get("spacing_px", 0.0))
        calib.num_corners = int(data.get("num_corners", 0))
        calib.method = str(data.get("method", "lattice"))
        calib.square_size_mm = float(data.get("square_size_mm", 0.0))
        calib.error = str(data.get("error", ""))
        return calib

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _store(self, result: CalibrationResult) -> None:
        """Mirror a CalibrationResult onto the instance state."""
        self.valid = result.valid
        self.scale_mm_per_px = result.scale_mm_per_px
        self.spacing_px = result.spacing_px
        self.num_corners = result.num_corners
        self.method = result.method
        self.error = result.error

    @staticmethod
    def _validate_square_size(square_size_mm: float) -> None:
        if square_size_mm is None or not isinstance(square_size_mm, (int, float)):
            raise ValueError(f"棋盘格边长必须是数字，得到 {square_size_mm!r}")
        if square_size_mm <= 0:
            raise ValueError(f"棋盘格边长必须 > 0，得到 {square_size_mm}")

    def _detect_corners(
        self,
        image: np.ndarray,
        board_grid: Optional[Tuple[int, int]] = None,
    ) -> Tuple[List[Tuple[float, float]], str]:
        """Return ((row, col) corner list, method name)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 \
            else image

        # Optional precise full-grid detection (only when grid size is known).
        if board_grid is not None:
            sb_corners = self._try_find_chessboard(gray, board_grid)
            if sb_corners is not None:
                return sb_corners, "findChessboardCornersSB"

        # Primary path: Shi-Tomasi corners + lattice spacing estimation.
        corners = self._shitiomasi_corners(gray)
        return corners, "lattice"

    @staticmethod
    def _try_find_chessboard(
        gray: np.ndarray,
        grid: Tuple[int, int],
    ) -> Optional[List[Tuple[float, float]]]:
        """Try full-grid detection; return None if the grid is not visible."""
        flags = cv2.CALIB_CB_FAST_CHECK
        found, pts = cv2.findChessboardCornersSB(gray, grid, flags)
        if not found or pts is None:
            return None
        # pts: shape (N, 1, 2) with (col, row) -> convert to (row, col).
        pts = pts.reshape(-1, 2)
        return [(float(row), float(col)) for col, row in pts]

    @staticmethod
    def _shitiomasi_corners(
        gray: np.ndarray,
        max_corners: int = 2000,
        quality_level: float = 0.05,
        min_distance: float = 4.0,
    ) -> List[Tuple[float, float]]:
        """Detect checkerboard corners via Shi-Tomasi + subpixel refinement.

        Returns (row, col) corner list (possibly empty).
        """
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        pts = cv2.goodFeaturesToTrack(
            blurred,
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=min_distance,
        )
        if pts is None or len(pts) == 0:
            return []
        pts = pts.reshape(-1, 2)  # (col, row)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        pts = cv2.cornerSubPix(blurred, np.float32(pts), (5, 5), (-1, -1), criteria)
        return [(float(row), float(col)) for col, row in pts]

    @staticmethod
    def _estimate_spacing(
        corners: List[Tuple[float, float]],
    ) -> Optional[float]:
        """Estimate the checkerboard lattice spacing (px) from corners.

        Uses the median nearest-neighbour distance with outlier rejection.
        Interior corners have 4 orthogonal neighbours at exactly one square
        spacing; the median over all corners is robust to a partial board and
        to a few stray corners.

        Returns None when the estimate is not meaningful (too few corners).
        """
        n = len(corners)
        if n < MIN_CORNERS:
            return None

        pts = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
        from scipy.spatial import cKDTree

        tree = cKDTree(pts)
        k = min(_NEIGHBOUR_K + 1, n)  # +1 to exclude the point itself
        dists, _ = tree.query(pts, k=k)
        if k == 1:
            dists = dists.reshape(-1, 1)
        # Exclude self (first column).
        nn = dists[:, 1:]
        all_d = nn[nn > 0]
        if all_d.size < MIN_CORNERS:
            return None

        median_d = float(np.median(all_d))
        lo = _OUTLIER_FACTOR * median_d
        hi = median_d / _OUTLIER_FACTOR
        keep = all_d[(all_d >= lo) & (all_d <= hi)]
        if keep.size == 0:
            return None
        return float(np.median(keep))

    @staticmethod
    def _draw_corners(
        image: np.ndarray,
        corners: List[Tuple[float, float]],
    ) -> np.ndarray:
        """Return a BGR copy of the image with detected corners drawn."""
        vis = to_bgr(image)
        for row, col in corners:
            x, y = int(round(col)), int(round(row))
            cv2.circle(vis, (x, y), 3, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.drawMarker(
                vis, (x, y), (255, 0, 0), cv2.MARKER_CROSS, 12, 1, cv2.LINE_AA,
            )
        return vis


# ===========================================================================
# Module-level conversion helpers (shared by GUI display layers)
# ===========================================================================


def to_physical(value_px: float, scale_mm_per_px: float) -> float:
    """Convert a pixel length to physical length (mm)."""
    return value_px * scale_mm_per_px


def format_length(
    value_mm: float,
    value_px: float,
    decimals: int = 3,
) -> str:
    """Format a physical + pixel length pair, e.g. '12.345 mm (123.5 px)'."""
    return f"{value_mm:.{decimals}f} mm ({value_px:.{decimals}f} px)"
