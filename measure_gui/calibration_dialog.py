"""
Interactive checkerboard pixel-scale calibration dialog.

Workflow (anti false-positive):
  1. Load a (possibly partially visible) checkerboard image.
  2. Draw a ROI box around the board (click center → drag to size →
     scroll to rotate → double-click to confirm).  Only corners INSIDE the
     box are detected, excluding background texture.
  3. Detection thresholds (角点质量 / 最小间距 / 最大角点数) are tunable and
     re-run detection live.  Corners that do not fit the checkerboard
     lattice are filtered out automatically.
  4. Detected corners are shown on the canvas and can be edited
     interactively: left-click empty = add, left-drag = move, right-click =
     delete.  The mm/px scale is recomputed live after every edit.
  5. "应用标定" writes the calibration onto the app (and workflow);
     "清除标定" removes it.  Calibration is optional — without it
     measurements stay in px.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from measure.pixel_calibration import CheckerboardScaleCalibration
from .image_canvas import CanvasMode, ImageCanvas
from .utils import compute_rotated_box_corners

# Interaction hint strings.
_HINT_DRAW_ROI = ("请在棋盘格区域拉框：点击定中心 → 拖拽调大小 → "
                  "滚轮旋转 → 双击确认")
_HINT_EDIT_CORNERS = "左键空白=添加角点 ｜ 左键拖拽=移动角点 ｜ 右键=删除角点"


class _CornerCanvas(ImageCanvas):
    """ImageCanvas that renders corners as compact markers.

    The base control-point renderer draws per-point index labels and a
    dashed polyline connecting all points in order, which is cluttered for a
    dense checkerboard lattice.  This override renders small cross markers
    only, while keeping the base add / drag / delete hit-testing.
    """

    def _redraw_control_points(self) -> None:
        canvas = self._canvas
        for row, col in self._control_points:
            cx, cy = self._img_to_canvas(row, col)
            self._cp_items.append(
                canvas.create_oval(cx - 3, cy - 3, cx + 3, cy + 3,
                                   outline="#00e676", width=1)
            )
            self._cp_items.append(
                canvas.create_line(cx - 7, cy, cx + 7, cy, fill="#00e676")
            )
            self._cp_items.append(
                canvas.create_line(cx, cy - 7, cx, cy + 7, fill="#00e676")
            )


class CalibrationDialog(tk.Toplevel):
    """Interactive checkerboard pixel-scale calibration dialog.

    Args:
        master: Parent widget (the MeasureApp).
        on_apply: ``Callable[[dict], None]`` — receives a calibration dict
            with keys ``scale_mm_per_px``, ``square_size_mm``, ``num_corners``,
            ``method``, ``board_image`` (np.ndarray or None).
        on_clear: ``Callable[[], None]`` — clears the applied calibration.
    """

    def __init__(
        self,
        master: tk.Widget,
        on_apply: Callable[[Dict[str, Any]], None],
        on_clear: Callable[[], None],
    ) -> None:
        super().__init__(master)
        self.title("棋盘格标定")
        self.geometry("920x680")
        self.resizable(True, True)

        self.on_apply = on_apply
        self.on_clear = on_clear

        self._calib = CheckerboardScaleCalibration()
        self._board_image: Optional[np.ndarray] = None
        self._square_mm: Optional[float] = None
        self._current: Optional[Any] = None  # last CalibrationResult
        self._roi_bbox: Optional[Tuple[int, int, int, int]] = None
        #   (row0, col0, row1, col1) detection region in full-image coords.

        self.transient(master)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build_ui()
        self.after(120, self._safe_grab_set)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        top = ttk.Frame(self)
        top.pack(fill=tk.X, padx=8, pady=6)

        ttk.Button(top, text="打开标定板图...",
                   command=self._load_image).pack(side=tk.LEFT)

        ttk.Label(top, text="  棋盘格边长 (mm):").pack(side=tk.LEFT, padx=(14, 2))
        self._size_var = tk.StringVar(value="5.0")
        self._size_entry = ttk.Entry(top, textvariable=self._size_var, width=8)
        self._size_entry.pack(side=tk.LEFT)

        self._detect_btn = ttk.Button(top, text="检测框内角点",
                                      command=self._detect, state=tk.DISABLED)
        self._detect_btn.pack(side=tk.LEFT, padx=(14, 0))

        ttk.Button(top, text="重新拉框",
                   command=self._redraw_roi).pack(side=tk.LEFT, padx=(8, 0))

        # --- Tunable detection thresholds (live re-detect) ---
        tune = ttk.Frame(self)
        tune.pack(fill=tk.X, padx=8, pady=(0, 2))
        ttk.Label(tune, text="检测阈值:").pack(side=tk.LEFT)
        ttk.Label(tune, text="角点质量").pack(side=tk.LEFT, padx=(10, 2))
        self._quality_var = tk.StringVar(value="0.05")
        ttk.Spinbox(tune, from_=0.001, to=0.5, increment=0.005, width=7,
                    textvariable=self._quality_var).pack(side=tk.LEFT)
        ttk.Label(tune, text="最小间距(px)").pack(side=tk.LEFT, padx=(10, 2))
        self._min_dist_var = tk.StringVar(value="4")
        ttk.Spinbox(tune, from_=1, to=50, increment=1, width=5,
                    textvariable=self._min_dist_var).pack(side=tk.LEFT)
        ttk.Label(tune, text="最大角点数").pack(side=tk.LEFT, padx=(10, 2))
        self._max_corners_var = tk.StringVar(value="2000")
        ttk.Spinbox(tune, from_=10, to=10000, increment=100, width=7,
                    textvariable=self._max_corners_var).pack(side=tk.LEFT)
        ttk.Label(tune, text="  （误检多 → 调高角点质量/最小间距）",
                  foreground="gray").pack(side=tk.LEFT, padx=(14, 0))
        for _var in (self._quality_var, self._min_dist_var,
                     self._max_corners_var):
            _var.trace_add("write", self._on_params_changed)

        # Canvas for the board image + interactive corners
        self._canvas = _CornerCanvas(self, width=880, height=520)
        self._canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=2)
        self._canvas.on_control_points_changed = self._on_corners_changed
        self._canvas.on_roi_confirmed = self._on_roi_confirmed

        self._hint_var = tk.StringVar(value=_HINT_DRAW_ROI)
        ttk.Label(self, textvariable=self._hint_var,
                  foreground="gray").pack(fill=tk.X, padx=8)

        self._result_var = tk.StringVar(value="未检测")
        ttk.Label(self, textvariable=self._result_var,
                  foreground="#00a05e").pack(fill=tk.X, padx=8, pady=(2, 0))

        bottom = ttk.Frame(self)
        bottom.pack(fill=tk.X, padx=8, pady=6)
        ttk.Button(bottom, text="应用标定", command=self._apply).pack(side=tk.LEFT)
        ttk.Button(bottom, text="清除标定",
                   command=self._clear).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(bottom, text="关闭", command=self._on_close).pack(side=tk.RIGHT)

    def _safe_grab_set(self) -> None:
        try:
            self.grab_set()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _load_image(self) -> None:
        path = filedialog.askopenfilename(
            parent=self, title="选择标定板图片",
            filetypes=[("图片", "*.png *.jpg *.jpeg *.bmp"), ("所有文件", "*.*")],
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("错误", f"无法读取图片:\n{path}", parent=self)
            return
        self._board_image = img
        self._canvas.load_image(img)
        self._canvas.zoom_to_fit()
        self._canvas.set_mode(CanvasMode.DRAW_ROI)
        self._canvas.reset_roi()
        self._canvas.clear_control_points()
        self._roi_bbox = None
        self._current = None
        self._detect_btn.config(state=tk.NORMAL)
        self._hint_var.set(_HINT_DRAW_ROI)
        self._result_var.set("已加载图片，请拉框框住棋盘格区域")

    def _read_square_size(self) -> Optional[float]:
        """Read + validate the square side length.  Returns None on error."""
        try:
            value = float(self._size_var.get())
        except (ValueError, TypeError):
            messagebox.showerror("输入错误", "棋盘格边长必须是数字（mm）", parent=self)
            return None
        if value <= 0:
            messagebox.showerror("输入错误", "棋盘格边长必须大于 0", parent=self)
            return None
        return value

    def _on_roi_confirmed(self, center_row: float, center_col: float,
                          height: float, width: float, angle_deg: float) -> None:
        """ROI drawn/confirmed by double-click → store bbox + auto-detect.

        Detection is deferred with ``after(0)`` because ``ImageCanvas.
        confirm_roi()`` switches the canvas back to BROWSE *after* invoking
        this callback; running detection on the next idle tick lets it finish
        with DRAW_CONTROL_POINTS as the final mode.
        """
        if self._board_image is None:
            return
        corners = compute_rotated_box_corners(
            (center_row, center_col), (height, width), angle_deg)
        rows = [r for r, _ in corners]
        cols = [c for _, c in corners]
        # Axis-aligned bounding box of the (possibly rotated) ROI.
        self._roi_bbox = (
            int(round(min(rows))), int(round(min(cols))),
            int(round(max(rows))), int(round(max(cols))),
        )
        self.after(0, self._detect)

    def _redraw_roi(self) -> None:
        """Clear the box/corners and go back to ROI-drawing mode."""
        self._canvas.set_mode(CanvasMode.DRAW_ROI)
        self._canvas.reset_roi()
        self._canvas.clear_control_points()
        self._roi_bbox = None
        self._current = None
        self._hint_var.set(_HINT_DRAW_ROI)
        self._result_var.set("请重新拉框框住棋盘格区域")

    def _on_params_changed(self, *_args) -> None:
        """Debounced handler for the threshold spinboxes."""
        after_id = getattr(self, "_param_after", None)
        if after_id:
            self.after_cancel(after_id)
        self._param_after = self.after(250, self._apply_params)

    def _apply_params(self) -> None:
        """Push tuned thresholds into the detector and re-run detection."""
        self._param_after = None
        try:
            q = float(self._quality_var.get())
            md = float(self._min_dist_var.get())
            mc = int(float(self._max_corners_var.get()))
        except ValueError:
            return
        if not (0 < q <= 1 and md >= 1 and mc >= 1):
            return
        self._calib.quality_level = q
        self._calib.min_distance = md
        self._calib.max_corners = mc
        if self._board_image is not None and self._roi_bbox is not None:
            self._detect(quiet=True)

    def _detect(self, quiet: bool = False) -> None:
        if self._board_image is None:
            return
        if self._roi_bbox is None:
            messagebox.showwarning(
                "提示", "请先在棋盘格区域拉框（点击定中心 → 拖拽调大小 → "
                        "滚轮旋转 → 双击确认）。", parent=self)
            return
        square_mm = self._read_square_size()
        if square_mm is None:
            return
        try:
            result = self._calib.calibrate(
                self._board_image, square_mm, roi=self._roi_bbox)
        except ValueError as e:
            if not quiet:
                messagebox.showerror("输入错误", str(e), parent=self)
            return

        self._current = result
        self._square_mm = square_mm
        # Switch to corner-editing mode (even on failure, so the user can add
        # corners manually).
        self._canvas.set_mode(CanvasMode.DRAW_CONTROL_POINTS)
        self._canvas.set_control_points(result.corners)
        self._hint_var.set(_HINT_EDIT_CORNERS)
        if result.valid:
            self._show_result(result)
        else:
            self._result_var.set("检测失败")
            if not quiet:
                messagebox.showerror("标定失败", result.error, parent=self)

    def _on_corners_changed(self, points: List[Tuple[float, float]]) -> None:
        """Recompute the scale live from the edited corner set."""
        square_mm = self._read_square_size()
        if square_mm is None:
            return
        self._square_mm = square_mm
        result = self._calib.compute_scale(points, square_mm)
        self._current = result
        self._show_result(result)

    def _show_result(self, result: Any) -> None:
        if result.valid:
            self._result_var.set(
                f"比例: {result.scale_mm_per_px:.4f} mm/px  ｜  "
                f"角点: {result.num_corners}  ｜  "
                f"网格间距: {result.spacing_px:.2f} px  ｜  "
                f"边长: {self._square_mm:.3f} mm"
            )
        else:
            self._result_var.set(
                f"角点不足，无法计算比例（{result.error}）"
            )

    def _apply(self) -> None:
        if self._current is None or not self._current.valid:
            messagebox.showwarning(
                "提示", "请先完成有效的标定检测/角点编辑", parent=self)
            return
        data: Dict[str, Any] = {
            "scale_mm_per_px": self._current.scale_mm_per_px,
            "square_size_mm": self._square_mm or self._current.square_size_mm,
            "num_corners": self._current.num_corners,
            "method": self._current.method,
            "board_image": self._board_image,
        }
        if self.on_apply:
            self.on_apply(data)
        self.destroy()

    def _clear(self) -> None:
        if self.on_clear:
            self.on_clear()
        self._canvas.set_mode(CanvasMode.DRAW_ROI)
        self._canvas.reset_roi()
        self._canvas.clear_control_points()
        self._current = None
        self._square_mm = None
        self._roi_bbox = None
        self._hint_var.set(_HINT_DRAW_ROI)
        self._result_var.set("已清除标定")
        messagebox.showinfo(
            "已清除", "标定已清除，测量结果将使用像素单位。", parent=self)

    def _on_close(self) -> None:
        self.destroy()
