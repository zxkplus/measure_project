"""Preprocessing preview dialog.

Lets an operator see, before committing a template, what each preprocessor does
to the template patch and to the inspection image.  The dialog is read-only: it
renders ``measure.preview`` output and never touches the workflow, the model
asset on disk, or the saved ``preprocessor`` setting.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from measure.measure_template import preprocessor_from_config
from measure.preview import (
    PREPROCESSOR_TYPES,
    PreviewColumn,
    default_configs,
    preview_matrix,
    preview_pair,
)

from .utils import cv2_to_tk

_ZOOM_MODES = ("适应窗口", "50%", "100%")

#: The default named font reports an 11 px line space, which clips CJK glyphs.
#: Naming the family explicitly asks Tk for the font's real extent instead.
_CAPTION_FONT = ("TkDefaultFont", 12)


class _ImageView(ttk.Frame):
    """Canvas-backed image view with fit/fixed zoom and drag-to-pan."""

    def __init__(
        self,
        parent: tk.Widget,
        caption: str = "",
        width: int = 440,
        height: int = 240,
    ) -> None:
        super().__init__(parent)
        self._caption_var = tk.StringVar(value=caption)
        ttk.Label(self, textvariable=self._caption_var, font=_CAPTION_FONT,
                  wraplength=width).pack(anchor=tk.W, pady=(0, 2))

        self._zoom_var = tk.StringVar(value=_ZOOM_MODES[0])
        zoom = ttk.Combobox(
            self, textvariable=self._zoom_var, values=list(_ZOOM_MODES),
            state="readonly", width=12,
        )
        zoom.pack(anchor=tk.W, pady=(2, 4))
        zoom.bind("<<ComboboxSelected>>", lambda _e: self._on_zoom_changed())

        self._canvas = tk.Canvas(
            self, width=width, height=height, bg="#1e1e1e",
            highlightthickness=1, highlightbackground="#555555",
        )
        self._canvas.pack(fill=tk.BOTH, expand=True)
        self._canvas.bind("<Configure>", lambda _e: self._redraw())
        self._canvas.bind("<ButtonPress-1>", self._on_press)
        self._canvas.bind("<B1-Motion>", self._on_drag)

        self._stats_var = tk.StringVar(value="")
        ttk.Label(self, textvariable=self._stats_var, foreground="#555555",
                  wraplength=width).pack(anchor=tk.W)

        self._image: Optional[np.ndarray] = None
        self._photo: Optional[Any] = None
        self._uniform: bool = False
        self._scale: float = 1.0
        self._offset: Tuple[float, float] = (0.0, 0.0)
        self._drag_origin: Optional[Tuple[int, int]] = None
        self._placeholder = "（未就绪）"

    # -- public API ---------------------------------------------------
    def set_image(
        self,
        image: Optional[np.ndarray],
        summary: str = "",
        placeholder: str = "（未就绪）",
        uniform: bool = False,
    ) -> None:
        """Show ``image`` (uint8) with an optional one-line summary below it."""
        self._image = None if image is None else np.asarray(image)
        self._uniform = bool(uniform)
        self._placeholder = placeholder
        self._stats_var.set(summary)
        self._offset = (0.0, 0.0)
        if self._image is None:
            self._photo = None
        self._redraw()

    def stats_uniform(self) -> bool:
        """Whether the last image shown had no grey-level variation at all."""
        return self._uniform

    def set_caption(self, text: str) -> None:
        self._caption_var.set(text)

    # -- rendering ----------------------------------------------------
    def _on_zoom_changed(self) -> None:
        # Centre the new viewport instead of snapping to the top-left corner.
        if self._image is not None:
            h, w = self._image.shape[:2]
            self._offset = (max(0.0, (w - self._viewport()[0]) / 2.0),
                            max(0.0, (h - self._viewport()[1]) / 2.0))
        self._redraw()

    def _fit_scale(self) -> float:
        cw = max(1, self._canvas.winfo_width())
        ch = max(1, self._canvas.winfo_height())
        h, w = self._image.shape[:2]
        return min(cw / float(w), ch / float(h))

    def _scale_for_mode(self) -> float:
        mode = self._zoom_var.get()
        if mode == "适应窗口" or self._image is None:
            return max(self._fit_scale(), 1e-6)
        return 0.5 if mode == "50%" else 1.0

    def _viewport(self) -> Tuple[float, float]:
        """Visible size in image pixels for the current scale."""
        cw = max(1, self._canvas.winfo_width())
        ch = max(1, self._canvas.winfo_height())
        scale = max(self._scale, 1e-6)
        return cw / scale, ch / scale

    def _redraw(self) -> None:
        self._canvas.delete("all")
        cw = max(1, self._canvas.winfo_width())
        ch = max(1, self._canvas.winfo_height())

        if self._image is None:
            self._canvas.create_text(cw // 2, ch // 2, text=self._placeholder,
                                     fill="#aaaaaa")
            return

        h, w = self._image.shape[:2]
        self._scale = self._scale_for_mode()
        vis_w, vis_h = self._viewport()

        ox, oy = self._offset
        ox = min(max(0.0, ox), max(0.0, w - vis_w))
        oy = min(max(0.0, oy), max(0.0, h - vis_h))
        self._offset = (ox, oy)

        x0, y0 = int(ox), int(oy)
        x1 = int(min(float(w), ox + vis_w + 1.0))
        y1 = int(min(float(h), oy + vis_h + 1.0))
        region = self._image[y0:y1, x0:x1]
        if region.size == 0:
            return

        out_w = max(1, int(round((x1 - x0) * self._scale)))
        out_h = max(1, int(round((y1 - y0) * self._scale)))
        disp = cv2.resize(region, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        self._photo = cv2_to_tk(disp)
        # Centre the image when it is smaller than the canvas; when zoomed past
        # the viewport both terms clamp to 0 and the pan offset takes over.
        draw_x = max(0.0, (cw - out_w) / 2.0)
        draw_y = max(0.0, (ch - out_h) / 2.0)
        self._canvas.create_image(draw_x, draw_y, anchor=tk.NW,
                                  image=self._photo)

    # -- panning ------------------------------------------------------
    def _on_press(self, event: "tk.Event") -> None:
        self._drag_origin = (int(event.x), int(event.y))

    def _on_drag(self, event: "tk.Event") -> None:
        if self._drag_origin is None or self._image is None:
            return
        px, py = self._drag_origin
        dx = (int(event.x) - px) / max(self._scale, 1e-6)
        dy = (int(event.y) - py) / max(self._scale, 1e-6)
        self._drag_origin = (int(event.x), int(event.y))
        self._offset = (self._offset[0] - dx, self._offset[1] - dy)
        self._redraw()


class PreprocessorPreviewDialog(tk.Toplevel):
    """Non-modal before/after preview of the preprocessing step."""

    def __init__(
        self,
        parent: tk.Widget,
        template_patch: Optional[np.ndarray] = None,
        inspection_image: Optional[np.ndarray] = None,
        initial_type: str = "raw",
        on_type_changed: Optional[Any] = None,
    ) -> None:
        super().__init__(parent)
        self.title("预处理预览")
        try:
            self.transient(parent.winfo_toplevel())
        except Exception:  # pragma: no cover - parent may be a plain widget
            pass

        self._template_patch = template_patch
        self._inspection_image = inspection_image
        self._on_type_changed = on_type_changed
        self._configs: Dict[str, Dict[str, Any]] = {
            cfg["type"]: dict(cfg) for cfg in default_configs()
        }
        self._vars: Dict[str, Any] = {}
        self._views: Dict[str, _ImageView] = {}
        self._after_id: Optional[str] = None
        self._compare_window: Optional["_CompareWindow"] = None
        self._status_var = tk.StringVar(value="")

        if initial_type not in PREPROCESSOR_TYPES:
            initial_type = "raw"
        self._type_var = tk.StringVar(value=initial_type)

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.geometry("1040x840")
        self.minsize(880, 640)
        self.after(60, self.refresh_now)

    # -- construction -------------------------------------------------
    def _build_ui(self) -> None:
        outer = ttk.Frame(self, padding=8)
        outer.pack(fill=tk.BOTH, expand=True)

        controls = ttk.Frame(outer)
        controls.pack(fill=tk.X)

        ttk.Label(controls, text="预处理").pack(side=tk.LEFT)
        type_combo = ttk.Combobox(
            controls, textvariable=self._type_var, values=list(PREPROCESSOR_TYPES),
            state="readonly", width=12,
        )
        type_combo.pack(side=tk.LEFT, padx=(4, 12))
        type_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_type_selected())

        self._param_frame = ttk.Frame(controls)
        self._param_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Button(controls, text="全部对比…", command=self.open_compare).pack(
            side=tk.RIGHT, padx=(6, 0)
        )

        grid = ttk.Frame(outer, padding=(0, 8))
        grid.pack(fill=tk.BOTH, expand=True)
        for col in range(2):
            grid.columnconfigure(col, weight=1)
        for row in range(2):
            grid.rowconfigure(row, weight=1)

        captions = {
            "template_raw": "模板 · 原图（未增强）",
            "template_enh": "模板 · 增强后（参与匹配）",
            "inspection_raw": "目标图 · 原图（未增强）",
            "inspection_enh": "目标图 · 增强后（参与匹配）",
        }
        positions = {
            "template_raw": (0, 0), "template_enh": (0, 1),
            "inspection_raw": (1, 0), "inspection_enh": (1, 1),
        }
        for key, (row, col) in positions.items():
            view = _ImageView(grid, caption=captions[key], width=440, height=240)
            view.grid(row=row, column=col, sticky="nsew", padx=4, pady=4)
            self._views[key] = view

        footer = ttk.Frame(outer)
        footer.pack(fill=tk.X)
        ttk.Label(footer, textvariable=self._status_var,
                  foreground="#8a4b00", wraplength=760).pack(side=tk.LEFT)
        ttk.Button(footer, text="关闭", command=self.close).pack(side=tk.RIGHT)

        self._build_param_controls()

    def _build_param_controls(self) -> None:
        for child in self._param_frame.winfo_children():
            child.destroy()
        self._vars.clear()

        kind = self._type_var.get()
        config = self._configs.get(kind, {"type": kind})

        if kind == "raw":
            ttk.Label(self._param_frame, text="无参数").pack(side=tk.LEFT, padx=4)
            return
        if kind == "canny":
            self._add_scale("t1", "低阈值", 0.0, 255.0, float(config.get("t1", 50.0)))
            self._add_scale("t2", "高阈值", 0.0, 255.0, float(config.get("t2", 150.0)))
        elif kind == "sobel":
            self._add_combo("ksize", "核大小", ["1", "3", "5", "7"],
                            str(config.get("ksize", 3)))
        elif kind == "clahe":
            self._add_scale("clip_limit", "clip_limit", 0.5, 40.0,
                            float(config.get("clip_limit", 2.0)), digits=1)
            tile = list(config.get("tile_grid_size", [8, 8]))
            self._add_spin("tile_x", "tile 宽", 2, 16, int(tile[0]))
            self._add_spin("tile_y", "tile 高", 2, 16, int(tile[1]))
        elif kind == "threshold":
            self._add_scale("threshold", "阈值", 0.0, 255.0,
                            float(config.get("threshold", 128.0)))
            self._add_combo("mode", "模式", ["binary", "binary_inv"],
                            str(config.get("mode", "binary")))

    def _add_scale(self, key: str, label: str, lo: float, hi: float,
                   default: float, digits: int = 1) -> None:
        ttk.Label(self._param_frame, text=label).pack(side=tk.LEFT, padx=(8, 2))
        var = tk.DoubleVar(value=default)
        self._vars[key] = var
        value_label = ttk.Label(self._param_frame, width=6)
        value_label.pack(side=tk.LEFT)

        def _on_move(value: str) -> None:
            value_label.config(text=f"{float(value):.{digits}f}")
            self._schedule_refresh()

        scale = ttk.Scale(self._param_frame, from_=lo, to=hi, variable=var,
                          orient=tk.HORIZONTAL, length=130, command=_on_move)
        scale.pack(side=tk.LEFT)
        value_label.config(text=f"{default:.{digits}f}")

    def _add_spin(self, key: str, label: str, lo: int, hi: int, default: int) -> None:
        ttk.Label(self._param_frame, text=label).pack(side=tk.LEFT, padx=(8, 2))
        var = tk.IntVar(value=default)
        self._vars[key] = var
        spin = ttk.Spinbox(self._param_frame, from_=lo, to=hi, width=4,
                           textvariable=var, command=self._schedule_refresh)
        spin.pack(side=tk.LEFT)
        spin.bind("<FocusOut>", lambda _e: self._schedule_refresh())

    def _add_combo(self, key: str, label: str, values: list, default: str) -> None:
        ttk.Label(self._param_frame, text=label).pack(side=tk.LEFT, padx=(8, 2))
        var = tk.StringVar(value=default)
        self._vars[key] = var
        combo = ttk.Combobox(self._param_frame, textvariable=var, values=values,
                             state="readonly", width=10)
        combo.pack(side=tk.LEFT)
        combo.bind("<<ComboboxSelected>>", lambda _e: self._schedule_refresh())

    # -- behaviour ----------------------------------------------------
    def set_type(self, name: str) -> None:
        """Switch the displayed preprocessor (used to mirror the main window)."""
        if name not in PREPROCESSOR_TYPES or name == self._type_var.get():
            return
        self._type_var.set(name)
        self._build_param_controls()
        self._schedule_refresh()

    def set_sources(
        self,
        template_patch: Optional[np.ndarray],
        inspection_image: Optional[np.ndarray],
    ) -> None:
        """Replace the previewed images (e.g. after the user teaches a template)."""
        self._template_patch = template_patch
        self._inspection_image = inspection_image
        self._schedule_refresh()

    def _on_type_selected(self) -> None:
        self._build_param_controls()
        if self._on_type_changed is not None:
            try:
                self._on_type_changed(self._type_var.get())
            except Exception:
                pass
        self._schedule_refresh()

    def _current_config(self) -> Dict[str, Any]:
        kind = self._type_var.get()
        if kind == "raw":
            return {"type": "raw"}
        if kind == "canny":
            return {"type": "canny", "t1": float(self._vars["t1"].get()),
                    "t2": float(self._vars["t2"].get())}
        if kind == "sobel":
            return {"type": "sobel", "ksize": int(self._vars["ksize"].get())}
        if kind == "clahe":
            return {
                "type": "clahe",
                "clip_limit": float(self._vars["clip_limit"].get()),
                "tile_grid_size": [max(2, int(self._vars["tile_x"].get())),
                                   max(2, int(self._vars["tile_y"].get()))],
            }
        if kind == "threshold":
            return {"type": "threshold",
                    "threshold": float(self._vars["threshold"].get()),
                    "mode": str(self._vars["mode"].get())}
        return {"type": kind}

    def _schedule_refresh(self) -> None:
        if self._after_id is not None:
            try:
                self.after_cancel(self._after_id)
            except tk.TclError:
                pass
        self._after_id = self.after(120, self.refresh_now)

    def refresh_now(self) -> None:
        """Recompute and redraw both sides for the current settings."""
        self._after_id = None
        try:
            preprocessor = preprocessor_from_config(self._current_config())
        except Exception as exc:  # invalid widget state while typing
            self._status_var.set(f"参数无效：{exc}")
            return

        template_raw, inspection_raw = preview_pair(
            self._template_patch, self._inspection_image,
            preprocessor_from_config({"type": "raw"}),
        )
        template_enh, inspection_enh = preview_pair(
            self._template_patch, self._inspection_image, preprocessor,
        )

        self._views["template_raw"].set_image(
            None if template_raw is None else template_raw.image,
            self._raw_summary(template_raw),
            placeholder="（未创建模板）",
        )
        self._views["inspection_raw"].set_image(
            None if inspection_raw is None else inspection_raw.image,
            self._raw_summary(inspection_raw),
            placeholder="（未加载检测图）",
        )
        self._views["template_enh"].set_image(
            None if template_enh is None else template_enh.image,
            self._enhanced_summary(template_enh),
            placeholder="（未创建模板）",
            uniform=bool(template_enh is not None
                         and template_enh.stats.get("uniform")),
        )
        self._views["inspection_enh"].set_image(
            None if inspection_enh is None else inspection_enh.image,
            self._enhanced_summary(inspection_enh),
            placeholder="（未加载检测图）",
            uniform=bool(inspection_enh is not None
                         and inspection_enh.stats.get("uniform")),
        )

        notes = []
        if template_enh is not None and template_enh.stats.get("uniform"):
            notes.append("模板增强后整块饱和（可能阈值超出模板灰阶范围）")
        if not notes:
            notes.append("预览为只读：不修改工程配置，也不会重建模板")
        self._status_var.set("；".join(notes))

    @staticmethod
    def _raw_summary(result: Any) -> str:
        if result is None:
            return ""
        h, w = result.stats["shape"]
        return f"{h}×{w}  mean {result.stats['mean']}  std {result.stats['std']}"

    @staticmethod
    def _enhanced_summary(result: Any) -> str:
        if result is None:
            return ""
        stats = result.stats
        h, w = stats["shape"]
        text = (f"{h}×{w}  mean {stats['mean']}  std {stats['std']}  "
                f"{stats['elapsed_ms']} ms")
        if stats.get("normalized"):
            text += f"  已拉伸（原始 {stats['source_range'][0]}~{stats['source_range'][1]}）"
        if stats.get("uniform"):
            text += "  整块饱和"
        return text

    # -- compare window ------------------------------------------------
    def open_compare(self) -> None:
        """Show every built-in preprocessor side by side."""
        if self._compare_window is not None and self._compare_window.winfo_exists():
            self._compare_window.lift()
            return
        self._compare_window = _CompareWindow(
            self, self._template_patch, self._inspection_image
        )

    def close(self) -> None:
        try:
            if not self.winfo_exists():
                return
        except tk.TclError:
            return
        if self._after_id is not None:
            try:
                self.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None
        if self._compare_window is not None and self._compare_window.winfo_exists():
            self._compare_window.destroy()
        self._compare_window = None
        self.destroy()


class _CompareWindow(tk.Toplevel):
    """Two rows (template / target) by one column per built-in preprocessor."""

    def __init__(
        self,
        parent: tk.Widget,
        template_patch: Optional[np.ndarray],
        inspection_image: Optional[np.ndarray],
    ) -> None:
        super().__init__(parent)
        self.title("预处理对比 · 全部内置方式")
        self.transient(parent)

        outer = ttk.Frame(self, padding=8)
        outer.pack(fill=tk.BOTH, expand=True)

        columns: List[PreviewColumn] = preview_matrix(template_patch, inspection_image)
        for index in range(len(columns)):
            outer.columnconfigure(index, weight=1)

        for index, column in enumerate(columns):
            header = column.name
            if column.template is not None and column.inspection is not None:
                header += (f"\n模板σ {column.template.stats['std']}"
                           f"\n目标σ {column.inspection.stats['std']}")
            ttk.Label(outer, text=header, justify=tk.CENTER, font=_CAPTION_FONT,
                      wraplength=180).grid(
                row=0, column=index, sticky="ew", padx=3, pady=(0, 2)
            )

            tmpl_view = _ImageView(
                outer, caption="模板", width=180, height=150
            )
            tmpl_view.grid(row=1, column=index, sticky="nsew", padx=3, pady=3)
            tmpl_view.set_image(
                None if column.template is None else column.template.image,
                placeholder="（无）",
            )

            insp_view = _ImageView(
                outer, caption="目标图", width=180, height=150
            )
            insp_view.grid(row=2, column=index, sticky="nsew", padx=3, pady=3)
            insp_view.set_image(
                None if column.inspection is None else column.inspection.image,
                placeholder="（无）",
            )

        ttk.Button(outer, text="关闭", command=self.destroy).grid(
            row=3, column=0, columnspan=max(1, len(columns)), pady=(6, 0)
        )
        self.geometry("1080x540")