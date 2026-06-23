"""
Left-side tool panel for measurement configuration.

Contains:
  - File operations (load/save)
  - Template settings (preprocessor, matching params)
  - Measurement tool list with reorder/delete
  - Composed measurement creation (distance, angle)
  - Execute button
"""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .dialogs import ComposedMeasureDialog


class ToolPanel(ttk.Frame):
    """
    Left-side measurement configuration panel.

    Callbacks:
        on_load_reference: () -> None
        on_load_inspection: () -> None
        on_save_project: () -> None
        on_load_project: () -> None
        on_create_template: () -> None
        on_execute: () -> None
        on_preprocessor_changed: (name: str) -> None
        on_score_threshold_changed: (value: float) -> None
        on_angle_range_changed: (value: str) -> None  # '±30', '±45', '±90'
        on_max_matches_changed: (value: int) -> None
        on_add_composed: (obj_type: str, label: str, params: dict) -> None
        on_tool_edit: (label: str) -> None
        on_tool_delete: (label: str) -> None
        on_export_csv: () -> None
    """

    def __init__(self, parent: tk.Widget, **kwargs):
        super().__init__(parent, **kwargs)

        # Callbacks
        self.on_load_reference: Optional[Callable] = None
        self.on_load_inspection: Optional[Callable] = None
        self.on_save_project: Optional[Callable] = None
        self.on_load_project: Optional[Callable] = None
        self.on_create_template: Optional[Callable] = None
        self.on_execute: Optional[Callable] = None
        self.on_preprocessor_changed: Optional[Callable] = None
        self.on_score_threshold_changed: Optional[Callable] = None
        self.on_angle_range_changed: Optional[Callable] = None
        self.on_max_matches_changed: Optional[Callable] = None
        self.on_overlap_changed: Optional[Callable] = None
        self.on_add_composed: Optional[Callable] = None
        self.on_tool_edit: Optional[Callable] = None
        self.on_tool_delete: Optional[Callable] = None
        self.on_export_csv: Optional[Callable] = None

        # State
        self._template_created: bool = False

        self._build_ui()

    def _build_ui(self):
        # Scrollable container
        canvas = tk.Canvas(self, width=280, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=canvas.yview)
        self._scroll_frame = ttk.Frame(canvas)

        self._scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self._scroll_frame, anchor=tk.NW)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind("<MouseWheel>", _on_mousewheel)
        canvas.bind("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
        canvas.bind("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))

        f = self._scroll_frame
        pad = {"padx": 4, "pady": 2}

        # --- File operations ---
        file_frame = ttk.LabelFrame(f, text="文件操作", padding=5)
        file_frame.pack(fill=tk.X, padx=4, pady=2)

        ttk.Button(file_frame, text="加载参考图",
                   command=lambda: self._fire(self.on_load_reference),
                   width=14).pack(fill=tk.X, **pad)
        ttk.Button(file_frame, text="加载检测图",
                   command=lambda: self._fire(self.on_load_inspection),
                   width=14).pack(fill=tk.X, **pad)
        ttk.Separator(file_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, **pad)
        ttk.Button(file_frame, text="保存项目",
                   command=lambda: self._fire(self.on_save_project),
                   width=14).pack(fill=tk.X, **pad)
        ttk.Button(file_frame, text="加载项目",
                   command=lambda: self._fire(self.on_load_project),
                   width=14).pack(fill=tk.X, **pad)
        ttk.Button(file_frame, text="导出 CSV",
                   command=lambda: self._fire(self.on_export_csv),
                   width=14).pack(fill=tk.X, **pad)

        # --- Template settings ---
        tmpl_frame = ttk.LabelFrame(f, text="模板设置", padding=5)
        tmpl_frame.pack(fill=tk.X, padx=4, pady=2)

        # Preprocessor
        ttk.Label(tmpl_frame, text="预处理").pack(anchor=tk.W, **pad)
        self._preproc_var = tk.StringVar(value="raw")
        preproc_combo = ttk.Combobox(
            tmpl_frame, textvariable=self._preproc_var,
            values=["raw", "canny", "sobel", "clahe", "threshold"],
            state="readonly", width=18,
        )
        preproc_combo.pack(fill=tk.X, **pad)
        preproc_combo.bind("<<ComboboxSelected>>",
                          lambda e: self._fire(self.on_preprocessor_changed,
                                              self._preproc_var.get()))

        # Match score threshold
        ttk.Label(tmpl_frame, text="匹配分数阈值").pack(anchor=tk.W, **pad)
        self._score_var = tk.DoubleVar(value=0.5)
        score_scale = ttk.Scale(
            tmpl_frame, from_=0.1, to=1.0, variable=self._score_var,
            orient=tk.HORIZONTAL,
            command=lambda v: self._fire(self.on_score_threshold_changed,
                                        float(v)),
        )
        score_scale.pack(fill=tk.X, **pad)
        score_label = ttk.Label(tmpl_frame, textvariable=self._score_var)
        score_label.pack(anchor=tk.E, **pad)

        # Angle range
        ttk.Label(tmpl_frame, text="角度搜索范围").pack(anchor=tk.W, **pad)
        self._angle_var = tk.StringVar(value="±30°")
        angle_combo = ttk.Combobox(
            tmpl_frame, textvariable=self._angle_var,
            values=["±15°", "±30°", "±45°", "±60°", "±90°", "±180°"],
            state="readonly", width=18,
        )
        angle_combo.pack(fill=tk.X, **pad)
        angle_combo.bind("<<ComboboxSelected>>",
                        lambda e: self._fire(self.on_angle_range_changed,
                                            self._angle_var.get()))

        # Max matches
        ttk.Label(tmpl_frame, text="最大匹配数 (0=无限)").pack(anchor=tk.W, **pad)
        self._max_matches_var = tk.IntVar(value=0)
        max_spin = ttk.Spinbox(
            tmpl_frame, textvariable=self._max_matches_var,
            from_=0, to=100, increment=1, width=5,
        )
        max_spin.pack(anchor=tk.W, **pad)
        max_spin.bind("<FocusOut>",
                     lambda e: self._fire(self.on_max_matches_changed,
                                         self._max_matches_var.get()))

        # Overlap (NMS)
        ttk.Label(tmpl_frame, text="最大重叠比例 (NMS)").pack(anchor=tk.W, **pad)
        self._overlap_var = tk.DoubleVar(value=0.3)
        overlap_scale = ttk.Scale(
            tmpl_frame, from_=0.0, to=1.0, variable=self._overlap_var,
            orient=tk.HORIZONTAL,
            command=lambda v: self._fire(self.on_overlap_changed,
                                        float(v)),
        )
        overlap_scale.pack(fill=tk.X, **pad)
        self._overlap_label = ttk.Label(tmpl_frame, text="30%")
        self._overlap_label.pack(anchor=tk.E, **pad)
        # Update label as slider moves
        def _update_overlap_label(*args):
            pct = int(self._overlap_var.get() * 100)
            self._overlap_label.config(text=f"{pct}%")
        self._overlap_var.trace_add("write", _update_overlap_label)

        # ROI display (read-only)
        self._roi_label = ttk.Label(tmpl_frame, text="ROI: (未设置)", foreground="gray")
        self._roi_label.pack(anchor=tk.W, **pad)

        # Create template button
        self._create_btn = ttk.Button(
            tmpl_frame, text="▶ 创建/更新模板",
            command=lambda: self._fire(self.on_create_template),
        )
        self._create_btn.pack(fill=tk.X, pady=5)

        # --- Measurement tool list ---
        tool_frame = ttk.LabelFrame(f, text="测量工具", padding=5)
        tool_frame.pack(fill=tk.X, padx=4, pady=2)

        # Treeview for tool list
        columns = ("label", "type")
        self._tool_tree = ttk.Treeview(tool_frame, columns=columns,
                                       show="headings", height=8)
        self._tool_tree.heading("label", text="标签")
        self._tool_tree.heading("type", text="类型")
        self._tool_tree.column("label", width=100)
        self._tool_tree.column("type", width=80)
        self._tool_tree.pack(fill=tk.X, **pad)

        # Double-click to edit
        self._tool_tree.bind("<Double-1>", self._on_tool_double_click)

        # Edit/Delete buttons
        btn_row = ttk.Frame(tool_frame)
        btn_row.pack(fill=tk.X, **pad)
        ttk.Button(btn_row, text="编辑", command=self._on_edit_tool).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="删除", command=self._on_delete_tool).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="上移", command=self._on_move_up).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="下移", command=self._on_move_down).pack(side=tk.LEFT, padx=2)

        # --- Composed measurement ---
        comp_frame = ttk.LabelFrame(f, text="组合测量", padding=5)
        comp_frame.pack(fill=tk.X, padx=4, pady=2)

        ttk.Button(comp_frame, text="＋两点距离",
                   command=lambda: self._add_composed("TwoPointsDistance",
                                                      "两点距离",
                                                      ["point_a_label", "point_b_label"]),
                   width=14).pack(fill=tk.X, **pad)
        ttk.Button(comp_frame, text="＋两线夹角",
                   command=lambda: self._add_composed("TwoLinesAngle",
                                                      "两线夹角",
                                                      ["line_a_label", "line_b_label"]),
                   width=14).pack(fill=tk.X, **pad)
        ttk.Button(comp_frame, text="＋点线距离",
                   command=lambda: self._add_composed("PointLineDistance",
                                                      "点线距离",
                                                      ["point_label", "line_label"]),
                   width=14).pack(fill=tk.X, **pad)
        ttk.Button(comp_frame, text="＋点圆距离",
                   command=lambda: self._add_composed("PointCircleDistance",
                                                      "点圆距离",
                                                      ["point_label", "circle_label"]),
                   width=14).pack(fill=tk.X, **pad)

        # --- Execute ---
        exec_frame = ttk.Frame(f, padding=5)
        exec_frame.pack(fill=tk.X, padx=4, pady=5)

        self._exec_btn = ttk.Button(
            exec_frame, text="▶ 执行测量",
            command=lambda: self._fire(self.on_execute),
        )
        self._exec_btn.pack(fill=tk.X, ipady=5)

        self._progress = ttk.Progressbar(exec_frame, mode="indeterminate")
        self._progress.pack(fill=tk.X, pady=2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_roi_info(self, center, size, angle_deg):
        """Update the ROI info label."""
        self._roi_label.config(
            text=f"ROI: ({center[0]:.0f},{center[1]:.0f}) {size[0]:.0f}×{size[1]:.0f} {angle_deg:.1f}°"
        )

    def clear_roi_info(self):
        """Clear ROI info."""
        self._roi_label.config(text="ROI: (未设置)", foreground="gray")

    def set_template_created(self, created: bool):
        """Enable/disable measurement tool controls based on template state."""
        self._template_created = created
        if created:
            self._roi_label.config(foreground="lime")
            self._exec_btn.config(state=tk.NORMAL)
        else:
            self._exec_btn.config(state=tk.DISABLED)

    def add_tool_to_list(self, label: str, obj_type: str):
        """Add a measurement tool to the list display."""
        self._tool_tree.insert("", tk.END, iid=label, values=(label, obj_type))

    def remove_tool_from_list(self, label: str):
        """Remove a tool from the list display."""
        if self._tool_tree.exists(label):
            self._tool_tree.delete(label)

    def clear_tool_list(self):
        """Clear all tools from the list."""
        for item in self._tool_tree.get_children():
            self._tool_tree.delete(item)

    def get_tool_list_labels(self) -> List[str]:
        """Get all tool labels currently visible."""
        return list(self._tool_tree.get_children())

    def get_tool_list_order(self) -> List[str]:
        """Get ordered list of tool labels (for project manifest)."""
        return self.get_tool_list_labels()

    def restore_tool_list(self, tools: List[dict], order: List[str]):
        """Restore the tool list treeview from saved state.

        Args:
            tools: List of {object_type, label, params} from workflow.measurement_defs.
            order: Ordered list of labels from project manifest.
        """
        self.clear_tool_list()
        # Build lookup by label
        tool_map = {}
        for t in tools:
            label = t.get("label", "")
            obj_type = t.get("object_type", "?")
            if label:
                tool_map[label] = obj_type

        # Insert in saved order
        for label in order:
            obj_type = tool_map.pop(label, "?")
            if not self._tool_tree.exists(label):
                self._tool_tree.insert("", tk.END, iid=label,
                                       values=(label, obj_type))

        # Insert any remaining tools not in order
        for label, obj_type in tool_map.items():
            if not self._tool_tree.exists(label):
                self._tool_tree.insert("", tk.END, iid=label,
                                       values=(label, obj_type))

    def set_matching_params(self, preprocessor_type: str,
                            score_threshold: float,
                            angle_range_half: float,
                            max_matches: int,
                            overlap: float = 0.3):
        """Set matching parameter widgets programmatically (for project restore).

        Args:
            preprocessor_type: One of raw/canny/sobel/clahe/threshold.
            score_threshold: Match score threshold (0.1–1.0).
            angle_range_half: Half-range in degrees (e.g. 30 for ±30°).
            max_matches: Max number of matches (0 = unlimited).
            overlap: Max allowed IoU overlap in [0, 1] (default 0.3).
        """
        self._preproc_var.set(preprocessor_type)
        self._score_var.set(score_threshold)
        self._angle_var.set(f"±{int(angle_range_half)}°")
        self._max_matches_var.set(max_matches)
        self._overlap_var.set(overlap)

    def set_progress(self, running: bool):
        """Start/stop the progress bar."""
        if running:
            self._progress.start(10)
        else:
            self._progress.stop()

    def enable_ui(self, enabled: bool):
        """Enable or disable all UI controls."""
        state = tk.NORMAL if enabled else tk.DISABLED
        for child in self.winfo_children():
            try:
                child.config(state=state)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fire(self, callback, *args):
        """Safely call a callback if set."""
        if callback is not None:
            if args:
                callback(*args)
            else:
                callback()

    def _on_tool_double_click(self, event):
        """Double-click a tool to edit."""
        selection = self._tool_tree.selection()
        if selection:
            label = selection[0]
            self._fire(self.on_tool_edit, label)

    def _on_edit_tool(self):
        selection = self._tool_tree.selection()
        if selection:
            self._fire(self.on_tool_edit, selection[0])

    def _on_delete_tool(self):
        selection = self._tool_tree.selection()
        if selection:
            label = selection[0]
            self._fire(self.on_tool_delete, label)

    def _on_move_up(self):
        selection = self._tool_tree.selection()
        if selection:
            label = selection[0]
            prev = self._tool_tree.prev(label)
            if prev:
                self._tool_tree.move(label, "", self._tool_tree.index(prev))

    def _on_move_down(self):
        selection = self._tool_tree.selection()
        if selection:
            label = selection[0]
            nxt = self._tool_tree.next(label)
            if nxt:
                self._tool_tree.move(label, "", self._tool_tree.index(nxt) + 1)

    def _add_composed(self, obj_type: str, title: str, param_keys: list):
        """Show dialog to add a composed measurement."""
        if not self._template_created:
            messagebox.showwarning("提示", "请先创建模板")
            return

        # Get available labels from the tree
        available_labels = list(self._tool_tree.get_children())

        result = ComposedMeasureDialog.ask(
            self, title=title, input_labels=available_labels,
            param_keys=param_keys,
        )

        if result:
            label = f"{obj_type}_{len(self._tool_tree.get_children()) + 1}"
            self._fire(self.on_add_composed, obj_type, label, result)
