"""
Main application — integrates all GUI components with the MultiTargetWorkflow backend.

Layout:
  ┌──────────────────────────────────────────────────────────┐
  │ [Menu] File  View  Help                                  │
  │ [Toolbar] LoadRef LoadInsp SaveProject ... │ Execute     │
  ├──────────┬───────────────────────────┬───────────────────┤
  │          │ [Reference] [Inspection]  │                   │
  │ ToolPanel│ ┌───────────────────────┐ │  TemplateView     │
  │          │ │   ImageCanvas         │ │  (straightened    │
  │          │ │   + ROI               │ │   template        │
  │          │ │   + Result overlay    │ │   + tool overlay) │
  │          │ └───────────────────────┘ │                   │
  ├──────────┴───────────────────────────┴───────────────────┤
  │ ResultPanel: Targets │ Measurements │ Summary Text       │
  └──────────────────────────────────────────────────────────┘

Usage:
    python -m measure_gui.app
"""

from __future__ import annotations

import os
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .image_canvas import CanvasMode, ImageCanvas
from .multi_target import MultiTargetWorkflow, TargetResult
from .result_panel import ResultPanel
from .template_view import TemplateTool, TemplateView
from .tool_panel import ToolPanel
from .utils import crop_and_straighten


class MeasureApp:
    """Main application class."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("工业视觉测量系统 — Measure GUI")
        self.root.geometry("1600x900")

        # Backend
        self._workflow: Optional[MultiTargetWorkflow] = None
        self._reference_image: Optional[np.ndarray] = None
        self._inspection_image: Optional[np.ndarray] = None
        self._current_project_path: Optional[str] = None

        # State
        self._teaching: bool = True  # True = teaching mode, False = inspection mode

        # Build UI
        self._build_menu()
        self._build_toolbar()
        self._build_main_area()
        self._build_statusbar()

        # Configure styles
        self._setup_styles()

        # Bind keyboard shortcuts
        self.root.bind("<Control-o>", lambda e: self._load_reference())
        self.root.bind("<Control-i>", lambda e: self._load_inspection())
        self.root.bind("<Control-s>", lambda e: self._save_project())
        self.root.bind("<Control-e>", lambda e: self._execute())
        self.root.bind("<Escape>", lambda e: self._cancel_drawing())

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Execute.TButton", font=("TkDefaultFont", 11, "bold"))

    def _build_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="文件", menu=file_menu)
        file_menu.add_command(label="加载参考图 (Ctrl+O)", command=self._load_reference)
        file_menu.add_command(label="加载检测图 (Ctrl+I)", command=self._load_inspection)
        file_menu.add_separator()
        file_menu.add_command(label="保存项目 (Ctrl+S)", command=self._save_project)
        file_menu.add_command(label="加载项目", command=self._load_project)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.root.quit)

        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="视图", menu=view_menu)
        view_menu.add_command(label="适应窗口", command=self._zoom_to_fit)
        view_menu.add_command(label="100%", command=self._zoom_100)
        view_menu.add_separator()
        view_menu.add_command(label="重置 ROI", command=self._reset_roi)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="帮助", menu=help_menu)
        help_menu.add_command(label="使用说明", command=self._show_help)
        help_menu.add_command(label="关于", command=self._show_about)

    def _build_toolbar(self):
        toolbar = ttk.Frame(self.root)
        toolbar.pack(fill=tk.X, padx=2, pady=2)

        ttk.Button(toolbar, text="📁 参考图", command=self._load_reference).pack(side=tk.LEFT, padx=1)
        ttk.Button(toolbar, text="📁 检测图", command=self._load_inspection).pack(side=tk.LEFT, padx=1)
        ttk.Button(toolbar, text="💾 保存", command=self._save_project).pack(side=tk.LEFT, padx=1)
        ttk.Button(toolbar, text="📂 加载", command=self._load_project).pack(side=tk.LEFT, padx=1)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)

        self._exec_btn = ttk.Button(
            toolbar, text="▶ 执行测量 (Ctrl+E)",
            command=self._execute, style="Execute.TButton",
        )
        self._exec_btn.pack(side=tk.LEFT, padx=1)
        self._exec_btn.state(["disabled"])

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)

        self._mode_label = ttk.Label(toolbar, text="教学模式", foreground="blue")
        self._mode_label.pack(side=tk.LEFT, padx=5)

    def _build_main_area(self):
        # Horizontal PanedWindow
        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # Left: ToolPanel
        self.tool_panel = ToolPanel(main_pane)
        main_pane.add(self.tool_panel, weight=0)

        # Connect tool panel callbacks
        self.tool_panel.on_load_reference = self._load_reference
        self.tool_panel.on_load_inspection = self._load_inspection
        self.tool_panel.on_save_project = self._save_project
        self.tool_panel.on_load_project = self._load_project
        self.tool_panel.on_create_template = self._create_template
        self.tool_panel.on_execute = self._execute
        self.tool_panel.on_preprocessor_changed = self._on_preprocessor_changed
        self.tool_panel.on_score_threshold_changed = self._on_score_threshold_changed
        self.tool_panel.on_angle_range_changed = self._on_angle_range_changed
        self.tool_panel.on_max_matches_changed = self._on_max_matches_changed
        self.tool_panel.on_add_composed = self._on_add_composed
        self.tool_panel.on_tool_edit = self._on_tool_edit
        self.tool_panel.on_tool_delete = self._on_tool_delete
        self.tool_panel.on_export_csv = self._on_export_csv

        # Right side: vertical paned window
        right_pane = ttk.PanedWindow(main_pane, orient=tk.VERTICAL)
        main_pane.add(right_pane, weight=1)

        # Center: Notebook for reference/inspection + template preview
        center_frame = ttk.Frame(right_pane)
        right_pane.add(center_frame, weight=3)

        # Center sub-pane: image notebook + template preview
        center_pane = ttk.PanedWindow(center_frame, orient=tk.HORIZONTAL)
        center_pane.pack(fill=tk.BOTH, expand=True)

        # Image notebook (reference / inspection tabs)
        self._notebook = ttk.Notebook(center_pane)
        center_pane.add(self._notebook, weight=3)

        # Reference image tab
        self._ref_frame = ttk.Frame(self._notebook)
        self._notebook.add(self._ref_frame, text="参考图")
        self._ref_canvas = ImageCanvas(self._ref_frame, width=800, height=600)
        self._ref_canvas.pack(fill=tk.BOTH, expand=True)
        self._ref_canvas.on_roi_changed = self._on_roi_changed
        self._ref_canvas.on_roi_confirmed = self._on_roi_confirmed

        # Inspection image tab
        self._insp_frame = ttk.Frame(self._notebook)
        self._notebook.add(self._insp_frame, text="检测图")
        self._insp_canvas = ImageCanvas(self._insp_frame, width=800, height=600)
        self._insp_canvas.pack(fill=tk.BOTH, expand=True)
        self._insp_canvas.set_mode(CanvasMode.VIEW_RESULT)

        self._notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        # Template preview (right side)
        self.template_view = TemplateView(center_pane, width=350, height=350)
        center_pane.add(self.template_view, weight=1)

        # Connect template view callbacks
        self.template_view.on_tool_added = self._on_template_tool_added
        self.template_view.on_tool_removed = self._on_template_tool_removed

        # Bottom: ResultPanel
        self.result_panel = ResultPanel(right_pane)
        right_pane.add(self.result_panel, weight=1)

        # Connect result panel callback
        self.result_panel.on_target_selected = self._on_target_selected

    def _build_statusbar(self):
        statusbar = ttk.Frame(self.root)
        statusbar.pack(fill=tk.X, side=tk.BOTTOM)

        self._status_text = tk.StringVar(value="就绪")
        ttk.Label(statusbar, textvariable=self._status_text, foreground="gray").pack(
            side=tk.LEFT, padx=5,
        )

        self._coord_text = tk.StringVar(value="")
        ttk.Label(statusbar, textvariable=self._coord_text, foreground="gray").pack(
            side=tk.RIGHT, padx=5,
        )

    # ------------------------------------------------------------------
    # Menu / Toolbar actions
    # ------------------------------------------------------------------

    def _load_reference(self):
        """Load a reference (teaching) image."""
        filepath = filedialog.askopenfilename(
            parent=self.root,
            title="加载参考图",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("All files", "*.*"),
            ],
        )
        if not filepath:
            return

        try:
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError("无法读取图像")

            self._reference_image = img
            self._ref_canvas.load_image(img)
            self._ref_canvas.set_mode(CanvasMode.DRAW_ROI)
            self._ref_canvas.reset_roi()
            self._status_text.set(f"参考图: {os.path.basename(filepath)} ({img.shape[1]}×{img.shape[0]})")

            # Switch to reference tab
            self._notebook.select(0)

            # Reset workflow
            self._workflow = MultiTargetWorkflow()
            self.template_view.clear_tools()
            self.tool_panel.clear_tool_list()
            self.result_panel.clear_results()

        except Exception as e:
            messagebox.showerror("加载失败", str(e))

    def _load_inspection(self):
        """Load an inspection image."""
        filepath = filedialog.askopenfilename(
            parent=self.root,
            title="加载检测图",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("All files", "*.*"),
            ],
        )
        if not filepath:
            return

        try:
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError("无法读取图像")

            self._inspection_image = img
            self._insp_canvas.load_image(img)
            self._status_text.set(f"检测图: {os.path.basename(filepath)} ({img.shape[1]}×{img.shape[0]})")

            # Switch to inspection tab
            self._notebook.select(1)

        except Exception as e:
            messagebox.showerror("加载失败", str(e))

    def _save_project(self):
        """Save the current project."""
        if self._workflow is None or self._workflow._template_point is None:
            messagebox.showwarning("提示", "请先创建模板")
            return

        filepath = filedialog.asksaveasfilename(
            parent=self.root,
            title="保存项目",
            defaultextension=".npz",
            filetypes=[("Project files", "*.npz"), ("All files", "*.*")],
        )
        if not filepath:
            return

        try:
            self._workflow.save(filepath)
            self._current_project_path = filepath
            self._status_text.set(f"项目已保存: {os.path.basename(filepath)}")
        except Exception as e:
            messagebox.showerror("保存失败", str(e))

    def _load_project(self):
        """Load a saved project."""
        filepath = filedialog.askopenfilename(
            parent=self.root,
            title="加载项目",
            filetypes=[("Project files", "*.npz"), ("All files", "*.*")],
        )
        if not filepath:
            return

        try:
            self._workflow = MultiTargetWorkflow.load(filepath)
            self._current_project_path = filepath

            # Restore reference image
            ref_img = self._workflow._reference_image
            if ref_img is not None:
                self._reference_image = ref_img
                self._ref_canvas.load_image(ref_img)

                # Restore ROI
                self._ref_canvas.set_roi(
                    self._workflow.box_center,
                    self._workflow.box_size,
                    self._workflow.box_angle_deg,
                )
                self._ref_canvas.confirm_roi()

            # Restore template
            tmpl = self._workflow.template_image
            if tmpl is not None:
                self.template_view.load_template(tmpl)

            # Restore measurement tools
            self.tool_panel.clear_tool_list()
            self.template_view.clear_tools()
            for d in self._workflow.measurement_defs:
                self.tool_panel.add_tool_to_list(d["label"], d["object_type"])
                # Note: TemplateView tool overlays are not fully restored —
                # the user would need to re-draw them for visual feedback

            self.tool_panel.set_template_created(True)
            self._exec_btn.state(["!disabled"])
            self._status_text.set(f"项目已加载: {os.path.basename(filepath)}")
            self._notebook.select(0)

        except Exception as e:
            messagebox.showerror("加载失败", str(e))

    def _execute(self):
        """Execute multi-target measurement on the inspection image."""
        if self._workflow is None or self._workflow._template_point is None:
            messagebox.showwarning("提示", "请先创建/加载模板")
            return

        if self._inspection_image is None:
            messagebox.showwarning("提示", "请先加载检测图")
            return

        try:
            self.tool_panel.set_progress(True)
            self._status_text.set("正在执行多目标测量...")
            self.root.update_idletasks()

            # Sync workflow params from tool panel
            self._sync_workflow_params()

            results = self._workflow.measure(self._inspection_image)

            # Display results
            self.result_panel.set_results(
                results,
                self._workflow.summary_text(),
            )

            # Visualize on inspection canvas
            self._insp_canvas.clear_overlays()
            vis_img = self._workflow.visualize(self._inspection_image)
            self._insp_canvas.load_image(vis_img)

            self._status_text.set(
                f"测量完成: {len(results)} 个目标, "
                f"{sum(1 for t in results if t.valid)} 个有效"
            )

            # Switch to inspection tab
            self._notebook.select(1)

        except Exception as e:
            messagebox.showerror("测量失败", str(e))
        finally:
            self.tool_panel.set_progress(False)

    def _export_csv(self):
        """Export results (delegated to ResultPanel)."""
        pass  # Handled by ResultPanel._export_csv internally

    # ------------------------------------------------------------------
    # ROI callbacks
    # ------------------------------------------------------------------

    def _on_roi_changed(self, center_row, center_col, height, width, angle_deg):
        """Called when the ROI is being edited."""
        self.tool_panel.update_roi_info(
            (center_row, center_col), (height, width), angle_deg,
        )

        # Live preview: update template view
        if self._reference_image is not None:
            patch, _ = crop_and_straighten(
                self._reference_image,
                (center_row, center_col),
                (height, width),
                angle_deg,
            )
            self.template_view.load_template(patch)

    def _on_roi_confirmed(self, center_row, center_col, height, width, angle_deg):
        """Called when the user confirms the ROI (double-click)."""
        self._create_template_from_roi(center_row, center_col, height, width, angle_deg)

    # ------------------------------------------------------------------
    # Template creation
    # ------------------------------------------------------------------

    def _create_template(self):
        """Create template from current ROI."""
        if self._reference_image is None:
            messagebox.showwarning("提示", "请先加载参考图")
            return

        roi = self._ref_canvas.roi_center
        if roi is None:
            messagebox.showwarning("提示", "请在参考图上绘制旋转框")
            return

        self._create_template_from_roi(
            roi[0], roi[1],
            self._ref_canvas.roi_size[0],
            self._ref_canvas.roi_size[1],
            self._ref_canvas.roi_angle,
        )

    def _create_template_from_roi(self, center_row, center_col, height, width, angle_deg):
        """Create the MultiTargetWorkflow template from ROI parameters."""
        if self._reference_image is None:
            return

        # Parse angle range
        angle_str = self.tool_panel._angle_var.get()
        angle_val = float(angle_str.replace("±", "").replace("°", ""))
        angle_range = (-angle_val, angle_val)

        # Create/update workflow
        if self._workflow is None:
            self._workflow = MultiTargetWorkflow()

        # Apply preprocessor selection
        from measure_template import (
            CannyPreprocessor,
            CLAHEPreprocessor,
            RawPreprocessor,
            SobelPreprocessor,
            ThresholdPreprocessor,
        )

        preproc_map = {
            "raw": RawPreprocessor(),
            "canny": CannyPreprocessor(50.0, 150.0),
            "sobel": SobelPreprocessor(3),
            "clahe": CLAHEPreprocessor(2.0),
            "threshold": ThresholdPreprocessor(128.0),
        }
        preproc_name = self.tool_panel._preproc_var.get()
        preprocessor = preproc_map.get(preproc_name, RawPreprocessor())

        self._workflow.teach_template(
            self._reference_image,
            center=(center_row, center_col),
            size=(height, width),
            angle_deg=angle_deg,
            preprocessor=preprocessor,
            match_score_threshold=self.tool_panel._score_var.get(),
            angle_range=angle_range,
            angle_step=1.0,
            max_matches=self.tool_panel._max_matches_var.get(),
            coarse_fine=True,
            coarse_angle_step=5.0,
        )

        # Update template view
        self.template_view.load_template(self._workflow.template_image)

        # Clear existing measurement tools
        self.tool_panel.clear_tool_list()
        self.template_view.clear_tools()
        self._workflow.clear_measurements()

        # Enable execute
        self.tool_panel.set_template_created(True)
        self._exec_btn.state(["!disabled"])
        self._status_text.set(f"模板已创建: {width:.0f}×{height:.0f} @ {angle_deg:.1f}°")

    # ------------------------------------------------------------------
    # Template tool callbacks
    # ------------------------------------------------------------------

    def _on_template_tool_added(self, object_type: str, label: str, params: dict):
        """A measurement tool was added on the template view."""
        if self._workflow is None:
            return

        self._workflow.add_measurement(object_type, label, **params)
        self.tool_panel.add_tool_to_list(label, object_type)

        self._status_text.set(f"已添加测量工具: {label} ({object_type})")

    def _on_template_tool_removed(self, label: str):
        """A measurement tool was removed from the template view."""
        if self._workflow is None:
            return

        self._workflow.remove_measurement(label)
        self.tool_panel.remove_tool_from_list(label)

    def _on_tool_edit(self, label: str):
        """Edit a tool's parameters."""
        messagebox.showinfo("提示", f"编辑功能: {label}\n(可通过模板预览图重新绘制该工具)")

    def _on_tool_delete(self, label: str):
        """Delete a tool from the tool panel."""
        if self._workflow is None:
            return

        self._workflow.remove_measurement(label)
        self.template_view.remove_tool(label)
        self.tool_panel.remove_tool_from_list(label)

        self._status_text.set(f"已删除测量工具: {label}")

    def _on_add_composed(self, obj_type: str, label: str, params: dict):
        """Add a composed measurement from the tool panel."""
        if self._workflow is None:
            return

        self._workflow.add_measurement(obj_type, label, **params)
        self.tool_panel.add_tool_to_list(label, obj_type)

        self._status_text.set(f"已添加组合测量: {label} ({obj_type})")

    # ------------------------------------------------------------------
    # Parameter change callbacks
    # ------------------------------------------------------------------

    def _sync_workflow_params(self):
        """Sync tool panel params to the workflow before execution."""
        if self._workflow is None:
            return
        self._workflow.match_score_threshold = self.tool_panel._score_var.get()
        self._workflow.max_matches = self.tool_panel._max_matches_var.get()
        angle_str = self.tool_panel._angle_var.get()
        angle_val = float(angle_str.replace("±", "").replace("°", ""))
        self._workflow.angle_range = (-angle_val, angle_val)

    def _on_preprocessor_changed(self, name: str):
        if self._workflow:
            self._status_text.set(f"预处理已切换: {name} (重新创建模板生效)")

    def _on_score_threshold_changed(self, value: float):
        if self._workflow:
            self._workflow.match_score_threshold = value

    def _on_angle_range_changed(self, value: str):
        if self._workflow:
            angle_val = float(value.replace("±", "").replace("°", ""))
            self._workflow.angle_range = (-angle_val, angle_val)

    def _on_max_matches_changed(self, value: int):
        if self._workflow:
            self._workflow.max_matches = value

    def _on_export_csv(self):
        """Export results (handled by ResultPanel)."""
        self.result_panel._export_csv()

    # ------------------------------------------------------------------
    # Result callbacks
    # ------------------------------------------------------------------

    def _on_target_selected(self, target: TargetResult):
        """A target was selected in the result panel."""
        # Scroll the inspection canvas to show this target
        if self._inspection_image is not None:
            # Draw a highlight on the inspection canvas or just navigate
            self._status_text.set(
                f"选中 Target #{target.id}: "
                f"({target.center_row:.1f}, {target.center_col:.1f}) "
                f"score={target.score:.3f}"
            )

    # ------------------------------------------------------------------
    # Zoom
    # ------------------------------------------------------------------

    def _zoom_to_fit(self):
        canvas = self._get_active_canvas()
        if canvas:
            canvas.zoom_to_fit()

    def _zoom_100(self):
        canvas = self._get_active_canvas()
        if canvas:
            canvas.zoom_to_100()

    def _reset_roi(self):
        if self._ref_canvas.get_mode() == CanvasMode.DRAW_ROI:
            self._ref_canvas.reset_roi()
            self.tool_panel.clear_roi_info()

    def _cancel_drawing(self):
        """Escape key: cancel current operation."""
        if self._ref_canvas.get_mode() == CanvasMode.DRAW_ROI:
            self._ref_canvas.set_mode(CanvasMode.BROWSE)

    def _get_active_canvas(self) -> Optional[ImageCanvas]:
        tab_idx = self._notebook.index(self._notebook.select())
        if tab_idx == 0:
            return self._ref_canvas
        else:
            return self._insp_canvas

    def _on_tab_changed(self, event):
        """Update mode based on active tab."""
        tab_idx = self._notebook.index(self._notebook.select())
        if tab_idx == 0:
            self._mode_label.config(text="教学模式", foreground="blue")
            if self._ref_canvas.roi_center is None or not self._ref_canvas.roi_confirmed:
                self._ref_canvas.set_mode(CanvasMode.DRAW_ROI)
        else:
            self._mode_label.config(text="检测模式", foreground="green")
            self._insp_canvas.set_mode(CanvasMode.VIEW_RESULT)

    # ------------------------------------------------------------------
    # Help
    # ------------------------------------------------------------------

    def _show_help(self):
        help_text = """工业视觉测量系统 — 使用说明

【教学模式】
1. 加载参考图 (Ctrl+O)
2. 在参考图上画旋转目标框:
   - 点击确定中心点
   - 拖拽定义框的宽高
   - 鼠标滚轮调整旋转角度
   - 双击确认 ROI
3. 在右侧模板预览上添加测量工具:
   - 选择工具类型（边缘点/边缘对/拟合直线/拟合圆）
   - 在模板图上点击拖拽放置
   - 在弹出的对话框中调整算法参数
4. 可在工具栏添加组合测量（距离/角度等）
5. 保存项目 (Ctrl+S)

【检测模式】
1. 加载检测图 (Ctrl+I)
2. 点击执行测量 (Ctrl+E)
3. 在底部结果面板查看每个目标的测量结果
4. 可导出 CSV

【快捷键】
Ctrl+O: 加载参考图
Ctrl+I: 加载检测图
Ctrl+S: 保存项目
Ctrl+E: 执行测量
Escape: 取消当前操作
"""
        messagebox.showinfo("使用说明", help_text)

    def _show_about(self):
        messagebox.showinfo(
            "关于",
            "工业视觉测量系统 v1.0\n\n"
            "基于 Halcon 1D/2D 测量算法的 Python 实现\n"
            "整合多目标模板匹配与可组合测量工作流\n\n"
            "© 2026 Measure Project",
        )


# ===========================================================================
# Entry point
# ===========================================================================


def main():
    """Launch the GUI application."""
    root = tk.Tk()
    app = MeasureApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
