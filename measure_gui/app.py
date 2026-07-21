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

from .alignment import MultiPointAlignment, SingleBoxAlignment
from .image_canvas import CanvasMode, ImageCanvas
from .multi_target import MultiTargetWorkflow, TargetResult
from .project_manager import ProjectManager
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
        self._current_project_dir: Optional[str] = None

        # File path tracking
        self._reference_image_path: Optional[str] = None
        self._inspection_image_path: Optional[str] = None
        self._created_at: Optional[str] = None

        # State
        self._teaching: bool = True  # True = teaching mode, False = inspection mode

        # Build UI
        self._build_menu()
        self._build_main_area()
        self._build_statusbar()

        # Configure styles
        self._setup_styles()

        # Bind keyboard shortcuts
        self.root.bind("<Control-o>", lambda e: self._load_project())
        self.root.bind("<Control-Shift-O>", lambda e: self._load_reference())
        self.root.bind("<Control-i>", lambda e: self._load_inspection())
        self.root.bind("<Control-s>", lambda e: self._save_project())
        self.root.bind("<Control-Shift-S>", lambda e: self._save_project_as())
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

        # Project operations
        file_menu.add_command(label="新建项目...", command=self._new_project)
        file_menu.add_command(label="打开项目... (Ctrl+O)", command=self._load_project)
        file_menu.add_separator()
        file_menu.add_command(label="保存项目 (Ctrl+S)", command=self._save_project)
        file_menu.add_command(label="另存项目... (Ctrl+Shift+S)", command=self._save_project_as)
        file_menu.add_separator()

        # Image loading
        file_menu.add_command(label="加载参考图... (Ctrl+Shift+O)",
                              command=self._load_reference)
        file_menu.add_command(label="加载检测图... (Ctrl+I)",
                              command=self._load_inspection)
        file_menu.add_separator()

        # Recent projects submenu
        self._recent_menu = tk.Menu(file_menu, tearoff=0)
        file_menu.add_cascade(label="最近项目", menu=self._recent_menu)
        self._rebuild_recent_menu()

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

    def _build_main_area(self):
        # Horizontal PanedWindow
        self._main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self._main_pane.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # Left: ToolPanel
        self.tool_panel = ToolPanel(self._main_pane)
        self._main_pane.add(self.tool_panel, weight=0)

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
        self.tool_panel.on_overlap_changed = self._on_overlap_changed
        self.tool_panel.on_add_composed = self._on_add_composed
        self.tool_panel.on_tool_edit = self._on_tool_edit
        self.tool_panel.on_tool_delete = self._on_tool_delete
        self.tool_panel.on_export_csv = self._on_export_csv
        self.tool_panel.on_alignment_mode_changed = self._on_alignment_mode_changed
        self.tool_panel.on_tool_visibility_changed = self._on_tool_visibility_changed
        self.tool_panel.on_debug_save_changed = self._on_debug_save_changed

        # Right side: vertical paned window
        self._right_pane = ttk.PanedWindow(self._main_pane, orient=tk.VERTICAL)
        self._main_pane.add(self._right_pane, weight=1)

        # Center: Notebook for reference/inspection + template preview
        center_frame = ttk.Frame(self._right_pane)
        self._right_pane.add(center_frame, weight=3)

        # Center sub-pane: image notebook + template preview
        self._center_pane = ttk.PanedWindow(center_frame, orient=tk.HORIZONTAL)
        self._center_pane.pack(fill=tk.BOTH, expand=True)

        # Image notebook (reference / inspection tabs)
        self._notebook = ttk.Notebook(self._center_pane)
        self._center_pane.add(self._notebook, weight=3)

        # Reference image tab
        self._ref_frame = ttk.Frame(self._notebook)
        self._notebook.add(self._ref_frame, text="参考图")
        self._ref_canvas = ImageCanvas(self._ref_frame, width=800, height=600)
        self._ref_canvas.pack(fill=tk.BOTH, expand=True)
        self._ref_canvas.on_roi_changed = self._on_roi_changed
        self._ref_canvas.on_roi_confirmed = self._on_roi_confirmed
        self._ref_canvas.on_control_points_confirmed = self._on_control_points_confirmed

        # Inspection image tab
        self._insp_frame = ttk.Frame(self._notebook)
        self._notebook.add(self._insp_frame, text="检测图")
        self._insp_canvas = ImageCanvas(self._insp_frame, width=800, height=600)
        self._insp_canvas.pack(fill=tk.BOTH, expand=True)
        self._insp_canvas.set_mode(CanvasMode.VIEW_RESULT)

        self._notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        # Template preview (right side)
        self.template_view = TemplateView(self._center_pane, width=350, height=350)
        self._center_pane.add(self.template_view, weight=1)

        # Connect template view callbacks
        self.template_view.on_tool_added = self._on_template_tool_added
        self.template_view.on_tool_removed = self._on_template_tool_removed
        self.template_view.on_tool_edited = self._on_template_tool_edited

        # Bottom: ResultPanel
        self.result_panel = ResultPanel(self._right_pane)
        self._right_pane.add(self.result_panel, weight=1)

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
            self._reference_image_path = filepath
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
            self._inspection_image_path = filepath
            self._insp_canvas.load_image(img)
            self._status_text.set(f"检测图: {os.path.basename(filepath)} ({img.shape[1]}×{img.shape[0]})")

            # Switch to inspection tab
            self._notebook.select(1)

        except Exception as e:
            messagebox.showerror("加载失败", str(e))

    def _save_project(self):
        """Save the current project. Uses current project dir or prompts for one."""
        if self._workflow is None or self._workflow._alignment.template_point is None:
            messagebox.showwarning("提示", "请先创建模板")
            return

        # Determine project directory
        if self._current_project_dir and os.path.isdir(self._current_project_dir):
            project_dir = self._current_project_dir
        else:
            project_dir = filedialog.askdirectory(
                parent=self.root,
                title="选择项目保存目录",
                mustexist=False,
            )
            if not project_dir:
                return
            os.makedirs(project_dir, exist_ok=True)

        try:
            ProjectManager.save_project(project_dir, self)
            self._current_project_dir = project_dir
            self._current_project_path = None
            project_name = os.path.basename(project_dir)
            self.root.title(f"工业视觉测量系统 — {project_name}")
            self._status_text.set(f"项目已保存: {project_dir}")
            self._rebuild_recent_menu()
        except Exception as e:
            messagebox.showerror("保存失败", str(e))

    def _save_project_as(self):
        """Save project to a new directory (always prompts)."""
        if self._workflow is None or self._workflow._alignment.template_point is None:
            messagebox.showwarning("提示", "请先创建模板")
            return

        project_dir = filedialog.askdirectory(
            parent=self.root,
            title="选择项目保存目录",
            mustexist=False,
        )
        if not project_dir:
            return

        os.makedirs(project_dir, exist_ok=True)

        try:
            ProjectManager.save_project(project_dir, self)
            self._current_project_dir = project_dir
            self._current_project_path = None
            project_name = os.path.basename(project_dir)
            self.root.title(f"工业视觉测量系统 — {project_name}")
            self._status_text.set(f"项目已另存为: {project_dir}")
            self._rebuild_recent_menu()
        except Exception as e:
            messagebox.showerror("保存失败", str(e))

    def _load_project(self, project_path: str = None):
        """Load a project. Supports both new project directories and legacy .npz files.

        Args:
            project_path: If given, load this specific path directly
                          (used by recent projects menu).
        """
        if project_path:
            filepath = project_path
        else:
            filepath = filedialog.askopenfilename(
                parent=self.root,
                title="加载项目",
                filetypes=[
                    ("Project files", "*.npz"),
                    ("All files", "*.*"),
                ],
            )
            if not filepath:
                return

        try:
            # Detect whether this is a project directory or legacy .npz
            parent_dir = os.path.dirname(filepath)
            manifest_path = os.path.join(parent_dir, "project.json")

            if os.path.exists(manifest_path):
                # Full project load
                ProjectManager.load_project(parent_dir, self)
                self._rebuild_recent_menu()
            else:
                # Legacy .npz — prompt for migration
                result = messagebox.askyesnocancel(
                    "旧格式项目",
                    "这是一个旧格式的 .npz 项目文件。\n\n"
                    "选择「是」→ 迁移到新项目目录格式（推荐，可保存完整编辑状态）\n"
                    "选择「否」→ 以旧格式打开（部分状态无法恢复）\n"
                    "选择「取消」→ 不打开",
                )
                if result is True:
                    # Migrate
                    project_dir = filedialog.askdirectory(
                        parent=self.root,
                        title="选择迁移目标目录",
                        mustexist=False,
                    )
                    if not project_dir:
                        return
                    os.makedirs(project_dir, exist_ok=True)

                    # Copy the .npz to the project dir as workflow.npz
                    import shutil
                    dst_npz = os.path.join(project_dir, "workflow.npz")
                    shutil.copy2(filepath, dst_npz)

                    # Load workflow from the copy
                    self._workflow = MultiTargetWorkflow.load(dst_npz)
                    self._current_project_dir = project_dir
                    self._current_project_path = None
                    self._created_at = datetime.now().isoformat()

                    # Restore reference image
                    ref_img = self._workflow.reference_image
                    if ref_img is not None:
                        self._reference_image = ref_img
                        self._ref_canvas.load_image(ref_img)
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

                    # Restore measurement tools (tool panel only, no overlays)
                    self.tool_panel.clear_tool_list()
                    self.template_view.clear_tools()
                    for d in self._workflow.measurement_defs:
                        self.tool_panel.add_tool_to_list(d["label"], d["object_type"])

                    self.tool_panel.set_template_created(True)
                    pass  # toolbar removed, state managed by tool_panel

                    # Save as full project now
                    ProjectManager.save_project(project_dir, self)
                    self._rebuild_recent_menu()

                    project_name = os.path.basename(project_dir)
                    self.root.title(f"工业视觉测量系统 — {project_name}")
                    self._status_text.set(f"项目已迁移并保存: {project_dir}")
                    self._notebook.select(0)

                elif result is False:
                    # Open as legacy
                    ProjectManager.load_legacy_npz(filepath, self)
                # else: cancelled — do nothing

        except Exception as e:
            messagebox.showerror("加载失败", str(e))

    def _new_project(self):
        """Create a new empty project directory."""
        project_dir = filedialog.askdirectory(
            parent=self.root,
            title="选择/创建项目目录",
            mustexist=False,
        )
        if not project_dir:
            return

        os.makedirs(project_dir, exist_ok=True)
        self._current_project_dir = project_dir
        self._current_project_path = None
        self._created_at = datetime.now().isoformat()

        # Reset all state
        self._workflow = None
        self._reference_image = None
        self._inspection_image = None
        self._reference_image_path = None
        self._inspection_image_path = None
        self._ref_canvas.reset_roi()
        self.template_view.clear_tools()
        self.tool_panel.clear_tool_list()
        self.result_panel.clear_results()
        pass  # toolbar removed, state managed by tool_panel

        project_name = os.path.basename(project_dir)
        self.root.title(f"工业视觉测量系统 — {project_name}")
        self._status_text.set(f"新项目已创建: {project_dir}")

    def _recent_project_opened(self, path: str):
        """Open a project from the recent projects list."""
        if not os.path.exists(path):
            messagebox.showwarning("提示", f"项目目录不存在:\n{path}")
            ProjectManager.remove_recent_project(path)
            self._rebuild_recent_menu()
            return
        self._load_project(path)

    def _rebuild_recent_menu(self):
        """Rebuild the recent projects submenu."""
        self._recent_menu.delete(0, tk.END)
        projects = ProjectManager.get_recent_projects()
        if not projects:
            self._recent_menu.add_command(label="(无最近项目)", state=tk.DISABLED)
        else:
            for p in projects:
                label = f"{p['name']} — {p['path']}"
                self._recent_menu.add_command(
                    label=label,
                    command=lambda path=p["path"]: self._recent_project_opened(path),
                )

    def _execute(self):
        """Execute multi-target measurement on the inspection image."""
        if self._workflow is None or self._workflow._alignment.template_point is None:
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

            # Sync debug save toggle state (refresh dir timestamp)
            self._on_debug_save_changed(self.tool_panel.debug_save_enabled)

            import time
            _t_start = time.perf_counter()
            results = self._workflow.measure(self._inspection_image)
            _elapsed = time.perf_counter() - _t_start

            # 生成带耗时的摘要文本
            summary = self._workflow.summary_text()
            summary += f"\n\n{'='*40}\n"
            summary += f"执行耗时: {_elapsed*1000:.0f}ms ({_elapsed:.2f}s)"

            # Display results
            self.result_panel.set_results(
                results,
                summary,
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

            # Popup notification
            n_valid = sum(1 for t in results if t.valid)
            n_total = len(results)
            messagebox.showinfo(
                "测量完成",
                f"多目标测量执行完毕！\n\n"
                f"检测到 {n_total} 个目标\n"
                f"其中 {n_valid} 个测量有效\n\n"
                f"详情请查看底部结果面板。",
            )

        except Exception as e:
            messagebox.showerror("测量失败", str(e))
        finally:
            self.tool_panel.set_progress(False)

    def _export_csv(self):
        """Export results (delegated to ResultPanel)."""
        pass  # Handled by ResultPanel._export_csv internally

    # ------------------------------------------------------------------
    # State getters (for project serialization)
    # ------------------------------------------------------------------

    def get_gui_state(self) -> dict:
        """Collect GUI layout state for project manifest."""
        try:
            geometry = self.root.geometry()  # "1600x900+100+50"
        except Exception:
            geometry = "1600x900"
        try:
            window_state = self.root.state()  # "normal" or "zoomed"
        except Exception:
            window_state = "normal"

        # Collect sash positions from PanedWindows
        main_sash = []
        try:
            for i in range(10):  # arbitrary safe max
                coord = self._main_pane.sash_coord(i)
                if coord:
                    main_sash.append(coord[0] if isinstance(coord, tuple) else coord)
                else:
                    break
        except Exception:
            pass

        right_sash = []
        try:
            for i in range(10):
                coord = self._right_pane.sash_coord(i)
                if coord:
                    right_sash.append(coord[0] if isinstance(coord, tuple) else coord)
                else:
                    break
        except Exception:
            pass

        center_sash = []
        try:
            for i in range(10):
                coord = self._center_pane.sash_coord(i)
                if coord:
                    center_sash.append(coord[0] if isinstance(coord, tuple) else coord)
                else:
                    break
        except Exception:
            pass

        return {
            "active_notebook_tab": self._notebook.index(self._notebook.select()),
            "window_geometry": geometry,
            "window_state": window_state,
            "main_pane_sash_positions": main_sash,
            "right_pane_sash_positions": right_sash,
            "center_pane_sash_positions": center_sash,
        }

    def get_matching_state(self) -> dict:
        """Collect template matching settings for project manifest."""
        return {
            "preprocessor_type": self.tool_panel._preproc_var.get(),
            "match_score_threshold": self.tool_panel._score_var.get(),
            "angle_range_deg": self._parse_angle_range(),
            "angle_step_deg": self.tool_panel._angle_step_var.get(),
            "max_matches": self.tool_panel._max_matches_var.get(),
            "overlap": self.tool_panel._overlap_var.get(),
        }

    def get_tool_list_order(self) -> list:
        """Get ordered list of tool labels from the tool panel treeview."""
        return self.tool_panel.get_tool_list_order()

    def _get_roi_state(self) -> dict:
        """Collect current ROI state from reference canvas."""
        if self._ref_canvas.roi_center is None:
            return {}
        state = {
            "center_row": self._ref_canvas.roi_center[0],
            "center_col": self._ref_canvas.roi_center[1],
            "height": self._ref_canvas.roi_size[0],
            "width": self._ref_canvas.roi_size[1],
            "angle_deg": self._ref_canvas.roi_angle,
            "confirmed": self._ref_canvas.roi_confirmed,
            "alignment_mode": self.tool_panel.get_alignment_mode(),
        }
        if state["alignment_mode"] == "multi_point":
            state["control_points"] = [
                {"row": r, "col": c}
                for r, c in self._ref_canvas.get_control_points()
            ]
        return state

    def _parse_angle_range(self) -> float:
        """Parse angle range half-value from the tool panel combo string."""
        angle_str = self.tool_panel._angle_var.get()
        return float(angle_str.replace("±", "").replace("°", ""))

    # ------------------------------------------------------------------
    # State setters (for project deserialization)
    # ------------------------------------------------------------------

    def restore_gui_state(self, gui_state: dict):
        """Restore window geometry, notebook tab, and sash positions."""
        # Restore window geometry
        geometry = gui_state.get("window_geometry", "")
        if geometry:
            try:
                self.root.geometry(geometry)
            except Exception:
                pass

        # Restore window state (normal/zoomed)
        window_state = gui_state.get("window_state", "normal")
        if window_state == "zoomed":
            try:
                self.root.state("zoomed")
            except Exception:
                pass

        # Restore notebook tab
        tab = gui_state.get("active_notebook_tab", 0)
        try:
            self._notebook.select(tab)
        except Exception:
            pass

        # Restore sash positions (deferred via after_idle to let layout settle)
        def _restore_sashes():
            for i, pos in enumerate(gui_state.get("main_pane_sash_positions", [])):
                try:
                    self._main_pane.sashpos(i, int(pos))
                except Exception:
                    pass
            for i, pos in enumerate(gui_state.get("right_pane_sash_positions", [])):
                try:
                    self._right_pane.sashpos(i, int(pos))
                except Exception:
                    pass
            for i, pos in enumerate(gui_state.get("center_pane_sash_positions", [])):
                try:
                    self._center_pane.sashpos(i, int(pos))
                except Exception:
                    pass

        self.root.after(200, _restore_sashes)

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
        if self.tool_panel.get_alignment_mode() == "multi_point":
            # Switch to control-point mode; user places points, then
            # double-clicking again calls _on_control_points_confirmed
            self._ref_canvas.set_mode(CanvasMode.DRAW_CONTROL_POINTS)
            self._ref_canvas.set_roi(
                (center_row, center_col), (height, width), angle_deg
            )
        else:
            self._create_template_from_roi(center_row, center_col,
                                           height, width, angle_deg)

    def _on_control_points_confirmed(self, points):
        """Called when user confirms control points (double-click in
        DRAW_CONTROL_POINTS mode)."""
        if len(points) < 3:
            messagebox.showwarning("提示", "至少需要 3 个控制点")
            return
        roi = self._ref_canvas.roi_center
        if roi is None:
            messagebox.showwarning("提示", "请先绘制 ROI 框")
            return
        self._create_template_from_roi(
            roi[0], roi[1],
            self._ref_canvas.roi_size[0],
            self._ref_canvas.roi_size[1],
            self._ref_canvas.roi_angle,
            control_points=[
                (r, c) for r, c in points
            ],
        )

    def _on_alignment_mode_changed(self, mode_str: str):
        """Handle alignment mode change from the tool panel."""
        if mode_str == "多点仿射":
            # When multi-point is selected, confirm ROI first then move
            # to control-point placement
            pass
        else:
            # Single box: restore normal DRAW_ROI mode if needed
            if self._ref_canvas.get_mode() == CanvasMode.DRAW_CONTROL_POINTS:
                self._ref_canvas.set_mode(CanvasMode.DRAW_ROI)
                self._ref_canvas.clear_control_points()

    def _parse_pyramid_decimate(self) -> int:
        """Parse the pyramid decimation level from the tool panel combobox."""
        val = self.tool_panel._pyramid_var.get()
        mapping = {
            "禁用": 0,
            "2x": 1,
            "4x (推荐)": 2,
            "8x": 3,
        }
        return mapping.get(val, 2)

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

    def _create_template_from_roi(self, center_row, center_col, height, width,
                                   angle_deg, control_points=None):
        """Create the MultiTargetWorkflow template from ROI parameters.

        Args:
            control_points: Optional list of (row, col) for multi-point
                            affine alignment.  If given, the alignment
                            strategy is switched to MultiPointAlignment.
        """
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
        from measure.measure_template import (
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

        # Set alignment strategy based on mode
        use_multi_point = (
            control_points is not None and len(control_points) >= 3
        )
        if use_multi_point:
            strategy = MultiPointAlignment()
        else:
            strategy = SingleBoxAlignment()
        self._workflow._alignment = strategy

        self._workflow.teach_template(
            self._reference_image,
            center=(center_row, center_col),
            size=(height, width),
            angle_deg=angle_deg,
            preprocessor=preprocessor,
            match_score_threshold=self.tool_panel._score_var.get(),
            angle_range=angle_range,
            angle_step=self.tool_panel._angle_step_var.get(),
            max_matches=self.tool_panel._max_matches_var.get(),
            overlap=self.tool_panel._overlap_var.get(),
            coarse_fine=True,
            coarse_angle_step=5.0,
            pyramid_decimate=self._parse_pyramid_decimate(),
            pyramid_max_template_size=400,
        )

        # For multi-point, register control points with the alignment
        if use_multi_point and isinstance(strategy, MultiPointAlignment):
            for i, (r, c) in enumerate(control_points):
                strategy.add_control_point(
                    f"cp_{i}", float(r), float(c))
            strategy.build_all_control_point_templates()

        # Update template view
        self.template_view.load_template(self._workflow.template_image)

        # Clear existing measurement tools
        self.tool_panel.clear_tool_list()
        self.template_view.clear_tools()
        self._workflow.clear_measurements()

        # Enable execute
        self.tool_panel.set_template_created(True)
        pass  # toolbar removed, state managed by tool_panel
        self._status_text.set(f"模板已创建: {width:.0f}×{height:.0f} @ {angle_deg:.1f}°")

        # Popup notification
        messagebox.showinfo(
            "模板创建完成",
            f"模板已成功创建！\n\n"
            f"尺寸: {width:.0f} × {height:.0f} px\n"
            f"角度: {angle_deg:.1f}°\n"
            f"预处理: {preproc_name}\n\n"
            f"现在可以在右侧模板预览上添加测量工具。",
        )

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

    def _on_template_tool_edited(self, label: str, params: dict):
        """A measurement tool was edited on the template view (double-click or re-click)."""
        if self._workflow is None:
            return

        # Sync edited params to the workflow measurement_def
        self._workflow.update_measurement(label, **params)

        self._status_text.set(f"已更新测量工具: {label}")

    def _on_tool_edit(self, label: str):
        """Edit a tool's parameters (triggered from tool panel Edit button).

        Opens the parameter dialog pre-filled with the tool's current params.
        Updates both the template_view overlay and workflow measurement_def.
        """
        # Find the tool in template_view
        tool = None
        for t in self.template_view._tools:
            if t["label"] == label:
                tool = t
                break

        if tool is None:
            messagebox.showwarning("提示", f"未找到工具: {label}")
            return

        obj_type = tool["object_type"]
        params = tool["params"]

        from .dialogs import (
            EdgePairDialog,
            EdgePointDialog,
            FitCircleDialog,
            FitLineDialog,
            TemplateMatchPointDialog,
        )

        dlg_params = None
        if obj_type == "EdgePoint":
            dlg_params = EdgePointDialog.ask(self.root, params=dict(params))
        elif obj_type == "EdgePair":
            dlg_params = EdgePairDialog.ask(self.root, params=dict(params))
        elif obj_type == "FitLine":
            dlg_params = FitLineDialog.ask(self.root, params=dict(params))
        elif obj_type == "FitCircle":
            dlg_params = FitCircleDialog.ask(self.root, params=dict(params))
        elif obj_type == "TemplateMatchPoint":
            dlg_params = TemplateMatchPointDialog.ask(self.root, params=dict(params))

        if dlg_params is None:
            return  # User cancelled

        # Update template_view tool params and redraw overlay
        params.update(dlg_params)
        self.template_view._redraw_tools()

        # Update workflow measurement_def
        if self._workflow is not None:
            self._workflow.update_measurement(label, **dlg_params)

        self._status_text.set(f"已更新测量工具: {label}")

    def _on_tool_delete(self, label: str):
        """Delete a tool from the tool panel."""
        if self._workflow is None:
            return

        self._workflow.remove_measurement(label)
        self.template_view.remove_tool(label)
        self.tool_panel.remove_tool_from_list(label)

        self._status_text.set(f"已删除测量工具: {label}")

    def _on_debug_save_changed(self, enabled: bool):
        """Toggle debug image saving."""
        if self._workflow is None:
            return
        if enabled and self._current_project_dir:
            import time
            debug_dir = os.path.join(
                self._current_project_dir,
                f"debug_{time.strftime('%Y%m%d_%H%M%S')}",
            )
            self._workflow.debug_dir = debug_dir
        else:
            self._workflow.debug_dir = None

    def _on_tool_visibility_changed(self, label: str, visible: bool):
        """Toggle tool overlay visibility."""
        if visible:
            self.template_view.show_tool(label)
        else:
            self.template_view.hide_tool(label)

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
        self._workflow.overlap = self.tool_panel._overlap_var.get()
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

    def _on_overlap_changed(self, value: float):
        if self._workflow:
            self._workflow.overlap = value

    def _on_alignment_mode_changed(self, mode_name: str):
        """Switch between single-box and multi-point alignment modes."""
        if mode_name == "多点仿射":
            self._ref_canvas.set_mode(CanvasMode.DRAW_CONTROL_POINTS)
            self._status_text.set("对齐模式: 多点仿射 — 点击放置 ≥3 个控制点")
        else:
            self._ref_canvas.set_mode(CanvasMode.DRAW_ROI)
            self._status_text.set("对齐模式: 旋转框 — 画旋转矩形")

    def _on_control_points_confirmed(self, points: list):
        """Handle multi-point control point confirmation (double-click)."""
        if self._reference_image is None:
            return
        if len(points) < 3:
            messagebox.showwarning("提示", "多点仿射需要至少 3 个控制点")
            return

        alignment = MultiPointAlignment()
        angle_str = self.tool_panel._angle_var.get()
        angle_val = float(angle_str.replace("±", "").replace("°", ""))
        alignment.set_parent_search_angle_range((-angle_val, angle_val))

        for i, (row, col) in enumerate(points):
            alignment.add_control_point(
                label=f"cp_{i}", ref_row=float(row), ref_col=float(col),
                template_size=40,
            )

        if self._workflow is None:
            self._workflow = MultiTargetWorkflow()

        from measure.measure_template import (
            CannyPreprocessor, CLAHEPreprocessor, RawPreprocessor,
            SobelPreprocessor, ThresholdPreprocessor,
        )
        preproc_map = {
            "raw": RawPreprocessor(), "canny": CannyPreprocessor(50.0, 150.0),
            "sobel": SobelPreprocessor(3), "clahe": CLAHEPreprocessor(2.0),
            "threshold": ThresholdPreprocessor(128.0),
        }
        preprocessor = preproc_map.get(
            self.tool_panel._preproc_var.get(), RawPreprocessor()
        )

        alignment.teach(self._reference_image, preprocessor)
        self._workflow.alignment = alignment

        tmpl_img = alignment.template_image
        if tmpl_img is not None:
            self.template_view.load_template(tmpl_img)

        self.tool_panel.clear_tool_list()
        self.tool_panel.set_template_created(True)
        self._teaching = True
        self._ref_canvas.confirm_control_points()

        h, w = tmpl_img.shape if tmpl_img is not None else (0, 0)
        pp_name = self.tool_panel._preproc_var.get()
        messagebox.showinfo(
            "模板创建成功",
            f"对齐模式: 多点仿射\n控制点: {len(points)} 个\n"
            f"模板尺寸: {w}×{h} px\n预处理: {pp_name}",
        )
        self._status_text.set(
            f"模板已创建 (多点仿射, {len(points)} 控制点, {w}×{h})"
        )

    def get_alignment_state(self) -> dict:
        """Collect alignment mode and control points for project serialization."""
        mode = self.tool_panel._alignment_var.get()
        state: dict = {"mode": "single_box" if mode != "多点仿射" else "multi_point"}
        if state["mode"] == "multi_point":
            points = self._ref_canvas.get_control_points()
            state["control_points"] = [
                {"label": f"cp_{i}", "ref_row": r, "ref_col": c}
                for i, (r, c) in enumerate(points)
            ]
        return state

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
            pass  # toolbar removed, mode handled by tool_panel
            if self._ref_canvas.roi_center is None or not self._ref_canvas.roi_confirmed:
                self._ref_canvas.set_mode(CanvasMode.DRAW_ROI)
        else:
            pass  # toolbar removed, mode handled by tool_panel
            self._insp_canvas.set_mode(CanvasMode.VIEW_RESULT)

    # ------------------------------------------------------------------
    # Help
    # ------------------------------------------------------------------

    def _show_help(self):
        help_text = """工业视觉测量系统 — 使用说明

【项目工程】
  - 新建项目: 文件 → 新建项目... 创建项目目录
  - 打开项目: 文件 → 打开项目... (Ctrl+O) 加载已有项目
  - 保存项目: 文件 → 保存项目 (Ctrl+S) 保存到项目目录
  - 另存为:   文件 → 另存项目... (Ctrl+Shift+S)
  - 最近项目: 文件 → 最近项目 → 快速打开
项目目录包含 project.json (设置+状态) + workflow.npz (模板数据) + 图片副本

【教学模式】
1. 加载参考图 (Ctrl+Shift+O)
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
Ctrl+O: 打开项目
Ctrl+S: 保存项目
Ctrl+Shift+S: 另存项目
Ctrl+Shift+O: 加载参考图
Ctrl+I: 加载检测图
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
