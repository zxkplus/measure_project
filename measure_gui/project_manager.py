"""
Project file management for the Measure GUI.

Handles:
  - Full project save/load (project.json manifest + workflow.npz)
  - Legacy .npz migration
  - Recent projects tracking (~/.measure_gui_recent.json)

A project is a directory containing:
  project.json   — human-readable manifest (settings + GUI state)
  workflow.npz   — core data (template pixels, measurement definitions)
  reference.png  — copy of the reference image
  inspection.png — copy of the inspection image (optional)
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import cv2
import numpy as np

from .multi_target import MultiTargetWorkflow

if TYPE_CHECKING:
    from .app import MeasureApp

MANIFEST_FILENAME = "project.json"
WORKFLOW_FILENAME = "workflow.npz"
REF_IMAGE_FILENAME = "reference.png"
INSP_IMAGE_FILENAME = "inspection.png"
MANIFEST_VERSION = 2
MAX_RECENT = 10


def _json_default(obj: Any) -> Any:
    """Convert numpy / Python types for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _to_json(obj: Any) -> Any:
    """Recursively walk a dict/list and convert tuple/numpy values to JSON-safe types."""
    if isinstance(obj, dict):
        return {k: _to_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


class ProjectManager:
    """Static methods for project I/O. No instance state."""

    RECENT_FILE = os.path.expanduser("~/.measure_gui_recent.json")

    # =====================================================================
    # Save
    # =====================================================================

    @staticmethod
    def save_project(project_dir: str, app: MeasureApp) -> None:
        """Save complete project state to a directory.

        1. Writes project.json manifest
        2. Saves workflow.npz
        3. Copies reference/inspection images into the directory
        4. Updates recent projects list
        """
        os.makedirs(project_dir, exist_ok=True)

        # 1. Build and save manifest
        manifest = ProjectManager._build_manifest(app, project_dir)
        manifest_path = os.path.join(project_dir, MANIFEST_FILENAME)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False,
                      default=_json_default)

        # 2. Save workflow (pixel data)
        workflow_path = os.path.join(project_dir, WORKFLOW_FILENAME)
        app._workflow.save(workflow_path)

        # 3. Copy images into project dir
        if app._reference_image is not None:
            ref_path = os.path.join(project_dir, REF_IMAGE_FILENAME)
            cv2.imwrite(ref_path, app._reference_image)

        if app._inspection_image is not None:
            insp_path = os.path.join(project_dir, INSP_IMAGE_FILENAME)
            cv2.imwrite(insp_path, app._inspection_image)

        # 4. Track recent projects
        ProjectManager.add_recent_project(
            project_dir,
            os.path.basename(project_dir),
        )

    @staticmethod
    def _build_manifest(app: MeasureApp, project_dir: str) -> Dict[str, Any]:
        """Collect all state from the app into a manifest dict."""
        now = datetime.now(timezone.utc).isoformat()

        # Determine created_at: keep existing if already set, else now
        created_at = getattr(app, '_created_at', None) or now

        manifest: Dict[str, Any] = {
            "version": MANIFEST_VERSION,
            "project_name": os.path.basename(project_dir),
            "created_at": created_at,
            "updated_at": now,
            "reference_image_path": getattr(app, '_reference_image_path', None),
            "inspection_image_path": getattr(app, '_inspection_image_path', None),
            "matching": _to_json(app.get_matching_state()),
            "roi": _to_json(app._get_roi_state()),
            "ref_canvas_state": _to_json(app._ref_canvas.get_view_state()),
            "insp_canvas_state": _to_json(app._insp_canvas.get_view_state()),
            "template_view": _to_json(app.template_view.get_state()),
            "tool_list_order": _to_json(app.get_tool_list_order()),
            "tool_visibility": _to_json(dict(app.tool_panel._tool_visibility)),
            "gui": _to_json(app.get_gui_state()),
            "alignment": _to_json(app.get_alignment_state()),
        }
        return manifest

    # =====================================================================
    # Load
    # =====================================================================

    @staticmethod
    def load_project(project_dir: str, app: MeasureApp) -> None:
        """Load a complete project from a directory.

        1. Reads project.json manifest
        2. Loads workflow.npz
        3. Restores all GUI state (ROI, tools, canvas views, layout)
        4. Updates recent projects list
        """
        manifest_path = os.path.join(project_dir, MANIFEST_FILENAME)
        workflow_path = os.path.join(project_dir, WORKFLOW_FILENAME)

        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"项目清单不存在: {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        version = manifest.get("version", 0)
        if version > MANIFEST_VERSION:
            raise ValueError(
                f"项目由更高版本软件保存 (v{version})，"
                f"当前支持最高 v{MANIFEST_VERSION}。请升级软件。"
            )

        # Load workflow
        if not os.path.exists(workflow_path):
            raise FileNotFoundError(f"工作流文件不存在: {workflow_path}")

        app._workflow = MultiTargetWorkflow.load(workflow_path)
        app._current_project_dir = project_dir
        app._current_project_path = None
        app._created_at = manifest.get("created_at")

        # Restore reference image
        ref_img = app._workflow.reference_image
        if ref_img is not None:
            app._reference_image = ref_img
            app._ref_canvas.load_image(ref_img)

            # Restore ROI
            roi = manifest.get("roi", {})
            if roi:
                app._ref_canvas.set_roi(
                    (roi["center_row"], roi["center_col"]),
                    (roi["height"], roi["width"]),
                    roi["angle_deg"],
                )
                if roi.get("confirmed", False):
                    app._ref_canvas.confirm_roi()

                # Restore alignment mode
                alignment_mode = roi.get("alignment_mode", "single_box")
                app.tool_panel.set_alignment_mode(alignment_mode)

                # Restore control points if multi-point
                if alignment_mode == "multi_point":
                    cps = roi.get("control_points", [])
                    if cps:
                        points = [(cp["row"], cp["col"]) for cp in cps]
                        app._ref_canvas.set_control_points(points)

            # Restore ref canvas view state
            ref_state = manifest.get("ref_canvas_state", {})
            if ref_state:
                app._ref_canvas.set_view_state(ref_state)

        # Restore template image + tools in template view
        tmpl = app._workflow.template_image
        if tmpl is not None:
            app.template_view.load_template(tmpl, clear_tools=False)
            tmpl_state = manifest.get("template_view", {})
            if tmpl_state:
                app.template_view.set_state(tmpl_state)

        # Restore tool panel treeview (order from manifest, data from workflow)
        tools_from_wf = app._workflow.measurement_defs
        tool_order = manifest.get("tool_list_order", [])
        tool_visibility = manifest.get("tool_visibility", {})
        app.tool_panel.clear_tool_list()
        if tool_visibility:
            app.tool_panel._tool_visibility = dict(tool_visibility)
            app.template_view.set_tool_visibility(tool_visibility)
        app.tool_panel.restore_tool_list(tools_from_wf, tool_order)

        # Restore matching params in UI widgets
        matching = manifest.get("matching", {})
        if matching:
            app.tool_panel.set_matching_params(
                matching.get("preprocessor_type", "raw"),
                matching.get("match_score_threshold", 0.5),
                matching.get("angle_range_deg", 30.0),
                matching.get("max_matches", 0),
                matching.get("overlap", 0.3),
            )

        # Restore inspection image if available
        insp_path = manifest.get("inspection_image_path")
        if insp_path:
            abs_insp = os.path.join(project_dir, insp_path)
            if os.path.exists(abs_insp):
                insp_img = cv2.imread(abs_insp, cv2.IMREAD_GRAYSCALE)
                if insp_img is not None:
                    app._inspection_image = insp_img
                    app._insp_canvas.load_image(insp_img)
                    insp_state = manifest.get("insp_canvas_state", {})
                    if insp_state:
                        app._insp_canvas.set_view_state(insp_state)

        # Restore reference/inspection image path tracking
        app._reference_image_path = manifest.get("reference_image_path")
        app._inspection_image_path = manifest.get("inspection_image_path")

        # Restore GUI layout state (geometry, tab, sash positions)
        gui_state = manifest.get("gui", {})
        if gui_state:
            app.restore_gui_state(gui_state)

        # Enable execute button
        app.tool_panel.set_template_created(True)
        app._exec_btn.state(["!disabled"])

        # Update window title
        project_name = os.path.basename(project_dir)
        app.root.title(f"工业视觉测量系统 — {project_name}")
        app._status_text.set(f"项目已加载: {project_dir}")

        # Update recent projects
        ProjectManager.add_recent_project(project_dir, project_name)

    @staticmethod
    def load_legacy_npz(filepath: str, app: MeasureApp) -> None:
        """Load a legacy standalone .npz file (partial state restore).

        This is the backward-compatible path for old-format saves.
        Tool overlays on the template view will NOT be restored —
        only the reference image, ROI, template image, and measurement
        definitions in the tool panel list are restored.
        """
        app._workflow = MultiTargetWorkflow.load(filepath)
        app._current_project_path = filepath
        app._current_project_dir = None

        ref_img = app._workflow.reference_image
        if ref_img is not None:
            app._reference_image = ref_img
            app._ref_canvas.load_image(ref_img)
            app._ref_canvas.set_roi(
                app._workflow.box_center,
                app._workflow.box_size,
                app._workflow.box_angle_deg,
            )
            app._ref_canvas.confirm_roi()

        tmpl = app._workflow.template_image
        if tmpl is not None:
            app.template_view.load_template(tmpl)

        app.tool_panel.clear_tool_list()
        app.template_view.clear_tools()
        for d in app._workflow.measurement_defs:
            app.tool_panel.add_tool_to_list(d["label"], d["object_type"])

        app.tool_panel.set_template_created(True)
        app._exec_btn.state(["!disabled"])
        app._status_text.set(f"项目已加载: {os.path.basename(filepath)} (旧格式)")
        app._notebook.select(0)

    # =====================================================================
    # Recent Projects
    # =====================================================================

    @staticmethod
    def get_recent_projects() -> List[Dict[str, str]]:
        """Return list of recent project entries, newest first."""
        return ProjectManager._read_recent_projects()

    @staticmethod
    def add_recent_project(project_dir: str, name: str) -> None:
        """Add or move a project to the top of the recent list."""
        projects = ProjectManager._read_recent_projects()
        # Remove existing entry for this path
        projects = [p for p in projects if p["path"] != project_dir]
        # Insert at front
        projects.insert(0, {
            "path": project_dir,
            "name": name,
            "last_opened": datetime.now(timezone.utc).isoformat(),
        })
        # Trim
        projects = projects[:MAX_RECENT]
        ProjectManager._write_recent_projects(projects)

    @staticmethod
    def remove_recent_project(project_dir: str) -> None:
        """Remove a project from the recent list."""
        projects = ProjectManager._read_recent_projects()
        projects = [p for p in projects if p["path"] != project_dir]
        ProjectManager._write_recent_projects(projects)

    @staticmethod
    def _read_recent_projects() -> List[Dict[str, str]]:
        try:
            if os.path.exists(ProjectManager.RECENT_FILE):
                with open(ProjectManager.RECENT_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
        except (json.JSONDecodeError, IOError):
            pass
        return []

    @staticmethod
    def _write_recent_projects(projects: List[Dict[str, str]]) -> None:
        os.makedirs(os.path.dirname(ProjectManager.RECENT_FILE), exist_ok=True)
        with open(ProjectManager.RECENT_FILE, "w", encoding="utf-8") as f:
            json.dump(projects, f, indent=2, ensure_ascii=False)
