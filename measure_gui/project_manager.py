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

from measure.multi_target_workflow import MultiTargetWorkflow

if TYPE_CHECKING:
    from .app import MeasureApp

MANIFEST_FILENAME = "project.json"
WORKFLOW_FILENAME = "workflow.npz"
REF_IMAGE_FILENAME = "reference.png"
INSP_IMAGE_FILENAME = "inspection.png"
CALIB_BOARD_FILENAME = "calibration_board.png"
# Model asset manifest written by MultiTargetWorkflow.save().
MODEL_MANIFEST_FILENAME = "manifest.json"

# project.json schema version.
#   4 — measurement config (matching / roi / alignment) is no longer duplicated
#       here: the model asset's manifest.json owns it.  Purely GUI state is
#       grouped under "view".
MANIFEST_VERSION = 4

# Keys older project.json files carry; still read (as a fallback) but never written.
_LEGACY_MEASUREMENT_KEYS = ("matching", "roi", "alignment")
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

        # 3. Copy calibration board image into project dir (portable copy)
        if app._calibration_board_image is not None:
            calib_path = os.path.join(project_dir, CALIB_BOARD_FILENAME)
            cv2.imwrite(calib_path, app._calibration_board_image)

        # 4. Copy images into project dir
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

        # Normalize the recorded inspection image path so the manifest itself
        # is portable. save_project always writes an "inspection.png" copy into
        # the project dir, so if the original path points inside the project
        # dir, record the relative name instead of the absolute path.
        inspection_path = getattr(app, '_inspection_image_path', None)
        if inspection_path and os.path.isfile(inspection_path):
            try:
                if os.path.abspath(inspection_path).startswith(
                        os.path.abspath(project_dir) + os.sep):
                    inspection_path = INSP_IMAGE_FILENAME
            except (OSError, TypeError):
                pass

        # Measurement configuration (ROI geometry, matching parameters and
        # alignment mode) is NOT written here any more.  It belongs to the model
        # asset (manifest.json), saved by MultiTargetWorkflow.save() below, which
        # keeps the two in lockstep by construction.  Duplicating it let the GUI
        # restore a ROI the engine was never taught with.
        #
        # What stays here is state the model asset deliberately does not carry:
        # window/canvas presentation plus the user's ROI *interaction* flag.
        roi_state = _to_json(app._get_roi_state())
        view: Dict[str, Any] = {
            "roi_confirmed": bool(roi_state.get("confirmed", False)),
            "ref_canvas_state": _to_json(app._ref_canvas.get_view_state()),
            "insp_canvas_state": _to_json(app._insp_canvas.get_view_state()),
            "template_view": _to_json(app.template_view.get_state()),
        }

        manifest: Dict[str, Any] = {
            "version": MANIFEST_VERSION,
            "project_name": os.path.basename(project_dir),
            "created_at": created_at,
            "updated_at": now,
            "reference_image_path": getattr(app, '_reference_image_path', None),
            "inspection_image_path": inspection_path,
            "view": view,
            "tool_list_order": _to_json(app.get_tool_list_order()),
            "tool_visibility": _to_json(dict(app.tool_panel._tool_visibility)),
            "gui": _to_json(app.get_gui_state()),
            "calibration": _to_json(app.get_calibration_state()),
        }
        return manifest

    # =====================================================================
    # Load
    # =====================================================================

    @staticmethod
    def load_project(project_dir: str, app: MeasureApp) -> None:
        """Load a complete project from a directory.

        1. Loads the model asset (manifest.json, plus its derived template cache
           when there is one) — this owns the ROI geometry, matching parameters
           and alignment mode
        2. Reads project.json for purely GUI state (views, layout, tool order)
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

        # Load workflow.  ``workflow.npz`` must NOT be required here: it is a
        # derived template cache, and search-free alignment modes (``none``,
        # ``fixed_roi``) legitimately have no template to cache — such a project
        # saves only manifest.json + reference.npy.  Requiring the file made the
        # GUI reject assets the engine can load perfectly, contradicting
        # MultiTargetWorkflow.load()'s own "never required" contract.
        # _load_workflow() decides what is actually needed (manifest vs legacy
        # single-file npz) and raises with an accurate message if neither exists.
        app._workflow = ProjectManager._load_workflow(project_dir, workflow_path)
        ProjectManager._restore_into_app(project_dir, app, manifest, workflow_path)
        # Update recent projects
        ProjectManager.add_recent_project(
            project_dir, os.path.basename(project_dir))


    @staticmethod
    def _restore_into_app(project_dir: str, app: MeasureApp,
                           manifest: Dict[str, Any], workflow_path: str) -> None:
        """Rebuild all engine and UI state from the loaded project.

        Runs with ``app._restoring_project`` set so that repainting and locking
        the ROI cannot fire the interactive confirm callback: that callback
        rebuilds the template from the combobox and would replace the alignment
        strategy just loaded from the model asset.
        """
        app._restoring_project = True
        try:
            app._current_project_dir = project_dir
            app._current_project_path = None
            app._created_at = manifest.get("created_at")
        
            # The model asset is the single source of truth for measurement config.
            model = ProjectManager._read_model_manifest(project_dir)
            view = manifest.get("view") or {}
        
            # Restore reference image
            ref_img = app._workflow.reference_image
            if ref_img is not None:
                app._reference_image = ref_img
                app._ref_canvas.load_image(ref_img)
        
                roi = ProjectManager._roi_from_model(model) or manifest.get("roi", {})
                if roi:
                    app._ref_canvas.set_roi(
                        (roi["center_row"], roi["center_col"]),
                        (roi["height"], roi["width"]),
                        roi["angle_deg"],
                    )
                    # 'confirmed' is GUI interaction state, never model state: it
                    # comes from project.json ("view" now, "roi" in older files).
                    confirmed = view.get("roi_confirmed",
                                         manifest.get("roi", {}).get("confirmed", False))
                    if confirmed:
                        app._ref_canvas.confirm_roi()
        
                    alignment_mode = (ProjectManager._alignment_mode(model)
                                      or roi.get("alignment_mode")
                                      or "single_box")
                    app.tool_panel.set_alignment_mode(alignment_mode)
        
                    # Restore control points if multi-point
                    if alignment_mode == "multi_point":
                        cps = (ProjectManager._control_points(model)
                               or roi.get("control_points") or [])
                        if cps:
                            points = [(cp["row"], cp["col"]) for cp in cps]
                            app._ref_canvas.set_control_points(points)
        
                # Restore ref canvas view state
                ref_state = view.get("ref_canvas_state") or manifest.get(
                    "ref_canvas_state", {})
                if ref_state:
                    app._ref_canvas.set_view_state(ref_state)
        
            # Restore template image + tools in template view
            tmpl = app._workflow.template_image
            if tmpl is not None:
                app.template_view.load_template(tmpl, clear_tools=False)
                tmpl_state = view.get("template_view") or manifest.get(
                    "template_view", {})
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
        
            # Restore matching params in UI widgets — from the model asset, falling
            # back to a legacy project.json block for older projects.
            matching = ProjectManager._matching_from_model(model) or manifest.get(
                "matching", {})
            if matching:
                app.tool_panel.set_matching_params(
                    matching["preprocessor_type"],
                    matching["match_score_threshold"],
                    matching["angle_range_half"],
                    matching["max_matches"],
                    overlap=matching["overlap"],
                    angle_step_deg=matching["angle_step_deg"],
                )
        
            # Restore inspection image if available
            insp_img = ProjectManager._resolve_inspection_image(project_dir, manifest)
            if insp_img is not None:
                app._inspection_image = insp_img
                app._insp_canvas.load_image(insp_img)
                insp_state = view.get("insp_canvas_state") or manifest.get(
                    "insp_canvas_state", {})
                if insp_state:
                    app._insp_canvas.set_view_state(insp_state)
        
            # Restore reference/inspection image path tracking
            app._reference_image_path = manifest.get("reference_image_path")
            app._inspection_image_path = manifest.get("inspection_image_path")
        
            # Restore checkerboard calibration (optional; old projects skip this)
            calibration = manifest.get("calibration")
            if calibration and calibration.get("scale_mm_per_px"):
                app._calibration = dict(calibration)
                app._workflow.physical_scale_mm = float(
                    calibration["scale_mm_per_px"])
                calib_path = os.path.join(project_dir, CALIB_BOARD_FILENAME)
                if os.path.exists(calib_path):
                    board = cv2.imread(calib_path)
                    if board is not None:
                        app._calibration_board_image = board
                app._update_calibration_status()
        
            # Restore GUI layout state (geometry, tab, sash positions)
            gui_state = manifest.get("gui", {})
            if gui_state:
                app.restore_gui_state(gui_state)
        
            # Enable execute button
            app.tool_panel.set_template_created(True)
            # _exec_btn removed with toolbar in app.py, state managed by tool_panel
        
            # Update window title
            project_name = os.path.basename(project_dir)
            app.root.title(f"工业视觉测量系统 — {project_name}")
            app._status_text.set(f"项目已加载: {project_dir}")
        
        finally:
            app._restoring_project = False

    # =====================================================================
    # Model-asset readers (the single source of measurement truth)
    # =====================================================================

    @staticmethod
    def _load_workflow(project_dir: str, workflow_path: str) -> MultiTargetWorkflow:
        """Load the workflow, upgrading a legacy single-file model in place first.

        Projects saved before the manifest format carry only ``workflow.npz``
        with a ``workflow_meta`` blob inside.  Rather than failing, convert them
        to a model asset on the spot and keep the original file alongside as
        ``workflow.npz.legacy`` — so an old project opens, and its ROI/matching
        parameters become available to restore from.

        Args:
            project_dir: Project directory to write the converted asset into.
            workflow_path: Path of the project's ``workflow.npz``.

        Returns:
            The loaded MultiTargetWorkflow.
        """
        manifest_present = os.path.isfile(
            os.path.join(project_dir, MODEL_MANIFEST_FILENAME))
        if manifest_present:
            return MultiTargetWorkflow.load(workflow_path)

        if not ProjectManager._is_legacy_single_file(workflow_path):
            # No model manifest and nothing convertible: this is a broken or
            # half-migrated asset, not an old project.  Say so actionably
            # instead of surfacing a KeyError from inside the legacy reader.
            raise ValueError(
                f"'{project_dir}' has no {MODEL_MANIFEST_FILENAME} and "
                f"'{workflow_path}' is not a readable legacy model. Legacy "
                f"single-file .npz models are converted automatically; run "
                f"scripts/migrate_models_to_manifest.py for standalone models."
            )

        workflow = MultiTargetWorkflow.load_legacy(workflow_path)
        backup = workflow_path + ".legacy"
        if not os.path.exists(backup):
            shutil.copy2(workflow_path, backup)
        # Re-save through the engine so the directory gains manifest.json +
        # reference.npy; load again to prove the conversion is usable.
        workflow.save(project_dir)
        return MultiTargetWorkflow.load(workflow_path)

    @staticmethod
    def _is_legacy_single_file(path: str) -> bool:
        """True when *path* is a pre-manifest single-file workflow archive."""
        try:
            with np.load(path, allow_pickle=True) as data:
                return "workflow_meta" in data
        except (OSError, ValueError):
            return False

    @staticmethod
    def _read_model_manifest(project_dir: str) -> Dict[str, Any]:
        """Read the model asset's manifest.json next to workflow.npz.

        Returns:
            The model manifest, or ``{}`` when it is absent (a project saved by
            an older build).
        """
        path = os.path.join(project_dir, MODEL_MANIFEST_FILENAME)
        if not os.path.isfile(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _roi_from_model(model: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Translate the model manifest's ROI into the canvas' ROI state form.

        Returns ``None`` when the model carries no ROI, so callers can fall back
        to a legacy project.json.
        """
        alignment = model.get("alignment") or {}
        center = alignment.get("box_center")
        size = alignment.get("box_size")
        if not center or not size:
            return None
        roi = {
            "center_row": float(center[0]),
            "center_col": float(center[1]),
            "height": float(size[0]),
            "width": float(size[1]),
            "angle_deg": float(alignment.get("box_angle_deg", 0.0)),
            "alignment_mode": alignment.get("mode", "single_box"),
        }
        control_points = ProjectManager._control_points(model)
        if control_points:
            roi["control_points"] = control_points
        return roi

    @staticmethod
    def _alignment_mode(model: Dict[str, Any]) -> Optional[str]:
        """Alignment mode recorded in the model asset."""
        mode = (model.get("alignment") or {}).get("mode")
        return mode or None

    @staticmethod
    def _control_points(model: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Control points stored in the model asset's ROI state."""
        roi_state = (model.get("alignment") or {}).get("roi_state") or {}
        points = roi_state.get("control_points") or []
        return list(points)

    @staticmethod
    def _matching_from_model(model: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Translate the model manifest's matching block into widget values.

        The model stores ``angle_range`` as a [min, max] pair while the tool
        panel works with a half-range, so the conversion happens here rather
        than leaving two incompatible spellings of the same setting around.
        """
        matching = model.get("matching") or {}
        if not matching:
            return None
        angle_range = matching.get("angle_range") or (-30.0, 30.0)
        preprocessor = (model.get("alignment") or {}).get("preprocessor") or {}
        return {
            "preprocessor_type": preprocessor.get("type", "raw"),
            "match_score_threshold": float(matching.get("match_score_threshold", 0.5)),
            "angle_range_half": max(abs(float(v)) for v in angle_range),
            "angle_step_deg": float(matching.get("angle_step", 1.0)),
            "max_matches": int(matching.get("max_matches", 0)),
            "overlap": float(matching.get("overlap", 0.3)),
        }

    @staticmethod
    def _resolve_inspection_image(
        project_dir: str,
        manifest: Dict[str, Any],
    ) -> Optional[np.ndarray]:
        """Resolve the inspection image to restore for a project.

        Prefers the project-local "inspection.png" copy saved by
        ``save_project`` so projects stay portable across machines. Falls back
        to ``manifest["inspection_image_path"]`` when the local copy is missing
        (e.g. legacy/partial projects). Returns None (does not raise) when no
        recoverable inspection image exists.
        """
        candidates = [os.path.join(project_dir, INSP_IMAGE_FILENAME)]
        insp_path = manifest.get("inspection_image_path")
        if isinstance(insp_path, str) and not os.path.isabs(insp_path):
            candidates.append(os.path.join(project_dir, insp_path))
        elif isinstance(insp_path, str):
            candidates.append(insp_path)

        for candidate in candidates:
            if os.path.exists(candidate):
                img = cv2.imread(candidate, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    return img
        return None

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
        # _exec_btn removed with toolbar in app.py, state managed by tool_panel
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
