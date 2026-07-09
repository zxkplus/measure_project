"""
MeasureProject — core SDK for measurement modeling and execution.

Two-phase lifecycle:

  Phase 1 — Modeling (Teach):
      1. create / load project
      2. load reference image
      3. set template (ROI)
      4. add / test / tune measurement objects individually
      5. compose relationships
      6. save project

  Phase 2 — Measurement (Run):
      1. load saved project
      2. measure(inspection_image) -> full results

All public methods return plain dicts (JSON-serializable).
"""

from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from measure_api.logger import get_logger
from measure_api.config import Config
from measure_api.quality import compute_quality
from measure_api.schemas import COMPOSED_DEPS, COMPOSED_TYPES, PRIMITIVE_TYPES
from measure_api.visualizer import generate_overview_visual, generate_visual

logger = get_logger("project")

# ----------------------------------------------------------------------------
# Re-use the existing object factories from the GUI module
# ----------------------------------------------------------------------------

from measure_gui.multi_target import _OBJECT_FACTORIES  # noqa: E402
from measure_gui.multi_target import MultiTargetWorkflow as _MTW  # noqa: E402
from measure.measure_workflow import MeasurementWorkflow  # noqa: E402


# ----------------------------------------------------------------------------
# State machine constants
# ----------------------------------------------------------------------------

PHASE_CREATED = "created"
PHASE_REF_LOADED = "reference_loaded"
PHASE_TEMPLATE_READY = "template_ready"
PHASE_HAS_MEASUREMENTS = "has_measurements"
PHASE_HAS_COMPOSED = "has_composed"
PHASE_MEASURED = "measured"


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _geometric_result_to_dict(result) -> Dict[str, Any]:
    """Convert a GeometricResult (or subclass) to a plain dict."""
    if result is None:
        return {"valid": False}
    d = {
        "type": getattr(result, "type", "unknown"),
        "label": getattr(result, "label", ""),
        "valid": bool(getattr(result, "valid", False)),
    }
    # PointResult: row, col
    if hasattr(result, "row") and hasattr(result, "col"):
        d["row"] = float(getattr(result, "row", 0))
        d["col"] = float(getattr(result, "col", 0))
    # LineResult: a, b, c, start, end
    if hasattr(result, "a"):
        d["a"] = float(getattr(result, "a", 0))
        d["b"] = float(getattr(result, "b", 0))
        d["c"] = float(getattr(result, "c", 0))
    if hasattr(result, "start_row"):
        d["start_row"] = float(getattr(result, "start_row", 0))
        d["start_col"] = float(getattr(result, "start_col", 0))
        d["end_row"] = float(getattr(result, "end_row", 0))
        d["end_col"] = float(getattr(result, "end_col", 0))
    # CircleResult
    if hasattr(result, "center_row"):
        d["center_row"] = float(getattr(result, "center_row", 0))
        d["center_col"] = float(getattr(result, "center_col", 0))
    if hasattr(result, "radius"):
        d["radius"] = float(getattr(result, "radius", 0))
    # DistanceResult / AngleResult
    if hasattr(result, "value"):
        d["value"] = float(getattr(result, "value", 0))
    if hasattr(result, "value_deg"):
        d["value_deg"] = float(getattr(result, "value_deg", 0))
    # Meta
    meta = getattr(result, "meta", {})
    if meta:
        d["meta"] = dict(meta)
    return d


def _phase_label(phase: str) -> str:
    labels = {
        PHASE_CREATED: "created",
        PHASE_REF_LOADED: "reference_loaded",
        PHASE_TEMPLATE_READY: "template_ready",
        PHASE_HAS_MEASUREMENTS: "has_measurements",
        PHASE_HAS_COMPOSED: "has_composed",
        PHASE_MEASURED: "measured",
    }
    return labels.get(phase, phase)


# ============================================================================
# MeasureProject
# ============================================================================


class MeasureProject:
    """
    Core SDK for measurement modeling and execution.

    Args:
        project_dir: Directory path for the project (will be created if it
                     doesn't exist on ``save()``).
    """

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(self, project_dir: str) -> None:
        self.project_dir = str(project_dir)
        self._phase: str = PHASE_CREATED

        # The underlying multi-target workflow
        self._workflow: Optional[_MTW] = None

        # Reference image
        self._reference_image: Optional[np.ndarray] = None

        # Measurement definitions (ordered list of dicts with keys:
        #   object_type, label, params)
        self._measurement_defs: List[Dict[str, Any]] = []

        # Composed measurement definitions (ordered list of dicts with keys:
        #   composed_type, label, deps)
        self._composed_defs: List[Dict[str, Any]] = []

        # Last measured results cache: list of target dicts
        self._results: List[Dict[str, Any]] = []

        logger.info("Project created: %s", project_dir)

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    @property
    def phase(self) -> str:
        """Current phase label (human-readable)."""
        return _phase_label(self._phase)

    @property
    def has_template(self) -> bool:
        return self._workflow is not None and self._workflow.template_image is not None

    @property
    def template_image(self) -> Optional[np.ndarray]:
        if self._workflow is None:
            return None
        return self._workflow.template_image

    @property
    def template_shape(self) -> Optional[Tuple[int, int]]:
        img = self.template_image
        if img is None:
            return None
        return (int(img.shape[0]), int(img.shape[1]))

    def _check_phase(self, *required: str) -> None:
        """Raise RuntimeError if phase is not one of the required values."""
        if self._phase not in required:
            raise RuntimeError(
                f"Operation not allowed in phase '{self.phase}'. "
                f"Requires one of: {[_phase_label(p) for p in required]}"
            )

    def _update_phase(self) -> None:
        """Recompute phase based on current state."""
        if self._composed_defs:
            self._phase = PHASE_HAS_COMPOSED
        elif self._measurement_defs:
            self._phase = PHASE_HAS_MEASUREMENTS
        elif self.has_template:
            self._phase = PHASE_TEMPLATE_READY
        elif self._reference_image is not None:
            self._phase = PHASE_REF_LOADED
        else:
            self._phase = PHASE_CREATED

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load_reference(self, image_path: str) -> Dict[str, Any]:
        """
        Load a reference (teaching) image.

        Args:
            image_path: Absolute path or path relative to ``project_dir``.

        Returns:
            Dict with ``width``, ``height``, ``path``.
        """
        # Resolve relative paths
        path = self._resolve_path(image_path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Reference image not found: {path}")

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to read image: {path}")

        self._reference_image = img
        self._workflow = _MTW()
        self._phase = PHASE_REF_LOADED

        info = {"width": img.shape[1], "height": img.shape[0], "path": path}
        logger.info("Reference loaded: %s", info)
        return info

    def set_template(
        self,
        center: Tuple[float, float],
        size: Tuple[float, float],
        angle_deg: float,
        preprocessor: str = "raw",
        match_score_threshold: float = 0.5,
        angle_range_deg: float = 30.0,
        max_matches: int = 0,
        angle_step: float = 1.0,
        coarse_angle_step: float = 5.0,
        coarse_fine: bool = True,
        overlap: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Define the ROI template.

        Args:
            center: ``(row, col)`` of the rotated bounding box.
            size: ``(height, width)`` of the bounding box.
            angle_deg: Rotation angle in degrees.
            preprocessor: Preprocessor type (``"raw"``, ``"canny"``, etc.).
            match_score_threshold: Minimum NCC score.
            angle_range_deg: Half-range ±degrees for angle search.
            max_matches: 0 = unlimited, 1 = single target, N = max N targets.

        Returns:
            Dict with ``template_shape`` info.
        """
        self._check_phase(PHASE_REF_LOADED, PHASE_TEMPLATE_READY,
                          PHASE_HAS_MEASUREMENTS, PHASE_HAS_COMPOSED)

        if self._workflow is None:
            raise RuntimeError("Workflow not initialized. Call load_reference() first.")

        from measure.measure_template import (
            CannyPreprocessor,
            CLAHEPreprocessor,
            RawPreprocessor,
            SobelPreprocessor,
            ThresholdPreprocessor,
        )

        proc_map = {
            "raw": RawPreprocessor(),
            "canny": CannyPreprocessor(50.0, 150.0),
            "sobel": SobelPreprocessor(3),
            "clahe": CLAHEPreprocessor(2.0),
            "threshold": ThresholdPreprocessor(128.0),
        }
        preprocessor_obj = proc_map.get(preprocessor, RawPreprocessor())

        # Apply global matching config as fallback for default parameters
        _cfg_match = Config.load().get("matching", {}) or {}
        if abs(angle_range_deg - 30.0) < 1e-9 and "angle_range_deg" in _cfg_match:
            angle_range_deg = float(_cfg_match["angle_range_deg"])
        if abs(angle_step - 1.0) < 1e-9 and "angle_step" in _cfg_match:
            angle_step = float(_cfg_match["angle_step"])
        if abs(coarse_angle_step - 5.0) < 1e-9 and "coarse_angle_step" in _cfg_match:
            coarse_angle_step = float(_cfg_match["coarse_angle_step"])
        if coarse_fine is True and "coarse_fine" in _cfg_match:
            coarse_fine = bool(_cfg_match["coarse_fine"])
        if abs(overlap - 0.3) < 1e-9 and "overlap" in _cfg_match:
            overlap = float(_cfg_match["overlap"])
        self._workflow.teach_template(
            self._reference_image,
            center=tuple(center),
            size=tuple(size),
            angle_deg=float(angle_deg),
            preprocessor=preprocessor_obj,
            match_score_threshold=float(match_score_threshold),
            angle_range=(-float(angle_range_deg), float(angle_range_deg)),
            angle_step=float(angle_step),
            coarse_angle_step=float(coarse_angle_step),
            coarse_fine=bool(coarse_fine),
            overlap=float(overlap),
            max_matches=int(max_matches),
        )

        self._phase = PHASE_TEMPLATE_READY

        info = {"template_shape": list(self.template_shape)}
        logger.info("Template set: center=%s size=%s angle=%.1f preproc=%s",
                    center, size, angle_deg, preprocessor)
        return info

    # ------------------------------------------------------------------
    # Single measurement CRUD (with immediate test feedback)
    # ------------------------------------------------------------------

    def add_measurement(
        self, object_type: str, label: str, params: Dict[str, Any],
        include_visual: bool = False,
    ) -> Dict[str, Any]:
        """
        Add a measurement tool and run an immediate test on the template.

        Args:
            object_type: Type string (``"FitCircle"``, ``"FitLine"``, etc.).
            label: Unique label for this measurement.
            params: Parameter dict (depends on object_type).
            include_visual: Whether to include a base64 visual snapshot.

        Returns:
            Dict with ``label``, ``valid``, ``result``, ``quality``,
            and optionally ``visual_b64``.
        """
        self._check_phase(PHASE_TEMPLATE_READY, PHASE_HAS_MEASUREMENTS,
                          PHASE_HAS_COMPOSED)

        if object_type not in PRIMITIVE_TYPES:
            raise ValueError(
                f"Unknown primitive type '{object_type}'. "
                f"Supported: {sorted(PRIMITIVE_TYPES)}"
            )

        # Check duplicate label
        for d in self._measurement_defs:
            if d["label"] == label:
                raise ValueError(f"Duplicate measurement label: '{label}'")
        for d in self._composed_defs:
            if d["label"] == label:
                raise ValueError(f"Label '{label}' already used by a composed measurement")

        # Test on template first
        test_result = self._test_on_template(object_type, label, params,
                                              include_visual=include_visual)

        # Add to defs
        self._measurement_defs.append({
            "object_type": object_type,
            "label": label,
            "params": copy.deepcopy(params),
        })

        # Sync to the underlying workflow
        self._workflow.add_measurement(object_type, label, **copy.deepcopy(params))

        self._update_phase()
        logger.info("add_measurement: %s type=%s valid=%s quality=%s",
                    label, object_type, test_result.get("valid"),
                    test_result.get("quality", {}))

        return test_result

    def update_measurement(
        self, label: str, params: Dict[str, Any],
        include_visual: bool = False,
    ) -> Dict[str, Any]:
        """
        Update a measurement tool's parameters and re-test on the template.

        Args:
            label: Label of the measurement to update.
            params: Updated parameter dict (partial update — merged with existing).
            include_visual: Whether to include a base64 visual snapshot.

        Returns:
            Dict with feedback from the re-test.
        """
        self._check_phase(PHASE_HAS_MEASUREMENTS, PHASE_HAS_COMPOSED)

        # Find existing def
        def_idx = None
        for i, d in enumerate(self._measurement_defs):
            if d["label"] == label:
                def_idx = i
                break
        if def_idx is None:
            raise ValueError(f"Measurement '{label}' not found")

        defn = self._measurement_defs[def_idx]
        object_type = defn["object_type"]

        # Merge params
        merged = copy.deepcopy(defn["params"])
        merged.update(params)

        # Test
        test_result = self._test_on_template(object_type, label, merged,
                                              include_visual=include_visual)

        # Update def
        defn["params"] = merged

        # Sync to workflow (remove + re-add)
        self._workflow.remove_measurement(label)
        self._workflow.add_measurement(object_type, label, **copy.deepcopy(merged))

        logger.info("update_measurement: %s valid=%s", label, test_result.get("valid"))
        return test_result

    def remove_measurement(self, label: str) -> Dict[str, Any]:
        """
        Remove a measurement tool.  Cascades to any composed measurements
        that depend on it (they are marked invalid or removed).

        Returns:
            Dict with ``status`` and optional ``cascaded`` list.
        """
        self._check_phase(PHASE_HAS_MEASUREMENTS, PHASE_HAS_COMPOSED)

        # Find the def
        def_idx = None
        for i, d in enumerate(self._measurement_defs):
            if d["label"] == label:
                def_idx = i
                break
        if def_idx is None:
            raise ValueError(f"Measurement '{label}' not found")

        # Cascade — find composed measurements that depend on this label
        cascaded = self._resolve_cascade_delete(label)

        # Remove from defs
        removed = self._measurement_defs.pop(def_idx)

        # Remove from workflow
        self._workflow.remove_measurement(label)

        # If no more primitives, drop back to template state
        self._update_phase()

        response = {"status": "deleted", "label": label}
        if cascaded:
            response["cascaded"] = cascaded
            if not self._composed_defs:
                self._update_phase()

        logger.info("remove_measurement: %s cascaded=%d", label, len(cascaded))
        return response

    def get_measurement(self, label: str) -> Dict[str, Any]:
        """
        Get a single measurement's definition and its last test result.

        Returns:
            Dict with ``label``, ``object_type``, ``params``, and
            optionally ``last_result`` if available.
        """
        for d in self._measurement_defs:
            if d["label"] == label:
                return {
                    "label": label,
                    "object_type": d["object_type"],
                    "params": copy.deepcopy(d["params"]),
                }
        raise ValueError(f"Measurement '{label}' not found")

    def list_measurements(self) -> Dict[str, Any]:
        """
        List all measurements (primitive and composed) with their current status.

        Returns:
            Dict with keys ``"measurements"`` and ``"composed"``.
        """
        return {
            "measurements": [
                {
                    "label": d["label"],
                    "object_type": d["object_type"],
                    "params": copy.deepcopy(d["params"]),
                }
                for d in self._measurement_defs
            ],
            "composed": [
                {
                    "label": d["label"],
                    "composed_type": d["composed_type"],
                    "dependencies": copy.deepcopy(d["deps"]),
                }
                for d in self._composed_defs
            ],
        }

    def test_measurement(
        self, object_type: str, label: str, params: Dict[str, Any],
        include_visual: bool = False,
    ) -> Dict[str, Any]:
        """
        Test a measurement on the template image **without** saving it.

        Use for parameter exploration before committing.

        Args:
            object_type: Type string.
            label: Temporary label for identification in the response.
            params: Parameter dict.
            include_visual: Whether to include base64 visual snapshot.

        Returns:
            Dict with feedback.
        """
        self._check_phase(PHASE_TEMPLATE_READY, PHASE_HAS_MEASUREMENTS,
                          PHASE_HAS_COMPOSED)

        return self._test_on_template(object_type, label, params,
                                       include_visual=include_visual)

    # ------------------------------------------------------------------
    # Composed measurements
    # ------------------------------------------------------------------

    def add_composed(
        self, composed_type: str, label: str, deps: Dict[str, str],
        include_visual: bool = False,
    ) -> Dict[str, Any]:
        """
        Add a composed (derived) measurement.

        Composed measurements derive their value from other measurements.
        E.g. ``"TwoPointsDistance"`` computes distance between two point results.

        Args:
            composed_type: Type string (``"TwoPointsDistance"``, etc.).
            label: Unique label.
            deps: Dependency mapping, e.g.
                  ``{"point_a_label": "pt1", "point_b_label": "pt2"}``.
            include_visual: Whether to include base64 visual snapshot.

        Returns:
            Dict with feedback (tested on template).
        """
        self._check_phase(PHASE_HAS_MEASUREMENTS, PHASE_HAS_COMPOSED)

        if composed_type not in COMPOSED_TYPES:
            raise ValueError(
                f"Unknown composed type '{composed_type}'. "
                f"Supported: {sorted(COMPOSED_TYPES)}"
            )

        # Validate dependencies exist
        expected_deps = COMPOSED_DEPS.get(composed_type, [])
        for key in expected_deps:
            dep_label = deps.get(key)
            if dep_label is None:
                raise ValueError(f"Missing dependency '{key}' for {composed_type}")
            found = any(d["label"] == dep_label for d in self._measurement_defs)
            if not found:
                raise ValueError(
                    f"Dependency '{dep_label}' (key={key}) not found "
                    f"among measurements"
                )

        # Check duplicate label
        for d in self._composed_defs:
            if d["label"] == label:
                raise ValueError(f"Duplicate composed label: '{label}'")
        for d in self._measurement_defs:
            if d["label"] == label:
                raise ValueError(f"Label '{label}' already used by a primitive measurement")

        # Test
        test_result = self._test_composed_on_template(
            composed_type, label, deps, include_visual=include_visual,
        )

        # Add to defs
        self._composed_defs.append({
            "composed_type": composed_type,
            "label": label,
            "deps": copy.deepcopy(deps),
        })

        self._update_phase()
        logger.info("add_composed: %s type=%s deps=%s valid=%s",
                    label, composed_type, deps, test_result.get("valid"))
        return test_result

    def remove_composed(self, label: str) -> Dict[str, Any]:
        """
        Remove a composed measurement.

        Returns:
            Dict with ``status``.
        """
        self._check_phase(PHASE_HAS_COMPOSED)

        for i, d in enumerate(self._composed_defs):
            if d["label"] == label:
                self._composed_defs.pop(i)
                self._update_phase()
                logger.info("remove_composed: %s", label)
                return {"status": "deleted", "label": label}

        raise ValueError(f"Composed measurement '{label}' not found")

    def list_composed(self) -> Dict[str, Any]:
        """
        List composed measurements with their current dependency status.

        Returns:
            Dict with key ``"composed"``.
        """
        return {
            "composed": [
                {
                    "label": d["label"],
                    "composed_type": d["composed_type"],
                    "dependencies": copy.deepcopy(d["deps"]),
                }
                for d in self._composed_defs
            ],
        }

    # ------------------------------------------------------------------
    # DAG (dependency graph)
    # ------------------------------------------------------------------

    def get_dag(self, format: str = "json") -> Dict[str, Any]:
        """
        Return the dependency graph of all measurements.

        Args:
            format: ``"json"`` (default) for structured data,
                    ``"text"`` for a human-readable summary.

        Returns:
            Dict with ``nodes``, ``edges``, ``execution_order``,
            and ``is_valid``.
        """
        nodes = []
        edges = []

        # Primitive nodes
        for d in self._measurement_defs:
            nodes.append({
                "label": d["label"],
                "type": d["object_type"],
                "category": "primitive",
            })

        # Composed nodes + edges
        for d in self._composed_defs:
            nodes.append({
                "label": d["label"],
                "type": d["composed_type"],
                "category": "composed",
            })
            expected_deps = COMPOSED_DEPS.get(d["composed_type"], [])
            for key in expected_deps:
                dep_label = d["deps"].get(key)
                if dep_label:
                    edges.append({
                        "from": dep_label,
                        "to": d["label"],
                        "role": key,
                    })

        # Compute execution order and validity via topological sort
        exec_order, is_valid, errors = self._topological_sort()

        result: Dict[str, Any] = {
            "nodes": nodes,
            "edges": edges,
            "execution_order": exec_order,
            "is_valid": is_valid,
        }
        if errors:
            result["errors"] = errors

        if format == "text":
            result["text"] = self._dag_to_text(nodes, edges, exec_order, is_valid, errors)

        return result

    def _dag_to_text(
        self, nodes, edges, exec_order, is_valid, errors,
    ) -> str:
        """Generate a human-readable DAG summary."""
        lines = ["DAG Summary:"]
        for i, label in enumerate(exec_order):
            node = next((n for n in nodes if n["label"] == label), None)
            if node is None:
                continue
            prefix = f"  {i+1}) "
            if node["category"] == "primitive":
                deps = [e for e in edges if e["from"] == label]
                used_by = [e["to"] for e in edges if e["from"] == label]
                dep_str = f"  → 用于: {', '.join(used_by)}" if used_by else ""
                lines.append(f"{prefix}[Primitive] {label} ({node['type']}){dep_str}")
            else:
                input_edges = [e for e in edges if e["to"] == label]
                dep_detail = ", ".join(f"{e['role']}={e['from']}" for e in input_edges)
                lines.append(f"{prefix}[Composed]  {label} ({node['type']})  deps: {dep_detail}")
        lines.append("  ────")
        if is_valid:
            lines.append("  DAG 完整可解析 ✓")
        else:
            lines.append(f"  DAG 不可解析: {'; '.join(errors)}")
        return "\n".join(lines)

    def _topological_sort(self) -> Tuple[List[str], bool, List[str]]:
        """
        Kahn's algorithm for topological sort.

        Returns:
            ``(execution_order, is_valid, errors)``.
        """
        # Build adjacency
        all_labels: List[str] = (
            [d["label"] for d in self._measurement_defs] +
            [d["label"] for d in self._composed_defs]
        )

        in_degree: Dict[str, int] = {lbl: 0 for lbl in all_labels}
        adjacency: Dict[str, List[str]] = {lbl: [] for lbl in all_labels}

        errors: List[str] = []

        for d in self._composed_defs:
            label = d["label"]
            expected_deps = COMPOSED_DEPS.get(d["composed_type"], [])
            for key in expected_deps:
                dep_label = d["deps"].get(key)
                if dep_label is None:
                    errors.append(f"'{label}': missing dependency '{key}'")
                    continue
                if dep_label not in all_labels:
                    errors.append(f"'{label}': dependency '{dep_label}' not found")
                    continue
                adjacency.setdefault(dep_label, []).append(label)
                in_degree[label] = in_degree.get(label, 0) + 1

        # Kahn
        queue = [lbl for lbl, d in in_degree.items() if d == 0]
        order = []
        while queue:
            current = queue.pop(0)
            order.append(current)
            for neighbor in adjacency.get(current, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(all_labels):
            missing = set(all_labels) - set(order)
            errors.append(f"Cyclic dependency involving: {missing}")
            return order, False, errors

        return order, True, errors

    def _resolve_cascade_delete(self, label: str) -> List[Dict[str, Any]]:
        """
        Find composed measurements that depend on ``label`` and
        mark/remove them.
        """
        cascaded = []
        to_remove = []
        for i, d in enumerate(self._composed_defs):
            expected_deps = COMPOSED_DEPS.get(d["composed_type"], [])
            for key in expected_deps:
                if d["deps"].get(key) == label:
                    cascaded.append({
                        "label": d["label"],
                        "reason": f"depends on {label} via {key}",
                        "action": "removed",
                    })
                    to_remove.append(i)
                    break

        # Remove in reverse order
        for i in reversed(to_remove):
            self._composed_defs.pop(i)

        return cascaded

    # ------------------------------------------------------------------
    # Internal: test on template
    # ------------------------------------------------------------------

    def _build_measurement_object(self, object_type: str, label: str, params: Dict[str, Any]):
        """Build a MeasureObject from type + params using the existing factory."""
        factory_tuple = _OBJECT_FACTORIES.get(object_type)
        if factory_tuple is None:
            raise ValueError(
                f"Unknown object type '{object_type}'. "
                f"Available: {sorted(_OBJECT_FACTORIES.keys())}"
            )
        factory_fn, param_keys = factory_tuple
        return factory_fn(label=label, **params)

    def _test_on_template(
        self, object_type: str, label: str, params: Dict[str, Any],
        include_visual: bool = False,
    ) -> Dict[str, Any]:
        """
        Run a single measurement on the straightened template image.

        This is the core of the "test during teaching" workflow — it
        executes the measurement in template-local coordinates without
        requiring an inspection image.
        """
        if self._workflow is None or not self.has_template:
            raise RuntimeError("Template not defined. Call set_template() first.")

        template_img = self._workflow.template_image
        if template_img is None:
            raise RuntimeError("Template image is None.")

        # Build a temporary single-object workflow
        temp_wf = MeasurementWorkflow()
        obj = self._build_measurement_object(object_type, label, params)
        temp_wf.add(obj)

        # Measure on the template image
        try:
            start = time.perf_counter()
            temp_wf.measure(template_img)
            elapsed_ms = (time.perf_counter() - start) * 1000
        except Exception as e:
            # Measurement failed entirely
            return {
                "label": label,
                "valid": False,
                "result": {"valid": False, "error": str(e)},
                "quality": {},
                "elapsed_ms": round((time.perf_counter() - time.monotonic()) * 1000, 1),
            }

        # Get result
        result = obj.result
        result_dict = _geometric_result_to_dict(result)

        # Compute quality
        quality = compute_quality(object_type, result)

        # Visual
        response: Dict[str, Any] = {
            "label": label,
            "object_type": object_type,
            "valid": bool(result and result.valid),
            "result": result_dict,
            "quality": quality,
            "elapsed_ms": round(elapsed_ms, 1),
        }

        if include_visual:
            visual_b64 = generate_visual(template_img, object_type, params, result, quality)
            if visual_b64:
                response["visual_b64"] = visual_b64

        return response

    def _test_composed_on_template(
        self, composed_type: str, label: str, deps: Dict[str, str],
        include_visual: bool = False,
    ) -> Dict[str, Any]:
        """
        Test a composed measurement + all its dependencies on the template image.
        """
        if self._workflow is None or not self.has_template:
            raise RuntimeError("Template not defined.")

        template_img = self._workflow.template_image
        if template_img is None:
            raise RuntimeError("Template image is None.")

        temp_wf = MeasurementWorkflow()

        # Add all dependencies first
        dep_labels = set(deps.values())
        dep_map = {}  # dep_label -> actual label in temp_wf

        # Find the defs for the dependencies
        for dep_label in dep_labels:
            defn = next(
                (d for d in self._measurement_defs if d["label"] == dep_label),
                None,
            )
            if defn is None:
                return {
                    "label": label,
                    "valid": False,
                    "result": {"valid": False},
                    "error": f"Dependency '{dep_label}' not found",
                }
            obj = self._build_measurement_object(
                defn["object_type"], defn["label"], defn["params"],
            )
            temp_wf.add(obj)

        # Add the composed object
        factory_tuple = _OBJECT_FACTORIES.get(composed_type)
        if factory_tuple is None:
            raise ValueError(f"Unknown composed type '{composed_type}'")
        factory_fn, _ = factory_tuple
        composed_obj = factory_fn(label=label, **deps)
        temp_wf.add(composed_obj)

        # Measure
        try:
            start = time.perf_counter()
            temp_wf.measure(template_img)
            elapsed_ms = (time.perf_counter() - start) * 1000
        except Exception as e:
            return {
                "label": label,
                "valid": False,
                "result": {"valid": False, "error": str(e)},
                "quality": {},
                "elapsed_ms": round((time.perf_counter() - time.monotonic()) * 1000, 1),
            }

        result = composed_obj.result
        result_dict = _geometric_result_to_dict(result)
        quality = compute_quality(composed_type, result)

        response: Dict[str, Any] = {
            "label": label,
            "composed_type": composed_type,
            "valid": bool(result and result.valid),
            "result": result_dict,
            "quality": quality,
            "dependencies": dict(deps),
            "elapsed_ms": round(elapsed_ms, 1),
        }

        if include_visual:
            visual_b64 = generate_visual(template_img, composed_type, deps, result, quality)
            if visual_b64:
                response["visual_b64"] = visual_b64

        return response

    # ------------------------------------------------------------------
    # DAG validation
    # ------------------------------------------------------------------

    def _validate_dag(self) -> bool:
        """Check if the composed measurement DAG is complete and has no errors."""
        _, is_valid, errors = self._topological_sort()
        return is_valid and len(errors) == 0

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self) -> Dict[str, Any]:
        """
        Save the current project to ``project_dir/``.

        Writes:
          - ``config.json``: measurement defs, composed defs, params.
          - ``template.npz``: template pixels (via underlying workflow).
          - ``reference.png``: reference image.

        Returns:
            Dict listing saved files.
        """
        os.makedirs(self.project_dir, exist_ok=True)

        # 1. Save workflow (binary data)
        workflow_path = os.path.join(self.project_dir, "template.npz")
        if self._workflow is not None:
            try:
                self._workflow.save(workflow_path)
            except Exception as e:
                # Workflow may not have a template yet; that's OK
                logger.warning("Workflow save skipped: %s", e)

        # 2. Save config.json
        config = {
            "version": 2,
            "project_name": os.path.basename(self.project_dir),
            "phase": self.phase,
            "measurement_defs": copy.deepcopy(self._measurement_defs),
            "composed_defs": [
                {
                    "composed_type": d["composed_type"],
                    "label": d["label"],
                    "deps": copy.deepcopy(d["deps"]),
                }
                for d in self._composed_defs
            ],
        }

        config_path = os.path.join(self.project_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        # 3. Save reference image
        if self._reference_image is not None:
            ref_path = os.path.join(self.project_dir, "reference.png")
            cv2.imwrite(ref_path, self._reference_image)

        saved_files = [config_path, workflow_path]
        logger.info("Project saved: %s (files: %s)", self.project_dir, saved_files)
        return {"saved_to": self.project_dir, "files": saved_files}

    def load(self) -> Dict[str, Any]:
        """
        Load project state from ``project_dir/``.

        Reads:
          - ``config.json``: measurement and composed defs.
          - ``template.npz``: template (via the underlying workflow).
          - ``reference.png``: reference image.

        Returns:
            Dict with loaded state summary.
        """
        config_path = os.path.join(self.project_dir, "config.json")
        workflow_path = os.path.join(self.project_dir, "template.npz")
        ref_path = os.path.join(self.project_dir, "reference.png")

        # Load workflow
        if os.path.isfile(workflow_path):
            self._workflow = _MTW.load(workflow_path)

        # Load config
        if os.path.isfile(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            self._measurement_defs = config.get("measurement_defs", [])
            self._composed_defs = config.get("composed_defs", [])

        # Load reference image
        if os.path.isfile(ref_path):
            img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                self._reference_image = img
                if self._workflow is None:
                    self._workflow = _MTW()

        self._update_phase()

        # Re-sync measurement defs to workflow
        if self._workflow is not None:
            for d in self._measurement_defs:
                try:
                    self._workflow.add_measurement(
                        d["object_type"], d["label"], **copy.deepcopy(d["params"]),
                    )
                except Exception as e:
                    logger.warning("Failed to re-add measurement '%s': %s", d["label"], e)

        info = {
            "phase": self.phase,
            "has_template": self.has_template,
            "template_shape": self.template_shape,
            "num_measurements": len(self._measurement_defs),
            "num_composed": len(self._composed_defs),
        }
        logger.info("Project loaded: %s (%s)", self.project_dir, info)
        return info

    # ------------------------------------------------------------------
    # Measurement (full pipeline)
    # ------------------------------------------------------------------

    def measure(
        self, inspection_image: str,
        include_visual: bool = False,
    ) -> Dict[str, Any]:
        """
        Run the full measurement pipeline on an inspection image.

        Steps:
          1. Multi-target detection → targets.
          2. For each target: rectify sub-image → run all measurements.
          3. Map results back to original image coordinates.

        Args:
            inspection_image: Path to the inspection image.
            include_visual: Whether to include an overview visual.

        Returns:
            Dict with ``status``, ``elapsed_ms``, ``num_targets``,
            ``targets``, and optionally ``visual_b64``.
        """
        self._check_phase(PHASE_TEMPLATE_READY, PHASE_HAS_MEASUREMENTS,
                          PHASE_HAS_COMPOSED, PHASE_MEASURED)

        if self._workflow is None:
            raise RuntimeError("Workflow not initialized.")

        path = self._resolve_path(inspection_image)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Inspection image not found: {path}")

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to read inspection image: {path}")
        t_read = time.perf_counter()

        t0 = time.perf_counter()

        try:
            raw_results = self._workflow.measure(img)
        except Exception as e:
            logger.error("measure() failed: %s", e)
            return {
                "status": "error",
                "error": str(e),
                "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
                "num_targets": 0,
                "targets": [],
            }

        t_after_workflow = time.perf_counter()
        elapsed_ms = (t_after_workflow - t0) * 1000
        elapsed_read_ms = (t0 - t_read) * 1000
        logger.info("[TIMING] measure: read=%+.1fms  workflow=%+.1fms",
                     elapsed_read_ms, elapsed_ms)

        # Convert TargetResult dataclass objects to plain dicts
        # Note: tr.measurements values are already plain dicts from
        # _map_result_to_original, so we pass them through directly.
        targets = []
        for tr in raw_results:
            targets.append({
                "target_id": int(getattr(tr, "id", 0)),
                "score": float(getattr(tr, "score", 0)),
                "row": float(getattr(tr, "center_row", getattr(tr, "row", 0))),
                "col": float(getattr(tr, "center_col", getattr(tr, "col", 0))),
                "rotation_deg": float(getattr(tr, "rotation_deg", 0)),
                "scale": float(getattr(tr, "scale", 1.0)),
                "valid": bool(getattr(tr, "valid", False)),
                "measurements": dict(getattr(tr, "measurements", {})),
            })

        self._results = targets
        self._phase = PHASE_MEASURED

        response: Dict[str, Any] = {
            "status": "ok",
            "elapsed_ms": round(elapsed_ms, 1),
            "num_targets": len(targets),
            "targets": targets,
        }

        if include_visual:
            t_vis = time.perf_counter()
            visual_b64 = generate_overview_visual(img, targets)
            logger.info("[TIMING] measure: generate_visual=%+.1fms",
                         (time.perf_counter() - t_vis) * 1000)
            if visual_b64:
                response["visual_b64"] = visual_b64

        logger.info("measure() completed: %d targets in %.1fms (total)",
                    len(targets), (time.perf_counter() - t0) * 1000)
        return response
        return response

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _resolve_path(self, path: str) -> str:
        """Resolve a potentially relative path against project_dir."""
        if os.path.isabs(path):
            return path
        return os.path.abspath(os.path.join(self.project_dir, path))

    # ------------------------------------------------------------------
    # Status summary
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        """
        Return a summary of the current project state.

        Returns:
            Dict with phase, template info, counts, etc.
        """
        return {
            "phase": self.phase,
            "project_dir": self.project_dir,
            "has_reference": self._reference_image is not None,
            "has_template": self.has_template,
            "template_shape": self.template_shape,
            "num_measurements": len(self._measurement_defs),
            "num_composed": len(self._composed_defs),
            "dag_valid": self._validate_dag(),
        }
