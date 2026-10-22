"""
Unified multi-target measurement workflow.

This is the single implementation shared by the GUI, the v1 REST API and the
v2 engine.  It was promoted from ``measure_gui.multi_target`` so that no API
layer has to import from a GUI package.

Pipeline
--------
Teach:
    1. ``teach_template`` defines an oriented ROI (a rotated box) on the
       reference image and derives the straightened patch template.
    2. Measurement tools are then defined in **patch coordinates**.
Inspect:
    1. Multi-target template matching finds all instances of the ROI.
    2. For each instance the patch is straightened and the tools run on it.
    3. Results are mapped back to the inspection image via the alignment
       inverse transform (``AlignResult.M_inv``).

Alignment is pluggable via :mod:`measure.alignment` (single-box rigid or
multi-control-point affine refinement).
"""

from __future__ import annotations
import time

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import cv2
import logging
logger = logging.getLogger(__name__)
import numpy as np

from measure.constants import EPS
from measure.viz import to_bgr, draw_text_shadow
from measure.measure_workflow import (
    EdgePairObject,
    EdgePointObject,
    FitCircleObject,
    FitLineObject,
    MeasurementWorkflow,
    PointCircleDistanceObject,
    PointLineDistanceObject,
    TemplateMatchPointObject,
    TwoLinesAngleObject,
    TwoPointsDistanceObject,
    TwoPointsLineObject,
)
from measure.measure_template import (
    CannyPreprocessor,
    CLAHEPreprocessor,
    Preprocessor,
    RawPreprocessor,
    SobelPreprocessor,
    TemplatePoint,
    ThresholdPreprocessor,
    _PREPROCESSOR_REGISTRY,
    _deserialize_preprocessor,
)

from measure.object_specs import (
    OBJECT_SPECS,
    defaults_for,
    factory_table,
    get_spec,
)
from measure.alignment import (
    AlignmentStrategy,
    AlignResult,
    FixedRoiAlignment,
    NoneAlignment,
    ObbAlignment,
    SingleBoxAlignment,
    MultiPointAlignment,
    strategy_from_roi_state,
)
from measure.rectify import (
    crop_and_straighten,
    map_point_to_original,
    map_point_via_affine,
)
from measure.result_format import (
    _format_geometric_result,
    _format_result_dict,
)

# The object-type registry lives in measure.object_specs; these aliases
# keep the historical private names working inside this module and for
# callers that imported them from here.
_OBJECT_FACTORIES = factory_table()
_get_defaults = defaults_for


@dataclass
class TargetResult:
    """Aggregated measurement results for a single matched target.

    ``id`` is a 1-based display index matching the order returned by
    :meth:`MultiTargetWorkflow.measure` (both for valid instances and for
    matches that fell below the score threshold).

    ``measurements`` maps measurement label -> result.  Primitive results are
    stored as the JSON-serializable dicts produced by
    :func:`_map_result_to_original` (coordinates already mapped back to the
    inspection image); this is why the values are dicts rather than
    ``GeometricResult`` instances.

    ``valid`` means "this target's output can be trusted", defined as:

      * no tools defined  -> decided by the locator alone (a match above the
        score threshold).  This is a supported configuration, not a degenerate
        one: for a pure localisation task the pose *is* the deliverable.
      * tools defined     -> locator matched AND every tool produced a valid
        result.

    A model with no tools therefore reports ``valid=True`` with an empty
    ``measurements`` — do not read that emptiness as failure; check
    :attr:`is_localization_only` if you need to distinguish the cases.
    """

    id: int
    #: Correlation score from stage 1, or ``None`` when nothing was searched
    #: (the ``none`` / ``fixed_roi`` alignment modes).  Callers comparing this
    #: against a threshold must handle None explicitly — ``None < t`` raises.
    score: Optional[float]
    rotation_deg: float
    scale: float
    center_row: float
    center_col: float
    valid: bool
    measurements: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def pose(self) -> Dict[str, float]:
        """Stage-1 localisation result as one dict.

        These five values are produced by the locator alone — they exist even
        when the model defines no measurement tools, which is the whole output
        of a pure localisation task.  Exposed as a view over the existing fields
        (never copied) so the two cannot drift apart.
        """
        return {
            "center_row": self.center_row,
            "center_col": self.center_col,
            "angle_deg": self.rotation_deg,
            "scale": self.scale,
            "score": self.score,
        }

    @property
    def is_localization_only(self) -> bool:
        """True when the model defined no tools, so ``pose`` is the entire result."""
        return not self.measurements

    def to_dict(self) -> Dict[str, Any]:
        # Existing keys are load-bearing for shipped callers; only ever append.
        return {
            "id": self.id,
            "score": self.score,
            "rotation_deg": self.rotation_deg,
            "scale": self.scale,
            "center_row": self.center_row,
            "center_col": self.center_col,
            "valid": self.valid,
            "measurements": {
                label: _result_to_dict(r)
                for label, r in self.measurements.items()
            },
            "meta": self.meta,
            "pose": self.pose,
            "is_localization_only": self.is_localization_only,
        }

    def summary_text(self) -> str:
        """Single-line summary for target list display."""
        # ``score`` is None for search-free models; a bare format spec would
        # raise TypeError deep inside the GUI's result rendering.
        score_txt = "n/a" if self.score is None else f"{self.score:.3f}"
        return (
            f"[Target #{self.id}] score={score_txt}, "
            f"rot={self.rotation_deg:.1f}°, "
            f"center=({self.center_row:.1f}, {self.center_col:.1f}), "
            f"{'✓' if self.valid else '✗'}"
        )


def _fmt_score(score: Optional[float], digits: int = 3) -> str:
    """Render a stage-1 score, or ``"n/a"`` when nothing was searched.

    Search-free alignment modes leave ``TargetResult.score`` as None; every
    display path formats it, so the guard lives here instead of at each site.
    """
    if score is None:
        return "n/a"
    return f"{float(score):.{digits}f}"


def _result_to_dict(result) -> Dict[str, Any]:
    """Convert a GeometricResult to a plain dict."""
    if result is None:
        return {"valid": False}
    d = {
        "type": result.type,
        "label": result.label,
        "valid": result.valid,
    }
    # Add type-specific values
    if hasattr(result, "row") and hasattr(result, "col"):
        d["row"] = result.row
        d["col"] = result.col
    if hasattr(result, "value"):
        d["value"] = result.value
    if hasattr(result, "a") and hasattr(result, "b") and hasattr(result, "c"):
        d["a"] = result.a
        d["b"] = result.b
        d["c"] = result.c
    if hasattr(result, "start_row"):
        d["start_row"] = result.start_row
        d["start_col"] = result.start_col
        d["end_row"] = result.end_row
        d["end_col"] = result.end_col
    if hasattr(result, "radius"):
        d["radius"] = result.radius
        d["center_row"] = result.center_row
        d["center_col"] = result.center_col
    if hasattr(result, "value_deg"):
        d["value_deg"] = result.value_deg
    if result.meta:
        d["meta"] = result.meta
    return d


# ===========================================================================
# Known object types for serialization
# ===========================================================================

# Maps object_type string -> (constructor_fn, params_from_dict_fn)

class _VirtualPoint:
    """Lightweight wrapper for virtual EdgePair endpoints in measurement_objects."""

    def __init__(self, label: str, result: "PointResult"):
        self.label = label
        self.result = result
        self._cached_measure = None












# ===========================================================================
# Profile debug image builder
# ===========================================================================


def _build_profile_image(
    roi_img: Optional[np.ndarray],
    gradient: np.ndarray,
    threshold: float,
    label: str,
    obj_type: str,
    img_width: int = 600,
    roi_height: int = 80,
    curve_height: int = 150,
) -> np.ndarray:
    """生成 ROI + 曲线上下拼接的 profile 调试图。

    上半部分：摆正的 ROI 小图（灰度转BGR）
    下半部分：gradient 曲线 + 阈值线

    Args:
        roi_img: ROI图像（可选）
        gradient: 梯度数据
        threshold: 阈值
        label: 标签名称
        obj_type: 对象类型
        img_width: 输出图像宽度
        roi_height: ROI区域高度
        curve_height: 曲线区域高度

    Returns:
        BGR 图像
    """
    margin = 10
    label_h = 25  # 标签区域高度
    total_h = label_h + roi_height + curve_height
    total_w = img_width

    canvas = np.ones((total_h, total_w, 3), dtype=np.uint8) * 240

    # --- 标签区域 ---
    type_str = "EdgePt" if obj_type == "EdgePoint" else "EdgePr"
    label_text = f"{label} ({type_str}) threshold={threshold:.1f}"
    cv2.putText(canvas, label_text, (margin, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.line(canvas, (0, label_h), (total_w, label_h), (200, 200, 200), 1)

    # --- ROI 区域 ---
    roi_y_start = label_h

    # 计算曲线绘图区左右边界（ROI 与曲线共用，确保 X 轴一一对齐）
    plot_x0_roi = margin + 30  # 左边留空给Y轴标签
    plot_x1_roi = total_w - margin

    if roi_img is not None:
        # 确保是 BGR 格式
        if len(roi_img.shape) == 2:
            roi_img = cv2.cvtColor(roi_img, cv2.COLOR_GRAY2BGR)
        roi_h, roi_w = roi_img.shape[:2]
        if roi_h > 0 and roi_w > 0:
            # 缩放到 roi_height 高度，宽度按比例
            scale = roi_height / roi_h
            new_w = int(roi_w * scale)
            plot_roi_w = plot_x1_roi - plot_x0_roi
            new_w = min(new_w, plot_roi_w)
            if new_w > 10:
                roi_resized = cv2.resize(roi_img, (new_w, roi_height))
                # 与曲线共用相同的左边界居中
                x_offset = plot_x0_roi + (plot_roi_w - new_w) // 2
                canvas[roi_y_start:roi_y_start + roi_height, x_offset:x_offset + new_w] = roi_resized
                # 边框
                cv2.rectangle(canvas, (x_offset, roi_y_start),
                              (x_offset + new_w, roi_y_start + roi_height), (180, 180, 180), 1)

    # --- 曲线区域 ---
    curve_y_start = roi_y_start + roi_height
    cv2.line(canvas, (0, curve_y_start), (total_w, curve_y_start), (200, 200, 200), 1)

    plot_x0 = plot_x0_roi  # 与 ROI 共用左边界
    plot_x1 = total_w - margin
    plot_y0 = curve_y_start + margin
    plot_y1 = curve_y_start + curve_height - margin
    plot_h = plot_y1 - plot_y0
    plot_w = plot_x1 - plot_x0

    # 绘制边框
    cv2.rectangle(canvas, (plot_x0, plot_y0), (plot_x1, plot_y1), (200, 200, 200), 1)

    # 归一化 gradient
    g_min = gradient.min()
    g_max = gradient.max()
    g_range = g_max - g_min
    if g_range < 1e-6:
        g_range = 1.0

    # 绘制零线
    zero_y = plot_y0 + int((0 - g_min) / g_range * plot_h)
    zero_y = max(plot_y0, min(plot_y1, zero_y))
    cv2.line(canvas, (plot_x0, zero_y), (plot_x1, zero_y), (180, 180, 180), 1, cv2.LINE_AA)

    # 绘制阈值线
    if threshold > 0:
        thresh_pos_y = plot_y0 + int((threshold - g_min) / g_range * plot_h)
        thresh_pos_y = max(plot_y0, min(plot_y1, thresh_pos_y))
        cv2.line(canvas, (plot_x0, thresh_pos_y), (plot_x1, thresh_pos_y), (0, 0, 200), 1, cv2.LINE_AA)

        thresh_neg_y = plot_y0 + int((-threshold - g_min) / g_range * plot_h)
        thresh_neg_y = max(plot_y0, min(plot_y1, thresh_neg_y))
        cv2.line(canvas, (plot_x0, thresh_neg_y), (plot_x1, thresh_neg_y), (0, 0, 200), 1, cv2.LINE_AA)

    # 绘制 Y 轴标签
    cv2.putText(canvas, f"{g_max:.0f}", (margin, plot_y0 + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 120), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{g_min:.0f}", (margin, plot_y1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 120), 1, cv2.LINE_AA)
    cv2.putText(canvas, "0", (margin, zero_y + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 80, 80), 1, cv2.LINE_AA)

    # 绘制 gradient 曲线（与采样框长度对齐）
    n = len(gradient)
    if n > 1 and plot_w > 0:
        # 使用与上方 ROI 相同的缩放比例，使曲线与 ROI 显示宽度对齐
        if roi_img is not None and roi_img.shape[0] > 0:
            roi_h = roi_img.shape[0]
            scale = roi_height / roi_h
            curve_width = min(int(n * scale), plot_w)
        else:
            curve_width = plot_w
        curve_x0 = plot_x0 + (plot_w - curve_width) // 2
        curve_x1 = curve_x0 + curve_width
        
        # 绘制采样框长度的标记线
        cv2.line(canvas, (curve_x0, plot_y1 + 2), (curve_x0, plot_y1 + 8), (100, 100, 100), 1)
        cv2.line(canvas, (curve_x1, plot_y1 + 2), (curve_x1, plot_y1 + 8), (100, 100, 100), 1)
        
        # 绘制宽度标签（纯数字，对应gradient数据点个数）
        length_text = f"{n}pts"
        text_size = cv2.getTextSize(length_text, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]
        text_x = (curve_x0 + curve_x1 - text_size[0]) // 2
        cv2.putText(canvas, length_text, (text_x, plot_y1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1, cv2.LINE_AA)
        
        # 绘制曲线
        points = []
        for i in range(n):
            x = curve_x0 + int(i * curve_width / (n - 1))
            y = plot_y0 + int((gradient[i] - g_min) / g_range * plot_h)
            y = max(plot_y0, min(plot_y1, y))
            points.append([x, y])
        points = np.array(points, dtype=np.int32)
        cv2.polylines(canvas, [points], False, (0, 180, 180), 1, cv2.LINE_AA)

    return canvas


















# ===========================================================================
# Model asset format (manifest v3.0)
# ===========================================================================

MANIFEST_FILENAME = "manifest.json"
REFERENCE_FILENAME = "reference.npy"
DERIVED_FILENAME = "workflow.npz"
MANIFEST_CONTRACT_VERSION = "3.1"

#: Contracts a current reader can still open.  Kept as an explicit list rather
#: than a numeric comparison: a version is only compatible if someone has decided
#: it is, and silently accepting an arbitrary older number is how a reader ends up
#: guessing at fields it does not understand.
#:
#: 3.0 -> 3.1 added detector-based localisation (``alignment.mode == "obb"``,
#: carrying ``model_ref`` and ``nominal_size`` inside ``roi_state``).  Nothing was
#: removed or reinterpreted, so 3.0 assets load unchanged and keep their template
#: or fixed-ROI behaviour.
COMPATIBLE_MANIFEST_CONTRACT_VERSIONS = ("3.0", "3.1")


def _sha256_bytes(payload: bytes) -> str:
    """Hex sha256 of a byte string."""
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    """Hex sha256 of a file's contents."""
    return _sha256_bytes(Path(path).read_bytes())


def _stable_object_id(label: str) -> str:
    """Deterministic object id for a measurement label.

    Model assets use stable ids (derived from the label) rather than the
    session-scoped random ids the modelling API hands out, so that re-saving a
    model produces a byte-identical manifest.
    """
    return hashlib.sha1(label.encode("utf-8")).hexdigest()[:16]


# ===========================================================================
# MultiTargetWorkflow
# ===========================================================================




class MultiTargetWorkflow:
    """
    Multi-target measurement workflow.

    Teaching:
        wf = MultiTargetWorkflow()
        wf.teach_template(ref_img, center=(200, 300), size=(120, 180), angle_deg=15)
        wf.add_measurement("EdgePoint", "edge_top", row=10, col=60, angle=0, length1=50, length2=5)
        wf.add_measurement("EdgePoint", "edge_bot", row=110, col=60, angle=0, length1=50, length2=5)
        wf.add_measurement("TwoPointsDistance", "gap", point_a_label="edge_top", point_b_label="edge_bot")
        wf.save("project.mtwf")

    Inspection:
        wf = MultiTargetWorkflow.load("project.mtwf")
        results = wf.measure(insp_img)
        for r in results:
            print(r.summary_text())
            for label, m in r.measurements.items():
                print(f"  {label}: {m}")
        vis = wf.visualize(insp_img)
    """

    def __init__(self):
        # Alignment strategy (defaults to single-box mode)
        self._alignment: AlignmentStrategy = SingleBoxAlignment()

        # Matching parameters (may be overridden before teach_template)
        self._match_score_threshold: float = 0.5
        self._angle_range: Tuple[float, float] = (-30.0, 30.0)
        self._angle_step: float = 1.0
        self._max_matches: int = 0
        self._overlap: float = 0.3
        self._coarse_fine: bool = True
        self._coarse_angle_step: float = 5.0

        # Measurement definitions (ordered list)
        self._measurement_defs: List[Dict[str, Any]] = []

        # TemplateMatchPoint template cache: label -> TemplatePoint
        self._template_match_points: Dict[str, Any] = {}

        # Last results
        self._results: List[TargetResult] = []
        self._last_inspection_image: Optional[np.ndarray] = None

        # Debug image saving
        self._debug_dir: Optional[str] = None

        # Optional pixel->physical scale (mm/pixel) from checkerboard
        # calibration.  None (or 0) means "uncalibrated" — results stay in px.
        self._physical_scale_mm: Optional[float] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def physical_scale_mm(self) -> Optional[float]:
        """mm/pixel scale from checkerboard calibration (None = uncalibrated)."""
        return self._physical_scale_mm

    @physical_scale_mm.setter
    def physical_scale_mm(self, value: Optional[float]) -> None:
        self._physical_scale_mm = float(value) if value else None

    @property
    def alignment(self) -> AlignmentStrategy:
        """The active alignment strategy."""
        return self._alignment

    @alignment.setter
    def alignment(self, value: AlignmentStrategy):
        self._alignment = value

    @property
    def template_image(self) -> Optional[np.ndarray]:
        """The straightened template image (for display in GUI)."""
        return self._alignment.template_image

    @property
    def reference_image(self) -> Optional[np.ndarray]:
        """The stored reference image (for project save/load)."""
        return self._alignment._reference_image

    @property
    def is_taught(self) -> bool:
        """Whether a model has been taught and can be saved / executed.

        Deliberately *not* "has a template": the search-free alignment modes
        (``none``, ``fixed_roi``) are fully usable without one, so callers that
        gate on ``template_point`` wrongly reject them.  The real requirement is
        a reference image — without it nothing can be reconstructed — which is
        exactly what :meth:`save` and :meth:`_manifest` check internally.  Use
        this instead of poking private attributes from the UI or API layers.
        """
        return self._alignment._reference_image is not None

    @property
    def box_center(self) -> Tuple[float, float]:
        return self._alignment.box_center

    @property
    def box_size(self) -> Tuple[float, float]:
        return self._alignment.box_size

    @property
    def box_angle_deg(self) -> float:
        return self._alignment.box_angle_deg

    @property
    def measurement_defs(self) -> List[Dict[str, Any]]:
        return list(self._measurement_defs)

    @property
    def results(self) -> List[TargetResult]:
        return list(self._results)

    @property
    def angle_range(self) -> Tuple[float, float]:
        return self._angle_range

    @angle_range.setter
    def angle_range(self, value: Tuple[float, float]):
        self._angle_range = value

    @property
    def match_score_threshold(self) -> float:
        return self._match_score_threshold

    @match_score_threshold.setter
    def match_score_threshold(self, value: float):
        self._match_score_threshold = value

    @property
    def max_matches(self) -> int:
        return self._max_matches

    @max_matches.setter
    def max_matches(self, value: int):
        self._max_matches = value
        tp = self._alignment.template_point
        if tp is not None:
            tp.max_matches = value

    @property
    def overlap(self) -> float:
        return self._overlap

    @overlap.setter
    def overlap(self, value: float):
        self._overlap = float(value)
        tp = self._alignment.template_point
        if tp is not None:
            tp.overlap = float(value)

    @property
    def preprocessor_type(self) -> str:
        tp = self._alignment.template_point
        if tp is None:
            return "raw"
        return tp.preprocessor.name

    @property
    def debug_dir(self) -> Optional[str]:
        """Directory for saving debug images. None disables debug output."""
        return self._debug_dir

    @debug_dir.setter
    def debug_dir(self, path: Optional[str]):
        self._debug_dir = path

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------

    def _debug_save(self, filename: str, image: np.ndarray):
        """Save a debug image if debug_dir is set."""
        if not self._debug_dir:
            return
        import os
        os.makedirs(self._debug_dir, exist_ok=True)
        filepath = os.path.join(self._debug_dir, filename)
        cv2.imwrite(filepath, image)

    def _debug_save_inspection_boxes(self, inspection_image: np.ndarray):
        """Draw rotated target boxes on the inspection image and save."""
        vis = to_bgr(inspection_image)

        color_palette = [
            (0, 255, 0),     # green
            (0, 255, 255),   # yellow
            (255, 0, 255),   # magenta
            (255, 255, 0),   # cyan
            (0, 165, 255),   # orange
            (255, 0, 0),     # blue
            (128, 0, 128),   # dark magenta
            (0, 128, 128),   # dark yellow
        ]

        for target in self._results:
            color = color_palette[target.id % len(color_palette)]

            # Draw rotated box
            corners = _compute_target_box_corners(target, self._alignment.box_size)
            pts = corners[:, ::-1].astype(np.int32).reshape((-1, 1, 2))
            thickness = 2 if target.valid else 1
            cv2.polylines(vis, [pts], True, color, thickness=thickness)

            # Label
            label_text = (
                f"T#{target.id} s={_fmt_score(target.score)} "
                f"r={target.rotation_deg:.1f}° "
                f"{'OK' if target.valid else 'FAIL'}"
            )
            font_scale = 0.45
            cv2.putText(
                vis, label_text,
                (int(target.center_col) + 5, int(target.center_row) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA,
            )

        # Summary
        n_valid = sum(1 for t in self._results if t.valid)
        summary = (
            f"MultiTarget: {len(self._results)} targets, "
            f"{n_valid} valid"
        )
        cv2.putText(
            vis, summary,
            (10, vis.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )

        self._debug_save("inspection_boxes.png", vis)

    def _debug_draw_measurements_on_patch(
        self,
        patch: np.ndarray,
        raw_results: Dict[str, Any],
    ) -> np.ndarray:
        """Draw measurement tool ROIs and detected results on a patch copy.

        Uses OpenCV drawing to overlay:
        - Tool ROIs in light blue (where each tool is looking)
        - Valid detections in green
        - Invalid detections in red
        - Composed measurement connecting lines in yellow

        Returns:
            BGR image with overlays drawn.
        """
        vis = to_bgr(patch)

        h, w = vis.shape[:2]

        # Colors
        ROI_COLOR = (255, 180, 80)       # light blue (BGR)
        VALID_COLOR = (0, 255, 0)        # green
        INVALID_COLOR = (0, 0, 255)      # red
        COMPOSED_COLOR = (0, 255, 255)   # yellow
        TICK_COLOR = (180, 180, 180)     # gray

        # --- Pass 1: draw each measurement tool's ROI ---
        for d in self._measurement_defs:
            obj_type = d["object_type"]
            label = d["label"]
            params = d["params"]

            result = raw_results.get(label)
            is_valid = (
                result.valid
                if (result is not None and hasattr(result, "valid"))
                else False
            )

            if obj_type in ("EdgePoint", "EdgePair"):
                self._debug_draw_edge_roi(vis, params, is_valid)

            elif obj_type == "FitLine":
                self._debug_draw_fit_line_roi(vis, params, is_valid)

            elif obj_type == "FitCircle":
                self._debug_draw_fit_circle_roi(vis, params, is_valid)

            elif obj_type == "TemplateMatchPoint":
                self._debug_draw_tmpl_match_roi(vis, params, is_valid)

        # --- Pass 2: draw measurement results ---
        from measure.measure_workflow import CircleResult, LineResult, PointResult

        for label, result in raw_results.items():
            if label == "_error" or not hasattr(result, "valid"):
                continue

            color = VALID_COLOR if result.valid else INVALID_COLOR

            if isinstance(result, PointResult) and result.valid:
                pt = (int(round(result.col)), int(round(result.row)))
                cv2.drawMarker(
                    vis, pt, color,
                    markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2,
                )
                cv2.putText(
                    vis, label, (pt[0] + 8, pt[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
                )

            elif isinstance(result, LineResult) and result.valid:
                pt1 = (int(round(result.start_col)), int(round(result.start_row)))
                pt2 = (int(round(result.end_col)), int(round(result.end_row)))
                cv2.line(vis, pt1, pt2, color, thickness=2)
                mid = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
                cv2.putText(
                    vis, label, (mid[0] + 5, mid[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
                )

            elif isinstance(result, CircleResult) and result.valid:
                ct = (int(round(result.center_col)), int(round(result.center_row)))
                cv2.circle(vis, ct, int(round(result.radius)), color, thickness=2)
                cv2.putText(
                    vis, label, (ct[0] + 5, ct[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
                )
                # 绘制最长半径和最短半径线
                meta = getattr(result, "meta", {})
                max_radius_point = meta.get("max_radius_point")
                min_radius_point = meta.get("min_radius_point")
                max_radius = meta.get("max_radius", 0)
                min_radius = meta.get("min_radius", 0)
                if max_radius_point:
                    max_pt = (int(round(max_radius_point[0])), int(round(max_radius_point[1])))
                    cv2.line(vis, ct, max_pt, (0, 0, 255), 2, cv2.LINE_AA)  # 红色
                    cv2.circle(vis, max_pt, 4, (0, 0, 255), -1)
                    # 显示最长半径值
                    max_label = f"Rmax={max_radius:.1f}"
                    max_label_pos = ((ct[0] + max_pt[0]) // 2, (ct[1] + max_pt[1]) // 2 - 10)
                    cv2.putText(vis, max_label, max_label_pos,
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
                if min_radius_point:
                    min_pt = (int(round(min_radius_point[0])), int(round(min_radius_point[1])))
                    cv2.line(vis, ct, min_pt, (255, 0, 0), 2, cv2.LINE_AA)  # 蓝色
                    cv2.circle(vis, min_pt, 4, (255, 0, 0), -1)
                    # 显示最短半径值
                    min_label = f"Rmin={min_radius:.1f}"
                    min_label_pos = ((ct[0] + min_pt[0]) // 2, (ct[1] + min_pt[1]) // 2 + 15)
                    cv2.putText(vis, min_label, min_label_pos,
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
                # 显示椭圆度
                ellipticity = meta.get("ellipticity", 0)
                if ellipticity > 0:
                    ell_label = f"Ellipticity={ellipticity:.1f}"
                    cv2.putText(vis, ell_label, (ct[0] + 5, ct[1] + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

        # --- Pass 3: draw composed measurement connections ---
        for d in self._measurement_defs:
            obj_type = d["object_type"]
            params = d["params"]

            if obj_type in ("TwoPointsDistance", "TwoPointsLine"):
                p1 = raw_results.get(params.get("point_a_label", ""))
                p2 = raw_results.get(params.get("point_b_label", ""))
                if p1 and p2 and p1.valid and p2.valid:
                    if not (hasattr(p1, "col") and hasattr(p1, "row")
                            and hasattr(p2, "col") and hasattr(p2, "row")):
                        continue
                    pt1 = (int(p1.col), int(p1.row))
                    pt2 = (int(p2.col), int(p2.row))
                    cv2.line(vis, pt1, pt2, COMPOSED_COLOR, thickness=2)

            elif obj_type == "PointLineDistance":
                pt = raw_results.get(params.get("point_label", ""))
                line = raw_results.get(params.get("line_label", ""))
                if (pt and line and pt.valid and line.valid
                        and hasattr(pt, "col") and hasattr(pt, "row")
                        and hasattr(line, "a") and hasattr(line, "c")):
                    a, b, c = line.a, line.b, line.c
                    denom = a * a + b * b
                    if denom > EPS:
                        proj_col = (b * b * pt.col - a * b * pt.row - a * c) / denom
                        proj_row = (-a * b * pt.col + a * a * pt.row - b * c) / denom
                        p1 = (int(pt.col), int(pt.row))
                        p2 = (int(proj_col), int(proj_row))
                        cv2.line(vis, p1, p2, COMPOSED_COLOR, thickness=1,
                                 lineType=cv2.LINE_AA)

        # Legend
        cv2.putText(
            vis, "Green=Valid  Red=Invalid  Blue=ROI  Yellow=Composed",
            (5, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1,
            cv2.LINE_AA,
        )

        return vis

    def _save_profile_debug_images(
        self,
        target_index: int,
        patch: np.ndarray,
        measurement_objects: Dict[str, Any],
    ) -> None:
        """为每条 EdgePoint/EdgePair 保存独立的 profile 调试图。"""
        if not self._debug_dir:
            return

        patch_gray = patch if len(patch.shape) == 2 else cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

        for d in self._measurement_defs:
            obj_type = d["object_type"]
            label = d["label"]
            if obj_type not in ("EdgePoint", "EdgePair"):
                continue

            obj = measurement_objects.get(label)
            if obj is None:
                continue
            measure = getattr(obj, "_cached_measure", None)
            if measure is None or measure.last_gradient is None:
                continue

            # 提取摆正的 ROI
            roi_img = None
            try:
                roi_img = measure.extract_roi(patch_gray)
                if len(roi_img.shape) == 2:
                    roi_img = cv2.cvtColor(roi_img, cv2.COLOR_GRAY2BGR)
            except Exception:
                pass

            # 生成上下拼接图
            profile_img = _build_profile_image(
                roi_img, measure.last_gradient, measure.last_threshold,
                label, obj_type,
            )

            # 保存
            self._debug_save(
                f"target_{target_index + 1:02d}_profile_{label}.png",
                profile_img,
            )


    # --- Per-type ROI drawing helpers ---

    @staticmethod
    def _debug_draw_edge_roi(vis: np.ndarray, params: dict, is_valid: bool):
        """Draw caliper ROI rectangle and direction arrow."""
        color = (0, 255, 0) if is_valid else (0, 0, 255)
        roi_color = (255, 180, 80)

        row, col = params["row"], params["col"]
        angle = params["angle"]          # radians
        length1 = params["length1"]      # half-length along probe direction
        length2 = params["length2"]      # half-width perpendicular

        # Direction vector (matching measure1D convention: 0=right, pi/2=down)
        dr = np.sin(angle)
        dc = np.cos(angle)
        # Perpendicular direction (rotate 90 degrees counterclockwise)
        pr = -dc
        pc = dr

        corners = np.array([
            [row - dr * length1 - pr * length2, col - dc * length1 - pc * length2],
            [row - dr * length1 + pr * length2, col - dc * length1 + pc * length2],
            [row + dr * length1 + pr * length2, col + dc * length1 + pc * length2],
            [row + dr * length1 - pr * length2, col + dc * length1 - pc * length2],
        ], dtype=np.float64)
        pts = corners[:, ::-1].reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(vis, [pts], True, roi_color, thickness=1)

        # Direction arrow from center
        pt_start = (int(col), int(row))
        pt_end = (int(col + dc * length1 * 0.7), int(row + dr * length1 * 0.7))
        cv2.arrowedLine(vis, pt_start, pt_end, roi_color, thickness=1, tipLength=0.2)

    @staticmethod
    def _debug_draw_fit_line_roi(vis: np.ndarray, params: dict, is_valid: bool):
        """Draw fit line ROI with perpendicular measurement ticks."""
        color = (0, 255, 0) if is_valid else (0, 0, 255)
        roi_color = (255, 180, 80)
        tick_color = (180, 180, 180)

        start = params["start"]
        end = params["end"]
        pt1 = (int(start[1]), int(start[0]))
        pt2 = (int(end[1]), int(end[0]))

        # Dashed main line
        num_segments = 20
        for i in range(0, num_segments, 2):
            t1 = i / num_segments
            t2 = min((i + 1) / num_segments, 1.0)
            s1 = (int(start[1] + t1 * (end[1] - start[1])),
                  int(start[0] + t1 * (end[0] - start[0])))
            s2 = (int(start[1] + t2 * (end[1] - start[1])),
                  int(start[0] + t2 * (end[0] - start[0])))
            cv2.line(vis, s1, s2, roi_color, thickness=1)

        # Endpoint dots
        cv2.circle(vis, pt1, 3, roi_color, -1)
        cv2.circle(vis, pt2, 3, roi_color, -1)

        # Perpendicular measurement ticks
        num_measures = params.get("num_measures", 10)
        measure_length2 = params.get("measure_length2", 25.0)
        dr = end[0] - start[0]
        dc = end[1] - start[1]
        line_len = np.sqrt(dr**2 + dc**2)
        if line_len > EPS:
            for i in range(num_measures):
                t = (i + 0.5) / num_measures
                mr = start[0] + t * dr
                mc = start[1] + t * dc
                pdr = -dc / line_len * measure_length2
                pdc = dr / line_len * measure_length2
                t1 = (int(mc - pdc), int(mr - pdr))
                t2 = (int(mc + pdc), int(mr + pdr))
                cv2.line(vis, t1, t2, tick_color, thickness=1)

    @staticmethod
    def _debug_draw_fit_circle_roi(vis: np.ndarray, params: dict, is_valid: bool):
        """Draw fit circle ROI (center cross + expected radius circle)."""
        color = (0, 255, 0) if is_valid else (0, 0, 255)
        roi_color = (255, 180, 80)

        center = params["center"]
        radius = params.get("radius", 50.0)
        ct = (int(center[1]), int(center[0]))

        # Expected radius circle
        cv2.circle(vis, ct, int(radius), roi_color, thickness=1)
        # Center crosshair
        cv2.drawMarker(
            vis, ct, roi_color,
            markerType=cv2.MARKER_CROSS, markerSize=12, thickness=1,
        )

    @staticmethod
    def _debug_draw_tmpl_match_roi(vis: np.ndarray, params: dict, is_valid: bool):
        """Draw template-matching point ROI (bounding box + crosshair)."""
        color = (0, 255, 0) if is_valid else (0, 0, 255)
        roi_color = (255, 180, 80)

        row, col = params["row"], params["col"]
        size = params.get("template_size", 40)
        half = size // 2

        pt1 = (int(col - half), int(row - half))
        pt2 = (int(col + half), int(row + half))
        cv2.rectangle(vis, pt1, pt2, roi_color, thickness=1)
        cv2.drawMarker(
            vis, (int(col), int(row)), roi_color,
            markerType=cv2.MARKER_CROSS, markerSize=8, thickness=1,
        )

    # ------------------------------------------------------------------
    # Teaching
    # ------------------------------------------------------------------

    def teach_without_search(
        self,
        reference_image: np.ndarray,
        center: Tuple[float, float],
        size: Tuple[float, float],
        angle_deg: float = 0.0,
    ) -> None:
        """Teach a model that performs no template search.

        Public entry point for the ``none`` and ``fixed_roi`` alignment modes:
        stage 1 is skipped, so there are no matching parameters to tune and no
        template to build.  The ROI geometry is still recorded because it defines
        the measurement frame and drives visualisation.

        Args:
            reference_image: The reference (teaching) image.
            center: ``(row, col)`` of the fixed ROI; ignored by ``none`` mode.
            size: ``(height, width)`` of the fixed ROI.
            angle_deg: ROI rotation in degrees.
        """
        strategy = self._alignment
        if not isinstance(strategy, (NoneAlignment, FixedRoiAlignment)):
            raise RuntimeError(
                f"Alignment mode '{strategy.mode_name}' requires a template "
                f"search; use teach_template() instead."
            )
        strategy.teach(reference_image, center, size, angle_deg)
        self._after_teach()

    def _teach_without_search(
        self,
        reference_image: np.ndarray,
        alignment: Dict[str, Any],
    ) -> None:
        """Restore a search-free model from its manifest ``alignment`` block."""
        self.teach_without_search(
            reference_image,
            center=tuple(alignment["box_center"]),
            size=tuple(alignment["box_size"]),
            angle_deg=float(alignment["box_angle_deg"]),
        )

    def _teach_obb(
        self,
        reference_image: np.ndarray,
        alignment: Dict[str, Any],
    ) -> None:
        """Restore a detector-backed model from its manifest ``alignment`` block.

        The strategy was already rebuilt by ``strategy_from_roi_state``, so this
        only re-establishes the reference image and the shared caches.  Detection
        is deliberately *not* re-run: it would need a live detector (an injected
        capability, absent at load time) and would re-derive a nominal patch size
        from whatever the model happened to report today instead of honouring the
        frame the tools were taught in.
        """
        stored = self._alignment
        stored._reference_image = np.asarray(reference_image).copy()
        stored._template_image = stored._reference_image
        self._after_teach()

    def _after_teach(self) -> None:
        """Refresh the workflow-level caches shared by every teach path."""
        # Convenience: keep local references to frequently-accessed items
        self._template_image_legacy = self._alignment.template_image
        self._box_center_legacy = self._alignment.box_center
        self._box_size_legacy = self._alignment.box_size
        self._box_angle_deg_legacy = self._alignment.box_angle_deg
        # Build TemplateMatchPoint instances from the template image
        self._build_template_match_points()

    def teach_template(
        self,
        reference_image: np.ndarray,
        center: Tuple[float, float],
        size: Tuple[float, float],
        angle_deg: float,
        preprocessor: Optional[Preprocessor] = None,
        match_score_threshold: float = 0.5,
        angle_range: Tuple[float, float] = (-30.0, 30.0),
        angle_step: float = 1.0,
        max_matches: int = 0,
        overlap: float = 0.3,
        coarse_fine: bool = True,
        coarse_angle_step: float = 5.0,
        pyramid_decimate: int = 0,
        pyramid_max_template_size: int = 400,
    ):
        """
        Define the template from a rotated bounding box on the reference image.

        Args:
            reference_image: Grayscale reference image.
            center: (row, col) center of the rotated box.
            size: (height, width) of the box (in unrotated space).
            angle_deg: Rotation angle of the box in degrees.
            preprocessor: Preprocessor for template matching (default RawPreprocessor).
            match_score_threshold: Minimum NCC score for valid match.
            angle_range: (min, max) search range in degrees.
            angle_step: Step size for fine angle search.
            max_matches: Max matches to return (0 = unlimited).
            overlap: Maximum allowed IoU overlap in [0, 1] for NMS.
                     0 = no overlap at all (default 0.3).
            coarse_fine: Use two-stage coarse-to-fine search.
            coarse_angle_step: Step size for coarse angle search.
            pyramid_decimate: Pyramid decimation level (0=disabled).
            pyramid_max_template_size: Max template side length after decimation.
        """
        self._alignment.teach(
            reference_image, center, size, angle_deg,
            preprocessor=preprocessor,
            match_score_threshold=match_score_threshold,
            angle_range=angle_range, angle_step=angle_step,
            max_matches=max_matches, overlap=overlap,
            coarse_fine=coarse_fine, coarse_angle_step=coarse_angle_step,
            pyramid_decimate=pyramid_decimate,
            pyramid_max_template_size=pyramid_max_template_size,
        )

        # Keep the workflow-level matching cache in sync with what was taught.
        # Without this the public match_score_threshold / angle_range / ...
        # properties keep reporting stale defaults after a re-teach.
        self._match_score_threshold = float(match_score_threshold)
        self._angle_range = tuple(angle_range)
        self._angle_step = float(angle_step)
        self._max_matches = int(max_matches)
        self._overlap = float(overlap)
        self._coarse_fine = bool(coarse_fine)
        self._coarse_angle_step = float(coarse_angle_step)

        self._after_teach()

    def _build_template_match_points(self):
        """Create TemplatePoint instances for all TemplateMatchPoint definitions.

        Must be called after teach_template() when alignment.template_image is available.
        Uses the straightened template image as the reference for cropping.
        """
        preproc_map = {
            "raw": RawPreprocessor(),
            "canny": CannyPreprocessor(50.0, 150.0),
            "sobel": SobelPreprocessor(3),
            "clahe": CLAHEPreprocessor(2.0),
            "threshold": ThresholdPreprocessor(128.0),
        }

        self._template_match_points.clear()
        for d in self._measurement_defs:
            if d["object_type"] == "TemplateMatchPoint":
                label = d["label"]
                params = d["params"]
                preprocessor = preproc_map.get(
                    params.get("preprocessor_type", "raw"), RawPreprocessor()
                )
                angle_range_half = float(params.get("angle_range_half", 15.0))
                _pd_raw = params.get("pyramid_decimate", 0)
                _pd_map = {"禁用": 0, "2x": 1, "4x": 2, "8x": 3}
                if isinstance(_pd_raw, str):
                    # 兼容 "2x"/"4x"/"8x" 字符串和 "禁用" 中文
                    pyramid_decimate = _pd_map.get(_pd_raw, 0)
                else:
                    try:
                        pyramid_decimate = int(_pd_raw)
                    except (ValueError, TypeError):
                        pyramid_decimate = 0
                # When pyramid is enabled, compute max_template_size to enforce
                # downsampling: target scaled template side length ~ 200px
                _pms = int(params.get("pyramid_max_template_size", 400))
                if pyramid_decimate > 0:
                    _ts = int(params.get("template_size", 40))
                    _pms = min(_pms, max(_ts // 2, 80))
                tp = TemplatePoint(
                    self._alignment.template_image,
                    click_row=float(params["row"]),
                    click_col=float(params["col"]),
                    template_size=int(params.get("template_size", 40)),
                    preprocessor=preprocessor,
                    match_score_threshold=float(
                        params.get("match_score_threshold", 0.5)
                    ),
                    use_subpixel=bool(params.get("use_subpixel", True)),
                    rotation_invariant=(angle_range_half > 0),
                    angle_range=(-angle_range_half, angle_range_half),
                    angle_step=float(params.get("angle_step", 1.0)),
                    multi_target=False,
                    pyramid_decimate=pyramid_decimate,
                    pyramid_max_template_size=_pms,
                )
                self._template_match_points[label] = tp

    def add_measurement(self, object_type: str, label: str, **params):
        """
        Add a measurement definition.

        Args:
            object_type: One of 'EdgePoint', 'EdgePair', 'FitLine', 'FitCircle',
                        'TwoPointsLine', 'TwoPointsDistance', 'PointLineDistance',
                        'TwoLinesAngle', 'PointCircleDistance'.
            label: Unique label for this measurement.
            **params: Parameters for the measurement object constructor.
                     Coordinates should be in the **straightened template** space.
        """
        if object_type not in _OBJECT_FACTORIES:
            raise ValueError(
                f"Unknown object_type: {object_type}. "
                f"Known types: {list(_OBJECT_FACTORIES.keys())}"
            )

        # Validate params completeness
        _, expected_keys = _OBJECT_FACTORIES[object_type]
        for key in expected_keys:
            if key not in params and key not in _get_defaults(object_type):
                raise ValueError(
                    f"Missing required parameter '{key}' for {object_type}"
                )

        # Merge defaults
        full_params = _get_defaults(object_type).copy()
        full_params.update(params)

        self._measurement_defs.append({
            "object_type": object_type,
            "label": label,
            "params": full_params,
        })

        # If adding a TemplateMatchPoint and template image is available,
        # immediately build its TemplatePoint instance.
        if object_type == "TemplateMatchPoint" and self._alignment.template_image is not None:
            self._build_template_match_points()

        return self

    def remove_measurement(self, label: str):
        """Remove a measurement definition by label."""
        self._measurement_defs = [
            d for d in self._measurement_defs if d["label"] != label
        ]
        # Also remove cached TemplateMatchPoint
        self._template_match_points.pop(label, None)
        # Also remove any composed measurements that reference this label
        self._measurement_defs = [
            d for d in self._measurement_defs
            if not _references_label(d, label)
        ]

    def update_measurement(self, label: str, **params):
        """Update a measurement definition's params by label.

        Merges the given params into the existing definition, then rebuilds
        cached TemplateMatchPoint instances if needed.
        """
        for d in self._measurement_defs:
            if d["label"] == label:
                d["params"].update(params)
                break
        # Rebuild cached TemplateMatchPoint if the updated tool is one
        if self._alignment.template_image is not None:
            self._build_template_match_points()

    def clear_measurements(self):
        """Remove all measurement definitions."""
        self._measurement_defs = []
        self._template_match_points.clear()

    def move_measurement_up(self, label: str):
        """Move a measurement up in the execution order."""
        for i, d in enumerate(self._measurement_defs):
            if d["label"] == label and i > 0:
                self._measurement_defs[i], self._measurement_defs[i - 1] = (
                    self._measurement_defs[i - 1],
                    self._measurement_defs[i],
                )
                break

    def move_measurement_down(self, label: str):
        """Move a measurement down in the execution order."""
        for i, d in enumerate(self._measurement_defs):
            if d["label"] == label and i < len(self._measurement_defs) - 1:
                self._measurement_defs[i], self._measurement_defs[i + 1] = (
                    self._measurement_defs[i + 1],
                    self._measurement_defs[i],
                )
                break

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def measure(self, inspection_image: np.ndarray) -> List[TargetResult]:
        """
        Execute multi-target measurement on an inspection image.

        Steps:
        1. Multi-target template matching
        2. For each match: crop+straighten, run measurements, map results back

        Args:
            inspection_image: Grayscale inspection image.

        Returns:
            List of TargetResult, one per detected target.
        """
        self._last_inspection_image = inspection_image.copy()
        self._results = []

        # Stage 1 is pluggable, and "no stage 1" is a legitimate choice.
        # Strategies without a template (``none``, ``fixed_roi``) synthesise one
        # pseudo-match so the rest of the pipeline — per-target measurement,
        # coordinate mapping, debug capture — stays on a single path instead of
        # branching.  The score is 1.0 because there was nothing to score.
        t0 = time.perf_counter()
        self._timer_match = t0
        if hasattr(self._alignment, "detect"):
            # Detector-backed stage 1 (OBB).  `detect()` returns the same match
            # dicts the template matcher produces, so everything downstream —
            # per-target measurement, coordinate mapping, debug capture — stays
            # on one path.  It also carries `box_size`, because unlike a template
            # each instance has its own extent.
            if self._debug_dir and self._alignment.template_image is not None:
                self._debug_save("reference.png", self._alignment.template_image)
            matches = self._alignment.detect(inspection_image)
            if not matches:
                logger.info("ObbAlignment detected no targets")
                return []
        elif self._alignment.template_point is None:
            centre_row, centre_col = self._alignment.box_center
            height, width = self._alignment.box_size
            if height <= 0 or width <= 0:
                raise RuntimeError(
                    "No template defined and no ROI taught. Call teach_template() "
                    "first, or use an alignment mode that stores a fixed ROI."
                )
            # Score is None, not 1.0. Nothing was searched, so there is no
            # correlation value to report; claiming a perfect match made every
            # image look located (a blank frame scored exactly like the
            # reference) and let any downstream threshold pass silently.
            # None is honest *and* JSON-safe, unlike NaN which breaks
            # json.dumps(allow_nan=False) on every API response.
            matches = [{
                "matched_row": centre_row,
                "matched_col": centre_col,
                "match_score": None,
                "best_rotation_deg": 0.0,
                "best_scale": 1.0,
                "valid": True,
            }]
        else:
            # Debug: save template image used for matching
            if self._debug_dir and self._alignment.template_image is not None:
                self._debug_save("template.png", self._alignment.template_image)

            # Step 1: Multi-target matching
            match_result = self._alignment.template_point.measure(inspection_image)
            matches = match_result.get("matches", [])

            if not matches:
                # Try single-target result
                if match_result.get("valid", False):
                    matches = [match_result]
                else:
                    return []

        # Step 2: Process each match
        t_match_end = time.perf_counter()
        logger.info("[TIMING] MultiTargetWorkflow.measure: template_matching=%+.1fms  n_matches=%d",
                     (t_match_end - t0) * 1000, len(matches))
        for i, m in enumerate(matches):
            target_result = self._measure_one_target(inspection_image, m, i)
            self._results.append(target_result)
        logger.info("[TIMING] MultiTargetWorkflow.measure: per_target_done=%+.1fms  target=%d",
                     (time.perf_counter() - t_match_end) * 1000, i)

        # Debug: save inspection image with all matched target boxes
        if self._debug_dir:
            self._debug_save_inspection_boxes(inspection_image)

        return self._results

    def _measure_one_target(
        self,
        inspection_image: np.ndarray,
        match: Dict[str, Any],
        index: int,
    ) -> TargetResult:
        """Run all measurements on a single matched target.

        Rather than using MeasurementWorkflow (which requires localization
        calibration and complex dependency resolution), we directly:
        1. Create and run each measurement object on the straightened patch
        2. Compute composed measurements from the collected results
        3. Map point results back to inspection image coordinates
        """
        matched_row = match["matched_row"]
        matched_col = match["matched_col"]
        match_score = match.get("match_score", 0.0)
        rotation_deg = match.get("best_rotation_deg", 0.0)
        scale = match.get("best_scale", 1.0)
        valid_match = match.get("valid", True)
        # Detector-based locators report an extent per instance; template
        # matches do not and fall back to the strategy's taught box.
        box_size = match.get("box_size")

        if not valid_match:
            # Even for invalid matches, store the absolute rotation for correct
            # box drawing in debug visualizations.
            absolute_rotation = self._alignment.box_angle_deg - rotation_deg
            return TargetResult(
                id=index + 1,
                score=match_score,
                rotation_deg=absolute_rotation,
                scale=scale,
                center_row=matched_row,
                center_col=matched_col,
                valid=False,
                meta={"reason": "match below threshold"},
            )

        # Delegate alignment to the strategy (handles both rigid and
        # affine-refined alignment modes transparently).
        align_result = self._alignment.align(
            inspection_image, matched_row, matched_col, rotation_deg,
            box_size=box_size,
        )
        patch = align_result.patch
        M_inv = align_result.M_inv  # 2x3 affine matrix
        t_align = time.perf_counter()

        # Debug: save straightened patch
        self._debug_save(f"target_{index + 1:02d}_patch.png", patch)

        # Run measurements directly on the straightened patch
        raw_results: Dict[str, Any] = {}  # label -> GeometricResult
        measurement_objects: Dict[str, Any] = {}  # label -> MeasureObject (for debug profile)
        all_valid = True
        t_meas_start = time.perf_counter()

        try:
            for d in self._measurement_defs:
                obj_type = d["object_type"]
                label = d["label"]
                params = d["params"]

                if obj_type in ("EdgePoint", "EdgePair", "FitLine", "FitCircle"):
                    # Primitive: create object and measure on patch
                    factory, _ = _OBJECT_FACTORIES[obj_type]
                    obj = factory(label, **params)
                    result = obj.measure(patch)
                    measurement_objects[label] = obj

                    if obj_type == "EdgePair" and result.valid:
                        # Store two virtual endpoints instead of center
                        pa = obj.point_a
                        pb = obj.point_b
                        raw_results[label + "_A"] = pa
                        raw_results[label + "_B"] = pb
                        measurement_objects[label + "_A"] = _VirtualPoint(label + "_A", pa)
                        measurement_objects[label + "_B"] = _VirtualPoint(label + "_B", pb)
                    else:
                        raw_results[label] = result
                        if not result.valid:
                            all_valid = False

                elif obj_type == "TemplateMatchPoint":
                    # Template-matching point: use pre-built TemplatePoint
                    tp = self._template_match_points.get(label)
                    if tp is not None:
                        match_result = tp.measure(patch)
                        from measure.measure_workflow import PointResult
                        result = PointResult(
                            label=label,
                            row=match_result["matched_row"],
                            col=match_result["matched_col"],
                            valid=match_result["valid"],
                            meta={
                                "match_score": match_result["match_score"],
                                "dx": match_result["dx"],
                                "dy": match_result["dy"],
                                "teach_row": params["row"],
                                "teach_col": params["col"],
                                "best_rotation_deg": match_result.get(
                                    "best_rotation_deg", 0.0
                                ),
                            },
                        )
                    else:
                        result = PointResult(
                            label=label,
                            row=params["row"],
                            col=params["col"],
                            valid=False,
                            meta={"reason": "TemplateMatchPoint not taught"},
                        )
                    raw_results[label] = result
                    if not result.valid:
                        all_valid = False

                elif obj_type == "TwoPointsLine":
                    p1 = raw_results.get(params["point_a_label"])
                    p2 = raw_results.get(params["point_b_label"])
                    if p1 and p2 and p1.valid and p2.valid:
                        dr = p2.row - p1.row
                        dc = p2.col - p1.col
                        norm = np.sqrt(dr**2 + dc**2)
                        if norm > EPS:
                            from measure.measure_workflow import LineResult
                            a = dc / norm
                            b = -dr / norm
                            c = -(a * p1.row + b * p1.col)
                            result = LineResult(
                                label=label, a=a, b=b, c=c,
                                start_row=p1.row, start_col=p1.col,
                                end_row=p2.row, end_col=p2.col,
                                valid=True,
                                meta={"point_a_label": params["point_a_label"],
                                      "point_b_label": params["point_b_label"],
                                      "length": norm},
                            )
                        else:
                            from measure.measure_workflow import LineResult
                            result = LineResult(label=label, valid=False,
                                               meta={"reason": "coincident points"})
                    else:
                        from measure.measure_workflow import LineResult
                        result = LineResult(label=label, valid=False,
                                           meta={"reason": "input points invalid"})
                    raw_results[label] = result

                elif obj_type == "TwoPointsDistance":
                    p1 = raw_results.get(params["point_a_label"])
                    p2 = raw_results.get(params["point_b_label"])
                    if p1 and p2 and p1.valid and p2.valid:
                        dr = p2.row - p1.row
                        dc = p2.col - p1.col
                        dist = np.sqrt(dr**2 + dc**2)
                        from measure.measure_workflow import DistanceResult
                        result = DistanceResult(label=label, value=dist, valid=True)
                    else:
                        from measure.measure_workflow import DistanceResult
                        result = DistanceResult(label=label, value=0.0, valid=False)
                    raw_results[label] = result

                elif obj_type == "PointLineDistance":
                    pt = raw_results.get(params["point_label"])
                    line = raw_results.get(params["line_label"])
                    if (pt and line and pt.valid and line.valid and
                        hasattr(line, "a") and hasattr(line, "b") and hasattr(line, "c")):
                        dist = abs(line.a * pt.row + line.b * pt.col + line.c) / \
                               np.sqrt(line.a**2 + line.b**2 + EPS)
                        from measure.measure_workflow import DistanceResult
                        result = DistanceResult(label=label, value=dist, valid=True)
                    else:
                        from measure.measure_workflow import DistanceResult
                        result = DistanceResult(label=label, value=0.0, valid=False)
                    raw_results[label] = result

                elif obj_type == "TwoLinesAngle":
                    l1 = raw_results.get(params["line_a_label"])
                    l2 = raw_results.get(params["line_b_label"])
                    if (l1 and l2 and l1.valid and l2.valid and
                        hasattr(l1, "start_row") and hasattr(l2, "start_row")):
                        # Compute angle between two lines
                        dr1 = l1.end_row - l1.start_row
                        dc1 = l1.end_col - l1.start_col
                        dr2 = l2.end_row - l2.start_row
                        dc2 = l2.end_col - l2.start_col
                        a1 = np.arctan2(dc1, dr1)
                        a2 = np.arctan2(dc2, dr2)
                        angle_rad = a2 - a1
                        # Normalize to [-pi/2, pi/2]
                        while angle_rad > np.pi / 2:
                            angle_rad -= np.pi
                        while angle_rad < -np.pi / 2:
                            angle_rad += np.pi
                        from measure.measure_workflow import AngleResult
                        result = AngleResult(label=label,
                                            value_rad=abs(angle_rad), valid=True)
                    else:
                        from measure.measure_workflow import AngleResult
                        result = AngleResult(label=label, value_rad=0.0, valid=False)
                    raw_results[label] = result

                elif obj_type == "PointCircleDistance":
                    pt = raw_results.get(params["point_label"])
                    circ = raw_results.get(params["circle_label"])
                    if (pt and circ and pt.valid and circ.valid and
                        hasattr(circ, "center_row") and hasattr(circ, "radius")):
                        dr = pt.row - circ.center_row
                        dc = pt.col - circ.center_col
                        dist = abs(np.sqrt(dr**2 + dc**2) - circ.radius)
                        from measure.measure_workflow import DistanceResult
                        result = DistanceResult(label=label, value=dist, valid=True)
                    else:
                        from measure.measure_workflow import DistanceResult
                        result = DistanceResult(label=label, value=0.0, valid=False)
                    raw_results[label] = result

        except Exception as e:
            all_valid = False
            raw_results["_error"] = str(e)
        t_meas_end = time.perf_counter()
        logger.info("[TIMING]   target=%d: align=%+.1fms  measurements=%+.1fms  patch=%dx%d",
                     index,
                     (t_meas_start - t_align) * 1000,
                     (t_meas_end - t_meas_start) * 1000,
                     patch.shape[1], patch.shape[0])

        # Debug: save patch with measurement overlays
        if self._debug_dir:
            debug_vis = self._debug_draw_measurements_on_patch(patch, raw_results)
            valid_str = "OK" if all_valid else "FAIL"
            self._debug_save(
                f"target_{index + 1:02d}_measured_{valid_str}.png", debug_vis
            )
            # 保存 profile 调试图
            self._save_profile_debug_images(index, patch, measurement_objects)

        # Map point results back to original image coordinates
        measurements = {}
        for label, result in raw_results.items():
            if label == "_error":
                measurements["_error"] = result
            else:
                measurements[label] = _map_result_to_original(result, M_inv)

        return TargetResult(
            id=index + 1,
            score=match_score,
            rotation_deg=self._alignment.box_angle_deg - rotation_deg,
            scale=scale,
            center_row=matched_row,
            center_col=matched_col,
            # With no tools defined `all_valid` stays at its True initial value,
            # which is the intended reading (see TargetResult.valid): validity is
            # then decided purely by the locator.  Anchored here because an
            # unrelated change to that initialiser would silently invert it.
            valid=valid_match and all_valid,
            measurements=measurements,
            meta={
                "match_score": match_score,
                "patch_shape": patch.shape,
                "M_inv": M_inv,
                "localization_only": not self._measurement_defs,
            },
        )

    def _build_workflow(self) -> MeasurementWorkflow:
        """Build a MeasurementWorkflow from stored definitions."""
        wf = MeasurementWorkflow()
        for d in self._measurement_defs:
            obj_type = d["object_type"]
            label = d["label"]
            params = d["params"]

            factory, _ = _OBJECT_FACTORIES[obj_type]
            obj = factory(label, **params)
            wf.add(obj)

        return wf

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def visualize(
        self,
        inspection_image: Optional[np.ndarray] = None,
        color_palette: Optional[List[Tuple[int, int, int]]] = None,
    ) -> np.ndarray:
        """
        Render all matched targets and their measurements on the inspection image.

        Args:
            inspection_image: Inspection image. Uses last measure() image if None.
            color_palette: List of BGR colors for each target.

        Returns:
            Annotated BGR image.
        """
        if inspection_image is None:
            if self._last_inspection_image is None:
                raise ValueError("No inspection image available.")
            inspection_image = self._last_inspection_image

        vis = to_bgr(inspection_image)

        if color_palette is None:
            color_palette = [
                (0, 255, 0),     # green
                (0, 255, 255),   # yellow
                (255, 0, 255),   # magenta
                (255, 255, 0),   # cyan
                (0, 165, 255),   # orange
                (255, 0, 0),     # blue
                (128, 0, 128),   # dark magenta
                (0, 128, 128),   # dark yellow
            ]

        for target in self._results:
            color = color_palette[target.id % len(color_palette)]

            # Draw rotated box around matched target
            if target.valid:
                corners = _compute_target_box_corners(target, self._alignment.box_size)
                pts = corners.astype(np.int32).reshape((-1, 1, 2))
                # Convert from (row, col) to (x, y) for OpenCV
                pts_xy = pts[:, :, ::-1]
                cv2.polylines(vis, [pts_xy], True, color, thickness=2)

                # Label the target
                label_text = (f"T#{target.id} s={_fmt_score(target.score)} "
                        f"r={target.rotation_deg:.1f}")
                cv2.putText(
                    vis,
                    label_text,
                    (int(target.center_col) + 5, int(target.center_row) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                    cv2.LINE_AA,
                )

                # Draw measurement results for this target
                for label, result in target.measurements.items():
                    if isinstance(result, dict):
                        _draw_result_on_image(vis, result, color)
                    elif hasattr(result, "valid") and result.valid:
                        _draw_geometric_result(vis, result, color)

        # Draw summary at bottom
        summary = (
            f"MultiTarget: {len(self._results)} targets, "
            f"{sum(1 for t in self._results if t.valid)} valid"
        )
        cv2.putText(
            vis,
            summary,
            (10, vis.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return vis

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def _manifest(self, model_id: str = "",
                  object_ids: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
        """Build the manifest payload for this workflow.

        Args:
            model_id: Optional platform identity for the model.
            object_ids: Optional ``{label: object_id}`` mapping so a caller that
                already minted object ids (the v2 modelling API does) keeps them
                across publish/measure.  When omitted, ids are derived from the
                labels so that saving is deterministic.

        Returns:
            JSON-serializable manifest body (without the ``files`` digests).
        """
        tp = self._alignment.template_point
        if self._alignment._reference_image is None:
            raise RuntimeError(
                "No reference image available on the alignment strategy; "
                "teach_template() must have been called with one."
            )

        spec_deps = {}
        objects = []
        for definition in self._measurement_defs:
            object_type = definition["object_type"]
            label = definition["label"]
            params = dict(definition["params"])
            spec = get_spec(object_type)
            dep_names = spec.dep_param_names
            input_labels: List[str] = []
            input_ids: List[str] = []
            for name in dep_names:
                target_label = params.pop(name, None)
                if target_label is None:
                    continue
                target_label = str(target_label)
                # Dependencies are labels, not necessarily objects: an EdgePair
                # exposes virtual endpoints named "<label>_A"/"<label>_B" which
                # composed tools may reference.  The label form is authoritative
                # (it round-trips those virtual endpoints); the id form is
                # derived for consumers that build an object graph.
                input_labels.append(target_label)
                input_ids.append(_stable_object_id(target_label))
            objects.append({
                "object_id": (
                    str(object_ids[label]) if object_ids and label in object_ids
                    else _stable_object_id(label)
                ),
                "object_type": object_type,
                "label": label,
                "parameters": params,
                "input_object_ids": input_ids,
                "input_labels": input_labels,
            })

        reference = np.ascontiguousarray(self._alignment._reference_image)
        return {
            "contract_version": MANIFEST_CONTRACT_VERSION,
            "model_id": model_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "reference_image": {
                "file": REFERENCE_FILENAME,
                "shape": [int(reference.shape[0]), int(reference.shape[1])],
            },
            "alignment": {
                "mode": self._alignment.mode_name,
                "roi_state": self._alignment.get_roi_state(),
                "box_center": list(self._alignment.box_center),
                "box_size": list(self._alignment.box_size),
                "box_angle_deg": self._alignment.box_angle_deg,
                "template_size": self._alignment._template_size,
                "preprocessor": (tp.preprocessor.serialize() if tp is not None
                                 else None),
            },
            # Read from the template point — that is what actually runs at match
            # time.  The workflow-level attributes are only a cache:
            # teach_template() forwards its parameters to the alignment
            # strategy, so trusting the cache would persist whatever defaults
            # the widget never touched.
            #
            # Search-free modes (``none``, ``fixed_roi``) have no template and
            # therefore no matching behaviour to record; an empty dict says
            # "nothing was searched" honestly instead of storing defaults that
            # would look like real tuning.
            "matching": ({} if tp is None else {
                "match_score_threshold": float(tp.match_score_threshold),
                "angle_range": list(tp.angle_range),
                "angle_step": float(tp.angle_step),
                "max_matches": int(tp.max_matches),
                "overlap": float(tp.overlap),
                "coarse_fine": bool(tp.coarse_fine),
                "coarse_angle_step": float(tp.coarse_angle_step),
                "pyramid_decimate": tp.pyramid_decimate,
                "pyramid_max_template_size": tp.pyramid_max_template_size,
            }),
            "physical_scale_mm": self._physical_scale_mm,
            "objects": objects,
            "derived_fingerprint": self._derived_fingerprint(),
        }

    def _derived_fingerprint(self) -> str:
        """Hash of everything the derived template state depends on.

        The template image and edge template are fully determined by the
        reference image, the ROI geometry and the preprocessor, so this
        fingerprint pins the cache in ``workflow.npz`` to its inputs.
        """
        tp = self._alignment.template_point
        payload = {
            "reference": _sha256_bytes(
                np.ascontiguousarray(self._alignment._reference_image).tobytes()
            ),
            "box_center": list(self._alignment.box_center),
            "box_size": list(self._alignment.box_size),
            "box_angle_deg": self._alignment.box_angle_deg,
            "template_size": self._alignment._template_size,
            "mode": self._alignment.mode_name,
            "roi_state": self._alignment.get_roi_state(),
            "preprocessor": tp.preprocessor.serialize() if tp else None,
        }
        return _sha256_bytes(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        )

    def save(self, path: str, model_id: str = "",
             object_ids: Optional[Mapping[str, str]] = None) -> None:
        """Write this workflow as a manifest model asset.

        Args:
            path: Model directory, or a ``.npz`` file path inside one.  A
                ``.npz`` path is accepted so existing callers that saved to
                ``<model_dir>/workflow.npz`` keep working: its parent directory
                becomes the model directory.
            model_id: Optional platform identity recorded in the manifest.
            object_ids: Optional ``{label: object_id}`` mapping preserved in the
                manifest (see :meth:`_manifest`).

        The asset directory receives:

        * ``manifest.json`` — authoritative model definition
        * ``reference.npy`` — the raw reference image
        * ``workflow.npz`` — **derived cache only** (template image and edge
          template).  Loading always re-derives this state from the manifest,
          so the cache is never required and can be deleted safely.
        """
        if self._alignment._reference_image is None:
            raise RuntimeError(
                "Nothing taught yet: call teach_template() before saving.  "
                "A model with no reference image cannot be reconstructed."
            )

        target = Path(path)
        if target.suffix == ".npz":
            model_dir = target.parent
            derived_path = target
        else:
            model_dir = target
            derived_path = model_dir / DERIVED_FILENAME

        model_dir.mkdir(parents=True, exist_ok=True)

        reference = np.ascontiguousarray(self._alignment._reference_image)
        reference_path = model_dir / REFERENCE_FILENAME
        np.save(str(reference_path), reference, allow_pickle=False)

        tp = self._alignment.template_point
        if tp is None:
            # Search-free modes derive nothing from a template, so there is no
            # cache to write.  The manifest already declares the cache optional
            # ("derived cache only, safe to delete"), so omitting it entirely is
            # consistent — and writing an npz without edge_template would make
            # the loader's cache path disagree with its own contract.
            if derived_path.exists():
                derived_path.unlink()
            manifest = self._manifest(model_id=model_id, object_ids=object_ids)
            manifest["files"] = {REFERENCE_FILENAME: _sha256_file(reference_path)}
        else:
            np.savez_compressed(
                str(derived_path),
                edge_template=tp.edge_template,
                template_image=self._alignment.template_image,
            )
            manifest = self._manifest(model_id=model_id, object_ids=object_ids)
            manifest["files"] = {
                REFERENCE_FILENAME: _sha256_file(reference_path),
                derived_path.name: _sha256_file(derived_path),
            }
        # Same digest as files[reference.npy] — kept here too so a reader can
        # verify the reference image without knowing the asset's file layout.
        manifest["reference_image"]["sha256"] = manifest["files"][REFERENCE_FILENAME]
        (model_dir / MANIFEST_FILENAME).write_text(
            json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str) -> "MultiTargetWorkflow":
        """Load a workflow from a manifest model asset.

        Args:
            path: Model directory, its ``manifest.json``, or a ``.npz`` file
                inside the model directory.

        Returns:
            A fully reconstructed MultiTargetWorkflow ready for ``measure()``.

        Raises:
            ValueError: When the path holds a legacy single-file ``.npz``
                (written before the manifest format) or a v2 engine asset.
                Run ``scripts/migrate_models_to_manifest.py`` to convert it.

        The template image and edge template are re-derived from the reference
        image plus the manifest (they are deterministic), so ``workflow.npz``
        is never required.
        """
        target = Path(path)
        model_dir = target if target.is_dir() else target.parent
        manifest_path = (target if target.name == MANIFEST_FILENAME
                         else model_dir / MANIFEST_FILENAME)

        if not manifest_path.is_file():
            legacy_hint = ""
            candidate = target if target.suffix == ".npz" else model_dir / DERIVED_FILENAME
            if candidate.is_file():
                legacy_hint = (
                    f" Found legacy workflow file '{candidate}'. Legacy single-file "
                    f".npz models are no longer supported; convert it with "
                    f"scripts/migrate_models_to_manifest.py."
                )
            raise ValueError(
                f"No manifest model asset at '{manifest_path}'.{legacy_hint}"
            )

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        contract = manifest.get("contract_version")
        if contract not in COMPATIBLE_MANIFEST_CONTRACT_VERSIONS:
            raise ValueError(
                f"Unsupported manifest contract_version: {contract!r}. "
                f"Supported: {', '.join(COMPATIBLE_MANIFEST_CONTRACT_VERSIONS)}."
            )

        reference_path = model_dir / manifest["reference_image"]["file"]
        if not reference_path.is_file():
            raise ValueError(
                f"Model asset is incomplete: '{reference_path}' is missing."
            )
        expected_sha = manifest["reference_image"].get("sha256")
        actual_sha = _sha256_file(reference_path)
        if expected_sha and actual_sha != expected_sha:
            raise ValueError(
                f"Model asset is corrupt: '{reference_path}' sha256 mismatch "
                f"(expected {expected_sha}, got {actual_sha})."
            )

        reference = np.load(str(reference_path), allow_pickle=False)

        alignment = manifest["alignment"]
        matching = manifest.get("matching") or {}

        wf = cls()
        wf._physical_scale_mm = manifest.get("physical_scale_mm")
        strategy = strategy_from_roi_state(
            alignment.get("roi_state", {}), reference
        )
        wf._alignment = strategy
        # Search-free modes store an empty ``matching`` block and must stay that
        # way.  Routing them through teach_template() would hand the base
        # implementation's TemplatePoint constructor a None preprocessor (it
        # defaults to RawPreprocessor, silently changing semantics) and would
        # rebuild exactly the matcher these modes exist to avoid.
        if isinstance(strategy, (NoneAlignment, FixedRoiAlignment)):
            wf._teach_without_search(reference, alignment)
        elif isinstance(strategy, ObbAlignment):
            # A detector-backed asset must NOT be routed through teach_template():
            # `matching` is empty for these, so rebuilding a template search would
            # raise on the missing keys — or worse, silently produce a model that
            # searches where it was taught to detect.  Reattach the stored
            # geometry only; the detector itself is an injected capability and is
            # supplied by the caller after load (see ObbAlignment.teach).
            wf._teach_obb(reference, alignment)
        else:
            wf.teach_template(
                reference,
                center=tuple(alignment["box_center"]),
                size=tuple(alignment["box_size"]),
                angle_deg=float(alignment["box_angle_deg"]),
                preprocessor=_deserialize_preprocessor(alignment["preprocessor"]),
                match_score_threshold=float(matching["match_score_threshold"]),
                angle_range=tuple(matching["angle_range"]),
                angle_step=float(matching["angle_step"]),
                max_matches=int(matching["max_matches"]),
                overlap=float(matching["overlap"]),
                coarse_fine=bool(matching["coarse_fine"]),
                coarse_angle_step=float(matching["coarse_angle_step"]),
                pyramid_decimate=int(matching["pyramid_decimate"]),
                pyramid_max_template_size=int(matching["pyramid_max_template_size"]),
            )

        # Translate the manifest's dependency references back to label form.
        label_by_id = {obj["object_id"]: obj["label"] for obj in manifest["objects"]}
        defs: List[Dict[str, Any]] = []
        for obj in manifest["objects"]:
            object_type = obj["object_type"]
            params = dict(obj.get("parameters", {}))
            dep_names = get_spec(object_type).dep_param_names
            if "input_labels" in obj:
                # authoritative: preserves virtual EdgePair endpoints
                labels = list(obj["input_labels"])
            else:
                labels = [label_by_id.get(i, i)
                          for i in obj.get("input_object_ids", [])]
            for name, target_label in zip(dep_names, labels):
                params[name] = target_label
            defs.append({
                "object_type": object_type,
                "label": obj["label"],
                "params": params,
            })
        wf._measurement_defs = defs
        wf._build_template_match_points()
        return wf

    @classmethod
    def load_legacy(cls, filepath: str) -> "MultiTargetWorkflow":
        """Load a legacy single-file ``.npz`` workflow (pre-manifest format).

        Kept solely so ``scripts/migrate_models_to_manifest.py`` can read old
        assets.  New code should use :meth:`load`.
        """
        data = np.load(filepath, allow_pickle=True)

        meta_raw = data["workflow_meta"]
        if isinstance(meta_raw, np.ndarray) and meta_raw.dtype == np.object_:
            meta = json.loads(str(meta_raw[0]))
        else:
            meta = json.loads(str(meta_raw))

        if meta.get("version", 0) != 1:
            raise ValueError(
                f"Unsupported project file version: {meta.get('version')}. "
                f"Expected version 1."
            )

        wf = cls()

        # Restore matching params
        wf._match_score_threshold = meta["match_score_threshold"]
        wf._angle_range = tuple(meta["angle_range"])
        wf._angle_step = meta["angle_step"]
        wf._max_matches = meta["max_matches"]
        wf._overlap = float(meta.get("overlap", 0.3))
        wf._coarse_fine = meta["coarse_fine"]
        wf._coarse_angle_step = meta["coarse_angle_step"]

        # Restore preprocessor
        preprocessor = _deserialize_preprocessor(meta["preprocessor_data"])

        # Restore template image and reference image
        template_image = None
        reference_image = None
        if "template_image" in data:
            template_image = data["template_image"]
        if "reference_image" in data:
            reference_image = data["reference_image"]

        # Restore alignment strategy
        alignment_mode = meta.get("alignment_mode", "single_box")
        if alignment_mode == "single_box" or "alignment_roi_state" not in meta:
            wf._alignment = SingleBoxAlignment()
        else:
            roi_state = meta["alignment_roi_state"]
            wf._alignment = strategy_from_roi_state(roi_state, reference_image)

        # Set alignment box params
        wf._alignment._box_center = tuple(meta["box_center"])
        wf._alignment._box_size = tuple(meta["box_size"])
        wf._alignment._box_angle_deg = meta["box_angle_deg"]
        wf._alignment._template_size = meta["template_size"]
        wf._alignment._template_image = template_image
        wf._alignment._reference_image = reference_image

        # Reconstruct TemplatePoint
        tp = TemplatePoint.__new__(TemplatePoint)
        tp.click_row = meta["click_row"]
        tp.click_col = meta["click_col"]
        tp.template_size = meta["template_size"]
        tp.match_score_threshold = meta["match_score_threshold"]
        tp.use_subpixel = True
        tp.edge_template = data["edge_template"]
        tp._crop_center_row = meta["crop_center_row"]
        tp._crop_center_col = meta["crop_center_col"]
        tp._crop_h = meta["crop_h"]
        tp._crop_w = meta["crop_w"]
        tp._actual_crop_bounds = (0, tp._crop_h, 0, tp._crop_w)
        tp.preprocessor = preprocessor
        tp.rotation_invariant = True
        tp.angle_range = wf._angle_range
        tp.angle_step = wf._angle_step
        tp.scale_invariant = False
        tp.scale_range = (0.9, 1.1)
        tp.scale_step = 0.02
        tp.coarse_fine = wf._coarse_fine
        tp.coarse_angle_step = wf._coarse_angle_step
        tp.coarse_scale_step = 0.1
        tp.multi_target = True
        tp.max_matches = wf._max_matches
        tp.overlap = wf._overlap
        tp.result = None
        # 初始化 pyramid 相关属性
        tp.pyramid_decimate = meta.get("pyramid_decimate", 0)
        tp.pyramid_max_template_size = meta.get("pyramid_max_template_size", 400)
        tp._pyramid_scale = 1.0  # 由 calc_pyramid_scale 延迟计算
        if tp.pyramid_decimate > 0 and tp.pyramid_max_template_size > 0:
            max_dim = max(tp._crop_h, tp._crop_w)
            tp._pyramid_scale = min(1.0, tp.pyramid_max_template_size / float(max_dim))
        wf._alignment._template_point = tp

        # For multi-point alignment, rebuild control point templates
        if alignment_mode == "multi_point" and template_image is not None:
            wf._alignment.build_all_control_point_templates()

        # Restore measurement definitions
        wf._measurement_defs = meta.get("measurement_defs", [])

        # Rebuild TemplateMatchPoint instances from restored template image
        if wf._alignment.template_image is not None:
            wf._build_template_match_points()

        return wf

    def summary_text(self) -> str:
        """
        Generate a human-readable summary of all results.

        This corresponds to the user's requirement Step 5:
        "在一个文本框输出每个目标的测量结果"
        """
        lines = []
        lines.append("=" * 60)
        lines.append("测量结果汇总 (Multi-Target Measurement Results)")
        lines.append("=" * 60)

        if not self._results:
            lines.append("(无结果 / No results)")
            return "\n".join(lines)

        lines.append(f"共检测到 {len(self._results)} 个目标")
        lines.append(f"其中有效目标: {sum(1 for t in self._results if t.valid)}")
        lines.append("")

        for target in self._results:
            lines.append(
                f"[Target #{target.id}] "
                f"score={_fmt_score(target.score)}, "
                f"rotation={target.rotation_deg:.2f}°, "
                f"center=({target.center_row:.1f}, {target.center_col:.1f}), "
                f"status={'✓ 有效' if target.valid else '✗ 无效'}"
            )

            if not target.measurements:
                lines.append("  (无测量项)")
                continue

            for label, result in target.measurements.items():
                if isinstance(result, dict):
                    if label == "_error":
                        lines.append(f"  [ERROR] {result}")
                    else:
                        lines.append(_format_result_dict(
                            label, result, scale_mm=self.physical_scale_mm))
                elif hasattr(result, "valid"):
                    lines.append(_format_geometric_result(
                        label, result, scale_mm=self.physical_scale_mm))
                else:
                    lines.append(f"  {label}: {result}")

            lines.append("")

        return "\n".join(lines)


# Helpers
# ===========================================================================


def _references_label(defn: Dict[str, Any], target_label: str) -> bool:
    """Check if a composed measurement references the given label
    or its virtual EdgePair endpoints (label_A, label_B)."""
    params = defn.get("params", {})
    for key in ["point_a_label", "point_b_label", "point_label",
                "line_a_label", "line_b_label", "line_label",
                "circle_label"]:
        val = params.get(key, "")
        if val == target_label:
            return True
        # Also match virtual EdgePair endpoints (label_A, label_B)
        if val in (target_label + "_A", target_label + "_B"):
            return True
    return False



def _compute_target_box_corners(
    target: TargetResult,
    size: Tuple[float, float],
) -> np.ndarray:
    """Compute the 4 corners of a target's rotated box in image coordinates."""
    from measure.rectify import compute_rotated_box_corners

    return compute_rotated_box_corners(
        (target.center_row, target.center_col),
        size,
        -target.rotation_deg,
    )


def _map_result_to_original(result, M_inv: np.ndarray):
    """Map a GeometricResult's point coordinates back to the original image.

    Returns a dict (serializable) instead of modifying the GeometricResult
    (which is a frozen dataclass).

    Handles both 2x3 affine matrices (from :class:`AlignmentStrategy`) and
    legacy 3x3 rigid matrices (from :func:`crop_and_straighten`).
    """
    from measure.measure_workflow import (
        CircleResult,
        DistanceResult,
        LineResult,
        PointResult,
    )

    # Pick the right mapper based on M_inv shape
    if M_inv.shape == (2, 3):
        mapper = map_point_via_affine
    else:
        mapper = map_point_to_original

    if isinstance(result, PointResult):
        if result.valid:
            orig_row, orig_col = mapper(
                (result.row, result.col), M_inv
            )
            return {
                "type": "point",
                "label": result.label,
                "valid": True,
                "row": orig_row,
                "col": orig_col,
                "meta": result.meta,
            }
        return {"type": "point", "label": result.label, "valid": False}

    elif isinstance(result, LineResult):
        if result.valid:
            sr, sc = mapper(
                (result.start_row, result.start_col), M_inv
            )
            er, ec = mapper(
                (result.end_row, result.end_col), M_inv
            )
            return {
                "type": "line",
                "label": result.label,
                "valid": True,
                "start_row": sr, "start_col": sc,
                "end_row": er, "end_col": ec,
                "a": result.a, "b": result.b, "c": result.c,
                "meta": result.meta,
            }
        return {"type": "line", "label": result.label, "valid": False}

    elif isinstance(result, CircleResult):
        if result.valid:
            cr, cc = mapper(
                (result.center_row, result.center_col), M_inv
            )
            return {
                "type": "circle",
                "label": result.label,
                "valid": True,
                "center_row": cr, "center_col": cc,
                "radius": result.radius,
                "meta": result.meta,
            }
        return {"type": "circle", "label": result.label, "valid": False}

    elif isinstance(result, DistanceResult):
        return {
            "type": "distance",
            "label": result.label,
            "valid": result.valid,
            "value": result.value,
        }

    else:  # AngleResult etc.
        return {
            "type": result.type,
            "label": result.label,
            "valid": result.valid,
            "value": getattr(result, "value", None),
            "value_deg": getattr(result, "value_deg", None),
        }


def _draw_result_on_image(
    vis: np.ndarray,
    result: dict,
    color: Tuple[int, int, int],
):
    """Draw a mapped result dict onto the visualization image."""
    rtype = result.get("type", "")
    if not result.get("valid", False):
        return

    if rtype == "point":
        pt = (int(round(result["col"])), int(round(result["row"])))
        cv2.drawMarker(
            vis, pt, color, markerType=cv2.MARKER_CROSS,
            markerSize=8, thickness=1,
        )

    elif rtype == "line":
        pt1 = (int(round(result["start_col"])), int(round(result["start_row"])))
        pt2 = (int(round(result["end_col"])), int(round(result["end_row"])))
        cv2.line(vis, pt1, pt2, color, thickness=1)

    elif rtype == "circle":
        ct = (int(round(result["center_col"])), int(round(result["center_row"])))
        r = int(round(result["radius"]))
        cv2.circle(vis, ct, r, color, thickness=1)

    elif rtype in ("distance", "angle"):
        pass  # Distance/angle don't have inherent geometry to draw


def _draw_geometric_result(
    vis: np.ndarray,
    result,
    color: Tuple[int, int, int],
):
    """Draw a GeometricResult onto the visualization image."""
    # This is called when results haven't been mapped yet (fallback)
    if not result.valid:
        return
    if hasattr(result, "row") and hasattr(result, "col"):
        pt = (int(round(result.col)), int(round(result.row)))
        cv2.drawMarker(
            vis, pt, color, markerType=cv2.MARKER_CROSS,
            markerSize=8, thickness=1,
        )
    # 绘制圆的最长半径和最短半径线
    if hasattr(result, "type") and result.type == "circle":
        center = (int(round(result.center_col)), int(round(result.center_row)))
        radius = int(round(result.radius))
        # 绘制圆
        cv2.circle(vis, center, radius, color, 2, cv2.LINE_AA)
        # 绘制最长半径和最短半径线
        meta = getattr(result, "meta", {})
        max_radius_point = meta.get("max_radius_point")
        min_radius_point = meta.get("min_radius_point")
        max_radius = meta.get("max_radius", 0)
        min_radius = meta.get("min_radius", 0)
        if max_radius_point:
            max_pt = (int(round(max_radius_point[0])), int(round(max_radius_point[1])))
            cv2.line(vis, center, max_pt, (0, 0, 255), 2, cv2.LINE_AA)  # 红色
            cv2.circle(vis, max_pt, 4, (0, 0, 255), -1)
            # 显示最长半径值
            max_label = f"Rmax={max_radius:.1f}"
            max_label_pos = ((center[0] + max_pt[0]) // 2, (center[1] + max_pt[1]) // 2 - 10)
            cv2.putText(vis, max_label, max_label_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        if min_radius_point:
            min_pt = (int(round(min_radius_point[0])), int(round(min_radius_point[1])))
            cv2.line(vis, center, min_pt, (255, 0, 0), 2, cv2.LINE_AA)  # 蓝色
            cv2.circle(vis, min_pt, 4, (255, 0, 0), -1)
            # 显示最短半径值
            min_label = f"Rmin={min_radius:.1f}"
            min_label_pos = ((center[0] + min_pt[0]) // 2, (center[1] + min_pt[1]) // 2 + 15)
            cv2.putText(vis, min_label, min_label_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
        # 显示椭圆度
        ellipticity = meta.get("ellipticity", 0)
        if ellipticity > 0:
            ell_label = f"Ellipticity={ellipticity:.1f}"
            cv2.putText(vis, ell_label, (center[0] + 5, center[1] + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
