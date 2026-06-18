"""
Unified composable measurement workflow system.

This module implements a "teach once, measure many" workflow where all
measurement objects share a common interface and can be freely composed
to form higher-level geometric measurements.

Architecture:
  MeasureObject (ABC)
    ├── Primitive (measure directly from image pixels)
    │   ├── TemplatePointObject   → Point
    │   ├── EdgePointObject       → Point
    │   ├── EdgePairObject        → Point
    │   ├── FitLineObject         → Line
    │   └── FitCircleObject       → Circle
    └── Composed (derive from other MeasureObjects)
        ├── TwoPointsLineObject      → Line
        ├── TwoPointsDistanceObject  → Distance
        ├── PointLineDistanceObject  → Distance
        ├── TwoLinesAngleObject      → Angle
        └── PointCircleDistanceObject→ Distance

Usage:
    wf = MeasurementWorkflow()

    # Localization anchors
    wf.add(TemplatePointObject("loc_A", ref_img, 80, 120, is_localization=True))
    wf.add(TemplatePointObject("loc_B", ref_img, 80, 600, is_localization=True))

    # Edge probes
    wf.add(EdgePointObject("e1", row=200, col=250, angle=np.pi/2, ...))
    wf.add(EdgePointObject("e2", row=400, col=250, angle=np.pi/2, ...))

    # Composed measurements
    wf.add(TwoPointsLineObject("edge_line", "e1", "e2"))
    wf.add(TwoPointsDistanceObject("gap", "e1", "e2"))

    # Execute
    results = wf.measure(inspection_image)
    print(results['gap'].value)

    # Persist
    wf.save("project.npz")
    wf2 = MeasurementWorkflow.load("project.npz")
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import cv2
import numpy as np

from measure1D import Halcon1DMeasure
from measure2D import CircleMeasureObject, LineMeasureObject
from measure_template import (
    Preprocessor,
    RawPreprocessor,
    TemplatePoint,
    _deserialize_preprocessor,
    _PREPROCESSOR_REGISTRY,
)


# ===========================================================================
# Shared visualization helpers
# ===========================================================================


def _draw_label(image: np.ndarray, text: str, position: Tuple[float, float],
                color: Tuple[int, int, int] = (0, 255, 0),
                offset: Tuple[int, int] = (10, -10),
                font_scale: float = 0.45) -> None:
    """
    Draw a label with dark outline near a position on the image.

    Args:
        image: BGR image (modified in-place).
        text: Label text.
        position: (col, row) anchor point.
        color: Text color (B, G, R).
        offset: (dx, dy) from anchor to text bottom-left.
        font_scale: OpenCV font scale.
    """
    x = int(position[0]) + offset[0]
    y = int(position[1]) + offset[1]
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, 1, cv2.LINE_AA)


def _draw_legend(image: np.ndarray, entries: List[Tuple[str, Tuple[int, int, int]]],
                 start_x: int = 10, start_y: int = 25, line_height: int = 18) -> None:
    """
    Draw a legend panel at the top-left corner.

    Args:
        image: BGR image (modified in-place).
        entries: List of (label, color) tuples.
        start_x, start_y: Top-left position of the first entry.
        line_height: Vertical spacing between entries.
    """
    x, y = start_x, start_y
    # Semi-transparent background
    overlay = image.copy()
    n = len(entries)
    cv2.rectangle(overlay, (x - 4, y - 16), (x + 200, y + n * line_height + 4),
                  (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
    for label, color in entries:
        cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, color, 1, cv2.LINE_AA)
        y += line_height


# ===========================================================================
# Geometric result types — the output vocabulary of the system
# ===========================================================================


@dataclass
class GeometricResult:
    """Base for all measurement outputs."""

    type: str  # 'point' | 'line' | 'circle' | 'distance' | 'angle'
    label: str = ""
    valid: bool = True
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PointResult(GeometricResult):
    """A single point in image coordinates."""

    type: str = "point"
    row: float = 0.0
    col: float = 0.0


@dataclass
class LineResult(GeometricResult):
    """A line: ax + by + c = 0 with endpoints."""

    type: str = "line"
    a: float = 0.0
    b: float = 0.0
    c: float = 0.0
    start_row: float = 0.0
    start_col: float = 0.0
    end_row: float = 0.0
    end_col: float = 0.0

    @property
    def angle_rad(self) -> float:
        """Angle of the line direction in radians (0 = right, pi/2 = down)."""
        return np.arctan2(-self.b, self.a)

    @property
    def angle_deg(self) -> float:
        return np.degrees(self.angle_rad)

    @property
    def length(self) -> float:
        dr = self.end_row - self.start_row
        dc = self.end_col - self.start_col
        return np.sqrt(dr**2 + dc**2)


@dataclass
class CircleResult(GeometricResult):
    """A circle: center and radius."""

    type: str = "circle"
    center_row: float = 0.0
    center_col: float = 0.0
    radius: float = 0.0


@dataclass
class DistanceResult(GeometricResult):
    """A distance value in pixels."""

    type: str = "distance"
    value: float = 0.0


@dataclass
class AngleResult(GeometricResult):
    """An angle in radians (always the acute/smaller angle, in [0, pi/2])."""

    type: str = "angle"
    value_rad: float = 0.0

    @property
    def value_deg(self) -> float:
        return np.degrees(self.value_rad)


# ===========================================================================
# SimilarityTransform — Umeyama least-squares similarity
# ===========================================================================


class SimilarityTransform:
    """
    2D similarity transform: rotation + uniform scale + translation.

    Computed from point correspondences using the Umeyama algorithm
    for least-squares similarity estimation.

    - 2+ point pairs: full similarity (rotation, scale, translation)
    - 1 point pair:   translation only
    - 0 point pairs:  invalid
    """

    def __init__(self):
        self.rotation: float = 0.0
        self.scale: float = 1.0
        self.translation_row: float = 0.0
        self.translation_col: float = 0.0
        self._valid: bool = False
        self._num_points: int = 0

    @property
    def is_valid(self) -> bool:
        return self._valid

    @classmethod
    def from_correspondences(
        cls,
        source_points: List[Tuple[float, float]],
        target_points: List[Tuple[float, float]],
    ) -> "SimilarityTransform":
        """
        Compute similarity transform from point pairs.

        Args:
            source_points: (row, col) in reference/teach coordinates.
            target_points: (row, col) in inspection coordinates (matched).

        Returns:
            SimilarityTransform mapping source → target.
        """
        t = cls()
        if len(source_points) != len(target_points):
            raise ValueError(
                f"source_points and target_points must have same length, "
                f"got {len(source_points)} vs {len(target_points)}"
            )

        n = len(source_points)
        if n == 0:
            return t  # invalid

        src = np.array(source_points, dtype=np.float64)
        dst = np.array(target_points, dtype=np.float64)
        t._num_points = n

        if n == 1:
            t.translation_row = dst[0, 0] - src[0, 0]
            t.translation_col = dst[0, 1] - src[0, 1]
            t.rotation = 0.0
            t.scale = 1.0
            t._valid = True
            return t

        # Umeyama algorithm for similarity (rotation + scale + translation)
        src_mean = np.mean(src, axis=0)
        dst_mean = np.mean(dst, axis=0)
        src_centered = src - src_mean
        dst_centered = dst - dst_mean

        # Covariance matrix: dst_centered.T @ src_centered / n
        cov = (dst_centered.T @ src_centered) / n
        U, S, Vt = np.linalg.svd(cov)

        # Handle reflection case (ensure det(R) = 1)
        d = np.ones(2)
        if np.linalg.det(U) * np.linalg.det(Vt) < 0:
            d[-1] = -1

        # Rotation matrix
        R = U @ np.diag(d) @ Vt

        # Scale
        src_var = np.sum(src_centered**2) / n
        if src_var > 1e-10:
            scale = np.sum(S * d) / src_var
        else:
            scale = 1.0

        t.rotation = np.arctan2(R[1, 0], R[0, 0])
        t.scale = scale
        t.translation_row = dst_mean[0] - scale * (
            R[0, 0] * src_mean[0] + R[0, 1] * src_mean[1]
        )
        t.translation_col = dst_mean[1] - scale * (
            R[1, 0] * src_mean[0] + R[1, 1] * src_mean[1]
        )
        t._valid = True
        return t

    def apply(self, row: float, col: float) -> Tuple[float, float]:
        """Transform a point from source to target coordinates."""
        if not self._valid:
            return (row, col)
        cos_r = np.cos(self.rotation)
        sin_r = np.sin(self.rotation)
        new_row = self.scale * (cos_r * row - sin_r * col) + self.translation_row
        new_col = self.scale * (sin_r * row + cos_r * col) + self.translation_col
        return (new_row, new_col)

    def apply_angle(self, angle: float) -> float:
        """Transform an angle (just adds the rotation)."""
        return angle - self.rotation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rotation": self.rotation,
            "scale": self.scale,
            "translation_row": self.translation_row,
            "translation_col": self.translation_col,
            "valid": self._valid,
            "num_points": self._num_points,
        }


# ===========================================================================
# MeasureObject — unified ABC
# ===========================================================================


class MeasureObject(ABC):
    """
    Unified measurement object contract.

    Lifecycle:
      1. Constructor stores configuration and dependency labels (strings).
      2. During Workflow.resolve(), _input_labels are bound to _input_objects.
      3. During calibration, primitive objects update their image-space positions.
      4. During execution, measure(image) is called in topological order.
      5. self.result is populated.

    Subclasses:
      - Primitive objects: measure directly from image pixels.
        Must implement calibrate() to update their positions.
      - Composed objects: compute results from their dependencies.
        calibrate() is a no-op (inputs are already calibrated).
    """

    def __init__(self, label: str):
        self.label = label
        self.result: Optional[GeometricResult] = None
        self._input_labels: List[str] = []  # labels of dependent objects
        self._input_objects: List[MeasureObject] = []  # bound after resolve()

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def measure(self, image: np.ndarray) -> GeometricResult:
        """
        Execute measurement.

        Primitive objects use the image directly.
        Composed objects read results from their input objects via get_input_result().
        """
        ...

    @abstractmethod
    def result_type(self) -> str:
        """One of: 'point', 'line', 'circle', 'distance', 'angle'."""
        ...

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self, transform: SimilarityTransform) -> None:
        """
        Apply a similarity transform to this object's internal coordinates.

        Primitive objects override this to update their image-space positions.
        Composed objects are no-ops because their inputs are already calibrated.
        """
        pass

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_primitive(self) -> bool:
        return len(self._input_labels) == 0

    @property
    def is_composed(self) -> bool:
        return not self.is_primitive

    # ------------------------------------------------------------------
    # Dependency helpers
    # ------------------------------------------------------------------

    def get_input_result(self, index: int) -> GeometricResult:
        """
        Retrieve a pre-computed input result.

        Args:
            index: Index into _input_objects (0-based).

        Returns:
            The GeometricResult of the indexed input object.

        Raises:
            RuntimeError: If the input hasn't been measured yet.
            IndexError: If index is out of range.
        """
        if index < 0 or index >= len(self._input_objects):
            raise IndexError(
                f"Input index {index} out of range for '{self.label}' "
                f"(has {len(self._input_objects)} inputs)"
            )
        obj = self._input_objects[index]
        if obj.result is None:
            raise RuntimeError(
                f"Input '{obj.label}' (index {index}) of '{self.label}' "
                f"has not been measured yet. This indicates a topological ordering bug."
            )
        return obj.result

    # ------------------------------------------------------------------
    # Serialization contracts
    # ------------------------------------------------------------------

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration (excluding _input_labels, handled by workflow)."""
        ...

    @classmethod
    @abstractmethod
    def from_dict(cls, label: str, data: Dict[str, Any]) -> "MeasureObject":
        """Deserialize a single object from its config dict."""
        ...

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def visualize(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Default no-op. Subclasses override for rendering."""
        return image


# ===========================================================================
# Primitive measurement objects
# ===========================================================================


class TemplatePointObject(MeasureObject):
    """
    Template-matching point. Can serve as a localization anchor.

    Wraps measure_template.TemplatePoint.
    """

    def __init__(
        self,
        label: str,
        reference_image: np.ndarray,
        click_row: float,
        click_col: float,
        template_size: int = 80,
        preprocessor: Optional[Preprocessor] = None,
        match_score_threshold: float = 0.5,
        use_subpixel: bool = True,
        is_localization: bool = False,
        search_region: Optional[Tuple[int, int, int, int]] = None,
    ):
        super().__init__(label)
        self._template_point = TemplatePoint(
            reference_image,
            click_row,
            click_col,
            template_size,
            preprocessor,
            match_score_threshold,
            use_subpixel,
        )
        self._teach_row = click_row
        self._teach_col = click_col
        self._calibrated_row = click_row
        self._calibrated_col = click_col
        self.template_size = template_size
        self.match_score_threshold = match_score_threshold
        self.use_subpixel = use_subpixel
        self.is_localization = is_localization
        self.search_region = search_region
        self._preprocessor = preprocessor

    def result_type(self) -> str:
        return "point"

    def calibrate(self, transform: SimilarityTransform) -> None:
        self._calibrated_row, self._calibrated_col = transform.apply(
            self._teach_row, self._teach_col
        )

    def measure(self, image: np.ndarray) -> GeometricResult:
        raw = self._template_point.measure(image, self.search_region)
        result = PointResult(
            label=self.label,
            row=raw["matched_row"],
            col=raw["matched_col"],
            valid=raw["valid"],
            meta={
                "match_score": raw["match_score"],
                "dx": raw["dx"],
                "dy": raw["dy"],
                "teach_row": self._teach_row,
                "teach_col": self._teach_col,
            },
        )
        self.result = result
        return result

    def to_dict(self) -> Dict[str, Any]:
        tp = self._template_point
        preprocessor_data = tp.preprocessor.serialize()
        # Ensure it's in the registry
        ptype = preprocessor_data.get("type", "raw")
        if ptype not in _PREPROCESSOR_REGISTRY:
            raise ValueError(
                f"Preprocessor type '{ptype}' is not registered. "
                f"Known types: {list(_PREPROCESSOR_REGISTRY.keys())}"
            )
        return {
            "object_type": "TemplatePointObject",
            "teach_row": self._teach_row,
            "teach_col": self._teach_col,
            "template_size": tp.template_size,
            "match_score_threshold": tp.match_score_threshold,
            "use_subpixel": tp.use_subpixel,
            "is_localization": self.is_localization,
            "preprocessor_data": preprocessor_data,
            "crop_center_row": tp._crop_center_row,
            "crop_center_col": tp._crop_center_col,
            "crop_h": tp._crop_h,
            "crop_w": tp._crop_w,
            "click_row": tp.click_row,
            "click_col": tp.click_col,
        }

    @classmethod
    def from_dict(cls, label: str, data: Dict[str, Any]) -> "TemplatePointObject":
        obj = cls.__new__(cls)
        MeasureObject.__init__(obj, label)

        # Reconstruct TemplatePoint via __new__ (same pattern as
        # TemplatePoint.from_file)
        tp = TemplatePoint.__new__(TemplatePoint)
        tp.click_row = data["click_row"]
        tp.click_col = data["click_col"]
        tp.template_size = data["template_size"]
        tp.match_score_threshold = data["match_score_threshold"]
        tp.use_subpixel = data["use_subpixel"]
        tp._crop_center_row = data["crop_center_row"]
        tp._crop_center_col = data["crop_center_col"]
        tp._crop_h = data["crop_h"]
        tp._crop_w = data["crop_w"]
        tp.result = None
        tp.preprocessor = _deserialize_preprocessor(data["preprocessor_data"])

        # edge_template and _actual_crop_bounds are stored as binary data
        # in the .npz file, restored separately by MeasurementWorkflow.load()
        tp.edge_template = data["edge_template"]
        tp._actual_crop_bounds = tuple(data["actual_crop_bounds"])

        obj._template_point = tp
        obj._teach_row = data["teach_row"]
        obj._teach_col = data["teach_col"]
        obj._calibrated_row = data["teach_row"]
        obj._calibrated_col = data["teach_col"]
        obj.template_size = data["template_size"]
        obj.match_score_threshold = data["match_score_threshold"]
        obj.use_subpixel = data["use_subpixel"]
        obj.is_localization = data.get("is_localization", False)
        obj._preprocessor = tp.preprocessor
        obj.search_region = None
        return obj

    def visualize(self, image: np.ndarray, **kwargs) -> np.ndarray:
        vis = self._template_point.visualize(image, **kwargs)
        # Draw label near the matched position (or teach position)
        if self.result is not None and self.result.valid:
            pos = (self.result.col, self.result.row)
        else:
            pos = (self._teach_col, self._teach_row)
        _draw_label(vis, self.label, pos, color=(0, 255, 0))
        return vis


class EdgePointObject(MeasureObject):
    """
    Single-edge point via 1D caliper.

    Places a Halcon1DMeasure probe at a specified position and returns the
    detected edge as a Point. Calibration adjusts the probe position.
    """

    def __init__(
        self,
        label: str,
        row: float,
        col: float,
        angle: float,
        length1: float,
        length2: float,
        sigma: float = 1.0,
        threshold: float = 30.0,
        transition: str = "all",
        select: str = "first",
        interpolation: str = "linear",
    ):
        super().__init__(label)
        self._teach_row = row
        self._teach_col = col
        self._teach_angle = angle
        self._calibrated_row = row
        self._calibrated_col = col
        self._calibrated_angle = angle
        self.length1 = length1
        self.length2 = length2
        self.sigma = sigma
        self.threshold = threshold
        self.transition = transition
        self.select = select
        self.interpolation = interpolation

    def result_type(self) -> str:
        return "point"

    def calibrate(self, transform: SimilarityTransform) -> None:
        self._calibrated_row, self._calibrated_col = transform.apply(
            self._teach_row, self._teach_col
        )
        self._calibrated_angle = transform.apply_angle(self._teach_angle)

    def measure(self, image: np.ndarray) -> GeometricResult:
        self._cached_measure = Halcon1DMeasure(
            row=self._calibrated_row,
            col=self._calibrated_col,
            angle=self._calibrated_angle,
            length1=self.length1,
            length2=self.length2,
            interpolation=self.interpolation,
        )
        rows, cols, amps, _ = self._cached_measure.measure_pos(
            image,
            sigma=self.sigma,
            threshold=self.threshold,
            transition=self.transition,
            select=self.select,
        )
        if rows:
            result = PointResult(
                label=self.label,
                row=rows[0],
                col=cols[0],
                valid=True,
                meta={
                    "amplitude": amps[0],
                    "num_edges_found": len(rows),
                    "calibrated_row": self._calibrated_row,
                    "calibrated_col": self._calibrated_col,
                    "calibrated_angle": self._calibrated_angle,
                },
            )
        else:
            result = PointResult(
                label=self.label,
                row=self._calibrated_row,
                col=self._calibrated_col,
                valid=False,
                meta={
                    "amplitude": 0.0,
                    "num_edges_found": 0,
                    "reason": "no edges found",
                },
            )
        self.result = result
        return result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_type": "EdgePointObject",
            "teach_row": self._teach_row,
            "teach_col": self._teach_col,
            "teach_angle": self._teach_angle,
            "length1": self.length1,
            "length2": self.length2,
            "sigma": self.sigma,
            "threshold": self.threshold,
            "transition": self.transition,
            "select": self.select,
            "interpolation": self.interpolation,
        }

    @classmethod
    def from_dict(cls, label: str, data: Dict[str, Any]) -> "EdgePointObject":
        return cls(
            label=label,
            row=data["teach_row"],
            col=data["teach_col"],
            angle=data["teach_angle"],
            length1=data["length1"],
            length2=data["length2"],
            sigma=data.get("sigma", 1.0),
            threshold=data.get("threshold", 30.0),
            transition=data.get("transition", "all"),
            select=data.get("select", "first"),
            interpolation=data.get("interpolation", "linear"),
        )

    def visualize(self, image: np.ndarray, **kwargs) -> np.ndarray:
        if len(image.shape) == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()

        # Draw the caliper position (reuse cached object from measure() if available)
        measure = getattr(self, "_cached_measure", None)
        if measure is None:
            measure = Halcon1DMeasure(
                row=self._calibrated_row,
                col=self._calibrated_col,
                angle=self._calibrated_angle,
                length1=self.length1,
                length2=self.length2,
                interpolation=self.interpolation,
            )
        vis = measure.draw_roi_on_image(vis)

        # Draw detected edge point
        if self.result is not None and self.result.valid:
            pt = (int(round(self.result.col)), int(round(self.result.row)))
            cv2.drawMarker(
                vis,
                pt,
                (0, 255, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=10,
                thickness=2,
            )
        _draw_label(vis, self.label, (self._calibrated_col, self._calibrated_row),
                    color=(0, 255, 0), offset=(12, -12))
        return vis


class EdgePairObject(MeasureObject):
    """
    Edge-pair center point via 1D caliper.

    Uses Halcon1DMeasure.measure_pairs to find edge pairs and returns
    the center of the first/last/all pairs as a Point.
    """

    def __init__(
        self,
        label: str,
        row: float,
        col: float,
        angle: float,
        length1: float,
        length2: float,
        sigma: float = 1.0,
        threshold: float = 30.0,
        transition: str = "negative",
        select: str = "first",
        interpolation: str = "linear",
    ):
        super().__init__(label)
        self._teach_row = row
        self._teach_col = col
        self._teach_angle = angle
        self._calibrated_row = row
        self._calibrated_col = col
        self._calibrated_angle = angle
        self.length1 = length1
        self.length2 = length2
        self.sigma = sigma
        self.threshold = threshold
        self.transition = transition
        self.select = select
        self.interpolation = interpolation

    def result_type(self) -> str:
        return "point"

    def calibrate(self, transform: SimilarityTransform) -> None:
        self._calibrated_row, self._calibrated_col = transform.apply(
            self._teach_row, self._teach_col
        )
        self._calibrated_angle = transform.apply_angle(self._teach_angle)

    def measure(self, image: np.ndarray) -> GeometricResult:
        self._cached_measure = Halcon1DMeasure(
            row=self._calibrated_row,
            col=self._calibrated_col,
            angle=self._calibrated_angle,
            length1=self.length1,
            length2=self.length2,
            interpolation=self.interpolation,
        )
        (
            pairs_row1,
            pairs_col1,
            _,
            pairs_row2,
            pairs_col2,
            _,
            centers_row,
            centers_col,
            intra_dist,
            inter_dist,
        ) = self._cached_measure.measure_pairs(
            image,
            sigma=self.sigma,
            threshold=self.threshold,
            transition=self.transition,
            select=self.select,
        )

        if centers_row:
            result = PointResult(
                label=self.label,
                row=centers_row[0],
                col=centers_col[0],
                valid=True,
                meta={
                    "num_pairs_found": len(centers_row),
                    "intra_distance": intra_dist[0] if intra_dist else 0.0,
                    "inter_distance": inter_dist[0] if inter_dist else 0.0,
                },
            )
        else:
            result = PointResult(
                label=self.label,
                row=self._calibrated_row,
                col=self._calibrated_col,
                valid=False,
                meta={"num_pairs_found": 0, "reason": "no edge pairs found"},
            )
        self.result = result
        return result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_type": "EdgePairObject",
            "teach_row": self._teach_row,
            "teach_col": self._teach_col,
            "teach_angle": self._teach_angle,
            "length1": self.length1,
            "length2": self.length2,
            "sigma": self.sigma,
            "threshold": self.threshold,
            "transition": self.transition,
            "select": self.select,
            "interpolation": self.interpolation,
        }

    @classmethod
    def from_dict(cls, label: str, data: Dict[str, Any]) -> "EdgePairObject":
        return cls(
            label=label,
            row=data["teach_row"],
            col=data["teach_col"],
            angle=data["teach_angle"],
            length1=data["length1"],
            length2=data["length2"],
            sigma=data.get("sigma", 1.0),
            threshold=data.get("threshold", 30.0),
            transition=data.get("transition", "negative"),
            select=data.get("select", "first"),
            interpolation=data.get("interpolation", "linear"),
        )

    def visualize(self, image: np.ndarray, **kwargs) -> np.ndarray:
        if len(image.shape) == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()
        measure = getattr(self, "_cached_measure", None)
        if measure is None:
            measure = Halcon1DMeasure(
                row=self._calibrated_row,
                col=self._calibrated_col,
                angle=self._calibrated_angle,
                length1=self.length1,
                length2=self.length2,
                interpolation=self.interpolation,
            )
        vis = measure.draw_roi_on_image(vis)
        if self.result is not None and self.result.valid:
            pt = (int(round(self.result.col)), int(round(self.result.row)))
            cv2.drawMarker(
                vis,
                pt,
                (0, 255, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=10,
                thickness=2,
            )
        _draw_label(vis, self.label, (self._calibrated_col, self._calibrated_row),
                    color=(0, 255, 255), offset=(12, -12))
        return vis


class FitLineObject(MeasureObject):
    """
    Edge-based line fitting using multiple measurement rectangles.

    Wraps measure2D.LineMeasureObject. Calibration transforms the line's
    start and end points.
    """

    def __init__(
        self,
        label: str,
        start: Tuple[float, float],
        end: Tuple[float, float],
        measure_length1: float,
        measure_length2: float,
        num_measures: int = 10,
        sigma: float = 1.0,
        threshold: float = 30.0,
        transition: str = "all",
    ):
        super().__init__(label)
        self._teach_start = (float(start[0]), float(start[1]))
        self._teach_end = (float(end[0]), float(end[1]))
        self._calibrated_start = (float(start[0]), float(start[1]))
        self._calibrated_end = (float(end[0]), float(end[1]))
        self.measure_length1 = measure_length1
        self.measure_length2 = measure_length2
        self.num_measures = num_measures
        self.sigma = sigma
        self.threshold = threshold
        self.transition = transition

    def result_type(self) -> str:
        return "line"

    def calibrate(self, transform: SimilarityTransform) -> None:
        sr, sc = transform.apply(self._teach_start[0], self._teach_start[1])
        er, ec = transform.apply(self._teach_end[0], self._teach_end[1])
        self._calibrated_start = (sr, sc)
        self._calibrated_end = (er, ec)

    def measure(self, image: np.ndarray) -> GeometricResult:
        self._cached_obj = LineMeasureObject(
            start=self._calibrated_start,
            end=self._calibrated_end,
            measure_length1=self.measure_length1,
            measure_length2=self.measure_length2,
            num_measures=self.num_measures,
            sigma=self.sigma,
            threshold=self.threshold,
            transition=self.transition,
        )
        raw = self._cached_obj.measure(image)
        if raw is None:
            result = LineResult(
                label=self.label,
                valid=False,
                meta={"reason": "insufficient edge points for fit"},
            )
        else:
            result = LineResult(
                label=self.label,
                a=raw["params"][0],
                b=raw["params"][1],
                c=raw["params"][2],
                start_row=raw["start"][0],
                start_col=raw["start"][1],
                end_row=raw["end"][0],
                end_col=raw["end"][1],
                valid=True,
                meta={
                    "num_points": raw["num_points"],
                    "mean_error": raw["mean_error"],
                    "max_error": raw["max_error"],
                    "angle_rad": raw["angle"],
                },
            )
        self.result = result
        return result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_type": "FitLineObject",
            "teach_start": list(self._teach_start),
            "teach_end": list(self._teach_end),
            "measure_length1": self.measure_length1,
            "measure_length2": self.measure_length2,
            "num_measures": self.num_measures,
            "sigma": self.sigma,
            "threshold": self.threshold,
            "transition": self.transition,
        }

    @classmethod
    def from_dict(cls, label: str, data: Dict[str, Any]) -> "FitLineObject":
        return cls(
            label=label,
            start=tuple(data["teach_start"]),
            end=tuple(data["teach_end"]),
            measure_length1=data["measure_length1"],
            measure_length2=data["measure_length2"],
            num_measures=data.get("num_measures", 10),
            sigma=data.get("sigma", 1.0),
            threshold=data.get("threshold", 30.0),
            transition=data.get("transition", "all"),
        )

    def visualize(self, image: np.ndarray, **kwargs) -> np.ndarray:
        obj = getattr(self, "_cached_obj", None)
        if obj is None:
            obj = LineMeasureObject(
                start=self._calibrated_start,
                end=self._calibrated_end,
                measure_length1=self.measure_length1,
                measure_length2=self.measure_length2,
                num_measures=self.num_measures,
                sigma=self.sigma,
                threshold=self.threshold,
                transition=self.transition,
            )
        # Set result on the temp object so visualize can use it
        if self.result is not None and self.result.valid:
            r = self.result
            dr = r.end_row - r.start_row
            dc = r.end_col - r.start_col
            obj.result = {
                "params": (r.a, r.b, r.c),
                "point": ((r.start_row + r.end_row) / 2, (r.start_col + r.end_col) / 2),
                "direction": [dr, dc],
                "start": (r.start_row, r.start_col),
                "end": (r.end_row, r.end_col),
                "angle": float(np.arctan2(dc, dr)),
                "num_points": r.meta.get("num_points", 0),
                "mean_error": r.meta.get("mean_error", 0.0),
                "max_error": r.meta.get("max_error", 0.0),
            }
        vis = obj.visualize(image, **kwargs)
        # Draw label near line midpoint
        mid_col = (self._calibrated_start[1] + self._calibrated_end[1]) / 2
        mid_row = (self._calibrated_start[0] + self._calibrated_end[0]) / 2
        _draw_label(vis, self.label, (mid_col, mid_row), color=(255, 255, 0))
        return vis


class FitCircleObject(MeasureObject):
    """
    Edge-based circle fitting using multiple measurement rectangles.

    Wraps measure2D.CircleMeasureObject. Calibration transforms the center
    position; radius is scaled by the transform.
    """

    def __init__(
        self,
        label: str,
        center: Tuple[float, float],
        radius: float,
        radius_min: float,
        radius_max: float,
        measure_length1: float,
        measure_length2: float,
        num_measures: int = 12,
        sigma: float = 1.0,
        threshold: float = 30.0,
        transition: str = "all",
        start_phi: float = 0.0,
        end_phi: float = 2 * np.pi,
    ):
        super().__init__(label)
        self._teach_center = (float(center[0]), float(center[1]))
        self._teach_radius = float(radius)
        self._calibrated_center = (float(center[0]), float(center[1]))
        self._calibrated_radius = float(radius)
        self.radius_min = radius_min
        self.radius_max = radius_max
        self.measure_length1 = measure_length1
        self.measure_length2 = measure_length2
        self.num_measures = num_measures
        self.sigma = sigma
        self.threshold = threshold
        self.transition = transition
        self.start_phi = start_phi
        self.end_phi = end_phi

    def result_type(self) -> str:
        return "circle"

    def calibrate(self, transform: SimilarityTransform) -> None:
        cr, cc = transform.apply(self._teach_center[0], self._teach_center[1])
        self._calibrated_center = (cr, cc)
        self._calibrated_radius = self._teach_radius * transform.scale

    def measure(self, image: np.ndarray) -> GeometricResult:
        self._cached_obj = CircleMeasureObject(
            center=self._calibrated_center,
            radius=self._calibrated_radius,
            radius_min=self.radius_min * (self._calibrated_radius / max(self._teach_radius, 1e-6)),
            radius_max=self.radius_max * (self._calibrated_radius / max(self._teach_radius, 1e-6)),
            measure_length1=self.measure_length1,
            measure_length2=self.measure_length2,
            num_measures=self.num_measures,
            sigma=self.sigma,
            threshold=self.threshold,
            transition=self.transition,
            start_phi=self.start_phi,
            end_phi=self.end_phi,
        )
        raw = self._cached_obj.measure(image)
        if raw is None:
            result = CircleResult(
                label=self.label,
                valid=False,
                meta={"reason": "insufficient edge points for fit"},
            )
        else:
            result = CircleResult(
                label=self.label,
                center_row=raw["center"][0],
                center_col=raw["center"][1],
                radius=raw["radius"],
                valid=True,
                meta={
                    "num_points": raw["num_points"],
                    "mean_error": raw["mean_error"],
                    "max_error": raw["max_error"],
                },
            )
        self.result = result
        return result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_type": "FitCircleObject",
            "teach_center": list(self._teach_center),
            "teach_radius": self._teach_radius,
            "radius_min": self.radius_min,
            "radius_max": self.radius_max,
            "measure_length1": self.measure_length1,
            "measure_length2": self.measure_length2,
            "num_measures": self.num_measures,
            "sigma": self.sigma,
            "threshold": self.threshold,
            "transition": self.transition,
            "start_phi": self.start_phi,
            "end_phi": self.end_phi,
        }

    @classmethod
    def from_dict(cls, label: str, data: Dict[str, Any]) -> "FitCircleObject":
        return cls(
            label=label,
            center=tuple(data["teach_center"]),
            radius=data["teach_radius"],
            radius_min=data["radius_min"],
            radius_max=data["radius_max"],
            measure_length1=data["measure_length1"],
            measure_length2=data["measure_length2"],
            num_measures=data.get("num_measures", 12),
            sigma=data.get("sigma", 1.0),
            threshold=data.get("threshold", 30.0),
            transition=data.get("transition", "all"),
            start_phi=data.get("start_phi", 0.0),
            end_phi=data.get("end_phi", 2 * np.pi),
        )

    def visualize(self, image: np.ndarray, **kwargs) -> np.ndarray:
        obj = getattr(self, "_cached_obj", None)
        if obj is None:
            obj = CircleMeasureObject(
                center=self._calibrated_center,
                radius=self._calibrated_radius,
                radius_min=self.radius_min,
                radius_max=self.radius_max,
                measure_length1=self.measure_length1,
                measure_length2=self.measure_length2,
                num_measures=self.num_measures,
                sigma=self.sigma,
                threshold=self.threshold,
                transition=self.transition,
                start_phi=self.start_phi,
                end_phi=self.end_phi,
            )
        if self.result is not None and self.result.valid:
            obj.result = {
                "center": (self.result.center_row, self.result.center_col),
                "radius": self.result.radius,
                "num_points": self.result.meta.get("num_points", 0),
                "mean_error": self.result.meta.get("mean_error", 0.0),
                "max_error": self.result.meta.get("max_error", 0.0),
            }
        vis = obj.visualize(image, **kwargs)
        _draw_label(vis, self.label,
                    (self._calibrated_center[1] + self._calibrated_radius,
                     self._calibrated_center[0] - self._calibrated_radius),
                    color=(255, 100, 255), offset=(8, 0))
        return vis


# ===========================================================================
# Composed measurement objects
# ===========================================================================


class TwoPointsLineObject(MeasureObject):
    """Line defined by two Point-producing objects."""

    def __init__(self, label: str, point_a_label: str, point_b_label: str):
        super().__init__(label)
        self._input_labels = [point_a_label, point_b_label]

    def result_type(self) -> str:
        return "line"

    def measure(self, image: np.ndarray) -> GeometricResult:
        p1 = self.get_input_result(0)
        p2 = self.get_input_result(1)

        if not (p1.valid and p2.valid):
            result = LineResult(
                label=self.label,
                valid=False,
                meta={"reason": "one or both input points are invalid"},
            )
            self.result = result
            return result

        dr = p2.row - p1.row
        dc = p2.col - p1.col
        norm = np.sqrt(dr**2 + dc**2)

        if norm < 1e-10:
            result = LineResult(
                label=self.label,
                valid=False,
                meta={"reason": "points are coincident"},
            )
            self.result = result
            return result

        # Line normal: (dc, -dr) normalized
        a = dc / norm
        b = -dr / norm
        c = -(a * p1.row + b * p1.col)

        result = LineResult(
            label=self.label,
            a=a,
            b=b,
            c=c,
            start_row=p1.row,
            start_col=p1.col,
            end_row=p2.row,
            end_col=p2.col,
            valid=True,
            meta={
                "point_a_label": self._input_labels[0],
                "point_b_label": self._input_labels[1],
                "length": norm,
            },
        )
        self.result = result
        return result

    def to_dict(self) -> Dict[str, Any]:
        return {"object_type": "TwoPointsLineObject"}

    @classmethod
    def from_dict(cls, label: str, data: Dict[str, Any]) -> "TwoPointsLineObject":
        obj = cls.__new__(cls)
        MeasureObject.__init__(obj, label)
        # _input_labels are restored separately by the workflow
        return obj

    def visualize(self, image: np.ndarray, **kwargs) -> np.ndarray:
        if len(image.shape) == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()
        if self.result is not None and self.result.valid:
            r = self.result
            pt1 = (int(round(r.start_col)), int(round(r.start_row)))
            pt2 = (int(round(r.end_col)), int(round(r.end_row)))
            # Double-stroke line
            cv2.line(vis, pt1, pt2, (0, 0, 0), thickness=3)
            cv2.line(vis, pt1, pt2, (255, 0, 255), thickness=2)
        if self.result is not None and self.result.valid and len(self._input_objects) >= 2:
            i0 = self._input_objects[0].result
            i1 = self._input_objects[1].result
            if i0 and i1:
                mid = ((i0.col + i1.col) / 2, (i0.row + i1.row) / 2)
                _draw_label(vis, self.label, mid, color=(255, 0, 255), offset=(0, 15))
        return vis


class TwoPointsDistanceObject(MeasureObject):
    """Euclidean distance between two Point objects."""

    def __init__(self, label: str, point_a_label: str, point_b_label: str):
        super().__init__(label)
        self._input_labels = [point_a_label, point_b_label]

    def result_type(self) -> str:
        return "distance"

    def measure(self, image: np.ndarray) -> GeometricResult:
        p1 = self.get_input_result(0)
        p2 = self.get_input_result(1)
        valid = p1.valid and p2.valid
        dr = p2.row - p1.row
        dc = p2.col - p1.col
        dist = np.sqrt(dr**2 + dc**2)
        result = DistanceResult(label=self.label, value=dist, valid=valid)
        self.result = result
        return result

    def to_dict(self) -> Dict[str, Any]:
        return {"object_type": "TwoPointsDistanceObject"}

    @classmethod
    def from_dict(cls, label: str, data: Dict[str, Any]) -> "TwoPointsDistanceObject":
        obj = cls.__new__(cls)
        MeasureObject.__init__(obj, label)
        return obj

    def visualize(self, image: np.ndarray, **kwargs) -> np.ndarray:
        if len(image.shape) == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()
        if (
            self.result is not None
            and self.result.valid
            and len(self._input_objects) == 2
        ):
            p1 = self._input_objects[0].result
            p2 = self._input_objects[1].result
            if p1 is not None and p2 is not None and p1.valid and p2.valid:
                pt1 = (int(round(p1.col)), int(round(p1.row)))
                pt2 = (int(round(p2.col)), int(round(p2.row)))
                cv2.line(vis, pt1, pt2, (0, 0, 0), thickness=3)
                cv2.line(vis, pt1, pt2, (255, 0, 255), thickness=2)
                # Label at midpoint
                mid = (
                    int(round((p1.col + p2.col) / 2)),
                    int(round((p1.row + p2.row) / 2)),
                )
                vis_label = f"{self.label} {self.result.value:.1f}px"
                _draw_label(vis, vis_label, mid, color=(255, 0, 255))
        return vis


class PointLineDistanceObject(MeasureObject):
    """Perpendicular distance from a Point to a Line."""

    def __init__(self, label: str, point_label: str, line_label: str):
        super().__init__(label)
        self._input_labels = [point_label, line_label]

    def result_type(self) -> str:
        return "distance"

    def measure(self, image: np.ndarray) -> GeometricResult:
        pt = self.get_input_result(0)
        line = self.get_input_result(1)
        if not (pt.valid and line.valid):
            result = DistanceResult(label=self.label, value=0.0, valid=False)
            self.result = result
            return result
        # Distance = |a*row + b*col + c| / sqrt(a^2 + b^2)
        dist = abs(line.a * pt.row + line.b * pt.col + line.c) / np.sqrt(
            line.a**2 + line.b**2 + 1e-10
        )
        result = DistanceResult(label=self.label, value=dist, valid=True)
        self.result = result
        return result

    def to_dict(self) -> Dict[str, Any]:
        return {"object_type": "PointLineDistanceObject"}

    @classmethod
    def from_dict(cls, label: str, data: Dict[str, Any]) -> "PointLineDistanceObject":
        obj = cls.__new__(cls)
        MeasureObject.__init__(obj, label)
        return obj

    def visualize(self, image: np.ndarray, **kwargs) -> np.ndarray:
        if len(image.shape) == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()
        if (
            self.result is not None
            and self.result.valid
            and len(self._input_objects) == 2
        ):
            pt = self._input_objects[0].result
            line = self._input_objects[1].result
            if (
                pt is not None
                and line is not None
                and pt.valid
                and line.valid
            ):
                # Draw dashed perpendicular from point to line
                # Project point onto line
                denom = line.a**2 + line.b**2
                proj_row = (
                    line.b * (line.b * pt.row - line.a * pt.col) - line.a * line.c
                ) / denom
                proj_col = (
                    line.a * (-line.b * pt.row + line.a * pt.col) - line.b * line.c
                ) / denom
                pt1 = (int(round(pt.col)), int(round(pt.row)))
                pt2 = (int(round(proj_col)), int(round(proj_row)))
                cv2.line(vis, pt1, pt2, (0, 0, 0), thickness=3)
                cv2.line(vis, pt1, pt2, (0, 255, 255), thickness=2)
                mid = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
                _draw_label(vis, f"{self.label} {self.result.value:.1f}px",
                            mid, color=(0, 255, 255))
        return vis


class TwoLinesAngleObject(MeasureObject):
    """Acute angle between two Line objects (in [0, pi/2])."""

    def __init__(self, label: str, line_a_label: str, line_b_label: str):
        super().__init__(label)
        self._input_labels = [line_a_label, line_b_label]

    def result_type(self) -> str:
        return "angle"

    def measure(self, image: np.ndarray) -> GeometricResult:
        line1 = self.get_input_result(0)
        line2 = self.get_input_result(1)
        if not (line1.valid and line2.valid):
            result = AngleResult(label=self.label, value_rad=0.0, valid=False)
            self.result = result
            return result
        # Angle between normal vectors (a, b)
        dot = line1.a * line2.a + line1.b * line2.b
        norm1 = np.sqrt(line1.a**2 + line1.b**2)
        norm2 = np.sqrt(line2.a**2 + line2.b**2)
        cos_angle = np.clip(dot / (norm1 * norm2 + 1e-10), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        # Return acute angle
        if angle > np.pi / 2:
            angle = np.pi - angle
        result = AngleResult(label=self.label, value_rad=angle, valid=True)
        self.result = result
        return result

    def to_dict(self) -> Dict[str, Any]:
        return {"object_type": "TwoLinesAngleObject"}

    @classmethod
    def from_dict(cls, label: str, data: Dict[str, Any]) -> "TwoLinesAngleObject":
        obj = cls.__new__(cls)
        MeasureObject.__init__(obj, label)
        return obj

    def visualize(self, image: np.ndarray, **kwargs) -> np.ndarray:
        if len(image.shape) == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()
        if (
            self.result is not None
            and self.result.valid
            and len(self._input_objects) == 2
        ):
            line1 = self._input_objects[0].result
            line2 = self._input_objects[1].result
            if (
                line1 is not None
                and line2 is not None
                and line1.valid
                and line2.valid
            ):
                # Find intersection of the two lines, and draw an arc
                a1, b1, c1 = line1.a, line1.b, line1.c
                a2, b2, c2 = line2.a, line2.b, line2.c
                det = a1 * b2 - a2 * b1
                if abs(det) > 1e-10:
                    ix_row = (b1 * c2 - b2 * c1) / det
                    ix_col = (a2 * c1 - a1 * c2) / det
                    ix = (int(round(ix_col)), int(round(ix_row)))
                    # Intersection marker + angle label
                    cv2.circle(vis, ix, 5, (0, 200, 255), -1)
                    _draw_label(vis, f"{self.label} {self.result.value_deg:.1f}deg",
                                ix, color=(0, 255, 255), offset=(12, -8))
        return vis


class PointCircleDistanceObject(MeasureObject):
    """Shortest distance from a Point to the circumference of a Circle."""

    def __init__(self, label: str, point_label: str, circle_label: str):
        super().__init__(label)
        self._input_labels = [point_label, circle_label]

    def result_type(self) -> str:
        return "distance"

    def measure(self, image: np.ndarray) -> GeometricResult:
        pt = self.get_input_result(0)
        circle = self.get_input_result(1)
        if not (pt.valid and circle.valid):
            result = DistanceResult(label=self.label, value=0.0, valid=False)
            self.result = result
            return result
        dist_to_center = np.sqrt(
            (pt.row - circle.center_row) ** 2 + (pt.col - circle.center_col) ** 2
        )
        result = DistanceResult(
            label=self.label,
            value=abs(dist_to_center - circle.radius),
            valid=True,
        )
        self.result = result
        return result

    def to_dict(self) -> Dict[str, Any]:
        return {"object_type": "PointCircleDistanceObject"}

    @classmethod
    def from_dict(
        cls, label: str, data: Dict[str, Any]
    ) -> "PointCircleDistanceObject":
        obj = cls.__new__(cls)
        MeasureObject.__init__(obj, label)
        return obj

    def visualize(self, image: np.ndarray, **kwargs) -> np.ndarray:
        if len(image.shape) == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()
        if (
            self.result is not None
            and self.result.valid
            and len(self._input_objects) == 2
        ):
            pt = self._input_objects[0].result
            circle = self._input_objects[1].result
            if (
                pt is not None
                and circle is not None
                and pt.valid
                and circle.valid
            ):
                ctr = (int(round(circle.center_col)), int(round(circle.center_row)))
                ptx = (int(round(pt.col)), int(round(pt.row)))
                cv2.line(vis, ctr, ptx, (0, 0, 0), thickness=3)
                cv2.line(vis, ctr, ptx, (0, 255, 255), thickness=2)
                mid = ((ctr[0] + ptx[0]) // 2, (ctr[1] + ptx[1]) // 2)
                _draw_label(vis, f"{self.label} {self.result.value:.1f}px",
                            mid, color=(0, 255, 255))
        return vis


# ===========================================================================
# MeasurementWorkflow — orchestrator
# ===========================================================================


# Registry of object types for deserialization
_OBJECT_TYPE_REGISTRY: Dict[str, Type[MeasureObject]] = {
    "TemplatePointObject": TemplatePointObject,
    "EdgePointObject": EdgePointObject,
    "EdgePairObject": EdgePairObject,
    "FitLineObject": FitLineObject,
    "FitCircleObject": FitCircleObject,
    "TwoPointsLineObject": TwoPointsLineObject,
    "TwoPointsDistanceObject": TwoPointsDistanceObject,
    "PointLineDistanceObject": PointLineDistanceObject,
    "TwoLinesAngleObject": TwoLinesAngleObject,
    "PointCircleDistanceObject": PointCircleDistanceObject,
}


class MeasurementWorkflow:
    """
    Unified measurement workflow manager.

    Manages the full lifecycle:
      1. Registration: add objects with labels
      2. Serialization: save/load the entire object graph
      3. Resolution: bind input labels to object references, topological sort
      4. Calibration: apply localization transform to all primitives
      5. Execution: measure() in topological order
      6. Results: retrieve by label

    Usage:
        wf = MeasurementWorkflow()

        # Localization anchors
        wf.add(TemplatePointObject("loc1", ref_img, 100, 150, is_localization=True))
        wf.add(TemplatePointObject("loc2", ref_img, 100, 450, is_localization=True))

        # Measurement tools
        wf.add(EdgePointObject("edge1", row=200, col=200, angle=pi/2, ...))
        wf.add(TwoPointsDistanceObject("gap", "loc1", "loc2"))

        # Execute
        results = wf.measure(inspection_image)

        # Persist
        wf.save("project.npz")
        wf2 = MeasurementWorkflow.load("project.npz")
    """

    def __init__(self):
        self._objects: Dict[str, MeasureObject] = {}
        self._registration_order: List[str] = []
        self._execution_order: List[str] = []
        self._transform: Optional[SimilarityTransform] = None
        self._resolved: bool = False
        self._results: Dict[str, GeometricResult] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def add(self, obj: MeasureObject) -> "MeasurementWorkflow":
        """
        Register a measurement object. Returns self for chaining.

        Raises ValueError if label already exists.
        """
        if obj.label in self._objects:
            raise ValueError(
                f"Duplicate label '{obj.label}'. Each measurement object "
                f"must have a unique label."
            )
        self._objects[obj.label] = obj
        self._registration_order.append(obj.label)
        self._resolved = False
        return self

    @property
    def object_labels(self) -> List[str]:
        """Return labels in registration order."""
        return list(self._registration_order)

    def has_object(self, label: str) -> bool:
        return label in self._objects

    # ------------------------------------------------------------------
    # Dependency resolution
    # ------------------------------------------------------------------

    def resolve(self) -> None:
        """
        Bind dependency labels to object references and compute
        topological execution order using Kahn's algorithm.

        Raises:
            ValueError: If a referenced label doesn't exist.
            ValueError: If a cyclic dependency is detected.
        """
        # Bind references
        for label, obj in self._objects.items():
            obj._input_objects = []
            for input_label in obj._input_labels:
                if input_label not in self._objects:
                    raise ValueError(
                        f"Object '{label}' references unknown input "
                        f"'{input_label}'. Available labels: "
                        f"{list(self._objects.keys())}"
                    )
                obj._input_objects.append(self._objects[input_label])

        # Topological sort (Kahn's algorithm)
        in_degree: Dict[str, int] = {label: 0 for label in self._objects}
        adjacency: Dict[str, List[str]] = {label: [] for label in self._objects}

        for label, obj in self._objects.items():
            for inp in obj._input_objects:
                adjacency[inp.label].append(label)
                in_degree[label] += 1

        queue = [l for l, d in in_degree.items() if d == 0]
        order = []

        while queue:
            current = queue.pop(0)
            order.append(current)
            for neighbor in adjacency[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(self._objects):
            remaining = set(self._objects.keys()) - set(order)
            raise ValueError(
                f"Cyclic dependency detected involving: {remaining}. "
                f"Check the _input_labels of these objects."
            )

        self._execution_order = order
        self._resolved = True

    # ------------------------------------------------------------------
    # Localization
    # ------------------------------------------------------------------

    def _get_localization_objects(self) -> List[TemplatePointObject]:
        """Find all objects designated as localization anchors."""
        loc_objs = []
        for label in self._registration_order:
            obj = self._objects[label]
            if isinstance(obj, TemplatePointObject) and obj.is_localization:
                loc_objs.append(obj)
        return loc_objs

    def _compute_localization_transform(
        self, image: np.ndarray
    ) -> SimilarityTransform:
        """
        Measure all localization objects and compute the similarity transform
        from their teach→match positions.
        """
        loc_objs = self._get_localization_objects()
        if not loc_objs:
            t = SimilarityTransform()
            t._valid = True
            return t

        source_points = []
        target_points = []

        for obj in loc_objs:
            result = obj.measure(image)
            if not result.valid:
                raise RuntimeError(
                    f"Localization template '{obj.label}' failed to match "
                    f"(score={result.meta.get('match_score', 'N/A'):.4f}). "
                    f"Cannot localize the measurement."
                )
            source_points.append((obj._teach_row, obj._teach_col))
            target_points.append((result.row, result.col))

        return SimilarityTransform.from_correspondences(source_points, target_points)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def measure(
        self, image: np.ndarray
    ) -> Dict[str, GeometricResult]:
        """
        Execute the complete measurement workflow.

        1. Resolve dependencies and compute execution order
        2. Match localization templates and compute similarity transform
        3. Calibrate all primitive objects with the transform
        4. Execute all objects in topological order
        5. Return {label: GeometricResult}

        Args:
            image: Inspection image (grayscale uint8 or BGR).

        Returns:
            Dict mapping object labels to their GeometricResult.
        """
        if not self._resolved:
            self.resolve()

        # Step 1: Compute localization transform
        try:
            self._transform = self._compute_localization_transform(image)
        except RuntimeError as e:
            # Localization failed — mark all objects as invalid
            self._transform = SimilarityTransform()
            for label in self._registration_order:
                obj = self._objects[label]
                obj.result = GeometricResult(
                    type=obj.result_type(),
                    label=label,
                    valid=False,
                    meta={"error": str(e), "reason": "localization failed"},
                )
            self._results = {
                label: self._objects[label].result
                for label in self._registration_order
                if self._objects[label].result is not None
            }
            return self._results

        # Step 2: Calibrate all primitives
        if self._transform.is_valid:
            for label in self._registration_order:
                obj = self._objects[label]
                if obj.is_primitive:
                    obj.calibrate(self._transform)

        # Step 3: Execute in topological order
        for label in self._execution_order:
            obj = self._objects[label]
            try:
                obj.measure(image)
            except Exception as e:
                if obj.result is None:
                    obj.result = GeometricResult(
                        type=obj.result_type(),
                        label=label,
                        valid=False,
                        meta={"error": str(e), "reason": "measurement raised exception"},
                    )

        self._results = {
            label: self._objects[label].result
            for label in self._registration_order
            if self._objects[label].result is not None
        }
        return self._results

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    @property
    def results(self) -> Dict[str, GeometricResult]:
        """Results from the last measure() call."""
        return self._results

    def get_result(self, label: str) -> Optional[GeometricResult]:
        """
        Get a single object's result by label.

        Returns None if the label is unknown or hasn't been measured yet.
        """
        obj = self._objects.get(label)
        if obj is None:
            return None
        return obj.result

    @property
    def is_valid(self) -> bool:
        """Check if all objects produced valid results."""
        if not self._results:
            return False
        return all(r.valid for r in self._results.values())

    @property
    def transform(self) -> Optional[SimilarityTransform]:
        """The localization transform from the last measure() call."""
        return self._transform

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def visualize(self, image: np.ndarray,
                  show_objects: Optional[List[str]] = None,
                  wait_ms: int = -1,
                  **kwargs) -> np.ndarray:
        """
        Render measurement objects onto the image in execution order.

        Args:
            image: Input image (grayscale or BGR).
            show_objects: Optional list of object labels to render.
                          If None, renders all objects.
            wait_ms: If >= 0, display the result in an OpenCV window
                     for wait_ms milliseconds. -1 means no display.
            **kwargs: Forwarded to each object's visualize() method
                      (e.g. show_labels, line_thickness, point_radius).

        Returns:
            Annotated BGR image.
        """
        if len(image.shape) == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()

        # Filter objects
        order = self._execution_order or self._registration_order
        if show_objects is not None:
            show_set = set(show_objects)
            order = [l for l in order if l in show_set]

        for label in order:
            obj = self._objects[label]
            vis = obj.visualize(vis, **kwargs)

        # Draw localization status
        if self._transform is not None and self._transform.is_valid:
            loc_objs = self._get_localization_objects()
            status = (
                f"Localization: {len(loc_objs)} templates, "
                f"dx={self._transform.translation_col:.1f}, "
                f"dy={self._transform.translation_row:.1f}, "
                f"rot={np.degrees(self._transform.rotation):.1f}deg, "
                f"s={self._transform.scale:.3f}"
            )
        else:
            status = "Localization: not performed"
        cv2.putText(
            vis,
            status,
            (10, vis.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        if wait_ms >= 0:
            cv2.imshow("MeasurementWorkflow.visualize", vis)
            cv2.waitKey(wait_ms)

        return vis

        return vis

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, filepath: str) -> None:
        """
        Serialize the entire workflow to a compressed .npz file.

        The file contains:
          - workflow_meta: JSON string with object graph configuration
          - For each TemplatePointObject:
              {label}_edge_template: preprocessed template array
              {label}_actual_crop_bounds: crop bounds array

        Raises:
            ValueError: If a preprocessor type is not registered.
        """
        if not self._resolved:
            self.resolve()

        meta: Dict[str, Any] = {
            "version": 1,
            "registration_order": self._registration_order,
            "objects": {},
        }

        save_dict: Dict[str, Any] = {}
        for label in self._registration_order:
            obj = self._objects[label]
            obj_data = obj.to_dict()

            # Extract binary template data for TemplatePointObject
            if isinstance(obj, TemplatePointObject):
                tp = obj._template_point
                save_dict[f"{label}_edge_template"] = tp.edge_template
                save_dict[f"{label}_actual_crop_bounds"] = np.array(
                    tp._actual_crop_bounds, dtype=np.int32
                )
                obj_data["actual_crop_bounds"] = list(tp._actual_crop_bounds)

            # Store dependency edges
            obj_data["_input_labels"] = obj._input_labels
            meta["objects"][label] = obj_data

        save_dict["workflow_meta"] = np.array([json.dumps(meta)], dtype=np.object_)
        np.savez_compressed(filepath, **save_dict)

    @classmethod
    def load(cls, filepath: str) -> "MeasurementWorkflow":
        """
        Deserialize a workflow from a .npz file.

        Args:
            filepath: Path to a .npz file saved by MeasurementWorkflow.save().

        Returns:
            A fully reconstructed MeasurementWorkflow ready for measure().

        Raises:
            ValueError: If the file contains an unknown object type.
        """
        data = np.load(filepath, allow_pickle=True)
        meta_raw = data["workflow_meta"]
        if isinstance(meta_raw, np.ndarray) and meta_raw.dtype == np.object_:
            meta = json.loads(str(meta_raw[0]))
        else:
            meta = json.loads(str(meta_raw))

        if meta.get("version", 0) != 1:
            raise ValueError(
                f"Unsupported workflow file version: {meta.get('version')}. "
                f"Expected version 1."
            )

        wf = cls()

        for label in meta["registration_order"]:
            obj_data = meta["objects"][label]
            object_type = obj_data.pop("object_type")
            input_labels = obj_data.pop("_input_labels", [])

            # Restore binary template data for TemplatePointObject
            if object_type == "TemplatePointObject":
                template_key = f"{label}_edge_template"
                bounds_key = f"{label}_actual_crop_bounds"
                if template_key in data:
                    obj_data["edge_template"] = data[template_key]
                if bounds_key in data:
                    obj_data["actual_crop_bounds"] = data[bounds_key].tolist()

            cls_type = _OBJECT_TYPE_REGISTRY.get(object_type)
            if cls_type is None:
                raise ValueError(
                    f"Unknown object type '{object_type}' for label '{label}'. "
                    f"Known types: {list(_OBJECT_TYPE_REGISTRY.keys())}"
                )

            obj = cls_type.from_dict(label, obj_data)
            obj._input_labels = input_labels
            wf._objects[label] = obj
            wf._registration_order.append(label)

        return wf
