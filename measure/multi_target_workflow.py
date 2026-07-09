"""
Multi-Target Measurement Workflow.

Provides a "teach once, measure many targets" pipeline:

1. Draw a rotated bounding box on a reference image → crop & rectify → upright template
2. Define measurement tools on the upright template (lines, circles, edge points, etc.)
3. On a new inspection image: detect ALL target instances (position + angle + scale)
4. For each detected target: crop + rectify → run measurements → transform results back
5. Output per-target measurement results

Usage:
    from multi_target_workflow import MultiTargetWorkflow

    # Teach
    mtw = MultiTargetWorkflow()
    mtw.set_template(ref_image, center_row=200, center_col=300,
                     bbox_width=80, bbox_height=60, bbox_angle_deg=15)

    # Define measurements on template (template-local pixel coordinates)
    mtw.add_line_measure("top_edge", start=(20, 10), end=(20, 70),
                         measure_length1=10, measure_length2=5)
    mtw.add_circle_measure("hole", center=(40, 40), radius=15,
                           radius_min=10, radius_max=20,
                           measure_length1=10, measure_length2=3)

    # Inspect
    results = mtw.inspect(inspection_image)
    for tr in results:
        print(f"Target {tr.target.index}: ({tr.target.row:.1f}, {tr.target.col:.1f})")
        line = tr.get("top_edge")
        if line:
            print(f"  Line length: {line.length:.2f}px")
"""

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np

from measure.measure_template import TemplatePoint, RawPreprocessor, Preprocessor
from measure.constants import EPS
from measure.viz import to_bgr, draw_text_shadow
from measure.measure_workflow import (
    MeasurementWorkflow,
    PointResult,
    LineResult,
    CircleResult,
    GeometricResult,
    FitLineObject,
    FitCircleObject,
)


# ===========================================================================
# Data classes
# ===========================================================================


@dataclass
class TargetInstance:
    """A single detected target instance."""
    index: int                              # target number 0,1,2,...
    row: float                              # center row in inspection image
    col: float                              # center col in inspection image
    angle_deg: float                        # detected rotation (relative to upright template)
    scale: float                            # detected scale
    match_score: float                      # NCC score
    rectified_image: Optional[np.ndarray] = None  # rectified sub-image (template-size)


@dataclass
class MeasureDef:
    """A measurement tool definition in template-local pixel coordinates."""
    measure_type: str        # 'edge_point' | 'edge_pair' | 'fit_line' | 'fit_circle'
    label: str               # unique label, e.g. "left_edge", "hole_center"
    params: Dict[str, Any] = field(default_factory=dict)
    # Example params for FitLine:
    #   {'start': (row1, col1), 'end': (row2, col2),
    #    'measure_length1': ..., 'measure_length2': ...,
    #    'num_measures': ..., 'sigma': ..., 'threshold': ..., 'transition': ...}
    # Example params for FitCircle:
    #   {'center': (row, col), 'radius': ...,
    #    'radius_min': ..., 'radius_max': ...,
    #    'measure_length1': ..., 'measure_length2': ...,
    #    'num_measures': ..., 'sigma': ..., 'threshold': ..., 'transition': ...}
    # Example params for EdgePoint/EdgePair:
    #   {'row': ..., 'col': ..., 'angle': ...,
    #    'length1': ..., 'length2': ...,
    #    'sigma': ..., 'threshold': ..., 'transition': ..., 'select': ...}


@dataclass
class TargetResult:
    """Complete measurement results for a single detected target."""
    target: TargetInstance
    measurements: Dict[str, GeometricResult] = field(default_factory=dict)

    def get(self, label: str) -> Optional[GeometricResult]:
        """Get a measurement result by label."""
        return self.measurements.get(label)

    @property
    def all_valid(self) -> bool:
        """Check if all measurements are valid."""
        return all(r.valid for r in self.measurements.values())


# ===========================================================================
# Coordinate transform helpers
# ===========================================================================


def _transform_point_result(
    result: PointResult,
    target: TargetInstance,
    template_h: int,
    template_w: int,
) -> PointResult:
    """Transform a PointResult from template-local coords to inspection image coords."""
    dx = result.col - template_w / 2.0
    dy = result.row - template_h / 2.0

    rad = math.radians(target.angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    new_dx = target.scale * (dx * cos_a - dy * sin_a)
    new_dy = target.scale * (dx * sin_a + dy * cos_a)

    return PointResult(
        label=result.label,
        row=target.row + new_dy,
        col=target.col + new_dx,
        valid=result.valid,
        meta={
            **result.meta,
            'template_row': result.row,
            'template_col': result.col,
        },
    )


def _transform_line_result(
    result: LineResult,
    target: TargetInstance,
    template_h: int,
    template_w: int,
) -> LineResult:
    """Transform a LineResult from template-local coords to inspection image coords.

    Transforms both endpoints and re-derives the line equation from them.
    """
    # Transform start point
    start_dx = result.start_col - template_w / 2.0
    start_dy = result.start_row - template_h / 2.0
    rad = math.radians(target.angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    new_start_dx = target.scale * (start_dx * cos_a - start_dy * sin_a)
    new_start_dy = target.scale * (start_dx * sin_a + start_dy * cos_a)
    new_start_row = target.row + new_start_dy
    new_start_col = target.col + new_start_dx

    # Transform end point
    end_dx = result.end_col - template_w / 2.0
    end_dy = result.end_row - template_h / 2.0
    new_end_dx = target.scale * (end_dx * cos_a - end_dy * sin_a)
    new_end_dy = target.scale * (end_dx * sin_a + end_dy * cos_a)
    new_end_row = target.row + new_end_dy
    new_end_col = target.col + new_end_dx

    # Recompute line equation a*row + b*col + c = 0 from endpoints
    # Line direction
    dr = new_end_row - new_start_row
    dc = new_end_col - new_start_col
    length = math.sqrt(dr * dr + dc * dc)
    if length < EPS:
        a, b, c = 0.0, 0.0, 0.0
    else:
        # Normal vector: (dc, -dr) / length gives direction perpendicular to line
        a = -dc   # coefficient for row
        b = dr    # coefficient for col
        # c = -(a*row + b*col) at any point on the line
        c = -(a * new_start_row + b * new_start_col)
        # Normalize
        norm = math.sqrt(a * a + b * b)
        if norm > EPS:
            a /= norm
            b /= norm
            c /= norm

    return LineResult(
        label=result.label,
        a=a, b=b, c=c,
        start_row=new_start_row, start_col=new_start_col,
        end_row=new_end_row, end_col=new_end_col,
        valid=result.valid,
        meta={
            **result.meta,
            'template_start_row': result.start_row,
            'template_start_col': result.start_col,
            'template_end_row': result.end_row,
            'template_end_col': result.end_col,
        },
    )


def _transform_circle_result(
    result: CircleResult,
    target: TargetInstance,
    template_h: int,
    template_w: int,
) -> CircleResult:
    """Transform a CircleResult from template-local coords to inspection image coords.

    Center is transformed as a point; radius is scaled.
    """
    dx = result.center_col - template_w / 2.0
    dy = result.center_row - template_h / 2.0

    rad = math.radians(target.angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    new_dx = target.scale * (dx * cos_a - dy * sin_a)
    new_dy = target.scale * (dx * sin_a + dy * cos_a)

    return CircleResult(
        label=result.label,
        center_row=target.row + new_dy,
        center_col=target.col + new_dx,
        radius=result.radius * target.scale,
        valid=result.valid,
        meta={
            **result.meta,
            'template_center_row': result.center_row,
            'template_center_col': result.center_col,
            'template_radius': result.radius,
        },
    )


# Mapping from result type to transform function
_TRANSFORM_MAP = {
    'point': _transform_point_result,
    'line': _transform_line_result,
    'circle': _transform_circle_result,
}


# ===========================================================================
# RotatedTemplate
# ===========================================================================


class RotatedTemplate:
    """
    A template defined by a rotated bounding box on a reference image.

    Crops the rotated rectangle, rectifies it to upright, and stores it
    as the template for multi-target matching.

    Internally uses TemplatePoint(multi_target=True, rotation_invariant=True)
    for detecting all instances in inspection images.
    """

    def __init__(
        self,
        reference_image: np.ndarray,
        center_row: float,
        center_col: float,
        bbox_width: float,
        bbox_height: float,
        bbox_angle_deg: float,
        preprocessor: Optional[Preprocessor] = None,
        match_score_threshold: float = 0.5,
        angle_range: Tuple[float, float] = (-30.0, 30.0),
        angle_step: float = 1.0,
        scale_invariant: bool = False,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        scale_step: float = 0.02,
        max_matches: int = 0,
    ):
        """
        Initialize a rotated template from a reference image.

        Args:
            reference_image: Grayscale or BGR reference image (uint8).
            center_row: Row of the rotated bbox center.
            center_col: Column of the rotated bbox center.
            bbox_width: Width of the bbox (column direction before rotation).
            bbox_height: Height of the bbox (row direction before rotation).
            bbox_angle_deg: Rotation angle of the bbox in degrees
                           (OpenCV convention: positive = CCW).
            preprocessor: Preprocessor for template edge enhancement.
            match_score_threshold: Minimum NCC score for valid match.
            angle_range: (min, max) search range in degrees.
            angle_step: Fine angle search step in degrees.
            scale_invariant: Enable multi-scale search.
            scale_range: (min, max) scale factor range.
            scale_step: Fine scale search step.
            max_matches: Maximum matches to return (0 = unlimited).
        """
        self._center_row = center_row
        self._center_col = center_col
        self._bbox_width = bbox_width
        self._bbox_height = bbox_height
        self._bbox_angle_deg = bbox_angle_deg
        self._match_score_threshold = match_score_threshold
        self._angle_range = angle_range
        self._angle_step = angle_step
        self._scale_invariant = scale_invariant
        self._scale_range = scale_range
        self._scale_step = scale_step
        self._max_matches = max_matches
        self._preprocessor = preprocessor

        # Convert to grayscale
        if len(reference_image.shape) == 3:
            gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = reference_image.copy()

        # Crop and rectify
        self.template_image, self._M_rectify = self._crop_and_rectify(
            gray, center_row, center_col, bbox_width, bbox_height, bbox_angle_deg,
        )
        self._th, self._tw = self.template_image.shape

        if self._th <= 0 or self._tw <= 0:
            raise ValueError(
                f"Template is empty after rectification. Check bbox parameters: "
                f"center=({center_row:.0f},{center_col:.0f}), "
                f"size=({bbox_width:.0f},{bbox_height:.0f}), "
                f"angle={bbox_angle_deg:.1f}°"
            )

        # Create internal TemplatePoint for matching
        # click is at the center of the upright template
        template_click_row = self._th / 2.0
        template_click_col = self._tw / 2.0
        template_size = min(self._th, self._tw)

        self._template_point = TemplatePoint(
            self.template_image,
            click_row=template_click_row,
            click_col=template_click_col,
            template_size=template_size,
            preprocessor=preprocessor,
            match_score_threshold=match_score_threshold,
            use_subpixel=True,
            rotation_invariant=True,
            angle_range=angle_range,
            angle_step=angle_step,
            scale_invariant=scale_invariant,
            scale_range=scale_range,
            scale_step=scale_step,
            coarse_fine=True,
            multi_target=True,
            max_matches=max_matches,
        )

    @staticmethod
    def _crop_and_rectify(
        image: np.ndarray,
        center_row: float,
        center_col: float,
        w: float,
        h: float,
        angle_deg: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crop a rotated rectangle from an image and rectify it to upright.

        Uses perspective transform to map the 4 corners of the rotated bbox
        to a target upright rectangle.

        Args:
            image: Input grayscale image.
            center_row, center_col: Center of the rotated bbox.
            w: Width of the bbox (column direction before rotation).
            h: Height of the bbox (row direction before rotation).
            angle_deg: Rotation angle in degrees (OpenCV convention).

        Returns:
            (rectified_image, M) where rectified_image is the upright crop
            and M is the 3×3 perspective transform matrix used.
        """
        # Build rotated rectangle in OpenCV format
        rect = ((float(center_col), float(center_row)),
                (float(w), float(h)), float(angle_deg))
        box = cv2.boxPoints(rect)  # 4 corner points

        # Target: upright rectangle of the same size
        dst_w = max(1, int(w))
        dst_h = max(1, int(h))
        src_pts = box.astype(np.float32)
        dst_pts = np.float32([
            [0, 0],
            [dst_w - 1, 0],
            [dst_w - 1, dst_h - 1],
            [0, dst_h - 1],
        ])

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        rectified = cv2.warpPerspective(
            image, M, (dst_w, dst_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=int(np.mean(image)),
        )
        return rectified, M

    def detect_all(self, inspection_image: np.ndarray) -> List[TargetInstance]:
        """
        Detect all target instances in an inspection image.

        Args:
            inspection_image: Grayscale or BGR inspection image (uint8).

        Returns:
            List of TargetInstance, sorted by match_score descending.
        """
        result = self._template_point.measure(inspection_image, search_region=None)
        matches = result.get('matches', [])

        targets = []
        for i, m in enumerate(matches):
            # Rectify the detected target sub-image
            rectified = self._rectify_target(
                inspection_image, m['matched_row'], m['matched_col'],
                m.get('best_rotation_deg', 0.0), m.get('best_scale', 1.0),
            )

            targets.append(TargetInstance(
                index=i,
                row=m['matched_row'],
                col=m['matched_col'],
                angle_deg=m.get('best_rotation_deg', 0.0),
                scale=m.get('best_scale', 1.0),
                match_score=m['match_score'],
                rectified_image=rectified,
            ))
        return targets

    def _rectify_target(
        self,
        image: np.ndarray,
        center_row: float,
        center_col: float,
        angle_deg: float,
        scale: float,
    ) -> np.ndarray:
        """
        Crop and rectify a detected target from the inspection image.

        Takes the target's detected position, rotation, and scale,
        applies the inverse of the teach-time rectification plus the
        target's own rotation, to produce an upright sub-image that
        matches the template.

        Args:
            image: Inspection image.
            center_row, center_col: Target center in image coords.
            angle_deg: Detected rotation (relative to upright template).
            scale: Detected scale factor.

        Returns:
            Rectified sub-image with the same dimensions as the template.
        """
        # The target in the inspection image:
        # 1. Start with the upright template at origin
        # 2. Rotate by (teach_angle + detected_angle) + scale by detected_scale
        # 3. Translate to detected center

        # Compute the rotated bbox corners in the inspection image
        h, w = self._th, self._tw
        total_angle = self._bbox_angle_deg + angle_deg

        # Build the rotated rectangle at the target location
        rect = ((float(center_col), float(center_row)),
                (float(w) * scale, float(h) * scale),
                float(total_angle))
        box = cv2.boxPoints(rect)

        # Source: 4 corners of rotated target in image
        # Destination: upright rectangle matching template dimensions
        src_pts = box.astype(np.float32)
        dst_pts = np.float32([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1],
        ])

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        rectified = cv2.warpPerspective(
            gray, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=int(np.mean(gray)),
        )
        return rectified


# ===========================================================================
# MultiTargetWorkflow
# ===========================================================================


class MultiTargetWorkflow:
    """
    Complete "teach once, measure many targets" workflow.

    Teach phase:
      - Set a rotated-bbox template on a reference image
      - Define measurement tools in template-local coordinates

    Inspect phase:
      - Detect all target instances
      - For each target: rectify → run measurements → transform results
      - Return per-target results

    Usage:
        mtw = MultiTargetWorkflow()
        mtw.set_template(ref, 200, 300, 80, 60, 15)
        mtw.add_line_measure("edge1", start=(20, 10), end=(20, 70),
                             measure_length1=10, measure_length2=5)
        results = mtw.inspect(inspection_image)
    """

    def __init__(self):
        self._template: Optional[RotatedTemplate] = None
        self._measure_defs: List[MeasureDef] = []
        self._template_params: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Teach
    # ------------------------------------------------------------------

    def set_template(
        self,
        reference_image: np.ndarray,
        center_row: float,
        center_col: float,
        bbox_width: float,
        bbox_height: float,
        bbox_angle_deg: float,
        preprocessor: Optional[Preprocessor] = None,
        match_score_threshold: float = 0.5,
        angle_range: Tuple[float, float] = (-30.0, 30.0),
        angle_step: float = 1.0,
        scale_invariant: bool = False,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        scale_step: float = 0.02,
        max_matches: int = 0,
    ) -> None:
        """
        Set the rotated-bbox template from a reference image.

        Args:
            reference_image: Reference image (grayscale or BGR).
            center_row, center_col: Center of the rotated bbox in image coords.
            bbox_width, bbox_height: Width and height of the bbox.
            bbox_angle_deg: Rotation of the bbox in degrees.
            preprocessor: Preprocessor for template enhancement.
            match_score_threshold: Minimum NCC score for valid match.
            angle_range: (min, max) search range in degrees.
            angle_step: Fine angle search step.
            scale_invariant: Enable multi-scale search.
            scale_range: (min, max) scale range.
            scale_step: Fine scale step.
            max_matches: Maximum matches to return (0 = unlimited).
        """
        self._template_params = {
            'center_row': center_row,
            'center_col': center_col,
            'bbox_width': bbox_width,
            'bbox_height': bbox_height,
            'bbox_angle_deg': bbox_angle_deg,
            'match_score_threshold': match_score_threshold,
            'angle_range': list(angle_range),
            'angle_step': angle_step,
            'scale_invariant': scale_invariant,
            'scale_range': list(scale_range),
            'scale_step': scale_step,
            'max_matches': max_matches,
        }

        self._template = RotatedTemplate(
            reference_image=reference_image,
            center_row=center_row,
            center_col=center_col,
            bbox_width=bbox_width,
            bbox_height=bbox_height,
            bbox_angle_deg=bbox_angle_deg,
            preprocessor=preprocessor,
            match_score_threshold=match_score_threshold,
            angle_range=angle_range,
            angle_step=angle_step,
            scale_invariant=scale_invariant,
            scale_range=scale_range,
            scale_step=scale_step,
            max_matches=max_matches,
        )

    def add_measurement(self, measure_type: str, label: str, **params) -> None:
        """
        Add a measurement definition in template-local pixel coordinates.

        Supported measure_type values:
          - 'fit_line': Fit a line along a specified segment.
              Required params: start=(row, col), end=(row, col)
              Optional: measure_length1, measure_length2, num_measures,
                        sigma, threshold, transition
          - 'fit_circle': Fit a circle near a specified center/radius.
              Required params: center=(row, col), radius (approximate)
              Optional: radius_min, radius_max, measure_length1, measure_length2,
                        num_measures, sigma, threshold, transition,
                        start_phi, end_phi

        Args:
            measure_type: Type of measurement ('fit_line' or 'fit_circle').
            label: Unique label for this measurement.
            **params: Parameters for the measurement, in template-local
                     pixel coordinates.
        """
        if measure_type not in ('fit_line', 'fit_circle'):
            raise ValueError(
                f"Unsupported measure_type: '{measure_type}'. "
                f"Supported types: 'fit_line', 'fit_circle'"
            )

        # Check for duplicate labels
        for existing in self._measure_defs:
            if existing.label == label:
                raise ValueError(f"Duplicate measurement label: '{label}'")

        self._measure_defs.append(MeasureDef(
            measure_type=measure_type,
            label=label,
            params=params,
        ))

    @property
    def template_image(self) -> Optional[np.ndarray]:
        """Return the upright template image, or None if not set."""
        if self._template is None:
            return None
        return self._template.template_image

    @property
    def template_shape(self) -> Optional[Tuple[int, int]]:
        """Return (height, width) of the template, or None if not set."""
        if self._template is None:
            return None
        return (self._template._th, self._template._tw)

    # ------------------------------------------------------------------
    # Inspect
    # ------------------------------------------------------------------

    def inspect(self, inspection_image: np.ndarray) -> List[TargetResult]:
        """
        Run full detection + measurement on an inspection image.

        1. Multi-target detection → List[TargetInstance]
        2. For each target: rectify sub-image → run measurements → transform
        3. Return List[TargetResult]

        Args:
            inspection_image: Inspection image (grayscale or BGR).

        Returns:
            List of TargetResult, one per detected target.
        """
        if self._template is None:
            raise RuntimeError("Template not set. Call set_template() first.")

        # Step 1: Detect all targets
        targets = self._template.detect_all(inspection_image)

        # Step 2: For each target, build a MeasurementWorkflow on the rectified
        #         sub-image and run all measurements
        results = []
        th, tw = self._template._th, self._template._tw

        for target in targets:
            if target.rectified_image is None:
                continue

            # Build a fresh MeasurementWorkflow for this target
            wf = self._build_target_workflow(target)

            # Run measurements on the rectified sub-image
            wf.measure(target.rectified_image)

            # Collect and transform results
            measurements = {}
            for md in self._measure_defs:
                raw = wf.get_result(md.label)
                if raw is None:
                    continue

                # Transform from template coords to image coords
                transformed = self._transform_result(raw, target, th, tw)
                if transformed is not None:
                    measurements[md.label] = transformed

            results.append(TargetResult(target=target, measurements=measurements))

        return results

    def _build_target_workflow(self, target: TargetInstance) -> MeasurementWorkflow:
        """
        Build a MeasurementWorkflow configured for a single target's
        rectified sub-image.
        """
        wf = MeasurementWorkflow()

        for md in self._measure_defs:
            params = md.params

            if md.measure_type == 'fit_line':
                obj = FitLineObject(
                    label=md.label,
                    start=params['start'],
                    end=params['end'],
                    measure_length1=params.get('measure_length1', 10),
                    measure_length2=params.get('measure_length2', 5),
                    num_measures=params.get('num_measures', 10),
                    sigma=params.get('sigma', 1.0),
                    threshold=params.get('threshold', 30.0),
                    transition=params.get('transition', 'all'),
                )
                wf.add(obj)
            elif md.measure_type == 'fit_circle':
                obj = FitCircleObject(
                    label=md.label,
                    center=params['center'],
                    radius=params['radius'],
                    radius_min=params.get('radius_min', params['radius'] * 0.5),
                    radius_max=params.get('radius_max', params['radius'] * 1.5),
                    measure_length1=params.get('measure_length1', 10),
                    measure_length2=params.get('measure_length2', 5),
                    num_measures=params.get('num_measures', 12),
                    sigma=params.get('sigma', 1.0),
                    threshold=params.get('threshold', 30.0),
                    transition=params.get('transition', 'all'),
                    start_phi=params.get('start_phi', 0.0),
                    end_phi=params.get('end_phi', 2 * math.pi),
                )
                wf.add(obj)

        return wf

    def _transform_result(
        self,
        result: GeometricResult,
        target: TargetInstance,
        th: int,
        tw: int,
    ) -> Optional[GeometricResult]:
        """Transform a measurement result from template coords to image coords."""
        transform_fn = _TRANSFORM_MAP.get(result.type)
        if transform_fn is None:
            return result  # pass through unknown types unchanged
        return transform_fn(result, target, th, tw)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, filepath: str) -> None:
        """
        Save the template, measurement definitions, and parameters to a .npz file.

        Args:
            filepath: Path to the output .npz file.
        """
        if self._template is None:
            raise RuntimeError("Cannot save: template not set.")

        # Serialize measurements
        measure_list = []
        for md in self._measure_defs:
            measure_list.append({
                'measure_type': md.measure_type,
                'label': md.label,
                'params': md.params,
            })

        # Serialize preprocessor if any
        pp = self._template._preprocessor
        pp_data = pp.serialize() if pp else RawPreprocessor().serialize()

        np.savez_compressed(
            filepath,
            version=1,
            template_image=self._template.template_image,
            template_params=json.dumps(self._template_params),
            preprocessor_json=json.dumps(pp_data),
            measure_defs=json.dumps(measure_list),
            edge_template=self._template._template_point.edge_template,
        )

    @classmethod
    def load(cls, filepath: str) -> 'MultiTargetWorkflow':
        """
        Load a MultiTargetWorkflow from a .npz file.

        Args:
            filepath: Path to the .npz file.

        Returns:
            A fully initialized MultiTargetWorkflow ready for inspect().
        """
        data = np.load(filepath, allow_pickle=False)

        version = int(data['version'])
        if version != 1:
            raise ValueError(f"Unsupported file version: {version}")

        template_image = data['template_image']
        template_params = json.loads(data['template_params'].item() if
                                     isinstance(data['template_params'], np.ndarray)
                                     else str(data['template_params']))
        measure_list = json.loads(data['measure_defs'].item() if
                                  isinstance(data['measure_defs'], np.ndarray)
                                  else str(data['measure_defs']))

        # Reconstruct preprocessor
        pp_json = data.get('preprocessor_json')
        if pp_json is not None:
            pp_str = str(pp_json) if isinstance(pp_json, np.ndarray) else pp_json
            pp_data = json.loads(pp_str)
            from measure.measure_template import _deserialize_preprocessor
            preprocessor = _deserialize_preprocessor(pp_data)
        else:
            preprocessor = None

        # Create instance via __new__ to bypass __init__
        obj = cls.__new__(cls)
        obj._template_params = template_params
        obj._measure_defs = []

        # Reconstruct RotatedTemplate directly from the stored template image
        # (the image is already rectified, so we pass it with angle=0 and
        # center at the image center). Also reconstruct the internal
        # TemplatePoint for matching.
        th, tw = template_image.shape
        tp = template_params

        rt = RotatedTemplate.__new__(RotatedTemplate)
        rt._center_row = tp['center_row']
        rt._center_col = tp['center_col']
        rt._bbox_width = tp['bbox_width']
        rt._bbox_height = tp['bbox_height']
        rt._bbox_angle_deg = tp['bbox_angle_deg']
        rt._match_score_threshold = tp.get('match_score_threshold', 0.5)
        rt._angle_range = tuple(tp.get('angle_range', (-30.0, 30.0)))
        rt._angle_step = tp.get('angle_step', 1.0)
        rt._scale_invariant = tp.get('scale_invariant', False)
        rt._scale_range = tuple(tp.get('scale_range', (0.9, 1.1)))
        rt._scale_step = tp.get('scale_step', 0.02)
        rt._max_matches = tp.get('max_matches', 0)
        rt._preprocessor = preprocessor
        rt.template_image = template_image
        rt._th, rt._tw = th, tw
        rt._M_rectify = np.eye(3, dtype=np.float64)  # identity (already rectified)

        # Rebuild TemplatePoint from the stored template image
        template_click_row = th / 2.0
        template_click_col = tw / 2.0
        template_size = min(th, tw)
        rt._template_point = TemplatePoint(
            template_image,
            click_row=template_click_row,
            click_col=template_click_col,
            template_size=template_size,
            preprocessor=preprocessor,
            match_score_threshold=rt._match_score_threshold,
            use_subpixel=True,
            rotation_invariant=True,
            angle_range=rt._angle_range,
            angle_step=rt._angle_step,
            scale_invariant=rt._scale_invariant,
            scale_range=rt._scale_range,
            scale_step=rt._scale_step,
            coarse_fine=True,
            multi_target=True,
            max_matches=rt._max_matches,
        )

        obj._template = rt

        # Restore measurements
        for md_dict in measure_list:
            obj._measure_defs.append(MeasureDef(
                measure_type=md_dict['measure_type'],
                label=md_dict['label'],
                params=md_dict['params'],
            ))

        return obj

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def visualize(
        self,
        image: np.ndarray,
        targets: Optional[List[TargetResult]] = None,
        show_template_box: bool = False,
        show_measurements: bool = True,
        line_thickness: int = 2,
        point_radius: int = 6,
        wait_time: int = -1,
    ) -> np.ndarray:
        """
        Visualize detected targets and measurement results on the inspection image.

        Args:
            image: Inspection image (grayscale or BGR).
            targets: List of TargetResult from inspect(). If None, draws nothing.
            show_template_box: Draw the original teach bbox (only useful on
                              the reference image).
            show_measurements: Draw measurement results for each target.
            line_thickness: Line thickness for drawings.
            point_radius: Radius of matched point markers.
            wait_time: OpenCV waitKey time in ms (-1 = no display).

        Returns:
            Annotated BGR image (copy).
        """
        # Convert to BGR
        vis = to_bgr(image)

        if targets is None:
            targets = []

        # Colour palette
        palette = [
            (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
            (255, 0, 255), (255, 255, 0), (128, 0, 255), (255, 128, 0),
            (0, 128, 255), (128, 255, 0),
        ]

        for tr in targets:
            t = tr.target
            color = palette[t.index % len(palette)]

            # Draw rotated bbox for this target
            h, w = self._template._th, self._template._tw
            total_angle = self._template._bbox_angle_deg + t.angle_deg
            rect = ((float(t.col), float(t.row)),
                    (float(w) * t.scale, float(h) * t.scale),
                    float(total_angle))
            box = cv2.boxPoints(rect)
            box = box.astype(np.int32)
            cv2.polylines(vis, [box], True, (0, 0, 0), line_thickness + 2)
            cv2.polylines(vis, [box], True, color, line_thickness)

            # Draw center crosshair
            r, c = int(t.row), int(t.col)
            cv2.line(vis, (c - point_radius, r), (c + point_radius, r),
                     (0, 0, 0), 3)
            cv2.line(vis, (c - point_radius, r), (c + point_radius, r),
                     color, 2)
            cv2.line(vis, (c, r - point_radius), (c, r + point_radius),
                     (0, 0, 0), 3)
            cv2.line(vis, (c, r - point_radius), (c, r + point_radius),
                     color, 2)

            # Index label
            draw_text_shadow(vis, str(t.index), (c + point_radius + 4, r - point_radius), color=(255, 255, 255), font_scale=0.55, thickness=1)

            # Draw measurements
            if show_measurements:
                for label, result in tr.measurements.items():
                    if not result.valid:
                        continue
                    if result.type == 'line':
                        self._draw_line(vis, result, color, line_thickness)
                    elif result.type == 'circle':
                        self._draw_circle(vis, result, color, line_thickness)
                    elif result.type == 'point':
                        self._draw_point(vis, result, color, point_radius)
                    # Label
                    cv2.putText(vis, f"{t.index}:{label}",
                                (int(result.start_col if result.type == 'line'
                                     else result.col if result.type == 'point'
                                     else result.center_col),
                                 int(result.start_row if result.type == 'line'
                                     else result.row if result.type == 'point'
                                     else result.center_row) - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2)
                    cv2.putText(vis, f"{t.index}:{label}",
                                (int(result.start_col if result.type == 'line'
                                     else result.col if result.type == 'point'
                                     else result.center_col),
                                 int(result.start_row if result.type == 'line'
                                     else result.row if result.type == 'point'
                                     else result.center_row) - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        # Info overlay
        y = 25
        draw_text_shadow(vis, f'Targets: {len(targets)}', (10, y), color=(255, 255, 255), font_scale=0.6, thickness=1)

        if wait_time >= 0:
            cv2.imshow("Multi-Target Workflow", vis)
            cv2.waitKey(wait_time)
            if wait_time > 0:
                cv2.destroyWindow("Multi-Target Workflow")

        return vis

    @staticmethod
    def _draw_line(vis, result, color, thickness):
        """Draw a LineResult."""
        pt1 = (int(result.start_col), int(result.start_row))
        pt2 = (int(result.end_col), int(result.end_row))
        cv2.line(vis, pt1, pt2, (0, 0, 0), thickness + 2)
        cv2.line(vis, pt1, pt2, color, thickness)

    @staticmethod
    def _draw_circle(vis, result, color, thickness):
        """Draw a CircleResult."""
        center = (int(result.center_col), int(result.center_row))
        radius = int(result.radius)
        cv2.circle(vis, center, radius, (0, 0, 0), thickness + 2)
        cv2.circle(vis, center, radius, color, thickness)
        cv2.circle(vis, center, 3, (0, 0, 0), -1)
        cv2.circle(vis, center, 2, color, -1)

    @staticmethod
    def _draw_point(vis, result, color, radius):
        """Draw a PointResult."""
        c = (int(result.col), int(result.row))
        cv2.drawMarker(vis, c, color, cv2.MARKER_CROSS, radius, 2)
