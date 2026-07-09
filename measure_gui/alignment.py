"""
Alignment strategies for ROI extraction and straightening.

Supports two modes:
  - SingleBoxAlignment: rigid (rotation + translation) via crop_and_straighten
  - MultiPointAlignment: affine refinement via control point matching

Both implement the AlignmentStrategy interface so downstream measurement
code works identically regardless of the alignment mode.

Architecture:
                ┌── SingleBoxAlignment ──→ crop_and_straighten (rigid)
  AlignmentStrategy ──┤
                └── MultiPointAlignment ──→ parent match → control-point
                                              refinement → getAffineTransform
"""

from __future__ import annotations

#from measure_api.config import Config
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from measure.measure_template import (
    Preprocessor,
    RawPreprocessor,
    TemplatePoint,
    _PREPROCESSOR_REGISTRY,
    _deserialize_preprocessor,
)

from .utils import crop_and_straighten


# =============================================================================
# AlignResult
# =============================================================================


@dataclass
class AlignResult:
    """Result of aligning a target ROI in an inspection image.

    Attributes:
        patch: Straightened (and optionally affine-refined) image patch.
        M_inv: 2x3 affine matrix mapping original image coords → patch coords.
               Use :func:`map_point_via_affine` to map back.
        control_matches: Per-control-point (row, col) match results, or None
                         for non-multi-point modes.
    """

    patch: np.ndarray
    M_inv: np.ndarray  # 2x3, original → patch
    control_matches: Optional[List[Optional[Tuple[float, float]]]] = None


# =============================================================================
# AlignmentStrategy
# =============================================================================


class AlignmentStrategy:
    """Abstract base for ROI alignment strategies.

    Each strategy handles two phases:

    **Teaching** (:meth:`teach`):
        Store the reference geometry and create the parent template for
        coarse localisation.

    **Inspection** (:meth:`align`):
        Given a matched target location (from the parent template),
        produce the straightened patch and the coordinate mapping needed
        to project measurement results back to the inspection image.
    """

    def __init__(self):
        self._box_center: Tuple[float, float] = (0.0, 0.0)
        self._box_size: Tuple[float, float] = (0.0, 0.0)
        self._box_angle_deg: float = 0.0
        self._template_size: int = 0
        self._template_image: Optional[np.ndarray] = None
        self._template_point: Optional[TemplatePoint] = None
        self._reference_image: Optional[np.ndarray] = None
        self._preprocessor: Optional[Preprocessor] = None

    # -- properties -----------------------------------------------------------

    @property
    def template_point(self) -> Optional[TemplatePoint]:
        """The parent TemplatePoint used for coarse multi-target matching."""
        return self._template_point

    @property
    def template_image(self) -> Optional[np.ndarray]:
        """The straightened template image (upright, deskewed)."""
        return self._template_image

    @property
    def box_center(self) -> Tuple[float, float]:
        return self._box_center

    @property
    def box_size(self) -> Tuple[float, float]:
        return self._box_size

    @property
    def box_angle_deg(self) -> float:
        return self._box_angle_deg

    # -- teaching -------------------------------------------------------------

    def teach(
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
    ):
        """Store the reference geometry and create the parent template.

        Subclasses should call ``super().teach(...)`` and then add their
        own initialisation.
        """
        self._reference_image = reference_image.copy()
        self._box_center = (float(center[0]), float(center[1]))
        self._box_size = (float(size[0]), float(size[1]))
        self._box_angle_deg = float(angle_deg)
        self._preprocessor = preprocessor

        template_size = int(max(size))
        self._template_size = template_size
        self._template_image, _ = crop_and_straighten(
            reference_image, center, size, angle_deg
        )

        self._template_point = TemplatePoint(
            reference_image,
            click_row=center[0],
            click_col=center[1],
            template_size=template_size,
            preprocessor=preprocessor,
            match_score_threshold=match_score_threshold,
            use_subpixel=True,
            rotation_invariant=True,
            angle_range=angle_range,
            angle_step=angle_step,
            coarse_fine=coarse_fine,
            coarse_angle_step=coarse_angle_step,
            multi_target=True,
            max_matches=max_matches,
            overlap=overlap,
        )

        self._angle_range = angle_range
        self._angle_step = angle_step
        self._max_matches = max_matches
        self._overlap = float(overlap)
        self._coarse_fine = coarse_fine
        self._coarse_angle_step = coarse_angle_step
        self._coarse_angle_step = coarse_angle_step

    # -- inspection -----------------------------------------------------------

    def align(
        self,
        inspection_image: np.ndarray,
        matched_row: float,
        matched_col: float,
        rotation_deg: float,
    ) -> AlignResult:
        """Align and crop the target from the inspection image.

        Subclasses must implement this.
        """
        raise NotImplementedError

    # -- serialization --------------------------------------------------------

    def get_roi_state(self) -> Dict[str, Any]:
        """Return serialisable ROI geometry for ``project.json``."""
        return {
            "alignment_mode": self.mode_name,
            "box_center": list(self._box_center),
            "box_size": list(self._box_size),
            "box_angle_deg": self._box_angle_deg,
        }

    @property
    def mode_name(self) -> str:
        raise NotImplementedError

    @classmethod
    def from_roi_state(
        cls, data: Dict[str, Any], **kwargs
    ) -> "AlignmentStrategy":
        """Restore from a ``get_roi_state()`` dict.

        The caller is responsible for calling :meth:`teach` afterwards
        with the full set of parameters.
        """
        raise NotImplementedError


# =============================================================================
# SingleBoxAlignment
# =============================================================================


class SingleBoxAlignment(AlignmentStrategy):
    """Rigid (rotation + translation) alignment via a single rotated box.

    This is the existing behaviour: :func:`crop_and_straighten` with the
    box geometry and the detected target rotation.
    """

    mode_name = "single_box"

    def align(
        self,
        inspection_image: np.ndarray,
        matched_row: float,
        matched_col: float,
        rotation_deg: float,
    ) -> AlignResult:
        target_angle = self._box_angle_deg + rotation_deg
        patch, M_inv_3x3 = crop_and_straighten(
            inspection_image,
            (matched_row, matched_col),
            self._box_size,
            target_angle,
        )
        return AlignResult(patch=patch, M_inv=M_inv_3x3[:2, :])

    @classmethod
    def from_roi_state(
        cls, data: Dict[str, Any], **kwargs
    ) -> "SingleBoxAlignment":
        return cls()


# =============================================================================
# ControlPoint
# =============================================================================


@dataclass
class ControlPoint:
    """A single alignment control point with its own template matcher.

    Used by :class:`MultiPointAlignment` to refine the affine transform
    after coarse parent-template matching.

    Attributes:
        label: Unique identifier, e.g. ``"cp_0"``.
        ref_row: Row coordinate in the **straightened template** space.
        ref_col: Column coordinate in the **straightened template** space.
        template_size: Side length of the square match template (px).
        preprocessor_data: Serialised preprocessor dict (see
            :class:`~measure_template.Preprocessor`).
        match_score_threshold: Minimum NCC score (0–1).
        angle_range: (min, max) search range in degrees.
        template_point: The :class:`~measure_template.TemplatePoint`
            instance created at teach time (not serialised directly).
    """

    label: str
    ref_row: float
    ref_col: float
    template_size: int = 40
    preprocessor_data: Dict[str, Any] = field(
        default_factory=lambda: {"type": "raw"}
    )
    match_score_threshold: float = 0.5
    angle_range: Tuple[float, float] = (-15.0, 15.0)

    # Set at teach time — not serialised directly
    template_point: Optional[TemplatePoint] = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "ref_row": self.ref_row,
            "ref_col": self.ref_col,
            "template_size": self.template_size,
            "preprocessor_data": self.preprocessor_data,
            "match_score_threshold": self.match_score_threshold,
            "angle_range": list(self.angle_range),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ControlPoint":
        return cls(
            label=data["label"],
            ref_row=float(data["ref_row"]),
            ref_col=float(data["ref_col"]),
            template_size=int(data.get("template_size", 40)),
            preprocessor_data=data.get("preprocessor_data", {"type": "raw"}),
            match_score_threshold=float(
                data.get("match_score_threshold", 0.5)
            ),
            angle_range=tuple(data.get("angle_range", (-15.0, 15.0))),
        )

    def build_template_point(
        self, template_image: np.ndarray
    ) -> TemplatePoint:
        """Create the :class:`TemplatePoint` from the straightened template.

        Args:
            template_image: The straightened template image (output of
                :func:`crop_and_straighten`) on which the control point
                was placed.
        """
        preprocessor = _deserialize_preprocessor(self.preprocessor_data)
        angle_range_half = max(abs(self.angle_range[0]), abs(self.angle_range[1]))
        use_rotation = angle_range_half > 1e-6

        self.template_point = TemplatePoint(
            template_image,
            click_row=self.ref_row,
            click_col=self.ref_col,
            template_size=self.template_size,
            preprocessor=preprocessor,
            match_score_threshold=self.match_score_threshold,
            use_subpixel=True,
            rotation_invariant=use_rotation,
            angle_range=(-angle_range_half, angle_range_half),
            angle_step=1.0,
            multi_target=False,
        )
        return self.template_point


# =============================================================================
# MultiPointAlignment
# =============================================================================


class MultiPointAlignment(AlignmentStrategy):
    """Affine-refined alignment using multiple control points.

    **Teaching phase:**
      1. User draws a rotated box ROI on the reference image (same as
         :class:`SingleBoxAlignment`).
      2. ``crop_and_straighten`` creates the straightened template.
      3. User clicks ≥ 3 control points **on the straightened template**.
      4. Each control point becomes a small :class:`TemplatePoint` for
         local matching.

    **Inspection phase** (single-target only):
      1. Parent template match → coarse (row, col, rotation).
      2. Rigid crop via ``crop_and_straighten``.
      3. Match each control point on the rigid-cropped patch.
      4. Compute 2D affine transform from (ref_pts → matched_pts) via
         ``cv2.estimateAffinePartial2D``.
      5. Rewarp the inspection image with the affine-refined matrix.
      6. Return the refined patch + M_inv.

    .. note::
       This strategy is **only applicable to single-target scenarios**.
       With multiple targets, control-point matches cannot be reliably
       grouped per instance (matching ambiguity).
    """

    mode_name = "multi_point"

    def __init__(self):
        super().__init__()
        self._control_points: List[ControlPoint] = []

    # -- control point management ---------------------------------------------

    @property
    def control_points(self) -> List[ControlPoint]:
        return self._control_points

    def add_control_point(
        self,
        label: str,
        ref_row: float,
        ref_col: float,
        template_size: int = 40,
        preprocessor_data: Optional[Dict[str, Any]] = None,
        match_score_threshold: float = 0.5,
        angle_range: Tuple[float, float] = (-15.0, 15.0),
    ) -> ControlPoint:
        """Add a control point (in straightened-template coordinates)."""
        cp = ControlPoint(
            label=label,
            ref_row=float(ref_row),
            ref_col=float(ref_col),
            template_size=template_size,
            preprocessor_data=preprocessor_data or {"type": "raw"},
            match_score_threshold=match_score_threshold,
            angle_range=angle_range,
        )
        if self._template_image is not None:
            cp.build_template_point(self._template_image)
        self._control_points.append(cp)
        return cp

    def remove_control_point(self, label: str) -> bool:
        """Remove a control point by label. Returns True if found."""
        for i, cp in enumerate(self._control_points):
            if cp.label == label:
                self._control_points.pop(i)
                return True
        return False

    def clear_control_points(self):
        self._control_points.clear()

    # -- teaching -------------------------------------------------------------

    def build_all_control_point_templates(self):
        """(Re-)build TemplatePoint instances for every control point.

        Call after :meth:`teach` when ``_template_image`` is available.
        """
        if self._template_image is None:
            return
        for cp in self._control_points:
            cp.build_template_point(self._template_image)

    # -- inspection -----------------------------------------------------------

    def align(
        self,
        inspection_image: np.ndarray,
        matched_row: float,
        matched_col: float,
        rotation_deg: float,
    ) -> AlignResult:
        """Align using coarse parent match + control-point affine refinement.

        Falls back to rigid alignment when fewer than 3 control points
        match successfully.
        """
        # Stage 1: rigid crop (same as SingleBoxAlignment)
        target_angle = self._box_angle_deg + rotation_deg
        rigid_patch, M_rigid_3x3 = crop_and_straighten(
            inspection_image,
            (matched_row, matched_col),
            self._box_size,
            target_angle,
        )
        M_rigid_2x3 = M_rigid_3x3[:2, :]

        # Stage 2: match control points on the rigid patch
        if not self._control_points:
            return AlignResult(patch=rigid_patch, M_inv=M_rigid_2x3)

        src_pts = []  # reference positions in straightened template
        dst_pts = []  # matched positions in rigid patch
        control_matches: List[Optional[Tuple[float, float]]] = []

        for cp in self._control_points:
            if cp.template_point is None:
                control_matches.append(None)
                continue
            result = cp.template_point.measure(rigid_patch)
            if result.get("valid", False):
                mr = float(result["matched_row"])
                mc = float(result["matched_col"])
                src_pts.append([cp.ref_col, cp.ref_row])  # OpenCV: (x, y) = (col, row)
                dst_pts.append([mc, mr])
                control_matches.append((mr, mc))
            else:
                control_matches.append(None)

        # Stage 3: compute affine refinement
        MIN_AFFINE_POINTS = 3
        if len(src_pts) >= MIN_AFFINE_POINTS:
            src_arr = np.array(src_pts, dtype=np.float32)
            dst_arr = np.array(dst_pts, dtype=np.float32)

            # estimateAffinePartial2D: similarity with optional anisotropic
            # scaling. For full 6-DOF affine use estimateAffine2D.
            M_refine, inliers = cv2.estimateAffinePartial2D(
                src_arr, dst_arr, method=cv2.RANSAC,
                ransacReprojThreshold=3.0,
            )

            if M_refine is not None and inliers is not None and inliers.sum() >= MIN_AFFINE_POINTS:
                # Compose: M_final = M_refine @ M_rigid
                # M_rigid maps inspection → rigid_patch
                # M_refine maps rigid_patch → refined_patch (src→dst)
                M_final = _compose_affine_2x3(M_refine, M_rigid_2x3)

                # Rewarp with the refined transform
                h, w = int(self._box_size[0]), int(self._box_size[1])
                refined_patch = cv2.warpAffine(
                    inspection_image,
                    M_final,
                    (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                return AlignResult(
                    patch=refined_patch,
                    M_inv=M_final,
                    control_matches=control_matches,
                )

        # Fallback: use rigid result
        return AlignResult(
            patch=rigid_patch,
            M_inv=M_rigid_2x3,
            control_matches=control_matches,
        )

    # -- serialization --------------------------------------------------------

    def get_roi_state(self) -> Dict[str, Any]:
        state = super().get_roi_state()
        state["control_points"] = [
            cp.to_dict() for cp in self._control_points
        ]
        return state

    @classmethod
    def from_roi_state(
        cls, data: Dict[str, Any], **kwargs
    ) -> "MultiPointAlignment":
        strategy = cls()
        for cp_data in data.get("control_points", []):
            strategy._control_points.append(
                ControlPoint.from_dict(cp_data)
            )
        return strategy


# =============================================================================
# Alignment helper: compose two 2x3 affine matrices
# =============================================================================


def _compose_affine_2x3(
    M1: np.ndarray, M2: np.ndarray
) -> np.ndarray:
    """Compose two 2x3 affine matrices: ``result = M1 @ M2``.

    Both matrices map **original → warped**, i.e. they are inverse
    transforms (like the M returned by ``cv2.invertAffineTransform``).
    The composition ``M1 @ M2`` means: first apply M2, then M1.

    Args:
        M1: 2x3 affine matrix.
        M2: 2x3 affine matrix.

    Returns:
        2x3 composed affine matrix.
    """
    M1_3x3 = np.eye(3, dtype=np.float64)
    M1_3x3[:2, :] = M1
    M2_3x3 = np.eye(3, dtype=np.float64)
    M2_3x3[:2, :] = M2
    result_3x3 = M1_3x3 @ M2_3x3
    return result_3x3[:2, :].astype(np.float64)


# =============================================================================
# Registry — maps mode_name → AlignmentStrategy subclass
# =============================================================================

_ALIGNMENT_STRATEGY_REGISTRY: Dict[str, type] = {
    "single_box": SingleBoxAlignment,
    "multi_point": MultiPointAlignment,
}


def strategy_from_roi_state(
    data: Dict[str, Any],
    reference_image: np.ndarray,
) -> AlignmentStrategy:
    """Restore an AlignmentStrategy from a serialised ROI state dict.

    Args:
        data: Dict from ``AlignmentStrategy.get_roi_state()``.
        reference_image: The reference image (needed for ``teach()``).

    Returns:
        A new strategy instance.  The caller must call ``teach()``
        afterwards with the full parameter set.
    """
    mode = data.get("alignment_mode", "single_box")
    cls_type = _ALIGNMENT_STRATEGY_REGISTRY.get(mode)
    if cls_type is None:
        raise ValueError(
            f"Unknown alignment mode: '{mode}'. "
            f"Known modes: {list(_ALIGNMENT_STRATEGY_REGISTRY.keys())}"
        )
    return cls_type.from_roi_state(data)
