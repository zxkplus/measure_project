"""
Alignment strategies for ROI extraction and straightening.

Modes:
  - NoneAlignment:       no localisation at all — the inspection image *is*
                         the measurement frame (fixed workpiece position)
  - FixedRoiAlignment:   crop a known ROI without searching for it
  - SingleBoxAlignment:  rigid (rotation + translation) via crop_and_straighten
  - MultiPointAlignment: affine refinement via control point matching

All implement the AlignmentStrategy interface so downstream measurement code
works identically regardless of the mode.

Architecture:
                              ┌── NoneAlignment ──────→ identity, full image
                              ├── FixedRoiAlignment ──→ crop, no search
  AlignmentStrategy ──────────┤
                              └── template-matching modes:
                                    SingleBoxAlignment → crop_and_straighten
                                    MultiPointAlignment → parent match →
                                        control-point refinement →
                                        getAffineTransform

Why the first two exist
-----------------------
Template matching is one way to answer "where is the part", but it is not free:
it costs time, can mismatch, and quantises orientation to ``angle_step``.  When
the part's pose is already known — a fixed fixture, or a detector upstream that
already returned a box — searching again is pure loss.  Worse, rectification
*removes* pose from the patch by design, so any measurement that needs the
orientation itself (for example a bearing between two detected centres) cannot be
expressed in a rectified frame.  These two modes make stage 1 explicitly
skippable instead of forcing every model through a template.
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

from measure.obb_source import ObbObservation, select_observations
from measure.rectify import crop_and_straighten


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

    # -- inspection-time angle conversion -------------------------------------

    def absolute_rotation(self, reported_rotation_deg: float) -> float:
        """Turn a match's reported rotation into the target's absolute orientation.

        Stage 1 reports a rotation *from the taught pose*, and the two locator
        families encode that same quantity with opposite signs, so the conversion
        cannot be one shared expression:

        * template matching (this base implementation) reports ``+delta``, i.e.
          ``absolute - taught`` — the angle the template was rotated by to match.
          An image rotated +10 degrees away from the reference therefore reports
          +10, making ``taught + reported`` the absolute orientation.
        * :class:`ObbAlignment` reports ``taught - absolute`` instead, because it
          formats a detector's *absolute* angle into the same field the template
          matcher uses.  It overrides this method.

        Getting the sign wrong is silent rather than loud: at the taught pose the
        reported value is zero for every locator, so a self-match or an
        unrotated inspection image looks perfect and only a rotated part exposes
        the error — as *twice* the deviation, mirrored.  That signature is what
        this method exists to prevent.
        """
        return float(self._box_angle_deg) + float(reported_rotation_deg)

    def crop_angle(self, rotation_deg: float) -> float:
        """Angle to hand to :func:`crop_and_straighten` for this target.

        Derived from :meth:`absolute_rotation` rather than re-derived, so the
        patch frame and the reported pose cannot disagree: straightening means
        rotating the crop *back* by the target's absolute orientation, and
        ``crop_and_straighten`` adds its angle to the content rather than
        removing it.

        Both halves of that sentence have been got wrong before.  Using the box
        angle alone left every target tilted by its own rotation; using the
        *negated* box-delta tilted them by **twice** their rotation, which still
        looks plausible in a small patch and only shows up as a measurement that
        follows the part's tilt instead of the taught direction.  Measured on a
        part rotated 14 degrees: the patch came out at 29 degrees.

        The reason this cannot be one formula at the call sites is the same as
        for :meth:`absolute_rotation` — the two locator families report the
        rotation with opposite signs.
        """
        return -self.absolute_rotation(float(rotation_deg))

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
        pyramid_decimate: int = 0,
        pyramid_max_template_size: int = 400,
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
            reference_image, center, size, _crop_angle(angle_deg)
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
            pyramid_decimate=pyramid_decimate,
            pyramid_max_template_size=pyramid_max_template_size,
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
        box_size: Optional[Tuple[float, float]] = None,
    ) -> AlignResult:
        """Align and crop the target from the inspection image.

        Args:
            inspection_image: Image being inspected.
            matched_row: Row of the detected target centre.
            matched_col: Column of the detected target centre.
            rotation_deg: Target rotation relative to the taught reference.
            box_size: ``(height, width)`` to crop, overriding the strategy's own
                geometry.  Strategies whose targets vary in size per instance —
                notably OBB detection, where every box is measured — need this;
                template-based modes leave it None and use the taught box.

        Returns:
            The straightened patch and its coordinate mapping.

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


def _crop_angle(box_angle_deg: float, rotation_deg: float = 0.0) -> float:
    """Convert a box orientation into the angle ``crop_and_straighten`` expects.

    The two conventions are negatives of each other, which is easy to miss because
    everything still "works" when the box is axis-aligned:

    * ``compute_rotated_box_corners`` (and every box angle in the manifest) is in
      ``(row, col)`` with positive meaning *visually* counter-clockwise, i.e. the
      long edge tilting upwards on screen.
    * ``crop_and_straighten``'s matrix acts on ``(x, y) = (col, row)`` in the
      usual maths sense, so feeding it the box angle rotates the patch the wrong
      way.

    Passing the un-negated angle leaves the part tilted inside its patch and pads
    the rest with background: measured on a 50x120 box, a 20 degree input gave a
    0.65 fill with ``+angle`` versus 1.00 with ``-angle``.  Detected angles came
    out systematically wrong by roughly twice the box angle.

    Kept as a function rather than a sign at each call site so the reasoning lives
    in one place and a new call site cannot silently pick the wrong convention.

    Three conventions exist in this codebase, and getting one wrong is silent
    because an axis-aligned box hides the sign entirely:

    ==========================================  =====================
    where                                       positive means
    ==========================================  =====================
    manifest / ``compute_rotated_box_corners``  visually CCW (row, col)
    ``crop_and_straighten`` (this function)     maths CCW on (x, y) = (col, row)
    ``ImageCanvas._roi_angle``                  visually CW
    ==========================================  =====================

    So a manifest angle reaching ``crop_and_straighten`` must be negated (this
    function), and one reaching the canvas must be negated too.  Five call sites
    have been wrong at one time or another; ``grep -n "set_roi("`` and this
    docstring are the checklist.

    This two-argument form follows the **template** locator's convention, where
    ``rotation_deg`` is ``absolute - taught``.  It is correct only for a box that
    is not moving; a locator-aware crop must go through
    :meth:`AlignmentStrategy.crop_angle`, which derives the angle from the
    strategy's own :meth:`AlignmentStrategy.absolute_rotation`.  Using this
    function with a detector's (oppositely signed) delta tilted every patch by
    twice the target's rotation.

    Args:
        box_angle_deg: Box orientation as stored (visually CCW positive).
        rotation_deg: Target rotation relative to the taught box, in the template
            locator's sign convention.

    Returns:
        The angle to hand to :func:`measure.rectify.crop_and_straighten`.
    """
    return -(float(box_angle_deg) + float(rotation_deg))


# =============================================================================
# NoneAlignment — stage 1 skipped entirely
# =============================================================================


class NoneAlignment(AlignmentStrategy):
    """Skip localisation: the inspection image is already the measurement frame.

    For a fixed-position workpiece, or when some upstream detector has already
    established the pose.  :meth:`align` returns the image untouched with an
    identity mapping, so measurement coordinates *are* image coordinates and
    orientation-dependent measurements remain expressible.

    Teaching still records geometry for visualisation, but deliberately creates
    no :class:`TemplatePoint`: there is nothing to search for, and a template
    would imply a search that this mode exists to avoid.
    """

    mode_name = "none"

    def teach(self, *args, **kwargs) -> None:
        """Store geometry only — never builds a matching template."""
        reference_image = args[0] if args else kwargs.get("reference_image")
        if reference_image is None:
            raise ValueError("teach() requires a reference image")
        center = args[1] if len(args) > 1 else kwargs.get("center", (0.0, 0.0))
        size = args[2] if len(args) > 2 else kwargs.get("size", (0.0, 0.0))
        angle_deg = args[3] if len(args) > 3 else kwargs.get("angle_deg", 0.0)

        self._reference_image = np.asarray(reference_image).copy()
        self._box_center = (float(center[0]), float(center[1]))
        self._box_size = (float(size[0]), float(size[1]))
        self._box_angle_deg = float(angle_deg)
        self._template_point = None
        self._template_image = self._reference_image
        self._template_size = int(max(self._reference_image.shape[:2]))

    def align(
        self,
        inspection_image: np.ndarray,
        matched_row: float,
        matched_col: float,
        rotation_deg: float,
        box_size: Optional[Tuple[float, float]] = None,
    ) -> AlignResult:
        """Return the full image with an identity mapping.

        ``matched_*`` / ``rotation_deg`` are ignored: with no stage 1 there is no
        match to consume.  The engine supplies the image centre and 0.0 in that
        case (see :meth:`MultiTargetWorkflow.measure`).
        """
        image = np.asarray(inspection_image)
        identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        return AlignResult(patch=image, M_inv=identity)

    @classmethod
    def from_roi_state(cls, data, **kwargs) -> "NoneAlignment":
        return cls()


# =============================================================================
# FixedRoiAlignment — crop a known ROI without searching
# =============================================================================


class FixedRoiAlignment(AlignmentStrategy):
    """Crop a taught ROI from every inspection image, with no template search.

    Use when the part's position is repeatable enough that finding it again is
    waste, but the measurement still wants a rectified patch rather than the raw
    frame.  This mirrors v2's ``Localization.fixed_roi``: exactly one source is
    active, and here that source is the stored ROI itself.

    Unlike :class:`NoneAlignment` this *does* deskew, so it removes pose from
    the patch; use ``none`` when the orientation must survive into the
    measurement stage.
    """

    mode_name = "fixed_roi"

    def teach(self, *args, **kwargs) -> None:
        """Store the ROI geometry; builds no TemplatePoint."""
        reference_image = args[0] if args else kwargs.get("reference_image")
        if reference_image is None:
            raise ValueError("teach() requires a reference image")
        center = args[1] if len(args) > 1 else kwargs.get("center")
        size = args[2] if len(args) > 2 else kwargs.get("size")
        angle_deg = args[3] if len(args) > 3 else kwargs.get("angle_deg", 0.0)
        if center is None or size is None:
            raise ValueError("FixedRoiAlignment.teach() requires center and size")

        self._reference_image = np.asarray(reference_image).copy()
        self._box_center = (float(center[0]), float(center[1]))
        self._box_size = (float(size[0]), float(size[1]))
        self._box_angle_deg = float(angle_deg)
        self._preprocessor = kwargs.get("preprocessor")
        self._template_point = None

        # The straightened reference ROI doubles as the patch frame of record.
        self._template_image, _ = crop_and_straighten(
            self._reference_image, self._box_center, self._box_size,
            self.crop_angle(0.0),
        )
        self._template_size = int(max(self._box_size))

    def align(
        self,
        inspection_image: np.ndarray,
        matched_row: float,
        matched_col: float,
        rotation_deg: float,
        box_size: Optional[Tuple[float, float]] = None,
    ) -> AlignResult:
        """Crop the fixed ROI.

        ``matched_*`` default back to the taught centre when the engine has no
        stage-1 result to offer, which keeps this strategy usable whether or not
        a caller happens to pass a pose.
        """
        if matched_row is None or matched_col is None:
            matched_row, matched_col = self._box_center
        if rotation_deg is None:
            rotation_deg = 0.0
        target_angle = self.crop_angle(rotation_deg)
        patch, M_inv_3x3 = crop_and_straighten(
            inspection_image,
            (matched_row, matched_col),
            self._box_size,
            target_angle,
        )
        return AlignResult(patch=patch, M_inv=M_inv_3x3[:2, :])

    @classmethod
    def from_roi_state(cls, data, **kwargs) -> "FixedRoiAlignment":
        return cls()


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
        box_size: Optional[Tuple[float, float]] = None,
    ) -> AlignResult:
        target_angle = self.crop_angle(rotation_deg)
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
        box_size: Optional[Tuple[float, float]] = None,
    ) -> AlignResult:
        """Align using coarse parent match + control-point affine refinement.

        Falls back to rigid alignment when fewer than 3 control points
        match successfully.
        """
        # Stage 1: rigid crop (same as SingleBoxAlignment)
        target_angle = self.crop_angle(rotation_deg)
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
# ObbAlignment — stage 1 by learned oriented-box detection
# =============================================================================


class ObbAlignment(AlignmentStrategy):
    """Locate targets with an OBB detector instead of a template search.

    Where template matching sweeps a discrete angle grid over a taught patch, a
    detector reports a box per instance: continuous orientation, no ROI to draw,
    no search cost.  The measurement stage is untouched — each detected box is
    cropped and straightened by the same :func:`crop_and_straighten` used
    everywhere else, so tools, composed objects and coordinate mapping behave
    identically.

    Patch size is fixed, not per-instance
    ------------------------------------
    Detection boxes vary in size, but a measurement tool's parameters are defined
    once, in patch coordinates — ``FitCircle(radius=320)`` means nothing unless
    the patch has a stable scale.  So the model records a *nominal* box size at
    teaching time and every instance is resampled into it.  The alternative
    (cropping each instance at its own detected size) would quietly change what
    every tool measures whenever the detector boxes the part slightly differently.

    The nominal size is derived from the detection on the reference image unless
    the caller supplies one, and it is stored in the manifest so a reloaded model
    reproduces the same patch frame.
    """

    mode_name = "obb"

    def __init__(self):
        super().__init__()
        self._source: Optional[Any] = None          # ObbSource; kept untyped to
        self._anchor_cls: Optional[int] = None      # avoid importing its module here
        self._anchor_observation: Optional[Any] = None
        self._model_ref: Dict[str, Any] = {}
        self._nominal_size: Optional[Tuple[float, float]] = None
        self._min_score: Optional[float] = None
        self._max_targets: int = 0

    # -- properties -----------------------------------------------------------

    @property
    def source(self) -> Optional[Any]:
        """The injected detector, if any."""
        return self._source

    @property
    def nominal_size(self) -> Optional[Tuple[float, float]]:
        """``(height, width)`` every instance is resampled into."""
        return self._nominal_size

    @property
    def model_ref(self) -> Dict[str, Any]:
        """Identity of the model behind this strategy (path, digest, thresholds)."""
        return dict(self._model_ref)

    @property
    def anchor_class_id(self) -> Optional[int]:
        """Class index treated as the part to locate."""
        return self._anchor_cls

    # -- teaching -------------------------------------------------------------

    def teach(
        self,
        reference_image: np.ndarray,
        *,
        source: Any = None,
        anchor_class_id: Optional[int] = None,
        anchor_observation: Optional[Any] = None,
        nominal_size: Optional[Tuple[float, float]] = None,
        min_score: Optional[float] = None,
        max_targets: int = 0,
        model_ref: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Attach a detector and fix the patch frame.

        Deliberately narrow compared with the template-based ``teach``: there is
        no ROI to draw and no matching parameters to tune, so accepting that
        signature here would mean silently ignoring most of it.  A caller that
        passes the wrong thing gets a TypeError rather than a model that looks
        configured but is not.

        Args:
            reference_image: Teaching image; also the detection that derives the
                nominal box size when one is not supplied.
            source: An ``ObbSource``.  Stored as given, so the engine never needs
                to import a detector implementation.
            anchor_class_id: Keep only this detector class.  ``None`` keeps all.
            anchor_observation: A specific detection to use as the teaching
                reference.  A class alone is ambiguous when the image holds many
                instances of it — 166 ships on one sample — and the anchor decides
                both the patch frame and the recorded pose, so a caller that let
                the user point at a box must be able to pin that box.  When None
                the highest-scoring detection is used.
            nominal_size: ``(height, width)`` for the patch frame.  When omitted
                it is taken from the anchor detection.
            min_score: Drop detections below this confidence.
            max_targets: Cap the number of instances; ``0`` means no cap.
            model_ref: Model identity to record (path, sha256, thresholds).

        Raises:
            ValueError: If no source is given, the reference image is unusable, or
                the anchor class cannot be detected on it.
        """
        if source is None:
            raise ValueError(
                "ObbAlignment.teach() requires a detection source; pass the "
                "ObbSource that will localise targets at inspection time."
            )
        if reference_image is None:
            raise ValueError("ObbAlignment.teach() requires a reference image")

        image = np.asarray(reference_image)
        self._reference_image = image.copy()
        self._source = source
        self._anchor_cls = anchor_class_id
        self._anchor_observation = anchor_observation
        self._min_score = min_score
        self._max_targets = int(max_targets)
        self._model_ref = dict(model_ref or {})
        # No template: this mode exists precisely to avoid building one.
        self._template_point = None
        self._template_image = image

        # Teach-time detection.  Failing here is better than shipping a model
        # that will report nothing at inspection time for reasons the user cannot
        # see; the anchor is required to be findable on its own reference image.
        observations = self._select(self._source.detect(image))
        if not observations:
            raise ValueError(
                "The detector found no target on the reference image"
                + (f" for class {anchor_class_id}" if anchor_class_id is not None
                   else "")
                + ". Teach against an image that contains the part, or widen the "
                  "class/score filters."
            )

        # The anchor box: an explicitly chosen detection when given, otherwise the
        # highest-scoring one.  This fixes the patch frame and the recorded pose, so
        # it must be the box the user pointed at, not merely a class.
        primary = self._resolve_anchor(observations, anchor_observation)

        if nominal_size is not None:
            height, width = float(nominal_size[0]), float(nominal_size[1])
            if height <= 0 or width <= 0:
                raise ValueError(
                    f"nominal_size must be positive (height, width), got {nominal_size}"
                )
            self._nominal_size = (height, width)
        else:
            self._nominal_size = (float(primary.height), float(primary.width))

        # Record the reference box centre so visualisation and the search-free
        # code paths have a meaningful box, and so the GUI can draw the frame.
        self._box_center = (float(primary.row), float(primary.col))
        self._box_size = self._nominal_size
        self._box_angle_deg = float(primary.angle_deg)
        self._template_size = int(max(self._nominal_size))

    @staticmethod
    def _resolve_anchor(observations, requested):
        """Pick the teaching anchor, honouring an explicit request.

        The requested observation must be one of the current detections.  Matching
        by identity would be fragile — detections are re-produced on every teach —
        so the comparison is geometric, to within a pixel.  A mismatch is an error
        rather than a silent fallback: quietly anchoring on a different box would
        change the patch frame the tools were taught against.
        """
        if requested is None:
            return observations[0]
        for candidate in observations:
            if (abs(candidate.row - requested.row) < 1.0
                    and abs(candidate.col - requested.col) < 1.0):
                return candidate
        raise ValueError(
            f"anchor_observation at ({requested.row:.1f}, {requested.col:.1f}) is "
            f"not among the {len(observations)} detections on this image; it may "
            f"have been filtered out by the class or score settings."
        )

    @property
    def anchor_observation(self):
        """The detection that fixed the patch frame, if one was recorded."""
        return self._anchor_observation

    # -- inspection -----------------------------------------------------------

    def observations_for(self, image: np.ndarray) -> List[Any]:
        """Return the selected observations for ``image`` without teaching.

        Used when reopening a project: the geometry is already stored, so this only
        re-runs detection to repopulate the anchor list.  Teaching would re-derive
        the nominal size from today's detections, moving the frame the tools were
        taught in.
        """
        if self._source is None:
            raise RuntimeError(
                "ObbAlignment has no detection source. Attach one before detecting."
            )
        return self._select(self._source.detect(image))

    def absolute_rotation(self, reported_rotation_deg: float) -> float:
        """Undo the sign flip :meth:`detect` applies to a detector's angle.

        ``detect`` stores ``taught - instance`` so the field keeps the
        template matcher's "rotation from the taught pose" meaning, hence the
        inverse here is a subtraction.  Using the base implementation's addition
        would mirror every instance away from the anchor.
        """
        return float(self._box_angle_deg) - float(reported_rotation_deg)

    def detect(self, inspection_image: np.ndarray) -> List[Dict[str, Any]]:
        """Localise targets, returning match dicts in the engine's usual shape.

        Returning the same keys as the template matcher keeps
        ``_measure_one_target`` on a single path.  ``box_size`` is included
        because unlike a template, each instance carries its own extent and the
        crop needs it.
        """
        if self._source is None:
            raise RuntimeError(
                "ObbAlignment has no detection source. Call teach() first."
            )
        observations = self._select(self._source.detect(inspection_image))
        matches: List[Dict[str, Any]] = []
        for obs in observations:
            matches.append({
                "matched_row": float(obs.row),
                "matched_col": float(obs.col),
                # Detector confidence, not an NCC score; None stays None so a
                # source without scores is not mistaken for a perfect match.
                "match_score": obs.score,
                # Deliberately the *negated* delta: ``taught - instance``.  The
                # field is shared with the template matcher, which reports
                # ``absolute - taught`` (positive), so a detector feeding its
                # absolute angle straight through would later be decoded with the
                # wrong sign — that is what produced `2*anchor - instance`, an
                # error of exactly twice the offset from the anchor and zero for
                # the anchor itself, which is why self-matching looked correct.
                # `absolute_rotation()` is the single inverse of this choice.
                "best_rotation_deg": float(self._box_angle_deg - obs.angle_deg),
                "best_scale": 1.0,
                "valid": True,
                "box_size": obs.size,
                "corners": None if obs.corners is None else obs.corners.tolist(),
                "class_id": obs.class_id,
            })
        return matches

    def align(
        self,
        inspection_image: np.ndarray,
        matched_row: float,
        matched_col: float,
        rotation_deg: float,
        box_size: Optional[Tuple[float, float]] = None,
    ) -> AlignResult:
        """Crop the detected box into the fixed nominal patch frame.

        ``box_size`` (the instance's own extent) is accepted but deliberately not
        used for the output shape: the patch frame is the model's, not the
        instance's, so tools defined at teaching time keep their meaning.  It is
        taken as an argument so the signature stays uniform across strategies.
        """
        if self._nominal_size is None:
            raise RuntimeError(
                "ObbAlignment has no nominal patch size. Call teach() first."
            )
        target_angle = self.crop_angle(rotation_deg)
        patch, M_inv_3x3 = crop_and_straighten(
            inspection_image,
            (matched_row, matched_col),
            self._nominal_size,
            target_angle,
        )
        return AlignResult(patch=patch, M_inv=M_inv_3x3[:2, :])

    # -- helpers --------------------------------------------------------------

    def _select(self, observations: Sequence[Any]) -> List[Any]:
        """Apply the stored selection policy (class, score, cap)."""
        return select_observations(
            observations,
            class_id=self._anchor_cls,
            min_score=self._min_score,
            max_targets=self._max_targets,
        )

    # -- serialization --------------------------------------------------------

    def get_roi_state(self) -> Dict[str, Any]:
        """Serialise the geometry the manifest needs to rebuild this strategy."""
        return {
            "alignment_mode": self.mode_name,
            "box_center": list(self._box_center),
            "box_size": list(self._box_size),
            "box_angle_deg": self._box_angle_deg,
            "anchor_class_id": self._anchor_cls,
            "nominal_size": (None if self._nominal_size is None
                             else list(self._nominal_size)),
            "min_score": self._min_score,
            "max_targets": self._max_targets,
            "model_ref": dict(self._model_ref),
        }

    @classmethod
    def from_roi_state(cls, data: Dict[str, Any], **kwargs) -> "ObbAlignment":
        """Rebuild the stored geometry; the source is re-attached by the loader.

        The detector itself is never serialised: it is an injected capability, so
        a caller must supply it again (which is what keeps torch out of the model
        asset and lets the same model run against a different backend).
        """
        strategy = cls()
        nominal = data.get("nominal_size")
        strategy._nominal_size = None if nominal is None else tuple(nominal)
        strategy._anchor_cls = data.get("anchor_class_id")
        strategy._min_score = data.get("min_score")
        strategy._max_targets = int(data.get("max_targets", 0) or 0)
        strategy._model_ref = dict(data.get("model_ref") or {})
        strategy._box_center = tuple(data.get("box_center", (0.0, 0.0)))
        strategy._box_size = tuple(data.get("box_size", (0.0, 0.0)))
        strategy._box_angle_deg = float(data.get("box_angle_deg", 0.0))
        if strategy._nominal_size is not None:
            strategy._template_size = int(max(strategy._nominal_size))
        return strategy


# =============================================================================
# Registry — maps mode_name → AlignmentStrategy subclass
# =============================================================================

_ALIGNMENT_STRATEGY_REGISTRY: Dict[str, type] = {
    "none": NoneAlignment,
    "fixed_roi": FixedRoiAlignment,
    "obb": ObbAlignment,
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
