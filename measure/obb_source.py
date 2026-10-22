"""Oriented bounding box (OBB) detection as a pluggable localisation source.

Stage 1 of a measurement model answers "where is the part".  Template matching is
one answer; a learned OBB detector is another, and for many lines it is the
better one — no ROI to draw, no angle grid to sweep, and a continuous orientation
instead of one quantised to ``angle_step``.

This module is deliberately free of any ML dependency.  It defines what a
detector must return and the geometry to convert that into the engine's ROI
frame; the actual model adapter lives outside ``measure/`` (see
``measure_obb``), so the engine stays importable — and testable — without torch.

Coordinate conventions (these match :mod:`measure.rectify`, and getting them
wrong rotates patches the wrong way):

* points are ``(row, col)``, i.e. y-then-x, matching numpy image indexing
* angles are **degrees, counter-clockwise positive**, measured from the image
  column axis, exactly as ``crop_and_straighten`` expects
* box ``size`` is ``(height, width)`` — rows first — again as the crop expects
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np

#: Tolerated deviation, in pixels, when deciding whether four points really form
#: a rectangle.  Detectors emit float boxes, so exact equality is unrealistic.
_RECTANGLE_TOLERANCE_PX = 1e-3

#: Tolerated ``|cos|`` between adjacent edges.
#:
#: Sized from the precision of the input rather than picked by feel.  Detectors
#: hand over float32 tensors (~7 significant digits), so a perfectly square box
#: still shows a residual: measured over 166 real detections from a YOLO-OBB
#: model the median was 1.2e-6 and the maximum 4.5e-6, i.e. 57% of genuine boxes
#: exceed 1e-6.  A threshold at that magnitude rejects real detections at random.
#:
#: 1e-3 corresponds to about 0.06 degrees of corner error — far looser than any
#: real detector noise, yet still tight enough to catch a genuinely skewed
#: quadrilateral (the test that guards this uses a 60-degree corner).
_RECTANGLE_ORTHOGONALITY_TOLERANCE = 1e-3


@dataclass(frozen=True)
class ObbObservation:
    """One oriented box reported by a detector, in inspection-image coordinates.

    Attributes:
        row: Box centre row.
        col: Box centre column.
        width: Extent along the box's own long axis.
        height: Extent perpendicular to that axis.
        angle_deg: Orientation in degrees, **counter-clockwise positive**,
            measured from the column axis, already normalised into
            ``[-90, 90)``.  Build instances through :meth:`from_corners` or
            :meth:`from_xywhr` rather than guessing the convention.
        class_id: Detector class index.
        score: Confidence, or ``None`` when the detector reports none.
        class_name: Optional human-readable class label.
        corners: ``(4, 2)`` array of ``(row, col)`` corners in the order
            top-left, top-right, bottom-right, bottom-left *relative to the
            normalised angle*.  Retained because it is the version-proof form of
            the same information.
    """

    row: float
    col: float
    width: float
    height: float
    angle_deg: float
    class_id: int = 0
    score: Optional[float] = None
    class_name: Optional[str] = None
    corners: Optional[np.ndarray] = None

    @property
    def size(self) -> Tuple[float, float]:
        """``(height, width)`` — the order :func:`measure.rectify.crop_and_straighten` wants."""
        return (float(self.height), float(self.width))

    @property
    def center(self) -> Tuple[float, float]:
        """``(row, col)`` centre."""
        return (float(self.row), float(self.col))

    # -- construction ---------------------------------------------------------

    @classmethod
    def from_corners(
        cls,
        corners: np.ndarray,
        *,
        class_id: int = 0,
        score: Optional[float] = None,
        class_name: Optional[str] = None,
    ) -> "ObbObservation":
        """Build from four ``(row, col)`` corners in any winding order.

        Preferred over :meth:`from_xywhr` because it does not depend on the
        detector's angle field.  That field's meaning has moved between library
        releases (radians vs degrees, and which edge the angle refers to), while
        four points are unambiguous.  The rectangle is recovered from the points,
        so a caller cannot pass an angle that disagrees with the box.

        Args:
            corners: ``(4, 2)`` array of ``(row, col)`` points, in perimeter
                order (either winding).
            class_id: Detector class index.
            score: Confidence, if any.
            class_name: Optional label.

        Returns:
            A normalised observation.

        Raises:
            ValueError: If four points are not supplied, or they do not form a
                rectangle within :data:`_RECTANGLE_TOLERANCE_PX`.
        """
        pts = np.asarray(corners, dtype=np.float64)
        if pts.shape != (4, 2):
            raise ValueError(
                f"corners must be a (4, 2) array of (row, col) points, got {pts.shape}"
            )

        # Edge vectors between consecutive corners; a rectangle alternates two
        # independent directions.  Use the two from corner 0 so the box's own
        # axes are recovered regardless of winding.
        e0 = pts[1] - pts[0]
        e1 = pts[3] - pts[0]
        len0 = float(np.hypot(*e0))
        len1 = float(np.hypot(*e1))
        if len0 <= 0.0 or len1 <= 0.0:
            raise ValueError("corners describe a degenerate box (zero-length edge)")

        # Verify closure: opposite corners must agree, which is what makes it a
        # parallelogram-rectangle rather than an arbitrary quadrilateral.
        expected_pt2 = pts[0] + e0 + e1
        if float(np.max(np.abs(expected_pt2 - pts[2]))) > _RECTANGLE_TOLERANCE_PX:
            raise ValueError(
                "corners do not form a rectangle; expected corner 2 to be at "
                f"{expected_pt2.tolist()}, got {pts[2].tolist()}"
            )
        # And the two edge directions must be perpendicular.
        cos_angle = float(np.dot(e0, e1)) / (len0 * len1)
        if abs(cos_angle) > _RECTANGLE_ORTHOGONALITY_TOLERANCE:
            raise ValueError(
                f"corners do not form a rectangle; edges meet at "
                f"{math.degrees(math.acos(max(-1.0, min(1.0, cos_angle)))):.3f}° "
                f"instead of 90°"
            )

        # Normalise so `width` is the longer axis and the angle refers to it,
        # giving one canonical representation for a box that could equally be
        # described by its other edge plus 90°.
        if len1 > len0:
            long_vec, short_len = e1, len0
            long_len = len1
        else:
            long_vec, short_len = e0, len1
            long_len = len0

        # Angle from the column axis, counter-clockwise positive.  Row grows
        # downwards, so a counter-clockwise turn in image terms is a negative
        # rotation in (row, col): hence atan2(-drow, dcol).
        angle_deg = math.degrees(math.atan2(-float(long_vec[0]), float(long_vec[1])))
        angle_deg = _normalize_angle_deg(angle_deg)

        # Re-emit the corners from the normalised angle rather than keeping the
        # caller's points: a box described by its short edge at theta is the same
        # box as its long edge at theta+90, and storing whichever form arrived
        # would make two identical boxes compare unequal.  Round-tripping through
        # the canonical form is what makes `corners` and `angle_deg` agree by
        # construction.
        center = pts.mean(axis=0)
        ordered = _rectangle_corners(
            float(center[0]), float(center[1]), long_len, short_len, angle_deg
        )
        return cls(
            row=float(center[0]),
            col=float(center[1]),
            width=long_len,
            height=short_len,
            angle_deg=angle_deg,
            class_id=int(class_id),
            score=None if score is None else float(score),
            class_name=class_name,
            corners=ordered,
        )

    @classmethod
    def from_xywhr(
        cls,
        center_row: float,
        center_col: float,
        width: float,
        height: float,
        angle_rad: float,
        *,
        class_id: int = 0,
        score: Optional[float] = None,
        class_name: Optional[str] = None,
    ) -> "ObbObservation":
        """Build from a centre/size/radians tuple, as many detectors report it.

        Convenience for adapters whose library documents its angle as radians.
        Internally this converts to corners and delegates to
        :meth:`from_corners`, so both constructors yield identical observations
        and there is only one implementation of the geometry.

        Args:
            center_row: Box centre row.
            center_col: Box centre column.
            width: Extent along the box's long axis.
            height: Extent perpendicular to it.
            angle_rad: Orientation in radians, counter-clockwise positive.
            class_id: Detector class index.
            score: Confidence, if any.
            class_name: Optional label.

        Returns:
            A normalised observation.
        """
        length = max(abs(float(width)), abs(float(height)))
        breadth = min(abs(float(width)), abs(float(height)))
        theta = float(angle_rad)
        if abs(width) < abs(height):
            # The angle was given for the shorter axis; the canonical form uses
            # the longer one, so rotate the reference by a quarter turn.
            theta += math.pi / 2.0
        corners = _rectangle_corners(center_row, center_col, length, breadth,
                                     math.degrees(theta))
        return cls.from_corners(
            corners, class_id=class_id, score=score, class_name=class_name
        )

    # -- conversion -----------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable form (corners become nested lists)."""
        return {
            "row": self.row,
            "col": self.col,
            "width": self.width,
            "height": self.height,
            "angle_deg": self.angle_deg,
            "class_id": self.class_id,
            "score": self.score,
            "class_name": self.class_name,
            "corners": None if self.corners is None else self.corners.tolist(),
        }


@runtime_checkable
class ObbSource(Protocol):
    """A detector the engine can localise with.

    Implementations live outside ``measure/`` so that importing the engine never
    pulls in torch or a model runtime.  The engine only ever calls :meth:`detect`
    and :meth:`serialize`.
    """

    #: Stable identifier recorded in the model manifest.
    name: str

    def detect(self, image: np.ndarray) -> List[ObbObservation]:
        """Return every observation on ``image``, best-first or in any order.

        Implementations must not filter by class or score beyond their own
        configuration; selection policy belongs to the caller so that it can be
        recorded with the model and reproduced later.
        """
        ...

    def serialize(self) -> Dict[str, Any]:
        """Describe this source for the manifest (model identity, thresholds)."""
        ...

    #: Whether this source needs a taught ROI.  False for detectors: they find
    #: their own boxes, which is what lets the GUI skip ROI drawing.
    requires_roi: bool


def select_observations(
    observations: Sequence[ObbObservation],
    *,
    class_id: Optional[int] = None,
    min_score: Optional[float] = None,
    max_targets: int = 0,
) -> List[ObbObservation]:
    """Apply an explicit, reproducible selection policy.

    Kept as a free function rather than baked into sources so the policy is
    visible at the call site and can be stored with the model.  Re-deriving the
    choice on every image instead would let a model silently measure a different
    object than the one it was taught against.

    Args:
        observations: Candidates, as returned by a source.
        class_id: Keep only this class when given.
        min_score: Drop observations scoring below this when given.
        max_targets: Cap the result, highest score first; ``0`` means no cap.

    Returns:
        The selected observations, ordered by descending score (observations
        without a score keep their original relative order, after scored ones).
    """
    kept = list(observations)
    if class_id is not None:
        kept = [o for o in kept if o.class_id == class_id]
    if min_score is not None:
        kept = [o for o in kept if o.score is not None and o.score >= min_score]

    # Stable sort: observations lacking a score cannot be ranked, so they stay in
    # source order below the scored ones rather than jumping to the front.
    kept.sort(key=lambda o: (o.score is None, -(o.score or 0.0)))
    if max_targets > 0:
        kept = kept[:max_targets]
    return kept


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _normalize_angle_deg(angle_deg: float) -> float:
    """Fold an angle into ``[-90, 90)``.

    A box has no head or tail, so 100° and -80° describe the same orientation.
    Normalising means the recorded angle is canonical and comparable instead of
    depending on which edge the detector happened to list first.
    """
    angle = float(angle_deg)
    while angle >= 90.0:
        angle -= 180.0
    while angle < -90.0:
        angle += 180.0
    return angle


def _rectangle_corners(
    center_row: float,
    center_col: float,
    length: float,
    breadth: float,
    angle_deg: float,
) -> np.ndarray:
    """Four ``(row, col)`` corners for a box, in perimeter order.

    Uses the same counter-clockwise-positive convention as
    :func:`measure.rectify.compute_rotated_box_corners`, so the result can be fed
    straight to :meth:`ObbObservation.from_corners`.
    """
    theta = math.radians(float(angle_deg))
    cos_a, sin_a = math.cos(theta), math.sin(theta)
    half_len, half_brd = length / 2.0, breadth / 2.0
    # Offsets along the box axes, then rotated.  Row is negated because image
    # rows grow downwards while the angle is counter-clockwise in display terms.
    local = np.array(
        [[-half_brd, -half_len],
         [-half_brd, half_len],
         [half_brd, half_len],
         [half_brd, -half_len]],
        dtype=np.float64,
    )
    # local columns are (along-perpendicular, along-length) in (row, col) terms
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)
    pts = local @ rot.T
    pts[:, 0] += float(center_row)
    pts[:, 1] += float(center_col)
    return pts
