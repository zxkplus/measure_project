"""
Measure package — portable, pure-Python reimplementation of geometric
measurement tools (1-D edge detection, line / circle fitting, template
matching).

This module consolidates all measurement-related code into a single package.
"""

# ---------------------------------------------------------------------------
# Display-mode control
# ---------------------------------------------------------------------------
# Call ``_apply_headless_patch()`` to make all ``cv2.imshow`` / ``cv2.waitKey``
# calls into no-ops globally.  This is used by the test suite when
# ``--headless`` is passed.


def _apply_headless_patch() -> None:
    """Monkey-patch cv2 display functions into no-ops (applied globally)."""
    import cv2 as _cv2

    _cv2.imshow = lambda *a, **kw: None  # type: ignore[assignment]
    _cv2.waitKey = lambda *a, **kw: -1   # type: ignore[assignment]
    _cv2.destroyAllWindows = lambda *a, **kw: None  # type: ignore[assignment]
    _cv2.destroyWindow = lambda *a, **kw: None  # type: ignore[assignment]
    _cv2.namedWindow = lambda *a, **kw: None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Core imports - these make the most commonly used classes available at the
# package level so users can write ``from measure import Halcon1DMeasure``.
# ---------------------------------------------------------------------------

# 1-D measurement
from measure.measure1D import Halcon1DMeasure

# 2-D measurement
from measure.measure2D import LineMeasureObject, CircleMeasureObject, MetrologyModel

# Calibration
from measure.measure_calibration import CameraCalibration, StereoRigCalibration

# Template matching
from measure.measure_template import (
    Preprocessor,
    RawPreprocessor,
    CannyPreprocessor,
    SobelPreprocessor,
    CLAHEPreprocessor,
    ThresholdPreprocessor,
    TemplatePoint,
    DistanceMeasure,
)

# Workflow
from measure.measure_workflow import (
    MeasurementWorkflow,
    MeasureObject,
    TemplatePointObject,
    EdgePointObject,
    TemplateMatchPointObject,
    EdgePairObject,
    FitLineObject,
    FitCircleObject,
    TwoPointsLineObject,
    TwoPointsDistanceObject,
    PointLineDistanceObject,
    TwoLinesAngleObject,
    PointCircleDistanceObject,
    GeometricResult,
    PointResult,
    LineResult,
    CircleResult,
    DistanceResult,
    AngleResult,
    SimilarityTransform,
)

# Multi-target workflow
from measure.multi_target_workflow import MultiTargetWorkflow, TargetInstance, TargetResult

# Core utilities (available but not exported by default to keep namespace clean)
# Users can still do: from measure.constants import EPS
# from measure.signal_ops import find_peaks_vectorized
# from measure.transforms import compute_rotated_rect_corners
# from measure.viz import to_bgr, draw_text_shadow
