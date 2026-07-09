"""
Type aliases and structure constants for Measure API.

All public API methods return plain dicts (JSON-serializable).
This module documents the expected shape of those dicts.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TypedDict

# ---------------------------------------------------------------------------
# Measurement result types
# ---------------------------------------------------------------------------


class PointResult(TypedDict, total=False):
    valid: bool
    type: str          # "point"
    row: float
    col: float


class LineResult(TypedDict, total=False):
    valid: bool
    type: str          # "line"
    a: float
    b: float
    c: float
    start_row: float
    start_col: float
    end_row: float
    end_col: float


class CircleResult(TypedDict, total=False):
    valid: bool
    type: str          # "circle"
    center_row: float
    center_col: float
    radius: float


class DistanceResult(TypedDict, total=False):
    valid: bool
    type: str          # "distance"
    value: float


class AngleResult(TypedDict, total=False):
    valid: bool
    type: str          # "angle"
    value_deg: float


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------


class QualityMetrics(TypedDict, total=False):
    num_edges: int
    expected_edges: int
    coverage_ratio: float
    rms: float
    edge_amplitude_mean: float
    edge_amplitude_min: float


# ---------------------------------------------------------------------------
# DAG
# ---------------------------------------------------------------------------


class DagNode(TypedDict, total=False):
    label: str
    type: str                        # primitive object type or composed type
    category: str                    # "primitive" | "composed"
    valid: bool


class DagEdge(TypedDict, total=False):
    from_: str                       # upstream label
    to: str                          # downstream label
    role: str                        # e.g. "point_label", "line_label"


class DagResponse(TypedDict, total=False):
    nodes: List[DagNode]
    edges: List[DagEdge]
    execution_order: List[str]
    is_valid: bool


# ---------------------------------------------------------------------------
# Call record
# ---------------------------------------------------------------------------


class CallRecord(TypedDict, total=False):
    call_id: str
    seq: int
    timestamp: str
    elapsed_ms: float
    function: str
    endpoint: str
    session_id: str
    session_state: Dict[str, Any]
    request: Dict[str, Any]
    response: Dict[str, Any]
    error: Optional[str]
    visual_file: Optional[str]
    copied_images: List[str]


# ---------------------------------------------------------------------------
# Supported measurement object types
# ---------------------------------------------------------------------------

PRIMITIVE_TYPES = {
    "EdgePoint",
    "EdgePair",
    "FitLine",
    "FitCircle",
    "TemplateMatchPoint",
}

COMPOSED_TYPES = {
    "TwoPointsLine",
    "TwoPointsDistance",
    "PointLineDistance",
    "TwoLinesAngle",
    "PointCircleDistance",
}

ALL_TYPES = PRIMITIVE_TYPES | COMPOSED_TYPES

# Mapping from composed type to its dependency parameter names
COMPOSED_DEPS: Dict[str, List[str]] = {
    "TwoPointsLine": ["point_a_label", "point_b_label"],
    "TwoPointsDistance": ["point_a_label", "point_b_label"],
    "PointLineDistance": ["point_label", "line_label"],
    "TwoLinesAngle": ["line_a_label", "line_b_label"],
    "PointCircleDistance": ["point_label", "circle_label"],
}
