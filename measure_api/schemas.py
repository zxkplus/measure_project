"""
Type aliases and structure constants for Measure API.

All public API methods return plain dicts (JSON-serializable).
This module documents the expected shape of those dicts.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TypedDict

from measure.object_specs import (
    OBJECT_SPECS,
    composed_deps,
    composed_types,
    primitive_types,
)

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
#
# Derived from the single registry in ``measure.object_specs`` so that this
# module can never drift from the GUI / v2 engine's view of object types.
# ---------------------------------------------------------------------------

PRIMITIVE_TYPES = set(primitive_types())

COMPOSED_TYPES = set(composed_types())

ALL_TYPES = PRIMITIVE_TYPES | COMPOSED_TYPES

# Mapping from composed type to its dependency parameter names
COMPOSED_DEPS: Dict[str, List[str]] = composed_deps()

# Mapping from any object type to the result type it produces.  Composed types
# declare which result type each of their dependency slots expects via
# ``COMPOSED_DEPS`` + this table, so a caller can reject "line wired into a
# point slot" before it silently fails at run time.
RESULT_TYPES: Dict[str, str] = {
    spec.object_type: spec.result_type for spec in OBJECT_SPECS.values()
}

# A circle is usable wherever a point is expected: ``CircleResult`` exposes
# ``row``/``col`` as documented compatibility aliases for its centre, and both
# the GUI dependency picker and the shipped models rely on that.  Keep this in
# sync with measure.measure_workflow.CircleResult.
TYPE_COMPATIBILITY: Dict[str, set] = {
    "point": {"point", "circle"},
    "line": {"line"},
    "circle": {"circle"},
    "distance": {"distance"},
    "angle": {"angle"},
}


def accepts_result_type(expected: str, actual: str) -> bool:
    """True when an object producing ``actual`` may feed an ``expected`` slot."""
    return actual in TYPE_COMPATIBILITY.get(expected, {expected})


def expected_input_types(composed_type: str) -> List[Tuple[str, str]]:
    """Return ``(dependency_param, expected_result_type)`` pairs in order."""
    spec = OBJECT_SPECS.get(composed_type)
    if spec is None:
        return []
    return list(spec.deps)
