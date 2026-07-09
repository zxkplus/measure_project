"""
Quality metrics for measurement results.

Provides quantitative feedback when testing individual measurement objects,
so backend colleagues can judge parameter quality without visual inspection.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np

from measure_workflow import (
    CircleResult,
    EdgePairObject,
    EdgePointObject,
    FitCircleObject,
    FitLineObject,
    GeometricResult,
    LineResult,
    PointResult,
)


def compute_quality(object_type: str, result: Any) -> Dict[str, Any]:
    """
    Compute quality metrics for a single measurement result.

    Args:
        object_type: Type string like ``"FitCircle"``, ``"FitLine"``, etc.
        result: The ``GeometricResult`` (or subclass) from measurement.

    Returns:
        Dict with keys like ``num_edges``, ``expected_edges``,
        ``coverage_ratio``, ``rms``, ``edge_amplitude_mean``, etc.
        Empty dict if not applicable.
    """
    q: Dict[str, Any] = {}

    if result is None or not getattr(result, "valid", False):
        return q

    meta = getattr(result, "meta", {}) or {}

    if object_type in ("FitCircle", "FitLine", "EdgePoint", "EdgePair"):
        num_edges = meta.get("num_points", meta.get("num_edges", 0))
        if isinstance(num_edges, (list, tuple, np.ndarray)):
            num_edges = int(np.sum(num_edges))
        else:
            num_edges = int(num_edges)
        q["num_edges"] = num_edges

        expected = meta.get("num_measures", 0)
        if isinstance(expected, (list, tuple, np.ndarray)):
            expected = int(np.sum(expected))
        else:
            expected = int(expected)
        q["expected_edges"] = expected

        if expected > 0:
            q["coverage_ratio"] = round(min(num_edges / expected, 1.0), 4)
        else:
            q["coverage_ratio"] = 0.0

    if object_type in ("FitCircle", "FitLine"):
        rms_val = meta.get("rms", meta.get("fit_rms", None))
        if rms_val is not None:
            q["rms"] = round(float(rms_val), 4)

    if object_type in ("EdgePoint", "EdgePair", "FitCircle", "FitLine"):
        amp = meta.get("amplitude_mean", meta.get("edge_amplitude", None))
        if amp is not None:
            q["edge_amplitude_mean"] = round(float(amp), 2)
        amp_min = meta.get("amplitude_min", None)
        if amp_min is not None:
            q["edge_amplitude_min"] = round(float(amp_min), 2)

    return q


def quality_summary(q: Dict[str, Any]) -> str:
    """
    One-line human-readable quality summary.

    Args:
        q: Quality dict from ``compute_quality()``.

    Returns:
        Short string like ``"edges=96/12 rate=0.92 rms=0.32"``.
    """
    parts = []
    if "num_edges" in q:
        parts.append(f"edges={q['num_edges']}/{q.get('expected_edges', '?')}")
    if "coverage_ratio" in q:
        parts.append(f"rate={q['coverage_ratio']:.2f}")
    if "rms" in q:
        parts.append(f"rms={q['rms']:.2f}")
    if "edge_amplitude_mean" in q:
        parts.append(f"amp={q['edge_amplitude_mean']:.1f}")
    return " ".join(parts)
