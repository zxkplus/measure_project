"""
Quality metrics for measurement results.

Provides quantitative feedback when testing individual measurement objects,
so backend colleagues can judge parameter quality without visual inspection.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np

from measure.evaluation import summarize_quality
from measure.measure_workflow import (
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
    if result is None or not getattr(result, "valid", False):
        return {}

    summary = summarize_quality(meta=getattr(result, "meta", {}) or {})
    q: Dict[str, Any] = {}

    if object_type in ("FitCircle", "FitLine", "EdgePoint", "EdgePair"):
        samples = summary.sample_count or 0
        expected = summary.requested_probe_count or 0
        q["num_edges"] = samples
        q["expected_edges"] = expected
        coverage = summary.coverage_ratio or 0.0
        q["coverage_ratio"] = round(min(coverage, 1.0), 4)

    if object_type in ("FitCircle", "FitLine"):
        if summary.rms_error is not None:
            q["rms"] = round(float(summary.rms_error), 4)

    if object_type in ("EdgePoint", "EdgePair", "FitCircle", "FitLine"):
        if summary.amplitude_mean is not None:
            q["edge_amplitude_mean"] = round(float(summary.amplitude_mean), 2)
        if getattr(result, "meta", None) and "amplitude_min" in (result.meta or {}):
            q["edge_amplitude_min"] = round(float(result.meta["amplitude_min"]), 2)

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
