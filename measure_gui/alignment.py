"""Backwards-compatibility shim.

The alignment strategies were promoted to :mod:`measure.alignment` so the API
layers can share them without importing from a GUI package.  Import from there
in new code.
"""

from __future__ import annotations

from measure.alignment import (  # noqa: F401
    AlignmentStrategy,
    AlignResult,
    ControlPoint,
    MultiPointAlignment,
    SingleBoxAlignment,
    _ALIGNMENT_STRATEGY_REGISTRY,
    _compose_affine_2x3,
    strategy_from_roi_state,
)

__all__ = [
    "AlignmentStrategy",
    "AlignResult",
    "ControlPoint",
    "SingleBoxAlignment",
    "MultiPointAlignment",
    "strategy_from_roi_state",
]