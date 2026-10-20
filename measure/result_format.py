"""
Result formatting helpers for text display of measurement results.

Moved out of measure_gui.multi_target so both the GUI and the API layers can
share one implementation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _fmt_length(px: float, scale_mm: Optional[float]) -> str:
    """Format a pixel length; append the physical value when calibrated."""
    if scale_mm:
        mm = px * scale_mm
        return f"{mm:.3f} mm ({px:.3f} px)"
    return f"{px:.3f} px"


def _result_dict_line_length(result: dict) -> float:
    """Pixel length of a line result dict (from its endpoints)."""
    dr = result.get('end_row', 0) - result.get('start_row', 0)
    dc = result.get('end_col', 0) - result.get('start_col', 0)
    return float(np.hypot(dr, dc))


def _format_result_dict(label: str, result: dict,
                        scale_mm: Optional[float] = None) -> str:
    """Format a single measurement result dict for text display."""
    rtype = result.get("type", "unknown")
    valid = result.get("valid", False)

    if not valid:
        return f"  {label}: [{rtype}] INVALID"

    if rtype == "point":
        return (
            f"  {label}: [{rtype}] "
            f"row={result['row']:.2f}, col={result['col']:.2f}"
        )
    elif rtype == "line":
        length = _result_dict_line_length(result)
        return (
            f"  {label}: [{rtype}] "
            f"start=({result['start_row']:.1f},{result['start_col']:.1f}), "
            f"end=({result['end_row']:.1f},{result['end_col']:.1f}), "
            f"len={_fmt_length(length, scale_mm)}"
        )
    elif rtype == "circle":
        radius = result['radius']
        if scale_mm:
            r_str = f"r={radius * scale_mm:.2f} mm ({radius:.2f} px)"
        else:
            r_str = f"r={radius:.2f}px"
        return (
            f"  {label}: [{rtype}] "
            f"center=({result['center_row']:.1f},{result['center_col']:.1f}), "
            f"{r_str}"
        )
    elif rtype == "distance":
        return f"  {label}: [{rtype}] {_fmt_length(result['value'], scale_mm)}"
    elif rtype == "angle":
        val = result.get("value_deg", result.get("value", 0))
        return f"  {label}: [{rtype}] {val:.2f}°"
    else:
        return f"  {label}: [{rtype}] VALID"


def _format_geometric_result(label: str, result,
                             scale_mm: Optional[float] = None) -> str:
    """Format a GeometricResult for text display."""
    rtype = result.type
    if not result.valid:
        return f"  {label}: [{rtype}] INVALID"

    if rtype == "point":
        return f"  {label}: [{rtype}] row={result.row:.2f}, col={result.col:.2f}"
    elif rtype == "line":
        return (
            f"  {label}: [{rtype}] "
            f"start=({result.start_row:.1f},{result.start_col:.1f}), "
            f"end=({result.end_row:.1f},{result.end_col:.1f}), "
            f"len={_fmt_length(result.length, scale_mm)}"
        )
    elif rtype == "circle":
        if scale_mm:
            r_str = f"r={result.radius * scale_mm:.2f} mm ({result.radius:.2f} px)"
        else:
            r_str = f"r={result.radius:.2f}px"
        return (
            f"  {label}: [{rtype}] "
            f"center=({result.center_row:.1f},{result.center_col:.1f}), "
            f"{r_str}"
        )
    elif rtype == "distance":
        return f"  {label}: [{rtype}] {_fmt_length(result.value, scale_mm)}"
    elif rtype == "angle":
        val = getattr(result, "value_deg", None)
        if val is None:
            val = getattr(result, "value", 0)
        return f"  {label}: [{rtype}] {val:.2f}°"
    return f"  {label}: [{rtype}] VALID"


# Public aliases (the underscore-prefixed names were the original private API)
fmt_length = _fmt_length
result_dict_line_length = _result_dict_line_length
format_result_dict = _format_result_dict
format_geometric_result = _format_geometric_result
