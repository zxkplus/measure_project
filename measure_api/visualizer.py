"""
Visual snapshot generation for measurement feedback.

Each ``test_measurement()`` call can optionally return a base64-encoded PNG
showing the measurement ROI and detected results overlaid on the template.
"""

from __future__ import annotations

import base64
import io
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from measurement.viz import to_bgr
from measure_workflow import (
    CircleResult,
    GeometricResult,
    LineResult,
    PointResult,
)


def generate_visual(
    template_image: np.ndarray,
    object_type: str,
    params: Dict[str, Any],
    result: Optional[GeometricResult],
    quality: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Draw the measurement ROI and result on the template image.

    Args:
        template_image: Grayscale or BGR template image.
        object_type: Type string like ``"FitCircle"``, ``"FitLine"``.
        params: Parameter dict used for the measurement.
        result: Measurement result (may be None or invalid).
        quality: Optional quality dict for overlay text.

    Returns:
        Base64-encoded PNG string (without ``data:`` prefix).
    """
    vis = to_bgr(template_image)

    # --- Draw ROI ---
    cv2.putText(vis, f"Type: {object_type}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    if object_type == "FitCircle":
        center = params.get("center", (0, 0))
        radius = params.get("radius", 50)
        if isinstance(center, (list, tuple)) and len(center) == 2:
            cx, cy = int(round(center[1])), int(round(center[0]))
            r = int(radius)
            # Draw search ring
            cv2.circle(vis, (cx, cy), r, (255, 180, 80), 1, cv2.LINE_AA)
            # Draw measurement lines
            num = params.get("num_measures", 12)
            for i in range(num):
                angle = 2 * np.pi * i / num
                x1 = int(cx + (r - params.get("measure_length1", 60)) * np.cos(angle))
                y1 = int(cy + (r - params.get("measure_length1", 60)) * np.sin(angle))
                x2 = int(cx + (r + params.get("measure_length2", 10)) * np.cos(angle))
                y2 = int(cy + (r + params.get("measure_length2", 10)) * np.sin(angle))
                cv2.line(vis, (x1, y1), (x2, y2), (255, 180, 80), 1)

        # Draw detected circle
        if result is not None and result.valid and hasattr(result, "center_row"):
            cx = int(round(result.center_col))
            cy = int(round(result.center_row))
            cr = int(round(result.radius))
            cv2.circle(vis, (cx, cy), cr, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(vis, (cx, cy), 4, (0, 255, 0), -1)
            cv2.putText(vis, f"R={result.radius:.1f}", (cx + 8, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

    elif object_type == "FitLine":
        start = params.get("start", (0, 0))
        end = params.get("end", (0, 0))
        if isinstance(start, (list, tuple)) and len(start) == 2:
            # Draw search region
            s_r, s_c = int(round(start[1])), int(round(start[0]))
            e_r, e_c = int(round(end[1])), int(round(end[0]))
            cv2.line(vis, (s_c, s_r), (e_c, e_r), (255, 180, 80), 1, cv2.LINE_AA)
            # Draw perpendicular measure lines
            num = params.get("num_measures", 10)
            ml1 = params.get("measure_length1", 10)
            dr = e_r - s_r
            dc = e_c - s_c
            length = max(1.0, np.sqrt(dr**2 + dc**2))
            for i in range(num):
                t = i / max(num - 1, 1)
                pr = s_r + dr * t
                pc = s_c + dc * t
                nr = -dc / length
                nc = dr / length
                x1 = int(pc + nr * ml1)
                y1 = int(pr + nc * ml1)
                x2 = int(pc - nr * ml1)
                y2 = int(pr - nc * ml1)
                cv2.line(vis, (x1, y1), (x2, y2), (255, 180, 80), 1)

        # Draw detected line
        if result is not None and result.valid and hasattr(result, "start_row"):
            pt1 = (int(round(result.start_col)), int(round(result.start_row)))
            pt2 = (int(round(result.end_col)), int(round(result.end_row)))
            cv2.line(vis, pt1, pt2, (0, 255, 0), 2, cv2.LINE_AA)

    elif object_type in ("EdgePoint", "EdgePair"):
        row = params.get("row", 0)
        col = params.get("col", 0)
        angle = params.get("angle", 0.0)
        l1 = params.get("length1", 50)
        if isinstance(row, (int, float)) and isinstance(col, (int, float)):
            r, c = int(round(row)), int(round(col))
            # Draw measurement rectangle
            p1 = (int(c - l1 * np.cos(angle + np.pi / 2)),
                  int(r - l1 * np.sin(angle + np.pi / 2)))
            p2 = (int(c + l1 * np.cos(angle + np.pi / 2)),
                  int(r + l1 * np.sin(angle + np.pi / 2)))
            cv2.line(vis, p1, p2, (255, 180, 80), 1, cv2.LINE_AA)

        if result is not None and result.valid and hasattr(result, "row"):
            pt = (int(round(result.col)), int(round(result.row)))
            cv2.drawMarker(vis, pt, (0, 255, 0), cv2.MARKER_CROSS, 10, 2)

    # --- Quality overlay ---
    if quality:
        y = 45
        for k, v in quality.items():
            cv2.putText(vis, f"{k}: {v}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
            y += 18

    # --- Status ---
    status = "OK" if (result is not None and result.valid) else "FAIL"
    color = (0, 255, 0) if status == "OK" else (0, 0, 255)
    cv2.putText(vis, status, (vis.shape[1] - 60, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    # --- Encode ---
    success, buf = cv2.imencode(".png", vis)
    if not success:
        return ""
    return base64.b64encode(buf).decode("ascii")


def generate_overview_visual(
    inspection_image: np.ndarray,
    targets: list,
) -> str:
    """
    Draw all detected targets and their measurements on the inspection image.

    Args:
        inspection_image: The original inspection image (grayscale or BGR).
        targets: List of target result dicts from ``measure()``.

    Returns:
        Base64-encoded PNG string.
    """
    vis = to_bgr(inspection_image)

    palette = [
        (0, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0),
        (0, 165, 255), (255, 0, 0), (128, 0, 128), (0, 128, 128),
    ]

    for ti, target in enumerate(targets):
        color = palette[ti % len(palette)]
        row = target.get("row", 0)
        col = target.get("col", 0)
        angle = target.get("rotation_deg", 0)

        # Draw center cross
        c = (int(round(col)), int(round(row)))
        cv2.drawMarker(vis, c, color, cv2.MARKER_CROSS, 14, 2)

        # Draw target label
        score = target.get("score", 0)
        cv2.putText(vis, f"T#{ti} s={score:.3f}", (c[0] + 10, c[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        # Draw measurements
        for label, meas in target.get("measurements", {}).items():
            if not meas.get("valid", False):
                continue
            mtype = meas.get("type", "")
            if mtype == "circle":
                cr = int(round(meas.get("center_row", 0)))
                cc = int(round(meas.get("center_col", 0)))
                r = int(round(meas.get("radius", 0)))
                cv2.circle(vis, (cc, cr), r, color, 1)
            elif mtype == "line":
                p1 = (int(round(meas.get("start_col", 0))),
                      int(round(meas.get("start_row", 0))))
                p2 = (int(round(meas.get("end_col", 0))),
                      int(round(meas.get("end_row", 0))))
                cv2.line(vis, p1, p2, color, 1)
            elif mtype == "point":
                pr = int(round(meas.get("row", 0)))
                pc = int(round(meas.get("col", 0)))
                cv2.drawMarker(vis, (pc, pr), color, cv2.MARKER_CROSS, 6, 1)

    # Summary
    cv2.putText(vis, f"Targets: {len(targets)}", (10, vis.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    success, buf = cv2.imencode(".png", vis)
    if not success:
        return ""
    return base64.b64encode(buf).decode("ascii")
