"""Headless decimal vector geometry and optional pixel rasterization."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterator, Tuple

import cv2
import numpy as np

from measure_api.v2.errors import ServiceError
from measure_api.v2.models import CoordinateMapping
from measure_api.v2.result_mapping import map_geometry
from measure_api.v2.result_models import OverlayPrimitive, VisualizationRequest
from measure_api.v2.result_support import result_nodes


def geometry_values(
    value: Any, port: str = "", frame: str = "source", unit: str = "px", valid: bool = True
) -> Iterator[Tuple[str, str, dict, str, str, bool]]:
    """Find actual observed geometry, without inventing locations for scalar values.

    Args:
        value: Supplied observation, geometry or numerical quality.
        port: Stable observation port path within the node.
        frame: Declared observation coordinate frame.
        unit: Declared observation measurement unit.
        valid: Enclosing observation validity.

    Yields:
        Actual primitive coordinates with ports, frames, units and validity.
    """
    if isinstance(value, list):
        for index, child in enumerate(value):
            yield from geometry_values(child, f"{port}[{index}]", frame, unit, valid)
        return
    if not isinstance(value, dict):
        return
    frame, unit, valid = (
        value.get("frame", frame),
        value.get("unit", unit),
        value.get("valid", valid),
    )
    if not valid:
        yield port, "annotation", {
            "reason": value.get("reason", "invalid_observation")
        }, frame, unit, False
        return
    if "point" in value and isinstance(value["point"], dict):
        yield from geometry_values(value["point"], port, frame, unit, valid)
        return
    if {"a", "b", "c"} <= set(value) and value["a"] is not None:
        yield port, "line", {
            key: value.get(key) for key in ["a", "b", "c", "start", "end"]
        }, frame, unit, valid
        return
    if "center" in value and "radius" in value and value.get("radius") is not None:
        yield port, "circle", {
            "center": value["center"],
            "radius": value["radius"],
        }, frame, unit, valid
        return
    if value.get("type") == "rectangle":
        angle = np.deg2rad(value.get("angle_degrees", 0))
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], np.float64)
        corners *= [value["width"] / 2, value["height"] / 2]
        corners = corners @ rotation.T + [value["center"]["x"], value["center"]["y"]]
        points = [{"x": float(x), "y": float(y)} for x, y in corners]
        yield port, "rectangle", {"points": points, "closed": True}, frame, unit, valid
        return
    if value.get("type") == "polyline":
        yield port, "polyline", {"points": value["points"], "closed": False}, frame, unit, valid
        return
    if set(value) == {"x", "y"}:
        yield port, "point", value, frame, unit, valid
        return
    for key, child in value.items():
        if key not in {
            "diagnostics",
            "gray",
            "gradient",
            "smoothed",
            "artifacts",
            "manifest",
            "sample_points",
            "samples",
            "distances",
            "center_distances",
        }:
            yield from geometry_values(child, f"{port}.{key}".strip("."), frame, unit, valid)


def overlays(request: VisualizationRequest) -> list:
    """Retain complete decimal primitives with per-request independent display flags.

    Args:
        request: Typed supplied observations and calculation controls.

    Returns:
        Stable vector primitives with request-local display flags.
    """
    values = []
    for item, run, target, node in result_nodes(request.result):
        data = node.get(request.space + "_result")
        if data is None:
            continue
        for port, shape, geometry, frame, unit, valid in geometry_values(data):
            if frame != request.display_frame:
                mapping = request.mapping
                if mapping is None and target.get("physical_mapping") is not None:
                    mapping = CoordinateMapping.model_validate(target["physical_mapping"])
                if mapping is None:
                    raise ServiceError(
                        "visualization.frame_mismatch", "An explicit display mapping is required"
                    )
                shape, geometry, unit = map_geometry(
                    shape, geometry, mapping, frame, request.display_frame, request.curve_samples
                )
                frame = request.display_frame
            identity = hashlib.sha256(
                json.dumps(
                    [
                        item.get("item_id"),
                        target["target_id"],
                        node["id"],
                        request.space,
                        port or "value",
                    ],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            values.append(
                OverlayPrimitive(
                    id=identity,
                    item_id=item.get("item_id"),
                    target_id=target["target_id"],
                    node_id=node["id"],
                    port=port or "value",
                    type=shape,
                    geometry=geometry,
                    frame=frame,
                    unit=unit,
                    label=node.get("label"),
                    valid=valid,
                    visible=request.visible_nodes is None or node["id"] in request.visible_nodes,
                    selected=node["id"] in request.selected_nodes,
                )
            )
            if len(values) > 100000:
                raise ServiceError("visualization.too_large", "Too many overlay primitives")
    return values


def rasterize(image: np.ndarray, values: list) -> np.ndarray:
    """Draw a copy; rounding affects pixels only and never the returned vector geometry.

    Args:
        image: Decoded source pixels; these remain unchanged.
        values: Selectable vector primitives or named numeric arrays.

    Returns:
        An independent BGR raster with visible measured geometry.
    """
    pixels = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()

    def point(value: dict) -> tuple:
        return tuple(int(np.clip(round(value[key]), -10000000, 10000000)) for key in ["x", "y"])

    for value in values:
        if not value.visible or not value.valid:
            continue
        if value.unit != "px":
            raise ServiceError(
                "visualization.pixel_units_required", "Raster geometry must use pixel units"
            )
        g = value.geometry
        if value.type == "point":
            cv2.drawMarker(pixels, point(g), (0, 255, 0), cv2.MARKER_CROSS, 7, 1)
        elif value.type == "line":
            if g.get("start") and g.get("end"):
                a, b = point(g["start"]), point(g["end"])
            elif abs(g["b"]) > abs(g["a"]):
                a, b = (0, int(round(-g["c"] / g["b"]))), (
                    pixels.shape[1] - 1,
                    int(round(-(g["a"] * (pixels.shape[1] - 1) + g["c"]) / g["b"])),
                )
            else:
                a, b = (int(round(-g["c"] / g["a"])), 0), (
                    int(round(-(g["b"] * (pixels.shape[0] - 1) + g["c"]) / g["a"])),
                    pixels.shape[0] - 1,
                )
            clipped, a, b = cv2.clipLine((0, 0, pixels.shape[1], pixels.shape[0]), a, b)
            if clipped:
                cv2.line(pixels, a, b, (0, 255, 0), 1)
        elif value.type == "circle":
            cv2.circle(pixels, point(g["center"]), int(round(g["radius"])), (0, 255, 0), 1)
        elif value.type in {"polyline", "rectangle"}:
            points = np.array([point(p) for p in g["points"]], np.int32)
            cv2.polylines(pixels, [points], g.get("closed", True), (0, 255, 0), 1)
    return pixels
