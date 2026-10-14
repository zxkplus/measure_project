"""Bounded numerical quality summaries and shared, non-pickle sampling diagnostics."""

from __future__ import annotations

import io
import json
import zipfile
from typing import Any, Optional

import cv2
import numpy as np

from measure_api.v2.errors import ServiceError
from measure_api.v2.models import FileReference
from measure_api.v2.registry import CalculationContext
from measure_api.v2.result_models import VisualizationRequest
from measure_api.v2.result_support import result_nodes

CURVE_FIELDS = {"gray", "smoothed", "gradient", "sample_points"}
SAMPLE_FIELDS = {"samples", "edges", "residuals"}


def numeric_arrays(value: Any, prefix: str = "") -> dict:
    """Collect supplied numeric samples with their explicit field provenance.

    Args:
        value: Supplied observation, geometry or numerical quality.
        prefix: Stable numeric field path within the observation.

    Returns:
        Numeric arrays indexed by their supplied field paths.
    """
    result = {}
    if isinstance(value, list):
        for index, child in enumerate(value):
            result.update(numeric_arrays(child, f"{prefix}_{index}"))
    if not isinstance(value, dict):
        return result
    if isinstance(value.get("point"), dict) and "amplitude" in value:
        p = value["point"]
        result[prefix or "edge"] = np.array([[p["x"], p["y"], value["amplitude"]]], np.float64)
        return result
    for key, child in value.items():
        name = f"{prefix}_{key}".strip("_")
        if key in CURVE_FIELDS | SAMPLE_FIELDS and isinstance(child, list) and child:
            if isinstance(child[0], dict):
                if "point" in child[0]:
                    array = [
                        [item["point"]["x"], item["point"]["y"], item.get("amplitude", 0)]
                        for item in child
                    ]
                else:
                    array = [[item["x"], item["y"]] for item in child]
            else:
                array = child
            result[name] = np.array(array, np.float64)
        else:
            result.update(numeric_arrays(child, name))
    return result


def read_numeric_npz(data: bytes, context: CalculationContext) -> dict:
    """Inspect ZIP bounds before decoding arrays; object/pickle payloads are rejected.

    Args:
        data: Supplied numerical or serialized result payload.
        context: Isolated calculation resources and output identity.

    Returns:
        Bounded finite numeric arrays without object payloads.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            entries = archive.infolist()
            if (
                len(entries) > 1000
                or sum(item.file_size for item in entries) > context.settings.max_input_bytes
            ):
                raise ServiceError(
                    "diagnostics.too_large", "Diagnostic array archive exceeds bounds"
                )
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            values = {key: archive[key] for key in archive.files}
        if any(
            array.dtype.kind not in "biuf" or array.size > 1000000 or not np.isfinite(array).all()
            for array in values.values()
        ):
            raise ValueError("Invalid numeric arrays")
        return values
    except (ValueError, OSError, zipfile.BadZipFile, KeyError) as error:
        raise ServiceError(
            "diagnostics.invalid", "Invalid non-pickle numeric diagnostics"
        ) from error


def curve_preview(arrays: dict, threshold: Optional[float]) -> bytes:
    """Draw actual gray/derivative curves without interactive windows.

    Args:
        arrays: Named numerical curves with sample provenance.
        threshold: Actual derivative threshold in gray levels.

    Returns:
        PNG bytes of the actual curves and derivative threshold.
    """
    from measure_api.v2.families.images import encode_png

    pixels = np.full((240, 480, 3), 255, np.uint8)
    for key, color, low, high in [
        ("gray", (120, 120, 120), 0, 255),
        ("smoothed", (255, 0, 0), 0, 255),
        ("gradient", (0, 0, 255), -255, 255),
    ]:
        values = np.asarray(arrays.get(key, []), np.float64).reshape(-1)
        if len(values) < 2:
            continue
        xs = np.linspace(10, 469, len(values))
        ys = 229 - np.clip((values - low) / (high - low), 0, 1) * 219
        points = np.round(np.column_stack([xs, ys])).astype(np.int32)
        cv2.polylines(pixels, [points], False, color, 1)
    for value in [threshold, -threshold] if threshold is not None else []:
        y = int(round(229 - (value + 255) / 510 * 219))
        cv2.line(pixels, (10, y), (469, y), (0, 180, 0), 1)
    return encode_png(pixels)


def selected_points(data: dict) -> list:
    """Count each selected observation once despite equivalent public geometry ports.

    Args:
        data: Supplied numerical or serialized result payload.

    Returns:
        Each actual selected xy observation exactly once.
    """
    if data.get("edges"):
        return [[edge["point"]["x"], edge["point"]["y"]] for edge in data["edges"]]
    if data.get("samples"):
        return [[point["x"], point["y"]] for point in data["samples"]]
    if data.get("pairs"):
        return [
            [pair[key]["point"]["x"], pair[key]["point"]["y"]]
            for pair in data["pairs"]
            for key in ["edge_a", "edge_b"]
        ]
    return []


def probe_provenance(run: dict, target: dict, node: dict) -> Any:
    """Describe actual configured sampling areas and their declared mapping to metrology.

    Args:
        run: Supplied run configuration and source provenance.
        target: Target-local frames, mappings and observations.
        node: Target-local node identity and observation state.

    Returns:
        Declared sampling polygons and their coordinate mapping, or None.
    """
    recipe = run.get("resolved_recipe", {})
    definition = next(
        (value for value in recipe.get("nodes", []) if value["id"] == node["id"]), None
    )
    if definition is None:
        return None
    params, areas = definition["parameters"], []
    probes = []
    if isinstance(params.get("probe"), dict):
        probes = [params["probe"]]
    elif node["operation"] in {"measurements.lines", "measurements.circles"}:
        count = params.get("probe_count", 20 if node["operation"].endswith("lines") else 24)
        if node["operation"].endswith("lines"):
            a, b = params["start"], params["end"]
            angle = np.rad2deg(np.arctan2(b["y"] - a["y"], b["x"] - a["x"])) + 90
            centers = [
                [a["x"] + fraction * (b["x"] - a["x"]), a["y"] + fraction * (b["y"] - a["y"])]
                for fraction in np.linspace(0, 1, count)
            ]
            directions = [angle] * count
        else:
            begin, end = params.get("start_angle_degrees", 0), params.get("end_angle_degrees", 360)
            span = (end - begin) % 360 or 360
            directions = begin + np.linspace(0, span, count, endpoint=False)
            radius, center = params["radius"], params["center"]
            centers = [
                [
                    center["x"] + radius * np.cos(np.deg2rad(a)),
                    center["y"] + radius * np.sin(np.deg2rad(a)),
                ]
                for a in directions
            ]
            directions = directions + 180
        probes = [
            {
                "center": {"x": x, "y": y},
                "angle_degrees": float(angle),
                "search_half_length": params.get("search_half_length", 15),
                "average_half_width": params.get("average_half_width", 5),
            }
            for (x, y), angle in zip(centers, directions)
        ]
    if not probes:
        return None
    for probe in probes:
        angle = np.deg2rad(probe.get("angle_degrees", 0))
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], np.float64)
        corners *= [probe["search_half_length"], probe["average_half_width"]]
        corners = corners @ rotation.T + [probe["center"]["x"], probe["center"]["y"]]
        areas.append(
            {
                "points": [{"x": float(x), "y": float(y)} for x, y in corners],
                "closed": True,
                "angle_degrees": probe.get("angle_degrees", 0),
            }
        )
    frame = (
        "aligned"
        if recipe.get("alignment") or recipe.get("localization")
        else recipe.get("source_frame", "source")
    )
    return {"frame": frame, "unit": "px", "areas": areas, "mapping": target.get("metric_mapping")}


def diagnostic_payloads(request: VisualizationRequest, context: CalculationContext) -> list:
    """Commit requested full samples as files, leaving normal HTTP JSON bounded.

    Args:
        request: Typed supplied observations and calculation controls.
        context: Isolated calculation resources and output identity.

    Returns:
        File payloads and their shared sampling/quality index.
    """
    payloads, index = [], []
    for serial, (item, run, target, node) in enumerate(result_nodes(request.result)):
        data = node.get(request.space + "_result") or node.get("result") or {}
        arrays = numeric_arrays(data)
        curves = {
            key: value
            for key, value in arrays.items()
            if any(key == field or key.endswith("_" + field) for field in CURVE_FIELDS)
        }
        samples = {key: value for key, value in arrays.items() if key not in curves}
        for descriptor in data.get("artifacts", []):
            if descriptor.get("kind") == "curves" and descriptor.get("format") == "npz":
                content, _ = context.files.read_bytes(
                    FileReference.model_validate(descriptor["file"])
                )
                curves.update(read_numeric_npz(content, context))
        for kind, values in [("curves", curves), ("selected_samples", samples)]:
            if not values:
                continue
            if sum(
                value.nbytes for value in values.values()
            ) > context.settings.max_input_bytes or any(
                value.size > 1000000 or not np.isfinite(value).all() for value in values.values()
            ):
                raise ServiceError("diagnostics.too_large", "Diagnostic arrays exceed bounds")
            buffer = io.BytesIO()
            np.savez_compressed(buffer, **values)
            payloads.append((f"{serial}-{kind}.npz", kind, "npz", buffer.getvalue()))
        if any(key in curves for key in ["gray", "smoothed", "gradient"]):
            payloads.append(
                (
                    f"{serial}-curve.png",
                    "curve_preview",
                    "png",
                    curve_preview(curves, data.get("threshold_gray")),
                )
            )
        index.append(
            {
                "item_id": item.get("item_id"),
                "target_id": target["target_id"],
                "node_id": node["id"],
                "operation": node["operation"],
                "frame": data.get("frame"),
                "unit": data.get("unit"),
                "valid": node["valid"],
                "quality": quality_summary(data.get("diagnostics")),
                "selected_sample_count": len(selected_points(data)),
                "probes": probe_provenance(run, target, node),
                "arrays": {
                    key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                    for key, value in {**curves, **samples}.items()
                },
                "error": node.get("error"),
            }
        )
    payloads.append(
        (
            "diagnostics.json",
            "sampling_diagnostics",
            "json",
            json.dumps(
                {"schema_version": "1.0", "nodes": index}, ensure_ascii=False, allow_nan=False
            ).encode("utf-8"),
        )
    )
    return payloads


def quality_summary(value: Any) -> Any:
    """Copy available scalar quality and declare omitted residual arrays by count.

    Args:
        value: Supplied observation, geometry or numerical quality.

    Returns:
        Available scalar quality, or None when no quality was supplied.
    """
    if not isinstance(value, dict):
        return None
    result = {key: item for key, item in value.items() if not isinstance(item, (dict, list))}
    if "residuals" in value:
        result["residual_count"] = len(value["residuals"])
    return result


def operation_diagnostics(result: dict, error: Any = None) -> dict:
    """Retain real stage timings, bounded selected samples and numerical failure details.

    Args:
        result: Supplied atomic, recipe or batch observations.
        error: Structured failure when a calculation did not complete.

    Returns:
        Bounded actual stage, sample, quality and failure summaries.
    """
    output = {}
    if error is not None:
        output["failure"] = error
    if not result:
        return output
    if "stage_timings" in result:
        output["stage_timings"] = result["stage_timings"]
    nodes = []
    if "targets" in result:
        for target in result["targets"]:
            for node in target.get("nodes", []):
                data = node.get("pixel_result") or node.get("result") or {}
                selected = selected_points(data)
                preview = selected[:32]
                count = len(selected)
                nodes.append(
                    {
                        "target_id": target["target_id"],
                        "node_id": node["id"],
                        "valid": node["valid"],
                        "quality": quality_summary(data.get("diagnostics")),
                        "selected_sample_count": count,
                        "selected_sample_preview": preview,
                        "samples_truncated": count > len(preview),
                        "error": node.get("error"),
                    }
                )
    elif "diagnostics" in result:
        output["quality"] = quality_summary(result["diagnostics"])
    if nodes:
        output["nodes"] = nodes
    if "items" in result:
        output["items"] = [
            {
                "id": item["id"],
                "diagnostics": operation_diagnostics(item.get("result"), item.get("error")),
            }
            for item in result["items"]
        ]
    if result.get("artifacts"):
        output["artifacts"] = result["artifacts"]
    return output
