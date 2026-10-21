"""Precision-preserving supplied-result CSV/JSON reports."""

from __future__ import annotations

import csv
import io
import json
from typing import Any, Iterator

from measure_api.v2.errors import ServiceError
from measure_api.v2.recipe_models import RunBatchResult
from measure_api.v2.result_models import SuppliedResult
from measure_api.v2.result_support import result_nodes

COLUMNS = [
    "recipe_id",
    "recipe_version",
    "configuration_sha256",
    "item_id",
    "target_id",
    "node_id",
    "label",
    "operation",
    "port",
    "space",
    "frame",
    "unit",
    "state",
    "valid",
    "evaluation_status",
    "value",
    "reason",
    "geometry_json",
    "error_json",
]


def observation_rows(
    value: Any, port: str = "value", frame: str = "source", unit: str = "px", valid: bool = True
) -> Iterator[dict]:
    """Flatten explicit scalar and geometry observations without zero-filling invalid data.

    Args:
        value: Supplied observation, geometry or numerical quality.
        port: Stable observation port path within the node.
        frame: Declared observation coordinate frame.
        unit: Declared observation measurement unit.
        valid: Enclosing observation validity.

    Returns:
        Scalar or geometry rows without zero-filling invalid observations.
    """
    if isinstance(value, list):
        for index, child in enumerate(value):
            yield from observation_rows(child, f"{port}[{index}]", frame, unit, valid)
        return
    if not isinstance(value, dict):
        return
    frame, unit, valid = (
        value.get("frame", frame),
        value.get("unit", unit),
        value.get("valid", valid),
    )
    fields = {
        "port": port,
        "frame": frame,
        "unit": unit,
        "valid": valid,
        "reason": value.get("reason"),
        "evaluation_status": value.get("evaluation_status"),
    }
    if not valid:
        yield {**fields, "value": None}
        return
    if "value" in value and isinstance(value["value"], (float, int, type(None))):
        yield {**fields, "value": value["value"]}
        return
    geometry = None
    if isinstance(value.get("point"), dict):
        geometry = value["point"]
    elif {"a", "b", "c"} <= set(value):
        geometry = {key: value.get(key) for key in ["a", "b", "c", "start", "end"]}
    elif "radius" in value and "center" in value:
        geometry = {"center": value["center"], "radius": value["radius"]}
    elif set(value) == {"x", "y"}:
        geometry = value
    if geometry is not None:
        yield {**fields, "geometry_json": json.dumps(geometry, ensure_ascii=False, allow_nan=False)}
        return
    for key, child in value.items():
        if key not in {
            "diagnostics",
            "gray",
            "smoothed",
            "gradient",
            "sample_points",
            "samples",
            "artifacts",
            "manifest",
            "distances",
            "center_distances",
        }:
            name = key if port == "value" else f"{port}.{key}"
            yield from observation_rows(child, name, frame, unit, valid)


def report_rows(result: SuppliedResult) -> list:
    """Keep pixel/physical rows and evaluation outcomes separate from measurement validity.

    Args:
        result: Supplied atomic, recipe or batch observations.

    Returns:
        Pixel, physical and failed-item records with separate evaluation outcomes.
    """
    rows = []
    for item, run, target, node in result_nodes(result):
        common = {
            "recipe_id": run.get("recipe_id"),
            "recipe_version": run.get("recipe_version"),
            "configuration_sha256": run.get("configuration_sha256"),
            **item,
            "target_id": target["target_id"],
            "node_id": node["id"],
            "label": node.get("label"),
            "operation": node["operation"],
            "state": node["state"],
            "error_json": (
                json.dumps(node["error"], ensure_ascii=False) if node.get("error") else None
            ),
        }
        emitted = False
        for space in ["pixel", "physical"]:
            data = node.get(space + "_result")
            if data is None:
                continue
            for observation in observation_rows(data):
                evaluation = (
                    observation.get("evaluation_status")
                    or data.get("evaluation_status")
                    or target["evaluation_status"]
                )
                rows.append(
                    {**common, "space": space, **observation, "evaluation_status": evaluation}
                )
                emitted = True
                if len(rows) > 200000:
                    raise ServiceError("result.too_large", "Too many report observations")
        if not emitted:
            data = node.get("result") or {}
            rows.append(
                {
                    **common,
                    "port": "value",
                    "valid": node["valid"],
                    "evaluation_status": data.get("evaluation_status", target["evaluation_status"]),
                    "reason": data.get("reason"),
                }
            )
    if isinstance(result, RunBatchResult):
        for item in result.items:
            if item.result is None:
                rows.append(
                    {
                        "item_id": item.id,
                        "state": item.state,
                        "valid": item.valid,
                        "evaluation_status": "not_evaluated",
                        "value": None,
                        "error_json": (
                            json.dumps(item.error, ensure_ascii=False) if item.error else None
                        ),
                    }
                )
    return rows


def encode_csv(rows: list) -> bytes:
    """Encode exact Python numeric representations with standard UTF-8 CSV quoting.

    Args:
        rows: Flattened report observations in caller order.

    Returns:
        UTF-8 CSV bytes with complete numerical precision.
    """
    target = io.StringIO(newline="")
    writer = csv.DictWriter(target, fieldnames=COLUMNS)
    writer.writeheader()
    writer.writerows(rows)
    return target.getvalue().encode("utf-8")
