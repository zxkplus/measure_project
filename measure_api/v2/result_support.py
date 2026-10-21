"""Resolve supplied result inputs without requiring an engine business-result store."""

from __future__ import annotations

import json
from typing import Any, Iterator, Tuple

from pydantic import TypeAdapter, ValidationError

from measure_api.v2.errors import ServiceError
from measure_api.v2.files import SharedFiles
from measure_api.v2.models import FileReference
from measure_api.v2.recipe_models import RunBatchResult, RunResult
from measure_api.v2.registry import operations
from measure_api.v2.result_models import AtomicSuppliedResult, SuppliedResult, SuppliedResultRequest


def prepare_result(request: SuppliedResultRequest, files: SharedFiles) -> SuppliedResultRequest:
    """Pin JSON and transitive file identities before job admission and retry matching.

    Args:
        request: Typed supplied observations and calculation controls.
        files: Configured shared-file access with content verification.

    Returns:
        The supplied record with verified transitive file identities.
    """
    result, source = request.result, None
    if request.result_file is not None:
        data, source = files.read_bytes(request.result_file)
        try:
            result = TypeAdapter(SuppliedResult).validate_json(data)
        except ValidationError as error:
            raise ServiceError(
                "result.invalid",
                "Supplied JSON does not match a result schema",
                details={"validation": str(error)},
            ) from error
    if isinstance(result, AtomicSuppliedResult):
        operation = operations().get(result.operation)
        if operation is None:
            raise ServiceError("result.unknown_operation", "Unknown atomic result operation")
        try:
            result = result.model_copy(
                update={
                    "value": operation.result_model.model_validate(result.value).model_dump(
                        mode="json"
                    )
                }
            )
        except ValidationError as error:
            raise ServiceError(
                "result.invalid",
                "Atomic result does not match its public schema",
                details={"validation": str(error)},
            ) from error
    runs = (
        [result]
        if isinstance(result, RunResult)
        else (
            [item.result for item in result.items if item.result is not None]
            if isinstance(result, RunBatchResult)
            else []
        )
    )
    for run in runs:
        for target in run.targets:
            for node in target.nodes:
                operation = operations().get(node.operation)
                if operation is None:
                    raise ServiceError(
                        "result.unknown_operation", "Unknown supplied node operation"
                    )
                if node.state != "succeeded":
                    continue
                for field in ["result", "pixel_result", "physical_result"]:
                    value = getattr(node, field)
                    if value is None:
                        continue
                    metadata = {key: value[key] for key in ["calibration_sha256"] if key in value}
                    data = {key: item for key, item in value.items() if key not in metadata}
                    try:
                        operation.result_model.model_validate(data, strict=True)
                    except ValidationError as error:
                        raise ServiceError(
                            "result.invalid",
                            "Node result does not match its atomic schema",
                            details={"node_id": node.id, "space": field, "validation": str(error)},
                        ) from error
    assets = []

    def pin(value: Any) -> Any:
        if isinstance(value, list):
            return [pin(item) for item in value]
        if isinstance(value, dict):
            if {"file", "sha256", "size_bytes", "kind", "format"} <= set(value):
                try:
                    reference = FileReference.model_validate(value["file"])
                    if (
                        reference.sha256 is not None
                        and reference.sha256.lower() != value["sha256"].lower()
                    ):
                        raise ServiceError(
                            "result.artifact_metadata_mismatch",
                            "Artifact digest fields disagree",
                            409,
                        )
                    reference = reference.model_copy(update={"sha256": value["sha256"]})
                    content, pinned = files.read_bytes(reference)
                    if len(content) != value["size_bytes"]:
                        raise ServiceError(
                            "result.artifact_metadata_mismatch",
                            "Artifact byte size is inconsistent",
                            409,
                        )
                    assets.append(pinned)
                    return {**value, "file": pinned.model_dump(mode="json")}
                except (ValidationError, TypeError, AttributeError) as error:
                    raise ServiceError(
                        "result.invalid_reference", "Invalid artifact descriptor"
                    ) from error
            if "path" in value and set(value) <= {"root", "path", "sha256", "provenance"}:
                try:
                    reference = FileReference.model_validate(value)
                except ValidationError as error:
                    raise ServiceError(
                        "result.invalid_reference", "Invalid nested file reference"
                    ) from error
                reference = files.read_bytes(reference)[1]
                assets.append(reference)
                return reference.model_dump(mode="json")
            return {key: pin(item) for key, item in value.items()}
        return value

    try:
        payload = pin(result.model_dump(mode="json"))
        result = TypeAdapter(SuppliedResult).validate_python(payload)
        # Non-finite nested dict values are not caught by Pydantic's float fields.
        json.dumps(payload, allow_nan=False)
    except (ValueError, TypeError) as error:
        raise ServiceError("result.invalid", "Result contains invalid numeric data") from error
    if hasattr(request, "image") and request.image is None and isinstance(result, RunResult):
        request = request.model_copy(update={"image": result.measurement_image})
    updates = {"result": result, "result_file": None, "source_result": source, "assets": assets}
    return request.model_copy(update=updates)


def result_nodes(result: SuppliedResult) -> Iterator[Tuple[dict, dict, dict, dict]]:
    """Yield item/run, target and node records with display labels attached.

    Args:
        result: Supplied atomic, recipe or batch observations.

    Returns:
        Item, run, target and node records with upstream display labels.
    """
    if isinstance(result, AtomicSuppliedResult):
        node = {
            "id": result.node_id,
            "label": result.label,
            "operation": result.operation,
            "state": "succeeded",
            "valid": result.value.get("valid", True),
            "result": result.value,
            "pixel_result": result.value if result.space == "pixel" else None,
            "physical_result": result.value if result.space == "physical" else None,
        }
        yield {}, {"evaluation_status": result.evaluation_status}, {
            "target_id": result.target_id,
            "evaluation_status": result.evaluation_status,
        }, node
        return
    runs = (
        [(None, result)]
        if isinstance(result, RunResult)
        else [(item.id, item.result) for item in result.items if item.result is not None]
    )
    for item_id, run in runs:
        labels = {node.id: node.label for node in run.resolved_recipe.nodes}
        info = run.model_dump(mode="json")
        for target in run.targets:
            for node in target.nodes:
                value = node.model_dump(mode="json")
                value["label"] = labels.get(node.id)
                yield {"item_id": item_id}, info, target.model_dump(mode="json"), value
