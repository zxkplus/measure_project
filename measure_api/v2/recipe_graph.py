"""Typed recipe validation, deterministic dependency closure and canonical identity."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Union, get_args, get_origin

from pydantic import TypeAdapter, ValidationError

from measure_api.v2.atomic_types import Circle, Line, PositionedPoint, Scalar
from measure_api.v2.models import CalculationRequest
from measure_api.v2.recipe_models import Binding, Recipe, RecipeValidationResult
from measure_api.v2.registry import Operation, operations


ALLOWED_FAMILIES = {"measurements", "geometry", "evaluations", "templates"}
SCHEDULING = set(CalculationRequest.model_fields)
PORT_TYPES = {
    "point": PositionedPoint,
    "line": Line,
    "circle": Circle,
    "number": Scalar,
    "scalar": Scalar,
}


def lookup(identifier: str) -> Operation:
    """Resolve registered atomic identifiers without allowing recursive recipe execution.

    Args:
        identifier: Public semantic operation identifier to resolve for a recipe node.

    Returns:
        Registered atomic operation allowed inside the measurement dependency graph.
    """
    identity = identifier.strip("/").replace("/", ".").replace(":", "_")
    operation = operations().get(identity)
    if operation is None or operation.family not in ALLOWED_FAMILIES:
        raise ValueError("Unsupported recipe node operation: " + identifier)
    if identity in {"template-builds", "template-inspections", "template-matches"}:
        raise ValueError("Teaching and parent matching belong outside measurement nodes")
    return operation


def bindings(value: Any) -> List[Binding]:
    return value if isinstance(value, list) else [value]


def port_type(operation: Operation, port: str) -> Any:
    if port not in operation.output_ports:
        raise ValueError("Unknown named output port: " + port)
    kind = operation.output_ports[port]
    if kind in PORT_TYPES:
        return PORT_TYPES[kind]
    field = operation.result_model.model_fields.get(port)
    if field is None:
        raise ValueError("Port has no result field: " + port)
    annotation = field.annotation
    if get_origin(annotation) is Union:
        alternatives = [a for a in get_args(annotation) if a is not type(None)]
        return alternatives[0] if len(alternatives) == 1 else annotation
    return annotation


def accepts(destination: Any, source: Any) -> bool:
    if destination == source:
        return True
    if get_origin(destination) is list:
        return get_origin(source) is list and accepts(get_args(destination)[0], get_args(source)[0])
    if get_origin(destination) is not None:
        return any(accepts(option, source) for option in get_args(destination))
    if isinstance(destination, type) and isinstance(source, type):
        return issubclass(source, destination)
    return False



def check_metric_frame(recipe: Recipe, asset: Any) -> None:
    """Require calibration to describe the frame in which recovered samples are measured.

    Args:
        recipe: Immutable upstream recipe with declared graph, image stages and metric
            frame.
        asset: Validated calibration value, including its declared setup and frame.
    """
    from measure_api.v2.calibration_assets import calibration_digest, validate_calibration
    from measure_api.v2.families.plane_stereo import PlaneCalibration

    stitches = [stage for stage in recipe.stages if stage.operation == "image-stitches"]
    if stitches:
        stage_calibration = stitches[0].parameters.get("calibration")
        stage_value = (
            stage_calibration.value
            if hasattr(stage_calibration, "value") else stage_calibration["value"]
        )
        stage_asset = validate_calibration(stage_value)
        if isinstance(asset, PlaneCalibration):
            if (recipe.source_frame != asset.parameters.canvas.frame
                    or calibration_digest(asset) != calibration_digest(stage_asset)):
                raise ValueError("Metric calibration does not describe the stitched canvas")
            return
        if asset.applicability.image_size != stage_asset.parameters.canvas.image_size:
            raise ValueError("Metric calibration image size does not match the stitched canvas")
    if recipe.metric.applicability.frame != recipe.source_frame:
        raise ValueError("Metric calibration frame does not match the measurement source frame")


def validate_recipe(recipe: Recipe) -> RecipeValidationResult:
    """Validate the full supplied graph without computing or fabricating observations.

    Args:
        recipe: Immutable upstream recipe with declared graph, image stages and metric
            frame.

    Returns:
        Configuration issues, deterministic dependency order and canonical recipe identity.
    """
    issues: List[Dict[str, Any]] = []
    ids = [node.id for node in recipe.nodes]
    nodes = {node.id: node for node in recipe.nodes}
    if len(nodes) != len(ids):
        issues.append({"code": "duplicate_id", "message": "Node IDs must be unique"})
    dependencies = {node.id: set() for node in recipe.nodes if node.enabled}
    registered = {}
    for node in recipe.nodes:
        try:
            operation = lookup(node.operation)
            registered[node.id] = operation
            fields = operation.request_model.model_fields
            if SCHEDULING.intersection(node.parameters) or {"image", "output"}.intersection(
                node.parameters
            ):
                raise ValueError("Node scheduling and output directories belong to the parent")
            if set(node.parameters).intersection(node.inputs):
                raise ValueError("Literal and bound input fields cannot overlap")
            if node.space == "physical" and recipe.metric is None:
                raise ValueError("Physical node requires a supplied metric calibration")
            for name, value in node.parameters.items():
                if name not in fields:
                    raise ValueError("Unknown parameter: " + name)
                TypeAdapter(fields[name].rebuild_annotation()).validate_python(value)
            missing = {name for name, field in fields.items() if field.is_required()}
            missing -= set(node.parameters) | set(node.inputs) | {"image", "output"}
            if missing:
                raise ValueError("Missing required inputs: " + ", ".join(sorted(missing)))
            for base in operation.request_model.__mro__[1:]:
                if (
                    isinstance(base, type)
                    and issubclass(base, CalculationRequest)
                    and base is not CalculationRequest
                ):
                    base_values = {
                        name: value
                        for name, value in node.parameters.items()
                        if name in base.model_fields
                    }
                    required = {
                        name for name, field in base.model_fields.items() if field.is_required()
                    }
                    if required <= set(base_values):
                        base.model_validate(base_values)
            if not node.inputs:
                values = dict(node.parameters)
                if "image" in fields:
                    values["image"] = {"root": "_validation", "path": "image.png"}
                operation.request_model.model_validate(values)
        except (ValueError, ValidationError) as error:
            issues.append({"code": "invalid_node", "node": node.id, "message": str(error)})
    for node in recipe.nodes:
        if not node.enabled:
            continue
        operation = registered.get(node.id)
        if operation is None:
            continue
        for field_name, supplied in node.inputs.items():
            try:
                field = operation.request_model.model_fields.get(field_name)
                if field is None or field_name in SCHEDULING or field_name in {"image", "output"}:
                    raise ValueError("Unknown or managed input field: " + field_name)
                destination = field.annotation
                if isinstance(supplied, list):
                    if get_origin(destination) is not list:
                        raise ValueError("Explicit binding list requires an array input")
                    destination = get_args(destination)[0]
                for binding in bindings(supplied):
                    source = nodes.get(binding.node)
                    if source is None or not source.enabled:
                        raise ValueError("Missing or disabled dependency: " + binding.node)
                    dependencies[node.id].add(source.id)
                    source_operation = registered.get(source.id)
                    if source_operation is None:
                        raise ValueError("Invalid source operation")
                    source_type = port_type(source_operation, binding.port)
                    if not accepts(destination, source_type):
                        raise ValueError("Incompatible single/array or observation port type")
                    if binding.space == "physical" and recipe.metric is None:
                        raise ValueError("Physical binding requires calibration")
                    if node.operation.startswith("geometry."):
                        explicit_frame = source.parameters.get("frame")
                        if explicit_frame and explicit_frame != recipe.source_frame:
                            raise ValueError("Dependency has incompatible declared frame")
                spaces = {binding.space for binding in bindings(supplied)}
                if len(spaces) > 1:
                    raise ValueError("Array binding mixes pixel and physical frames")
            except (ValueError, ValidationError) as error:
                issues.append(
                    {
                        "code": "invalid_binding",
                        "node": node.id,
                        "field": field_name,
                        "message": str(error),
                    }
                )
        spaces = {b.space for value in node.inputs.values() for b in bindings(value)}
        if node.space == "physical" and "pixel" in spaces:
            issues.append(
                {
                    "code": "frame_mismatch",
                    "node": node.id,
                    "message": "Physical node cannot consume pixel bindings",
                }
            )
        expected_space = "physical" if "physical" in spaces or node.space == "physical" else "pixel"
        expected_frame = (
            recipe.metric.physical_frame
            if expected_space == "physical" and recipe.metric
            else recipe.source_frame
        )
        expected_unit = (
            recipe.metric.calibration.value.unit
            if expected_space == "physical" and recipe.metric and recipe.metric.calibration.value
            else "px"
        )
        if operation.family == "geometry":
            if (
                node.parameters.get("frame", expected_frame) != expected_frame
                or node.parameters.get("unit", expected_unit) != expected_unit
            ):
                issues.append(
                    {
                        "code": "frame_mismatch",
                        "node": node.id,
                        "message": "Geometry frame/unit is incompatible with declared space",
                    }
                )
        for name, literal in node.parameters.items():
            if isinstance(literal, dict) and "valid" in literal:
                frame, unit = literal.get("frame", "source"), literal.get("unit", "px")
                if frame != expected_frame or unit != expected_unit:
                    issues.append(
                        {
                            "code": "frame_mismatch",
                            "node": node.id,
                            "message": "Literal frame/unit is incompatible with bindings",
                        }
                    )
        if len(spaces) > 1:
            issues.append(
                {
                    "code": "frame_mismatch",
                    "node": node.id,
                    "message": "Node inputs mix pixel and physical frames",
                }
            )
    for binding in recipe.required_outputs:
        try:
            source = nodes.get(binding.node)
            if source is None or not source.enabled:
                raise ValueError("Required output source is missing or disabled")
            port_type(registered[binding.node], binding.port)
            if binding.space == "physical" and recipe.metric is None:
                raise ValueError("Physical required output requires calibration")
            source_spaces = {b.space for value in source.inputs.values() for b in bindings(value)}
            if (
                (source.space == "physical" or "physical" in source_spaces)
                and "image" not in registered[source.id].request_model.model_fields
                and binding.space == "pixel"
            ):
                raise ValueError(
                    "Required output selects pixel space from a physical-only provider"
                )
        except (ValueError, KeyError) as error:
            issues.append({"code": "required_output_invalid", "message": str(error)})
    if recipe.alignment is not None:
        try:
            from measure_api.v2.families.alignment import validated_matrix

            validated_matrix(recipe.alignment.reference_to_patch, similarity=True)
            if recipe.alignment.reference_to_patch.unit != "px":
                raise ValueError("Target alignment operates in pixel coordinates")
        except Exception as error:
            issues.append({"code": "alignment_invalid", "message": str(error)})
    if recipe.metric is not None:
        try:
            from measure_api.v2.calibration_assets import check_applicability, validate_calibration

            asset = validate_calibration(recipe.metric.calibration.value)
            check_applicability(asset, recipe.metric.applicability)
            check_metric_frame(recipe, asset)
            if asset.kind == "camera":
                raise ValueError("Camera intrinsics alone cannot provide physical measurements")
        except Exception as error:
            issues.append({"code": "metric_invalid", "message": str(error)})
    stage_ids = [stage.id for stage in recipe.stages]
    if len(set(stage_ids)) != len(stage_ids):
        issues.append({"code": "stage_duplicate_id", "message": "Image stage IDs must be unique"})
    for index, stage in enumerate(recipe.stages):
        try:
            operation = operations()[stage.operation]
            if SCHEDULING.intersection(stage.parameters) or "output" in stage.parameters:
                raise ValueError("Stage scheduling/output belongs to the parent operation")
            values = dict(stage.parameters)
            if "image" in operation.request_model.model_fields:
                values["image"] = {"root": "_validation", "path": "image.png"}
            values["output"] = {"directory": {"root": "_validation", "path": "output"}}
            operation.request_model.model_validate(values)
            if stage.operation == "image-stitches":
                if index != 0:
                    raise ValueError("Stitch must be the first stage")
                from measure_api.v2.calibration_assets import validate_calibration

                calibration = values["calibration"]
                value = calibration.value if hasattr(calibration, "value") else calibration["value"]
                canvas = validate_calibration(value).parameters.canvas
                if recipe.source_frame != canvas.frame:
                    raise ValueError(
                        "Stitch metrology pixel frame must equal the calibrated canvas frame"
                    )
        except Exception as error:
            issues.append({"code": "stage_invalid", "stage": stage.id, "message": str(error)})
    order = []
    pending = {key: set(value) for key, value in dependencies.items()}
    while pending:
        ready = sorted(key for key, deps in pending.items() if not deps)
        if not ready:
            issues.append({"code": "cycle", "message": "Dependency graph contains a cycle"})
            break
        for key in ready:
            order.append(key)
            del pending[key]
        for deps in pending.values():
            deps.difference_update(ready)
    canonical = json.dumps(
        recipe.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return RecipeValidationResult(
        valid=not issues,
        recipe_id=recipe.id,
        recipe_version=recipe.version,
        configuration_sha256=hashlib.sha256(canonical).hexdigest(),
        execution_order=order,
        issues=issues,
        resolved_recipe=recipe,
    )


def dependency_closure(recipe: Recipe, selected: List[str]) -> set:
    nodes = {node.id: node for node in recipe.nodes}
    closure = set()

    def visit(identity: str) -> None:
        if identity not in nodes or not nodes[identity].enabled:
            raise ValueError("Preview selection is missing or disabled: " + identity)
        if identity in closure:
            return
        closure.add(identity)
        for value in nodes[identity].inputs.values():
            for binding in bindings(value):
                visit(binding.node)

    for identity in selected:
        visit(identity)
    return closure
