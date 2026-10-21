"""One headless target-local executor reusing directly callable atomic operations."""

from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from pydantic import BaseModel, TypeAdapter, ValidationError

from measure_api.v2.atomic_types import Scalar
from measure_api.v2.errors import ServiceError
from measure_api.v2.files import SharedFiles
from measure_api.v2.models import FileReference, OutputSpec
from measure_api.v2.recipe_metrology import SampleMapper
from measure_api.v2.families.templates import coordinate_mapping
from measure_api.v2.families.images import roi_to_patch_matrix
from measure_api.v2.recipe_graph import bindings, dependency_closure, lookup, validate_recipe
from measure_api.v2.recipe_models import (
    Binding,
    NodeOutcome,
    PreviewRequest,
    Recipe,
    RecipeRequest,
    RecipeValidationResult,
    RunRequest,
    RunResult,
    TargetOutcome,
    RunBatchRequest,
    RunBatchResult,
    BatchOutcome,
)
from measure_api.v2.registry import CalculationContext
from measure_api.v2.versions import runtime_identity


def pin_parameters(
    parameters: Dict[str, Any], model: type, files: SharedFiles, field_path: str,
) -> dict:
    """Materialize known typed fields so nested shared references are pinned before admission.

    Args:
        parameters: Literal node or legacy tool parameters before typed normalization.
        model: Registered atomic request schema defining each dynamic parameter field.
        files: Configured shared-directory boundary with size limits and digest
            verification.
        field_path: Recipe node or stage parameter prefix for field-level error locations.

    Returns:
        Typed parameter values with all nested shared input references pinned.
    """
    values = {}
    for name, value in parameters.items():
        field = model.model_fields.get(name)
        if field is None:
            values[name] = value
            continue
        try:
            typed = TypeAdapter(field.rebuild_annotation()).validate_python(value)
        except ValidationError as error:
            details = [
                {"field_path": ".".join([field_path, name, *map(str, item["loc"])]),
                 "message": item["msg"], "type": item["type"]}
                for item in error.errors()
            ]
            raise ServiceError(
                "recipe.invalid_parameter", "Recipe parameter validation failed", 422,
                details={"errors": details}, field_path=details[0]["field_path"],
            ) from error
        values[name] = files.snapshot_request(typed)
    return values


def gather_assets(value: Any) -> List[FileReference]:
    """Collect explicit typed references; untyped strings never imply asset access.

    Args:
        value: Supplied value to normalize under the public schema and coordinate
            convention.

    Returns:
        Explicit file references collected recursively from models, lists and dicts.
    """
    if isinstance(value, FileReference):
        return [value]
    if isinstance(value, BaseModel):
        return [
            ref for name in type(value).model_fields for ref in gather_assets(getattr(value, name))
        ]
    if isinstance(value, dict):
        return [ref for item in value.values() for ref in gather_assets(item)]
    if isinstance(value, list):
        return [ref for item in value for ref in gather_assets(item)]
    return []


def expand_calibrations(value: Any, files: SharedFiles, sources: list) -> Any:
    """Normalize generic calibration values into family schemas before recursive file pinning.

    Args:
        value: Supplied value to normalize under the public schema and coordinate
            convention.
        files: Configured shared-directory boundary with size limits and digest
            verification.
        sources: Mutable accumulator of original calibration references for provenance.

    Returns:
        Equivalent value with loaded typed assets and authorized output locations.
    """
    from measure_api.v2.calibration_assets import CalibrationInput, load_calibration

    if isinstance(value, CalibrationInput):
        if value.file is not None:
            sources.append(files.snapshot_request(value.file))
        context = CalculationContext(files.settings, "prepare", "prepare")
        asset = files.snapshot_request(load_calibration(value, context))
        return CalibrationInput(value=asset)
    if isinstance(value, OutputSpec):
        files.resolve(value.directory, write=True)
        return value
    if isinstance(value, BaseModel):
        return value.model_copy(
            update={
                name: expand_calibrations(getattr(value, name), files, sources)
                for name in type(value).model_fields
            }
        )
    if isinstance(value, list):
        return [expand_calibrations(item, files, sources) for item in value]
    if isinstance(value, dict):
        return {key: expand_calibrations(item, files, sources) for key, item in value.items()}
    return value


def prepare_recipe(request: RecipeRequest, files: SharedFiles) -> RecipeRequest:
    """Expand file recipes and pin every known transitive input before idempotent admission.

    Args:
        request: Typed inline or file-backed upstream recipe snapshot for complete
            validation.
        files: Configured shared-directory boundary with size limits and digest
            verification.

    Returns:
        Inline or file-backed upstream recipe snapshot for complete validation.
    """
    recipe = request.recipe
    source = request.source_recipe
    if request.recipe_file is not None:
        data, source = files.read_bytes(request.recipe_file)
        try:
            recipe = Recipe.model_validate_json(data)
        except ValueError as error:
            raise ServiceError(
                "recipe.invalid_file", "Recipe file is not valid schema 1.0 JSON"
            ) from error
    recipe = files.snapshot_request(recipe)
    for index, node in enumerate(recipe.nodes):
        try:
            operation = lookup(node.operation)
        except ValueError:
            continue
        node.parameters = pin_parameters(
            node.parameters, operation.request_model, files, f"recipe.nodes.{index}.parameters"
        )
    from measure_api.v2.registry import operations

    for index, stage in enumerate(recipe.stages):
        operation = operations()[stage.operation]
        stage.parameters = pin_parameters(
            stage.parameters, operation.request_model, files, f"recipe.stages.{index}.parameters"
        )
    from measure_api.v2.families.templates import load_template

    context = CalculationContext(files.settings, "prepare", "prepare")
    templates = []
    if recipe.localization and recipe.localization.template:
        templates.append(recipe.localization.template)
    if recipe.alignment:
        templates.extend(control.template for control in recipe.alignment.controls)
    for node in recipe.nodes:
        if "template" in node.parameters:
            templates.append(node.parameters["template"])
    for template in templates:
        reference = (
            template
            if isinstance(template, FileReference)
            else FileReference.model_validate(template)
        )
        _, metadata, _ = load_template(reference, context)
        if recipe.reference is None:
            raise ServiceError(
                "recipe.reference_required", "Template provenance requires a reference image"
            )
        if metadata.source.sha256 != recipe.reference.sha256:
            raise ServiceError(
                "recipe.template_provenance_mismatch",
                "Template was taught from a different source content identity",
            )
    calibration_sources = []
    recipe = files.snapshot_request(expand_calibrations(recipe, files, calibration_sources))
    assets = gather_assets(recipe) + calibration_sources
    if source is not None:
        assets.append(source)
    return request.model_copy(
        update={"recipe": recipe, "recipe_file": None, "source_recipe": source, "assets": assets}
    )


def validate_snapshot(request: RecipeRequest) -> RecipeValidationResult:
    """Return full configuration validation, preserving pinned source identity.

    Args:
        request: Typed inline or file-backed upstream recipe snapshot for complete
            validation.

    Returns:
        Configuration issues, deterministic dependency order and canonical recipe identity.
    """
    result = validate_recipe(request.recipe)
    result.assets = request.assets
    return result


def read_port(outcome: NodeOutcome, binding: Binding, operation: Any) -> Any:
    """Preserve enclosing scalar validity rather than promoting raw numbers to observations.

    Args:
        outcome: Validated result or error record to persist at the completion boundary.
        binding: Named upstream node port and pixel/physical observation space.
        operation: Registered atomic operation defining request fields and named result
            ports.

    Returns:
        Bound port observation retaining its validity, frame and unit metadata.
    """
    data = outcome.physical_result if binding.space == "physical" else outcome.pixel_result
    if data is None:
        raise ValueError("Dependency has no " + binding.space + " observation")
    value = data.get(binding.port)
    if operation.output_ports[binding.port] == "number":
        return {
            "valid": data.get("valid", True) and value is not None,
            "reason": data.get("reason"),
            "frame": data.get("frame", "source"),
            "unit": data.get("unit", "px"),
            "value": value,
        }
    if issubclass(operation.result_model, Scalar) and binding.port == "value":
        return {key: data.get(key) for key in Scalar.model_fields}
    if value is None:
        kind = operation.output_ports[binding.port]
        if kind in {"point", "line", "circle", "scalar", "number"}:
            return {
                "valid": False,
                "reason": data.get("reason") or "unavailable_single_port",
                "frame": data.get("frame", "source"),
                "unit": data.get("unit", "px"),
            }
        raise ValueError("Dependency output port is unavailable")
    return value


def execute_nodes(
    recipe: Recipe,
    order: List[str],
    image: FileReference,
    context: CalculationContext,
    mapper: Optional[SampleMapper] = None,
) -> List[NodeOutcome]:
    """Execute deterministic dependencies with fresh outcomes scoped to one target.

    Args:
        recipe: Immutable upstream recipe with declared graph, image stages and metric
            frame.
        order: Topologically ordered enabled node identifiers for this target.
        image: Verified current target image or private aligned patch reference.
        context: Isolated operation identity, resource limits and verified shared-file
            access.
        mapper: Optional mapping chain for recovering samples before pixel/physical fitting.

    Returns:
        Fresh ordered node outcomes with separate pixel and physical observations.
    """
    nodes = {node.id: node for node in recipe.nodes}
    outcomes = {}
    for identity in order:
        node = nodes[identity]
        operation = lookup(node.operation)
        try:
            values = dict(node.parameters)
            if "image" in operation.request_model.model_fields:
                values["image"] = image
                values["frame"] = recipe.source_frame
            for name, supplied in node.inputs.items():
                resolved = [
                    read_port(
                        outcomes[binding.node], binding, lookup(nodes[binding.node].operation)
                    )
                    for binding in bindings(supplied)
                ]
                values[name] = resolved if isinstance(supplied, list) else resolved[0]
            request = operation.request_model.model_validate(values)
            data = operation.result_model.model_validate(operation.execute(request, context))
            dumped = data.model_dump(mode="json")
            pixel, physical = dumped, None
            if mapper is not None and "image" in operation.request_model.model_fields:
                pixel = mapper.convert(dumped, operation.operation_id)
                if mapper.asset is not None:
                    physical = mapper.convert(dumped, operation.operation_id, True)
            explicit_physical = node.space == "physical" or any(
                b.space == "physical" for value in node.inputs.values() for b in bindings(value)
            )
            if explicit_physical and "image" not in operation.request_model.model_fields:
                pixel, physical = None, dumped
            elif mapper is not None and mapper.asset is not None and node.inputs:
                physical_values = dict(node.parameters)
                for name, supplied in node.inputs.items():
                    resolved = [
                        read_port(
                            outcomes[b.node],
                            b.model_copy(update={"space": "physical"}),
                            lookup(nodes[b.node].operation),
                        )
                        for b in bindings(supplied)
                    ]
                    physical_values[name] = resolved if isinstance(supplied, list) else resolved[0]
                if operation.family == "evaluations":
                    if node.parameters.get("unit", "px") != mapper.asset.unit:
                        physical_values = None
                if physical_values is not None:
                    child = operation.request_model.model_validate(physical_values)
                    physical = operation.result_model.model_validate(
                        operation.execute(child, context)
                    ).model_dump(mode="json")
            authoritative = physical if explicit_physical else pixel
            outcomes[identity] = NodeOutcome(
                id=identity,
                operation=operation.operation_id,
                state="succeeded",
                valid=authoritative.get("valid", True),
                result=authoritative,
                pixel_result=pixel,
                physical_result=physical,
            )
        except Exception as error:
            blocked = any(
                outcomes[b.node].state != "succeeded"
                for supplied in node.inputs.values()
                for b in bindings(supplied)
            )
            invalid = {
                "valid": False,
                "reason": "failed_dependency" if blocked else str(error),
                "frame": recipe.source_frame,
                "unit": "px",
            }
            physical_invalid = None
            if mapper is not None and mapper.asset is not None:
                physical_invalid = {
                    **invalid,
                    "frame": recipe.metric.physical_frame,
                    "unit": mapper.asset.unit,
                }
            outcomes[identity] = NodeOutcome(
                id=identity,
                operation=operation.operation_id,
                state="blocked" if blocked else "failed",
                valid=False,
                pixel_result=invalid,
                physical_result=physical_invalid,
                error={"code": getattr(error, "code", "recipe.node_failed"), "message": str(error)},
            )
    return list(outcomes.values())


def execute_stages(
    recipe: Recipe,
    image: FileReference,
    context: CalculationContext,
    timings: dict,
    prefix: str,
    retain: bool = False,
) -> tuple:
    """Run correction/stitching in nonfinal private stages and retain mapping semantics.

    Args:
        recipe: Immutable upstream recipe with declared graph, image stages and metric
            frame.
        image: Pinned source image reference before correction or calibrated stitching.
        context: Isolated operation identity, resource limits and verified shared-file
            access.
        timings: Mutable stage-duration map populated while executing image stages.
        prefix: Parent-owned namespace prefix for isolated stage or target outputs.
        retain: Whether private stage images are included in the final artifact set.

    Returns:
        Final image/context, inverse mapping, canvas flag and retained stage provenance.
    """
    from measure_api.v2.registry import operations

    current, active = image, context
    inverse_chain, provenance, sources, payloads = [], [], [], []
    canvas = False
    for index, stage in enumerate(recipe.stages):
        started = time.monotonic()
        operation = operations()[stage.operation]
        child, output = context.stage(prefix + "-stage-" + str(index))
        values = dict(stage.parameters)
        if "image" in operation.request_model.model_fields:
            values["image"] = current
        values["output"] = output
        typed = operation.request_model.model_validate(values)
        result = operation.result_model.model_validate(operation.execute(typed, child))
        image_kind = "stitched" if stage.operation == "image-stitches" else "undistorted"
        current = next(a.file for a in result.artifacts if a.kind == image_kind)
        if stage.operation == "image-undistortions":
            detail = result.mapping.details
            inverse_chain.append(detail)
        else:
            if index != 0 or inverse_chain:
                raise ServiceError("recipe.stitch_order", "Stitch must be the first image stage")
            canvas = True
        originals = result.sources if hasattr(result, "sources") else [result.source]
        sources.extend(ref for ref in originals if ref.root != "__stages__")
        diagnostic = {
            "id": stage.id,
            "operation": stage.operation,
            "metadata": result.metadata.model_dump(mode="json"),
        }
        if hasattr(result, "mappings"):
            diagnostic["mappings"] = [m.model_dump(mode="json") for m in result.mappings]
            diagnostic["canvas"] = result.canvas.model_dump(mode="json")
            diagnostic["provenance"] = result.provenance
        else:
            diagnostic["mapping"] = result.mapping.model_dump(mode="json")
            diagnostic["mapping"]["details"].pop("inverse_map_artifact", None)
        provenance.append(diagnostic)
        if retain:
            payloads.append(
                (stage.id + ".png", "stage_image", "png", child.files.read_bytes(current)[0])
            )
        active = child
        timings[stage.id + "_ms"] = (time.monotonic() - started) * 1000

    def inverse(points: np.ndarray) -> np.ndarray:
        for detail in reversed(inverse_chain):
            new_k = np.array(detail["corrected_camera_matrix"])
            homogeneous = np.column_stack([points, np.ones(len(points))]) @ np.linalg.inv(new_k).T
            normalized = homogeneous[:, :2] / homogeneous[:, 2, None]
            object_points = np.column_stack([normalized, np.ones(len(points))])
            points = cv2.projectPoints(
                object_points,
                np.zeros(3),
                np.zeros(3),
                np.array(detail["camera_matrix"]),
                np.array(detail["distortion"]),
            )[0].reshape(-1, 2)
        return points

    return current, active, inverse, canvas, inverse_chain, provenance, sources, payloads


def localized_targets(recipe: Recipe, image: FileReference, context: CalculationContext) -> list:
    """Return only target-local poses; absence remains an explicit no_targets outcome.

    Args:
        recipe: Immutable upstream recipe with declared graph, image stages and metric
            frame.
        image: Current source or corrected/canvas image reference to localize.
        context: Isolated operation identity, resource limits and verified shared-file
            access.

    Returns:
        Target poses and patch mappings; an empty list denotes no located targets.
    """
    from measure_api.v2.families.templates import TemplateMatchRequest, match_template

    if recipe.localization is None:
        return [(None, np.eye(3))]
    local = recipe.localization
    if local.template is not None:
        matched = match_template(
            TemplateMatchRequest(image=image, template=local.template, search=local.search), context
        )
        return [
            (m.model_dump(mode="json"), np.array(m.mapping.matrix))
            for m in matched.matches[: local.max_targets]
        ]
    return [(None, np.eye(3))]


def align_target(
    recipe: Recipe,
    image: FileReference,
    mapping: np.ndarray,
    context: CalculationContext,
    prefix: str,
) -> tuple:
    """Reuse rigid/control alignment with a private cancellable isolated target patch.

    Args:
        recipe: Immutable upstream recipe with declared graph, image stages and metric
            frame.
        image: Current verified source or stage image reference for target alignment.
        mapping: Localization pose from reference pixels to the current image. Template
            alignment inverts this pose when constructing the current-image-to-patch transform.
        context: Isolated operation identity, resource limits and verified shared-file
            access.
        prefix: Parent-owned namespace prefix for isolated stage or target outputs.

    Returns:
        Target image/context, recovery transform and optional alignment outcome.
    """
    from measure_api.v2.families.alignment import (
        ControlAlignmentRequest,
        RigidAlignmentRequest,
        align_controls,
        align_rigid,
    )

    alignment = recipe.alignment
    if recipe.localization and recipe.localization.fixed_roi:
        roi = recipe.localization.fixed_roi
        matrix = roi_to_patch_matrix(roi)
        width, height = roi.width, roi.height
        if alignment is not None:
            matrix = np.array(alignment.reference_to_patch.matrix) @ matrix
            width, height = alignment.width, alignment.height
    elif alignment is None and recipe.localization and recipe.localization.template:
        if recipe.reference is None:
            raise ServiceError(
                "recipe.reference_required", "Template targets require a reference image"
            )
        reference, _ = context.files.read_image(recipe.reference)
        matrix = np.linalg.inv(mapping)
        height, width = reference.shape[:2]
    elif alignment is None:
        return image, context, np.eye(3), None
    else:
        matrix = np.array(alignment.reference_to_patch.matrix) @ np.linalg.inv(mapping)
        width, height = alignment.width, alignment.height
    child, output = context.stage(prefix + "-align")
    values = dict(
        image=image,
        initial_mapping=coordinate_mapping(matrix, recipe.source_frame, "aligned"),
        width=width,
        height=height,
        output=output,
    )
    if alignment is not None:
        values["border_policy"] = alignment.border_policy
    if alignment is not None and alignment.controls:
        values.update(
            controls=alignment.controls,
            minimum_controls=alignment.minimum_controls,
            robust=alignment.robust,
            max_residual=alignment.max_residual,
            estimate_scale=alignment.estimate_scale,
            failure_policy=alignment.failure_policy,
        )
        result = align_controls(ControlAlignmentRequest(**values), child)
    else:
        result = align_rigid(RigidAlignmentRequest(**values), child)
    if not result.valid:
        return None, child, None, result
    return result.image, child, np.array(result.metric_mapping.matrix), result


def target_summary(recipe: Recipe, nodes: List[NodeOutcome], preview: bool) -> tuple:
    by_id = {node.id: node for node in nodes}
    valid = all(
        node.valid for node in nodes if next(n for n in recipe.nodes if n.id == node.id).required
    )
    for binding in recipe.required_outputs:
        if binding.node not in by_id and preview:
            continue
        try:
            source = next(n for n in recipe.nodes if n.id == binding.node)
            port = read_port(by_id[binding.node], binding, lookup(source.operation))
            valid = valid and (not isinstance(port, dict) or port.get("valid", True))
        except (ValueError, KeyError):
            valid = False
    evaluations = [
        node.result["evaluation_status"]
        for node in nodes
        if node.result and "evaluation_status" in node.result
    ]
    conformity = (
        "fail"
        if "fail" in evaluations
        else (
            "pass"
            if evaluations and all(value == "pass" for value in evaluations)
            else "not_evaluated"
        )
    )
    return valid, conformity


def execute_run(request: RunRequest, context: CalculationContext) -> RunResult:
    """Execute supplied recipes with fresh targets, sample recovery and one final commit.

    Args:
        request: Typed immutable recipe snapshot, inspection image and parent
            execution/output controls.
        context: Isolated operation identity, resource limits and verified shared-file
            access.

    Returns:
        Fresh target/node outcomes, distinct pixel/physical observations and immutable input
        provenance.
    """
    started = time.monotonic()
    validated = validate_snapshot(request)
    if not validated.valid:
        raise ServiceError(
            "recipe.invalid", "Recipe validation failed", details={"issues": validated.issues}
        )
    recipe, order = request.recipe, validated.execution_order
    preview = isinstance(request, PreviewRequest)
    if preview:
        try:
            closure = dependency_closure(recipe, request.selected_nodes)
        except ValueError as error:
            raise ServiceError("recipe.invalid_selection", str(error)) from error
        order = [identity for identity in order if identity in closure]
    pixels, source = context.files.read_image(request.image)
    if recipe.metric and not recipe.stages:
        size = recipe.metric.applicability.image_size
        if pixels.shape[:2] != (size.height, size.width):
            raise ServiceError(
                "calibration.image_size_mismatch", "Image does not match metric applicability"
            )
    timings = {}
    prefix = context.operation_id.replace("-", "_")
    stage_data = execute_stages(
        recipe, source, context, timings, prefix, request.retain_stage_images
    )
    image, active, inverse, canvas, inverse_chain, provenance, stage_sources, payloads = stage_data
    started_match = time.monotonic()
    targets = localized_targets(recipe, image, active)
    timings["localization_ms"] = (time.monotonic() - started_match) * 1000

    def public_mapping(matrix: np.ndarray, source_frame: str):
        if not inverse_chain:
            return coordinate_mapping(matrix, source_frame, recipe.source_frame)
        from measure_api.v2.models import CoordinateMapping

        return CoordinateMapping(
            source_frame=source_frame,
            destination_frame=recipe.source_frame,
            type="nonlinear",
            details={
                "model": "patch_to_corrected_to_source",
                "patch_to_image": matrix.tolist(),
                "image_to_source": [
                    {key: value for key, value in detail.items() if key != "inverse_map_artifact"}
                    for detail in inverse_chain
                ],
            },
        )

    results = []
    for index, (matched, mapping) in enumerate(targets):
        target_started = time.monotonic()
        identity = "target-" + str(index + 1)
        try:
            target_image, target_context, metric_matrix, alignment = align_target(
                recipe, image, mapping, active, prefix + "-" + identity
            )
            if target_image is None:
                results.append(
                    TargetOutcome(
                        target_id=identity,
                        valid=False,
                        evaluation_status="not_evaluated",
                        reason=alignment.reason,
                        alignment={
                            "valid": False,
                            "controls": [c.model_dump() for c in alignment.controls],
                        },
                    )
                )
                continue
            mapper = SampleMapper(recipe, target_context, metric_matrix, inverse, canvas)
            nodes = execute_nodes(recipe, order, target_image, target_context, mapper)
            valid, conformity = target_summary(recipe, nodes, preview)
            diagnostic = None
            if alignment is not None:
                diagnostic = {
                    "valid": alignment.valid,
                    "used_fallback": alignment.used_fallback,
                    "controls": [c.model_dump(mode="json") for c in alignment.controls],
                    "correction": (
                        alignment.correction.model_dump(mode="json")
                        if alignment.correction
                        else None
                    ),
                }
            metric_mapping = public_mapping(metric_matrix, "aligned")
            localization = None
            if matched is not None:
                frame = provenance[-1]["metadata"]["frame"] if provenance else recipe.source_frame
                matched["mapping"]["destination_frame"] = frame
                matched["support_mapping"]["destination_frame"] = frame
                localization = {"frame": frame, "match": matched}
            results.append(
                TargetOutcome(
                    target_id=identity,
                    valid=valid,
                    evaluation_status=conformity,
                    nodes=nodes,
                    mapping=public_mapping(mapping, "reference") if matched is not None else None,
                    localization=localization,
                    metric_mapping=metric_mapping,
                    physical_mapping=mapper.physical_mapping(),
                    alignment=diagnostic,
                )
            )
        except Exception as error:
            results.append(
                TargetOutcome(
                    target_id=identity,
                    valid=False,
                    evaluation_status="not_evaluated",
                    reason=str(error),
                )
            )
        timings[identity + "_ms"] = (time.monotonic() - target_started) * 1000
    count = len(results)
    policy = recipe.target_policy
    count_ok = (policy.minimum is None or count >= policy.minimum) and (
        policy.maximum is None or count <= policy.maximum
    )
    evaluations = [target.evaluation_status for target in results]
    conformity = (
        "fail"
        if not count_ok or "fail" in evaluations
        else "pass" if evaluations and all(v == "pass" for v in evaluations) else "not_evaluated"
    )
    if not count and count_ok:
        conformity = policy.no_targets
    valid = bool(results) and all(target.valid for target in results)
    status = (
        "no_targets"
        if not results
        else "partial_failure" if any(not target.valid for target in results) else "completed"
    )
    timings["execution_ms"] = (time.monotonic() - started) * 1000
    source_map = {(ref.root, ref.path, ref.sha256): ref for ref in [source] + stage_sources}
    result = RunResult(
        **{**validated.model_dump(), "valid": valid},
        runtime_identity=runtime_identity(),
        sources=list(source_map.values()),
        measurement_image=None if canvas else source,
        image_stages=provenance,
        status=status,
        evaluation_status=conformity,
        targets=results,
        stage_timings=timings,
    )
    if preview:
        result.upstream_revision, result.preview_token = (
            request.upstream_revision,
            request.preview_token,
        )
    if request.output is not None:
        from measure_api.v2.files import ArtifactWriter

        if canvas and request.retain_stage_images:
            name = recipe.stages[-1].id + ".png"
            content = next(payload[3] for payload in payloads if payload[0] == name)
            directory = context.files.resolve(request.output.directory, write=True)
            pathname = directory / context.operation_id / name
            alias = request.output.directory.root
            if alias is not None:
                pathname = pathname.relative_to(context.settings.shared_roots[alias])
            result.measurement_image = FileReference(
                root=alias, path=str(pathname), sha256=hashlib.sha256(content).hexdigest()
            )
        result.artifacts, result.manifest = ArtifactWriter(
            context.files, request.output, context.operation_id
        ).commit(
            (payloads if request.retain_stage_images else [])
            + [("result.json", "recipe_result", "json", result.model_dump_json().encode())]
        )
    return result


def prepare_batch(request: RunBatchRequest, files: SharedFiles) -> RunBatchRequest:
    """Snapshot each supplied recipe/image independently, retaining item preparation failures.

    Args:
        request: Typed independent recipe/image items with batch execution and output
            controls.
        files: Configured shared-directory boundary with size limits and digest
            verification.

    Returns:
        Independent recipe/image items with batch execution and output controls.
    """
    prepared = prepare_recipe(request, files)
    items = []
    for item in request.items:
        try:
            image = files.snapshot_request(item.image)
            if image is None:
                raise ServiceError("recipe.image_missing", "Batch item requires an image")
            child = (
                RunRequest(image=image, recipe=item.recipe, recipe_file=item.recipe_file)
                if (item.recipe or item.recipe_file)
                else (RunRequest(image=image, recipe=prepared.recipe))
            )
            child = prepare_recipe(child, files)
            items.append(
                item.model_copy(
                    update={
                        "image": image,
                        "recipe": child.recipe,
                        "recipe_file": None,
                        "assets": child.assets,
                        "error": None,
                    }
                )
            )
        except Exception as error:
            items.append(
                item.model_copy(
                    update={
                        "image": None,
                        "recipe": None,
                        "recipe_file": None,
                        "assets": [],
                        "error": {
                            "code": getattr(error, "code", "recipe.item_prepare_failed"),
                            "message": str(error),
                        },
                    }
                )
            )
    return prepared.model_copy(update={"items": items})


def execute_batch(request: RunBatchRequest, context: CalculationContext) -> RunBatchResult:
    """Reuse the complete executor per image and commit only one final business manifest.

    Args:
        request: Typed independent recipe/image items with batch execution and output
            controls.
        context: Isolated operation identity, resource limits and verified shared-file
            access.

    Returns:
        Per-item run outcomes, batch quality and jointly committed report artifacts.
    """
    items = []
    for index, item in enumerate(request.items):
        if item.error is not None:
            items.append(BatchOutcome(id=item.id, state="failed", valid=False, error=item.error))
            continue
        try:
            child, _ = context.stage("batch-" + str(index))
            result = execute_run(
                RunRequest(
                    image=item.image,
                    recipe=item.recipe or request.recipe,
                    assets=item.assets or request.assets,
                ),
                child,
            )
            items.append(
                BatchOutcome(id=item.id, state="succeeded", valid=result.valid, result=result)
            )
        except Exception as error:
            items.append(
                BatchOutcome(
                    id=item.id,
                    state="failed",
                    valid=False,
                    error={
                        "code": getattr(error, "code", "recipe.item_failed"),
                        "message": str(error),
                    },
                )
            )
    result = RunBatchResult(valid=all(item.valid for item in items), items=items)
    if request.output is not None:
        from measure_api.v2.files import ArtifactWriter

        result.artifacts, result.manifest = ArtifactWriter(
            context.files, request.output, context.operation_id
        ).commit([("batch.json", "recipe_batch", "json", result.model_dump_json().encode())])
    return result
