"""Frozen supplied manifests replay the same executor without historical job lookups."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Literal, Optional

from pydantic import Field, model_validator

from measure_api.v2.errors import ServiceError
from measure_api.v2.files import SharedFiles
from measure_api.v2.models import CalculationRequest, FileReference, OutputSpec, StrictModel
from measure_api.v2.recipe_execution import execute_run, prepare_recipe, validate_snapshot
from measure_api.v2.recipe_models import Recipe, RunRequest, RunResult
from measure_api.v2.registry import CalculationContext
from measure_api.v2.versions import RuntimeIdentity, runtime_identity


class FrozenManifest(StrictModel):
    """Complete verified upstream snapshot; prior IDs are optional provenance only."""

    schema_version: Literal["1.0"] = "1.0"
    format: Literal["measure-frozen-run"] = "measure-frozen-run"
    recipe: Recipe
    configuration_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    image: FileReference
    sources: List[FileReference] = Field(min_length=1, max_length=10000)
    assets: List[FileReference] = Field(default_factory=list, max_length=10000)
    runtime_identity: RuntimeIdentity
    expected: Dict[str, Any]
    tolerance: float = Field(default=1e-9, ge=0, le=1)
    provenance: Dict[str, Any] = Field(default_factory=dict)


def numerical_record(run: RunResult) -> Dict[str, Any]:
    """Retain business/numerical observations, excluding operational identities and files.

    Args:
        run: Complete observed run with immutable producer, recipe and input identities.

    Returns:
        Numerical/categorical business observations without runtime or file identities.
    """
    return {
        "valid": run.valid,
        "status": run.status,
        "evaluation_status": run.evaluation_status,
        "targets": [
            {
                "valid": target.valid,
                "evaluation_status": target.evaluation_status,
                "reason": target.reason,
                "nodes": [
                    {
                        "id": node.id,
                        "operation": node.operation,
                        "state": node.state,
                        "valid": node.valid,
                        "result": node.result,
                        "pixel_result": node.pixel_result,
                        "physical_result": node.physical_result,
                    }
                    for node in target.nodes
                ],
            }
            for target in run.targets
        ],
    }


def freeze_run(run: RunResult) -> FrozenManifest:
    """Freeze only complete observed runs with explicit immutable producer/input identities.

    Args:
        run: Complete observed run with immutable producer, recipe and input identities.

    Returns:
        Immutable run snapshot with producer versions and verified replay input identities.
    """
    if run.runtime_identity is None:
        raise ServiceError(
            "replay.unknown_producer", "Supplied run has no observed runtime identity"
        )
    if not run.sources or run.preview_token is not None:
        raise ServiceError(
            "replay.incomplete_run", "Only full runs with input provenance can freeze"
        )
    references = run.sources + run.assets
    if any(reference.sha256 is None for reference in references):
        raise ServiceError(
            "replay.unverified_input", "Every frozen input requires its original digest"
        )
    return FrozenManifest(
        recipe=run.resolved_recipe,
        configuration_sha256=run.configuration_sha256,
        image=run.sources[0],
        sources=run.sources,
        assets=run.assets,
        runtime_identity=run.runtime_identity,
        expected=numerical_record(run),
    )


class ReplayRequest(CalculationRequest):
    frozen: Optional[FrozenManifest] = None
    frozen_file: Optional[FileReference] = None
    output: OutputSpec
    source_manifest: Optional[FileReference] = None
    prepared_run: Optional[RunRequest] = None

    @model_validator(mode="after")
    def one_snapshot(self) -> ReplayRequest:
        if (self.frozen is None) == (self.frozen_file is None):
            raise ValueError("Supply exactly one frozen or frozen_file")
        return self


class ReplayResult(StrictModel):
    run: RunResult
    equivalent: bool
    differences: List[str] = Field(default_factory=list)
    original_configuration_sha256: str
    original_runtime_identity: RuntimeIdentity
    source_manifest: Optional[FileReference] = None
    provenance: Dict[str, Any] = Field(default_factory=dict)


def prepare_replay(request: ReplayRequest, files: SharedFiles) -> ReplayRequest:
    """Verify supplied snapshot/version and every transitive file before admission.

    Args:
        request: Typed frozen recipe/input identities and explicit comparison tolerance.
        files: Configured shared-directory boundary with size limits and digest
            verification.

    Returns:
        Frozen recipe/input identities and explicit comparison tolerance.
    """
    frozen, source = request.frozen, None
    if request.frozen_file is not None:
        data, source = files.read_bytes(request.frozen_file)
        try:
            frozen = FrozenManifest.model_validate_json(data)
        except ValueError as error:
            raise ServiceError(
                "replay.incompatible_manifest", "Frozen manifest must use supported schema 1.0"
            ) from error
    current = runtime_identity()
    if frozen.runtime_identity != current:
        raise ServiceError(
            "replay.incompatible_version",
            "Exact algorithm and runtime identity is required",
            409,
            details={
                "original": frozen.runtime_identity.model_dump(mode="json"),
                "current": current.model_dump(mode="json"),
            },
        )
    for reference in [frozen.image] + frozen.sources + frozen.assets:
        if reference.sha256 is None:
            raise ServiceError("replay.unverified_input", "Frozen input has no expected digest")
        try:
            files.read_bytes(reference)
        except ServiceError as error:
            error.details.update({"replay_input": reference.model_dump(mode="json")})
            raise
    run = prepare_recipe(
        RunRequest(recipe=frozen.recipe, image=frozen.image, output=request.output), files
    )
    validated = validate_snapshot(run)
    if validated.configuration_sha256 != frozen.configuration_sha256:
        raise ServiceError(
            "replay.configuration_mismatch", "Frozen configuration digest disagrees", 409
        )
    if not validated.valid:
        raise ServiceError(
            "recipe.invalid",
            "Frozen recipe cannot execute",
            details={"issues": validated.issues},
        )
    return request.model_copy(
        update={
            "frozen": frozen,
            "frozen_file": None,
            "source_manifest": source,
            "prepared_run": run,
        }
    )


def compare_records(expected: Any, actual: Any, tolerance: float, path: str = "") -> List[str]:
    """Compare ordered observations with explicit tolerance and categorical identity.

    Args:
        expected: Frozen numerical/categorical business observations for comparison.
        actual: Newly computed numerical and categorical observations to compare.
        tolerance: Allowed absolute difference for finite numeric observations.
        path: Current nested observation path used to locate reported differences.

    Returns:
        Paths of differing numerical or categorical observations; empty means equivalent.
    """
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected) != set(actual):
            return [path or "/"]
        return [
            item
            for key in expected
            for item in compare_records(expected[key], actual[key], tolerance, path + "/" + key)
        ]
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [path or "/"]
        return [
            item
            for index, (first, second) in enumerate(zip(expected, actual))
            for item in compare_records(first, second, tolerance, path + "/" + str(index))
        ]
    if (
        isinstance(expected, (int, float))
        and not isinstance(expected, bool)
        and isinstance(actual, (int, float))
        and not isinstance(actual, bool)
    ):
        return [] if math.isclose(expected, actual, rel_tol=0, abs_tol=tolerance) else [path]
    return [] if type(expected) is type(actual) and expected == actual else [path]


def replay(request: ReplayRequest, context: CalculationContext) -> ReplayResult:
    """Execute the same recipe chain in a new namespace and report numerical identity.

    Args:
        request: Typed frozen recipe/input identities and explicit comparison tolerance.
        context: Isolated operation identity, resource limits and verified shared-file
            access.

    Returns:
        New observed run and numerical equivalence diagnostics against the frozen snapshot.
    """
    run = execute_run(request.prepared_run, context)
    differences = compare_records(
        request.frozen.expected, numerical_record(run), request.frozen.tolerance
    )
    return ReplayResult(
        run=run,
        equivalent=not differences,
        differences=differences[:1000],
        original_configuration_sha256=request.frozen.configuration_sha256,
        original_runtime_identity=request.frozen.runtime_identity,
        source_manifest=request.source_manifest,
        provenance=request.frozen.provenance,
    )
