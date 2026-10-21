"""Bounded, versioned recipe packages with explicit shared asset relocation."""

from __future__ import annotations

import hashlib
import io
import json
import math
import stat
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Literal, Optional, Tuple

import cv2
import numpy as np
from pydantic import Field

from measure_api.v2.errors import ServiceError
from measure_api.v2.files import ArtifactWriter, SharedFiles, encoded_dimensions
from measure_api.v2.models import (
    CalculationRequest,
    FileDescriptor,
    FileReference,
    OutputSpec,
    StrictModel,
)
from measure_api.v2.recipe_execution import prepare_recipe, validate_snapshot
from measure_api.v2.recipe_models import (
    Recipe,
    RecipeRequest,
    RecipeValidationResult,
    RunResult,
)
from measure_api.v2.registry import CalculationContext


def canonical(value: Any) -> bytes:
    """Encode unambiguous finite UTF-8 JSON for manifests and identities.

    Args:
        value: Supplied value to normalize under the public schema and coordinate
            convention.

    Returns:
        Finite canonical UTF-8 JSON bytes with stable field ordering.
    """
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def rewrite_references(value: Any, replacements: Dict[str, dict]) -> Any:
    """Replace declared references recursively; ordinary strings remain ordinary strings.

    Args:
        value: Supplied value to normalize under the public schema and coordinate
            convention.
        replacements: Pinned reference identities mapped to their relocated portable
            references.

    Returns:
        Equivalent nested value with only explicitly declared references relocated.
    """
    if isinstance(value, dict):
        if "path" in value and set(value) <= set(FileReference.model_fields):
            reference = FileReference.model_validate(value)
            key = reference_key(reference)
            if key in replacements:
                replacement = dict(replacements[key])
                if reference.provenance is not None:
                    replacement["provenance"] = reference.provenance
                return replacement
        return {key: rewrite_references(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [rewrite_references(item, replacements) for item in value]
    return value


def reference_key(reference: FileReference) -> str:
    """Identify an already pinned pathname and content without incidental provenance.

    Args:
        reference: Shared pathname and optional expected content digest to verify.

    Returns:
        Canonical root, path and content-digest key without incidental provenance.
    """
    return canonical([reference.root, reference.path, reference.sha256]).decode()


class PackageAsset(StrictModel):
    id: str = Field(pattern=r"^asset-[0-9]+$")
    name: str
    kind: str
    format: str
    schema_version: str = "1.0"
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    size_bytes: int = Field(ge=0)
    source: FileReference


class PackageManifest(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    format: Literal["measure-recipe-package"] = "measure-recipe-package"
    packaging: Literal["embedded", "references"]
    recipe: Recipe
    source_configuration_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    assets: List[PackageAsset] = Field(default_factory=list, max_length=10000)


class RecipeExportRequest(RecipeRequest):
    output: OutputSpec
    packaging: Literal["embedded", "references"] = "embedded"
    frozen_run: Optional[RunResult] = None
    frozen_run_file: Optional[FileReference] = None


class RecipeImportRequest(CalculationRequest):
    package: FileReference
    output: OutputSpec
    root_bindings: Dict[str, str] = Field(default_factory=dict)
    asset_bindings: Dict[str, FileReference] = Field(default_factory=dict)
    resolved_assets: Dict[str, FileReference] = Field(default_factory=dict)


class PortableRecipeResult(RecipeValidationResult):
    source_package: Optional[FileReference] = None
    source_configuration_sha256: Optional[str] = None
    artifacts: List[FileDescriptor] = Field(default_factory=list)
    manifest: Optional[FileDescriptor] = None


def require_valid(request: RecipeRequest) -> RecipeValidationResult:
    """Reject semantically invalid packages before publishing any completed output.

    Args:
        request: Typed inline or file-backed upstream recipe snapshot for complete
            validation.

    Returns:
        Full recipe validation result; invalid configuration raises a service error.
    """
    result = validate_snapshot(request)
    if not result.valid:
        raise ServiceError(
            "recipe.invalid", "Recipe validation failed", details={"issues": result.issues}
        )
    return result


def bounded_zip(data: bytes, files: SharedFiles) -> Dict[str, bytes]:
    """Read allowed members in memory without extraction, symlinks or decompression bombs.

    Args:
        data: ZIP bytes checked for traversal, duplicate members and expansion limits.
        files: Configured shared-directory boundary with size limits and digest
            verification.

    Returns:
        Allowed archive member names mapped to verified, bounded expanded bytes.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            entries = archive.infolist()
            names = [item.filename for item in entries]
            if len(entries) > 10001 or len(set(names)) != len(names):
                raise ValueError("Duplicate or excessive archive entries")
            total = sum(item.file_size for item in entries)
            if total > min(
                files.settings.max_input_bytes, files.settings.max_worker_memory_bytes // 4
            ):
                raise ValueError("Expanded archive exceeds limits")
            for item in entries:
                path = PurePosixPath(item.filename)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or "\\" in item.filename
                    or ":" in item.filename
                    or item.is_dir()
                    or stat.S_ISLNK(item.external_attr >> 16)
                    or item.flag_bits & 1
                ):
                    raise ValueError("Unsafe archive member")
            return {item.filename: archive.read(item) for item in entries}
    except (ValueError, OSError, RuntimeError, zipfile.BadZipFile) as error:
        raise ServiceError("recipe_package.invalid", str(error)) from error


def read_package(reference: FileReference, files: SharedFiles) -> Tuple[
    PackageManifest,
    Dict[str, bytes],
    FileReference,
]:
    """Validate the complete package and all declared immutable payload identities.

    Args:
        reference: Upstream package manifest or ZIP reference with expected content digest.
        files: Configured shared-directory boundary with size limits and digest
            verification.

    Returns:
        Validated package manifest, verified payload bytes and pinned package reference.
    """
    data, verified = files.read_bytes(reference)
    members = bounded_zip(data, files) if data.startswith(b"PK") else {}
    try:
        manifest = PackageManifest.model_validate_json(
            members["manifest.json"] if members else data
        )
        identities = [asset.id for asset in manifest.assets]
        names = [asset.name for asset in manifest.assets]
        if len(set(identities)) != len(identities) or len(set(names)) != len(names):
            raise ValueError("Duplicate package asset identity")
        for asset in manifest.assets:
            if Path(asset.name).name != asset.name or asset.name in {
                ".",
                "..",
                "manifest.json",
            }:
                raise ValueError("Invalid package asset name")
            if asset.source.sha256 != asset.sha256:
                raise ValueError("Source and declared asset digests disagree")
        expected = {"manifest.json"} | {"assets/" + asset.name for asset in manifest.assets}
        if manifest.packaging == "embedded":
            if set(members) != expected:
                raise ValueError("Embedded package has missing or undeclared members")
            for asset in manifest.assets:
                payload = members["assets/" + asset.name]
                if (
                    len(payload) != asset.size_bytes
                    or hashlib.sha256(payload).hexdigest() != asset.sha256
                ):
                    raise ValueError("Embedded asset checksum mismatch")
                validate_payload(payload, asset.format, files)
        elif members:
            raise ValueError("Reference package must be standalone JSON")
        declared = {asset.id: asset for asset in manifest.assets}

        def inspect(value: Any) -> None:
            if isinstance(value, dict):
                if "path" in value and set(value) <= set(FileReference.model_fields):
                    ref = FileReference.model_validate(value)
                    if (
                        ref.root != "package"
                        or ref.path not in declared
                        or ref.sha256 != declared[ref.path].sha256
                    ):
                        raise ValueError("Recipe references an undeclared package asset")
                else:
                    for item in value.values():
                        inspect(item)
            elif isinstance(value, list):
                for item in value:
                    inspect(item)

        inspect(manifest.recipe.model_dump(mode="json"))
        return manifest, members, verified
    except (ValueError, KeyError, TypeError) as error:
        raise ServiceError("recipe_package.invalid", str(error)) from error


def validate_payload(data: bytes, format_name: str, files: SharedFiles) -> None:
    """Allow declared numeric/image/JSON data, never object arrays or arbitrary binaries.

    Args:
        data: Source payload to validate or transform without mutating the original.
        format_name: Declared artifact format to validate against the payload bytes.
        files: Configured shared-directory boundary with size limits and digest
            verification.
    """
    try:
        if format_name == "npz":
            for value in safe_numeric_archive(data, files).values():
                if value.dtype.kind in "fc" and not np.isfinite(value).all():
                    raise ValueError("Non-finite numeric payload")
        elif format_name == "json":
            json.loads(
                data,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError("Non-finite JSON constant")
                ),
            )
        elif format_name in {"png", "jpg", "jpeg"}:
            width, height = encoded_dimensions(data)
            if (
                width * height > files.settings.max_image_pixels
                or width * height * 4 + len(data) > files.settings.max_worker_memory_bytes
            ):
                raise ValueError("Image payload exceeds configured limits")
            pixels = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
            if (
                pixels is None
                or pixels.dtype != np.uint8
                or (pixels.ndim != 2 and (pixels.ndim != 3 or pixels.shape[2] != 3))
            ):
                raise ValueError("Expected supported GRAY8 or BGR8 image payload")
        else:
            raise ValueError("Unsupported payload format")
    except ServiceError as error:
        raise ServiceError("recipe_package.unsafe_payload", error.message) from error
    except (ValueError, TypeError, OSError, zipfile.BadZipFile) as error:
        raise ServiceError("recipe_package.unsafe_payload", str(error)) from error


def safe_numeric_archive(data: bytes, files: SharedFiles) -> Dict[str, np.ndarray]:
    """Validate NPY storage headers before any NumPy allocation or object deserialization.

    Args:
        data: NPZ bytes checked for bounded NPY shapes and nonobject dtypes.
        files: Configured shared-directory boundary with size limits and digest
            verification.

    Returns:
        Numeric arrays keyed by member name, without pickle or object loading.
    """
    members = bounded_zip(data, files)
    storage = 0
    limit = min(files.settings.max_input_bytes, files.settings.max_worker_memory_bytes // 4)
    for name, payload in members.items():
        if not name.endswith(".npy"):
            raise ValueError("Numeric archives contain only NPY members")
        buffer = io.BytesIO(payload)
        version = np.lib.format.read_magic(buffer)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(buffer)
        elif version == (2, 0):
            shape, _, dtype = np.lib.format.read_array_header_2_0(buffer)
        else:
            raise ValueError("Unsupported NPY header version")
        if dtype.hasobject:
            raise ValueError("Object arrays require pickle")
        if any(dimension < 0 for dimension in shape):
            raise ValueError("Invalid numeric shape")
        size = math.prod(shape) * dtype.itemsize
        storage += size
        if storage > limit:
            raise ValueError("Declared numeric storage exceeds configured limit")
        if size != len(payload) - buffer.tell():
            raise ValueError("Declared numeric storage disagrees with payload")
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def export_recipe(
    request: RecipeExportRequest, context: CalculationContext
) -> PortableRecipeResult:
    """Package all verified transitive inputs without registering or owning the recipe.

    Args:
        request: Typed recipe, portable-package mode and optional frozen observed run.
        context: Isolated operation identity, resource limits and verified shared-file
            access.

    Returns:
        Validated original recipe and configuration identity, with committed package
        artifact descriptors and their manifest. The returned recipe is not relocated.
    """
    validated = require_valid(request)
    replacements, assets, payloads = {}, [], {}
    seen = set()
    for reference in request.assets:
        key = reference_key(reference)
        if key in seen:
            continue
        seen.add(key)
        data, verified = context.files.read_bytes(reference)
        identity = "asset-" + str(len(assets))
        suffix = Path(verified.path).suffix.lower()
        suffix = suffix if suffix in {".png", ".jpg", ".jpeg", ".npz", ".json"} else ".bin"
        name = identity + suffix
        assets.append(
            PackageAsset(
                id=identity,
                name=name,
                kind="recipe_asset",
                format=suffix[1:],
                sha256=verified.sha256,
                size_bytes=len(data),
                source=verified,
            )
        )
        replacements[key] = FileReference(
            root="package", path=identity, sha256=verified.sha256
        ).model_dump(mode="json")
        payloads["assets/" + name] = data
    manifest = PackageManifest(
        packaging=request.packaging,
        recipe=Recipe.model_validate(
            rewrite_references(request.recipe.model_dump(mode="json"), replacements)
        ),
        source_configuration_sha256=validated.configuration_sha256,
        assets=assets,
    )
    content = canonical(manifest.model_dump(mode="json"))
    if request.packaging == "embedded":
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("manifest.json", content)
            for name, data in payloads.items():
                archive.writestr(name, data)
        content, format_name, name = buffer.getvalue(), "zip", "recipe.zip"
    else:
        format_name, name = "json", "recipe-package.json"
    if len(content) > context.settings.max_input_bytes:
        raise ServiceError("recipe_package.too_large", "Package exceeds configured file limit")
    final_payloads = [(name, "recipe_package", format_name, content)]
    if request.frozen_run is not None:
        from measure_api.v2.replay import freeze_run

        if request.frozen_run.configuration_sha256 != validated.configuration_sha256:
            raise ServiceError(
                "replay.configuration_mismatch", "Export recipe differs from supplied run", 409
            )
        frozen = freeze_run(request.frozen_run)
        final_payloads.append(
            ("frozen.json", "frozen_run", "json", frozen.model_dump_json().encode())
        )
    artifacts, completion = ArtifactWriter(
        context.files, request.output, context.operation_id
    ).commit(final_payloads)
    return PortableRecipeResult(
        **validated.model_dump(),
        artifacts=artifacts,
        manifest=completion,
        source_configuration_sha256=validated.configuration_sha256,
    )


def prepare_export(request: RecipeExportRequest, files: SharedFiles) -> RecipeExportRequest:
    """Load supplied run JSON and retain its producer and replay input identities.

    Args:
        request: Typed recipe, portable-package mode and optional frozen observed run.
        files: Configured shared-directory boundary with size limits and digest
            verification.

    Returns:
        Recipe, portable-package mode and optional frozen observed run.
    """
    if request.frozen_run is not None and request.frozen_run_file is not None:
        raise ServiceError("request.validation", "Supply at most one frozen run form")
    run = request.frozen_run
    if request.frozen_run_file is not None:
        data, _ = files.read_bytes(request.frozen_run_file)
        try:
            run = RunResult.model_validate_json(data)
        except ValueError as error:
            raise ServiceError("replay.invalid_run", "Supplied run file is invalid") from error
    if run is not None:
        from measure_api.v2.replay import freeze_run

        frozen = freeze_run(run)
        prepared_run = prepare_recipe(RecipeRequest(recipe=frozen.recipe), files)
        validated_run = require_valid(prepared_run)
        if validated_run.configuration_sha256 != frozen.configuration_sha256:
            raise ServiceError(
                "replay.configuration_mismatch",
                "Supplied run configuration disagrees with its hash",
                409,
            )
        for reference in [frozen.image] + frozen.sources + frozen.assets:
            files.read_bytes(reference)
        run = run.model_copy(
            update={"artifacts": [], "manifest": None, "measurement_image": None}
        )
    return prepare_recipe(request.model_copy(update={"frozen_run": run}), files)


def prepare_import(request: RecipeImportRequest, files: SharedFiles) -> RecipeImportRequest:
    """Resolve external aliases and pin linked assets before admission and retry identity.

    Args:
        request: Typed portable package, external root bindings and upstream output
            destination.
        files: Configured shared-directory boundary with size limits and digest
            verification.

    Returns:
        Portable package, external root bindings and upstream output destination.
    """
    manifest, _, verified = read_package(request.package, files)
    resolved = {}
    if manifest.packaging == "references":
        for asset in manifest.assets:
            ref = request.asset_bindings.get(asset.id, asset.source)
            if ref.root is None and asset.id not in request.asset_bindings:
                raise ServiceError(
                    "recipe_package.binding_required",
                    "Absolute source paths require explicit asset_bindings",
                )
            if ref.root in request.root_bindings:
                ref = ref.model_copy(update={"root": request.root_bindings[ref.root]})
            if ref.sha256 is not None and ref.sha256 != asset.sha256:
                raise ServiceError(
                    "recipe_package.binding_digest", "Binding changes asset identity", 409
                )
            data, ref = files.read_bytes(ref.model_copy(update={"sha256": asset.sha256}))
            if len(data) != asset.size_bytes:
                raise ServiceError(
                    "recipe_package.asset_size", "Linked asset size changed", 409
                )
            validate_payload(data, asset.format, files)
            resolved[asset.id] = ref
    return request.model_copy(update={"package": verified, "resolved_assets": resolved})


def output_reference(
    output: OutputSpec, name: str, data: bytes, context: CalculationContext
) -> FileReference:
    """Predict an isolated final descriptor for references inside a jointly committed recipe.

    Args:
        output: Upstream output directory and isolated namespace definition.
        name: Artifact filename beneath this operation's final output namespace.
        data: Immutable output payload bytes used to compute the reference digest.
        context: Isolated operation identity, resource limits and verified shared-file
            access.

    Returns:
        Final namespace reference and SHA-256 digest for the pending output bytes.
    """
    path = context.files.resolve(output.directory, write=True) / context.operation_id / name
    alias = output.directory.root
    if alias is not None:
        path = path.relative_to(context.settings.shared_roots[alias])
    return FileReference(root=alias, path=str(path), sha256=hashlib.sha256(data).hexdigest())


def import_recipe(
    request: RecipeImportRequest, context: CalculationContext
) -> PortableRecipeResult:
    """Validate in a private stage, relocate immutable assets, then commit one final manifest.

    Args:
        request: Typed portable package, external root bindings and upstream output
            destination.
        context: Isolated operation identity, resource limits and verified shared-file
            access.

    Returns:
        Validated relocated recipe and portable manifest with asset compatibility
        diagnostics.
    """
    manifest, members, verified = read_package(request.package, context.files)
    replacements, final_replacements, payloads = {}, {}, []
    stage_context = context
    final_assets = list(request.resolved_assets.values())
    if manifest.packaging == "embedded":
        stage_context, stage_output = context.stage("package-import")
        payloads = [
            (asset.name, asset.kind, asset.format, members["assets/" + asset.name])
            for asset in manifest.assets
        ]
        stage_assets, _ = ArtifactWriter(
            stage_context.files, stage_output, stage_context.operation_id
        ).commit(payloads)
        final_assets = []
        for asset, descriptor, payload in zip(manifest.assets, stage_assets, payloads):
            logical = FileReference(root="package", path=asset.id, sha256=asset.sha256)
            replacements[reference_key(logical)] = descriptor.file.model_dump(mode="json")
            final = output_reference(request.output, asset.name, payload[3], context)
            final = final.model_copy(update={"provenance": asset.source.provenance})
            final_replacements[reference_key(descriptor.file)] = final.model_dump(mode="json")
            final_assets.append(final)
    else:
        for asset in manifest.assets:
            logical = FileReference(root="package", path=asset.id, sha256=asset.sha256)
            replacements[reference_key(logical)] = request.resolved_assets[asset.id].model_dump(
                mode="json"
            )
    recipe = Recipe.model_validate(
        rewrite_references(manifest.recipe.model_dump(mode="json"), replacements)
    )
    prepared = prepare_recipe(RecipeRequest(recipe=recipe), stage_context.files)
    require_valid(prepared)
    relocated = Recipe.model_validate(
        rewrite_references(prepared.recipe.model_dump(mode="json"), final_replacements)
    )
    final_request = RecipeRequest(recipe=relocated, assets=final_assets)
    validated = require_valid(final_request)
    payloads.append(
        ("recipe.json", "recipe", "json", canonical(relocated.model_dump(mode="json")))
    )
    artifacts, completion = ArtifactWriter(
        context.files, request.output, context.operation_id
    ).commit(payloads)
    return PortableRecipeResult(
        **validated.model_dump(),
        source_package=verified,
        source_configuration_sha256=manifest.source_configuration_sha256,
        artifacts=artifacts,
        manifest=completion,
    )
