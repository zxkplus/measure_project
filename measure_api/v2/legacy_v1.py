"""One-way migration of saved V1 API projects into executable V2 recipes."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import cv2
import numpy as np
from pydantic import Field, model_validator

from measure_api.v2.calibration_assets import CalibrationApplicability
from measure_api.v2.errors import ServiceError
from measure_api.v2.files import SharedFiles
from measure_api.v2.legacy_conversion import (
    COMPOSITIONS,
    PARAMETERS,
    PRIMITIVES,
    CompatibilityDiagnostic,
    LegacyConversionResult,
    convert_normalized,
    diagnostic,
    legacy_preprocessing,
)
from measure_api.v2.models import (
    CalculationRequest,
    FileReference,
    OutputSpec,
    StrictModel,
)
from measure_api.v2.registry import CalculationContext


CONFIG_KEYS = {"version", "project_name", "phase", "measurement_defs", "composed_defs"}
MANIFEST_CONTRACTS = {"3.0", "3.1"}
REQUIRED_PARAMETERS = {
    "EdgePoint": {"row", "col", "angle", "length1", "length2"},
    "EdgePair": {"row", "col", "angle", "length1", "length2"},
    "FitLine": {"start", "end", "measure_length1", "measure_length2"},
    "FitCircle": {"center", "radius", "measure_length1", "measure_length2"},
    "TemplateMatchPoint": {"row", "col"},
}
LOSSLESS_IMAGE_SUFFIXES = {".png", ".bmp", ".tif", ".tiff"}


class LegacyV1ConversionRequest(CalculationRequest):
    """Explicit V1 project files, upstream identities and output destination."""

    config: FileReference
    model_manifest: Optional[FileReference] = None
    reference: Optional[FileReference] = None
    reference_npy: Optional[FileReference] = None
    recipe_id: str = Field(min_length=1)
    recipe_version: str = Field(min_length=1)
    applicability: Optional[CalibrationApplicability] = None
    square_size_mm: Optional[float] = Field(default=None, gt=0)
    advisory_policy: Literal["accept", "reject"] = "accept"
    output: OutputSpec
    normalized: Dict[str, Any] = Field(default_factory=dict)
    preparation_diagnostics: List[CompatibilityDiagnostic] = Field(default_factory=list)

    @model_validator(mode="after")
    def one_reference(self) -> "LegacyV1ConversionRequest":
        """Require at least one explicit reference image form.

        Returns:
            This request after checking that reference pixels are supplied.
        """
        if self.reference is None and self.reference_npy is None:
            raise ValueError("Provide reference or reference_npy")
        return self


class MigrationSummary(StrictModel):
    """Counts and explicit approximations for one migrated project."""

    nodes: int = Field(ge=0)
    templates: int = Field(ge=0)
    dropped_physical_units: bool = False
    approximations: List[str] = Field(default_factory=list)


class LegacyV1ConversionResult(LegacyConversionResult):
    """Converted recipe plus V1 source provenance and migration summary."""

    migrated_from: Literal["v1-api-project"] = "v1-api-project"
    source_config: FileReference
    source_model: Optional[FileReference] = None
    summary: MigrationSummary


def _read_json(
    reference: FileReference, files: SharedFiles, path: str
) -> Tuple[Dict[str, Any], FileReference, List[CompatibilityDiagnostic]]:
    """Read a bounded JSON object from a shared file reference.

    Args:
        reference: Verified shared-file reference to the JSON document.
        files: Configured shared-directory boundary with size limits and digest
            verification.
        path: Diagnostic field path used when parsing fails.

    Returns:
        Parsed object and located parse diagnostics; both are empty on failure except
        for the diagnostic list.
    """
    data, verified = files.read_bytes(reference)
    try:
        value = json.loads(data)
    except (ValueError, TypeError) as error:
        return {}, verified, [diagnostic(path, "v1.invalid_json", str(error))]
    if not isinstance(value, dict):
        return {}, verified, [diagnostic(path, "v1.invalid_json", "Expected a JSON object")]
    return value, verified, []


def _decode_reference_npy(data: bytes, max_pixels: int) -> np.ndarray:
    """Decode a bounded, non-object grayscale NPY reference array.

    Args:
        data: Raw NPY bytes read from the shared root.
        max_pixels: Deployment pixel-count ceiling.

    Returns:
        Two-dimensional uint8 reference pixels.
    """
    try:
        loaded = np.load(io.BytesIO(data), allow_pickle=False)
    except (ValueError, OSError, TypeError, EOFError) as error:
        raise ValueError("reference.npy is not a readable numeric array") from error
    if isinstance(loaded, np.lib.npyio.NpzFile):
        raise ValueError("reference.npy must be a single array, not an NPZ archive")
    pixels = np.asarray(loaded)
    if pixels.ndim != 2:
        raise ValueError("reference.npy must be a two-dimensional grayscale array")
    if pixels.dtype != np.uint8:
        raise ValueError("reference.npy must use uint8 grayscale samples")
    if pixels.size > max_pixels:
        raise ValueError("reference.npy exceeds the configured pixel limit")
    return pixels


def _finite_pair(value: Any) -> Optional[Tuple[float, float]]:
    """Return two finite floats, or ``None`` when the value is malformed.

    Args:
        value: Candidate list or tuple.

    Returns:
        Two finite floats, or ``None`` for any non-numeric/malformed pair.
    """
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        return None
    if not all(np.isfinite(item) for item in value):
        return None
    return float(value[0]), float(value[1])


def _validate_config(
    value: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[CompatibilityDiagnostic]]:
    """Validate a V1 ``config.json`` and collect normalized tool entries.

    Args:
        value: Parsed V1 project configuration.

    Returns:
        Normalized tools in declaration order and located diagnostics.
    """
    diagnostics: List[CompatibilityDiagnostic] = []
    version = value.get("version")
    if version not in (1, 2):
        diagnostics.append(
            diagnostic(
                "version", "v1.unsupported_version",
                "Only V1 config version 1/2 is supported",
            )
        )
    elif version == 1:
        diagnostics.append(
            diagnostic(
                "version", "v1.config_version_legacy",
                "V1 config version 1 is accepted with advisory diagnostics", "warning",
            )
        )
    for key in sorted(set(value) - CONFIG_KEYS):
        diagnostics.append(
            diagnostic(key, "v1.unknown_field", "Unknown V1 project configuration field")
        )
    for key in ("phase", "project_name"):
        if key in value:
            diagnostics.append(
                diagnostic(key, "v1." + key + "_dropped",
                           "V1 " + key + " is not part of a V2 recipe", "info")
            )

    measurement_defs = value.get("measurement_defs", [])
    composed_defs = value.get("composed_defs", [])
    if not isinstance(measurement_defs, list) or len(measurement_defs) > 1000:
        diagnostics.append(
            diagnostic("measurement_defs", "v1.invalid_config",
                       "measurement_defs must be a bounded array")
        )
        measurement_defs = []
    if not isinstance(composed_defs, list) or len(composed_defs) > 1000:
        diagnostics.append(
            diagnostic("composed_defs", "v1.invalid_config",
                       "composed_defs must be a bounded array")
        )
        composed_defs = []

    tools: List[Dict[str, Any]] = []
    labels: Dict[str, int] = {}

    def duplicate(label: str, path: str) -> None:
        if label in labels:
            diagnostics.append(
                diagnostic(path, "v1.duplicate_label", "Label '" + label + "' is already used")
            )
        else:
            labels[label] = 1

    for index, item in enumerate(measurement_defs):
        path = "measurement_defs." + str(index)
        if not isinstance(item, dict):
            diagnostics.append(diagnostic(path, "v1.invalid_config", "Expected a tool object"))
            continue
        kind, label, params = item.get("object_type"), item.get("label"), item.get("params", {})
        if not isinstance(label, str) or not label:
            diagnostics.append(diagnostic(path + ".label", "v1.invalid_config",
                                          "Nonempty string label is required"))
            continue
        duplicate(label, path + ".label")
        if not isinstance(params, dict):
            diagnostics.append(diagnostic(path + ".params", "v1.invalid_config",
                                          "Expected an object of parameters"))
            continue
        if kind not in PRIMITIVES:
            diagnostics.append(diagnostic(path + ".object_type", "v1.unsupported_tool", str(kind)))
            continue
        missing = REQUIRED_PARAMETERS[kind] - set(params)
        if missing:
            diagnostics.append(
                diagnostic(path + ".params", "v1.unsupported_parameter",
                           "Missing required parameters: " + ", ".join(sorted(missing)))
            )
        extra = set(params) - PARAMETERS[kind]
        if extra:
            diagnostics.append(
                diagnostic(path + ".params", "v1.unsupported_parameter",
                           ", ".join(sorted(extra)))
            )
        tools.append({"object_type": kind, "label": label, "params": dict(params)})

    for index, item in enumerate(composed_defs):
        path = "composed_defs." + str(index)
        if not isinstance(item, dict):
            diagnostics.append(diagnostic(path, "v1.invalid_config", "Expected a tool object"))
            continue
        kind, label, deps = item.get("composed_type"), item.get("label"), item.get("deps", {})
        if not isinstance(label, str) or not label:
            diagnostics.append(diagnostic(path + ".label", "v1.invalid_config",
                                          "Nonempty string label is required"))
            continue
        duplicate(label, path + ".label")
        if kind not in COMPOSITIONS:
            diagnostics.append(
                diagnostic(path + ".composed_type", "v1.unsupported_tool", str(kind))
            )
            continue
        expected = set(COMPOSITIONS[kind][1].values())
        if not isinstance(deps, dict):
            diagnostics.append(diagnostic(path + ".deps", "v1.invalid_config",
                                          "Expected a dependency object"))
            continue
        if set(deps) != expected:
            diagnostics.append(
                diagnostic(path + ".deps", "v1.unresolved_dependency",
                           "Expected keys " + ", ".join(sorted(expected)))
            )
        if any(not isinstance(value, str) or not value for value in deps.values()):
            diagnostics.append(
                diagnostic(path + ".deps", "v1.unresolved_dependency",
                           "Dependencies must be nonempty label strings")
            )
        tools.append({"object_type": kind, "label": label, "params": dict(deps)})

    providers = set(labels)
    providers.update(
        tool["label"] + suffix
        for tool in tools
        if tool["object_type"] == "EdgePair"
        for suffix in ("_A", "_B")
    )
    for index, tool in enumerate(tools):
        if tool["object_type"] not in COMPOSITIONS:
            continue
        for name, value in tool["params"].items():
            if isinstance(value, str) and value not in providers:
                diagnostics.append(
                    diagnostic(
                        "composed_defs." + str(index) + ".deps." + name,
                        "v1.unresolved_dependency",
                        "Dependency '" + value + "' does not name a declared tool or endpoint",
                    )
                )
    if not tools:
        diagnostics.append(
            diagnostic("template_view.tools", "v1.empty_project",
                       "A V2 recipe requires at least one measurement node")
        )
    return tools, diagnostics


def _validate_manifest(
    value: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], List[CompatibilityDiagnostic]]:
    """Validate a V1 manifest model asset and extract its teaching geometry.

    Args:
        value: Parsed ``manifest.json`` object.

    Returns:
        Alignment block and located diagnostics; the block is ``None`` when the
        manifest cannot define a usable V1 teaching frame.
    """
    diagnostics: List[CompatibilityDiagnostic] = []
    contract = value.get("contract_version")
    if contract not in MANIFEST_CONTRACTS:
        diagnostics.append(
            diagnostic("contract_version", "v1.invalid_model_manifest",
                       "Unsupported manifest contract version " + str(contract))
        )
    elif contract == "3.0":
        diagnostics.append(
            diagnostic("contract_version", "v1.manifest_contract_legacy",
                       "Manifest contract 3.0 is accepted with advisory diagnostics", "warning")
        )
    alignment = value.get("alignment")
    if not isinstance(alignment, dict):
        diagnostics.append(
            diagnostic("alignment", "v1.invalid_model_manifest", "Missing alignment object")
        )
        return None, diagnostics
    mode = alignment.get("mode")
    if mode != "single_box":
        diagnostics.append(
            diagnostic(
                "alignment.mode", "v1.unsupported_alignment",
                "Only single_box alignment has a safe V2 equivalent; got " + str(mode),
            )
        )
        return None, diagnostics
    center = _finite_pair(alignment.get("box_center"))
    size = _finite_pair(alignment.get("box_size"))
    angle = alignment.get("box_angle_deg")
    if center is None:
        diagnostics.append(
            diagnostic("alignment.box_center", "v1.invalid_model_manifest",
                       "Expected two finite row/column values")
        )
    if size is None or size[0] <= 0 or size[1] <= 0:
        diagnostics.append(
            diagnostic("alignment.box_size", "v1.invalid_model_manifest",
                       "Expected two positive height/width values")
        )
    if isinstance(angle, bool) or not isinstance(angle, (int, float)) or not np.isfinite(angle):
        diagnostics.append(
            diagnostic("alignment.box_angle_deg", "v1.invalid_model_manifest",
                       "Expected a finite angle")
        )
    matching = value.get("matching")
    if not isinstance(matching, dict) or not matching:
        diagnostics.append(
            diagnostic("matching", "v1.invalid_model_manifest",
                       "single_box migration requires recorded matching parameters")
        )
    return alignment, diagnostics


def _normalize_matching(
    manifest: Dict[str, Any],
    alignment: Dict[str, Any],
    diagnostics: List[CompatibilityDiagnostic],
) -> Optional[Dict[str, Any]]:
    """Translate V1 manifest matching parameters into the shared legacy shape.

    Args:
        manifest: Parsed V1 manifest carrying top-level matching parameters.
        alignment: Validated V1 alignment block carrying preprocessing.
        diagnostics: Diagnostic list extended in place.

    Returns:
        Normalized matching block, or ``None`` when required values are malformed.
    """
    raw = manifest.get("matching")
    if not isinstance(raw, dict) or not raw:
        diagnostics.append(
            diagnostic("matching", "v1.invalid_model_manifest", "Missing matching parameters")
        )
        return None
    angle_range = _finite_pair(raw.get("angle_range", [-30, 30]))
    if angle_range is None:
        diagnostics.append(
            diagnostic("matching.angle_range", "v1.invalid_model_manifest",
                       "Expected two finite angle bounds")
        )
        return None
    low, high = angle_range
    half = max(abs(low), abs(high))
    if abs(low) != abs(high):
        diagnostics.append(
            diagnostic("matching.angle_range", "v1.asymmetric_angle_range",
                       "V1 expected a symmetric range; using the larger half-range", "warning")
        )

    def number(key: str, default: float) -> float:
        value = raw.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            diagnostics.append(
                diagnostic("matching." + key, "v1.invalid_model_manifest",
                           "Expected a finite number")
            )
            return default
        return float(value)

    score = number("match_score_threshold", 0.5)
    step = number("angle_step", 1.0)
    overlap = number("overlap", 0.3)
    max_matches = raw.get("max_matches", 0)
    if isinstance(max_matches, bool) or not isinstance(max_matches, int) or max_matches < 0:
        diagnostics.append(
            diagnostic("matching.max_matches", "v1.invalid_model_manifest",
                       "Expected a nonnegative integer")
        )
        max_matches = 0
    preprocessor = alignment.get("preprocessor") or {"type": "raw"}
    try:
        preprocessing = legacy_preprocessing(preprocessor)
    except (KeyError, TypeError, ValueError) as error:
        diagnostics.append(
            diagnostic("alignment.preprocessor", "v1.invalid_preprocessing", str(error))
        )
        return None

    matching: Dict[str, Any] = {
        "preprocessing": preprocessing,
        "match_score_threshold": score,
        "angle_range_deg": half,
        "angle_step_deg": step,
        "max_matches": max_matches,
        "overlap": overlap,
    }
    for key, name in (
        ("coarse_fine", "coarse_fine"),
        ("coarse_angle_step", "coarse_angle_step_deg"),
        ("pyramid_max_template_size", "pyramid_max_template_size"),
        ("pyramid_decimate", "pyramid_decimate"),
        ("scale_min", "scale_min"),
        ("scale_max", "scale_max"),
        ("scale_step", "scale_step"),
    ):
        if key in raw:
            matching[name] = raw[key]
    return matching


def _reference_inputs(
    request: LegacyV1ConversionRequest,
    manifest: Optional[Dict[str, Any]],
    files: SharedFiles,
) -> Tuple[
    Optional[Tuple[int, int]],
    Optional[FileReference],
    Optional[FileReference],
    List[CompatibilityDiagnostic],
]:
    """Read and cross-check explicit reference image forms.

    Args:
        request: Parsed migration request with one or both reference references.
        manifest: Parsed V1 model manifest, when supplied.
        files: Configured shared-directory boundary.

    Returns:
        Decoded ``(height, width)``, verified image reference, verified NPY reference and
        located diagnostics.
    """
    diagnostics: List[CompatibilityDiagnostic] = []
    image_shape: Optional[Tuple[int, int]] = None
    image_pixels: Optional[np.ndarray] = None
    image_verified: Optional[FileReference] = None
    if request.reference is not None:
        image, image_verified = files.read_image(request.reference)
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_pixels = image
        image_shape = image.shape[:2]
    npy_pixels: Optional[np.ndarray] = None
    npy_verified: Optional[FileReference] = None
    if request.reference_npy is not None:
        data, npy_verified = files.read_bytes(request.reference_npy)
        try:
            npy_pixels = _decode_reference_npy(data, files.settings.max_image_pixels)
        except ValueError as error:
            diagnostics.append(
                diagnostic("reference_npy", "v1.reference_invalid", str(error))
            )
        if image_shape is not None and npy_pixels is not None:
            if image_shape != npy_pixels.shape:
                diagnostics.append(
                    diagnostic("reference_npy", "v1.reference_shape_mismatch",
                               "reference image and reference.npy shapes differ")
                )
            elif (
                request.reference is not None
                and Path(request.reference.path).suffix.lower() in LOSSLESS_IMAGE_SUFFIXES
                and not np.array_equal(image_pixels, npy_pixels)
            ):
                diagnostics.append(
                    diagnostic("reference_npy", "v1.reference_pixel_mismatch",
                               "reference image and reference.npy pixels differ")
                )
    declared_shape = None
    if manifest is not None:
        reference_image = manifest.get("reference_image")
        if isinstance(reference_image, dict):
            declared = reference_image.get("shape")
            if isinstance(declared, (list, tuple)) and len(declared) == 2:
                try:
                    declared_shape = (int(declared[0]), int(declared[1]))
                except (TypeError, ValueError):
                    diagnostics.append(
                        diagnostic(
                            "reference_image.shape", "v1.invalid_model_manifest",
                            "Declared reference shape must contain two integers",
                        )
                    )
    actual_shape = image_shape or (npy_pixels.shape if npy_pixels is not None else None)
    if declared_shape is not None and actual_shape is not None and declared_shape != actual_shape:
        diagnostics.append(
            diagnostic("reference_image.shape", "v1.reference_shape_mismatch",
                       "Reference dimensions do not match the manifest declaration")
        )
    if manifest is not None and npy_verified is not None:
        declared_digest = (manifest.get("files") or {}).get("reference.npy")
        if (
            isinstance(declared_digest, str)
            and npy_verified.sha256.lower() != declared_digest.lower()
        ):
            diagnostics.append(
                diagnostic("reference_npy", "v1.reference_digest_mismatch",
                           "reference.npy digest does not match the manifest")
            )
    return actual_shape, image_verified, npy_verified, diagnostics


def _normalize(
    request: LegacyV1ConversionRequest,
    config: Dict[str, Any],
    manifest: Optional[Dict[str, Any]],
    diagnostics: List[CompatibilityDiagnostic],
) -> Dict[str, Any]:
    """Build the shared legacy structure consumed by ``convert_normalized``.

    Args:
        request: Parsed migration request.
        config: Validated V1 configuration.
        manifest: Validated V1 model manifest, when present.
        diagnostics: Diagnostic list extended in place.

    Returns:
        Normalized legacy structure with tools, ROI, matching and optional calibration.
    """
    tools, config_diagnostics = _validate_config(config)
    diagnostics.extend(config_diagnostics)
    if manifest is None:
        if tools:
            diagnostics.append(
                diagnostic("model_manifest", "v1.missing_model_manifest",
                           "A V1 model manifest is required to reconstruct the taught frame")
            )
        return {}
    alignment, manifest_diagnostics = _validate_manifest(manifest)
    diagnostics.extend(manifest_diagnostics)
    if alignment is None:
        return {}
    matching = _normalize_matching(manifest, alignment, diagnostics)
    if matching is None:
        return {}
    center = _finite_pair(alignment.get("box_center"))
    size = _finite_pair(alignment.get("box_size"))
    # Manifest angles are visually counter-clockwise; V2 ROI/search angles are
    # clockwise.  Negating here keeps the taught support and the patch mapping
    # in the same frame.
    angle = -float(alignment.get("box_angle_deg", 0.0))
    if center is None or size is None:
        return {}
    height, width = int(round(size[0])), int(round(size[1]))
    if abs(size[0] - height) > 1e-6 or abs(size[1] - width) > 1e-6:
        diagnostics.append(
            diagnostic("alignment.box_size", "v1.roi_rounded",
                       "Non-integral V1 box size was rounded to whole pixels", "warning")
        )
    diagnostics.append(
        diagnostic(
            "alignment.box_angle_deg", "v1.angle_convention_converted",
            "Manifest CCW angle converted to the V2 clockwise ROI convention", "info",
        )
    )
    roi = {
        "center_row": center[0],
        "center_col": center[1],
        "width": width,
        "height": height,
        "angle_deg": angle,
        "alignment_mode": "single_box",
        "confirmed": True,
    }
    if width != height:
        diagnostics.append(
            diagnostic(
                "alignment.box_size", "v1.template_support_changed",
                "V1 used a square max(width, height) support; V2 uses the exact rectangle",
                "warning",
            )
        )
    normalized: Dict[str, Any] = {
        "version": 3,
        "template_view": {"tools": tools},
        "roi": roi,
        "matching": matching,
    }
    diagnostics.append(
        diagnostic(
            "template_view.tools", "v1.template_reconstructed",
            "Localization and template-point assets are rebuilt from the reference image",
            "warning",
        )
    )
    objects = manifest.get("objects")
    if isinstance(objects, list) and objects:
        declared = {
            item.get("label"): item.get("object_type")
            for item in objects
            if isinstance(item, dict) and isinstance(item.get("label"), str)
        }
        expected = {
            tool["label"]: tool["object_type"]
            for tool in tools
            if tool["object_type"] in PRIMITIVES
        }
        if declared != expected:
            diagnostics.append(
                diagnostic(
                    "objects", "v1.model_config_mismatch",
                    "config.json and manifest.json tool inventories differ; "
                    "config.json is authoritative",
                    "warning",
                )
            )
    scale = manifest.get("physical_scale_mm")
    if scale is not None:
        if (
            isinstance(scale, bool)
            or not isinstance(scale, (int, float))
            or not np.isfinite(scale)
            or scale <= 0
        ):
            diagnostics.append(
                diagnostic("physical_scale_mm", "v1.invalid_model_manifest",
                           "Expected a positive mm-per-pixel scale")
            )
        elif request.applicability is None:
            diagnostics.append(
                diagnostic(
                    "physical_scale_mm", "v1.physical_units_dropped",
                    "Physical scale needs explicit applicability; recipe stays in pixels",
                    "warning",
                )
            )
        else:
            square = request.square_size_mm or float(scale)
            if request.square_size_mm is None:
                diagnostics.append(
                    diagnostic(
                        "physical_scale_mm", "v1.calibration_derived",
                        "No original board square size supplied; a scalar calibration "
                        "is derived from the V1 scale",
                        "warning",
                    )
                )
            normalized["calibration"] = {
                "scale_mm_per_px": float(scale),
                "square_size_mm": float(square),
            }
    return normalized


def prepare_v1_conversion(
    request: LegacyV1ConversionRequest, files: SharedFiles
) -> LegacyV1ConversionRequest:
    """Read, pin and validate explicit V1 project files before admission.

    Args:
        request: Typed explicit V1 project files, upstream identities and output.
        files: Configured shared-directory boundary with size limits and digest
            verification.

    Returns:
        Request carrying normalized legacy data and preparation diagnostics.
    """
    diagnostics: List[CompatibilityDiagnostic] = []
    config, config_file, config_diagnostics = _read_json(
        request.config, files, "config"
    )
    diagnostics.extend(config_diagnostics)
    manifest: Optional[Dict[str, Any]] = None
    manifest_file: Optional[FileReference] = None
    if request.model_manifest is not None:
        manifest, manifest_file, manifest_diagnostics = _read_json(
            request.model_manifest, files, "model_manifest"
        )
        diagnostics.extend(manifest_diagnostics)
    _, reference_file, npy_file, reference_diagnostics = _reference_inputs(
        request, manifest, files
    )
    diagnostics.extend(reference_diagnostics)
    normalized = _normalize(request, config, manifest, diagnostics) if config else {}
    update: Dict[str, Any] = {
        "config": config_file,
        "normalized": normalized,
        "preparation_diagnostics": diagnostics,
    }
    if manifest_file is not None:
        update["model_manifest"] = manifest_file
    if reference_file is not None:
        update["reference"] = reference_file
    if npy_file is not None:
        update["reference_npy"] = npy_file
    return request.model_copy(update=update)


def _reference_image_input(
    request: LegacyV1ConversionRequest, context: CalculationContext
) -> Tuple[Optional[FileReference], Optional[bytes]]:
    """Return either an explicit image reference or encoded NPY pixels.

    Args:
        request: Parsed migration request with at least one reference form.
        context: Isolated operation identity with verified shared-file access.

    Returns:
        Existing image reference, or PNG bytes to stage and publish.
    """
    if request.reference is not None:
        return request.reference, None
    data, _ = context.files.read_bytes(request.reference_npy)
    pixels = _decode_reference_npy(data, context.settings.max_image_pixels)
    success, encoded = cv2.imencode(".png", pixels)
    if not success:
        raise ServiceError("v1.reference_invalid", "reference.npy cannot be encoded as PNG")
    return None, encoded.tobytes()


def _summary(
    result: LegacyConversionResult, request: LegacyV1ConversionRequest
) -> MigrationSummary:
    """Summarize a converted or rejected migration for callers.

    Args:
        result: Shared conversion result.
        request: Original request carrying normalized tools.

    Returns:
        Node/template counts and explicit approximation codes.
    """
    tools = request.normalized.get("template_view", {}).get("tools", [])
    nodes = len(result.resolved_recipe.nodes) if result.resolved_recipe is not None else len(tools)
    templates = sum(1 for item in result.artifacts if item.kind == "template")
    codes = {item.code for item in result.diagnostics if item.severity == "warning"}
    return MigrationSummary(
        nodes=nodes,
        templates=templates,
        dropped_physical_units="v1.physical_units_dropped" in codes,
        approximations=sorted(codes),
    )


def convert_v1_project(
    request: LegacyV1ConversionRequest, context: CalculationContext
) -> LegacyV1ConversionResult:
    """Migrate a normalized V1 project into one executable V2 recipe.

    Args:
        request: Prepared request carrying normalized tools and diagnostics.
        context: Isolated operation identity, resource limits and verified shared-file
            access.

    Returns:
        Converted recipe/assets with V1 provenance and a migration summary.
    """
    diagnostics = list(request.preparation_diagnostics)
    if any(item.severity == "error" for item in diagnostics) or not request.normalized:
        return LegacyV1ConversionResult(
            valid=False,
            source_project=request.config,
            source_config=request.config,
            source_model=request.model_manifest,
            diagnostics=diagnostics,
            summary=MigrationSummary(
                nodes=len(request.normalized.get("template_view", {}).get("tools", [])),
                templates=0,
            ),
        )
    reference, reference_bytes = _reference_image_input(request, context)
    result = convert_normalized(
        value=request.normalized,
        reference=reference,
        recipe_id=request.recipe_id,
        recipe_version=request.recipe_version,
        applicability=request.applicability,
        output=request.output,
        diagnostics=diagnostics,
        context=context,
        source_project=request.config,
        reject_warnings=request.advisory_policy == "reject",
        reference_bytes=reference_bytes,
    )
    return LegacyV1ConversionResult(
        **result.model_dump(),
        source_config=request.config,
        source_model=request.model_manifest,
        summary=_summary(result, request),
    )