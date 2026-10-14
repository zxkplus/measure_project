"""Safe one-way conversion of supported GUI JSON and Unicode core NPZ configuration."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np
from pydantic import Field

from measure_api.v2.calibration_assets import CalibrationApplicability
from measure_api.v2.errors import ServiceError
from measure_api.v2.files import ArtifactWriter, SharedFiles
from measure_api.v2.models import (
    CalculationRequest,
    FileDescriptor,
    FileReference,
    OutputSpec,
    StrictModel,
)
from measure_api.v2.portable_recipes import (
    canonical,
    output_reference,
    reference_key,
    rewrite_references,
    safe_numeric_archive,
)
from measure_api.v2.recipe_execution import prepare_recipe, validate_snapshot
from measure_api.v2.recipe_models import Recipe, RecipeRequest
from measure_api.v2.registry import CalculationContext


class CompatibilityDiagnostic(StrictModel):
    field_path: str
    severity: Literal["error", "warning", "info"]
    code: str
    message: str


class LegacyConversionRequest(CalculationRequest):
    project: FileReference
    reference: Optional[FileReference] = None
    recipe_id: str = Field(min_length=1)
    recipe_version: str = Field(min_length=1)
    applicability: Optional[CalibrationApplicability] = None
    output: OutputSpec
    project_data: Dict[str, Any] = Field(default_factory=dict)
    preparation_diagnostics: List[CompatibilityDiagnostic] = Field(default_factory=list)


class LegacyConversionResult(StrictModel):
    valid: bool
    resolved_recipe: Optional[Recipe] = None
    configuration_sha256: Optional[str] = None
    execution_order: List[str] = Field(default_factory=list)
    source_project: FileReference
    assets: List[FileReference] = Field(default_factory=list)
    diagnostics: List[CompatibilityDiagnostic] = Field(default_factory=list)
    artifacts: List[FileDescriptor] = Field(default_factory=list)
    manifest: Optional[FileDescriptor] = None


def diagnostic(
    path: str, code: str, message: str, severity: str = "error"
) -> CompatibilityDiagnostic:
    """Locate a compatibility outcome without silently discarding an unsupported field.

    Args:
        path: Filesystem destination or diagnostic field path for this operation.
        code: Stable compatibility diagnostic code for upstream handling.
        message: Actionable explanation of the unsupported or incompatible configuration.
        severity: Whether the compatibility outcome blocks conversion or is advisory.

    Returns:
        Located compatibility diagnostic with explicit code, message and severity.
    """
    return CompatibilityDiagnostic(
        field_path=path, code=code, message=message, severity=severity
    )


def legacy_preprocessing(value: Any) -> dict:
    """Preserve supported GUI presets and safely serialized core preprocessing parameters.

    Args:
        value: Supplied value to normalize under the public schema and coordinate
            convention.

    Returns:
        Supported typed preprocessing parameters translated from legacy values.
    """
    presets = {
        "raw": {"type": "raw"},
        "canny": {"type": "canny", "low_threshold": 50, "high_threshold": 150},
        "sobel": {"type": "sobel", "kernel_size": 3},
        "clahe": {"type": "clahe", "clip_limit": 2},
        "threshold": {"type": "threshold", "threshold": 128},
    }
    if isinstance(value, str):
        if value not in presets:
            raise ValueError("Unsupported legacy preprocessor " + value)
        return presets[value]
    kind = value["type"]
    fields = {
        "raw": {"type"},
        "canny": {"type", "t1", "t2"},
        "sobel": {"type", "ksize"},
        "clahe": {"type", "clip_limit", "tile_grid_size"},
        "threshold": {"type", "threshold", "mode"},
    }
    if kind not in fields or set(value) - fields[kind]:
        raise ValueError("Unsupported legacy preprocessing fields")
    converted = dict(presets[kind])
    names = {
        "t1": "low_threshold",
        "t2": "high_threshold",
        "ksize": "kernel_size",
        "clip_limit": "clip_limit",
        "threshold": "threshold",
        "mode": "mode",
    }
    converted.update({names[key]: item for key, item in value.items() if key in names})
    if "tile_grid_size" in value:
        grid = value["tile_grid_size"]
        if len(grid) != 2 or grid[0] != grid[1]:
            raise ValueError("Only square CLAHE tile grids can convert")
        converted["tile_grid_size"] = grid[0]
    return converted


def read_legacy(data: bytes, files: SharedFiles) -> tuple:
    """Read JSON or nonobject NPZ data without calling any legacy loader or pickle.

    Args:
        data: Original project JSON or safe nonobject NPZ bytes.
        files: Configured shared-directory boundary with size limits and digest
            verification.

    Returns:
        Decoded legacy configuration and explicit compatibility warning/error diagnostics.
    """
    if not data.startswith(b"PK"):
        try:
            value = json.loads(data)
            if not isinstance(value, dict) or value.get("version") != 3:
                raise ValueError("Only GUI project JSON version 3 is supported")
            return value, []
        except (ValueError, TypeError) as error:
            return {}, [diagnostic("project", "unsupported_project_format", str(error))]
    try:
        arrays = safe_numeric_archive(data, files)
        if {
            "version",
            "template_params",
            "measure_defs",
            "preprocessor_json",
            "template_image",
            "edge_template",
        } != set(arrays):
            raise ValueError("Unsupported safe core NPZ fields")
        for name in ("template_image", "edge_template"):
            pixels = arrays[name]
            if (
                pixels.dtype.kind not in "fiu"
                or pixels.ndim not in {2, 3}
                or not np.isfinite(pixels).all()
                or pixels.shape[0] * pixels.shape[1] > files.settings.max_image_pixels
            ):
                raise ValueError("Invalid numeric legacy template")
        if int(arrays["version"].item()) != 1:
            raise ValueError("Only core Unicode NPZ version 1 is supported")
        for name in ("template_params", "measure_defs", "preprocessor_json"):
            if arrays[name].dtype.kind not in "US" or arrays[name].ndim != 0:
                raise ValueError("Core JSON metadata must use scalar Unicode/string arrays")
        parameters = json.loads(str(arrays["template_params"].item()))
        measures = json.loads(str(arrays["measure_defs"].item()))
        pp = json.loads(str(arrays["preprocessor_json"].item()))
        tools = [
            {
                "object_type": measure["measure_type"],
                "label": measure["label"],
                "params": measure["params"],
            }
            for measure in measures
        ]
        angle_range = parameters.get("angle_range", [-30, 30])
        if angle_range[0] != -angle_range[1]:
            raise ValueError("Only symmetric core angular searches are supported")
        matching = {
            "preprocessor_type": pp.get("type", "raw"),
            "preprocessing": legacy_preprocessing(pp),
            "match_score_threshold": parameters.get("match_score_threshold", 0.5),
            "angle_range_deg": angle_range[1],
            "angle_step_deg": parameters.get("angle_step", 1),
            "max_matches": parameters.get("max_matches", 0),
        }
        value = {
            "version": 3,
            "template_view": {"tools": tools},
            "matching": matching,
            "roi": {
                "center_row": parameters["center_row"],
                "center_col": parameters["center_col"],
                "height": parameters["bbox_height"],
                "width": parameters["bbox_width"],
                "angle_deg": parameters["bbox_angle_deg"],
                "alignment_mode": "single_box",
                "confirmed": True,
            },
        }
        warnings = [
            diagnostic(
                "project",
                "core_template_rebuilt",
                "Numeric legacy templates are safely read; supplied reference teaches "
                "the declared rectangular support in the new matching engine",
                "warning",
            )
        ]
        if parameters.get("scale_invariant"):
            value["matching"].update(
                scale_min=parameters["scale_range"][0],
                scale_max=parameters["scale_range"][1],
                scale_step=parameters.get("scale_step", 0.02),
            )
        return value, warnings
    except (ValueError, TypeError, KeyError, IndexError) as error:
        code = (
            "pickle_metadata_requires_safe_manifest"
            if "Object arrays" in str(error)
            else "unsupported_project_format"
        )
        return {}, [diagnostic("project", code, str(error))]


def prepare_conversion(
    request: LegacyConversionRequest, files: SharedFiles
) -> LegacyConversionRequest:
    """Pin explicit or project-local source data before admission, without guessing paths.

    Args:
        request: Typed explicit legacy project/template files, source bindings and output
            destination.
        files: Configured shared-directory boundary with size limits and digest
            verification.

    Returns:
        Explicit legacy project/template files, source bindings and output destination.
    """
    data, project = files.read_bytes(request.project)
    value, diagnostics = read_legacy(data, files)
    reference = request.reference
    if value and reference is None:
        candidate = FileReference(
            root=project.root, path=str(Path(project.path).parent / "reference.png")
        )
        try:
            _, reference = files.read_bytes(candidate)
        except ServiceError:
            diagnostics.append(
                diagnostic(
                    "reference",
                    "reference_required",
                    "Supply reference explicitly or a project-local reference.png",
                )
            )
    if reference is not None:
        _, reference = files.read_image(reference)
    return request.model_copy(
        update={
            "project": project,
            "reference": reference,
            "project_data": value,
            "preparation_diagnostics": diagnostics,
        }
    )


COMPOSITIONS = {
    "TwoPointsLine": (
        "geometry.two-point-lines",
        {"point_a": "point_a_label", "point_b": "point_b_label"},
    ),
    "TwoPointsDistance": (
        "geometry.point-distances",
        {"point_a": "point_a_label", "point_b": "point_b_label"},
    ),
    "PointLineDistance": (
        "geometry.point-line-distances",
        {"point": "point_label", "line": "line_label"},
    ),
    "TwoLinesAngle": (
        "geometry.line-angles",
        {"line_a": "line_a_label", "line_b": "line_b_label"},
    ),
    "PointCircleDistance": (
        "geometry.point-circle-distances",
        {"point": "point_label", "circle": "circle_label"},
    ),
}
PRIMITIVES = {
    "EdgePoint": ("measurements.edge-points", "point"),
    "EdgePair": ("measurements.edge-pairs", "center"),
    "FitLine": ("measurements.lines", "line"),
    "FitCircle": ("measurements.circles", "circle"),
    "TemplateMatchPoint": ("measurements.template-points", "point"),
}
PARAMETERS = {
    "EdgePoint": {
        "row",
        "col",
        "angle",
        "length1",
        "length2",
        "sigma",
        "threshold",
        "transition",
        "select",
        "selection",
        "interpolation",
    },
    "EdgePair": {
        "row",
        "col",
        "angle",
        "length1",
        "length2",
        "sigma",
        "threshold",
        "transition",
        "select",
        "selection",
        "interpolation",
    },
    "FitLine": {
        "start",
        "end",
        "measure_length1",
        "measure_length2",
        "num_measures",
        "sigma",
        "threshold",
        "transition",
    },
    "FitCircle": {
        "center",
        "radius",
        "measure_length1",
        "measure_length2",
        "num_measures",
        "sigma",
        "threshold",
        "transition",
        "radius_min",
        "radius_max",
        "start_phi",
        "end_phi",
    },
    "TemplateMatchPoint": {
        "row",
        "col",
        "template_size",
        "preprocessor_type",
        "match_score_threshold",
        "angle_range_half",
        "angle_step",
        "use_subpixel",
    },
}


def xy(value: list) -> dict:
    """Translate legacy row/column pairs into the public xy convention.

    Args:
        value: Supplied value to normalize under the public schema and coordinate
            convention.

    Returns:
        Public xy point translated from the legacy row/column pair.
    """
    return {"x": value[1], "y": value[0]}


def convert_parameters(kind: str, parameters: dict) -> dict:
    """Translate documented GUI search/averaging dimensions, without GUI factories.

    Args:
        kind: Supported transformation or legacy measurement kind to interpret.
        parameters: Literal node or legacy tool parameters before typed normalization.

    Returns:
        Atomic measurement parameters preserving the GUI sampling convention.
    """
    p = parameters
    common = {
        "sigma": p.get("sigma", 1),
        "threshold": p.get("threshold", 30),
        "polarity": p.get("transition", "all"),
    }
    if kind in {"EdgePoint", "EdgePair"}:
        return {
            **common,
            "selection": p.get(
                "selection", p.get("select", "first" if kind == "EdgePoint" else "endpoints")
            ),
            "probe": {
                "center": {"x": p["col"], "y": p["row"]},
                "angle_degrees": math.degrees(p["angle"]),
                "search_half_length": p["length1"],
                "average_half_width": p["length2"],
                "interpolation": p.get("interpolation", "linear"),
            },
        }
    common.update(probe_count=p.get("num_measures", 10 if kind == "FitLine" else 12))
    if kind == "FitLine":
        return {
            **common,
            "start": xy(p["start"]),
            "end": xy(p["end"]),
            "search_half_length": p["measure_length2"],
            "average_half_width": p["measure_length1"],
        }
    if kind == "FitCircle":
        return {
            **common,
            "center": xy(p["center"]),
            "radius": p["radius"],
            "search_half_length": p["measure_length1"],
            "average_half_width": p["measure_length2"],
            "radius_min": p.get("radius_min"),
            "radius_max": p.get("radius_max"),
            "start_angle_degrees": math.degrees(p.get("start_phi", 0)),
            "end_angle_degrees": math.degrees(p.get("end_phi", 2 * math.pi)),
        }
    return {}


def convert_legacy(
    request: LegacyConversionRequest, context: CalculationContext
) -> LegacyConversionResult:
    """Convert supported configuration and assets into one upstream-owned package.

    Args:
        request: Typed explicit legacy project/template files, source bindings and output
            destination.
        context: Isolated operation identity, resource limits and verified shared-file
            access.

    Returns:
        Converted recipe/assets and explicit blocking or advisory compatibility diagnostics.
    """
    diagnostics = list(request.preparation_diagnostics)
    value = request.project_data

    def failure() -> LegacyConversionResult:
        return LegacyConversionResult(
            valid=False, source_project=request.project, diagnostics=diagnostics
        )

    if any(item.severity == "error" for item in diagnostics):
        return failure()
    template_view = value.get("template_view", {})
    if (
        not isinstance(template_view, dict)
        or not isinstance(template_view.get("tools", []), list)
        or len(template_view.get("tools", [])) > 1000
        or any(
            not isinstance(tool, dict)
            or not isinstance(tool.get("params", {}), dict)
            or not isinstance(tool.get("label"), str)
            or not isinstance(tool.get("object_type"), str)
            for tool in template_view.get("tools", [])
        )
    ):
        diagnostics.append(
            diagnostic(
                "template_view.tools",
                "malformed_configuration",
                "Expected bounded tools with typed labels and parameter objects",
            )
        )
        return failure()
    known = {
        "version",
        "project_name",
        "created_at",
        "updated_at",
        "reference_image_path",
        "inspection_image_path",
        "matching",
        "roi",
        "ref_canvas_state",
        "insp_canvas_state",
        "template_view",
        "tool_list_order",
        "tool_visibility",
        "gui",
        "alignment",
        "calibration",
    }
    for key in set(value) - known:
        diagnostics.append(
            diagnostic(key, "unsupported_field", "Unknown legacy configuration field")
        )
    sections = {
        "roi": {
            "center_row",
            "center_col",
            "width",
            "height",
            "angle_deg",
            "confirmed",
            "alignment_mode",
            "control_points",
        },
        "matching": {
            "preprocessor_type",
            "match_score_threshold",
            "angle_range_deg",
            "angle_step_deg",
            "max_matches",
            "overlap",
            "scale_min",
            "scale_max",
            "scale_step",
            "preprocessing",
        },
        "alignment": {"mode", "control_points"},
        "calibration": {"scale_mm_per_px", "square_size_mm", "num_corners", "method"},
    }
    for section, fields in sections.items():
        supplied = value.get(section) or {}
        if not isinstance(supplied, dict):
            diagnostics.append(
                diagnostic(section, "unsupported_field", "Expected configuration object")
            )
            return failure()
        for field in set(supplied) - fields:
            diagnostics.append(
                diagnostic(
                    section + "." + field,
                    "unsupported_field",
                    "Unknown legacy calculation field",
                )
            )
    for key in (
        "gui",
        "ref_canvas_state",
        "insp_canvas_state",
        "tool_list_order",
        "tool_visibility",
    ):
        if key in value:
            diagnostics.append(
                diagnostic(
                    key, "ignored_ui_state", "Upstream owns display/layout state", "info"
                )
            )
    roi = value.get("roi", {})
    if not roi and value.get("matching"):
        diagnostics.append(
            diagnostic(
                "roi",
                "unresolved_roi",
                "Configured localization requires the original teaching ROI",
            )
        )
    if (
        roi.get("alignment_mode", "single_box") != "single_box"
        or value.get("alignment", {}).get("mode", "single_box") != "single_box"
    ):
        diagnostics.append(
            diagnostic(
                "roi.alignment_mode",
                "unsupported_multi_point",
                "Legacy affine control composition has no safe equivalent",
            )
        )
    tools = value.get("template_view", {}).get("tools", [])
    labels = [tool.get("label") for tool in tools]
    if (
        not tools
        or len(set(labels)) != len(labels)
        or any(not isinstance(label, str) or not label for label in labels)
    ):
        diagnostics.append(
            diagnostic(
                "template_view.tools",
                "ambiguous_labels",
                "Nonempty unique labels are required for legacy dependencies",
            )
        )
    for key in set(value.get("template_view", {})) - {"tools"}:
        diagnostics.append(
            diagnostic(
                "template_view." + key,
                "ignored_ui_state",
                "Display transforms do not change taught measurement coordinates",
                "info",
            )
        )
    providers, nodes = {}, []
    for tool in tools:
        label, kind = tool.get("label"), tool.get("object_type")
        identity = "legacy-" + hashlib.sha256(canonical([kind, label])).hexdigest()[:16]
        port = PRIMITIVES.get(kind, (None, "line" if kind == "TwoPointsLine" else "value"))[1]
        providers[label] = {"node": identity, "port": port}
    for tool in tools:
        if tool.get("object_type") == "EdgePair":
            for suffix, port in (("_A", "edge_a"), ("_B", "edge_b")):
                label = tool["label"] + suffix
                if label in providers:
                    diagnostics.append(
                        diagnostic(
                            "template_view.tools",
                            "ambiguous_labels",
                            "Virtual endpoint label conflicts with a declared tool",
                        )
                    )
                providers[label] = {"node": providers[tool["label"]]["node"], "port": port}
    for index, tool in enumerate(tools):
        kind, label, p = tool.get("object_type"), tool.get("label"), tool.get("params", {})
        path = "template_view.tools." + str(index)
        for field in set(tool) - {"object_type", "label", "params", "_selected"}:
            diagnostics.append(
                diagnostic(path + "." + field, "unsupported_field", "Unknown legacy tool field")
            )
        if kind not in PRIMITIVES and kind not in COMPOSITIONS:
            diagnostics.append(diagnostic(path + ".object_type", "unsupported_tool", str(kind)))
            continue
        allowed = PARAMETERS.get(kind, set(COMPOSITIONS.get(kind, (None, {}))[1].values()))
        if set(p) - allowed:
            diagnostics.append(
                diagnostic(
                    path + ".params",
                    "unsupported_parameter",
                    ", ".join(sorted(set(p) - allowed)),
                )
            )
        try:
            if kind in COMPOSITIONS:
                operation, fields = COMPOSITIONS[kind]
                inputs = {name: providers[p[field]] for name, field in fields.items()}
                parameters = {}
            else:
                operation = PRIMITIVES[kind][0]
                inputs, parameters = {}, convert_parameters(kind, p)
            nodes.append(
                {
                    "id": providers[label]["node"],
                    "label": label,
                    "operation": operation,
                    "parameters": parameters,
                    "inputs": inputs,
                }
            )
        except (KeyError, TypeError, ValueError) as error:
            diagnostics.append(
                diagnostic(path + ".params", "unresolved_dependency_or_parameter", str(error))
            )
    calibration = value.get("calibration")
    if calibration and calibration.get("scale_mm_per_px") and request.applicability is None:
        diagnostics.append(
            diagnostic(
                "calibration",
                "unresolved_calibration",
                "Legacy scalar scale needs explicit camera/setup/frame applicability",
            )
        )
    if calibration and calibration.get("scale_mm_per_px"):
        if not calibration.get("square_size_mm") or calibration["square_size_mm"] <= 0:
            diagnostics.append(
                diagnostic(
                    "calibration.square_size_mm",
                    "unresolved_calibration",
                    "A positive original board square size is required",
                )
            )
        if request.applicability is not None:
            image, _ = context.files.read_image(request.reference)
            size = request.applicability.image_size
            if image.shape[:2] != (size.height, size.width):
                diagnostics.append(
                    diagnostic(
                        "applicability.image_size",
                        "unresolved_calibration",
                        "Applicability must match the teaching image dimensions",
                    )
                )
    if any(item.severity == "error" for item in diagnostics):
        return failure()
    from measure_api.v2.families.templates import (
        TemplateBuildRequest,
        build_template,
        coordinate_mapping,
    )

    payloads, replacements = [], {}

    def teach(
        name: str,
        roi_value: dict,
        anchor: Optional[dict] = None,
        preprocessing: Optional[dict] = None,
    ) -> FileReference:
        stage_context, output = context.stage(name)
        built = build_template(
            TemplateBuildRequest(
                image=request.reference,
                roi=roi_value,
                anchor=anchor,
                output=output,
                preprocessing=preprocessing or {"type": "raw"},
            ),
            stage_context,
        )
        content, _ = stage_context.files.read_bytes(built.template)
        filename = name + ".npz"
        payloads.append((filename, "template", "npz", content))
        final = output_reference(request.output, filename, content, context)
        replacements[reference_key(built.template)] = final.model_dump(mode="json")
        return built.template

    try:
        recipe_data = {
            "id": request.recipe_id,
            "version": request.recipe_version,
            "reference": request.reference,
            "nodes": nodes,
        }
        old_to_patch = np.eye(3)
        angle = float(roi.get("angle_deg", 0))
        if roi:
            width, height = int(roi["width"]), int(roi["height"])
            c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
            cx, cy = roi["center_col"], roi["center_row"]
            old_to_patch = np.array(
                [
                    [c, s, float(roi["width"]) / 2 - c * cx - s * cy],
                    [-s, c, float(roi["height"]) / 2 + s * cx - c * cy],
                    [0, 0, 1],
                ]
            )
            matching = value.get("matching", {})
            pp = matching.get("preprocessing") or legacy_preprocessing(
                matching.get("preprocessor_type", "raw")
            )
            template = teach(
                "localization",
                {
                    "center": {"x": cx, "y": cy},
                    "width": width,
                    "height": height,
                    "angle_degrees": angle,
                },
                preprocessing=pp,
            )
            half = matching.get("angle_range_deg", 30)
            search = {
                "score_threshold": matching.get("match_score_threshold", 0.5),
                "angle_min": angle - half,
                "angle_max": angle + half,
                "angle_step": matching.get("angle_step_deg", 1),
                "max_matches": matching.get("max_matches", 0),
                "overlap": matching.get("overlap", 0.3),
            }
            search.update(
                {
                    name: matching[name]
                    for name in ("scale_min", "scale_max", "scale_step")
                    if name in matching
                }
            )
            recipe_data["localization"] = {"template": template, "search": search}
            recipe_data["alignment"] = {
                "width": width,
                "height": height,
                "reference_to_patch": coordinate_mapping(old_to_patch, "reference", "aligned"),
            }
            diagnostics.append(
                diagnostic(
                    "matching",
                    "matching_engine_changed",
                    "Rectangular support, overlap and pyramid search use explicit V2 contracts",
                    "warning",
                )
            )
        for tool, node in zip(tools, nodes):
            if tool["object_type"] != "TemplateMatchPoint":
                continue
            p = tool["params"]
            source_point = np.linalg.inv(old_to_patch) @ [p["col"], p["row"], 1]
            size = int(p.get("template_size", 40))
            template = teach(
                node["id"],
                {
                    "center": {"x": source_point[0], "y": source_point[1]},
                    "width": size,
                    "height": size,
                    "angle_degrees": angle,
                },
                anchor={"x": source_point[0], "y": source_point[1]},
                preprocessing=legacy_preprocessing(p.get("preprocessor_type", "raw")),
            )
            half = p.get("angle_range_half", 15)
            node["parameters"] = {
                "template": template,
                "search": {
                    "max_matches": 1,
                    "score_threshold": p.get("match_score_threshold", 0.5),
                    "angle_min": -half,
                    "angle_max": half,
                    "angle_step": p.get("angle_step", 1),
                    "subpixel": p.get("use_subpixel", True),
                },
            }
        if calibration and calibration.get("scale_mm_per_px"):
            scale = calibration["scale_mm_per_px"]
            square = calibration.get("square_size_mm", scale)
            recipe_data["metric"] = {
                "calibration": {
                    "value": {
                        "kind": "scalar",
                        "unit": "mm",
                        "applicability": request.applicability,
                        "parameters": {
                            "scale_per_px": scale,
                            "square_size": square,
                            "spacing_px": square / scale,
                        },
                    }
                },
                "applicability": request.applicability,
            }
        recipe = Recipe.model_validate(recipe_data)
        # All generated teaching files live beneath the shared private stage root.
        preparation_context, _ = context.stage("conversion-validation")
        prepared = prepare_recipe(RecipeRequest(recipe=recipe), preparation_context.files)
        validated = validate_snapshot(prepared)
        if not validated.valid:
            diagnostics.extend(
                diagnostic(
                    issue.get("field_path", "recipe"),
                    issue.get("code", "invalid_recipe"),
                    issue.get("message", "Converted recipe is invalid"),
                )
                for issue in validated.issues
            )
            return failure()
        relocated = Recipe.model_validate(
            rewrite_references(prepared.recipe.model_dump(mode="json"), replacements)
        )
        final_assets = [
            FileReference.model_validate(
                rewrite_references(ref.model_dump(mode="json"), replacements)
            )
            for ref in prepared.assets
        ]
        validated = validate_snapshot(RecipeRequest(recipe=relocated, assets=final_assets))
        payloads.append(
            ("recipe.json", "recipe", "json", canonical(relocated.model_dump(mode="json")))
        )
        artifacts, manifest = ArtifactWriter(
            context.files, request.output, context.operation_id
        ).commit(payloads)
        return LegacyConversionResult(
            valid=True,
            resolved_recipe=relocated,
            configuration_sha256=validated.configuration_sha256,
            execution_order=validated.execution_order,
            source_project=request.project,
            assets=final_assets,
            diagnostics=diagnostics,
            artifacts=artifacts,
            manifest=manifest,
        )
    except (ValueError, KeyError, TypeError, np.linalg.LinAlgError, ServiceError) as error:
        if isinstance(error, ServiceError) and error.code.startswith(("output.", "operation.")):
            raise
        diagnostics.append(diagnostic("recipe", "incompatible_configuration", str(error)))
        return failure()
