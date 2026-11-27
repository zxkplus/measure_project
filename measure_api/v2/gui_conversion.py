"""Convert a GUI model asset into a recipe bundle that ``POST /api/v2/runs`` accepts.

The GUI writes a *model directory*: ``manifest.json`` (contract 3.0/3.1), a numeric
``reference.npy`` plus its portable ``reference.png``, and a derived ``workflow.npz``.
``/api/v2/runs`` cannot read that layout - it needs a schema-1.0 recipe whose templates
are portable ``pixels.npy``/``metadata.npy`` archives. This module closes the gap
offline, without importing the GUI and without teaching a second matching engine:

* geometry, object parameters and preprocessing are translated with the same helpers
  the ``/legacy-project-conversions`` route uses (``convert_parameters``,
  ``legacy_preprocessing``, ``PRIMITIVES``, ``COMPOSITIONS``);
* the localization template is taught by ``build_template`` so the published asset has
  exactly the metadata ``/api/v2/runs`` verifies;
* the finished recipe is validated with ``validate_snapshot`` before anything is
  written, and the bundle is self-contained: a single shared-root alias covers the
  recipe, the reference image and every template.

Run it as::

    python -m measure_api.v2.gui_conversion MODEL_DIR OUT_DIR --alias gui

The GUI directory is only read; every output goes to ``OUT_DIR``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from measure.computation_types import invert_matrix
from measure.preprocessing import to_display_uint8
from measure_api.v2.families.images import RotatedROI
from measure_api.v2.families.templates import (
    TemplateBuildRequest,
    TemplateSearch,
    build_template,
    coordinate_mapping,
)
from measure_api.v2.files import SharedFiles
from measure_api.v2.legacy_conversion import (
    COMPOSITIONS,
    PRIMITIVES,
    CompatibilityDiagnostic,
    convert_parameters,
    diagnostic,
    legacy_preprocessing,
)
from measure_api.v2.models import FileReference, OutputSpec, Point, ServiceSettings
from measure_api.v2.portable_recipes import canonical
from measure_api.v2.recipe_execution import validate_snapshot
from measure_api.v2.recipe_models import Recipe, RecipeRequest
from measure_api.v2.registry import CalculationContext, operations

GUI_MANIFEST = "manifest.json"
GUI_REFERENCE_NPY = "reference.npy"
GUI_REFERENCE_PNG = "reference.png"
GUI_INSPECTION_PNG = "inspection.png"

#: Manifest contract versions this converter understands.
SUPPORTED_CONTRACTS = ("3.0", "3.1")

#: Alignment modes with an exact V2 equivalent.
SUPPORTED_ALIGNMENT_MODES = ("single_box", "none", "fixed_roi")

STAGING = ".gui-conversion-staging"
RECIPE_NAME = "recipe.json"
BUNDLE_NAME = "bundle.json"
REQUEST_NAME = "run-request.json"


class GuiConversionError(ValueError):
    """Blocking incompatibility; ``diagnostics`` explains every located cause."""

    def __init__(self, message: str, diagnostics: Sequence[CompatibilityDiagnostic]) -> None:
        super().__init__(message)
        self.diagnostics = list(diagnostics)


@dataclass
class GuiModel:
    """Read-only view of one GUI model directory after normalization."""

    directory: Path
    manifest: Dict[str, Any]
    reference_png: bytes
    reference_sha256: str
    config: Dict[str, Any]
    diagnostics: List[CompatibilityDiagnostic] = field(default_factory=list)

    @property
    def contract_version(self) -> str:
        return str(self.manifest.get("contract_version"))

    @property
    def alignment_mode(self) -> str:
        return str(self.config["roi"].get("alignment_mode", "single_box"))

    @property
    def object_count(self) -> int:
        return len(self.config["template_view"]["tools"])


@dataclass
class ConversionBundle:
    """Committed, self-contained recipe bundle and its ready-to-POST request."""

    directory: Path
    recipe: Recipe
    recipe_path: Path
    request: Dict[str, Any]
    request_path: Path
    reference: FileReference
    templates: Dict[str, FileReference]
    configuration_sha256: str
    diagnostics: List[CompatibilityDiagnostic]

    def blocking(self) -> List[CompatibilityDiagnostic]:
        return [item for item in self.diagnostics if item.severity == "error"]


# ---------------------------------------------------------------------------
# Reading the GUI asset
# ---------------------------------------------------------------------------


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _encode_png(pixels: np.ndarray) -> bytes:
    """Encode one numeric reference array as lossless PNG bytes."""
    array = to_display_uint8(np.asarray(pixels))
    if array.ndim == 2:
        ok, buffer = cv2.imencode(".png", array)
    else:
        ok, buffer = cv2.imencode(".png", cv2.cvtColor(array, cv2.COLOR_BGR2RGB))
    if not ok:
        raise GuiConversionError(
            "reference.npy cannot be encoded as PNG", [diagnostic("reference", "reference_invalid")]
        )
    return buffer.tobytes()


def read_reference(manifest: Dict[str, Any], directory: Path) -> Tuple[bytes, str]:
    """Return portable reference PNG bytes, preferring the GUI's own copy.

    Args:
        manifest: Parsed ``manifest.json`` of the model asset.
        directory: Model directory holding the numeric reference and PNG copies.

    Returns:
        PNG bytes and their SHA-256 digest.

    Raises:
        GuiConversionError: When neither reference representation is readable or the
            numeric copy does not match the manifest digest.
    """
    declared = (manifest.get("reference_image") or {}).get("sha256")
    png = directory / GUI_REFERENCE_PNG
    numeric = directory / (manifest.get("reference_image") or {}).get("file", GUI_REFERENCE_NPY)
    if png.is_file():
        return png.read_bytes(), _hash(png.read_bytes())
    if not numeric.is_file():
        raise GuiConversionError(
            f"no reference image in {directory}",
            [
                diagnostic(
                    "reference_image",
                    "reference_missing",
                    f"Expected {GUI_REFERENCE_PNG} or {GUI_REFERENCE_NPY}",
                )
            ],
        )
    try:
        pixels = np.load(str(numeric), allow_pickle=False)
    except (OSError, ValueError) as error:
        raise GuiConversionError(
            "reference.npy is unreadable",
            [diagnostic("reference_image", "reference_invalid", str(error))],
        ) from error
    if not isinstance(pixels, np.ndarray) or pixels.ndim not in (2, 3) or not pixels.size:
        raise GuiConversionError(
            "reference.npy is not a grayscale or colour image",
            [diagnostic("reference_image", "reference_invalid", "Expected a 2-D or 3-D array")],
        )
    actual = _hash(numeric.read_bytes())
    if declared and declared != actual:
        raise GuiConversionError(
            "reference.npy does not match the manifest digest",
            [
                diagnostic(
                    "reference_image.sha256",
                    "reference_digest_mismatch",
                    f"manifest {declared} != file {actual}",
                )
            ],
        )
    data = _encode_png(pixels)
    return data, _hash(data)


def _angle_range(matching: Dict[str, Any], found: List[CompatibilityDiagnostic]) -> float:
    """Return a symmetric half-range, reporting an asymmetric GUI search window."""
    value = matching.get("angle_range")
    if value is None:
        return 30.0
    try:
        low, high = float(value[0]), float(value[1])
    except (TypeError, ValueError, IndexError) as error:
        raise GuiConversionError(
            "matching.angle_range must be a numeric pair",
            [diagnostic("matching.angle_range", "malformed_configuration", str(error))],
        ) from error
    if low > high:
        low, high = high, low
    half = max(abs(low), abs(high))
    if abs(low + high) > 1e-9:
        found.append(
            diagnostic(
                "matching.angle_range",
                "asymmetric_angle_range",
                f"[{low}, {high}] is centred on {(low + high) / 2:g} deg; the V2 search "
                "window is centred on the box angle",
                "warning",
            )
        )
    return half


def _pyramid_enabled(matching: Dict[str, Any]) -> bool:
    """Whether the GUI model had the pyramid coarse search switched on.

    The exponent is a real level number in v1 as well (``2**-n``); decimation
    additionally needs a positive size cap, so both must be set for the model to
    have searched a coarse level at all.
    """
    try:
        decimate = int(matching.get("decimate_exponent", 0) or 0)
        cap = int(matching.get("coarse_template_side", 128) or 0)
    except (TypeError, ValueError):
        return False
    return decimate > 0 and cap > 0


def _pyramid_cap(matching: Dict[str, Any], width: int, height: int) -> int:
    """Size cap that reproduces the GUI model's decimation in v2.

    v2 always applies ``min(1, cap / longest_side)``; when the GUI had the
    switch off, lifting the cap above the support keeps the factor at 1.0
    instead of silently introducing a downsampling step the operator disabled.
    """
    cap = int(matching.get("coarse_template_side", 128))
    enabled = matching.get("pyramid_enabled")
    if enabled is None:
        enabled = _pyramid_enabled(matching)
    if enabled:
        return cap
    return max(cap, int(width), int(height))


def normalize_manifest(
    manifest: Dict[str, Any], found: Optional[List[CompatibilityDiagnostic]] = None
) -> Dict[str, Any]:
    """Translate a manifest model asset into the normalized structure V2 maps from.

    Args:
        manifest: Parsed ``manifest.json`` with contract 3.0/3.1.
        found: Diagnostic sink extended in place.

    Returns:
        Normalized ``roi``/``matching``/``template_view``/``alignment``/``calibration``
        sections using the GUI ``(row, col)`` and radian conventions the shared V2
        parameter translator already documents.

    Raises:
        GuiConversionError: On an unsupported contract, alignment mode or object list.
    """
    found = found if found is not None else []
    contract = str(manifest.get("contract_version"))
    if contract not in SUPPORTED_CONTRACTS:
        raise GuiConversionError(
            f"unsupported GUI manifest contract {contract!r}",
            [
                diagnostic(
                    "contract_version",
                    "unsupported_contract",
                    f"Supported: {', '.join(SUPPORTED_CONTRACTS)}",
                )
            ],
        )
    alignment = manifest.get("alignment") or {}
    mode = str(alignment.get("mode", "single_box"))
    if mode not in SUPPORTED_ALIGNMENT_MODES:
        reason = {
            "obb": "an injected detector has no recipe-level equivalent",
            "multi_point": "GUI affine control composition is not translated",
        }.get(mode, "no translated stage-1 strategy")
        raise GuiConversionError(
            f"alignment mode {mode!r} has no V2 equivalent: {reason}",
            [
                diagnostic(
                    "alignment.mode",
                    "unsupported_alignment_mode",
                    f"{reason}; supported: {', '.join(SUPPORTED_ALIGNMENT_MODES)}",
                )
            ],
        )
    center = list(alignment.get("box_center") or [0.0, 0.0])
    size = list(alignment.get("box_size") or [0.0, 0.0])
    if len(center) != 2 or len(size) != 2:
        raise GuiConversionError(
            "alignment.box_center/box_size must be pairs",
            [diagnostic("alignment.box_size", "malformed_configuration", "Expected (row, col)")],
        )
    # The GUI patch is ``cv2.warpAffine(..., (int(w), int(h)))``: truncate, do not round.
    height, width = int(size[0]), int(size[1])
    if width < 2 or height < 2:
        raise GuiConversionError(
            "the taught box is degenerate",
            [diagnostic("alignment.box_size", "degenerate_box", str(size))],
        )
    if width > 32768 or height > 32768:
        raise GuiConversionError(
            "the taught box exceeds the V2 patch limit",
            [diagnostic("alignment.box_size", "box_too_large", str(size))],
        )
    roi = {
        "center_row": float(center[0]),
        "center_col": float(center[1]),
        "width": width,
        "height": height,
        "angle_deg": float(alignment.get("box_angle_deg", 0.0)),
        "confirmed": True,
        "alignment_mode": mode,
    }
    matching = manifest.get("matching") or {}
    preprocessor = alignment.get("preprocessor") or {"type": "raw"}
    if not isinstance(preprocessor, dict):
        raise GuiConversionError(
            "alignment.preprocessor must be an object",
            [diagnostic("alignment.preprocessor", "malformed_configuration", str(preprocessor))],
        )
    try:
        preprocessing = legacy_preprocessing(preprocessor)
    except (ValueError, KeyError, TypeError) as error:
        raise GuiConversionError(
            "unsupported template preprocessor",
            [diagnostic("alignment.preprocessor", "unsupported_preprocessor", str(error))],
        ) from error
    normalized_matching: Dict[str, Any] = {
        "preprocessor_type": preprocessing["type"],
        "preprocessing": preprocessing,
        "match_score_threshold": float(matching.get("match_score_threshold", 0.5)),
        "angle_range_deg": _angle_range(matching, found),
        "angle_step_deg": float(matching.get("angle_step", 1.0)),
        "max_matches": int(matching.get("max_matches", 1)),
        "overlap": float(matching.get("overlap", 0.3)),
        "coarse_fine": bool(matching.get("coarse_fine", True)),
        "coarse_angle_step_deg": float(matching.get("coarse_angle_step", 5.0)),
        "coarse_template_side": int(matching.get("coarse_template_side", 128)),
        # v1 disables decimation entirely when the switch is off, while v2's
        # size cap is always active. Keep the flag so the search builder can
        # lift the cap above the support and keep factor == 1.0.
        "pyramid_enabled": _pyramid_enabled(matching),
    }
    found.append(diagnostic(
        "matching.match_score_threshold",
        "score_estimator_changed",
        "GUI scores fill rotated corners with the template mean while v2 masks "
        "them, so the taught match threshold must be re-validated after conversion",
        "warning",
    ))
    for name in ("scale_min", "scale_max", "scale_step"):
        if name in matching:
            normalized_matching[name] = float(matching[name])
    objects = manifest.get("objects")
    if not isinstance(objects, list) or not objects:
        raise GuiConversionError(
            "the model declares no measurements",
            [diagnostic("objects", "no_measurements", "Expected at least one object")],
        )
    tools: List[Dict[str, Any]] = []
    for index, item in enumerate(objects):
        path = f"objects.{index}"
        if not isinstance(item, dict):
            raise GuiConversionError(
                "objects entries must be objects",
                [diagnostic(path, "malformed_configuration", str(type(item).__name__))],
            )
        kind = item.get("object_type")
        label = item.get("label")
        parameters = item.get("parameters") or {}
        if not isinstance(label, str) or not label:
            raise GuiConversionError(
                "every object needs a nonempty label",
                [diagnostic(path + ".label", "malformed_configuration", str(label))],
            )
        if not isinstance(parameters, dict):
            raise GuiConversionError(
                "object parameters must be an object",
                [diagnostic(path + ".parameters", "malformed_configuration", str(label))],
            )
        tools.append({"object_type": kind, "label": label, "params": dict(parameters)})
    labels = [tool["label"] for tool in tools]
    if len(set(labels)) != len(labels):
        raise GuiConversionError(
            "object labels must be unique",
            [diagnostic("objects", "ambiguous_labels", ", ".join(sorted(labels)))],
        )
    calibration: Optional[Dict[str, Any]] = None
    scale = manifest.get("physical_scale_mm")
    if scale:
        calibration = {"scale_mm_per_px": float(scale), "square_size_mm": float(scale) * 100.0}
    return {
        "version": 3,
        "roi": roi,
        "matching": normalized_matching,
        "template_view": {"tools": tools},
        "alignment": {"mode": mode},
        "calibration": calibration,
    }


def load_gui_model(directory: Path) -> GuiModel:
    """Read and normalize one GUI model directory without modifying it.

    Args:
        directory: Model directory holding ``manifest.json`` and a reference copy.

    Returns:
        Immutable model view plus advisory diagnostics.

    Raises:
        GuiConversionError: When the manifest, contract or reference cannot be used.
    """
    directory = Path(directory)
    manifest_path = directory / GUI_MANIFEST
    if not manifest_path.is_file():
        raise GuiConversionError(
            f"{manifest_path} does not exist",
            [diagnostic("manifest", "manifest_missing", "Expected a GUI model directory")],
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise GuiConversionError(
            "manifest.json is not valid JSON",
            [diagnostic("manifest", "malformed_configuration", str(error))],
        ) from error
    if not isinstance(manifest, dict):
        raise GuiConversionError(
            "manifest.json must contain an object",
            [diagnostic("manifest", "malformed_configuration", "Expected an object")],
        )
    found: List[CompatibilityDiagnostic] = []
    reference_png, reference_sha = read_reference(manifest, directory)
    config = normalize_manifest(manifest, found)
    alignment = manifest.get("alignment") or {}
    template_size = int(alignment.get("template_size") or 0)
    roi = config["roi"]
    if template_size and template_size > min(roi["width"], roi["height"]) + 2:
        found.append(
            diagnostic(
                "alignment.template_size",
                "support_geometry_changed",
                f"The GUI correlates a {template_size}x{template_size} square while the "
                f"converted support is the {roi['width']}x{roi['height']} taught box",
                "warning",
            )
        )
    if config.get("calibration") is None:
        found.append(
            diagnostic(
                "physical_scale_mm",
                "pixel_only",
                "The model declares no physical scale; results carry pixel units only",
                "info",
            )
        )
    return GuiModel(
        directory=directory,
        manifest=manifest,
        reference_png=reference_png,
        reference_sha256=reference_sha,
        config=config,
        diagnostics=found,
    )


# ---------------------------------------------------------------------------
# Geometry and teaching
# ---------------------------------------------------------------------------


def reference_to_patch(roi: Dict[str, Any]) -> np.ndarray:
    """Return the GUI-exact reference-to-aligned-patch transform.

    ``measure.rectify.crop_and_straighten`` maps a patch pixel ``(x, y) = (col, row)``
    through ``R(theta)`` around ``(width / 2, height / 2)`` and translates it to the box
    centre. This is its inverse, so numbers taught in the GUI's straightened space keep
    their meaning; note the GUI uses ``size / 2`` while ``roi_to_patch_matrix`` uses
    ``(size - 1) / 2`` for the *template support*.

    Args:
        roi: Normalized ROI with ``center_col``, ``center_row``, ``width``,
            ``height`` and ``angle_deg``.

    Returns:
        A 3-by-3 similarity mapping reference pixels to aligned patch pixels.
    """
    angle = math.radians(float(roi.get("angle_deg", 0.0)))
    cos, sin = math.cos(angle), math.sin(angle)
    width, height = float(roi["width"]), float(roi["height"])
    cx, cy = float(roi["center_col"]), float(roi["center_row"])
    return np.array(
        [
            [cos, sin, width / 2.0 - cos * cx - sin * cy],
            [-sin, cos, height / 2.0 + sin * cx - cos * cy],
            [0.0, 0.0, 1.0],
        ]
    )


class _Builder:
    """Teach templates and publish a validated recipe into one shared-root bundle."""

    def __init__(self, model: GuiModel, output: Path, alias: str) -> None:
        self.model = model
        self.output = output
        self.alias = alias
        self.staging = output / STAGING
        self.settings = ServiceSettings(
            shared_roots={alias: output}, writable_roots=[alias]
        )
        self.files = SharedFiles(self.settings)
        self.diagnostics: List[CompatibilityDiagnostic] = list(model.diagnostics)
        self.templates: Dict[str, FileReference] = {}
        self.reference = FileReference(
            root=alias, path=GUI_REFERENCE_PNG, sha256=model.reference_sha256
        )

    # -- publishing ------------------------------------------------------
    def teach(self, name: str, roi: RotatedROI, preprocessing: Dict[str, Any],
              anchor: Optional[Point] = None) -> FileReference:
        """Teach one portable template and publish it at ``<name>.npz``."""
        context = CalculationContext(self.settings, "template", "gui-conversion")
        built = build_template(
            TemplateBuildRequest(
                image=self.reference,
                roi=roi,
                anchor=anchor,
                preprocessing=preprocessing,
                output=OutputSpec(directory=FileReference(root=self.alias, path=STAGING)),
            ),
            context,
        )
        target = self.output / f"{name}.npz"
        target.write_bytes(context.files.read_bytes(built.template)[0])
        for artifact in built.artifacts:
            if artifact.kind == "preview":
                preview = self.output / f"{name}.preview.png"
                preview.write_bytes(context.files.read_bytes(artifact.file)[0])
        shutil.rmtree(self.staging, ignore_errors=True)
        reference = FileReference(
            root=self.alias, path=target.name, sha256=_hash(target.read_bytes())
        )
        self.templates[name] = reference
        return reference

    # -- conversion ------------------------------------------------------
    def convert(self, recipe_id: str, version: str) -> Recipe:
        """Build, validate and return the executable recipe for this model."""
        roi = self.model.config["roi"]
        matching = self.model.config["matching"]
        tools = self.model.config["template_view"]["tools"]
        for tool in tools:
            kind = tool["object_type"]
            if kind not in PRIMITIVES and kind not in COMPOSITIONS:
                raise GuiConversionError(
                    f"object type {kind!r} is unsupported",
                    [
                        diagnostic(
                            f"objects.{tool['label']}.object_type",
                            "unsupported_tool",
                            f"Known: {', '.join(sorted(set(PRIMITIVES) | set(COMPOSITIONS)))}",
                        )
                    ],
                )
        providers: Dict[str, Dict[str, str]] = {}
        for tool in tools:
            kind = tool["object_type"]
            port = PRIMITIVES.get(kind, (None, "line" if kind == "TwoPointsLine" else "value"))[1]
            providers[tool["label"]] = {"node": _node_id(kind, tool["label"]), "port": port}
        for tool in tools:
            if tool["object_type"] != "EdgePair":
                continue
            for suffix, port in (("_A", "edge_a"), ("_B", "edge_b")):
                providers[tool["label"] + suffix] = {
                    "node": providers[tool["label"]]["node"],
                    "port": port,
                }
        nodes: List[Dict[str, Any]] = []
        for tool in tools:
            kind, label, parameters = tool["object_type"], tool["label"], tool["params"]
            if kind in COMPOSITIONS:
                operation, fields = COMPOSITIONS[kind]
                try:
                    inputs = {name: providers[parameters[field]] for name, field in fields.items()}
                except KeyError as error:
                    raise GuiConversionError(
                        f"{label} depends on an unknown provider {error}",
                        [
                            diagnostic(
                                f"objects.{label}.parameters",
                                "unresolved_dependency",
                                "Compositions need declared upstream labels",
                            )
                        ],
                    ) from error
                node_parameters: Dict[str, Any] = {}
            else:
                operation = PRIMITIVES[kind][0]
                inputs, node_parameters = {}, convert_parameters(kind, parameters)
                # The GUI core always warps a probe with BORDER_REPLICATE, so a probe
                # that reaches past the patch edge samples edge pixels instead of
                # failing. Keep that behaviour instead of V2's rejecting default.
                if "border_policy" in operations()[operation].request_model.model_fields:
                    node_parameters.setdefault("border_policy", "replicate")
            nodes.append(
                {
                    "id": providers[label]["node"],
                    "label": label,
                    "operation": operation,
                    "parameters": node_parameters,
                    "inputs": inputs,
                }
            )
        recipe: Dict[str, Any] = {
            "id": recipe_id,
            "version": version,
            "reference": self.reference.model_dump(mode="json"),
            "nodes": nodes,
        }
        if self.model.alignment_mode == "none":
            if matching.get("max_matches"):
                self.diagnostics.append(
                    diagnostic(
                        "alignment.mode",
                        "localization_skipped",
                        "Mode 'none' has no stage-1 search; the whole image is the patch",
                        "warning",
                    )
                )
        elif self.model.alignment_mode == "fixed_roi":
            recipe["localization"] = {
                "fixed_roi": {
                    "center": {
                        "x": float(roi["center_col"]),
                        "y": float(roi["center_row"]),
                    },
                    "width": int(roi["width"]),
                    "height": int(roi["height"]),
                    "angle_degrees": float(roi["angle_deg"]),
                },
                "max_targets": 1,
            }
            recipe["alignment"] = {
                "width": int(roi["width"]),
                "height": int(roi["height"]),
                "reference_to_patch": coordinate_mapping(
                    reference_to_patch(roi), "reference", "aligned"
                ).model_dump(mode="json"),
            }
        else:
            support = self.teach(
                "localization",
                RotatedROI(
                    center=Point(x=float(roi["center_col"]), y=float(roi["center_row"])),
                    width=int(roi["width"]),
                    height=int(roi["height"]),
                    angle_degrees=float(roi["angle_deg"]),
                ),
                matching["preprocessing"],
            )
            half = float(matching["angle_range_deg"])
            angle = float(roi["angle_deg"])
            search: Dict[str, Any] = {
                "template": support.model_dump(mode="json"),
                "search": TemplateSearch(
                    score_threshold=float(matching["match_score_threshold"]),
                    max_matches=int(matching["max_matches"]),
                    overlap=float(matching["overlap"]),
                    angle_min=max(-360.0, angle - half),
                    angle_max=min(360.0, angle + half),
                    angle_step=float(matching["angle_step_deg"]),
                    coarse_fine=bool(matching["coarse_fine"]),
                    coarse_angle_step=float(matching["coarse_angle_step_deg"]),
                    pyramid_levels=0,
                    pyramid_max_template_size=_pyramid_cap(
                        matching, int(roi["width"]), int(roi["height"])),
                    **{
                        name: float(matching[name])
                        for name in ("scale_min", "scale_max", "scale_step")
                        if name in matching
                    },
                ).model_dump(mode="json"),
            }
            recipe["localization"] = search
            recipe["alignment"] = {
                "width": int(roi["width"]),
                "height": int(roi["height"]),
                "reference_to_patch": coordinate_mapping(
                    reference_to_patch(roi), "reference", "aligned"
                ).model_dump(mode="json"),
            }
        old_to_patch = reference_to_patch(roi)
        for tool, node in zip(tools, nodes):
            if tool["object_type"] != "TemplateMatchPoint":
                continue
            parameters = tool["params"]
            point = invert_matrix(old_to_patch) @ [
                float(parameters["col"]),
                float(parameters["row"]),
                1.0,
            ]
            size = int(parameters.get("template_size", 40))
            support = self.teach(
                node["id"],
                RotatedROI(
                    center=Point(x=float(point[0]), y=float(point[1])),
                    width=size,
                    height=size,
                    angle_degrees=float(roi["angle_deg"]),
                ),
                legacy_preprocessing(parameters.get("preprocessor_type", "raw")),
                anchor=Point(x=float(point[0]), y=float(point[1])),
            )
            half = float(parameters.get("angle_range_half", 15.0))
            node["parameters"] = {
                "template": support.model_dump(mode="json"),
                "search": TemplateSearch(
                    max_matches=1,
                    score_threshold=float(parameters.get("match_score_threshold", 0.5)),
                    angle_min=-half,
                    angle_max=half,
                    angle_step=float(parameters.get("angle_step", 1.0)),
                    subpixel=bool(parameters.get("use_subpixel", True)),
                    pyramid_levels=0,
                    pyramid_max_template_size=_pyramid_cap(parameters, size, size),
                    # The tool's matcher choice travels with it; an asset saved
                    # before these keys existed converts the same as before.
                    metric=str(parameters.get("metric", "use_polarity")),
                    algorithm=str(parameters.get("algorithm", "ncc")),
                    min_contrast=float(parameters.get("min_contrast", 0.0)),
                ).model_dump(mode="json"),
            }
        calibration = self.model.config.get("calibration")
        if calibration:
            raise GuiConversionError(
                "physical_scale_mm needs an explicit calibration applicability",
                [
                    diagnostic(
                        "physical_scale_mm",
                        "unresolved_calibration",
                        "A scalar scale requires camera_id, setup_id and image_size; "
                        "drop the scale or calibrate with a portable asset first",
                    )
                ],
            )
        validated = validate_snapshot(RecipeRequest(recipe=Recipe.model_validate(recipe)))
        if not validated.valid:
            raise GuiConversionError(
                "the converted recipe is invalid",
                self.diagnostics
                + [
                    diagnostic(
                        issue.get("field_path", "recipe"),
                        issue.get("code", "invalid_recipe"),
                        issue.get("message", "Converted recipe is invalid"),
                    )
                    for issue in validated.issues
                ],
            )
        self.configuration_sha256 = validated.configuration_sha256
        return validated.resolved_recipe


def _node_id(kind: str, label: str) -> str:
    """Return a stable, label-safe V2 node identifier."""
    digest = hashlib.sha256(canonical([kind, label])).hexdigest()[:16]
    return "gui-" + digest


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def convert_gui_model(
    model_directory: Path,
    output_directory: Path,
    *,
    recipe_id: Optional[str] = None,
    version: str = "1",
    alias: str = "gui",
    image: Optional[Path] = None,
    timeout_ms: int = 300000,
) -> ConversionBundle:
    """Convert one GUI model directory into a ``/api/v2/runs`` bundle.

    Args:
        model_directory: GUI model asset (``manifest.json`` + reference copies).
        output_directory: Destination bundle; becomes the ``alias`` shared root.
        recipe_id: Explicit recipe identity; defaults to the model directory name.
        version: Upstream-owned recipe version label.
        alias: Shared-root alias used in every published file reference.
        image: Optional inspection image copied into the bundle as ``inspection.png``.
        timeout_ms: Operation budget written into the ready-to-POST request.

    Returns:
        The committed bundle, its recipe and its ready-to-POST request body.

    Raises:
        GuiConversionError: When the model cannot be expressed as a valid recipe.
    """
    model = load_gui_model(Path(model_directory))
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()) and not (output / RECIPE_NAME).exists():
        raise GuiConversionError(
            f"{output} is not empty",
            [
                diagnostic(
                    "output.directory",
                    "output_not_empty",
                    "Use an empty or previously generated bundle directory",
                )
            ],
        )
    for item in output.iterdir():
        if item.is_file():
            item.unlink()
        else:
            shutil.rmtree(item, ignore_errors=True)
    (output / GUI_REFERENCE_PNG).write_bytes(model.reference_png)
    builder = _Builder(model, output, alias)
    recipe = builder.convert(recipe_id or model.directory.name, str(version))
    shutil.rmtree(builder.staging, ignore_errors=True)
    recipe_path = output / RECIPE_NAME
    recipe_path.write_bytes(canonical(recipe.model_dump(mode="json")))
    image_reference: Optional[Dict[str, Any]] = None
    candidate = Path(image) if image is not None else model.directory / GUI_INSPECTION_PNG
    if candidate.is_file():
        payload = candidate.read_bytes()
        (output / GUI_INSPECTION_PNG).write_bytes(payload)
        image_reference = {"root": alias, "path": GUI_INSPECTION_PNG, "sha256": _hash(payload)}
    request: Dict[str, Any] = {
        "mode": "sync",
        "timeout_ms": int(timeout_ms),
        "recipe_file": {"root": alias, "path": RECIPE_NAME},
        "image": image_reference or {"root": alias, "path": "inspection.png"},
    }
    request_path = output / REQUEST_NAME
    request_path.write_bytes(canonical(request))
    bundle = ConversionBundle(
        directory=output,
        recipe=recipe,
        recipe_path=recipe_path,
        request=request,
        request_path=request_path,
        reference=builder.reference,
        templates=dict(builder.templates),
        configuration_sha256=builder.configuration_sha256,
        diagnostics=list(builder.diagnostics),
    )
    (output / BUNDLE_NAME).write_bytes(
        canonical(
            {
                "schema_version": "1.0",
                "source_model": str(model.directory),
                "contract_version": model.contract_version,
                "alignment_mode": model.alignment_mode,
                "objects": model.object_count,
                "recipe_id": recipe.id,
                "recipe_version": recipe.version,
                "configuration_sha256": bundle.configuration_sha256,
                "files": {
                    path.name: _hash(path.read_bytes())
                    for path in sorted(output.iterdir())
                    if path.is_file() and path.name not in {BUNDLE_NAME}
                },
                "diagnostics": [
                    item.model_dump(mode="json") for item in builder.diagnostics
                ],
            }
        )
    )
    (output / "README.txt").write_text(_readme(bundle, alias), encoding="utf-8")
    return bundle


def _readme(bundle: ConversionBundle, alias: str) -> str:
    return f"""GUI model -> V2 recipe bundle
============================

Mount this directory as the "{alias}" shared root, then:

  curl -sS -X POST "$BASE_URL/api/v2/runs" \\
    -H 'content-type: application/json' \\
    --data-binary @{REQUEST_NAME}

Files
  {RECIPE_NAME}          recipe schema 1.0 executed by POST /api/v2/runs
  {GUI_REFERENCE_PNG}    teaching image referenced by the recipe
  *.npz           portable templates (pixels.npy + metadata.npy)
  {REQUEST_NAME}   ready-to-POST body (recipe_file + image)
  {BUNDLE_NAME}          provenance, digests and conversion diagnostics

The recipe was validated before publication: configuration_sha256
{bundle.configuration_sha256}

Caveats
  * Results are pixel-only unless the model carried a portable calibration.
  * The GUI correlates a square stage-1 template; the converted support is the
    taught box, so NCC scores are comparable but not bit-identical.
  * Node coordinates are the GUI's straightened-patch coordinates; the recipe
    declares the matching reference-to-patch transform.
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _run_bundle(bundle: ConversionBundle, alias: str) -> Dict[str, Any]:
    """Execute the published request through the real V2 HTTP application."""
    from fastapi.testclient import TestClient

    from measure_api.v2.app import create_app

    app = create_app(
        ServiceSettings(shared_roots={alias: bundle.directory}, writable_roots=[alias])
    )
    with TestClient(app) as client:
        response = client.post("/api/v2/runs", json=bundle.request)
    payload = response.json()
    return {"http_status": response.status_code, "body": payload}


def _summary(bundle: ConversionBundle, run: Optional[Dict[str, Any]]) -> str:
    lines = [
        f"bundle      {bundle.directory}",
        f"recipe      {bundle.recipe.id} v{bundle.recipe.version}",
        f"sha256      {bundle.configuration_sha256}",
        f"templates   {', '.join(sorted(bundle.templates)) or '-'}",
    ]
    for item in bundle.diagnostics:
        lines.append(f"{item.severity:<7} {item.field_path}: {item.code} - {item.message}")
    if run is not None:
        body = run["body"]
        lines.append(f"http        {run['http_status']}")
        if isinstance(body, dict) and body.get("result"):
            result = body["result"]
            lines.append(
                "run         state={} status={} evaluation={} targets={}".format(
                    body.get("state"), result.get("status"),
                    result.get("evaluation_status"), len(result.get("targets") or []),
                )
            )
            for target in result.get("targets") or []:
                lines.append(f"  target    {target['target_id']} valid={target['valid']}")
                for node in target.get("nodes") or []:
                    lines.append(
                        f"    node    {node['id']} state={node['state']} valid={node['valid']}"
                    )
        else:
            lines.append("error       " + json.dumps(body.get("error"), ensure_ascii=False))
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Convert a GUI model directory into a runnable V2 recipe bundle."""
    parser = argparse.ArgumentParser(
        prog="python -m measure_api.v2.gui_conversion",
        description="Convert a GUI model asset into a recipe POST /api/v2/runs accepts",
    )
    parser.add_argument("model", type=Path, help="GUI model directory (manifest.json)")
    parser.add_argument("output", type=Path, help="bundle directory, used as the shared root")
    parser.add_argument(
        "--recipe-id", default=None, help="recipe identity; defaults to the model name"
    )
    parser.add_argument("--version", default="1", help="upstream recipe version label")
    parser.add_argument("--alias", default="gui", help="shared-root alias written into references")
    parser.add_argument("--image", type=Path, default=None, help="inspection image to copy")
    parser.add_argument("--timeout-ms", type=int, default=300000, help="request operation budget")
    parser.add_argument("--run", action="store_true", help="POST the bundle to /api/v2/runs")
    parser.add_argument("--json", action="store_true", help="print a JSON summary")
    arguments = parser.parse_args(argv)
    try:
        bundle = convert_gui_model(
            arguments.model,
            arguments.output,
            recipe_id=arguments.recipe_id,
            version=arguments.version,
            alias=arguments.alias,
            image=arguments.image,
            timeout_ms=arguments.timeout_ms,
        )
    except GuiConversionError as error:
        print(f"conversion failed: {error}", file=sys.stderr)
        for item in error.diagnostics:
            print(
                f"  {item.severity:<7} {item.field_path}: {item.code} - {item.message}",
                file=sys.stderr,
            )
        return 2
    run = _run_bundle(bundle, arguments.alias) if arguments.run else None
    if arguments.json:
        print(
            json.dumps(
                {
                    "bundle": str(bundle.directory),
                    "recipe_id": bundle.recipe.id,
                    "recipe_version": bundle.recipe.version,
                    "configuration_sha256": bundle.configuration_sha256,
                    "request": bundle.request,
                    "templates": sorted(bundle.templates),
                    "diagnostics": [
                        item.model_dump(mode="json") for item in bundle.diagnostics
                    ],
                    "run": run,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(_summary(bundle, run))
    if run is not None:
        body = run["body"]
        if run["http_status"] != 200:
            return 1
        result = (body or {}).get("result") or {}
        if result.get("status") not in {"completed", "no_targets"}:
            return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
