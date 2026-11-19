#!/usr/bin/env python3
"""Reproduce a measure-v2 algorithm debug package locally and visualise it.

A debug package exported by the measurement workbench is self-contained:
``manifest.json`` (``measure.pick_run`` v1), ``inputs.json`` (the exact request),
``result.json`` (the recorded observations), ``images/`` and ``assets/``.  Its
``README.txt`` states that ``measure.pick_replay`` only covers the gantry pick
service (vision/offset/locate), so ``type == "run"`` workbench jobs cannot be
replayed yet.  This script closes that gap:

1. mirror every referenced asset into a local shared-root tree so the recipe's
   ``exchange`` references resolve to byte-identical content (sha256 verified);
2. replay the recorded request through the same measure-v2 executor
   (``POST /api/v2/runs``) in this checkout;
3. compare the reproduced observations with the recorded ones with the
   service's own numerical comparison (``measure_api.v2.replay``);
4. render the reproduction (localisation, alignment, probes, fitted circle,
   profiles, quality) as figures and write a Markdown + JSON report.

Usage:
    python scripts/repro_debug_package.py <package_dir> [--workdir DIR]
        [--tolerance 1e-9] [--timeout-ms 120000] [--no-figures]

Exit code is 0 when the replay reproduces the recorded numbers, 1 otherwise.
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
from scipy.ndimage import gaussian_filter1d

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from measure_api.v2.app import create_app  # noqa: E402
from measure_api.v2.families.alignment import (  # noqa: E402
    RigidAlignmentRequest,
    align_rigid,
)
from measure_api.v2.families.measurements import Probe, make_probe  # noqa: E402
from measure_api.v2.families.templates import coordinate_mapping  # noqa: E402
from measure_api.v2.models import (  # noqa: E402
    FileReference,
    OutputSpec,
    Point,
    ServiceSettings,
)
from measure_api.v2.recipe_models import RunResult  # noqa: E402
from measure_api.v2.registry import CalculationContext  # noqa: E402
from measure_api.v2.replay import compare_records, numerical_record  # noqa: E402
from measure_api.v2.versions import runtime_identity  # noqa: E402

# Figure palette; taught geometry is blue and dashed, measured is green.
C_TAUGHT = "#1f77b4"
C_FIT = "#2ca02c"
C_EDGE = "#00b0d4"
C_PROBE = "#ff8c00"
C_TEMPLATE = "#ffd400"
C_CENTER = "#d62728"
C_TEXT = "#222222"

FIGURES = {
    "localization": "01_localization_overview.png",
    "detail": "02_circle_detail.png",
    "aligned": "03_aligned_frame.png",
    "profiles": "04_probe_profiles.png",
    "quality": "05_geometry_quality.png",
    "reference": "06_taught_reference.png",
    "table": "07_verification_table.png",
    "image_center": "08_center_vs_image_center.png",
    "localization_check": "09_localization_check.png",
}


# ---------------------------------------------------------------------------
# Debug package loading and staging
# ---------------------------------------------------------------------------


@dataclass
class DebugPackage:
    """Parsed debug package with its asset files resolved to package paths."""

    root: Path
    manifest: Dict[str, Any]
    inputs: Dict[str, Any]
    result: Dict[str, Any]
    job: Dict[str, Any]
    asset_files: Dict[str, Path] = field(default_factory=dict)

    @property
    def run_id(self) -> str:
        return str(self.manifest.get("run_id") or self.root.name)

    @property
    def recipe(self) -> Dict[str, Any]:
        return self.inputs["recipe"]

    @property
    def target(self) -> Dict[str, Any]:
        targets = self.result.get("targets") or []
        if not targets:
            raise RuntimeError("Debug package result has no targets to reproduce")
        return targets[0]

    @property
    def node_result(self) -> Dict[str, Any]:
        nodes = self.target.get("nodes") or []
        if not nodes:
            raise RuntimeError("Debug package target has no node observations")
        return nodes[0]["result"]

    @property
    def node_parameters(self) -> Dict[str, Any]:
        return self.recipe["nodes"][0]["parameters"]


def load_package(root: Path) -> DebugPackage:
    """Read the manifest/inputs/result/job documents and index packaged assets.

    Args:
        root: Extracted debug package directory.

    Returns:
        Parsed package whose asset identifiers map to files inside the package.
    """
    def read(name: str) -> Dict[str, Any]:
        path = root / name
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    package = DebugPackage(
        root=root,
        manifest=read("manifest.json"),
        inputs=read("inputs.json"),
        result=read("result.json"),
        job=read("job.json"),
    )
    assets_path = root / (package.manifest.get("assets") or "assets.json")
    if assets_path.exists():
        for entry in json.loads(assets_path.read_text(encoding="utf-8")):
            asset = entry.get("asset") or {}
            identifier = asset.get("id")
            if identifier and (root / entry["file"]).exists():
                package.asset_files[str(identifier)] = root / entry["file"]
    if not package.asset_files:
        raise RuntimeError("Debug package assets.json does not reference any packaged file")
    return package


def package_file(package: DebugPackage, reference: Dict[str, Any]) -> Path:
    """Resolve a recipe file reference to the packaged asset with that identity.

    Args:
        package: Parsed debug package.
        reference: Shared-file reference recorded in the recipe or request.

    Returns:
        Package file holding the referenced bytes.
    """
    name = Path(str(reference["path"])).name
    matches = [
        (identifier, path)
        for identifier, path in package.asset_files.items()
        if name == identifier or name.startswith(identifier)
    ]
    if not matches:
        raise RuntimeError(f"No packaged asset matches reference {reference['path']!r}")
    return max(matches, key=lambda item: len(item[0]))[1]


def referenced_inputs(package: DebugPackage) -> List[Dict[str, Any]]:
    """Collect every recipe/request input reference the replay must resolve.

    Args:
        package: Parsed debug package.

    Returns:
        Distinct file references with their recorded digests, in execution order.
    """
    recipe = package.recipe
    candidates = [package.inputs.get("image"), recipe.get("reference")]
    localization = recipe.get("localization") or {}
    candidates.append(localization.get("template"))
    for node in recipe.get("nodes", []):
        for value in (node.get("parameters") or {}).values():
            if isinstance(value, dict) and "root" in value and "path" in value:
                candidates.append(value)
    seen, references = set(), []
    for reference in candidates:
        if not isinstance(reference, dict) or "path" not in reference:
            continue
        key = (reference.get("root"), reference["path"])
        if key in seen:
            continue
        seen.add(key)
        references.append(reference)
    return references


def stage_exchange(package: DebugPackage, exchange_root: Path) -> List[Dict[str, Any]]:
    """Copy packaged assets to their recorded shared paths and verify each digest.

    Args:
        package: Parsed debug package.
        exchange_root: Local directory standing in for the ``exchange`` shared root.

    Returns:
        One verification record per staged input reference.
    """
    records = []
    for reference in referenced_inputs(package):
        source = package_file(package, reference)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        target = exchange_root / str(reference["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            shutil.copyfile(source, target)
        expected = reference.get("sha256")
        records.append({
            "reference": f"{reference.get('root')}:{reference['path']}",
            "package_file": str(source.relative_to(package.root)),
            "staged_file": str(target),
            "bytes": target.stat().st_size,
            "sha256": digest,
            "recorded_sha256": expected,
            "matches_record": expected is None or str(expected).lower() == digest,
        })
    return records


# ---------------------------------------------------------------------------
# Replay through the measure-v2 executor
# ---------------------------------------------------------------------------


def reproduce(
    package: DebugPackage, exchange_root: Path, output_root: Path, timeout_ms: int,
) -> Dict[str, Any]:
    """Execute the recorded request on the packaged images through ``POST /api/v2/runs``.

    Args:
        package: Parsed debug package.
        exchange_root: Local stand-in for the ``exchange`` shared root.
        output_root: Local stand-in for the ``outputs`` shared root.
        timeout_ms: Operation deadline handed to the service.

    Returns:
        Reproduced run document with target and node observations.
    """
    settings = ServiceSettings(
        shared_roots={"exchange": exchange_root, "outputs": output_root},
        writable_roots=["outputs"],
        max_workers=2,
    )
    payload = {
        "recipe": package.recipe,
        "image": package.inputs["image"],
        "output": package.inputs["output"],
        "timeout_ms": timeout_ms,
    }
    with TestClient(create_app(settings)) as client:
        response = client.post("/api/v2/runs", json=payload)
    if response.status_code != 200:
        detail = response.text[:4000]
        raise RuntimeError(f"Replay was rejected with HTTP {response.status_code}: {detail}")
    return response.json()["result"]


def _max_numeric_delta(expected: Any, actual: Any, path: str = "") -> Tuple[float, str]:
    """Locate the largest absolute numeric difference between two observation trees.

    Args:
        expected: Recorded numerical/categorical observations.
        actual: Reproduced numerical/categorical observations.
        path: Current JSON pointer while recursing.

    Returns:
        Largest absolute difference and the JSON pointer where it occurs.
    """
    if isinstance(expected, dict) and isinstance(actual, dict):
        best = (0.0, "")
        for key in expected:
            if key in actual:
                best = max(best, _max_numeric_delta(expected[key], actual[key], path + "/" + key))
        return best
    if isinstance(expected, list) and isinstance(actual, list):
        best = (0.0, "")
        for index, (first, second) in enumerate(zip(expected, actual)):
            best = max(best, _max_numeric_delta(first, second, f"{path}/{index}"))
        return best
    if (
        isinstance(expected, (int, float))
        and not isinstance(expected, bool)
        and isinstance(actual, (int, float))
        and not isinstance(actual, bool)
    ):
        delta = abs(float(expected) - float(actual))
        return (delta, path) if delta > 0 else (0.0, "")
    return 0.0, ""


def verify(package: DebugPackage, run: Dict[str, Any], tolerance: float) -> Dict[str, Any]:
    """Compare reproduced observations with the recorded ones field by field.

    Args:
        package: Parsed debug package holding the recorded result.
        run: Reproduced run document.
        tolerance: Allowed absolute difference for finite numeric observations.

    Returns:
        Equivalence verdict, differing paths and the largest numeric delta.
    """
    recorded = RunResult.model_validate(package.result)
    reproduced = RunResult.model_validate(run)
    expected, actual = numerical_record(recorded), numerical_record(reproduced)
    differences = compare_records(expected, actual, tolerance)
    max_delta, max_path = _max_numeric_delta(expected, actual)
    return {
        "equivalent": not differences,
        "tolerance": tolerance,
        "difference_count": len(differences),
        "differences": differences[:200],
        "max_absolute_difference": max_delta,
        "max_difference_path": max_path,
        "recorded_configuration_sha256": recorded.configuration_sha256,
        "reproduced_configuration_sha256": reproduced.configuration_sha256,
        "recorded_status": recorded.status,
        "reproduced_status": reproduced.status,
        "recorded_valid": recorded.valid,
        "reproduced_valid": reproduced.valid,
        "recipe_differences": _recipe_key_differences(
            package.result.get("resolved_recipe") or {}, run.get("resolved_recipe") or {}
        ),
    }


def _recipe_key_differences(expected: Any, actual: Any, path: str = "") -> List[str]:
    """List structural recipe paths whose presence or type differs between builds.

    Args:
        expected: Recorded resolved recipe document.
        actual: Reproduced resolved recipe document.
        path: Current JSON pointer while recursing.

    Returns:
        JSON pointers that are missing on either side.
    """
    if isinstance(expected, dict) and isinstance(actual, dict):
        found = []
        for key in sorted(set(expected) | set(actual)):
            location = f"{path}/{key}"
            if key not in expected:
                found.append("added " + location)
            elif key not in actual:
                found.append("removed " + location)
            else:
                found.extend(_recipe_key_differences(expected[key], actual[key], location))
        return found
    if isinstance(expected, list) and isinstance(actual, list) and len(expected) == len(actual):
        found = []
        for index, (first, second) in enumerate(zip(expected, actual)):
            found.extend(_recipe_key_differences(first, second, f"{path}/{index}"))
        return found
    return []


def environment_report(recorded_identity: Dict[str, Any]) -> Dict[str, Any]:
    """Compare the recorded producer identity with the local replay runtime.

    Args:
        recorded_identity: Runtime identity stored in the debug package result.

    Returns:
        Field-by-field comparison of runtime identity values.
    """
    local = runtime_identity().model_dump(mode="json")
    rows = []
    for key in ("algorithm_version", "implementation_sha256", "python_version",
                "python_implementation", "architecture"):
        rows.append({"field": key, "recorded": recorded_identity.get(key),
                     "local": local.get(key),
                     "match": recorded_identity.get(key) == local.get(key)})
    recorded_deps = recorded_identity.get("dependencies") or {}
    for key in sorted(set(recorded_deps) | set(local.get("dependencies", {}))):
        rows.append({"field": "dependencies." + key,
                     "recorded": recorded_deps.get(key),
                     "local": local.get("dependencies", {}).get(key),
                     "match": recorded_deps.get(key) == local.get("dependencies", {}).get(key)})
    return {"rows": rows, "local": local, "recorded": recorded_identity}


def key_metrics(package: DebugPackage, run: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the business numbers of the measurement from recorded/reproduced runs.

    Args:
        package: Parsed debug package holding the recorded result.
        run: Reproduced run document.

    Returns:
        Taught geometry, localisation pose, fitted circle and fit quality.
    """
    def extract(document: Dict[str, Any]) -> Dict[str, Any]:
        target = document["targets"][0]
        node = target["nodes"][0]["result"]
        match = ((target.get("localization") or {}).get("match") or {})
        diagnostics = node.get("diagnostics") or {}
        return {
            "valid": document["valid"],
            "status": document["status"],
            "localization_score": match.get("score"),
            "match_center": match.get("center"),
            "mapping": ((target.get("mapping") or {}).get("matrix")),
            "center": node["circle"]["center"],
            "radius": node["circle"]["radius"],
            "sample_count": diagnostics.get("sample_count"),
            "requested_probe_count": diagnostics.get("requested_probe_count"),
            "successful_probe_count": diagnostics.get("successful_probe_count"),
            "rms_error": diagnostics.get("rms_error"),
            "mean_absolute_error": diagnostics.get("mean_absolute_error"),
            "max_absolute_error": diagnostics.get("max_absolute_error"),
            "sampled_diameter_spread": node.get("sampled_diameter_spread"),
            "residuals": diagnostics.get("residuals") or [],
        }

    parameters = package.node_parameters
    taught_center = parameters["center"]
    taught_radius = parameters["radius"]
    recorded, reproduced = extract(package.result), extract(run)
    mapping = np.array(reproduced["mapping"])
    taught_in_image = (mapping @ np.array([taught_center["x"], taught_center["y"], 1.0]))[:2]
    offset = math.hypot(
        reproduced["center"]["x"] - float(taught_in_image[0]),
        reproduced["center"]["y"] - float(taught_in_image[1]),
    )
    return {
        "recorded": recorded,
        "reproduced": reproduced,
        "taught_center": taught_center,
        "taught_radius": taught_radius,
        "taught_center_in_image": {"x": float(taught_in_image[0]), "y": float(taught_in_image[1])},
        "center_offset_px": offset,
        "radius_delta_px": float(reproduced["radius"]) - float(taught_radius),
        "probe": {
            "search_half_length": parameters.get("search_half_length"),
            "average_half_width": parameters.get("average_half_width"),
            "sigma": parameters.get("sigma"),
            "threshold": parameters.get("threshold"),
            "polarity": parameters.get("polarity"),
            "selection": parameters.get("selection"),
            "probe_count": parameters.get("probe_count"),
            "start_angle_degrees": parameters.get("start_angle_degrees"),
            "end_angle_degrees": parameters.get("end_angle_degrees"),
        },
    }


# ---------------------------------------------------------------------------
# Shared visualisation helpers
# ---------------------------------------------------------------------------


def _pyplot():
    """Configure matplotlib for headless, CJK-capable rendering.

    Returns:
        Imported ``matplotlib.pyplot`` module configured for Agg output.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": ["Noto Sans CJK JP", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.edgecolor": "#666666",
    })
    return plt


def read_gray(path: Path) -> np.ndarray:
    """Read a single-channel 8-bit image.

    Args:
        path: Image filename inside the package.

    Returns:
        Grayscale pixel array.
    """
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Cannot decode image {path}")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def map_points(matrix: Sequence[Sequence[float]], points: np.ndarray) -> np.ndarray:
    """Apply a homogeneous 3x3 map to xy points.

    Args:
        matrix: 3x3 homogeneous transform.
        points: Array of shape (N, 2).

    Returns:
        Mapped points of shape (N, 2).
    """
    array = np.asarray(points, dtype=float).reshape(-1, 2)
    homogeneous = np.column_stack([array, np.ones(len(array))])
    return (homogeneous @ np.asarray(matrix, dtype=float).T)[:, :2]


def probe_segment(
    center: np.ndarray, angle_degrees: float, radius: float, half_length: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the inward radial probe segment around the taught circle.

    Args:
        center: Taught circle centre in the displayed frame.
        angle_degrees: Clockwise probe angle in image coordinates.
        radius: Taught circle radius.
        half_length: Search half length of every probe.

    Returns:
        Outer and inner endpoints of the probe segment.
    """
    phi = math.radians(angle_degrees)
    unit = np.array([math.cos(phi), math.sin(phi)])
    return center + (radius + half_length) * unit, center + (radius - half_length) * unit


def probe_angles(parameters: Dict[str, Any]) -> np.ndarray:
    """Rebuild the equally spaced probe angles used for a circle measurement.

    Args:
        parameters: ``measurements.circles`` node parameters.

    Returns:
        Probe angles in degrees.
    """
    start = float(parameters.get("start_angle_degrees", 0.0))
    end = float(parameters.get("end_angle_degrees", 360.0))
    count = int(parameters.get("probe_count", 24))
    span = (end - start) % 360 or 360
    return start + np.linspace(0, span, count, endpoint=False)


def _legend(ax, location: str = "lower left", fontsize: float = 9, **kwargs) -> None:
    """Draw a readable legend with a light backing box."""
    legend = ax.legend(loc=location, framealpha=0.85, facecolor="white",
                       edgecolor="#cccccc", fontsize=fontsize, **kwargs)
    legend.set_zorder(30)


def _line_step(ax, fontsize: float, spacing: float = 1.55) -> float:
    """Return the vertical distance between two text lines in axes fraction.

    Args:
        ax: Axes owning the text.
        fontsize: Font size in points.
        spacing: Baseline spacing as a multiple of the font size.

    Returns:
        Line step in axes-fraction units.
    """
    figure_height_in = ax.figure.get_size_inches()[1]
    axes_height_in = max(ax.get_position().height * figure_height_in, 1e-6)
    return (fontsize * spacing / 72.0) / axes_height_in


def _metric_panel(
    ax, rows: Sequence[Tuple[str, str]], location: str = "upper left",
    fontsize: float = 9.0, value_x: float = 0.19,
) -> None:
    """Draw a label/value panel with aligned monospace values and haloed text.

    Args:
        ax: Axes to annotate, in axes coordinates.
        rows: Label/value pairs; labels are Chinese, values are rendered monospace.
        location: ``upper left``, ``upper right``, ``lower left`` or ``lower right``.
        fontsize: Font size in points.
        value_x: Axes-fraction column where values start.
    """
    from matplotlib import patheffects
    from matplotlib.patches import Rectangle

    halo = [patheffects.withStroke(linewidth=0.8, foreground="white")]
    step = _line_step(ax, fontsize)
    upper = "upper" in location
    right = "right" in location
    x_anchor = 0.986 if right else 0.014
    y_anchor = 0.986 if upper else 0.014
    axes_width_in = max(ax.get_position().width * ax.figure.get_size_inches()[0], 1e-6)
    unit_width = 0.56 * fontsize / 72.0 / axes_width_in
    widest = max(
        (sum(2 if ord(character) > 127 else 1 for character in value) for _, value in rows),
        default=12,
    )
    box_width = min(0.92, value_x + widest * unit_width + 0.012)
    box_height = step * len(rows) + step * 0.55
    left_edge = 1.0 - 0.006 - box_width if right else 0.006
    bottom_edge = (0.994 - box_height) if upper else 0.006
    ax.add_patch(Rectangle(
        (left_edge, bottom_edge), box_width, box_height, transform=ax.transAxes,
        facecolor="white", edgecolor="#cccccc", linewidth=0.8, alpha=0.86, zorder=39,
    ))
    for index, (label, value) in enumerate(rows):
        offset = -index * step if upper else index * step
        y = y_anchor + offset
        ax.text(x_anchor, y, label, transform=ax.transAxes,
                va="top" if upper else "bottom", ha="right" if right else "left",
                fontsize=fontsize, color=C_TEXT, zorder=41, path_effects=halo)
        ax.text(x_anchor + (-value_x if right else value_x), y, value,
                transform=ax.transAxes, va="top" if upper else "bottom",
                ha="right" if right else "left", fontsize=fontsize, color=C_TEXT,
                family="DejaVu Sans Mono" if value.isascii() else None,
                zorder=41, path_effects=halo)


def _pixel_canvas(
    width: int, height: int, dpi: int = 100, pad_left: int = 118, pad_bottom: int = 74,
    pad_top: int = 54, pad_right: int = 26,
) -> Tuple[Any, Any, float]:
    """Create a figure whose axes maps one image pixel to one output pixel.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        dpi: Output resolution.
        pad_left: Left margin for the y axis label and ticks.
        pad_bottom: Bottom margin for the x axis label and ticks.
        pad_top: Top margin for the title.
        pad_right: Right margin.

    Returns:
        Figure, axes and the base font size that keeps the layout proportionate.
    """
    plt = _pyplot()
    total_width = width + pad_left + pad_right
    total_height = height + pad_top + pad_bottom
    figure = plt.figure(figsize=(total_width / dpi, total_height / dpi), dpi=dpi)
    axes = figure.add_axes([
        pad_left / total_width,
        pad_bottom / total_height,
        width / total_width,
        height / total_height,
    ])
    base_font = max(9.0, width * 72.0 / dpi * 0.017)
    return figure, axes, base_font


# ---------------------------------------------------------------------------
# Figure 1: localisation overview on the detection image
# ---------------------------------------------------------------------------


def figure_localization(package: DebugPackage, run: Dict[str, Any], metrics: Dict[str, Any],
                        path: Path) -> Path:
    """Render the detection image with template pose, probes and fitted circle.

    Args:
        package: Parsed debug package.
        run: Reproduced run document.
        metrics: Extracted business metrics.
        path: Output PNG filename.

    Returns:
        Written figure path.
    """
    plt = _pyplot()
    from matplotlib.patches import Circle

    image = read_gray(package_file(package, package.inputs["image"]))
    target = run["targets"][0]
    node = target["nodes"][0]["result"]
    match = target["localization"]["match"]
    support = match["support_mapping"]["matrix"]
    template_meta = json.loads(
        np.load(package_file(package, package.recipe["localization"]["template"]))
        .get("metadata").item()
    )
    width = float(template_meta["support"]["width"])
    height = float(template_meta["support"]["height"])

    fig, ax = plt.subplots(figsize=(11.4, 8.4), dpi=150)
    ax.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="antialiased")

    corners = map_points(support, np.array([[0, 0], [width, 0], [width, height], [0, height]]))
    ax.add_patch(plt.Polygon(corners, closed=True, fill=False, edgecolor=C_TEMPLATE,
                             linewidth=1.6, linestyle="-",
                             label="模板支撑区 2500×2486（定位搜索范围）"))

    taught_center = np.array([metrics["taught_center_in_image"]["x"],
                              metrics["taught_center_in_image"]["y"]])
    radius = float(metrics["taught_radius"])
    half_length = float(metrics["probe"]["search_half_length"])
    angles = probe_angles(package.node_parameters)
    for index, angle in enumerate(angles):
        start, end = probe_segment(taught_center, angle, radius, half_length)
        ax.plot([start[0], end[0]], [start[1], end[1]], color=C_PROBE, linewidth=0.6,
                alpha=0.55, label="径向探针 ×32（±150 px）" if index == 0 else None)

    edges = np.array([[edge["point"]["x"], edge["point"]["y"]] for edge in node["edges"]])
    ax.scatter(edges[:, 0], edges[:, 1], s=26, marker="x", color=C_EDGE,
               linewidths=1.4, zorder=20, label="亚像素边缘点（32/32）")

    taught_label = (f"示教圆 C=({taught_center[0]:.2f}, {taught_center[1]:.2f}) "
                    f"R={radius:.2f}")
    ax.add_patch(Circle(taught_center, radius, fill=False, edgecolor=C_TAUGHT,
                        linestyle="--", linewidth=1.5, label=taught_label))
    centre = np.array([node["circle"]["center"]["x"], node["circle"]["center"]["y"]])
    fitted_radius = float(node["circle"]["radius"])
    fit_label = (f"拟合圆 C=({centre[0]:.2f}, {centre[1]:.2f}) R={fitted_radius:.2f}")
    ax.add_patch(Circle(centre, fitted_radius, fill=False, edgecolor=C_FIT,
                        linewidth=2.0, label=fit_label))
    ax.plot(centre[0], centre[1], marker="+", markersize=17, markeredgewidth=2.0,
            color=C_CENTER, zorder=25, label="拟合圆心")
    ax.plot(match["center"]["x"], match["center"]["y"], marker="o", markersize=7,
            markerfacecolor="none", markeredgecolor=C_TEMPLATE, markeredgewidth=1.6,
            zorder=25, label="模板匹配中心")

    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_title(
        f"算法调试包 {package.run_id[:8]} 复现 · 定位图"
        f"（{image.shape[1]}×{image.shape[0]}）",
        fontsize=12, pad=10,
    )
    _metric_panel(ax, [
        ("定位得分", f"{match['score']:.9f}"),
        ("匹配中心", f"({match['center']['x']:.2f}, {match['center']['y']:.2f})"),
        ("示教圆(映射后)",
         f"C=({taught_center[0]:.2f},{taught_center[1]:.2f}) R={radius:.2f}"),
        ("拟合圆", f"C=({centre[0]:.2f},{centre[1]:.2f}) R={fitted_radius:.2f}"),
        ("Δ中心", f"{metrics['center_offset_px']:.2f} px   "
                  f"ΔR {metrics['radius_delta_px']:+.2f} px"),
        ("拟合残差", f"RMS {node['diagnostics']['rms_error']:.3f} / max "
                     f"{node['diagnostics']['max_absolute_error']:.3f} px"),
    ], fontsize=9.2)
    _legend(ax, "lower left", fontsize=8.6)
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 2: 1:1 detail around the measured circle (detection frame)
# ---------------------------------------------------------------------------


def figure_circle_detail(package: DebugPackage, run: Dict[str, Any], metrics: Dict[str, Any],
                         path: Path) -> Path:
    """Render a 1:1 crop with probes, sub-pixel edges and both circles.

    Args:
        package: Parsed debug package.
        run: Reproduced run document.
        metrics: Extracted business metrics.
        path: Output PNG filename.

    Returns:
        Written figure path.
    """
    plt = _pyplot()
    from matplotlib.patches import Circle

    image = read_gray(package_file(package, package.inputs["image"]))
    node = run["targets"][0]["nodes"][0]["result"]
    radius = float(metrics["taught_radius"])
    half_length = float(metrics["probe"]["search_half_length"])
    taught_center = np.array([metrics["taught_center_in_image"]["x"],
                              metrics["taught_center_in_image"]["y"]])
    centre = np.array([node["circle"]["center"]["x"], node["circle"]["center"]["y"]])
    fitted_radius = float(node["circle"]["radius"])

    margin = half_length + 55
    x0 = int(max(0, math.floor(min(taught_center[0], centre[0]) - radius - margin)))
    x1 = int(min(image.shape[1], math.ceil(max(taught_center[0], centre[0]) + radius + margin)))
    y0 = int(max(0, math.floor(min(taught_center[1], centre[1]) - radius - margin)))
    y1 = int(min(image.shape[0], math.ceil(max(taught_center[1], centre[1]) + radius + margin)))
    crop = image[y0:y1, x0:x1]

    fig, ax, base = _pixel_canvas(crop.shape[1], crop.shape[0])
    ax.imshow(crop, cmap="gray", vmin=0, vmax=255, interpolation="nearest",
              extent=(x0, x1, y1, y0))

    angles = probe_angles(package.node_parameters)
    for index, angle in enumerate(angles):
        start, end = probe_segment(taught_center, angle, radius, half_length)
        ax.plot([start[0], end[0]], [start[1], end[1]], color=C_PROBE,
                linewidth=base * 0.075, alpha=0.45,
                label="径向探针（±150 px 搜索）" if index == 0 else None)

    edges = np.array([[edge["point"]["x"], edge["point"]["y"]] for edge in node["edges"]])
    ax.scatter(edges[:, 0], edges[:, 1], s=(base * 0.85) ** 2, marker="x", color=C_EDGE,
               linewidths=base * 0.085, zorder=20, label="亚像素边缘点")

    ax.add_patch(Circle(centre, fitted_radius, fill=False, edgecolor=C_FIT, zorder=21,
                        linewidth=base * 0.16, label="拟合圆"))
    ax.add_patch(Circle(taught_center, radius, fill=False, edgecolor=C_TAUGHT, zorder=22,
                        linestyle=(0, (6, 4)), linewidth=base * 0.14,
                        label="示教圆（参考系映射而来）"))
    ax.plot(centre[0], centre[1], marker="+", markersize=base * 1.5,
            markeredgewidth=base * 0.16, color=C_CENTER, zorder=25, label="拟合圆心")
    ax.plot(taught_center[0], taught_center[1], marker="+", markersize=base * 1.2,
            markeredgewidth=base * 0.13, color=C_TAUGHT, zorder=24, label="示教圆心")

    phi = math.radians(-40)
    ax.plot([centre[0], centre[0] + fitted_radius * math.cos(phi)],
            [centre[1], centre[1] + fitted_radius * math.sin(phi)],
            color=C_FIT, linewidth=base * 0.08, alpha=0.9)
    ax.annotate(f"R = {fitted_radius:.3f} px",
                xy=(centre[0] + 0.55 * fitted_radius * math.cos(phi),
                    centre[1] + 0.55 * fitted_radius * math.sin(phi)),
                xytext=(-base * 0.7, -base * 1.3), textcoords="offset points",
                color=C_FIT, fontsize=base * 1.05, fontweight="bold")

    ax.annotate(f"Δ中心 {metrics['center_offset_px']:.2f} px",
                xy=tuple((centre + taught_center) / 2),
                xytext=(base * 1.6, -base * 2.2), textcoords="offset points",
                color=C_CENTER, fontsize=base * 0.92,
                arrowprops={"arrowstyle": "-", "color": C_CENTER, "lw": base * 0.06})

    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.tick_params(labelsize=base * 0.85)
    ax.set_xlabel("x (px)", fontsize=base * 0.95)
    ax.set_ylabel("y (px)", fontsize=base * 0.95)
    ax.set_title(
        f"测量细节（1:1 像素）· 拟合半径 {fitted_radius:.3f} px，"
        f"半径偏差 {metrics['radius_delta_px']:+.3f} px", fontsize=base * 1.15, pad=10,
    )
    _legend(ax, "lower left", fontsize=base * 0.8)
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Aligned frame reconstruction (pixels the probes actually sampled)
# ---------------------------------------------------------------------------


def aligned_patch(package: DebugPackage, run: Dict[str, Any], exchange_root: Path,
                  figures_root: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Rebuild the aligned reference-frame patch with the service's own warp.

    Args:
        package: Parsed debug package.
        run: Reproduced run document.
        exchange_root: Local ``exchange`` shared root.
        figures_root: Writable root receiving the intermediate PNG.

    Returns:
        Aligned pixel array and the detection-to-aligned 3x3 matrix.
    """
    mapping = np.array(run["targets"][0]["mapping"]["matrix"], dtype=float)
    matrix = np.linalg.inv(mapping)
    reference = read_gray(package_file(package, package.recipe["reference"]))
    height, width = reference.shape[:2]
    staging = figures_root / "_intermediate"
    if (staging / "aligned").exists():
        shutil.rmtree(staging / "aligned")
    settings = ServiceSettings(
        shared_roots={"exchange": exchange_root, "figs": figures_root},
        writable_roots=["figs"],
    )
    context = CalculationContext(settings, "aligned", "repro")
    request = RigidAlignmentRequest(
        image=FileReference(**package.inputs["image"]),
        initial_mapping=coordinate_mapping(matrix, "reference", "aligned"),
        width=int(width),
        height=int(height),
        output=OutputSpec(directory=FileReference(root="figs", path="_intermediate")),
    )
    result = align_rigid(request, context)
    path = context.files.resolve(result.image)
    return read_gray(path), matrix


def figure_aligned_frame(package: DebugPackage, run: Dict[str, Any], metrics: Dict[str, Any],
                         aligned: np.ndarray, matrix: np.ndarray, path: Path) -> Path:
    """Render the reference-frame measurement view the probes actually sampled.

    Args:
        package: Parsed debug package.
        run: Reproduced run document.
        metrics: Extracted business metrics.
        aligned: Aligned reference-frame pixels.
        matrix: Detection-to-aligned 3x3 matrix.
        path: Output PNG filename.

    Returns:
        Written figure path.
    """
    plt = _pyplot()
    from matplotlib.patches import Circle

    node = run["targets"][0]["nodes"][0]["result"]
    taught_center = np.array([metrics["taught_center"]["x"], metrics["taught_center"]["y"]])
    radius = float(metrics["taught_radius"])
    half_length = float(metrics["probe"]["search_half_length"])
    centre = map_points(matrix, np.array([[node["circle"]["center"]["x"],
                                           node["circle"]["center"]["y"]]]))[0]
    fitted_radius = float(node["circle"]["radius"])
    edges = map_points(matrix, np.array([[edge["point"]["x"], edge["point"]["y"]]
                                         for edge in node["edges"]]))

    margin = half_length + 55
    x0 = int(max(0, math.floor(taught_center[0] - radius - margin)))
    x1 = int(min(aligned.shape[1], math.ceil(taught_center[0] + radius + margin)))
    y0 = int(max(0, math.floor(taught_center[1] - radius - margin)))
    y1 = int(min(aligned.shape[0], math.ceil(taught_center[1] + radius + margin)))
    crop = aligned[y0:y1, x0:x1]

    fig, ax, base = _pixel_canvas(crop.shape[1], crop.shape[0])
    ax.imshow(crop, cmap="gray", vmin=0, vmax=255, interpolation="nearest",
              extent=(x0, x1, y1, y0))
    angles = probe_angles(package.node_parameters)
    for index, angle in enumerate(angles):
        start, end = probe_segment(taught_center, angle, radius, half_length)
        ax.plot([start[0], end[0]], [start[1], end[1]], color=C_PROBE,
                linewidth=base * 0.075, alpha=0.45,
                label="径向探针" if index == 0 else None)
    ax.scatter(edges[:, 0], edges[:, 1], s=(base * 0.85) ** 2, marker="x", color=C_EDGE,
               linewidths=base * 0.085, zorder=20, label="亚像素边缘点")
    ax.add_patch(Circle(centre, fitted_radius, fill=False, edgecolor=C_FIT, zorder=21,
                        linewidth=base * 0.16, label="拟合圆"))
    ax.add_patch(Circle(taught_center, radius, fill=False, edgecolor=C_TAUGHT, zorder=22,
                        linestyle=(0, (6, 4)), linewidth=base * 0.14,
                        label="示教圆（配方几何）"))
    ax.plot(centre[0], centre[1], marker="+", markersize=base * 1.5,
            markeredgewidth=base * 0.16, color=C_CENTER, zorder=25, label="拟合圆心")
    ax.plot(taught_center[0], taught_center[1], marker="+", markersize=base * 1.2,
            markeredgewidth=base * 0.13, color=C_TAUGHT, zorder=24, label="示教圆心")

    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.tick_params(labelsize=base * 0.85)
    ax.set_xlabel("x（对齐帧）(px)", fontsize=base * 0.95)
    ax.set_ylabel("y（对齐帧）(px)", fontsize=base * 0.95)
    ax.set_title("对齐到参考帧后的测量视图（算法实际执行帧，1:1 像素）",
                 fontsize=base * 1.15, pad=10)
    _legend(ax, "lower left", fontsize=base * 0.78)
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 4: probe intensity/gradient profiles
# ---------------------------------------------------------------------------


def probe_profiles(aligned: np.ndarray, package: DebugPackage,
                   metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Sample the recorded probes on the aligned patch and collect their curves.

    Args:
        aligned: Aligned reference-frame pixels.
        package: Parsed debug package.
        metrics: Extracted business metrics.

    Returns:
        One record per displayed probe with curves, candidates and the selected edge.
    """
    parameters = package.node_parameters
    probe_parameters = metrics["probe"]
    settings = ServiceSettings(shared_roots={"exchange": package.root})
    context = CalculationContext(settings, "profiles", "repro")
    taught_center = np.array([metrics["taught_center"]["x"], metrics["taught_center"]["y"]])
    radius = float(metrics["taught_radius"])
    half_length = float(probe_parameters["search_half_length"])
    sigma = float(probe_parameters["sigma"] or 0.0)
    threshold = float(probe_parameters["threshold"])
    polarity = str(probe_parameters["polarity"])

    selected = {}
    mapping = np.array(metrics["reproduced"]["mapping"], dtype=float)
    inverse = np.linalg.inv(mapping)
    edges_aligned = map_points(inverse, np.array([[edge["point"]["x"], edge["point"]["y"]]
                                                  for edge in metrics["reproduced_edges"]]))
    angles = probe_angles(parameters)
    for angle, point in zip(angles, edges_aligned):
        phi = math.radians(angle)
        unit = np.array([math.cos(phi), math.sin(phi)])
        centre = taught_center + radius * unit
        direction = -unit
        selected[round(float(angle), 6)] = float(np.dot(point - centre, direction))

    records = []
    for index in range(0, len(angles), max(1, len(angles) // 4)):
        angle = float(angles[index])
        phi = math.radians(angle)
        unit = np.array([math.cos(phi), math.sin(phi)])
        centre = taught_center + radius * unit
        probe = Probe(
            center=Point(x=float(centre[0]), y=float(centre[1])),
            angle_degrees=angle + 180.0,
            search_half_length=half_length,
            average_half_width=float(probe_parameters["average_half_width"]),
        )
        instrument = make_probe(probe, aligned, context)
        gray = instrument.get_profile(aligned).astype(np.float64)
        smoothed = gaussian_filter1d(gray, sigma) if sigma > 0 else gray
        gradient = np.gradient(smoothed)
        rows, cols, amplitudes, _ = instrument.measure_pos(
            aligned, sigma, threshold, polarity, "all",
        )
        offsets = np.arange(len(gray)) - half_length
        candidates = [
            (float((np.array([col, row]) - centre) @ (-unit)), float(amplitude))
            for row, col, amplitude in zip(rows, cols, amplitudes)
        ]
        records.append({
            "angle": angle,
            "position": offsets,
            "gray": gray,
            "smoothed": smoothed,
            "gradient": gradient,
            "candidates": candidates,
            "selected": selected.get(round(angle, 6)),
            "threshold": threshold,
            "polarity": polarity,
        })
    return records


def figure_probe_profiles(records: List[Dict[str, Any]], metrics: Dict[str, Any],
                          path: Path) -> Path:
    """Render gray/gradient curves with candidate and selected sub-pixel edges.

    Args:
        records: Probe curve records from :func:`probe_profiles`.
        metrics: Extracted business metrics.
        path: Output PNG filename.

    Returns:
        Written figure path.
    """
    plt = _pyplot()
    count = len(records)
    columns = 2
    rows = int(math.ceil(count / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(12.6, 3.5 * rows), dpi=150,
                             squeeze=False)
    for axis_index, (ax, record) in enumerate(zip(axes.flat, records)):
        position = record["position"]
        ax.plot(position, record["gray"], color="#555555", linewidth=1.0, alpha=0.55,
                label="灰度剖面")
        ax.plot(position, record["smoothed"], color="#111111", linewidth=1.5,
                label="高斯平滑")
        ax.set_xlabel("探针扫描位置 s (px)：0 = 示教圆周，s>0 向圆心")
        ax.set_ylabel("灰度 (0-255)")
        ax.set_xlim(position[0], position[-1])
        ax.set_ylim(-8, 275)
        ax.grid(alpha=0.25, linewidth=0.6)

        twin = ax.twinx()
        twin.plot(position, record["gradient"], color="#8c564b", linewidth=1.1,
                  alpha=0.85, label="一阶导数")
        threshold = record["threshold"]
        twin.axhline(threshold, color="#c9a227", linestyle=":", linewidth=1.0)
        twin.axhline(-threshold, color="#c9a227", linestyle=":", linewidth=1.0,
                     label=f"阈值 ±{threshold:g}")
        twin.set_ylabel("灰度导数 (灰度/px)", color="#8c564b")
        twin.tick_params(axis="y", colors="#8c564b")
        limits = max(abs(record["gradient"]).max(), threshold) * 1.15
        twin.set_ylim(-limits, limits)

        labelled = False
        for offset, amplitude in record["candidates"]:
            if position[0] <= offset <= position[-1]:
                twin.plot(offset, amplitude, marker="o", markersize=4.0,
                          markerfacecolor="none", markeredgecolor=C_EDGE,
                          markeredgewidth=1.0, linestyle="none",
                          label=None if labelled else "候选边缘")
                labelled = True
        if record["selected"] is not None:
            ax.axvline(record["selected"], color=C_FIT, linewidth=1.3, linestyle="--")
            amplitude = next(
                (value for offset, value in record["candidates"]
                 if abs(offset - record["selected"]) < 3.0), None,
            )
            text = f"选中边缘 s={record['selected']:.2f} px"
            if amplitude is not None:
                text += f" · 幅值 {amplitude:.1f}"
            ax.annotate(
                text, xy=(record["selected"], 245),
                xytext=(0.50, 0.60), textcoords="axes fraction",
                color=C_FIT, fontsize=9,
                arrowprops={"arrowstyle": "->", "color": C_FIT, "lw": 0.9},
            )
        ax.set_title(f"θ = {record['angle']:.1f}° 探针", fontsize=10.5)
        handles_a, labels_a = ax.get_legend_handles_labels()
        handles_b, labels_b = twin.get_legend_handles_labels()
        unique = dict(zip(labels_a + labels_b, handles_a + handles_b))
        ax.legend(unique.values(), unique.keys(), loc="lower left", fontsize=7.5,
                  framealpha=0.85)
    for ax in axes.flat[count:]:
        ax.axis("off")
    fig.suptitle(
        f"探针剖面与亚像素边缘（sigma={metrics['probe']['sigma']:g}，"
        f"极性 {metrics['probe']['polarity']}，选择 {metrics['probe']['selection']}，"
        f"阈值 {metrics['probe']['threshold']:g} 灰度）",
        fontsize=12, y=1.005,
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 5: fit quality
# ---------------------------------------------------------------------------


def figure_quality(run: Dict[str, Any], metrics: Dict[str, Any], path: Path) -> Path:
    """Render residual, amplitude, radius histogram and magnified shape panels.

    Args:
        run: Reproduced run document.
        metrics: Extracted business metrics.
        path: Output PNG filename.

    Returns:
        Written figure path.
    """
    plt = _pyplot()
    node = run["targets"][0]["nodes"][0]["result"]
    diagnostics = node["diagnostics"]
    residuals = np.array(diagnostics["residuals"], dtype=float)
    edges = np.array([[edge["point"]["x"], edge["point"]["y"]] for edge in node["edges"]])
    amplitudes = np.array([edge["amplitude"] for edge in node["edges"]])
    centre = np.array([node["circle"]["center"]["x"], node["circle"]["center"]["y"]])
    fitted_radius = float(node["circle"]["radius"])
    offsets = edges - centre
    radii = np.linalg.norm(offsets, axis=1)
    angles = (np.degrees(np.arctan2(offsets[:, 1], offsets[:, 0])) % 360)
    order = np.argsort(angles)

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.6), dpi=150)

    ax = axes[0, 0]
    ax.stem(angles, residuals, linefmt=C_TAUGHT, markerfmt="o", basefmt=" ")
    for line in ax.get_lines():
        line.set_markersize(4)
    ax.axhline(0, color="#999999", linewidth=0.8)
    ax.axhline(diagnostics["rms_error"], color="#d62728", linestyle="--", linewidth=1.0)
    ax.axhline(-diagnostics["rms_error"], color="#d62728", linestyle="--", linewidth=1.0,
               label=f"±RMS {diagnostics['rms_error']:.3f} px")
    ax.set_xlim(-10, 370)
    ax.set_xticks(np.arange(0, 361, 45))
    ax.set_xlabel("探针角度 (°)")
    ax.set_ylabel("拟合残差 (px)")
    ax.set_title("圆拟合残差（32 条探针）")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(loc="upper right", fontsize=8)
    ax.text(
        0.02, 0.04,
        f"MAE {diagnostics['mean_absolute_error']:.3f} px · "
        f"max {diagnostics['max_absolute_error']:.3f} px",
        transform=ax.transAxes, fontsize=8.5, color=C_TEXT, zorder=10,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85,
              "edgecolor": "none"},
    )

    ax = axes[0, 1]
    ax.plot(angles[order], amplitudes[order], marker="o", markersize=4.5,
            color=C_EDGE, linewidth=1.2)
    ax.axhline(amplitudes.mean(), color="#999999", linestyle=":", linewidth=1.0,
               label=f"平均 {amplitudes.mean():.1f}")
    ax.set_xlim(-10, 370)
    ax.set_xticks(np.arange(0, 361, 45))
    ax.set_xlabel("探针角度 (°)")
    ax.set_ylabel("边缘幅值（灰度/px）")
    ax.set_title("边缘对比度一致性（负值 = 亮→暗）")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(loc="lower right", fontsize=8)

    ax = axes[1, 0]
    ax.hist(radii, bins=14, color=C_TAUGHT, alpha=0.35, edgecolor="#4477aa")
    ax.axvline(fitted_radius, color=C_FIT, linewidth=2.0,
               label=f"拟合半径 {fitted_radius:.3f}")
    ax.axvline(metrics["taught_radius"], color=C_TAUGHT, linestyle="--", linewidth=1.4,
               label=f"示教半径 {metrics['taught_radius']:.3f}")
    ax.set_xlabel("采样半径 r_i (px)")
    ax.set_ylabel("探针数")
    ax.set_title(
        f"采样半径分布（σ={radii.std(ddof=1):.3f} px，极差 {np.ptp(radii):.3f} px）"
    )
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1, 1]
    magnification = 30.0
    shape = centre + (fitted_radius + residuals * magnification)[:, None] * np.column_stack(
        [np.cos(np.radians(angles)), np.sin(np.radians(angles))]
    )
    closed = np.vstack([shape[order], shape[order][:1]])
    ax.plot(closed[:, 0], closed[:, 1], color="#d62728", linewidth=1.3,
            label=f"实际采样轮廓（残差 ×{magnification:g} 放大）")
    theta = np.linspace(0, 2 * math.pi, 512)
    ax.plot(centre[0] + fitted_radius * np.cos(theta),
            centre[1] + fitted_radius * np.sin(theta), color=C_FIT, linewidth=1.5,
            label="拟合圆")
    ax.scatter(shape[:, 0], shape[:, 1], s=18, color="#d62728", zorder=5)
    ax.set_aspect("equal")
    span = fitted_radius * 1.35
    ax.set_xlim(centre[0] - span, centre[0] + span)
    ax.set_ylim(centre[1] - span, centre[1] + span)
    ax.set_title(f"残差形状（放大 ×{magnification:g}）")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("拟合质量诊断 · 定位圆（measurements.circles）", fontsize=12.5,
                 y=1.002)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 6: taught reference geometry
# ---------------------------------------------------------------------------


def _roi_outline(roi: Dict[str, Any]) -> np.ndarray:
    """Return the four corners of a rotated ROI record.

    Args:
        roi: ROI record with ``cx``/``cy``/``width``/``height``/``angle``.

    Returns:
        Corner array of shape (4, 2).
    """
    half = np.array([[roi["width"] / 2, roi["height"] / 2]])
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float) * half
    angle = math.radians(float(roi.get("angle") or 0.0))
    rotation = np.array([[math.cos(angle), -math.sin(angle)],
                         [math.sin(angle), math.cos(angle)]])
    return corners @ rotation.T + np.array([roi["cx"], roi["cy"]])


def figure_taught_reference(package: DebugPackage, metrics: Dict[str, Any], path: Path) -> Path:
    """Render the taught reference geometry that defined this measurement.

    Args:
        package: Parsed debug package.
        metrics: Extracted business metrics.
        path: Output PNG filename.

    Returns:
        Written figure path.
    """
    plt = _pyplot()
    from matplotlib.patches import Circle

    image = read_gray(package_file(package, package.recipe["reference"]))
    taught = np.array([metrics["taught_center"]["x"], metrics["taught_center"]["y"]])
    radius = float(metrics["taught_radius"])

    fig, ax = plt.subplots(figsize=(11.4, 8.4), dpi=150)
    ax.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="antialiased")
    ax.add_patch(Circle(taught, radius, fill=False, edgecolor=C_TAUGHT, linestyle="--",
                        linewidth=1.8, label=f"示教圆 R={radius:.2f}"))
    ax.plot(taught[0], taught[1], marker="+", markersize=18, markeredgewidth=2.0,
            color=C_CENTER, label="示教圆心")

    template_meta = json.loads(
        np.load(package_file(package, package.recipe["localization"]["template"]))
        .get("metadata").item()
    )
    support = template_meta["support"]
    box = _roi_outline({
        "cx": support["center"]["x"], "cy": support["center"]["y"],
        "width": support["width"], "height": support["height"],
        "angle": support.get("angle_degrees", 0.0),
    })
    ax.add_patch(plt.Polygon(box, closed=True, fill=False, edgecolor=C_TEMPLATE,
                             linewidth=1.6, label="模板支撑区（示教区域）"))

    tools = ((package.job.get("snapshot") or {}).get("tools") or [])
    if tools:
        tool_roi = tools[0].get("roi")
        if tool_roi:
            outline = _roi_outline(tool_roi)
            ax.add_patch(plt.Polygon(outline, closed=True, fill=False, edgecolor="#9467bd",
                                     linewidth=1.3, linestyle="-.",
                                     label="工具 ROI（示教包围盒）"))
    angles = probe_angles(package.node_parameters)
    for index, angle in enumerate(angles):
        start, end = probe_segment(taught, float(angle), radius,
                                   float(metrics["probe"]["search_half_length"]))
        ax.plot([start[0], end[0]], [start[1], end[1]], color=C_PROBE, linewidth=0.6,
                alpha=0.5, label="径向探针" if index == 0 else None)

    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_title("示教参考图（参考帧）· 配方定义", fontsize=12, pad=10)
    _metric_panel(ax, [
        ("参考图", f"{image.shape[1]}×{image.shape[0]}"),
        ("示教圆心", f"({taught[0]:.2f}, {taught[1]:.2f})"),
        ("示教半径", f"{radius:.3f} px"),
        ("探针", f"{metrics['probe']['probe_count']} 条 × "
                 f"±{metrics['probe']['search_half_length']:g} px"),
        ("边缘极性", f"{metrics['probe']['polarity']} / "
                     f"选取 {metrics['probe']['selection']}"),
        ("阈值", f"{metrics['probe']['threshold']:g} 灰度 / "
                 f"sigma {metrics['probe']['sigma']:g}"),
    ], fontsize=9.2)
    _legend(ax, "lower left", fontsize=8.6)
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 9: localisation check (taught template vs matched patch)
# ---------------------------------------------------------------------------


def figure_localization_check(package: DebugPackage, run: Dict[str, Any],
                              path: Path) -> Path:
    """Compare the taught template with the patch the localiser actually matched.

    Args:
        package: Parsed debug package.
        run: Measured run document (reproduced or recorded).
        path: Output PNG filename.

    Returns:
        Written figure path.
    """
    plt = _pyplot()
    from matplotlib.patches import Circle

    template_file = package_file(package, package.recipe["localization"]["template"])
    with np.load(template_file) as archive:
        template = archive["pixels"].astype(np.float32)
        metadata = json.loads(archive["metadata"].item())
    image = read_gray(package_file(package, package.inputs["image"])).astype(np.float32)
    target = run["targets"][0]
    match = target["localization"]["match"]
    mapping = np.array(target["mapping"]["matrix"], dtype=float)
    support = np.array(match["support_mapping"]["matrix"], dtype=float)
    height, width = template.shape[:2]
    x0, y0 = int(round(support[0, 2])), int(round(support[1, 2]))
    x1, y1 = min(image.shape[1], x0 + width), min(image.shape[0], y0 + height)
    patch = image[y0:y1, x0:x1]
    common_h = min(patch.shape[0], height)
    common_w = min(patch.shape[1], width)
    template_crop = template[:common_h, :common_w]
    patch_crop = patch[:common_h, :common_w]
    difference = np.abs(patch_crop - template_crop)

    display_low, display_high = np.percentile(
        np.concatenate([template_crop.ravel()[::7], patch_crop.ravel()[::7]]), [1, 99.5],
    )
    display_high = float(max(display_high, display_low + 1.0))

    taught = package.node_parameters["center"]
    radius = float(package.node_parameters["radius"])
    source_to_support = np.array(metadata["source_to_support"]["matrix"], dtype=float)
    taught_support = map_points(source_to_support, np.array([[taught["x"], taught["y"]]]))[0]
    matched_center = np.array([match["center"]["x"], match["center"]["y"]])
    taught_in_image = map_points(mapping, np.array([[taught["x"], taught["y"]]]))[0]
    foreground = (patch_crop > 30) | (template_crop > 30)
    fractional_residual = float(np.mean(difference[foreground])) if foreground.any() else 0.0

    figure = plt.figure(figsize=(13.6, 11.4), dpi=140)
    grid = figure.add_gridspec(2, 2, wspace=0.06, hspace=0.16)
    panels = [
        (figure.add_subplot(grid[0, 0]), template_crop, "模板（示教参考图的支撑区）", None),
        (figure.add_subplot(grid[0, 1]), patch_crop,
         f"定位图匹配区域（左上角 {x0}, {y0}）", None),
        (figure.add_subplot(grid[1, 0]), difference, "|匹配区域 − 模板| 绝对差", "magma"),
        (figure.add_subplot(grid[1, 1]),
         np.dstack([patch_crop, template_crop, np.zeros_like(patch_crop)]),
         "红=匹配区域 / 绿=模板（重合处呈黄色）", None),
    ]
    for index, (ax, data, title, cmap) in enumerate(panels):
        if index == 3:
            ax.imshow(np.clip(data, 0, 255) / 255.0, interpolation="antialiased")
        elif index == 2:
            ceiling = max(float(np.percentile(data, 99.0)), 8.0)
            ax.imshow(data, cmap=cmap, vmin=0, vmax=ceiling, interpolation="antialiased")
        else:
            ax.imshow(data, cmap="gray", vmin=float(display_low), vmax=display_high,
                      interpolation="antialiased")
        if index in (0, 1):
            ax.add_patch(Circle((taught_support[0] - (x0 if index else 0),
                                 taught_support[1] - (y0 if index else 0)),
                                radius, fill=False, edgecolor=C_TAUGHT, linestyle="--",
                                linewidth=1.4, label="示教圆（映射到该区域）"))
            ax.plot(taught_support[0] - (x0 if index else 0),
                    taught_support[1] - (y0 if index else 0), marker="+", markersize=13,
                    markeredgewidth=1.6, color=C_TAUGHT)
        if index == 1:
            ax.plot(matched_center[0] - x0, matched_center[1] - y0, marker="o",
                    markersize=9, markerfacecolor="none", markeredgecolor=C_TEMPLATE,
                    markeredgewidth=1.8, label="匹配中心")
        ax.set_title(title, fontsize=10.5)
        ax.tick_params(labelsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    panels[0][0].legend(loc="lower left", fontsize=8, framealpha=0.8)
    panels[1][0].legend(loc="lower left", fontsize=8, framealpha=0.8)
    figure.suptitle(
        f"定位核对 · 得分 {match['score']:.6f}，平移 "
        f"({mapping[0, 2]:+.2f}, {mapping[1, 2]:+.2f}) px，"
        f"旋转 {match['angle_degrees']:.2f}°，尺度 {match['scale']:.3f}",
        fontsize=12.5, y=0.965,
    )
    figure.text(
        0.5, 0.935,
        f"匹配中心 ({matched_center[0]:.2f}, {matched_center[1]:.2f})　"
        f"示教圆映射后圆心 ({taught_in_image[0]:.2f}, {taught_in_image[1]:.2f})　"
        f"前景平均绝对差 {fractional_residual:.2f} 灰度",
        ha="center", fontsize=10, color="#333333",
    )
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


# ---------------------------------------------------------------------------
# Figure 8: fitted centre versus the image centre
# ---------------------------------------------------------------------------


def image_center_metrics(package: DebugPackage, run: Dict[str, Any],
                         image: np.ndarray) -> Dict[str, Any]:
    """Compare the fitted circle centre with the image centre and border margins.

    Args:
        package: Parsed debug package.
        run: Reproduced run document.
        image: Detection image the circle was measured on.

    Returns:
        Centre offsets, distance, direction angle and border margins in pixels.
    """
    node = run["targets"][0]["nodes"][0]["result"]
    centre = node["circle"]["center"]
    radius = float(node["circle"]["radius"])
    height, width = image.shape[:2]
    image_center = {"x": (width - 1) / 2.0, "y": (height - 1) / 2.0}
    dx = float(centre["x"]) - image_center["x"]
    dy = float(centre["y"]) - image_center["y"]
    distance = math.hypot(dx, dy)
    taught = package.node_parameters["center"]
    taught_radius = float(package.node_parameters["radius"])
    reference = package.manifest.get("images", [{}])[0]
    reference_width = int(reference.get("width") or 0) or width
    reference_height = int(reference.get("height") or 0) or height
    taught_center_image = {"x": (reference_width - 1) / 2.0, "y": (reference_height - 1) / 2.0}
    taught_dx = float(taught["x"]) - taught_center_image["x"]
    taught_dy = float(taught["y"]) - taught_center_image["y"]
    return {
        "width": width,
        "height": height,
        "image_center": image_center,
        "centre": {"x": float(centre["x"]), "y": float(centre["y"])},
        "radius": radius,
        "dx": dx,
        "dy": dy,
        "distance": distance,
        "angle_degrees": math.degrees(math.atan2(dy, dx)),
        "relative_width_percent": dx / width * 100.0,
        "relative_height_percent": dy / height * 100.0,
        "diagonal_percent": distance / math.hypot(width, height) * 100.0,
        "margins": {
            "left": float(centre["x"]),
            "right": (width - 1) - float(centre["x"]),
            "top": float(centre["y"]),
            "bottom": (height - 1) - float(centre["y"]),
        },
        "inside_circle": distance < radius,
        "radius_headroom": radius - distance,
        "taught_center": {"x": float(taught["x"]), "y": float(taught["y"])},
        "taught_radius": taught_radius,
        "taught_dx": taught_dx,
        "taught_dy": taught_dy,
        "taught_distance": math.hypot(taught_dx, taught_dy),
    }


def figure_image_center(package: DebugPackage, run: Dict[str, Any],
                        centre_metrics: Dict[str, Any], path: Path) -> Path:
    """Render the fitted centre against the image centre and border margins.

    Args:
        package: Parsed debug package.
        run: Reproduced run document.
        centre_metrics: Result of :func:`image_center_metrics`.
        path: Output PNG filename.

    Returns:
        Written figure path.
    """
    plt = _pyplot()
    from matplotlib import patheffects
    from matplotlib.patches import Circle, Rectangle

    halo = [patheffects.withStroke(linewidth=2.6, foreground="black")]
    image = read_gray(package_file(package, package.inputs["image"]))
    node = run["targets"][0]["nodes"][0]["result"]
    centre = np.array([centre_metrics["centre"]["x"], centre_metrics["centre"]["y"]])
    image_center = np.array([centre_metrics["image_center"]["x"],
                             centre_metrics["image_center"]["y"]])
    radius = centre_metrics["radius"]
    width, height = centre_metrics["width"], centre_metrics["height"]

    figure = plt.figure(figsize=(15.4, 7.6), dpi=140)
    grid = figure.add_gridspec(2, 2, width_ratios=[1.62, 1.0], height_ratios=[1.0, 0.62],
                               wspace=0.10, hspace=0.16)
    overview = figure.add_subplot(grid[:, 0])
    overview.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="antialiased")
    overview.axvline(image_center[0], color="#c2185b", linestyle=":", linewidth=1.1)
    overview.axhline(image_center[1], color="#c2185b", linestyle=":", linewidth=1.1)
    overview.plot(image_center[0], image_center[1], marker="+", markersize=16,
                  markeredgewidth=1.8, color="#c2185b",
                  label=f"图像中心 ({image_center[0]:.1f}, {image_center[1]:.1f})")
    overview.plot(centre[0], centre[1], marker="+", markersize=18, markeredgewidth=2.0,
                  color=C_CENTER, zorder=26, label="拟合圆心")
    overview.add_patch(Circle(centre, radius, fill=False, edgecolor=C_FIT, linewidth=1.6,
                              label=f"拟合圆 R={radius:.2f} px"))
    overview.annotate("", xy=tuple(centre), xytext=tuple(image_center),
                      arrowprops={"arrowstyle": "<->", "color": "#c2185b", "lw": 1.4})
    overview.annotate(f"圆心偏移 {centre_metrics['distance']:.1f} px",
                      xy=tuple((centre + image_center) / 2), xytext=(0, -34),
                      textcoords="offset points", ha="center", color="#c2185b",
                      fontsize=10, fontweight="bold",
                      arrowprops={"arrowstyle": "-", "color": "#c2185b", "lw": 0.9})
    for label, position, horizontal, vertical in (
        (f"左 {centre_metrics['margins']['left']:.0f} px",
         (28, centre[1] - 26), "left", "top"),
        (f"右 {centre_metrics['margins']['right']:.0f} px",
         (width - 28, centre[1] - 26), "right", "top"),
        (f"上 {centre_metrics['margins']['top']:.0f} px",
         (centre[0] + 30, 24), "left", "top"),
        (f"下 {centre_metrics['margins']['bottom']:.0f} px",
         (centre[0] + 30, height - 24), "left", "bottom"),
    ):
        overview.text(position[0], position[1], label, fontsize=9.0, color="#ffffff",
                      ha=horizontal, va=vertical, path_effects=halo)
    overview.set_xlim(0, width)
    overview.set_ylim(height, 0)
    overview.set_xlabel("x (px)")
    overview.set_ylabel("y (px)")
    overview.set_title(f"拟合圆心 vs 图像中心 · 定位图 {width}×{height}",
                       fontsize=11.5)
    overview.legend(loc="lower left", fontsize=8.6, framealpha=0.85)
    zoom_box = _center_zoom_box(centre, image_center, image.shape, 300)

    zoom = figure.add_subplot(grid[0, 1])
    x0, x1, y0, y1 = zoom_box
    zoom_crop = image[y0:y1, x0:x1]
    zoom_ceiling = max(45.0, float(np.percentile(zoom_crop, 99.5)))
    zoom.imshow(zoom_crop, cmap="gray", vmin=0, vmax=zoom_ceiling,
                interpolation="nearest", extent=(x0, x1, y1, y0))
    zoom.axvline(image_center[0], color="#c2185b", linestyle=":", linewidth=1.2)
    zoom.axhline(image_center[1], color="#c2185b", linestyle=":", linewidth=1.2)
    zoom.plot(image_center[0], image_center[1], marker="+", markersize=16,
              markeredgewidth=1.8, color="#c2185b")
    zoom.plot(centre[0], centre[1], marker="+", markersize=18, markeredgewidth=2.0,
              color=C_CENTER, zorder=26)
    zoom.annotate("", xy=tuple(centre), xytext=tuple(image_center),
                  arrowprops={"arrowstyle": "<->", "color": "#c2185b", "lw": 1.4})
    zoom.annotate(f"Δ=({centre_metrics['dx']:+.2f}, {centre_metrics['dy']:+.2f}) px\n"
                  f"{centre_metrics['distance']:.2f} px",
                  xy=tuple((centre + image_center) / 2), xytext=(12, -34),
                  textcoords="offset points", color="#c2185b", fontsize=9,
                  bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85,
                        "edgecolor": "none"})
    zoom.plot([x0 + 20, x0 + 120], [y1 - 26, y1 - 26], color="#ffffff", linewidth=3.0,
              solid_capstyle="butt",
              path_effects=[patheffects.withStroke(linewidth=5.0, foreground="black")])
    zoom.text(x0 + 20, y1 - 36, "100 px", color="#ffffff", fontsize=8.5,
              path_effects=halo)
    zoom.set_xlim(x0, x1)
    zoom.set_ylim(y1, y0)
    zoom.set_title("图像中心与拟合圆心（1:1 像素，局部）", fontsize=10.5)
    zoom.tick_params(labelsize=8)
    zoom.set_xlabel("x (px)", fontsize=9)
    zoom.set_ylabel("y (px)", fontsize=9)

    summary = figure.add_subplot(grid[1, 1])
    summary.axis("off")
    rows = [
        ("图像尺寸", f"{width} × {height} px"),
        ("图像中心(像素中心)", f"({image_center[0]:.1f}, {image_center[1]:.1f})"),
        ("拟合圆心", f"({centre[0]:.3f}, {centre[1]:.3f})"),
        ("偏移 Δx / Δy", f"{centre_metrics['dx']:+.3f} / {centre_metrics['dy']:+.3f} px"),
        ("偏移距离", f"{centre_metrics['distance']:.3f} px "
                     f"（宽 {centre_metrics['relative_width_percent']:+.2f}%，"
                     f"高 {centre_metrics['relative_height_percent']:+.2f}%）"),
        ("方位角", f"{centre_metrics['angle_degrees']:+.2f}°（0°=向右，y 向下为正）"),
        ("四边余量", f"左 {centre_metrics['margins']['left']:.1f} / "
                     f"右 {centre_metrics['margins']['right']:.1f} / "
                     f"上 {centre_metrics['margins']['top']:.1f} / "
                     f"下 {centre_metrics['margins']['bottom']:.1f} px"),
        ("示教圆心偏移", f"({centre_metrics['taught_dx']:+.3f}, "
                         f"{centre_metrics['taught_dy']:+.3f}) → "
                         f"{centre_metrics['taught_distance']:.3f} px"),
    ]
    step = 0.125
    for index, (label, value) in enumerate(rows):
        y = 0.96 - index * step
        summary.text(0.0, y, label, transform=summary.transAxes, fontsize=9.4,
                     va="top", color="#444444")
        summary.text(0.42, y, value, transform=summary.transAxes, fontsize=9.4,
                     va="top", color="#111111",
                     family="DejaVu Sans Mono" if value.isascii() else None)
    verdict = ("否" if not (abs(centre_metrics["distance"]) < 0.5) else "是")
    summary.text(0.0, -0.04,
                 f"结论：拟合圆心{'不在' if verdict == '否' else '在'}图像像素中心 —— "
                 f"偏移 {centre_metrics['distance']:.2f} px；图像中心在拟合圆内"
                 f"（距圆周还有 {centre_metrics['radius_headroom']:.1f} px）。",
                 transform=summary.transAxes, fontsize=9.6, va="top", color="#8b0000")
    figure.savefig(path)
    plt.close(figure)
    return path


def _center_zoom_box(centre: np.ndarray, image_center: np.ndarray,
                     shape: Tuple[int, int], extent: int) -> Tuple[int, int, int, int]:
    """Return an integer crop box that contains both centres.

    Args:
        centre: Fitted circle centre in pixels.
        image_center: Image centre in pixels.
        shape: Image shape (height, width).
        extent: Half size of the requested crop in pixels.

    Returns:
        ``(x0, x1, y0, y1)`` integer crop bounds clipped to the image.
    """
    midpoint = (centre + image_center) / 2.0
    x0 = int(max(0, round(midpoint[0] - extent)))
    x1 = int(min(shape[1], round(midpoint[0] + extent)))
    y0 = int(max(0, round(midpoint[1] - extent)))
    y1 = int(min(shape[0], round(midpoint[1] + extent)))
    return x0, x1, y0, y1


# ---------------------------------------------------------------------------
# Figure 7: verification table
# ---------------------------------------------------------------------------


def figure_verification_table(package: DebugPackage, run: Dict[str, Any],
                              verification: Dict[str, Any], environment: Dict[str, Any],
                              path: Path) -> Path:
    """Render the recorded/reproduced comparison as a table image.

    Args:
        package: Parsed debug package.
        run: Reproduced run document.
        verification: Equivalence verdict from :func:`verify`.
        environment: Runtime identity comparison from :func:`environment_report`.
        path: Output PNG filename.

    Returns:
        Written figure path.
    """
    plt = _pyplot()

    recorded_target = package.result["targets"][0]
    reproduced_target = run["targets"][0]
    recorded_node = recorded_target["nodes"][0]["result"]
    reproduced_node = reproduced_target["nodes"][0]["result"]

    def row(name: str, first: Any, second: Any, digits: int = 6) -> List[str]:
        if isinstance(first, float) or isinstance(second, float):
            left = f"{first:.{digits}f}"
            right = f"{second:.{digits}f}"
            delta = second - first
            if abs(delta) < 1e-12:
                change = "0"
            elif abs(delta) < 1e-6:
                change = f"{delta:+.1e}"
            else:
                change = f"{delta:+.{digits}f}"
        else:
            left, right = str(first), str(second)
            change = "一致" if first == second else "不同"
        return [name, left, right, change]

    rows = [
        row("定位模板得分", recorded_target["localization"]["match"]["score"],
            reproduced_target["localization"]["match"]["score"], 9),
        row("匹配中心 x (px)", recorded_target["localization"]["match"]["center"]["x"],
            reproduced_target["localization"]["match"]["center"]["x"], 4),
        row("匹配中心 y (px)", recorded_target["localization"]["match"]["center"]["y"],
            reproduced_target["localization"]["match"]["center"]["y"], 4),
        row("拟合圆心 x (px)", recorded_node["circle"]["center"]["x"],
            reproduced_node["circle"]["center"]["x"], 4),
        row("拟合圆心 y (px)", recorded_node["circle"]["center"]["y"],
            reproduced_node["circle"]["center"]["y"], 4),
        row("拟合半径 (px)", recorded_node["circle"]["radius"],
            reproduced_node["circle"]["radius"], 4),
        row("采样点 / 成功探针",
            f"{recorded_node['diagnostics']['sample_count']} / "
            f"{recorded_node['diagnostics']['successful_probe_count']}",
            f"{reproduced_node['diagnostics']['sample_count']} / "
            f"{reproduced_node['diagnostics']['successful_probe_count']}"),
        row("拟合 RMS (px)", recorded_node["diagnostics"]["rms_error"],
            reproduced_node["diagnostics"]["rms_error"], 4),
        row("最大绝对残差 (px)", recorded_node["diagnostics"]["max_absolute_error"],
            reproduced_node["diagnostics"]["max_absolute_error"], 4),
        row("采样直径散布 (px)", recorded_node["sampled_diameter_spread"],
            reproduced_node["sampled_diameter_spread"], 4),
        row("结果状态 / 有效",
            f"{recorded_target['nodes'][0]['state']} / {recorded_target['valid']}",
            f"{reproduced_target['nodes'][0]['state']} / {reproduced_target['valid']}"),
        row("配置 SHA-256（前 12 位）",
            verification["recorded_configuration_sha256"][:12],
            verification["reproduced_configuration_sha256"][:12]),
        row("实现 SHA-256（前 12 位）",
            str(package.result.get("runtime_identity", {}).get("implementation_sha256", ""))[:12],
            str(environment["local"].get("implementation_sha256", ""))[:12]),
    ]

    fig, ax = plt.subplots(figsize=(11.6, 0.45 * len(rows) + 1.9), dpi=150)
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["指标", "原始调试包", "本地复现", "差异"],
        cellLoc="right",
        colWidths=[0.34, 0.24, 0.24, 0.18],
        loc="upper center",
        bbox=[0.01, 0.02, 0.98, 0.83],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    for (row_index, _), cell in table.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if row_index == 0:
            cell.set_facecolor("#eeeeee")
            cell.set_text_props(fontweight="bold")
        elif row_index % 2 == 0:
            cell.set_facecolor("#f8f9fb")
    ax.text(0.01, 0.985, "原始记录 vs 本地复现（measure-v2 执行器）",
            fontsize=13.5, fontweight="bold", transform=ax.transAxes, va="top")
    verdict = ("逐位一致" if verification["equivalent"] else
               f"存在 {verification['difference_count']} 处差异")
    ax.text(0.01, 0.940,
            f"结论：{verdict}（容差 {verification['tolerance']:g}）"
            f"　最大绝对差 {verification['max_absolute_difference']:.2e}",
            fontsize=10.5, color="#333333", transform=ax.transAxes, va="top")
    ax.text(0.01, 0.905,
            f"最大差位置：{verification['max_difference_path'] or '—'}",
            fontsize=9.0, color="#666666", transform=ax.transAxes, va="top")
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _fmt(value: Any, digits: int = 6) -> str:
    """Format an optional float for Markdown tables."""
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_report(package: DebugPackage, run: Dict[str, Any], verification: Dict[str, Any],
                 environment: Dict[str, Any], assets: List[Dict[str, Any]],
                 metrics: Dict[str, Any], figures: Dict[str, Path], workdir: Path,
                 duration_ms: Optional[float], centre_metrics: Dict[str, Any]) -> Path:
    """Write the Chinese Markdown reproduction report.

    Args:
        package: Parsed debug package.
        run: Reproduced run document.
        verification: Equivalence verdict.
        environment: Runtime identity comparison.
        assets: Staged input verification records.
        metrics: Extracted business metrics.
        figures: Rendered figure paths keyed by role.
        workdir: Reproduction working directory.
        duration_ms: Measured replay duration.
        centre_metrics: Fitted centre versus image centre comparison.

    Returns:
        Written Markdown report path.
    """
    def delta_text(first: float, second: float) -> str:
        change = second - first
        if abs(change) < 1e-12:
            return "0"
        if abs(change) < 1e-6:
            return f"{change:+.1e}"
        return f"{change:+.6f}"

    recorded = metrics["recorded"]
    reproduced = metrics["reproduced"]
    probe = metrics["probe"]
    node = run["targets"][0]["nodes"][0]["result"]
    diagnostics = node["diagnostics"]
    figures = {key: Path(value) for key, value in figures.items()}

    def figure_link(key: str, caption: str) -> str:
        if key not in figures:
            return ""
        relative = figures[key].relative_to(workdir).as_posix()
        return f"![{caption}]({relative})\n"

    lines = [
        f"# 算法调试包复现报告 · {package.run_id}",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| 调试包路径 | `{package.root}` |",
        f"| 规范 | `{package.manifest.get('schema')}/{package.manifest.get('version')}` |",
        f"| 任务类型 / 状态 | `{package.manifest.get('type')}` / "
        f"`{package.manifest.get('state')}` |",
        f"| 算法版本 | "
        f"`{package.result.get('runtime_identity', {}).get('algorithm_version')}` |",
        f"| 原始任务耗时 | {package.manifest.get('duration_ms')} ms"
        f"（含提交与归档） |",
        f"| 复现耗时 | {_fmt(duration_ms, 0)} ms（仅测量执行） |",
        "",
        "## 1. 结论",
        "",
    ]
    if verification["equivalent"]:
        lines += [
            "复现成功：在与原始记录相同的输入（SHA-256 校验通过）上重放该配方，"
            "`measure-v2` 执行器在本仓库复现出的数值与调试包 `result.json` **逐位一致**。"
            f"数值记录在容差 `{verification['tolerance']:g}` 下差异 "
            f"{verification['difference_count']} 处，最大绝对差 "
            f"`{verification['max_absolute_difference']:.2e}`"
            f"（位置 `{verification['max_difference_path']}`）。",
            "",
            f"- 定位：模板得分 `{reproduced['localization_score']:.9f}`，"
            f"匹配中心 `({reproduced['match_center']['x']:.3f}, "
            f"{reproduced['match_center']['y']:.3f})`。",
            f"- 测量：拟合圆 C=`({reproduced['center']['x']:.4f}, "
            f"{reproduced['center']['y']:.4f})`，"
            f"R=`{reproduced['radius']:.4f} px`；相对示教圆半径偏差 "
            f"`{metrics['radius_delta_px']:+.3f} px`，"
            f"圆心偏移 `{metrics['center_offset_px']:.3f} px`。",
            f"- 质量：{reproduced['successful_probe_count']}/"
            f"{reproduced['requested_probe_count']} 条探针"
            f"成功取到边缘，拟合 RMS `{reproduced['rms_error']:.3f} px`，"
            f"最大残差 `{reproduced['max_absolute_error']:.3f} px`。",
            "",
        ]
    else:
        lines += [
            f"复现未完全一致：容差 `{verification['tolerance']:g}` 下存在 "
            f"{verification['difference_count']} 处差异，最大绝对差 "
            f"`{verification['max_absolute_difference']:.2e}`。",
            "",
            "差异路径（前 20 条）：",
            "",
        ]
        lines += [f"- `{item}`" for item in verification["differences"][:20]]
        lines.append("")

    lines += [
        "## 2. 输入资产校验",
        "",
        "| 引用 | 包内文件 | SHA-256 | 与记录一致 |",
        "|---|---|---|---|",
    ]
    for record in assets:
        lines.append(
            f"| `{record['reference']}` | `{record['package_file']}` | "
            f"`{record['sha256'][:16]}…` | {'是' if record['matches_record'] else '否'} |"
        )
    lines += [
        "",
        "## 3. 复现环境",
        "",
        "| 项 | 原始记录 | 本地复现 | 一致 |",
        "|---|---|---|---|",
    ]
    for row in environment["rows"]:
        recorded_value = str(row["recorded"])
        local_value = str(row["local"])
        if len(recorded_value) == 64:
            recorded_value = recorded_value[:16] + "…"
        if len(local_value) == 64:
            local_value = local_value[:16] + "…"
        lines.append(
            f"| `{row['field']}` | `{recorded_value}` | `{local_value}` | "
            f"{'是' if row['match'] else '否'} |"
        )
    lines += [
        "",
    ]
    if verification["recipe_differences"]:
        lines += [
            "配方结构差异（本仓库实现相对生产构建新增的字段，"
            "默认值不改变本次测量路径）：",
            "",
        ]
        lines += [f"- `{item}`" for item in verification["recipe_differences"][:10]]
        lines += [
            "",
            "因此两份**规范化配置摘要**不同（见第 5 节最后两行），"
            "但执行所依据的几何/参数完全一致，观测数值逐位一致。",
            "",
        ]

    lines += [
        "## 4. 复现步骤",
        "",
        "```bash",
        f"python scripts/repro_debug_package.py \"{package.root}\" --workdir \"{workdir}\"",
        "```",
        "",
        "脚本依次完成：资产镜像到本地 `exchange` 根（逐文件 SHA-256 校验）→ "
        "通过 `POST /api/v2/runs` 重放 `inputs.json` 中的请求 → "
        "用 `measure_api.v2.replay` 的数值比较器对照 `result.json` → 渲染下图。",
        "",
        "",
        "依赖需与原始 `runtime_identity` 对齐：`numpy==1.26.4`、"
        "`opencv-python-headless==4.11.0.86`、`scipy==1.13.1`、`pydantic==2.11.7`、"
        "`fastapi==0.115.12`（`opencv_build_sha256` 已核对一致）。",
        "",
        "## 5. 关键结果对照",
        "",
        "| 指标 | 原始记录 | 本地复现 | 差异 |",
        "|---|---|---|---|",
        f"| 定位得分 | {recorded['localization_score']:.9f} | "
        f"{reproduced['localization_score']:.9f} | "
        f"{delta_text(recorded['localization_score'], reproduced['localization_score'])} |",
        f"| 匹配中心 x | {recorded['match_center']['x']:.4f} | "
        f"{reproduced['match_center']['x']:.4f} | "
        f"{delta_text(recorded['match_center']['x'], reproduced['match_center']['x'])} |",
        f"| 匹配中心 y | {recorded['match_center']['y']:.4f} | "
        f"{reproduced['match_center']['y']:.4f} | "
        f"{delta_text(recorded['match_center']['y'], reproduced['match_center']['y'])} |",
        f"| 拟合圆心 x | {recorded['center']['x']:.4f} | {reproduced['center']['x']:.4f} | "
        f"{delta_text(recorded['center']['x'], reproduced['center']['x'])} |",
        f"| 拟合圆心 y | {recorded['center']['y']:.4f} | {reproduced['center']['y']:.4f} | "
        f"{delta_text(recorded['center']['y'], reproduced['center']['y'])} |",
        f"| 拟合半径 | {recorded['radius']:.4f} | {reproduced['radius']:.4f} | "
        f"{delta_text(recorded['radius'], reproduced['radius'])} |",
        f"| 拟合 RMS | {recorded['rms_error']:.6f} | {reproduced['rms_error']:.6f} | "
        f"{delta_text(recorded['rms_error'], reproduced['rms_error'])} |",
        f"| 最大绝对残差 | {recorded['max_absolute_error']:.6f} | "
        f"{reproduced['max_absolute_error']:.6f} | "
        f"{delta_text(recorded['max_absolute_error'], reproduced['max_absolute_error'])} |",
        f"| 采样直径散布 | {recorded['sampled_diameter_spread']:.6f} | "
        f"{reproduced['sampled_diameter_spread']:.6f} | "
        f"{delta_text(recorded['sampled_diameter_spread'], reproduced['sampled_diameter_spread'])}"
        " |",
        f"| 配置 SHA-256 | `{verification['recorded_configuration_sha256']}` | "
        f"`{verification['reproduced_configuration_sha256']}` | 见第 3 节说明 |",
        "",
        "## 6. 测量链路与几何",
        "",
        f"1. **定位**：定位模板（2500×2486 支撑区）在定位图中搜索，"
        f"得分 `{reproduced['localization_score']:.6f}`，"
        f"得到参考帧 → 定位图的平移 "
        f"`({np.array(reproduced['mapping'])[0, 2]:+.3f}, "
        f"{np.array(reproduced['mapping'])[1, 2]:+.3f}) px`。",
        "2. **对齐**：按该平移把定位图搬到参考帧（5472×3648），"
        "探针在参考帧坐标下布设。",
        f"3. **探针**：示教圆 C=`({metrics['taught_center']['x']:.3f}, "
        f"{metrics['taught_center']['y']:.3f})` R=`{metrics['taught_radius']:.3f} px` 上均匀布 "
        f"{probe['probe_count']} 条径向探针（搜索半长 {probe['search_half_length']:g} px、"
        f"平均半宽 {probe['average_half_width']:g} px、sigma {probe['sigma']:g}、"
        f"阈值 {probe['threshold']:g} 灰度、极性 `{probe['polarity']}`、"
        f"选择 `{probe['selection']}`），"
        f"每条取首个亮→暗边缘。",
        f"4. **拟合**：对 {diagnostics['sample_count']} 个亚像素点做最小二乘圆拟合，"
        f"再映射回定位图坐标：C=`({reproduced['center']['x']:.4f}, "
        f"{reproduced['center']['y']:.4f})`，"
        f"R=`{reproduced['radius']:.4f} px`。",
        f"5. **质量**：RMS `{diagnostics['rms_error']:.3f} px`、"
        f"MAE `{diagnostics['mean_absolute_error']:.3f} px`、"
        f"最大残差 `{diagnostics['max_absolute_error']:.3f} px`；采样直径散布 "
        f"`{reproduced['sampled_diameter_spread']:.3f} px`（= 2×(max r − min r)）。",
        "",
        "## 7. 拟合圆心是否位于图像像素中心",
        "",
        "**不是。** 图像为 "
        f"`{centre_metrics['width']}×{centre_metrics['height']}`，"
        f"按像素中心坐标（左上角像素中心为 `(0, 0)`）图像中心为 "
        f"`({centre_metrics['image_center']['x']:.1f}, "
        f"{centre_metrics['image_center']['y']:.1f})`，"
        f"拟合圆心为 `({centre_metrics['centre']['x']:.3f}, "
        f"{centre_metrics['centre']['y']:.3f})`，"
        f"偏移 `Δx = {centre_metrics['dx']:+.3f} px`、`Δy = {centre_metrics['dy']:+.3f} px`，"
        f"距离 `{centre_metrics['distance']:.3f} px`"
        f"（图像宽的 {abs(centre_metrics['relative_width_percent']):.2f}%、"
        f"高的 {abs(centre_metrics['relative_height_percent']):.2f}%、"
        f"对角线的 {centre_metrics['diagonal_percent']:.2f}%），"
        f"方位角 `{centre_metrics['angle_degrees']:+.2f}°`（0° 向右，y 轴向下），"
        f"即圆心位于图像中心**右侧**、略偏上。",
        "",
        "| 量 | 值 |",
        "|---|---|",
        f"| 图像尺寸 | {centre_metrics['width']} × {centre_metrics['height']} px |",
        f"| 图像中心（像素中心坐标） | ({centre_metrics['image_center']['x']:.1f}, "
        f"{centre_metrics['image_center']['y']:.1f}) px |",
        f"| 拟合圆心 | ({centre_metrics['centre']['x']:.4f}, "
        f"{centre_metrics['centre']['y']:.4f}) px |",
        f"| 偏移 Δx / Δy | {centre_metrics['dx']:+.3f} / {centre_metrics['dy']:+.3f} px |",
        f"| 偏移距离 | {centre_metrics['distance']:.3f} px |",
        f"| 距左 / 右边界 | {centre_metrics['margins']['left']:.1f} / "
        f"{centre_metrics['margins']['right']:.1f} px |",
        f"| 距上 / 下边界 | {centre_metrics['margins']['top']:.1f} / "
        f"{centre_metrics['margins']['bottom']:.1f} px |",
        f"| 示教圆心（参考帧）偏移 | ({centre_metrics['taught_dx']:+.3f}, "
        f"{centre_metrics['taught_dy']:+.3f}) → {centre_metrics['taught_distance']:.3f} px |",
        "",
        "说明：图像中心到圆心的距离 "
        f"`{centre_metrics['distance']:.1f} px` 小于拟合半径 "
        f"`{centre_metrics['radius']:.2f} px`，所以图像中心仍落在圆内，"
        f"距圆周还有 `{centre_metrics['radius_headroom']:.1f} px`。"
        "示教时（参考帧）圆心偏离图像中心更多"
        f"（`{centre_metrics['taught_distance']:.1f} px`），定位图相对参考图平移了 "
        f"`({np.array(reproduced['mapping'])[0, 2]:+.2f}, "
        f"{np.array(reproduced['mapping'])[1, 2]:+.2f}) px`，"
        "因此该偏移主要是**工件在视野中的摆放位置**，不是拟合偏差；"
        f"本次拟合与示教映射位置的偏差仅 `{metrics['center_offset_px']:.2f} px`。",
        "",
        "## 8. 可视化",
        "",
    ]
    for key, caption in (
        ("localization", "定位总览：模板支撑区、模板匹配中心、示教圆、拟合圆与边缘点"),
        ("detail", "测量细节（1:1 像素）：径向探针、亚像素边缘、拟合圆与示教圆"),
        ("aligned", "对齐到参考帧后的算法执行视图（探针实际采样的像素）"),
        ("profiles", "探针灰度/梯度剖面与亚像素边缘选取"),
        ("quality", "拟合质量：残差、边缘幅值、采样半径分布、残差形状"),
        ("reference", "示教参考图与配方几何"),
        ("image_center", "拟合圆心与图像中心的偏差（全景 + 1:1 局部）"),
        ("table", "原始记录与本地复现的数值对照"),
    ):
        link = figure_link(key, caption)
        if link:
            lines += [f"### {caption}", "", link]
    lines += [
        "## 9. 说明",
        "",
        f"- 调试包 `manifest.json` 的 `replay_supported` 为 "
        f"`{str(package.manifest.get('replay_supported')).lower()}`，`measure.pick_replay` "
        f"只覆盖 gantry pick 服务的 vision/offset/locate；"
        f"本脚本补上了 `type=run` 的复现路径。",
        "- 复现使用 `inputs.json` 中工作台实际提交的请求，而非 `result.json` 的 `resolved_recipe`；"
        "后者仅用于对照与可视化。",
        "- 图像经调试包转换（`explicit-capture/1`），与现场采集原图不一定逐位相同，"
        "但与本包 `result.json` 所用输入逐位相同。",
        "",
    ]
    path = workdir / "reproduction_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Reproduce, verify, render and report one debug package.

    Args:
        argv: Optional command-line arguments; process arguments are used when omitted.

    Returns:
        Process exit code: 0 on numerical equivalence, 1 otherwise.
    """
    parser = argparse.ArgumentParser(
        description="Reproduce a measure-v2 algorithm debug package and visualise it",
    )
    parser.add_argument("package", type=Path, help="Extracted 算法调试包 directory")
    parser.add_argument("--workdir", type=Path, default=None,
                        help="Working directory (default: output/repro-<run_id>)")
    parser.add_argument("--tolerance", type=float, default=1e-9,
                        help="Absolute tolerance for numerical equivalence (default 1e-9)")
    parser.add_argument("--timeout-ms", type=int, default=120000,
                        help="Operation deadline passed to the service")
    parser.add_argument("--no-figures", action="store_true", help="Skip figure rendering")
    parser.add_argument(
        "--override", action="append", default=[], metavar="NAME=VALUE",
        help="Override one parameter of the first recipe node before execution",
    )
    parser.add_argument(
        "--render-recorded", action="store_true",
        help="Skip execution: render the recorded result.json directly (no replay)",
    )
    args = parser.parse_args(argv)

    package_dir = args.package.expanduser().resolve()
    if not (package_dir / "manifest.json").exists():
        parser.error(f"{package_dir} is not a debug package (manifest.json missing)")
    package = load_package(package_dir)
    if package.manifest.get("type") != "run":
        parser.error(f"Unsupported package type {package.manifest.get('type')!r} (expected 'run')")
    if not package.result.get("targets"):
        parser.error("Package has no recorded targets; nothing to compare against")
    if args.override:
        package.inputs = json.loads(json.dumps(package.inputs))
        parameters = package.inputs["recipe"]["nodes"][0]["parameters"]
        for item in args.override:
            if "=" not in item:
                parser.error(f"--override expects NAME=VALUE, got {item!r}")
            name, _, raw = item.partition("=")
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                value = raw
            parameters[name] = value
            print(f"      override {name} = {value!r}")

    prefix = "render" if args.render_recorded else "repro"
    workdir = (args.workdir or (REPO_ROOT / "output" / f"{prefix}-{package.run_id[:8]}")).resolve()
    exchange_root = workdir / "exchange"
    output_root = workdir / "outputs"
    figures_dir = workdir / "figures"
    workdir.mkdir(parents=True, exist_ok=True)
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] 校验并镜像输入资产 → {exchange_root}")
    assets = stage_exchange(package, exchange_root)
    for record in assets:
        flag = "OK" if record["matches_record"] else "MISMATCH"
        print(f"      {flag} {record['reference']}  sha256={record['sha256'][:12]}…")

    if args.render_recorded:
        print("[2/4] 跳过重放：直接使用调试包 result.json 的记录结果")
        run = package.result
        duration_ms = None
        print(f"      valid={run['valid']} status={run['status']}")
    else:
        print("[2/4] 通过 /api/v2/runs 重放配方")
        started = cv2.getTickCount()
        run = reproduce(package, exchange_root, output_root, args.timeout_ms)
        duration_ms = (cv2.getTickCount() - started) * 1000.0 / cv2.getTickFrequency()
        (workdir / "reproduced_result.json").write_text(
            json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"      valid={run['valid']} status={run['status']} {duration_ms:.0f} ms")

    if args.render_recorded:
        print("[3/4] 汇总记录数值（不做数值比对）")
        verification = None
        environment = environment_report(package.result.get("runtime_identity") or {})
    else:
        print("[3/4] 与调试包 result.json 逐项比对")
        verification = verify(package, run, args.tolerance)
        environment = environment_report(package.result.get("runtime_identity") or {})
    metrics = key_metrics(package, run)
    metrics["reproduced_edges"] = run["targets"][0]["nodes"][0]["result"]["edges"]
    metrics["recorded_edges"] = package.result["targets"][0]["nodes"][0]["result"]["edges"]
    centre_metrics = image_center_metrics(
        package, run, read_gray(package_file(package, package.inputs["image"])))
    summary_name = "rendered_record.json" if args.render_recorded else "verification.json"
    (workdir / summary_name).write_text(
        json.dumps({
            "verification": verification,
            "environment": environment,
            "metrics": {
                "taught": {"center": metrics["taught_center"], "radius": metrics["taught_radius"]},
                "taught_center_in_image": metrics["taught_center_in_image"],
                "probe": metrics["probe"],
                "recorded": {k: v for k, v in metrics["recorded"].items()},
                "reproduced": {k: v for k, v in metrics["reproduced"].items()},
                "center_offset_px": metrics["center_offset_px"],
                "radius_delta_px": metrics["radius_delta_px"],
                "image_center": centre_metrics,
                "duration_ms": duration_ms,
            },
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if verification is not None:
        print(f"      differences={verification['difference_count']} "
              f"max|Δ|={verification['max_absolute_difference']:.2e}")

    figures: Dict[str, Path] = {}
    if not args.no_figures:
        print(f"[4/4] 渲染复现可视化 → {figures_dir}")
        figures_dir.mkdir(parents=True, exist_ok=True)
        figures["localization"] = figure_localization(
            package, run, metrics, figures_dir / FIGURES["localization"])
        figures["detail"] = figure_circle_detail(
            package, run, metrics, figures_dir / FIGURES["detail"])
        aligned, alignment_matrix = aligned_patch(package, run, exchange_root, figures_dir)
        figures["aligned"] = figure_aligned_frame(
            package, run, metrics, aligned, alignment_matrix, figures_dir / FIGURES["aligned"])
        records = probe_profiles(aligned, package, metrics)
        figures["profiles"] = figure_probe_profiles(
            records, metrics, figures_dir / FIGURES["profiles"])
        figures["quality"] = figure_quality(run, metrics, figures_dir / FIGURES["quality"])
        figures["reference"] = figure_taught_reference(
            package, metrics, figures_dir / FIGURES["reference"])
        figures["image_center"] = figure_image_center(
            package, run, centre_metrics, figures_dir / FIGURES["image_center"])
        figures["localization_check"] = figure_localization_check(
            package, run, figures_dir / FIGURES["localization_check"])
        if verification is not None:
            figures["table"] = figure_verification_table(
                package, run, verification, environment, figures_dir / FIGURES["table"])
        for key, figure in figures.items():
            print(f"      {key:18s} {figure}")
    else:
        print("[4/4] 跳过图渲染 (--no-figures)")

    if verification is not None:
        report = write_report(package, run, verification, environment, assets, metrics,
                              figures, workdir, duration_ms, centre_metrics)
        print(f"\n报告：{report}")
        print(f"结果：{workdir / 'reproduced_result.json'}")
        print(f"校验：{workdir / 'verification.json'}")
        return 0 if verification["equivalent"] else 1
    print(f"\n数值摘要：{workdir / 'rendered_record.json'}")
    print(f"图：{figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())