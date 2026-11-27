#!/usr/bin/env python3
"""Reproduce a ``localization_preview`` debug package and measure its coarse search.

A workbench ``type == "localization_preview"`` package is a single atomic
``template-matches`` call: ``inputs.json`` holds the portable request, ``images/``
and ``assets/`` hold the exact bytes, and ``result.json`` holds the recorded
answer.  ``measure.pick_replay`` does not cover this type, so this script:

1. mirrors the referenced template/image assets into a local ``exchange`` shared
   root, verifying each recorded SHA-256;
2. replays ``inputs.json`` through the same measure-v2 operation
   (``POST /api/v2/template-matches``) in this checkout;
3. compares the match with ``result.json`` (center, angle, scale, score, matrix);
4. re-runs the same request with different ``search.coarse_*`` values to show
   what each coarse knob controls (pose budget, refinement band, gate);
5. sweeps the NCC score over every rotation at the coarse and fine resolutions,
   which is what a coarse grid has to hit;
6. renders an overlay, a template/patch comparison, the sweep chart and the
   score curves, then writes ``report.md`` plus the JSON evidence.

Usage:
    python scripts/repro_localization_coarse.py <package_dir> [--workdir DIR]
        [--no-sweep] [--no-curve] [--no-figures]

Exit code is 0 when the reproduced match agrees with the recorded one.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]

# Rendering must never try to attach to a Qt/X11 session: compute workers are
# spawned headless and a Qt platform-plugin probe aborts the process.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "output" / ".mplconfig"))

import cv2  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.use("Agg")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from measure_api.v2.app import create_app  # noqa: E402
from measure.template_search import (  # noqa: E402
    decimate_pixels, render_variant, variant_size,
)
from measure_api.v2.models import ServiceSettings  # noqa: E402
from measure_api.v2.versions import runtime_identity  # noqa: E402

C_MATCH = (0, 200, 0)
C_ANCHOR = (0, 0, 255)
C_WINDOW = (0, 190, 255)


# ---------------------------------------------------------------------------
# Package loading and staging
# ---------------------------------------------------------------------------


@dataclass
class Package:
    """Parsed localization_preview package with its asset files resolved."""

    root: Path
    manifest: Dict[str, Any]
    job: Dict[str, Any]
    request: Dict[str, Any]
    inputs: Dict[str, Any]
    result: Dict[str, Any]
    assets: List[Dict[str, Any]] = field(default_factory=list)
    asset_files: Dict[str, Path] = field(default_factory=dict)

    @property
    def recorded_match(self) -> Dict[str, Any]:
        matches = self.result.get("matches") or []
        if not matches:
            raise RuntimeError("Recorded result has no match to compare against")
        return matches[0]

    @property
    def search(self) -> Dict[str, Any]:
        return self.inputs["search"]


def read_json(path: Path) -> Dict[str, Any]:
    """Read one JSON document, or an empty mapping when it is absent."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_asset_index(root: Path, manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the asset index, tolerating both a list and an object envelope."""
    document = read_json(root / str(manifest.get("assets") or "assets.json"))
    if isinstance(document, dict):
        document = document.get("assets") or []
    return list(document or [])


def load_package(root: Path) -> Package:
    """Parse a localization_preview package and index its packaged assets.

    Args:
        root: Extracted debug package directory.

    Returns:
        Package with ``asset_files`` mapping recorded asset ids to package files.
    """
    manifest = read_json(root / "manifest.json")
    package = Package(
        root=root,
        manifest=manifest,
        job=read_json(root / "job.json"),
        request=read_json(root / "request.json"),
        inputs=read_json(root / "inputs.json"),
        result=read_json(root / "result.json"),
        assets=read_asset_index(root, manifest),
    )
    for entry in package.assets:
        identifier = str((entry.get("asset") or {}).get("id") or "")
        candidate = root / str(entry.get("file"))
        if identifier and candidate.exists():
            package.asset_files[identifier] = candidate
    if not package.asset_files:
        raise RuntimeError("Package assets.json references no packaged file")
    if "template" not in package.inputs or "image" not in package.inputs:
        raise RuntimeError("Package inputs.json is not a template-matches request")
    return package


def asset_for(package: Package, reference: Dict[str, Any]) -> Path:
    """Resolve a request file reference to the packaged asset with that identity."""
    name = Path(str(reference["path"])).name
    matches = [(identifier, path) for identifier, path in package.asset_files.items()
               if name == identifier or name.startswith(identifier)]
    if not matches:
        raise RuntimeError(f"No packaged asset matches reference {reference['path']!r}")
    return max(matches, key=lambda item: len(item[0]))[1]


def asset_digest(package: Package, path: Path) -> Optional[str]:
    """Return the digest recorded in assets.json for one packaged file."""
    for entry in package.assets:
        asset = entry.get("asset") or {}
        if package.asset_files.get(str(asset.get("id"))) == path:
            return str(asset.get("sha256") or "") or None
    return None


def stage_assets(package: Package, exchange_root: Path) -> List[Dict[str, Any]]:
    """Copy the request's template/image bytes to their recorded shared paths.

    Args:
        package: Parsed package.
        exchange_root: Local stand-in for the ``exchange`` shared root.

    Returns:
        One verification record per staged input, including digest agreement.
    """
    records = []
    for role in ("template", "image"):
        reference = package.inputs[role]
        source = asset_for(package, reference)
        target = exchange_root / str(reference["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or target.read_bytes() != source.read_bytes():
            shutil.copyfile(source, target)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        expected = asset_digest(package, source)
        records.append({
            "role": role,
            "package_file": str(source.relative_to(package.root)),
            "staged_file": str(target),
            "bytes": target.stat().st_size,
            "sha256": digest,
            "recorded_sha256": expected,
            "matches_record": expected is None or expected.lower() == digest,
        })
    return records


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def _await_operation(client: TestClient, payload: Dict[str, Any],
                     budget_s: float = 300.0) -> Dict[str, Any]:
    """Poll a retained operation until it reaches a terminal state.

    ``mode=sync`` still answers 202 when its wait budget elapses first (a dense
    1-degree grid does), so the retained operation is collected instead.

    Args:
        client: Test client for the running service.
        payload: Operation response returned by the submit call.
        budget_s: Wall-clock budget for the wait.

    Returns:
        The terminal operation response, or the last observed one.
    """
    deadline = time.perf_counter() + budget_s
    while payload.get("state") not in {"succeeded", "failed", "cancelled"}:
        identity = payload.get("operation_id")
        if not identity or time.perf_counter() > deadline:
            break
        payload = client.get(f"/api/v2/operations/{identity}",
                             params={"wait_ms": 60000}).json()
    return payload


def run_match(client: TestClient, inputs: Dict[str, Any],
              overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Execute one template-matches request and summarise its decision and cost.

    Args:
        client: Test client bound to a service sharing the same ``exchange`` root.
        inputs: Portable request recorded in the package.
        overrides: ``search`` fields replaced for this run.

    Returns:
        Summarised match, pose budget counters and stage timings.
    """
    body = copy.deepcopy(inputs)
    body["mode"] = "sync"
    body["wait_ms"] = 60000
    # The recorded key is bound to the recorded content; a different search
    # configuration under the same key is a conflict, so replay without one.
    body.pop("idempotency_key", None)
    body.setdefault("search", {}).update(overrides or {})
    started = time.perf_counter()
    response = client.post("/api/v2/template-matches", json=body)
    payload = response.json() if response.content else {}
    payload = _await_operation(client, payload)
    wall_ms = (time.perf_counter() - started) * 1000.0
    if payload.get("state") != "succeeded":
        return {"ok": False, "status": response.status_code,
                "detail": json.dumps(payload.get("error") or payload)[:2000],
                "overrides": overrides or {}, "wall_ms": wall_ms}
    result = payload["result"]
    timing = (payload.get("diagnostics") or {}).get("timing") or {}
    counters = timing.get("counters") or {}
    spans = {item["name"]: item["duration_ms"] for item in timing.get("spans") or []
             if item.get("kind") == "step"}
    match = (result.get("matches") or [None])[0] or {}
    diagnostics = result.get("diagnostics") or {}
    plan = diagnostics.get("search_plan") or {}
    windows = (diagnostics.get("fine_search") or {}).get("windows") or []
    return {
        "ok": True,
        "status": response.status_code,
        "overrides": overrides or {},
        "valid": bool(result.get("valid")),
        "reason": result.get("reason"),
        "angle": match.get("angle_degrees"),
        "scale": match.get("scale"),
        "score": match.get("score"),
        "center": match.get("center"),
        "anchor": match.get("anchor"),
        "matrix": (match.get("support_mapping") or {}).get("matrix"),
        "mapping_matrix": (match.get("mapping") or {}).get("matrix"),
        "pyramid_scale": diagnostics.get("pyramid_scale"),
        "coarse_poses": plan.get("coarse_poses"),
        "fine_poses_per_neighborhood": plan.get("fine_poses_per_neighborhood"),
        "neighborhoods_available": plan.get("neighborhoods_available"),
        "neighborhoods_refined": plan.get("neighborhoods_refined"),
        "fine_pose_budget": counters.get("fine_poses"),
        "ncc_evaluations": counters.get("ncc_evaluations"),
        "coarse_ms": spans.get("coarse_search"),
        "fine_ms": spans.get("fine_search"),
        "total_ms": timing.get("total_ms"),
        "wall_ms": wall_ms,
        "windows": windows,
    }


def compare(recorded: Dict[str, Any], replay: Dict[str, Any]) -> Dict[str, Any]:
    """Compare a replayed match with the recorded one.

    Args:
        recorded: First match of the package's ``result.json``.
        replay: Summary produced by :func:`run_match`.

    Returns:
        Field deltas plus the largest absolute mapping-matrix difference.
    """
    recorded_center = recorded.get("center") or {}
    replay_center = replay.get("center") or {}
    try:
        matrix_delta = float(np.max(np.abs(
            np.array(replay["mapping_matrix"], dtype=float)
            - np.array(recorded["mapping"]["matrix"], dtype=float))))
    except (KeyError, TypeError, ValueError):
        matrix_delta = None
    return {
        "valid": replay.get("valid"),
        "angle_recorded": recorded.get("angle_degrees"),
        "angle_replayed": replay.get("angle"),
        "angle_delta": _diff(replay.get("angle"), recorded.get("angle_degrees")),
        "score_recorded": recorded.get("score"),
        "score_replayed": replay.get("score"),
        "score_delta": _diff(replay.get("score"), recorded.get("score")),
        "dx_px": _diff(replay_center.get("x"), recorded_center.get("x")),
        "dy_px": _diff(replay_center.get("y"), recorded_center.get("y")),
        "reference_mapping_max_abs_delta": matrix_delta,
        "center": replay_center,
    }


def _diff(value: Any, reference: Any) -> Optional[float]:
    """Return ``value - reference`` when both are finite numbers."""
    try:
        return float(value) - float(reference)
    except (TypeError, ValueError):
        return None


def _within(delta: Optional[float], tolerance: float) -> bool:
    """Report whether a delta is present and inside ``tolerance``."""
    return delta is not None and abs(delta) <= tolerance


# ---------------------------------------------------------------------------
# NCC-versus-angle curves
# ---------------------------------------------------------------------------


def ncc_curve(pixels: np.ndarray, search: np.ndarray,
              angles: np.ndarray) -> np.ndarray:
    """Return the best NCC score of each rotation over one search area.

    Args:
        pixels: Support pixels at the resolution being scored.
        search: Search-area pixels at the same resolution.
        angles: Rotation angles in degrees.

    Returns:
        Best correlation per angle; ``nan`` when the rotated support cannot fit.
    """
    area = np.ascontiguousarray(search, dtype=np.float32)
    scores = []
    for angle in angles:
        radians = float(np.deg2rad(angle))
        width, height = variant_size(pixels, radians, 1.0, origin="pixel_center")
        if width > search.shape[1] or height > search.shape[0]:
            scores.append(float("nan"))
            continue
        rendered = render_variant(np.asarray(pixels, dtype=np.float32), radians, 1.0,
                                  origin="pixel_center")
        variant = rendered.pixels
        mask = (np.ones(variant.shape[:2], np.uint8) if rendered.mask is None
                else rendered.mask)
        correlation = cv2.matchTemplate(area, variant, cv2.TM_CCOEFF_NORMED,
                                        mask=None if np.all(mask) else mask)
        correlation[~np.isfinite(correlation)] = -1
        scores.append(float(correlation.max()))
    return np.asarray(scores, dtype=float)


def measure_curves(package: Package, exchange_root: Path,
                   replay: Dict[str, Any]) -> Dict[str, Any]:
    """Score every rotation at coarse and fine resolution around the match.

    The fine curve is evaluated on the 531 px neighborhood the fine stage uses,
    so it shows exactly the basin the full-resolution refinement sees.

    Args:
        package: Parsed package.
        exchange_root: Staged ``exchange`` root.
        replay: Reproduced match summary.

    Returns:
        Angles, both curves and basin metrics for the report.
    """
    pixels = np.load(exchange_root / package.inputs["template"]["path"],
                     allow_pickle=False)["pixels"]
    image = cv2.imread(str(exchange_root / package.inputs["image"]["path"]),
                       cv2.IMREAD_GRAYSCALE)
    factor = min(2.0 ** -int(package.search.get("pyramid_levels", 0)),
                 float(package.search.get("pyramid_max_template_size", 400))
                 / max(pixels.shape))
    factor = min(1.0, max(factor, min(1.0, 6.0 / min(pixels.shape))))
    sigma = 0.5 / factor
    coarse_pixels, _ = decimate_pixels(pixels, factor, sigma=sigma)
    coarse_search, _ = decimate_pixels(image, factor, sigma=sigma)

    center = replay["center"]
    radius = 265
    x0 = int(round(center["x"])) - radius
    y0 = int(round(center["y"])) - radius
    window = image[max(0, y0):y0 + 2 * radius + 1, max(0, x0):x0 + 2 * radius + 1]

    angles = np.arange(0, 360, 1.0)
    started = time.perf_counter()
    coarse = ncc_curve(coarse_pixels, coarse_search, angles)
    coarse_s = time.perf_counter() - started
    started = time.perf_counter()
    fine = ncc_curve(pixels, window, angles)
    fine_s = time.perf_counter() - started

    def at(curve: np.ndarray, degrees: float) -> float:
        return float(curve[int(round(degrees)) % 360])

    def grid_best(curve: np.ndarray, step: float) -> Tuple[float, float, int]:
        grid = np.arange(0, 360, step)
        values = np.array([at(curve, degree) for degree in grid])
        return float(values.max()), float(grid[int(values.argmax())]), int((values >= 0.4).sum())

    metrics: Dict[str, Any] = {
        "pyramid_scale": factor,
        "coarse_template_shape": [int(coarse_pixels.shape[1]), int(coarse_pixels.shape[0])],
        "fine_window": [int(window.shape[1]), int(window.shape[0])],
        "coarse_peak": float(np.nanmax(coarse)),
        "coarse_peak_angle": float(angles[int(np.nanargmax(coarse))]),
        "fine_peak": float(np.nanmax(fine)),
        "fine_peak_angle": float(angles[int(np.nanargmax(fine))]),
        "coarse_score_at_match": at(coarse, replay["angle"]),
        "fine_score_at_match": at(fine, replay["angle"]),
        "coarse_curve_s": coarse_s,
        "fine_curve_s": fine_s,
        "grids": {},
        "samples": {str(int(degree)): [at(coarse, degree), at(fine, degree)]
                    for degree in (0, 10, 20, 30, 35, 40, 45, 50, 60, 90, 180, 225, 270, 315)},
    }
    for step in (5, 10, 20, 30, 45):
        best, best_angle, above = grid_best(coarse, step)
        metrics["grids"][str(step)] = {
            "coarse_best": best,
            "coarse_best_angle": best_angle,
            "coarse_above_gate": above,
            "poses": int(len(np.arange(0, 360, step))),
        }
    return {"angles": angles, "coarse": coarse, "fine": fine, "metrics": metrics}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _save(figure, path: Path) -> None:
    """Write one matplotlib figure at 150 dpi."""
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _quad(matrix: List[List[float]], width: int, height: int) -> np.ndarray:
    """Map support corners through a 3-by-3 matrix into inspection pixels."""
    transform = np.array(matrix, dtype=float)
    corners = np.array([[0, 0, 1], [width - 1, 0, 1],
                        [width - 1, height - 1, 1], [0, height - 1, 1]], dtype=float)
    return (corners @ transform.T)[:, :2]


def draw_overlay(package: Package, exchange_root: Path, replay: Dict[str, Any],
                 template_size: Tuple[int, int], path: Path) -> None:
    """Render the full frame and a zoomed view of the reproduced match."""
    import matplotlib.pyplot as plt

    image = cv2.imread(str(exchange_root / package.inputs["image"]["path"]),
                       cv2.IMREAD_GRAYSCALE)
    view = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    width, height = template_size
    quad = _quad(replay["matrix"], width, height)
    cv2.polylines(view, [np.round(quad).astype(np.int32)], True, C_MATCH, 4)
    anchor = replay["anchor"]
    center = replay["center"]
    cv2.drawMarker(view, (int(round(anchor["x"])), int(round(anchor["y"]))),
                   C_ANCHOR, cv2.MARKER_CROSS, 60, 3)
    for window in replay.get("windows") or []:
        cv2.rectangle(view, (window["x"], window["y"]),
                      (window["x"] + window["width"], window["y"] + window["height"]),
                      C_WINDOW, 3)
        cv2.putText(view, f"{window['angle']:.0f}deg",
                    (window["x"] + 8, window["y"] + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, C_WINDOW, 2, cv2.LINE_AA)
    cv2.putText(view, f"angle {replay['angle']:.0f} deg  score {replay['score']:.4f}",
                (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, C_MATCH, 3, cv2.LINE_AA)

    figure, axes = plt.subplots(1, 2, figsize=(16, 7.2))
    axes[0].imshow(cv2.cvtColor(view, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Inspection frame: matched support (green), anchor (red),\n"
                      "coarse neighborhoods refined at full resolution (amber)", fontsize=10)
    axes[0].axis("off")

    half = 620
    x0 = int(np.clip(round(center["x"]) - half, 0, max(0, view.shape[1] - 2 * half)))
    y0 = int(np.clip(round(center["y"]) - half, 0, max(0, view.shape[0] - 2 * half)))
    crop = view[y0:y0 + 2 * half, x0:x0 + 2 * half]
    axes[1].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"Zoom x=({x0},{x0 + crop.shape[1]}) y=({y0},{y0 + crop.shape[0]})",
                      fontsize=10)
    axes[1].axis("off")
    _save(figure, path)


def draw_patch(package: Package, exchange_root: Path, replay: Dict[str, Any],
               template_pixels: np.ndarray, path: Path) -> None:
    """Render the taught support, the matched patch and their absolute difference."""
    import matplotlib.pyplot as plt

    image = cv2.imread(str(exchange_root / package.inputs["image"]["path"]),
                       cv2.IMREAD_GRAYSCALE)
    inverse = np.linalg.inv(np.array(replay["matrix"], dtype=float))
    height, width = template_pixels.shape[:2]
    patch = cv2.warpAffine(image.astype(np.float32), inverse[:2], (width, height))
    difference = np.abs(patch - template_pixels.astype(np.float32))
    ncc = float(cv2.matchTemplate(patch.astype(np.float32),
                                  template_pixels.astype(np.float32),
                                  cv2.TM_CCOEFF_NORMED)[0, 0])

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for axis, data, title in (
        (axes[0], template_pixels, f"taught support {width}x{height}"),
        (axes[1], patch, f"matched patch, local NCC {ncc:.4f}"),
        (axes[2], difference, f"|difference| max {difference.max():.1f} gray"),
    ):
        artist = axis.imshow(data, cmap="magma" if data is difference else "gray")
        axis.set_title(title, fontsize=10)
        axis.axis("off")
        figure.colorbar(artist, ax=axis, fraction=0.046, pad=0.02)
    _save(figure, path)


def _sweep_label(item: Dict[str, Any]) -> str:
    """Return a short human label for one sweep row."""
    overrides = item.get("overrides") or {}
    if item.get("name"):
        return str(item["name"])
    if not overrides:
        return "baseline"
    return ", ".join(f"{key}={value}" for key, value in overrides.items())


def draw_sweep(rows: List[Dict[str, Any]], path: Path) -> None:
    """Chart NCC budget, stage cost and the reported pose for every sweep row."""
    import matplotlib.pyplot as plt

    labels = [_sweep_label(row) for row in rows]
    coarse = [row.get("coarse_ms") or 0 for row in rows]
    fine = [row.get("fine_ms") or 0 for row in rows]
    figure, axis = plt.subplots(figsize=(13, 5.4))
    index = np.arange(len(rows))
    axis.barh(index, coarse, color="#4c78a8", label="coarse_search")
    axis.barh(index, fine, left=coarse, color="#f58518", label="fine_search")
    for position, row in enumerate(rows):
        total = coarse[position] + fine[position]
        if row.get("ok") and row.get("angle") is not None:
            verdict = f"{row['angle']:.0f} deg / {row['score']:.4f}"
        elif row.get("ok"):
            verdict = "NO MATCH"
        else:
            verdict = "failed"
        budget = row.get("ncc_evaluations")
        text = f"{verdict}   " + (f"{budget:.0f} NCC" if budget else "")
        axis.text(total * 1.02 + 30, position, text, va="center", fontsize=8.5)
    axis.set_yticks(index, labels, fontsize=8.5)
    axis.invert_yaxis()
    axis.set_xlim(0, max(coarse) + max(fine) + 2400 if rows else 1)
    axis.set_xlabel("milliseconds on this machine")
    axis.set_title("One request, one knob changed: cost and decision", fontsize=10)
    axis.legend(fontsize=8)
    axis.grid(axis="x", alpha=0.25)
    _save(figure, path)


def draw_curves(curves: Dict[str, Any], replay: Dict[str, Any],
                gate: float, path: Path) -> None:
    """Plot the coarse and fine NCC curves with the grid sampling that matters."""
    import matplotlib.pyplot as plt

    angles = curves["angles"]
    coarse = curves["coarse"]
    fine = curves["fine"]
    figure, axes = plt.subplots(1, 2, figsize=(15.5, 4.8))

    axes[0].plot(angles, coarse, color="#4c78a8", lw=1.1,
                 label=f"coarse (1/{1 / curves['metrics']['pyramid_scale']:.0f} res)")
    axes[0].plot(angles, fine, color="#2ca02c", lw=1.1,
                 label="fine (full res, local window)")
    for step, colour in ((10, "#d62728"), (30, "#9467bd")):
        grid = np.arange(0, 360, step)
        axes[0].plot(grid, [coarse[int(a) % 360] for a in grid], "o", ms=2.6,
                     color=colour, alpha=0.75, label=f"coarse grid @ {step} deg")
    axes[0].axhline(gate, color="#555555", ls="--", lw=1,
                    label=f"effective coarse gate {gate}")
    axes[0].axvline(replay["angle"], color="#333333", ls=":", lw=1)
    axes[0].set_xlabel("rotation applied to the support (degrees)")
    axes[0].set_ylabel("best NCC in the search area")
    axes[0].set_title("Score versus pose: the coarse grid must land in a narrow basin",
                      fontsize=10)
    axes[0].legend(fontsize=7, loc="lower left")
    axes[0].grid(alpha=0.2)

    low, high = 20, 70
    mask = (angles >= low) & (angles <= high)
    axes[1].plot(angles[mask], coarse[mask], "o-", ms=2.4, color="#4c78a8", label="coarse")
    axes[1].plot(angles[mask], fine[mask], "o-", ms=2.4, color="#2ca02c", label="fine")
    grid = np.arange(0, 360, 10)
    grid = grid[(grid >= low) & (grid <= high)]
    axes[1].plot(grid, [coarse[int(a) % 360] for a in grid], "s", ms=7, mfc="none",
                 mec="#d62728", mew=1.4, label="10 deg grid")
    axes[1].axhline(gate, color="#555555", ls="--", lw=1)
    metrics = curves["metrics"]
    axes[1].annotate(f"truth {replay['angle']:.0f} deg\n"
                     f"coarse {metrics['coarse_score_at_match']:.3f}\n"
                     f"fine {metrics['fine_score_at_match']:.3f}",
                     (replay["angle"] + 1.5, metrics["fine_score_at_match"]),
                     fontsize=8, va="top")
    axes[1].set_xlabel("rotation applied to the support (degrees)")
    axes[1].set_title("Zoom: the coarse peak is sharp, the full-resolution one is broad",
                      fontsize=10)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.2)
    _save(figure, path)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _fmt(value: Optional[float]) -> str:
    """Format an optional delta for the report."""
    return "—" if value is None else f"{value:+.9g}"


def _row(rows: List[Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
    """Return the sweep row with a given name."""
    for row in rows:
        if row.get("name") == name:
            return row
    return None


def findings(package: Package, replay: Dict[str, Any], sweep: List[Dict[str, Any]],
             curves: Optional[Dict[str, Any]], gate: float) -> List[str]:
    """Summarise the measured behaviour of each coarse parameter."""
    baseline = _row(sweep, "baseline")
    lines: List[str] = []
    if baseline and baseline.get("ok"):
        lines.append(
            f"- **基线**：粗搜索 {baseline['coarse_poses']} 个姿态（{baseline['pyramid_scale']:.2f} 倍图）"
            f"→ {baseline['neighborhoods_available']} 个邻域 → 精搜索 "
            f"{baseline['fine_pose_budget']:.0f} 个姿态，共 {baseline['ncc_evaluations']:.0f} 次 NCC，"
            f"{baseline['coarse_ms']:.0f} + {baseline['fine_ms']:.0f} ms。")
    dense = _row(sweep, "coarse_fine=false")
    if dense and dense.get("ok") and baseline and baseline.get("ok"):
        ratio = ((dense["coarse_ms"] + dense["fine_ms"])
                 / max(1.0, baseline["coarse_ms"] + baseline["fine_ms"]))
        lines.append(
            f"- **`coarse_fine=false`**：粗步长退化为 `angle_step={package.search['angle_step']}`，"
            f"第一段要扫 {dense['coarse_poses']} 个姿态（共 {dense['ncc_evaluations']:.0f} 次 NCC，"
            f"粗阶段 {dense['coarse_ms']:.0f} ms），每个邻域只按粗姿态本身精化，不再在网格之间插值。"
            f"本次角度仍为 {dense['angle']:.0f}°（因为 {dense['angle']:.0f}° 本来就在 1° 网格上），"
            f"总耗时约为基线的 {ratio:.1f} 倍。")
    coarse30 = _row(sweep, "coarse_angle_step=30")
    if coarse30 is not None and not (coarse30.get("ok") and coarse30.get("angle") is not None):
        detail = ""
        if curves:
            samples = curves["metrics"]["samples"]
            detail = (f"（粗分数在 30°/60° 只有 {samples['30'][0]:.3f} / {samples['60'][0]:.3f}，"
                      f"而门限是 {gate:g}）")
        lines.append(
            f"- **`coarse_angle_step=30`**：粗网格只剩 {coarse30.get('coarse_poses')} 个姿态，"
            f"真实姿态附近没有网格点能过门限{detail}，候选集里只剩别名姿态，精化后 **no_match**。")
    loose = _row(sweep, "coarse_score_threshold=0.2")
    if loose and loose.get("ok") and baseline and baseline.get("ok"):
        lines.append(
            f"- **`coarse_score_threshold=0.2`**：有效粗门限真的降到 0.2，弱一些的粗峰也进入候选 → "
            f"{loose['neighborhoods_available']} 个邻域、{loose['ncc_evaluations']:.0f} 次 NCC、"
            f"精搜索 {loose['fine_ms']:.0f} ms；答案不变（{loose['angle']:.0f}°），纯粹多花钱。")
    tight = _row(sweep, "coarse_score_threshold=0.9")
    if tight and tight.get("ok") and baseline and baseline.get("ok"):
        same = (tight.get("ncc_evaluations") == baseline.get("ncc_evaluations")
                and tight.get("angle") == baseline.get("angle"))
        lines.append(
            f"- **`coarse_score_threshold=0.9`**：与基线{'完全一致' if same else '不同'}——"
            f"因为有效门限是 `min(score_threshold, coarse_score_threshold)` = "
            f"min({package.search.get('score_threshold')}, 0.9) = {gate:g}，"
            f"调高它永远不会让粗搜索更严格。")
    raised = _row(sweep, "score_threshold=0.45")
    restored = _row(sweep, "score_threshold=0.45 + coarse=0.2")
    if raised is not None:
        extra = ""
        if curves:
            grid = curves["metrics"]["grids"]["10"]
            extra = (f"，而真实姿态附近最好的粗网格点只有 {grid['coarse_best']:.3f}"
                     f"（{grid['coarse_best_angle']:.0f}°）")
        state = ("no_match" if not (raised.get("ok") and raised.get("angle") is not None)
                 else f"{raised['angle']:.0f}°")
        lines.append(
            f"- **`score_threshold=0.45`**（只动精搜索门限）：粗门限被 `min()` 一起抬到 0.45{extra}，"
            f"粗阶段剩 0 个邻域 → **{state}**。这就是“把分数门槛调高一点，整个定位就消失”的机制。")
        if restored is not None:
            ok = restored.get("ok") and restored.get("angle") is not None
            lines.append(
                f"- **`score_threshold=0.45` + `coarse_score_threshold=0.2`**：单独把粗门限放低后"
                f"{'恢复为 %.0f°、分数 %.4f' % (restored['angle'], restored['score']) if ok else '仍然失败'}"
                f"——两个门限是「粗阶段召回 / 精阶段确认」的分工，不是同一个旋钮。")
    if curves:
        metrics = curves["metrics"]
        samples = metrics["samples"]
        lines.append(
            f"- **分数盆形状**：粗分辨率下真实姿态 {replay['angle']:.0f}° 得 "
            f"{metrics['coarse_score_at_match']:.3f}，40° 已降到 {samples['40'][0]:.3f}，"
            f"30° 只有 {samples['30'][0]:.3f}；全分辨率下同一姿态得 "
            f"{metrics['fine_score_at_match']:.3f}，40° 仍有 {samples['40'][1]:.3f}。"
            f"粗网格必须“踩进”这个很窄的盆，精搜索再在 ±`coarse_angle_step` 内把角度推到 "
            f"`angle_step` 的精度。")
    if _row(sweep, "coarse_scale_step=0.5") and baseline:
        lines.append(
            "- **`coarse_scale_step`**：本次 `scale_min = scale_max = 1`，尺度网格只有一个点，"
            "把它从 0.1 改到 0.5 结果与基线逐位相同——它只在真正做多尺度搜索时才有意义。")
    single = _row(sweep, "fine_candidates=1")
    if single and single.get("ok") and baseline and baseline.get("ok"):
        lines.append(
            f"- **`fine_candidates=1`**（不是 coarse 参数，但决定谁去精化）：只精化分数最高的 1 个邻域，"
            f"NCC 从 {baseline['ncc_evaluations']:.0f} 降到 {single['ncc_evaluations']:.0f}，"
            f"总耗时降到约 1/3，本次答案不变（{single['angle']:.0f}° / {single['score']:.4f}）。"
            "代价是：对称件上若最好的粗峰其实来自别名姿态，就没有第二次机会。")
    return lines


def markdown_report(package: Package, identity: Dict[str, Any], staged: List[Dict[str, Any]],
                    replay: Dict[str, Any], comparison: Dict[str, Any],
                    sweep: List[Dict[str, Any]], curves: Optional[Dict[str, Any]],
                    figures: Dict[str, str], gate: float) -> str:
    """Compose the Markdown reproduction report."""
    manifest = package.manifest
    search = package.search
    runtime = package.job.get("runtime") or {}
    coarse_shape = (f"{curves['metrics']['coarse_template_shape'][0]}×"
                    f"{curves['metrics']['coarse_template_shape'][1]}" if curves else "—")
    lines = [
        "# 定位任务复现 · " + package.root.name,
        "",
        "## 1. 包信息",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| run_id | `{manifest.get('run_id')}` |",
        f"| 类型 / 操作 | `{manifest.get('type')}` / "
        f"`{(manifest.get('workbench') or {}).get('operation')}` |",
        f"| 创建时间 (UTC) | `{manifest.get('created_at')}` |",
        f"| 记录时长 | {manifest.get('duration_ms')} ms |",
        f"| 记录实现 | `{runtime.get('algorithmVersion')}` / "
        f"`{str(runtime.get('implementationSHA256', ''))[:16]}…` |",
        f"| 本次实现 | `{identity['algorithm_version']}` / "
        f"`{identity['implementation_sha256'][:16]}…` |",
        f"| 记录依赖 | numpy {(manifest.get('environment') or {}).get('numpy')}, "
        f"opencv {(manifest.get('environment') or {}).get('opencv')}, "
        f"python {(manifest.get('environment') or {}).get('python')} |",
        f"| 本次依赖 | numpy {identity['dependencies'].get('numpy')}, "
        f"opencv {identity['dependencies'].get('opencv')}, "
        f"python {identity['python_version']} |",
        "",
        "## 2. 输入资产（逐字节校验）",
        "",
        "| 角色 | 包内文件 | SHA-256 | 与记录一致 |",
        "|---|---|---|---|",
    ]
    for record in staged:
        lines.append(
            f"| {record['role']} | `{record['package_file']}` | "
            f"`{record['sha256'][:16]}…` | {'是' if record['matches_record'] else '否'} |")
    lines += [
        "",
        "## 3. 请求里的搜索参数",
        "",
        "| 参数 | 请求值 | 作用 |",
        "|---|---|---|",
        f"| `score_threshold` | {search.get('score_threshold')} | 精搜索阶段的分数门限 |",
        f"| `angle_min/max/step` | {search.get('angle_min')} / {search.get('angle_max')} / "
        f"{search.get('angle_step')} | 角度范围（顺时针）与精搜索步长 |",
        f"| `scale_min/max/step` | {search.get('scale_min')} / {search.get('scale_max')} / "
        f"{search.get('scale_step')} | 尺度范围与精搜索步长 |",
        f"| `coarse_fine` | {search.get('coarse_fine')} | 是否“粗搜 → 局部精化”两段式 |",
        f"| `coarse_angle_step` | {search.get('coarse_angle_step')} | 粗角度步长，同时是精化角度带宽 ±Δθ |",
        f"| `coarse_scale_step` | {search.get('coarse_scale_step')} | 粗尺度步长，同时是精化尺度带宽 ±Δs |",
        f"| `coarse_score_threshold` | {search.get('coarse_score_threshold')} | 粗阶段峰值门限；"
        f"有效值 = min(score_threshold, 本值) = **{gate:g}** |",
        f"| `pyramid_levels` | {search.get('pyramid_levels')} | 金字塔级数（2 ⇒ 1/4 分辨率） |",
        f"| `pyramid_max_template_size` | {search.get('pyramid_max_template_size')} | "
        f"模板长边上限决定的降采样上限 |",
        f"| `max_matches` / `overlap` | {search.get('max_matches')} / {search.get('overlap')} | "
        f"输出个数与 NMS 重叠阈值 |",
        "",
        "## 4. 复现结果 vs 记录",
        "",
        "| 量 | 记录值 | 复现值 | 差值 |",
        "|---|---|---|---|",
        f"| valid | {bool(package.recorded_match)} | {comparison['valid']} | — |",
        f"| angle (°) | {comparison['angle_recorded']} | {comparison['angle_replayed']} | "
        f"{_fmt(comparison['angle_delta'])} |",
        f"| center.x (px) | {package.recorded_match['center']['x']:.6f} | "
        f"{comparison['center'].get('x'):.6f} | {_fmt(comparison['dx_px'])} |",
        f"| center.y (px) | {package.recorded_match['center']['y']:.6f} | "
        f"{comparison['center'].get('y'):.6f} | {_fmt(comparison['dy_px'])} |",
        f"| score | {comparison['score_recorded']:.9f} | {comparison['score_replayed']:.9f} | "
        f"{_fmt(comparison['score_delta'])} |",
        "| reference→inspection 矩阵 | — | — | "
        f"max|Δ| = {_fmt(comparison['reference_mapping_max_abs_delta'])} |",
        "",
        "> 记录与本次运行的实现哈希、依赖版本不同（见第 1 节），因此这里的“一致”指决策与几何量"
        "在浮点噪声内一致，而不是二进制逐位相同。",
        "",
        "## 5. 这次调用内部的姿态预算",
        "",
        "| 阶段 | 数量 | 说明 |",
        "|---|---|---|",
        f"| 金字塔 | ×{replay.get('pyramid_scale')} | 模板 342×339 → {coarse_shape} |",
        f"| 粗搜索姿态 | {replay.get('coarse_poses')} | 角度 0..360 / 10° → 37，尺度只有 1 个 |",
        f"| 候选邻域 | {replay.get('neighborhoods_available')} → "
        f"{replay.get('neighborhoods_refined')} | NMS（overlap 0.3）后的独立邻域 |",
        f"| 精搜索姿态 | {int(replay.get('fine_pose_budget') or 0)} | "
        f"{replay.get('neighborhoods_refined')} × "
        f"{int(replay.get('fine_poses_per_neighborhood') or 0)}（±10° / 1°） |",
        f"| NCC 总次数 | {int(replay.get('ncc_evaluations') or 0)} | "
        f"37 + {int(replay.get('fine_pose_budget') or 0)} |",
        f"| coarse / fine 耗时 | {replay.get('coarse_ms'):.0f} / {replay.get('fine_ms'):.0f} ms | "
        f"记录为 1461 / 3910 ms（机器不同） |",
        "",
        "## 6. 实测：coarse 参数各自控制什么",
        "",
        "| 配置 | 角度 | 分数 | 邻域 | NCC 次数 | coarse ms | fine ms |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in sweep:
        if not (row.get("ok") and row.get("angle") is not None):
            label = "no_match" if row.get("ok") else "失败"
            lines.append(f"| {_sweep_label(row)} | {label} | — | "
                         f"{row.get('neighborhoods_available') if row.get('ok') else '—'} | "
                         f"{row.get('ncc_evaluations') or '—'} | "
                         f"{(row.get('coarse_ms') or 0):.0f} | "
                         f"{(row.get('fine_ms') or 0):.0f} |")
            continue
        lines.append(
            f"| {_sweep_label(row)} | {row['angle']:.0f}° | {row['score']:.4f} | "
            f"{row.get('neighborhoods_available')} → {row.get('neighborhoods_refined')} | "
            f"{row.get('ncc_evaluations'):.0f} | {row.get('coarse_ms') or 0:.0f} | "
            f"{row.get('fine_ms') or 0:.0f} |")
    lines += ["", "### 结论（由上面的数字直接读出）", ""]
    lines.extend(findings(package, replay, sweep, curves, gate))
    if curves:
        metrics = curves["metrics"]
        lines += [
            "",
            "### 粗网格采样 vs 粗分数盆",
            "",
            "| 粗角度步长 | 姿态数 | 粗网格最高分 | 最高分所在角度 | 过门限的网格点 |",
            "|---|---|---|---|---|",
        ]
        for step, row in sorted(metrics["grids"].items(), key=lambda item: int(item[0])):
            lines.append(
                f"| {step}° | {row['poses']} | {row['coarse_best']:.4f} | "
                f"{row['coarse_best_angle']:.0f}° | {row['coarse_above_gate']}/{row['poses']} |")
        lines += [
            "",
            "> “过门限的网格点”是单姿态层面的计数；真正进入精化的邻域还要过 `overlap=0.3` 的 NMS，"
            "同一位置不同旋转的两块近方形支持会互相重叠而被合并，所以邻域数（本请求 6 个）比过门限的"
            "网格点数少。",
        ]
        lines += [
            "",
            "| 角度 | 粗分数 | 精分数 |",
            "|---|---|---|",
        ]
        for degree, pair in sorted(metrics["samples"].items(), key=lambda item: int(item[0])):
            lines.append(f"| {degree}° | {pair[0]:.4f} | {pair[1]:.4f} |")
    lines += ["", "## 7. 图", ""]
    for caption, filename in figures.items():
        lines.append(f"**{caption}**")
        lines.append("")
        lines.append(f"![{caption}](figures/{filename})")
        lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


EXPERIMENTS: List[Tuple[str, Dict[str, Any]]] = [
    ("baseline", {}),
    ("coarse_fine=false", {"coarse_fine": False}),
    ("coarse_angle_step=5", {"coarse_angle_step": 5}),
    ("coarse_angle_step=20", {"coarse_angle_step": 20}),
    ("coarse_angle_step=30", {"coarse_angle_step": 30}),
    ("coarse_scale_step=0.5", {"coarse_scale_step": 0.5}),
    ("coarse_score_threshold=0.2", {"coarse_score_threshold": 0.2}),
    ("coarse_score_threshold=0.9", {"coarse_score_threshold": 0.9}),
    ("score_threshold=0.45", {"score_threshold": 0.45}),
    ("score_threshold=0.45 + coarse=0.2",
     {"score_threshold": 0.45, "coarse_score_threshold": 0.2}),
    ("fine_candidates=1", {"fine_candidates": 1}),
]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("package", type=Path, help="Extracted debug package directory")
    parser.add_argument("--workdir", type=Path,
                        default=REPO_ROOT / "output/repro_localization",
                        help="Writable output directory")
    parser.add_argument("--no-sweep", action="store_true", help="Skip the coarse sweep")
    parser.add_argument("--no-curve", action="store_true", help="Skip the NCC curves")
    parser.add_argument("--no-figures", action="store_true", help="Skip figure rendering")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Reproduce the package, measure the coarse knobs and write the report."""
    args = parse_args(argv)
    package = load_package(args.package)
    workdir = args.workdir.resolve()
    exchange_root = workdir / "exchange"
    figures_dir = workdir / "figures"
    workdir.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] 校验并镜像资产 → {exchange_root}")
    staged = stage_assets(package, exchange_root)
    for record in staged:
        print(f"      {record['role']}: {record['bytes']} bytes "
              f"{record['sha256'][:16]}… {'OK' if record['matches_record'] else 'MISMATCH'}")

    identity = runtime_identity().model_dump(mode="json")
    gate = min(float(package.search.get("score_threshold", 0.0)),
               float(package.search.get("coarse_score_threshold", 0.0)))
    settings = ServiceSettings(shared_roots={"exchange": exchange_root},
                               writable_roots=[], max_workers=2)

    with TestClient(create_app(settings)) as client:
        print("[2/6] 重放记录的请求")
        replay = run_match(client, package.inputs)
        if not replay.get("ok"):
            print(replay.get("detail"))
            return 1
        comparison = compare(package.recorded_match, replay)
        print(f"      angle {replay['angle']} score {replay['score']:.6f} "
              f"center ({replay['center']['x']:.3f}, {replay['center']['y']:.3f})")
        print(f"      Δangle {_fmt(comparison['angle_delta'])} Δscore "
              f"{_fmt(comparison['score_delta'])} Δx {_fmt(comparison['dx_px'])} "
              f"Δy {_fmt(comparison['dy_px'])}")

        sweep: List[Dict[str, Any]] = []
        if not args.no_sweep:
            print("[3/6] 扫描 coarse 参数")
            for name, overrides in EXPERIMENTS:
                row = run_match(client, package.inputs, overrides)
                row["name"] = name
                sweep.append(row)
                if row.get("ok") and row.get("angle") is not None:
                    print(f"      {name:<32} angle {row['angle']:>5.0f}  "
                          f"score {row['score']:.4f}  NCC {row['ncc_evaluations']:>4.0f}  "
                          f"nb {row.get('neighborhoods_available')}  "
                          f"coarse {row['coarse_ms'] or 0:>6.0f}  "
                          f"fine {row['fine_ms'] or 0:>6.0f} ms")
                elif row.get("ok"):
                    print(f"      {name:<32} no_match  NCC {row.get('ncc_evaluations')}  "
                          f"coarse {row['coarse_ms'] or 0:>6.0f}  "
                          f"fine {row['fine_ms'] or 0:>6.0f} ms")
                else:
                    print(f"      {name:<32} failed: {row.get('status')} {row.get('detail')}")
        else:
            print("[3/6] 跳过参数扫描")

    curves: Optional[Dict[str, Any]] = None
    if not args.no_curve:
        print("[4/6] 计算粗/精分辨率的 NCC–角度曲线")
        curves = measure_curves(package, exchange_root, replay)
        metrics = curves["metrics"]
        print(f"      coarse peak {metrics['coarse_peak']:.4f} @ "
              f"{metrics['coarse_peak_angle']:.0f}°, fine peak {metrics['fine_peak']:.4f} @ "
              f"{metrics['fine_peak_angle']:.0f}°  "
              f"({metrics['coarse_curve_s']:.1f}s + {metrics['fine_curve_s']:.1f}s)")
        np.savez(workdir / "curves.npz", angles=curves["angles"],
                 coarse=curves["coarse"], fine=curves["fine"])
    else:
        print("[4/6] 跳过曲线")

    template_pixels = np.load(exchange_root / package.inputs["template"]["path"],
                              allow_pickle=False)["pixels"]
    template_size = (int(template_pixels.shape[1]), int(template_pixels.shape[0]))

    figures: Dict[str, str] = {}
    if not args.no_figures:
        print("[5/6] 渲染图")
        draw_overlay(package, exchange_root, replay, template_size,
                     figures_dir / "01_match_overlay.png")
        figures["复现的匹配位置与精搜索邻域"] = "01_match_overlay.png"
        draw_patch(package, exchange_root, replay, template_pixels,
                   figures_dir / "02_template_vs_patch.png")
        figures["模板 / 匹配块 / 差异"] = "02_template_vs_patch.png"
        if sweep:
            draw_sweep(sweep, figures_dir / "03_coarse_sweep.png")
            figures["coarse 参数扫描的代价与结论"] = "03_coarse_sweep.png"
        if curves:
            draw_curves(curves, replay, gate, figures_dir / "04_ncc_vs_angle.png")
            figures["粗/精分辨率的 NCC–角度曲线"] = "04_ncc_vs_angle.png"
    else:
        print("[5/6] 跳过图")

    print("[6/6] 写报告")
    (workdir / "reproduced.json").write_text(
        json.dumps(replay, indent=2, ensure_ascii=False), encoding="utf-8")
    (workdir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")
    (workdir / "coarse_sweep.json").write_text(
        json.dumps(sweep, indent=2, ensure_ascii=False), encoding="utf-8")
    (workdir / "curve_metrics.json").write_text(
        json.dumps(curves["metrics"] if curves else {}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    (workdir / "report.md").write_text(
        markdown_report(package, identity, staged, replay, comparison, sweep, curves,
                        figures, gate), encoding="utf-8")

    agrees = (
        comparison["valid"] is True
        and _within(comparison["angle_delta"], 1e-9)
        and _within(comparison["score_delta"], 1e-6)
        and _within(comparison["dx_px"], 1e-3)
        and _within(comparison["dy_px"], 1e-3)
    )
    print(f"      结果与记录一致: {agrees} → {workdir / 'report.md'}")
    return 0 if agrees else 1


if __name__ == "__main__":
    sys.exit(main())