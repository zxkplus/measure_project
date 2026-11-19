"""Reproduce the 2026-11-09 tooth-outer ``template-matches`` operation on v2.

The sample directory holds one modeling capture (``frame-05-best.png``), the
portable template the upstream produced from its search configuration
(``template.npz``) and the algorithm job records. This script re-submits the
same work to a running v2 service, compares the observation with the recorded
one, renders the localization on the image and collects timings.

Example::

    python scripts/repro_tooth_profile.py \
        --sample-dir "/media/.../mvp-tooth-profile-20261109" \
        --workdir output/repro-tooth \
        --base-url http://127.0.0.1:9014

Only files under ``--workdir`` are written; the sample directory is read-only.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

IMAGE_NAME = "frame-05-best.png"
TEMPLATE_NAME = "template.npz"
IMAGE_SHA256 = "ef95cdcf0fe9183575a72156d5c4cee40bc2c335d5ba4fe9e5781328a47ddd46"
TEMPLATE_SHA256 = "d28902c9934ec2f2bb3547b0acd7b57a14063d13fa28e38dff38c9e5c37bf688"

# Upstream camelCase search configuration -> v2 TemplateSearch fields.
SEARCH_FIELD_MAP = {
    "scoreThreshold": "score_threshold",
    "maxMatches": "max_matches",
    "overlap": "overlap",
    "subpixel": "subpixel",
    "angleMin": "angle_min",
    "angleMax": "angle_max",
    "angleStep": "angle_step",
    "scaleMin": "scale_min",
    "scaleMax": "scale_max",
    "scaleStep": "scale_step",
    "coarseFine": "coarse_fine",
    "coarseScoreThreshold": "coarse_score_threshold",
    "coarseAngleStep": "coarse_angle_step",
    "coarseScaleStep": "coarse_scale_step",
    "pyramidLevels": "pyramid_levels",
    "pyramidMaxTemplateSize": "pyramid_max_template_size",
}


class Client:
    """Minimal HTTP caller that separates request, first byte and terminal timings."""

    def __init__(self, base_url: str, caller_id: str = "tooth-repro") -> None:
        self.base_url = base_url.rstrip("/")
        self.caller_id = caller_id

    def _open(self, path: str, payload: Optional[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """POST/GET one route and return its JSON body plus wall-clock marks.

        Args:
            path: Route relative to ``/api/v2``.
            payload: Optional JSON body; ``None`` issues a read request.

        Returns:
            Decoded response and millisecond marks for the round trip.
        """
        data = None if payload is None else json.dumps(payload, allow_nan=False).encode()
        request = urllib.request.Request(
            self.base_url + "/api/v2/" + path.lstrip("/"), data=data,
            headers={"Content-Type": "application/json", "X-Caller-Id": self.caller_id},
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                raw = response.read()
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", "replace")
            raise RuntimeError(f"HTTP {error.code}: {body}") from error
        finished = time.perf_counter()
        return json.loads(raw), {"wall_ms": (finished - started) * 1000.0}

    def calculate(self, route: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Submit a calculation and poll it without resubmitting work.

        Args:
            route: Operation route (e.g. ``/template-matches``).
            payload: JSON body including scheduling controls.

        Returns:
            Terminal operation record extended with ``client_timing``.
        """
        submitted = time.perf_counter()
        record, timing = self._open(route, payload)
        submit_ms = timing["wall_ms"]
        polls = 0
        poll_ms: List[float] = []
        while record["state"] in {"queued", "running", "cancelling"}:
            body, timing = self._open(
                f"/operations/{record['operation_id']}?wait_ms=1000", None,
            )
            polls += 1
            poll_ms.append(timing["wall_ms"])
            if time.perf_counter() - submitted > payload.get("timeout_ms", 60000) / 1000 + 60:
                raise TimeoutError("Polling budget exceeded")
            record = body
        record["client_timing"] = {
            "submit_wall_ms": submit_ms,
            "terminal_wall_ms": (time.perf_counter() - submitted) * 1000.0,
            "poll_calls": polls,
            "poll_wall_ms_total": float(np.sum(poll_ms)) if poll_ms else 0.0,
        }
        return record


def file_reference(ns: str, name: str, digest: str) -> Dict[str, Any]:
    """Return an exchange-root file reference with a verified digest.

    Args:
        ns: Namespace directory under the ``exchange`` root.
        name: File name inside that namespace.
        digest: Expected SHA-256 of the exact bytes.

    Returns:
        JSON file reference accepted by v2.
    """
    return {"root": "exchange", "path": f"{ns}/{name}", "sha256": digest}


def upstream_search(sample: Path) -> Dict[str, Any]:
    """Translate the saved upstream search configuration into a v2 search body.

    Args:
        sample: Sample directory holding ``search-configuration.json``.

    Returns:
        v2 ``TemplateSearch`` payload; ``region`` is omitted for whole-image search.
    """
    configuration = json.loads((sample / "search-configuration.json").read_text(encoding="utf-8"))
    localization = configuration["edit"]["localization"]
    search: Dict[str, Any] = {}
    for source, target in SEARCH_FIELD_MAP.items():
        if source in localization["search"]:
            search[target] = localization["search"][source]
    return search


def localization_roi(sample: Path) -> Dict[str, Any]:
    """Return the taught support rectangle in v2 ``RotatedROI`` shape.

    Args:
        sample: Sample directory holding ``search-configuration.json``.

    Returns:
        ROI payload for ``/template-builds``.
    """
    roi = json.loads((sample / "search-configuration.json").read_text(
        encoding="utf-8"))["edit"]["localization"]["roi"]
    return {
        "center": {"x": roi["cx"], "y": roi["cy"]},
        "width": roi["width"], "height": roi["height"],
        "angle_degrees": roi["angle"],
    }


def recorded_operation(sample: Path) -> Dict[str, Any]:
    """Load the recorded best operation result.

    Args:
        sample: Sample directory holding ``operation-best.json``.

    Returns:
        Recorded operation record.
    """
    return json.loads((sample / "operation-best.json").read_text(encoding="utf-8"))


def compare(recorded: Dict[str, Any], actual: Dict[str, Any]) -> Dict[str, Any]:
    """Diff observable localization outcomes field by field.

    Args:
        recorded: Recorded operation from the upstream job.
        actual: Replayed operation from the local v2 service.

    Returns:
        Per-field differences with tolerances and pass flags.
    """
    rows = []

    def add(label: str, expected: Any, got: Any, tolerance: float, kind: str = "abs") -> None:
        delta = None
        if isinstance(expected, (int, float)) and isinstance(got, (int, float)):
            delta = float(got) - float(expected)
        ok = delta is not None and abs(delta) <= tolerance
        rows.append({"field": label, "recorded": expected, "reproduced": got,
                     "delta": delta, "tolerance": tolerance, "kind": kind, "pass": bool(ok)})

    want = recorded["result"]
    got = actual["result"]
    add("state.succeeded", True, actual["state"] == "succeeded", 0, "exact")
    add("result.valid", want["valid"], got["valid"], 0, "exact")
    rows.append({"field": "result.reason", "recorded": want.get("reason"),
                 "reproduced": got.get("reason"), "delta": None,
                 "tolerance": 0, "kind": "exact",
                 "pass": want.get("reason") == got.get("reason")})
    add("diagnostics.pyramid_scale", want["diagnostics"]["pyramid_scale"],
        got["diagnostics"]["pyramid_scale"], 0, "exact")
    for key in ("preprocessing", "metric_frame", "normalization_recovers_size"):
        rows.append({"field": f"diagnostics.{key}",
                     "recorded": want["diagnostics"].get(key),
                     "reproduced": got["diagnostics"].get(key), "delta": None,
                     "tolerance": 0, "kind": "exact",
                     "pass": want["diagnostics"].get(key) == got["diagnostics"].get(key)})
    add("matches.count", len(want["matches"]), len(got["matches"]), 0, "exact")
    if want["matches"] and got["matches"]:
        w, g = want["matches"][0], got["matches"][0]
        add("match.score", w["score"], g["score"], 1e-5)
        add("match.angle_degrees", w["angle_degrees"], g["angle_degrees"], 1e-9)
        add("match.scale", w["scale"], g["scale"], 1e-9)
        for axis in ("x", "y"):
            add(f"match.center.{axis}", w["center"][axis], g["center"][axis], 1e-3, "px")
            add(f"match.anchor.{axis}", w["anchor"][axis], g["anchor"][axis], 1e-3, "px")
            add(f"match.support_mapping.{axis}", w["support_mapping"]["matrix"][0 if axis == "x" else 1][2],
                g["support_mapping"]["matrix"][0 if axis == "x" else 1][2], 1e-3, "px")
            add(f"match.mapping.{axis}", w["mapping"]["matrix"][0 if axis == "x" else 1][2],
                g["mapping"]["matrix"][0 if axis == "x" else 1][2], 1e-3, "px")
    return {"rows": rows, "all_pass": all(row["pass"] for row in rows)}


def render_result(image: np.ndarray, match: Dict[str, Any], path: Path,
                  crop: bool = False) -> Path:
    """Draw the located support rectangle, center and reference shift.

    Args:
        image: Inspection image (gray or BGR).
        match: One match observation from ``/template-matches``.
        path: Output PNG path.
        crop: When true, render a full-resolution window around the center.

    Returns:
        Written path.
    """
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    matrix = np.array(match["support_mapping"]["matrix"], dtype=float)
    width = float(match.get("__width", 1975))
    height = float(match.get("__height", 487))
    half_w, half_h = (width - 1) / 2, (height - 1) / 2
    corners = np.array([[-half_w, -half_h, 1], [half_w, -half_h, 1],
                        [half_w, half_h, 1], [-half_w, half_h, 1]])
    box = (matrix @ corners.T).T[:, :2]
    center = (float(match["center"]["x"]), float(match["center"]["y"]))
    reference = (float(match["anchor"]["x"]), float(match["anchor"]["y"]))
    dx = matrix[0, 2]
    dy = matrix[1, 2]

    if crop:
        x0 = int(max(0, center[0] - 600))
        y0 = int(max(0, center[1] - 300))
        canvas = canvas[y0:y0 + 600, x0:x0 + 1200].copy()
        shift = np.array([x0, y0], dtype=float)
        box = box - shift
        center = (center[0] - shift[0], center[1] - shift[1])
        reference = (reference[0] - shift[0], reference[1] - shift[1])
        scale = 1.0
    else:
        scale = min(1600.0 / canvas.shape[1], 1600.0 / canvas.shape[0])
        shift = np.zeros(2)

    def s(points: np.ndarray) -> np.ndarray:
        return np.round((points - shift) * scale).astype(np.int32)

    poly = s(box)
    cv2.polylines(canvas, [poly.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
    center_px = tuple(np.round(np.array(center)).astype(int))
    ref_px = tuple(np.round(np.array(reference)).astype(int))
    for point, color in ((ref_px, (255, 128, 0)), (center_px, (0, 0, 255))):
        size = 40 if crop else 12
        cv2.line(canvas, (point[0] - size, point[1]), (point[0] + size, point[1]), color, 2)
        cv2.line(canvas, (point[0], point[1] - size), (point[0], point[1] + size), color, 2)
    for value, origin, color in ((f"inspected seed ({center[0]:.1f}, {center[1]:.1f})", center_px, (0, 0, 255)),
                                 (f"reference anchor ({reference[0]:.1f}, {reference[1]:.1f})", ref_px, (255, 128, 0))):
        text_y = origin[1] - (12 if origin is center_px else 26)
        cv2.putText(canvas, value, (origin[0] - 260, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6 if crop else 0.9, color, 2, cv2.LINE_AA)
    label = (f"score={match['score']:.6f}  angle={match['angle_degrees']:.1f} deg  "
             f"scale={match['scale']:.3f}   shift dx={dx:.3f}px dy={dy:.3f}px")
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 46 if crop else 34), (0, 0, 0), -1)
    cv2.putText(canvas, label, (12, 24 if crop else 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.6 if crop else 0.75, (0, 255, 255), 2, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)
    return path


def profile_in_process(workdir: Path, search: Dict[str, Any], with_profile: bool) -> Dict[str, Any]:
    """Run the same request in-process to get per-function timings cProfile can see.

    Args:
        workdir: Repro workspace providing settings and exchange files.
        search: v2 search payload.
        with_profile: Attach cProfile for ``matchTemplate``-level breakdown.

    Returns:
        Wall time and, when profiled, cProfile text and function shares.
    """
    import sys

    repository = Path(__file__).resolve().parents[1]
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))
    from measure_api.v2.configuration import load_settings
    from measure_api.v2.families.templates import TemplateMatchRequest, match_template
    from measure_api.v2.models import FileReference
    from measure_api.v2.registry import CalculationContext

    settings = load_settings(workdir / "settings.json")
    request = TemplateMatchRequest(
        template=FileReference(**file_reference("tooth-20261109", TEMPLATE_NAME, TEMPLATE_SHA256)),
        image=FileReference(**file_reference("tooth-20261109", IMAGE_NAME, IMAGE_SHA256)),
        search=search,
    )
    context = CalculationContext(
        settings=settings, operation_id="in-process-profile", trace_id="in-process-profile",
    )
    started = time.perf_counter()
    result = match_template(request, context)
    wall_ms = (time.perf_counter() - started) * 1000.0
    summary = {"wall_ms": wall_ms, "score": result.matches[0].score if result.matches else None}
    if with_profile:
        profiler = cProfile.Profile()
        profiler.enable()
        match_template(request, context)
        profiler.disable()
        stream = io.StringIO()
        pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(18)
        summary["cprofile"] = stream.getvalue()
    return summary


def main() -> None:
    """Replay the recorded operation, compare it, render and time it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:9014")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--skip-rebuild", action="store_true")
    args = parser.parse_args()

    sample = args.sample_dir.resolve()
    workdir = args.workdir.resolve()
    client = Client(args.base_url)
    namespace = "tooth-20261109"
    search = upstream_search(sample)
    report: Dict[str, Any] = {"sample_dir": str(sample), "base_url": args.base_url,
                              "search": search, "stages": {}}

    if client._open("health/ready", None)[0]["status"] != "ready":
        raise SystemExit("v2 is not ready")

    record = client.calculate("/template-inspections", {
        "template": file_reference(namespace, TEMPLATE_NAME, TEMPLATE_SHA256),
        "timeout_ms": 60000, "idempotency_key": "tooth-inspect-1",
    })
    report["stages"]["template_inspections"] = {
        "state": record["state"], "compute_ms": record.get("diagnostics", {}).get("compute_ms"),
        "wall_ms": record["client_timing"]["terminal_wall_ms"],
        "support": (record.get("result") or {}).get("metadata", {}).get("support"),
    }

    if not args.skip_rebuild:
        rebuild = client.calculate("/template-builds", {
            "image": file_reference(namespace, IMAGE_NAME, IMAGE_SHA256),
            "roi": localization_roi(sample), "support_mode": "tight_roi",
            "preprocessing": {"type": "raw"},
            "output": {"directory": {"root": "outputs", "path": "rebuild"}},
            "timeout_ms": 60000, "idempotency_key": "tooth-rebuild-1",
        })
        rebuilt = rebuild["result"]["template"]
        roots = {"outputs": workdir / "outputs", "exchange": workdir / "exchange",
                 "assets": workdir / "assets"}
        rebuilt_file = roots[rebuilt["root"]] / rebuilt["path"]
        with np.load(rebuilt_file) as archive:
            rebuilt_pixels = archive["pixels"]
        with np.load(sample / TEMPLATE_NAME) as archive:
            given_pixels = archive["pixels"]
        difference = np.abs(rebuilt_pixels.astype(np.float64) - given_pixels.astype(np.float64))
        report["stages"]["template_build_rebuild"] = {
            "state": rebuild["state"], "compute_ms": rebuild["diagnostics"].get("compute_ms"),
            "sha256_matches_supplied": rebuilt["sha256"] == TEMPLATE_SHA256,
            "max_abs_diff": float(difference.max()),
            "identical_fraction": float((difference == 0).mean()),
            "bit_identical": bool(np.array_equal(rebuilt_pixels, given_pixels)),
        }

    payload = {
        "template": file_reference(namespace, TEMPLATE_NAME, TEMPLATE_SHA256),
        "image": file_reference(namespace, IMAGE_NAME, IMAGE_SHA256),
        "search": search, "mode": "async", "timeout_ms": 180000,
        "idempotency_key": "tooth-match-20261109-op4",
    }
    (workdir / "request-match.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    runs = []
    for index in range(args.repeat):
        body = dict(payload)
        body["idempotency_key"] = f"tooth-match-20261109-r{index + 1}"
        operation = client.calculate("/template-matches", body)
        if index == 0:
            (workdir / "response-match.json").write_text(
                json.dumps(operation, indent=2, ensure_ascii=False), encoding="utf-8")
        diagnostics = operation.get("diagnostics", {})
        runs.append({
            "run": index + 1, "state": operation["state"],
            "compute_ms": diagnostics.get("compute_ms"),
            "queue_ms": diagnostics.get("queue_ms"),
            "preparation_ms": diagnostics.get("preparation_ms"),
            "worker_boot_spawn_ms": diagnostics.get("worker_boot_spawn_ms"),
            "worker_reused": diagnostics.get("worker_reused"),
            "peak_worker_bytes": diagnostics.get("peak_worker_bytes"),
            "cache": diagnostics.get("cache"),
            "client": operation["client_timing"],
            "score": operation["result"]["matches"][0]["score"] if operation["result"]["matches"] else None,
            "center": operation["result"]["matches"][0]["center"] if operation["result"]["matches"] else None,
        })
        first = operation
    report["stages"]["template_matches_runs"] = runs
    comparison = compare(recorded_operation(sample), first)
    report["comparison"] = comparison
    (workdir / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    image = cv2.imread(str(workdir / "exchange" / namespace / IMAGE_NAME), cv2.IMREAD_UNCHANGED)
    match = dict(first["result"]["matches"][0])
    match["__width"], match["__height"] = 1975, 487
    started = time.perf_counter()
    overlay = render_result(image, match, workdir / "overlay.png")
    crop = render_result(image, match, workdir / "overlay-crop.png", crop=True)
    report["render_ms"] = (time.perf_counter() - started) * 1000.0
    report["overlay"] = str(overlay)
    report["overlay_crop"] = str(crop)

    import cv2 as cv2_module
    report["environment"] = {
        "python": __import__("sys").version.split()[0],
        "numpy": np.__version__, "opencv": cv2_module.__version__,
        "cv2_threads": cv2_module.getNumThreads(), "cpu_count": __import__("os").cpu_count(),
    }
    report["stages"]["in_process"] = profile_in_process(workdir, search, with_profile=True)
    (workdir / "repro-report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "stages"}, indent=2))
    print(json.dumps(report["stages"], indent=2)[:6000])


if __name__ == "__main__":
    main()