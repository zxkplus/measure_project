#!/usr/bin/env python3
"""Replay exported measure-v2 debug packages and report where their time goes.

The measurement workbench exports one directory per algorithm call containing
``request.json`` (the exact submitted request), ``result.json`` (the recorded
observations), ``assets.json`` and the referenced images/templates.  This script
stages that content into a private shared-root tree, replays the recorded
request through the same in-process executor the HTTP service uses, and reports
the bounded operation timing plus a numeric comparison against the recorded
result.

It is the offline counterpart of ``scripts/v2_benchmark.py``: no server, no
cameras, deterministic inputs, and per-span attribution instead of one latency
number.

Usage::

    # one exported cycle (directories named "任务/..." are discovered)
    python scripts/v2_timing_benchmark.py \
        --package-dir "/media/.../检测数据-.../算法调试包/任务" \
        --repeats 3 --threads 1 --tag baseline

    # a single package
    python scripts/v2_timing_benchmark.py --package-dir "algorithm debug/任务/007-底图 · ..."

Exit code is 0 when every replay reproduced the recorded observations within
tolerance and no operation failed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import shutil
import statistics
import sys
import tempfile
import time
from typing import Any, Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from measure_api.v2.app import create_app  # noqa: E402
from measure_api.v2.models import ServiceSettings  # noqa: E402

PREFIX = "/api/v2"

#: Recorded-result paths that carry business observations.  Anything outside
#: these subtrees is provenance, artifacts or timing and is not compared.
COMPARED_PATHS = {
    "template-matches": ["/matches", "/valid", "/reason"],
    "runs": ["/targets", "/valid", "/status"],
}

#: Span families reported in the summary, slowest first.
REPORT_SPANS = (
    "template_match", "fine_search", "coarse_search", "pyramid_prepare",
    "image_decode_cache", "template_load", "file_read_verify", "preprocessing",
    "grayscale", "localization", "target", "target_alignment", "artifacts_commit",
    "terminal_persist",
)


@dataclass
class Package:
    """One exported algorithm call with everything needed to replay it."""

    root: Path
    name: str
    request: Dict[str, Any]
    result: Dict[str, Any]
    manifest: Dict[str, Any]
    assets: List[Dict[str, Any]] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)

    @property
    def route(self) -> str:
        """Return the public calculation route this request was submitted to."""
        if "recipe" in self.request:
            return "/runs"
        if "template" in self.request and "image" in self.request:
            return "/template-matches"
        raise RuntimeError(f"{self.name}: unrecognised request body {sorted(self.request)}")


def read_json(path: Path, default: Any) -> Any:
    """Read a JSON document, returning ``default`` when the file is absent."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def discover(target: Path) -> List[Package]:
    """Collect replayable packages below ``target``.

    Args:
        target: A single package directory or a directory holding package
            subdirectories (the exported ``算法调试包/任务`` layout).

    Returns:
        Packages that carry both ``request.json`` and ``result.json``.

    Raises:
        FileNotFoundError: No package directory matched.
    """
    candidates: List[Path]
    if (target / "request.json").exists():
        candidates = [target]
    else:
        candidates = sorted({path.parent for path in target.rglob("request.json")})
    packages: List[Package] = []
    for root in candidates:
        if (root / "导出失败.txt").exists():
            continue
        request = read_json(root / "request.json", None)
        result = read_json(root / "result.json", None)
        if not isinstance(request, dict) or not isinstance(result, dict):
            continue
        packages.append(Package(
            root=root, name=root.name, request=request, result=result,
            manifest=read_json(root / "manifest.json", {}),
            assets=read_json(root / "assets.json", []),
        ))
    if not packages:
        raise FileNotFoundError(f"No debug package with request.json/result.json under {target}")
    return packages


def references(value: Any) -> Iterable[Dict[str, Any]]:
    """Yield every shared-file reference (a mapping with ``root`` and ``path``)."""
    if isinstance(value, dict):
        if "root" in value and "path" in value and isinstance(value["path"], str):
            yield value
        for item in value.values():
            yield from references(item)
    elif isinstance(value, list):
        for item in value:
            yield from references(item)


def staged_sources(package: Package) -> Dict[str, Path]:
    """Map recorded shared paths to the packaged file holding their bytes.

    Args:
        package: Parsed export.

    Returns:
        ``path`` -> packaged source file.  Assets recorded with an explicit
        reference win; otherwise the packaged asset identifier is matched
        against the reference basename, which is how the export names files.
    """
    by_path: Dict[str, Path] = {}
    by_identifier: Dict[str, Path] = {}
    for entry in package.assets:
        source = package.root / str(entry.get("file") or "")
        if not source.is_file():
            continue
        asset = entry.get("asset") or {}
        identifier = str(asset.get("id") or "")
        if identifier:
            by_identifier[identifier] = source
        reference = entry.get("reference") or {}
        path = reference.get("path")
        if isinstance(path, str):
            by_path[path] = source
    for reference in references(package.request):
        path = str(reference["path"])
        if path in by_path:
            continue
        name = Path(path).name
        matches = [key for key in by_identifier if name == key or name.startswith(key)]
        if matches:
            by_path[path] = by_identifier[max(matches, key=len)]
    return by_path


def stage(package: Package, workdir: Path) -> Dict[str, Any]:
    """Copy packaged assets to the shared paths the recorded request names."""
    exchange = workdir / "exchange"
    outputs = workdir / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    sources = staged_sources(package)
    copied, missing = [], []
    for reference in references(package.request):
        path = str(reference["path"])
        source = sources.get(path)
        if source is None:
            missing.append(path)
            continue
        destination = exchange / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        copied.append({"path": path, "bytes": destination.stat().st_size})
    return {"exchange": exchange, "outputs": outputs, "copied": copied, "missing": missing}


def payload_for(package: Package, attempt: int) -> Dict[str, Any]:
    """Return the request body for one replay attempt.

    The scheduling controls are replaced so every repeat actually executes
    (the exported idempotency key would otherwise return the retained
    operation) and output directories never collide.
    """
    body = json.loads(json.dumps(package.request))
    body["mode"] = "sync"
    body["wait_ms"] = 60000
    body.pop("idempotency_key", None)
    directory = body.get("output", {}).get("directory") if isinstance(body.get("output"), dict) else None
    if isinstance(directory, dict) and isinstance(directory.get("path"), str):
        directory["path"] = f"{directory['path'].rstrip('/')}/attempt-{attempt}"
    return body


def replay(package: Package, client: TestClient, attempt: int) -> Dict[str, Any]:
    """Submit one attempt and return its operation document."""
    started = time.perf_counter()
    response = client.post(PREFIX + package.route, json=payload_for(package, attempt))
    elapsed_ms = (time.perf_counter() - started) * 1000
    if response.status_code not in (200, 201, 202):
        raise RuntimeError(f"{package.name}: HTTP {response.status_code} {response.text[:600]}")
    document = response.json()
    if document.get("state") not in {"succeeded", "failed", "cancelled", "timed_out"}:
        raise RuntimeError(f"{package.name}: operation stayed {document.get('state')}")
    document["wall_ms"] = elapsed_ms
    return document


def numeric_differences(expected: Any, actual: Any, path: str = "",
                        tolerance: float = 0.0) -> List[str]:
    """List numeric or structural differences between two observation trees."""
    found: List[str] = []
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            location = f"{path}/{key}"
            if key not in expected or key not in actual:
                found.append(f"{location}: missing on one side")
            else:
                found.extend(numeric_differences(expected[key], actual[key], location, tolerance))
        return found
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            found.append(f"{path}: length {len(expected)} != {len(actual)}")
            return found
        for index, (first, second) in enumerate(zip(expected, actual)):
            found.extend(numeric_differences(first, second, f"{path}/{index}", tolerance))
        return found
    if isinstance(expected, bool) or isinstance(actual, bool):
        if expected != actual:
            found.append(f"{path}: {expected!r} != {actual!r}")
        return found
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        delta = abs(float(expected) - float(actual))
        if delta > tolerance:
            found.append(f"{path}: {expected!r} != {actual!r} (delta {delta:.3g})")
        return found
    if isinstance(expected, str) and isinstance(actual, str):
        return found
    if expected != actual:
        found.append(f"{path}: {expected!r} != {actual!r}")
    return found


def observations(package: Package, document: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the business observations this benchmark gates on."""
    reproduced = document.get("result") or {}
    return {pointer: reproduced.get(pointer.strip("/"))
            for pointer in COMPARED_PATHS.get(package.route.strip("/"), [])}


def compare(package: Package, document: Dict[str, Any], tolerance: float) -> Dict[str, Any]:
    """Compare the replay with the recorded observations and report drift."""
    reproduced = document.get("result") or {}
    differences: List[str] = []
    for pointer in COMPARED_PATHS.get(package.route.strip("/"), []):
        key = pointer.strip("/")
        differences.extend(numeric_differences(
            package.result.get(key), reproduced.get(key), pointer, tolerance))
    return {"differences": differences[:50], "difference_count": len(differences),
            "valid": reproduced.get("valid", reproduced.get("status"))}


def compare_reports(previous: Dict[str, Any], current: Dict[str, Any],
                    tolerance: float) -> List[str]:
    """Compare two benchmark reports package by package on their observations.

    Args:
        previous: Earlier report holding ``observations`` per package.
        current: Report produced by this run.
        tolerance: Allowed absolute numeric difference.

    Returns:
        Human-readable drift descriptions; empty when every observation agrees.
    """
    by_name = {package["name"]: package for package in previous.get("packages") or []}
    drift: List[str] = []
    for package in current.get("packages") or []:
        before = by_name.get(package["name"])
        if before is None:
            drift.append(f"{package['name']}: absent from the baseline report")
            continue
        for pointer, expected in (before.get("observations") or {}).items():
            actual = (package.get("observations") or {}).get(pointer)
            for difference in numeric_differences(expected, actual, pointer, tolerance):
                drift.append(f"{package['name'][:40]}: {difference}")
    return drift


def span_table(timings: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Summarise retained spans by name across attempts (median and max)."""
    collected: Dict[str, List[float]] = {}
    for timing in timings:
        for span in timing.get("spans") or []:
            if "duration_ms" not in span:
                continue
            collected.setdefault(str(span["name"]), []).append(float(span["duration_ms"]))
    summary = {}
    for name, values in collected.items():
        summary[name] = {"count": len(values), "median_ms": statistics.median(values),
                         "max_ms": max(values), "total_ms": sum(values)}
    return summary


def counter_table(timings: List[Dict[str, Any]]) -> Dict[str, float]:
    """Median of every accumulated counter across attempts."""
    collected: Dict[str, List[float]] = {}
    for timing in timings:
        for name, value in (timing.get("counters") or {}).items():
            collected.setdefault(str(name), []).append(float(value))
    return {name: statistics.median(values) for name, values in sorted(collected.items())}


def markdown(summary: Dict[str, Any]) -> str:
    """Render one package summary as a Markdown section."""
    lines = [f"### {summary['name']}", "",
             f"route `{summary['route']}`, attempts {summary['attempts']}, "
             f"engine median **{summary['total_ms_median']:.0f} ms** "
             f"(max {summary['total_ms_max']:.0f} ms), "
             f"HTTP median {summary['wall_ms_median']:.0f} ms", ""]
    if summary["differences"]:
        lines.append(f"Numeric drift: {summary['difference_count']} field(s), "
                     f"first: `{summary['differences'][0]}`")
        lines.append("")
    rows = [(name, values) for name, values in summary["spans"].items()]
    rows.sort(key=lambda item: -item[1]["total_ms"])
    lines += ["| span | n | median ms | total ms |", "|---|---|---|---|"]
    for name, values in rows[:12]:
        lines.append(f"| `{name}` | {values['count']} | {values['median_ms']:.1f} "
                     f"| {values['total_ms']:.1f} |")
    if summary["counters"]:
        lines += ["", "| counter | value |", "|---|---|"]
        for name, value in summary["counters"].items():
            lines.append(f"| `{name}` | {value:.0f} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    """Replay every discovered package and write the timing report."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1,
                        help="MEASURE_V2_SEARCH_THREADS for pose-grid parallelism")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--tolerance", type=float, default=1e-6,
                        help="Allowed drift against the recorded result; the export may "
                             "have been produced by a different OpenCV build")
    parser.add_argument("--compare-to", type=Path, default=None,
                        help="Previous benchmark JSON; gates observations with strict tolerance")
    parser.add_argument("--strict-tolerance", type=float, default=1e-9)
    parser.add_argument("--tag", default="run")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "output" / "bench")
    parser.add_argument("--limit", type=int, default=0, help="Replay at most N packages")
    arguments = parser.parse_args()

    session_started = time.perf_counter()
    os.environ["MEASURE_V2_SEARCH_THREADS"] = str(arguments.threads)
    packages = discover(arguments.package_dir.resolve())
    if arguments.limit:
        packages = packages[:arguments.limit]

    arguments.output.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {"tag": arguments.tag, "threads": arguments.threads,
                              "repeats": arguments.repeats, "packages": []}
    failed = 0
    for package in packages:
        workdir = Path(tempfile.mkdtemp(prefix="measure-bench-"))
        try:
            staged = stage(package, workdir)
            settings = ServiceSettings(
                shared_roots={"exchange": staged["exchange"], "outputs": staged["outputs"]},
                writable_roots=["outputs"], max_workers=max(1, arguments.workers),
                state_directory=workdir / "state",
            )
            timings: List[Dict[str, Any]] = []
            walls: List[float] = []
            documents: List[Dict[str, Any]] = []
            with TestClient(create_app(settings)) as client:
                for attempt in range(arguments.repeats):
                    document = replay(package, client, attempt)
                    timing = (document.get("diagnostics") or {}).get("timing") or {}
                    timings.append(timing)
                    walls.append(float(document["wall_ms"]))
                    documents.append(document)
            totals = [float(timing.get("total_ms") or 0.0) for timing in timings]
            verdicts = [compare(package, document, arguments.tolerance) for document in documents]
            summary = {
                "name": package.name,
                "route": package.route,
                "attempts": len(timings),
                "missing_inputs": staged["missing"],
                "total_ms_median": statistics.median(totals) if totals else 0.0,
                "total_ms_max": max(totals) if totals else 0.0,
                "wall_ms_median": statistics.median(walls) if walls else 0.0,
                "spans": span_table(timings),
                "counters": counter_table(timings),
                "differences": verdicts[0]["differences"] if verdicts else [],
                "difference_count": max((item["difference_count"] for item in verdicts), default=0),
                "valid": verdicts[0]["valid"] if verdicts else None,
                "observations": observations(package, documents[0]) if documents else {},
                "spans_detail": [timing.get("spans") or [] for timing in timings],
                "search_plan": ((documents[0].get("result") or {}).get("diagnostics") or {}
                                ).get("search_plan") if documents else None,
            }
            report["packages"].append(summary)
            status = "OK" if not summary["difference_count"] else f"DRIFT {summary['difference_count']}"
            print(f"{package.route:18} {package.name[:52]:52} "
                  f"{summary['total_ms_median']:8.1f} ms  "
                  f"{arguments.repeats}x  {status}")
            if summary["difference_count"]:
                failed += 1
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    report["wall_seconds"] = round(time.perf_counter() - session_started, 3)
    output_json = arguments.output / f"{arguments.tag}.json"
    output_markdown = arguments.output / f"{arguments.tag}.md"
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    output_markdown.write_text(
        "# measure-v2 timing benchmark: " + arguments.tag + "\n\n" + "\n".join(
            markdown(summary) for summary in report["packages"]), encoding="utf-8")
    print(f"\nwrote {output_json}\nwrote {output_markdown}")

    strict_failed = 0
    if arguments.compare_to is not None:
        previous = json.loads(arguments.compare_to.read_text(encoding="utf-8"))
        drift = compare_reports(previous, report, arguments.strict_tolerance)
        report["drift_vs"] = str(arguments.compare_to)
        report["drift"] = drift[:200]
        output_json.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        if drift:
            strict_failed = 1
            print(f"\nSTRICT DRIFT vs {arguments.compare_to} ({len(drift)} field(s)):")
            for line in drift[:20]:
                print("  " + line)
        else:
            print(f"\nstrict comparison vs {arguments.compare_to}: identical "
                  f"(tolerance {arguments.strict_tolerance:g})")
    return 1 if (failed or strict_failed) else 0


if __name__ == "__main__":
    raise SystemExit(main())