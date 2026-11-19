"""Replay exported v2 debug packages and compare their measurements.

A production export contains, per task, a ``request.json`` (the original
``POST /api/v2/runs`` body), ``assets.json`` (which local file backs each
exchange path) and ``result.json`` (what the service answered at the time).

This script stages the same bytes into a temporary exchange root, replays the
request through the in-process app, and dumps a normalised result so two code
states can be compared field by field.

Usage::

    # record what the current tree answers for every exported task
    python scripts/replay_v2_debug_package.py --data "<export dir>" \
        --out tests/output/real/head.json

    # after a refactor, record again and compare
    python scripts/replay_v2_debug_package.py --data "<export dir>" \
        --out tests/output/real/after.json
    python scripts/replay_v2_debug_package.py --compare \
        tests/output/real/head.json tests/output/real/after.json

    # also diff a recorded replay against the exported production result
    python scripts/replay_v2_debug_package.py --data "<export dir>" \
        --out tests/output/real/head.json --against-export
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from fastapi.testclient import TestClient

from measure_api.v2.app import create_app
from measure_api.v2.models import ServiceSettings

#: Keys that legitimately differ between two runs of the same request.
VOLATILE_KEYS = {
    "operation_id", "trace_id", "preview_token", "duration_ms", "duration",
    "started_at", "finished_at", "created_at", "updated_at", "queued_at",
    "completed_at", "stage_timings", "engine_timing", "performance", "manifest",
    "runtime_identity", "upstream_revision", "idempotency_key", "storage",
    "worker_pid", "worker_id", "cache_hits", "cache_misses", "elapsed_ms",
}
VOLATILE_SUFFIXES = ("_ms", "_at", "_monotonic", "_bytes", "_nanos")


def _prune(value: Any) -> Any:
    """Drop volatile keys and absolute paths so two runs are comparable."""
    if isinstance(value, dict):
        kept = {}
        for key, item in value.items():
            if key in VOLATILE_KEYS or key.endswith(VOLATILE_SUFFIXES):
                continue
            if key in {"sha256", "path", "root", "file"}:
                continue
            kept[key] = _prune(item)
        return kept
    if isinstance(value, list):
        return [_prune(item) for item in value]
    return value


def _stage_assets(package: Path, exchange: Path) -> None:
    """Copy every packaged byte to the exchange path the request expects."""
    record = json.loads((package / "assets.json").read_text(encoding="utf-8"))
    for entry in record:
        target = exchange / entry["reference"]["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(package / entry["file"], target)


def _route(request: Dict[str, Any]) -> str:
    """Return the v2 route an exported request belongs to.

    Exports mix whole recipe runs with single-operation previews, so the route
    is chosen from the request shape rather than assumed.
    """
    if "recipe" in request:
        return "/runs"
    if "template" in request and "search" in request:
        return "/template-matches"
    if "template" in request:
        return "/template-inspections"
    raise ValueError(f"unrecognised exported request: {sorted(request)}")


def replay_one(package: Path) -> Dict[str, Any]:
    """Replay one exported task and return its operation result."""
    request = json.loads((package / "request.json").read_text(encoding="utf-8"))
    request.pop("idempotency_key", None)
    route = _route(request)
    with tempfile.TemporaryDirectory(prefix="v2replay-") as scratch:
        root = Path(scratch)
        for alias in ("exchange", "assets", "outputs"):
            (root / alias).mkdir()
        _stage_assets(package, root / "exchange")
        settings = ServiceSettings(
            shared_roots={"exchange": root / "exchange", "assets": root / "assets",
                          "outputs": root / "outputs"},
            writable_roots=["outputs"],
            state_directory=root / "state",
            max_workers=2,
        )
        with TestClient(create_app(settings)) as client:
            response = client.post("/api/v2" + route, json=request)
            if response.status_code not in (200, 202):
                return {"status": "http_error", "code": response.status_code,
                        "text": response.text[:2000]}
            operation = response.json()
            while operation["state"] in {"queued", "running", "cancelling"}:
                operation = client.get(
                    f"/api/v2/operations/{operation['operation_id']}?wait_ms=30000",
                ).json()
            if operation["state"] != "succeeded":
                return {"status": operation["state"],
                        "error": operation.get("error"),
                        "diagnostics": _prune(operation.get("diagnostics") or {})}
            return _prune(operation.get("result") or {})


def _tasks(data: Path) -> List[Path]:
    packages = []
    for index_file in sorted(data.glob("算法调试包/任务索引.json")):
        for entry in json.loads(index_file.read_text(encoding="utf-8")):
            package = index_file.parent / entry["folder"]
            if (package / "request.json").exists():
                packages.append(package)
    return packages


def record(data: Path, out: Path) -> Dict[str, Any]:
    """Replay every exported task and write the normalised results."""
    results = {}
    for package in _tasks(data):
        label = package.name.split(" · ")[0]
        print(f"replaying {label} ...", flush=True)
        results[label] = replay_one(package)
        status = results[label].get("status", "ok")
        print(f"  -> {status}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1, sort_keys=True),
                   encoding="utf-8")
    print(f"wrote {out}")
    return results


def _diff(base: Any, candidate: Any, path: str, tol: float, out: List[dict]) -> None:
    """Collect numeric or structural differences below one JSON node."""
    if isinstance(base, dict) and isinstance(candidate, dict):
        for key in sorted(set(base) | set(candidate)):
            _diff(base.get(key), candidate.get(key), f"{path}.{key}", tol, out)
        return
    if isinstance(base, list) and isinstance(candidate, list):
        if len(base) != len(candidate):
            out.append({"path": path, "kind": "length", "base": len(base),
                        "candidate": len(candidate)})
            return
        for index, (left, right) in enumerate(zip(base, candidate)):
            _diff(left, right, f"{path}[{index}]", tol, out)
        return
    if isinstance(base, bool) or isinstance(candidate, bool):
        if base != candidate:
            out.append({"path": path, "kind": "value", "base": base, "candidate": candidate})
        return
    if isinstance(base, (int, float)) and isinstance(candidate, (int, float)):
        delta = abs(float(base) - float(candidate))
        if delta > tol * max(1.0, abs(float(base))):
            out.append({"path": path, "kind": "numeric", "base": base,
                        "candidate": candidate, "delta": delta})
        return
    if base != candidate:
        out.append({"path": path, "kind": "value", "base": base, "candidate": candidate})


def compare(base_path: Path, candidate_path: Path, tol: float) -> int:
    """Print every difference between two recorded replays."""
    base = json.loads(base_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    total = 0
    for task in sorted(set(base) | set(candidate)):
        differences: List[dict] = []
        _diff(base.get(task), candidate.get(task), task, tol, differences)
        total += len(differences)
        print(f"{task}: {len(differences)} difference(s)")
        for entry in differences[:20]:
            print(f"  {entry['path']}: {entry}")
    print(f"\n{total} difference(s) across {len(base)} task(s)")
    return total


def against_export(data: Path, recorded: Dict[str, Any], tol: float) -> int:
    """Compare a replay against the production result shipped in the export."""
    total = 0
    for package in _tasks(data):
        label = package.name.split(" · ")[0]
        exported = json.loads((package / "result.json").read_text(encoding="utf-8"))
        replayed = recorded.get(label, {})
        differences: List[dict] = []
        _diff(_prune(exported), replayed, label, tol, differences)
        total += len(differences)
        print(f"{label}: {len(differences)} difference(s) vs exported result")
        for entry in differences[:10]:
            print(f"  {entry['path']}: {entry}")
    print(f"\n{total} difference(s) against exported results")
    return total


def main(argv: List[str] | None = None) -> int:
    """Parse arguments and either record or compare a replay."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="exported detection-data directory")
    parser.add_argument("--out", type=Path, help="where to write the normalised replay")
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("BASE", "CANDIDATE"),
                        help="compare two recorded replays")
    parser.add_argument("--against-export", action="store_true",
                        help="also diff the replay against the shipped result.json")
    parser.add_argument("--tol", type=float, default=1e-9, help="relative numeric tolerance")
    args = parser.parse_args(argv)
    if args.compare:
        return 1 if compare(args.compare[0], args.compare[1], args.tol) else 0
    if not args.data or not args.out:
        parser.error("--data and --out are required unless --compare is used")
    recorded = record(args.data, args.out)
    if args.against_export:
        return 1 if against_export(args.data, recorded, args.tol) else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())