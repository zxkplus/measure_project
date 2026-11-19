"""A/B comparison harness for v2 operations.

Runs a directory of recorded requests against the in-process FastAPI app and
dumps a normalised JSON result, so two worktrees can be compared field by
field instead of relying on the v2 test suite alone.

Usage::

    # record a baseline from the current tree
    python scripts/v2_ab_compare.py --dump tests/output/ab/base.json

    # after a refactor, record again and compare
    python scripts/v2_ab_compare.py --dump tests/output/ab/after.json
    python scripts/v2_ab_compare.py --compare tests/output/ab/base.json \
        tests/output/ab/after.json

    # prove the harness itself is deterministic
    python scripts/v2_ab_compare.py --self-check
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from measure_api.v2.app import create_app
from measure_api.v2.models import ServiceSettings

#: Keys that legitimately change between runs and never participate in a diff.
VOLATILE_KEYS = frozenset({
    "trace_id", "operation_id", "duration_ms", "elapsed_ms", "created_at",
    "timestamp", "started_at", "finished_at", "run_id", "wall_ms", "timing",
    "storage", "at",
})


def _is_volatile(key: str) -> bool:
    """Return whether a payload key carries run-to-run noise."""
    return (key in VOLATILE_KEYS or key.startswith("worker_")
            or key.endswith(("_ms", "_at", "_monotonic", "_bytes")))

#: Numeric tolerance for comparison.
DEFAULT_TOLERANCE = 1e-9


def _fixture_images(root: Path) -> None:
    """Write the synthetic images the recorded cases refer to."""
    stripes = np.zeros((120, 240), np.uint8)
    stripes[:, 40:60] = 255
    stripes[:, 130:160] = 255
    cv2.imwrite(str(root / "stripes.png"), stripes)

    ramp = np.tile(np.arange(240, dtype=np.uint8), (120, 1))
    cv2.imwrite(str(root / "ramp.png"), ramp)

    block = np.zeros((160, 200), np.uint8)
    block[40:120, 50:150] = 200
    cv2.imwrite(str(root / "block.png"), block)

    line = np.zeros((300, 400), np.uint8)
    line[100:] = 255
    cv2.imwrite(str(root / "line.png"), line)

    circle = np.zeros((200, 200), np.uint8)
    cv2.circle(circle, (100, 100), 50, 255, -1)
    cv2.imwrite(str(root / "circle.png"), circle)

    rings = np.zeros((220, 220), np.uint8)
    for radius, level in [(80, 255), (60, 0), (40, 255)]:
        cv2.circle(rings, (110, 110), radius, level, -1)
    cv2.imwrite(str(root / "rings.png"), rings)


def _normalise(value: Any) -> Any:
    """Make a response payload comparable: drop volatile keys, sort, round."""
    if isinstance(value, dict):
        return {
            key: _normalise(item)
            for key, item in sorted(value.items())
            if not _is_volatile(key)
        }
    if isinstance(value, list):
        return [_normalise(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        return round(value, 12)
    return value


def _load_cases(directory: Path) -> List[Dict[str, Any]]:
    """Load every ``*.json`` case file, sorted by name."""
    cases = []
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.setdefault("name", path.stem)
        cases.append(payload)
    return cases


def run_cases(directory: Path) -> Dict[str, Any]:
    """Execute every recorded case and return its normalised outcome."""
    cases = _load_cases(directory)
    results: Dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _fixture_images(root)
        settings = ServiceSettings(shared_roots={"exchange": root})
        with TestClient(create_app(settings)) as client:
            for case in cases:
                path = "/api/v2" + case["path"]
                method = case.get("method", "POST").upper()
                if method == "GET":
                    response = client.get(path)
                else:
                    response = client.post(path, json=case.get("body"))
                body: Optional[Any]
                try:
                    body = _normalise(response.json())
                except ValueError:
                    body = response.text
                results[case["name"]] = {
                    "status": response.status_code,
                    "body": body,
                }
    return results


def _diff_node(left: Any, right: Any, path: str, tolerance: float,
               out: List[str]) -> None:
    """Recursively compare two normalised payloads."""
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            if key not in left:
                out.append(f"{path}.{key}: missing on the left")
            elif key not in right:
                out.append(f"{path}.{key}: missing on the right")
            else:
                _diff_node(left[key], right[key], f"{path}.{key}", tolerance, out)
        return
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            out.append(f"{path}: length {len(left)} != {len(right)}")
            return
        for index, (a, b) in enumerate(zip(left, right)):
            _diff_node(a, b, f"{path}[{index}]", tolerance, out)
        return
    if isinstance(left, bool) or isinstance(right, bool):
        if left != right:
            out.append(f"{path}: {left!r} != {right!r}")
        return
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if math.isclose(float(left), float(right), rel_tol=tolerance,
                        abs_tol=tolerance * max(1.0, abs(float(left)))):
            return
        out.append(f"{path}: {left} != {right}")
        return
    if left != right:
        out.append(f"{path}: {left!r} != {right!r}")


def compare(left: Dict[str, Any], right: Dict[str, Any],
            tolerance: float = DEFAULT_TOLERANCE) -> List[str]:
    """Return every difference between two recorded runs."""
    differences: List[str] = []
    for name in sorted(set(left) | set(right)):
        if name not in left:
            differences.append(f"{name}: missing in the baseline")
            continue
        if name not in right:
            differences.append(f"{name}: missing in the candidate")
            continue
        _diff_node(left[name], right[name], name, tolerance, differences)
    return differences


def main(argv: Optional[List[str]] = None) -> int:
    """Run the CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default="tests/v2/cases/base",
                        help="Directory of recorded request cases")
    parser.add_argument("--dump", help="Write the normalised run to this file")
    parser.add_argument("--compare", nargs=2, metavar=("BASE", "CANDIDATE"),
                        help="Compare two recorded runs")
    parser.add_argument("--self-check", action="store_true",
                        help="Run the cases twice and require identical output")
    parser.add_argument("--tol", type=float, default=DEFAULT_TOLERANCE,
                        help="Relative tolerance for numeric fields")
    args = parser.parse_args(argv)

    directory = Path(args.cases)
    if not directory.is_dir():
        parser.error(f"case directory not found: {directory}")

    if args.compare:
        left = json.loads(Path(args.compare[0]).read_text(encoding="utf-8"))
        right = json.loads(Path(args.compare[1]).read_text(encoding="utf-8"))
        differences = compare(left, right, args.tol)
        if differences:
            print(f"{len(differences)} difference(s):")
            for line in differences[:200]:
                print("  " + line)
            return 1
        print(f"0 difference(s) across {len(left)} case(s)")
        return 0

    first = run_cases(directory)
    if args.dump:
        Path(args.dump).parent.mkdir(parents=True, exist_ok=True)
        Path(args.dump).write_text(json.dumps(first, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
        print(f"wrote {args.dump} ({len(first)} case(s))")

    if args.self_check:
        second = run_cases(directory)
        differences = compare(first, second, args.tol)
        if differences:
            print(f"self-check FAILED with {len(differences)} difference(s):")
            for line in differences[:50]:
                print("  " + line)
            return 1
        print(f"self-check passed: {len(first)} case(s) deterministic")
    return 0


if __name__ == "__main__":
    sys.exit(main())