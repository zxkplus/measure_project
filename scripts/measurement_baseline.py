"""Capture and compare numeric measurement baselines for every taught model.

The shared-computation refactor deliberately changes some numbers (D01-D10).
Before touching those code paths, record what the current engine produces and
compare afterwards, so "which model moved and by how much" is a fact rather
than a guess.

Usage::

    python scripts/measurement_baseline.py --out tests/output/baseline/before.json
    python scripts/measurement_baseline.py --out tests/output/baseline/after.json
    python scripts/measurement_baseline.py --compare \
        tests/output/baseline/before.json tests/output/baseline/after.json

Only engine outputs are compared: target pose, validity and the per-label
measurement dictionaries.  No timing, path or trace data is written.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import cv2

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from measure.multi_target_workflow import MultiTargetWorkflow  # noqa: E402

MODELS_DIR = REPO / "models"
IMAGES = ("reference.png", "inspection.png")
TOLERANCE = 1e-9


def _jsonable(value: Any) -> Any:
    """Convert NumPy scalars/arrays and tuples into plain JSON values."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _jsonable(tolist())
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (ValueError, TypeError):
            pass
    return str(value)


def capture_model(model_dir: Path) -> Dict[str, Any]:
    """Measure every available image of one model and return its numbers.

    Args:
        model_dir: Directory holding a manifest model asset.

    Returns:
        Mapping of image name to the captured target list, or an ``error`` entry.
    """
    record: Dict[str, Any] = {"model": model_dir.name, "images": {}}
    try:
        workflow = MultiTargetWorkflow.load(str(model_dir))
    except Exception as error:  # noqa: BLE001 - recorded, not hidden
        record["load_error"] = f"{type(error).__name__}: {error}"
        return record
    for name in IMAGES:
        path = model_dir / name
        if not path.is_file():
            continue
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            record["images"][name] = {"error": "image unreadable"}
            continue
        try:
            targets = workflow.measure(image)
        except Exception as error:  # noqa: BLE001 - recorded, not hidden
            record["images"][name] = {"error": f"{type(error).__name__}: {error}"}
            continue
        record["images"][name] = _jsonable([
            {
                "id": target.id,
                "score": target.score,
                "rotation_deg": target.rotation_deg,
                "scale": target.scale,
                "center_row": target.center_row,
                "center_col": target.center_col,
                "valid": target.valid,
                "measurements": target.measurements,
            }
            for target in targets
        ])
    return record


def capture(root: Path) -> Dict[str, Any]:
    """Capture every model under ``root``.

    Args:
        root: Directory containing one sub-directory per model.

    Returns:
        Mapping of model name to its captured record.
    """
    records = {}
    for model_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if not (model_dir / "manifest.json").is_file():
            continue
        records[model_dir.name] = capture_model(model_dir)
        print(f"captured {model_dir.name}")
    return records


def _compare_values(path: str, before: Any, after: Any, out: List[Dict[str, Any]]) -> None:
    """Record every leaf difference, including numeric drift."""
    if isinstance(before, dict) and isinstance(after, dict):
        for key in sorted(set(before) | set(after)):
            if key not in before or key not in after:
                out.append({"path": f"{path}.{key}", "kind": "missing"})
                continue
            _compare_values(f"{path}.{key}", before[key], after[key], out)
        return
    if isinstance(before, list) and isinstance(after, list):
        if len(before) != len(after):
            out.append({"path": path, "kind": "length", "before": len(before),
                        "after": len(after)})
            return
        for index, (left, right) in enumerate(zip(before, after)):
            _compare_values(f"{path}[{index}]", left, right, out)
        return
    if isinstance(before, bool) or isinstance(after, bool):
        if before != after:
            out.append({"path": path, "kind": "bool", "before": before, "after": after})
        return
    if isinstance(before, (int, float)) and isinstance(after, (int, float)):
        if math.isclose(float(before), float(after), rel_tol=TOLERANCE, abs_tol=TOLERANCE):
            return
        out.append({"path": path, "kind": "number", "before": before, "after": after,
                    "delta": float(after) - float(before)})
        return
    if before != after:
        out.append({"path": path, "kind": "value", "before": before, "after": after})


def compare(before_path: Path, after_path: Path) -> int:
    """Print a per-model difference report and return a process exit code.

    Args:
        before_path: Baseline capture.
        after_path: Capture taken after the change.

    Returns:
        ``0`` when the two captures match within tolerance, ``1`` otherwise.
    """
    before = json.loads(before_path.read_text(encoding="utf-8"))
    after = json.loads(after_path.read_text(encoding="utf-8"))
    differences = []
    _compare_values("", before, after, differences)
    for item in differences:
        print(f"{item['path']}: {item['kind']} "
              f"{item.get('before')!r} -> {item.get('after')!r}")
    print(f"\n{len(differences)} difference(s) across {len(before)} model(s)")
    return 0 if not differences else 1


def main() -> int:
    """Run the capture or comparison command.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="Write a capture to this file")
    parser.add_argument("--root", type=Path, default=MODELS_DIR,
                        help="Directory containing the model assets")
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("BEFORE", "AFTER"),
                        help="Compare two captures and report differences")
    args = parser.parse_args()
    if args.compare:
        return compare(*args.compare)
    if not args.out:
        parser.error("one of --out or --compare is required")
    payload = capture(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())