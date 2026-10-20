"""Migrate legacy single-file ``workflow.npz`` models to manifest v3.0 assets.

The legacy format packed everything (reference image, template image, edge
template, measurement definitions) into one ``.npz`` with a ``workflow_meta``
JSON blob.  Model assets are now a manifest directory:

    <model_dir>/
        manifest.json    authoritative definition
        reference.npy    raw reference image
        workflow.npz     derived cache only (safe to delete)

Usage:
    python scripts/migrate_models_to_manifest.py --dry-run models/
    python scripts/migrate_models_to_manifest.py models/
    python scripts/migrate_models_to_manifest.py models/瓶全高 --out /tmp/migrated

Each model is verified before its files are replaced: the migrated asset is
reloaded and measured on the reference image, and every measurement field is
compared against the legacy workflow's own result.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from measure.multi_target_workflow import (  # noqa: E402
    MANIFEST_FILENAME,
    REFERENCE_FILENAME,
    MultiTargetWorkflow,
)

LEGACY_BACKUP_SUFFIX = ".legacy.npz"


def find_legacy_models(root: Path) -> list[Path]:
    """Return model directories under *root* that hold a legacy workflow.npz."""
    candidates: list[Path] = []
    if (root / "workflow.npz").is_file() and not (root / MANIFEST_FILENAME).is_file():
        candidates.append(root)
    for child in sorted(root.iterdir()) if root.is_dir() else []:
        if not child.is_dir():
            continue
        if (child / "workflow.npz").is_file() and not (child / MANIFEST_FILENAME).is_file():
            candidates.append(child)
    return candidates


def _normalise(results) -> list:
    """JSON-normalise measurement results for exact comparison."""
    out = []
    for tr in results:
        out.append({
            "id": tr.id,
            "score": float(tr.score),
            "rotation_deg": float(tr.rotation_deg),
            "scale": float(tr.scale),
            "center_row": float(tr.center_row),
            "center_col": float(tr.center_col),
            "valid": bool(tr.valid),
            "measurements": json.loads(
                json.dumps(tr.measurements, default=str, sort_keys=True)
            ),
        })
    return out


def migrate_model(model_dir: Path, out_dir: Path | None, *, keep_backup: bool) -> dict:
    """Migrate one model directory.  Returns a small report dict."""
    legacy_path = model_dir / "workflow.npz"
    target_dir = Path(out_dir) if out_dir else model_dir

    legacy = MultiTargetWorkflow.load_legacy(str(legacy_path))
    reference = legacy._alignment._reference_image
    if reference is None:
        raise RuntimeError("legacy model has no reference image")

    before = _normalise(legacy.measure(reference))

    target_dir.mkdir(parents=True, exist_ok=True)
    legacy.save(str(target_dir))

    # verify: reload from disk and re-measure
    reloaded = MultiTargetWorkflow.load(str(target_dir))
    after = _normalise(reloaded.measure(reference))
    if before != after:
        raise AssertionError(
            f"migration changed measurement results for {model_dir.name}"
        )

    backup = None
    if target_dir == model_dir and keep_backup:
        backup = legacy_path.with_name(legacy_path.name + LEGACY_BACKUP_SUFFIX)
        if not backup.exists():
            shutil.copy2(legacy_path, backup)

    return {
        "model": model_dir.name,
        "target_dir": str(target_dir),
        "instances": len(before),
        "backup": str(backup) if backup else None,
        "objects": len(legacy.measurement_defs),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="model directory, or a directory of models")
    parser.add_argument("--out", default=None,
                        help="write migrated assets here instead of in place")
    parser.add_argument("--dry-run", action="store_true",
                        help="only list the models that would be migrated")
    parser.add_argument("--force", action="store_true",
                        help="re-migrate models that already have a manifest")
    parser.add_argument("--no-backup", action="store_true",
                        help="do not keep <model>/workflow.legacy.npz")
    args = parser.parse_args(argv)

    root = Path(args.path)
    if not root.exists():
        print(f"error: {root} does not exist", file=sys.stderr)
        return 2

    models = find_legacy_models(root)
    if args.force and root.is_dir() and (root / MANIFEST_FILENAME).is_file():
        models = sorted({*models, root})
    if root.is_dir() and (root / "workflow.npz").is_file():
        models = sorted({root, *models})

    if not models:
        print("no legacy models found (nothing to migrate)")
        return 0

    print(f"found {len(models)} legacy model(s):")
    for model in models:
        print(f"  {model}")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    failures = 0
    for model in models:
        try:
            report = migrate_model(
                model, Path(args.out) / model.name if args.out else None,
                keep_backup=not args.no_backup,
            )
            print(f"  migrated {report['model']}: {report['objects']} object(s), "
                  f"{report['instances']} instance(s) verified"
                  + (f", backup {Path(report['backup']).name}" if report["backup"] else ""))
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures += 1
            print(f"  FAILED {model.name}: {type(exc).__name__}: {exc}", file=sys.stderr)

    print(f"\ndone: {len(models) - failures} migrated, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())