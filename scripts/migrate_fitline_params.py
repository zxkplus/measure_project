#!/usr/bin/env python3
"""
迁移脚本：交换 FitLineObject / LineMeasureObject 的 measure_length1 ↔ measure_length2。

用法:
    python scripts/migrate_fitline_params.py models/xxx/project.json [--dry-run]
"""

import json
import sys
import argparse


def migrate_params(data: dict, dry_run: bool = False) -> bool:
    """递归交换 project.json 中 fit line 对象的 measure_length1/measure_length2。"""
    changed = False

    if isinstance(data, dict):
        obj_type = data.get("object_type", "")
        if obj_type in ("FitLineObject", "LineMeasureObject"):
            if "measure_length1" in data and "measure_length2" in data:
                l1 = data["measure_length1"]
                l2 = data["measure_length2"]
                if not dry_run:
                    data["measure_length1"] = l2
                    data["measure_length2"] = l1
                print(f"  {obj_type}: swap {l1:.2f} ↔ {l2:.2f}")
                changed = True
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                changed |= migrate_params(value, dry_run)
    elif isinstance(data, list):
        for item in data:
            changed |= migrate_params(item, dry_run)

    return changed


def main():
    parser = argparse.ArgumentParser(
        description="Swap measure_length1/measure_length2 in FitLine/LineMeasure project JSONs"
    )
    parser.add_argument("path", nargs="+", help="Project JSON file(s)")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    args = parser.parse_args()

    for path in args.path:
        print(f"\n--- {path} ---")
        with open(path, "r") as f:
            data = json.load(f)
        changed = migrate_params(data, dry_run=args.dry_run)
        if changed and not args.dry_run:
            with open(path, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"  ✓ Written")
        elif changed:
            print(f"  (dry-run, not written)")
        else:
            print(f"  (no FitLine/LineMeasure objects found)")


if __name__ == "__main__":
    main()
