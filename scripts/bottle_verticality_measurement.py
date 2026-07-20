#!/usr/bin/env python3
"""
瓶身垂直偏差测量 — 多图批量测量与垂直度判定脚本

从侧面拍摄的瓶身图片中，自动发现多个 pair 边缘对，计算每对连线的中点，
取所有中点 X 坐标极差作为垂直度指标，判定每张图是否通过阈值。

用法:
    python scripts/bottle_verticality_measurement.py data/sample/bottlebody_1.jpg data/sample/bottlebody_2.jpg data/sample/bottlebody_3.jpg
    python scripts/bottle_verticality_measurement.py -c my_config.json data/sample/bottlebody_*.jpg
    python scripts/bottle_verticality_measurement.py -o result.json data/sample/bottlebody_*.jpg
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# 将项目根目录加入 sys.path，以便 import measure_inference
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.measure_inference import load_model, measure_image


# ============================================================================
# 默认配置 (配置文件不存在时使用)
# ============================================================================
DEFAULT_CONFIG: Dict[str, Any] = {
    "model_path": "models/瓶身垂直偏差/",
    "threshold_px": 5.0,
    "target_index": 0,
}

DEFAULT_CONFIG_PATH = Path(__file__).parent / "bottle_verticality_config.json"

# pair 分组正则: pair_1_A / pair_1_B → 组名 pair_1, 侧别 A 或 B
_PAIR_RE = re.compile(r"^(pair_\d+)_([AB])$")


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """加载配置文件，若不存在则使用默认值并打印警告。"""
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    path = Path(config_path)
    if path.is_file():
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return {**DEFAULT_CONFIG, **cfg}
    else:
        print(f"[WARN] 配置文件不存在: {path}，使用默认配置", file=sys.stderr)
        return dict(DEFAULT_CONFIG)


def discover_pairs(
    measurements: Dict[str, Any],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    从测量结果中自动发现 pair 分组。

    Args:
        measurements: 单目标的 measurements 字典，键如 pair_1_A, pair_1_B, pair_2_A...

    Returns:
        {pair_name: {"A": {"row":..., "col":...}, "B": {...}}, ...}
    """
    pairs: Dict[str, Dict[str, Dict[str, float]]] = {}
    for key, entry in measurements.items():
        m = _PAIR_RE.match(key)
        if not m:
            continue
        if entry.get("type") != "point" or not entry.get("valid"):
            continue
        pair_name, side = m.group(1), m.group(2)
        pairs.setdefault(pair_name, {})[side] = {
            "row": float(entry["row"]),
            "col": float(entry["col"]),
        }
    return pairs


def compute_verticality(pairs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    计算垂直度：所有 pair 中点 X 坐标的极差。

    Returns:
        None 如果有效 pair 少于 2 个；否则返回包含 centers 和 max_x_diff_px 的字典。
    """
    centers: List[Dict[str, Any]] = []
    for pname in sorted(pairs.keys()):
        pdict = pairs[pname]
        if "A" not in pdict or "B" not in pdict:
            continue
        cx = round((pdict["A"]["col"] + pdict["B"]["col"]) / 2, 2)
        centers.append({"pair": pname, "x": cx})

    if len(centers) < 2:
        return None

    xs = [c["x"] for c in centers]
    max_x_diff = round(max(xs) - min(xs), 2)

    return {
        "centers": centers,
        "max_x_diff_px": max_x_diff,
    }


def measure_one_image(
    model: Any,
    image_path: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """对单张图片执行垂直度测量并判定。"""
    threshold_px = config["threshold_px"]
    target_index = config.get("target_index", 0)

    result = measure_image(model, image_path)

    if not result.get("success"):
        return {
            "path": image_path,
            "success": False,
            "valid": False,
            "num_pairs": 0,
            "centers": [],
            "max_x_diff_px": None,
            "passed": False,
            "error": result.get("error", "unknown"),
        }

    targets = result.get("targets", [])
    if len(targets) <= target_index:
        return {
            "path": image_path,
            "success": True,
            "valid": False,
            "num_pairs": 0,
            "centers": [],
            "max_x_diff_px": None,
            "passed": False,
            "error": f"目标 index={target_index} 不存在 (共 {len(targets)} 个目标)",
        }

    target = targets[target_index]
    if not target.get("valid"):
        return {
            "path": image_path,
            "success": True,
            "valid": False,
            "num_pairs": 0,
            "centers": [],
            "max_x_diff_px": None,
            "passed": False,
            "error": f"目标 #{target.get('id')} 被标记为 invalid (score={target.get('score')})",
            "score": target.get("score"),
        }

    measurements = target.get("measurements", {})
    pairs = discover_pairs(measurements)
    v = compute_verticality(pairs)

    if v is None:
        return {
            "path": image_path,
            "success": True,
            "valid": True,
            "num_pairs": len(pairs),
            "centers": [],
            "max_x_diff_px": None,
            "passed": False,
            "error": f"有效 pair 不足 2 个 (发现 {len(pairs)} 个)",
        }

    passed = v["max_x_diff_px"] <= threshold_px
    detail = "OK" if passed else f"垂直偏差 {v['max_x_diff_px']} px > 阈值 {threshold_px} px"

    return {
        "path": image_path,
        "success": True,
        "valid": True,
        "num_pairs": len(pairs),
        "centers": v["centers"],
        "max_x_diff_px": v["max_x_diff_px"],
        "passed": passed,
        "detail": detail,
        "score": target.get("score"),
    }


def main():
    parser = argparse.ArgumentParser(
        description="瓶身垂直偏差测量 — 多图批量测量与垂直度判定",
    )
    parser.add_argument(
        "images", nargs="+",
        help="待测量图片路径 (支持 glob 展开，3 张及以上)",
    )
    parser.add_argument(
        "-c", "--config",
        default=None,
        help=f"配置文件路径 (默认: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="输出 JSON 文件路径 (默认输出到控制台)",
    )
    parser.add_argument(
        "--indent",
        type=int, default=2,
        help="JSON 缩进空格数 (默认 2)",
    )
    args = parser.parse_args()

    if len(args.images) < 3:
        print("[ERROR] 至少需要传入 3 张图片", file=sys.stderr)
        sys.exit(2)

    # 加载配置
    config = load_config(args.config)
    model_path = Path(config["model_path"])
    if not model_path.exists():
        print(f"[ERROR] 模型路径不存在: {model_path}", file=sys.stderr)
        sys.exit(2)

    # 加载模型 (只加载一次)
    print(f"[INFO] 加载模型: {model_path}", file=sys.stderr)
    model = load_model(str(model_path))

    # 逐张测量
    image_results: List[Dict[str, Any]] = []
    for i, img in enumerate(args.images):
        print(f"[INFO] 测量 ({i + 1}/{len(args.images)}): {img}", file=sys.stderr)
        r = measure_one_image(model, img, config)
        image_results.append(r)
        status = "OK" if r.get("passed") else "FAIL"
        xdiff = r.get("max_x_diff_px", "N/A")
        print(f"       -> {status}  x_diff={xdiff} px  (pairs={r.get('num_pairs', 0)})", file=sys.stderr)

    # 汇总
    all_passed = all(bool(r.get("passed", False)) for r in image_results)
    valid_images = [r for r in image_results if r.get("valid")]

    output = {
        "passed": bool(all_passed),
        "summary": {
            "num_images": len(image_results),
            "num_valid": len(valid_images),
            "threshold_px": float(config["threshold_px"]),
            "all_passed": all_passed,
        },
        "images": image_results,
    }

    json_str = json.dumps(output, ensure_ascii=False, indent=args.indent)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(json_str)
        print(f"[INFO] 结果已保存到: {out_path}", file=sys.stderr)
    else:
        print(json_str)

    sys.exit(0 if output["passed"] else 1)


if __name__ == "__main__":
    main()
