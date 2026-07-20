#!/usr/bin/env python3
"""
瓶身直径测量 — 多图批量测量与圆度判定脚本

对多张不同旋转角度的瓶身侧面图片进行直径测量，根据配置的阈值判定：
  1. 单次测量直径是否在允许范围内
  2. 多次测量的直径极差是否在允许范围内

用法:
    python scripts/bottle_diameter_measurement.py data/sample/bottlebody_1.jpg data/sample/bottlebody_2.jpg data/sample/bottlebody_3.jpg
    python scripts/bottle_diameter_measurement.py -c my_config.json data/sample/bottlebody_*.jpg
    python scripts/bottle_diameter_measurement.py -o result.json data/sample/bottlebody_*.jpg
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 将项目根目录加入 sys.path，以便 import measure_inference
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.measure_inference import load_model, measure_image


# ============================================================================
# 默认配置 (配置文件不存在时使用)
# ============================================================================
DEFAULT_CONFIG: Dict[str, Any] = {
    "model_path": "models/瓶身直径测量/",
    "measurement_key": "TwoPointsDistance_2",
    "diameter_range": {"min_px": 1800.0, "max_px": 1840.0},
    "max_diameter_variation_px": 5.0,
    "target_index": 0,
}

# 配置文件默认路径 (脚本同目录)
DEFAULT_CONFIG_PATH = Path(__file__).parent / "bottle_diameter_config.json"


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """加载配置文件，若不存在则使用默认值并打印警告。"""
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    path = Path(config_path)
    if path.is_file():
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        # 用默认值补齐可能缺失的字段
        merged = {**DEFAULT_CONFIG, **cfg}
        return merged
    else:
        print(f"[WARN] 配置文件不存在: {path}，使用默认配置", file=sys.stderr)
        return DEFAULT_CONFIG


def extract_diameter(
    target: Dict[str, Any],
    measurement_key: str,
) -> Optional[float]:
    """从单个目标的测量结果中提取直径值 (px)。"""
    measurements = target.get("measurements", {})
    entry = measurements.get(measurement_key)
    if entry is None:
        return None
    if entry.get("type") != "distance":
        return None
    if not entry.get("valid", False):
        return None
    value = entry.get("value", None)
    if value is not None:
        value = float(value)
    return value


def measure_one_image(
    model: Any,
    image_path: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """对单张图片执行测量并判定。"""
    measurement_key = config["measurement_key"]
    target_index = config.get("target_index", 0)
    min_px = config["diameter_range"]["min_px"]
    max_px = config["diameter_range"]["max_px"]

    result = measure_image(model, image_path)

    if not result.get("success"):
        return {
            "path": image_path,
            "success": False,
            "valid": False,
            "diameter_px": None,
            "in_range": False,
            "error": result.get("error", "unknown"),
        }

    targets = result.get("targets", [])
    if len(targets) <= target_index:
        return {
            "path": image_path,
            "success": True,
            "valid": False,
            "diameter_px": None,
            "in_range": False,
            "error": f"目标 index={target_index} 不存在 (共 {len(targets)} 个目标)",
        }

    target = targets[target_index]
    if not target.get("valid"):
        return {
            "path": image_path,
            "success": True,
            "valid": False,
            "diameter_px": None,
            "in_range": False,
            "error": f"目标 #{target.get('id')} 被标记为 invalid (score={target.get('score')})",
            "score": target.get("score"),
        }

    diameter = extract_diameter(target, measurement_key)
    if diameter is None:
        # 尝试列出可用的 measurement keys 帮助排查
        available = list(target.get("measurements", {}).keys())
        return {
            "path": image_path,
            "success": True,
            "valid": True,
            "diameter_px": None,
            "in_range": False,
            "error": f"未找到测量字段 '{measurement_key}'，可用字段: {available}",
        }

    in_range = bool(min_px <= diameter <= max_px)
    detail = "OK" if in_range else f"超出范围 [{min_px}, {max_px}]"

    return {
        "path": image_path,
        "success": True,
        "valid": True,
        "diameter_px": round(diameter, 2),
        "in_range": in_range,
        "detail": detail,
        "score": target.get("score"),
    }


def build_summary(
    image_results: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """汇总多图结果并做整体圆度判定。"""
    valid_diameters = [
        r["diameter_px"] for r in image_results
        if r["valid"] and r["diameter_px"] is not None
    ]
    all_in_range = all(r.get("in_range", False) for r in image_results)

    variation_ok = True
    diameter_range_px = None
    if len(valid_diameters) >= 2:
        diameter_range_px = round(max(valid_diameters) - min(valid_diameters), 2)
        max_var = config["max_diameter_variation_px"]
        variation_ok = diameter_range_px <= max_var

    all_in_range = bool(all_in_range)
    variation_ok = bool(variation_ok)
    passed = all_in_range and variation_ok

    return {
        "passed": passed,
        "diameter_values": valid_diameters,
        "diameter_range_px": diameter_range_px,
        "max_variation_px": config["max_diameter_variation_px"],
        "range_ok": all_in_range,
        "variation_ok": variation_ok,
    }


def main():
    parser = argparse.ArgumentParser(
        description="瓶身直径测量 — 多图批量测量与圆度判定",
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
        status = "OK" if r.get("in_range") else "FAIL"
        diam = r.get("diameter_px", "N/A")
        print(f"       -> {status}  diameter={diam} px", file=sys.stderr)

    # 汇总
    summary = build_summary(image_results, config)

    output = {
        "passed": summary["passed"],
        "summary": {
            "num_images": len(image_results),
            "num_valid": len(summary["diameter_values"]),
            "diameter_values": summary["diameter_values"],
            "diameter_range_px": summary["diameter_range_px"],
            "max_variation_px": summary["max_variation_px"],
            "range_ok": summary["range_ok"],
            "variation_ok": summary["variation_ok"],
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
