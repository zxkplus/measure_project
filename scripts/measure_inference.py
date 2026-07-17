#!/usr/bin/env python3
"""
测量推理脚本

使用训练好的测量模型对图片进行检测，返回JSON格式的测量结果。

使用示例:
    python scripts/measure_inference.py --image path/to/image.png --model path/to/model_dir
    python scripts/measure_inference.py --image path/to/image.png --model path/to/workflow.npz
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from measure_gui.multi_target import MultiTargetWorkflow


def load_model(model_path: str) -> MultiTargetWorkflow:
    """
    加载测量模型
    
    Args:
        model_path: 模型路径，可以是：
                   - workflow.npz 文件路径
                   - 包含 workflow.npz 的目录路径
    
    Returns:
        MultiTargetWorkflow 实例
    """
    model_path = Path(model_path)
    
    # 如果是目录，查找 workflow.npz
    if model_path.is_dir():
        workflow_path = model_path / "workflow.npz"
        if not workflow_path.exists():
            raise FileNotFoundError(f"在目录 {model_path} 中未找到 workflow.npz")
        model_path = workflow_path
    
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    return MultiTargetWorkflow.load(str(model_path))


def measure_image(
    model: MultiTargetWorkflow,
    image_path: str,
    save_debug: bool = False,
    debug_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    对图片执行测量
    
    Args:
        model: 已加载的 MultiTargetWorkflow 模型
        image_path: 待检测图片路径
        save_debug: 是否保存调试图片
        debug_dir: 调试图片保存目录
    
    Returns:
        测量结果字典，包含:
        - success: 是否成功
        - num_targets: 检测到的目标数量
        - num_valid: 有效目标数量
        - targets: 目标列表，每个目标包含:
          - id: 目标ID
          - valid: 是否有效
          - score: 匹配分数
          - rotation_deg: 旋转角度
          - center_row: 中心行坐标
          - center_col: 中心列坐标
          - measurements: 测量结果字典
    """
    # 加载图片
    image_path = Path(image_path)
    if not image_path.exists():
        return {
            "success": False,
            "error": f"图片文件不存在: {image_path}",
            "num_targets": 0,
            "num_valid": 0,
            "targets": [],
        }
    
    # 读取图片（支持灰度和彩色）
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {
            "success": False,
            "error": f"无法读取图片: {image_path}",
            "num_targets": 0,
            "num_valid": 0,
            "targets": [],
        }
    
    # 设置调试目录
    if save_debug:
        if debug_dir is None:
            debug_dir = str(image_path.parent / "debug_output")
        model._debug_dir = debug_dir
        os.makedirs(debug_dir, exist_ok=True)
    else:
        model._debug_dir = None
    
    # 执行测量
    try:
        results = model.measure(img)
    except Exception as e:
        return {
            "success": False,
            "error": f"测量执行失败: {str(e)}",
            "num_targets": 0,
            "num_valid": 0,
            "targets": [],
        }
    
    # 构建返回结果
    targets = []
    for target in results:
        target_dict = {
            "id": target.id,
            "valid": target.valid,
            "score": round(target.score, 4),
            "rotation_deg": round(target.rotation_deg, 2),
            "center_row": round(target.center_row, 2),
            "center_col": round(target.center_col, 2),
            "measurements": {},
        }
        
        # 处理测量结果
        for label, measurement in target.measurements.items():
            if isinstance(measurement, dict):
                # 测量结果是字典格式
                meas_dict = {
                    "type": measurement.get("type", "unknown"),
                    "valid": measurement.get("valid", False),
                }
                
                # 根据类型添加特定字段
                meas_type = measurement.get("type", "")
                if meas_type == "circle":
                    meas_dict.update({
                        "center_row": round(measurement.get("center_row", 0), 2),
                        "center_col": round(measurement.get("center_col", 0), 2),
                        "radius": round(measurement.get("radius", 0), 2),
                    })
                    # 添加 meta 信息
                    meta = measurement.get("meta", {})
                    if meta:
                        meas_dict["ellipticity"] = round(meta.get("ellipticity", 0), 2)
                        meas_dict["max_radius"] = round(meta.get("max_radius", 0), 2)
                        meas_dict["min_radius"] = round(meta.get("min_radius", 0), 2)
                        meas_dict["mean_error"] = round(meta.get("mean_error", 0), 4)
                        meas_dict["num_points"] = meta.get("num_points", 0)
                elif meas_type == "point":
                    meas_dict.update({
                        "row": round(measurement.get("row", 0), 2),
                        "col": round(measurement.get("col", 0), 2),
                    })
                elif meas_type == "line":
                    meas_dict.update({
                        "start_row": round(measurement.get("start_row", 0), 2),
                        "start_col": round(measurement.get("start_col", 0), 2),
                        "end_row": round(measurement.get("end_row", 0), 2),
                        "end_col": round(measurement.get("end_col", 0), 2),
                        "length": round(measurement.get("length", 0), 2),
                    })
                elif meas_type == "distance":
                    meas_dict["value"] = round(measurement.get("value", 0), 2)
                elif meas_type == "angle":
                    meas_dict["value_deg"] = round(measurement.get("value_deg", 0), 2)
                
                target_dict["measurements"][label] = meas_dict
            else:
                # 测量结果是对象格式
                meas_dict = {
                    "type": getattr(measurement, "type", "unknown"),
                    "valid": getattr(measurement, "valid", False),
                }
                
                meas_type = getattr(measurement, "type", "")
                if meas_type == "circle":
                    meas_dict.update({
                        "center_row": round(getattr(measurement, "center_row", 0), 2),
                        "center_col": round(getattr(measurement, "center_col", 0), 2),
                        "radius": round(getattr(measurement, "radius", 0), 2),
                    })
                    meta = getattr(measurement, "meta", {})
                    if meta:
                        meas_dict["ellipticity"] = round(meta.get("ellipticity", 0), 2)
                        meas_dict["max_radius"] = round(meta.get("max_radius", 0), 2)
                        meas_dict["min_radius"] = round(meta.get("min_radius", 0), 2)
                elif meas_type == "point":
                    meas_dict.update({
                        "row": round(getattr(measurement, "row", 0), 2),
                        "col": round(getattr(measurement, "col", 0), 2),
                    })
                
                target_dict["measurements"][label] = meas_dict
        
        targets.append(target_dict)
    
    return {
        "success": True,
        "image_path": str(image_path),
        "image_size": {"height": img.shape[0], "width": img.shape[1]},
        "num_targets": len(targets),
        "num_valid": sum(1 for t in targets if t["valid"]),
        "targets": targets,
    }


def measure(
    image_path: str,
    model_path: str,
    save_debug: bool = False,
    debug_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    测量接口函数
    
    Args:
        image_path: 待检测图片路径
        model_path: 模型路径（目录或 workflow.npz 文件）
        save_debug: 是否保存调试图片
        debug_dir: 调试图片保存目录
    
    Returns:
        测量结果字典（JSON 格式）
    """
    try:
        # 加载模型
        model = load_model(model_path)
        
        # 执行测量
        result = measure_image(model, image_path, save_debug, debug_dir)
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "num_targets": 0,
            "num_valid": 0,
            "targets": [],
        }


def main():
    parser = argparse.ArgumentParser(description="测量推理脚本")
    parser.add_argument(
        "--image", "-i",
        required=True,
        help="待检测图片路径",
    )
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="模型路径（目录或 workflow.npz 文件）",
    )
    parser.add_argument(
        "--save-debug",
        action="store_true",
        help="保存调试图片",
    )
    parser.add_argument(
        "--debug-dir",
        default=None,
        help="调试图片保存目录",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="输出 JSON 文件路径（默认输出到控制台）",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON 缩进空格数",
    )
    
    args = parser.parse_args()
    
    # 执行测量
    result = measure(
        image_path=args.image,
        model_path=args.model,
        save_debug=args.save_debug,
        debug_dir=args.debug_dir,
    )
    
    # 输出结果
    json_str = json.dumps(result, ensure_ascii=False, indent=args.indent)
    
    if args.output:
        # 保存到文件
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(json_str)
        print(f"结果已保存到: {output_path}")
    else:
        # 输出到控制台
        print(json_str)
    
    # 返回退出码
    sys.exit(0 if result.get("success", False) else 1)


if __name__ == "__main__":
    main()
