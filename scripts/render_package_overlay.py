#!/usr/bin/env python3
"""Draw the recorded localisation box and fitted circle back onto the test image.

Reads a measure-v2 ``measure.pick_run`` debug package and renders only two
recorded observations on top of the detection image:

* the matched template support rectangle (localisation result),
* the fitted circle of the first ``run`` node and its sample points,
* the radial search band (inner/outer radius), its nominal circle and polarity.

Usage:
    python scripts/render_package_overlay.py <package_dir> [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
)

COLOUR_BOX = (255, 208, 0)
COLOUR_CIRCLE = (0, 220, 90)
COLOUR_CENTRE = (255, 40, 40)
COLOUR_SAMPLE = (0, 210, 255)
COLOUR_SEARCH = (150, 205, 255)
COLOUR_BAND = (205, 230, 255)
COLOUR_NOMINAL = (255, 255, 255)
COLOUR_TEXT = (255, 255, 255)


def load_package(root: Path) -> Dict[str, Any]:
    """Read the manifest, request, result and asset index of a debug package.

    Args:
        root: Extracted debug package directory.

    Returns:
        Documents and an ``asset id -> file`` map for the packaged assets.
    """
    def read(name: str) -> Dict[str, Any]:
        path = root / name
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    manifest, inputs, result = read("manifest.json"), read("inputs.json"), read("result.json")
    assets: Dict[str, Path] = {}
    index = root / (manifest.get("assets") or "assets.json")
    if index.exists():
        for entry in json.loads(index.read_text(encoding="utf-8")):
            asset = entry.get("asset") or {}
            if asset.get("id"):
                assets[str(asset["id"])] = root / entry["file"]
    return {"root": root, "manifest": manifest, "inputs": inputs, "result": result,
            "assets": assets}


def package_file(package: Dict[str, Any], reference: Dict[str, Any]) -> Path:
    """Resolve a recipe file reference to its packaged asset file.

    Args:
        package: Parsed debug package.
        reference: Shared-file reference recorded in the request.

    Returns:
        Package file holding the referenced bytes.
    """
    name = Path(str(reference["path"])).name
    matches = [(identifier, path) for identifier, path in package["assets"].items()
               if name == identifier or name.startswith(identifier)]
    if not matches:
        raise RuntimeError(f"No packaged asset matches {reference['path']!r}")
    return max(matches, key=lambda item: len(item[0]))[1]


def support_rectangle(package: Dict[str, Any], result: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """Return the matched template rectangle in detection-image pixels.

    Args:
        package: Parsed debug package.
        result: Recorded run document.

    Returns:
        ``(x0, y0, width, height)`` of the matched template support region.
    """
    template = np.load(package_file(package, package["inputs"]["recipe"]["localization"]["template"]))
    metadata = json.loads(template["metadata"].item())
    width = float(metadata["support"]["width"])
    height = float(metadata["support"]["height"])
    mapping = np.array(result["targets"][0]["localization"]["match"]["support_mapping"]["matrix"])
    corners = np.array([[0, 0, 1], [width, 0, 1], [width, height, 1], [0, height, 1]])
    points = (corners @ mapping.T)[:, :2]
    return float(points[:, 0].min()), float(points[:, 1].min()), width, height


def font(size: int) -> ImageFont.FreeTypeFont:
    """Load a CJK-capable TrueType font at the requested size."""
    for candidate in FONT_CANDIDATES:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def draw_dashed_circle(draw: ImageDraw.ImageDraw, centre: Tuple[float, float], radius: float,
                       colour: Tuple[int, int, int], width: int,
                       dash_degrees: float = 4.0, gap_degrees: float = 4.0) -> None:
    """Draw a dashed circle using short arc segments.

    Args:
        draw: PIL drawing context.
        centre: Circle centre in pixels.
        radius: Circle radius in pixels.
        colour: Stroke colour.
        width: Stroke width in pixels.
        dash_degrees: Angular length of each dash.
        gap_degrees: Angular gap between dashes.
    """
    angle = 0.0
    while angle < 360.0:
        box = [centre[0] - radius, centre[1] - radius,
               centre[0] + radius, centre[1] + radius]
        draw.arc(box, start=angle, end=angle + dash_degrees, fill=colour, width=width)
        angle += dash_degrees + gap_degrees


def draw_arrow(draw: ImageDraw.ImageDraw, start: Tuple[float, float], end: Tuple[float, float],
               colour: Tuple[int, int, int], width: int, head_length: float = 22.0) -> None:
    """Draw a straight arrow from ``start`` to ``end``.

    Args:
        draw: PIL drawing context.
        start: Tail position in pixels.
        end: Head position in pixels.
        colour: Stroke colour.
        width: Stroke width in pixels.
        head_length: Arrow head length in pixels.
    """
    draw.line([start, end], fill=colour, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    for side in (-0.5, 0.5):
        draw.line([end, (end[0] - head_length * math.cos(angle + side),
                         end[1] - head_length * math.sin(angle + side))], fill=colour, width=width)


def search_geometry(package: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    """Return the recorded radial search band in detection-image pixels.

    Args:
        package: Parsed debug package.
        result: Recorded run document.

    Returns:
        Taught/nominal radius, inner and outer search radius, polarity and threshold.
    """
    parameters = package["inputs"]["recipe"]["nodes"][0]["parameters"]
    mapping = np.array(result["targets"][0]["mapping"]["matrix"], dtype=float)
    taught = parameters["center"]
    nominal = float(parameters["radius"])
    half = float(parameters["search_half_length"])
    centre = (mapping[0, 0] * taught["x"] + mapping[0, 1] * taught["y"] + mapping[0, 2],
              mapping[1, 0] * taught["x"] + mapping[1, 1] * taught["y"] + mapping[1, 2])
    return {
        "centre": centre,
        "nominal": nominal,
        "half": half,
        "inner": max(nominal - half, 0.0),
        "outer": nominal + half,
        "polarity": parameters.get("polarity"),
        "selection": parameters.get("selection"),
        "threshold": parameters.get("threshold"),
        "sigma": parameters.get("sigma"),
    }


def draw_label(draw: ImageDraw.ImageDraw, xy: Tuple[float, float], text: str,
               size: int, colour: Tuple[int, int, int]) -> None:
    """Draw white text with a dark halo so it stays readable on any background."""
    face = font(size)
    x, y = xy
    for offset in ((-2, 0), (2, 0), (0, -2), (0, 2), (2, 2), (-2, -2)):
        draw.text((x + offset[0], y + offset[1]), text, font=face, fill=(0, 0, 0))
    draw.text((x, y), text, font=face, fill=colour)


def render(package: Dict[str, Any], out_dir: Path) -> List[Path]:
    """Render the template rectangle and fitted circle onto the detection image.

    Args:
        package: Parsed debug package.
        out_dir: Destination directory for the rendered PNG files.

    Returns:
        Written file paths (full image first, then the 1:1 detail crop).
    """
    result = package["result"]
    target = result["targets"][0]
    node = target["nodes"][0]["result"]
    image_path = package_file(package, package["inputs"]["image"])
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    x0, y0, box_w, box_h = support_rectangle(package, result)
    centre = node["circle"]["center"]
    radius = float(node["circle"]["radius"])
    score = target["localization"]["match"]["score"]
    samples = node.get("samples") or [edge["point"] for edge in node.get("edges", [])]
    search = search_geometry(package, result)
    draw = ImageDraw.Draw(image)

    draw.rectangle([x0, y0, x0 + box_w, y0 + box_h], outline=COLOUR_BOX, width=6)
    draw.ellipse([centre["x"] - radius, centre["y"] - radius,
                  centre["x"] + radius, centre["y"] + radius],
                 outline=COLOUR_CIRCLE, width=6)
    mask = Image.new("L", image.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    def circle_box(cx: float, cy: float, r: float) -> List[float]:
        return [cx - r, cy - r, cx + r, cy + r]

    band_centre = search["centre"]
    mask_draw.ellipse(circle_box(*band_centre, search["outer"]), fill=42)
    mask_draw.ellipse(circle_box(*band_centre, search["inner"]), fill=0)
    image = Image.composite(Image.new("RGB", image.size, COLOUR_BAND), image, mask)
    draw = ImageDraw.Draw(image)
    draw.ellipse(circle_box(*band_centre, search["outer"]), outline=COLOUR_SEARCH, width=5)
    draw.ellipse(circle_box(*band_centre, search["inner"]), outline=COLOUR_SEARCH, width=5)
    draw_dashed_circle(draw, band_centre, search["nominal"], COLOUR_NOMINAL, 4)

    marker = 9
    for sample in samples:
        draw.ellipse([sample["x"] - marker, sample["y"] - marker,
                      sample["x"] + marker, sample["y"] + marker],
                     fill=COLOUR_SAMPLE, outline=(0, 0, 0), width=2)
    arm = 48
    draw.line([centre["x"] - arm, centre["y"], centre["x"] + arm, centre["y"]],
              fill=COLOUR_CENTRE, width=4)
    draw.line([centre["x"], centre["y"] - arm, centre["x"], centre["y"] + arm],
              fill=COLOUR_CENTRE, width=4)

    draw_label(draw, (x0 + 12, y0 - 124),
               f"定位模板框 {int(box_w)}×{int(box_h)}  得分 {score:.6f}", 44, COLOUR_BOX)
    draw_label(draw, (centre["x"] + radius * 0.62, centre["y"] - radius * 0.62),
               f"拟合圆 R={radius:.2f} px", 44, COLOUR_CIRCLE)
    draw_label(draw, (centre["x"] + arm + 16, centre["y"] + arm + 4),
               f"圆心 ({centre['x']:.2f}, {centre['y']:.2f})", 40, COLOUR_CENTRE)
    draw_label(draw, (centre["x"] + radius * 0.62, centre["y"] - radius * 0.62 + 58),
               f"采样点 {len(samples)} 个", 44, COLOUR_SAMPLE)
    draw_label(draw, (x0 + 12, y0 - 66),
               f"搜索环 R {search['inner']:.2f} ~ {search['outer']:.2f} px"
               f"（±{search['half']:.0f}）  极性 {search['polarity']}"
               f"（亮→暗）  阈值 {search['threshold']:g} 灰度", 40, COLOUR_SEARCH)
    draw_label(draw, (band_centre[0] - search["outer"] * 0.72,
                      band_centre[1] + search["outer"] * 0.72 - 46),
               f"内圆 R={search['inner']:.2f}", 38, COLOUR_SEARCH)
    draw_label(draw, (band_centre[0] + search["outer"] * 0.30,
                      band_centre[1] - search["outer"] * 0.95),
               f"外圆 R={search['outer']:.2f}", 38, COLOUR_SEARCH)
    draw_label(draw, (band_centre[0] - search["nominal"] * 0.66,
                      band_centre[1] - search["nominal"] * 0.66),
               f"示教圆 R={search['nominal']:.2f}（搜索中心）", 38, (255, 255, 255))
    for degrees in (45, 135, 225, 315):
        phi = math.radians(degrees)
        unit = (math.cos(phi), math.sin(phi))
        tail = (band_centre[0] + (search["outer"] - 26) * unit[0],
                band_centre[1] + (search["outer"] - 26) * unit[1])
        head = (band_centre[0] + (search["outer"] - 156) * unit[0],
                band_centre[1] + (search["outer"] - 156) * unit[1])
        draw_arrow(draw, tail, head, (255, 255, 255), 8, head_length=38)

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    full_path = out_dir / "overlay_full.png"
    image.save(full_path)
    written.append(full_path)
    print(f"sample points drawn: {len(samples)}")

    margin = 150
    crop_box = (
        int(max(0, min(centre["x"] - radius, x0) - margin)),
        int(max(0, min(centre["y"] - radius, y0) - margin)),
        int(min(width, max(centre["x"] + radius, x0 + box_w) + margin)),
        int(min(height, max(centre["y"] + radius, y0 + box_h) + margin)),
    )
    crop_path = out_dir / "overlay_crop.png"
    image.crop(crop_box).save(crop_path)
    written.append(crop_path)
    return written


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Render one package overlay.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Draw the localisation box and circle")
    parser.add_argument("package", type=Path, help="Extracted 算法调试包 directory")
    parser.add_argument("--out", type=Path, default=None, help="Output directory")
    parser.add_argument("--result", type=Path, default=None,
                        help="Alternative result JSON (default: the recorded result.json)")
    args = parser.parse_args(argv)

    root = args.package.expanduser().resolve()
    package = load_package(root)
    if args.result is not None:
        package["result"] = json.loads(args.result.read_text(encoding="utf-8"))
    if not package["result"].get("targets"):
        parser.error("Package has no targets to render")
    out_dir = (args.out or (Path.cwd() / "output" / f"render-{package['manifest']['run_id'][:8]}"))
    for path in render(package, out_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())