#!/usr/bin/env python3
"""
Reproduce an online "algorithm measurement" call locally from a debug dump.

The production service (not available in this repo) answered JSON protocol
requests of the form saved under ``measurement-debug-*/cameras/camera_*/``:
  algorithm_request.json   (request: objects -> elements with roi/line/thr)
  algorithm_response.json  (ground truth: per-object res + element coords)
  current_image.jpg        (8-bit grayscale image)

Detected semantics (verified numerically on camera_3 / camera_5):
  * coordinate system is x=col, y=row; images are grayscale uint8
  * an "edge" is the sub-pixel position where the grey profile along a scan
    line crosses ``threshold`` (a grey level, e.g. 185), interpolated
  * element types:
      point   -> one edge on the scan line (direction filtered by isInv)
      segment -> the two outer edges on the scan line
      circle  -> radial edges around an expected centre, then circle fit
  * object result (pixels):
      height(point)        = image_height - y
      distance(segment)    = length of segment
      distance(2 points)   = |dx| between them (axis aligned data)
      distance(pt+segment) = vertical gap |y_seg - y_pt|
      diameter(circle)     = fitted radius

Tunable assumptions live in CONFIG so the reproducer can be calibrated
against the saved responses (the online sub-pixel/rounding details are not
documented and had to be reverse engineered).

Usage:
    python scripts/repro_protocol_measurement.py <debug_dir> \
        [--out report.json] [--params tune.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Calibratable assumptions (baseline, tuned while diffing against responses)
# ---------------------------------------------------------------------------

CONFIG: Dict[str, Any] = {
    # Grey-level pre-smoothing of the 1-D profile before crossing detection.
    # Calibrated to 0.5 against camera_3: it flips the half-pixel rounding of
    # point edges (327.5 -> 327, 328.5 -> 329) without moving other edges.
    "sigma": 0.5,
    # Point selection: 'nearest' to the expected position | 'first' along line.
    "point_select": "nearest",
    # isInv -> crossing direction to keep ('up' = dark->bright, 'down' = bright->dark)
    "is_inv_map": {True: "down", False: "up"},
    # Rounding of output coordinates / res (banker's rounding vs half-away).
    "round_mode": "half_away",
    # --- circle element ---
    "circle_angle_step_deg": 1.0,
    "circle_band": 150.0,
    # radius band (px) around the expected radius searched per direction
}

# Error codes in responses that mean success.


def _round_half_away(v: float) -> int:
    return int(np.floor(v + 0.5)) if v >= 0 else int(np.ceil(v - 0.5))


def _round_val(v: float, mode: str) -> int:
    if mode == "half_away":
        return _round_half_away(v)
    return int(round(v))  # banker's


# ---------------------------------------------------------------------------
# 1-D profile + grey crossing detection
# ---------------------------------------------------------------------------


def _clip_line_to_rect(
    line: Tuple[float, float, float, float],
    rect: Tuple[float, float, float, float],
) -> Optional[Tuple[float, float]]:
    """Clip segment line=(x0,y0,x1,y1) to axis-aligned rect=(x0r,y0r,x1r,y1r)
    (inclusive).  Returns the parameter interval (t_lo, t_hi) in [0,1] kept
    inside the rect, or None when the segment misses the rect (Liang-Barsky).
    """
    x0, y0, x1, y1 = line
    rx0, ry0, rx1, ry1 = rect
    xmin, xmax = min(rx0, rx1), max(rx0, rx1)
    ymin, ymax = min(ry0, ry1), max(ry0, ry1)

    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0
    # (p, q) pairs: left/right/bottom/top
    for p, q in ((-dx, x0 - xmin), (dx, xmax - x0),
                 (-dy, y0 - ymin), (dy, ymax - y0)):
        if p == 0:
            if q < 0:
                return None
            continue
        r = q / p
        if p < 0:          # entering the slab
            if r > t1:
                return None
            t0 = max(t0, r)
        else:              # leaving the slab
            if r < t0:
                return None
            t1 = min(t1, r)
    return t0, t1


def _profile_along(line, img: np.ndarray):
    """Sample the grey profile along an axis-aligned scan line.

    Walks one pixel at a time along the dominant axis (the production data is
    axis aligned).  Returns arrays (xs, ys, vals) of integer pixel samples.
    """
    x0, y0, x1, y1 = (float(v) for v in line)
    h, w = img.shape[:2]
    if abs(x1 - x0) >= abs(y1 - y0):
        n = int(abs(x1 - x0)) + 1
        xs = np.linspace(x0, x1, n)
        ys = np.full(n, y0)
    else:
        n = int(abs(y1 - y0)) + 1
        ys = np.linspace(y0, y1, n)
        xs = np.full(n, x0)
    xi = np.clip(np.round(xs).astype(int), 0, w - 1)
    yi = np.clip(np.round(ys).astype(int), 0, h - 1)
    vals = img[yi, xi].astype(np.float64)
    return xs, ys, vals


def _smooth1d(vals: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return vals
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(vals, sigma)


def edge_crossings(
    line: Tuple[float, float, float, float],
    roi: Tuple[float, float, float, float],
    threshold: float,
    img: np.ndarray,
    sigma: float = 0.0,
) -> List[Dict[str, Any]]:
    """All grey-level crossings of ``threshold`` along ``line`` ∩ ``roi``.

    Each crossing: {x, y (image coords, sub-pixel), t (position along the
    integer pixel walk), dir ('up' | 'down')}.
    """
    t_range = _clip_line_to_rect(line, roi)
    if t_range is None:
        return []
    t_lo, t_hi = t_range

    x0, y0, x1, y1 = (float(v) for v in line)
    xs, ys, vals = _profile_along(line, img)
    n = len(vals)
    if n < 2:
        return []

    vals = _smooth1d(vals, sigma)
    # indices covered by the clipped interval
    i0 = int(np.ceil(t_lo * (n - 1)))
    i1 = int(np.floor(t_hi * (n - 1)))
    i0, i1 = max(0, i0), min(n - 1, i1)

    out: List[Dict[str, Any]] = []
    for i in range(i0, i1):
        a, b = vals[i], vals[i + 1]
        # A crossing happens when the <=threshold state changes (185 itself
        # counts as "dark": verified against camera_3 segments whose far end
        # sits exactly on a pixel of grey 185).
        if (a <= threshold) == (b <= threshold):
            continue
        t = (threshold - a) / (b - a)
        # sub-pixel position along the walk (i -> i+1 is one pixel)
        p = i + t
        # map to image coordinates (axis aligned: only one axis moves)
        if abs(x1 - x0) >= abs(y1 - y0):
            x = x0 + (x1 - x0) * (p / (n - 1)) if n > 1 else x0
            y = y0
        else:
            y = y0 + (y1 - y0) * (p / (n - 1)) if n > 1 else y0
            x = x0
        out.append({
            "x": float(x), "y": float(y), "t": float(p),
            "dir": "up" if b > a else "down",
            "grey_before": float(a), "grey_after": float(b),
        })
    return out


# ---------------------------------------------------------------------------
# Element detectors
# ---------------------------------------------------------------------------


def _pick_point(
    crossings: List[Dict[str, Any]],
    expected: Tuple[float, float],
    want_dir: Optional[str],
    mode: str,
) -> Optional[Dict[str, Any]]:
    cand = [c for c in crossings if want_dir is None or c["dir"] == want_dir]
    if not cand:
        return None
    if mode == "first":
        # first along the walk direction of the line (t ascending)
        return cand[0]
    # nearest to expected position (euclidean distance in image space)
    ex, ey = expected
    return min(cand, key=lambda c: (c["x"] - ex) ** 2 + (c["y"] - ey) ** 2)


def detect_point(
    elem: Dict[str, Any],
    img: np.ndarray,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    line = tuple(float(v) for v in elem["line"])
    roi = tuple(float(v) for v in elem["roi"])
    thr = float(elem.get("threshold", cfg.get("threshold", 185)))
    want_dir = cfg["is_inv_map"].get(bool(elem.get("isInv", True)))
    cr = edge_crossings(line, roi, thr, img, sigma=cfg["sigma"])
    expected = (float(elem["coordinates"][0]), float(elem["coordinates"][1]))
    pick = _pick_point(cr, expected, want_dir, cfg["point_select"])
    if pick is None:
        return {"coordinates": None, "crossings": cr}
    return {"coordinates": [pick["x"], pick["y"]], "crossings": cr}


def detect_segment(
    elem: Dict[str, Any],
    img: np.ndarray,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    line = tuple(float(v) for v in elem["line"])
    roi = tuple(float(v) for v in elem["roi"])
    thr = float(elem.get("threshold", cfg.get("threshold", 185)))
    cr = edge_crossings(line, roi, thr, img, sigma=cfg["sigma"])
    if len(cr) < 2:
        return {"coordinates": None, "crossings": cr}
    # outer two crossings along the walk direction
    c0 = min(cr, key=lambda c: c["t"])
    c1 = max(cr, key=lambda c: c["t"])
    return {"coordinates": [c0["x"], c0["y"], c1["x"], c1["y"]], "crossings": cr}


def detect_circle(
    elem: Dict[str, Any],
    img: np.ndarray,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Radial crossings in a band around the expected radius, then fit.

    From the expected centre we walk radially inside a band centred on the
    expected radius (the production element's own circle has no exact
    geometry available, so the band is the sanest working assumption).  For
    every angle the crossing closest to the expected radius is kept, then an
    algebraic (Kasa) circle fit estimates centre + radius.
    """
    cx, cy, r_exp = (float(v) for v in elem["coordinates"])
    thr = float(elem.get("threshold", cfg.get("threshold", 185)))
    h, w = img.shape[:2]

    band = float(cfg.get("circle_band", 150.0))
    step = max(0.1, float(cfg["circle_angle_step_deg"]))
    r_lo = max(1.0, r_exp - band)
    r_hi = r_exp + band
    pts: List[Tuple[float, float]] = []
    n_rays = 0
    for deg in np.arange(0.0, 360.0, step):
        a = np.deg2rad(deg)
        n = int(r_hi - r_lo) + 1
        if n < 2:
            continue
        n_rays += 1
        ts = np.arange(r_lo, r_lo + n)
        xs = cx + ts * np.cos(a)
        ys = cy + ts * np.sin(a)
        xi = np.clip(np.round(xs).astype(int), 0, w - 1)
        yi = np.clip(np.round(ys).astype(int), 0, h - 1)
        vals = img[yi, xi].astype(np.float64)
        vals = _smooth1d(vals, cfg["sigma"])
        crs = []
        for i in range(len(vals) - 1):
            va, vb = vals[i], vals[i + 1]
            if (va <= thr) != (vb <= thr):
                crs.append(r_lo + i + (thr - va) / (vb - va))
        if not crs:
            continue
        rr = min(crs, key=lambda v: abs(v - r_exp))
        pts.append((cx + rr * np.cos(a), cy + rr * np.sin(a)))

    if len(pts) < 3:
        return {"coordinates": None, "points": pts, "n_rays": n_rays}
    P = np.asarray(pts, dtype=np.float64)
    x, y = P[:, 0], P[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x ** 2 + y ** 2)
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return {"coordinates": None, "points": pts, "n_rays": n_rays}
    ccx, ccy = -sol[0] / 2, -sol[1] / 2
    r = float(np.sqrt(max(ccx ** 2 + ccy ** 2 - sol[2], 0.0)))
    return {"coordinates": [ccx, ccy, r], "points": pts, "n_rays": n_rays}


# ---------------------------------------------------------------------------
# Object result computation
# ---------------------------------------------------------------------------


def _seg_len(coords) -> float:
    if coords is None or len(coords) < 4:
        return 0.0
    return float(np.hypot(coords[2] - coords[0], coords[3] - coords[1]))


def compute_res(
    cal_mode: str,
    elements: List[Dict[str, Any]],
    img_h: int,
) -> float:
    """Compute the object-level res from element detection outputs (pixels).

    Uses the ROUNDED element coordinates — the production service emits
    integer element coordinates and derives res from those (verified: a
    sub-pixel res of 5143.5 corresponds to gt 5143 with y rounded to 329).
    """
    coords = [e.get("coords_rounded") for e in elements]

    if cal_mode == "height":
        # point element -> image_height - y
        if coords and coords[0] is not None:
            return float(img_h) - coords[0][1]
        return 0.0

    if cal_mode == "diameter":
        # circle element -> radius
        if coords and coords[0] is not None and len(coords[0]) >= 3:
            return float(coords[0][2])
        return 0.0

    if cal_mode == "distance":
        if len(elements) == 1:
            return _seg_len(coords[0])
        # multi-element combination
        if len(elements) >= 2 and coords[0] is not None and coords[1] is not None:
            t0 = elements[0].get("type")
            t1 = elements[1].get("type")
            c0, c1 = coords[0], coords[1]
            if t0 == "point" and t1 == "point":
                return float(np.hypot(c1[0] - c0[0], c1[1] - c0[1]))
            if t0 == "point" and t1 == "segment":
                # distance from point to segment line (axis aligned -> |dy|)
                return float(abs(c1[1] - c0[1]))
            if t0 == "segment" and t1 == "point":
                return float(abs(c0[1] - c1[1]))
    return 0.0


# ---------------------------------------------------------------------------
# Reproduce one camera
# ---------------------------------------------------------------------------


def reproduce_camera(
    cam_dir: str,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    req_path = os.path.join(cam_dir, "algorithm_request.json")
    resp_path = os.path.join(cam_dir, "algorithm_response.json")
    img_path = os.path.join(cam_dir, "current_image.jpg")
    req = json.load(open(req_path, encoding="utf-8"))
    gt = json.load(open(resp_path, encoding="utf-8"))
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"cannot decode image {img_path}")
    h, w = img.shape[:2]

    results = []
    for obj in req.get("objects", []):
        det_elems = []
        for e in obj.get("elements", []):
            etype = e.get("type")
            if etype == "point":
                r = detect_point(e, img, cfg)
            elif etype == "segment":
                r = detect_segment(e, img, cfg)
            elif etype == "circle":
                r = detect_circle(e, img, cfg)
            else:
                r = {"coordinates": None}
            det_elems.append({"type": etype, **r})
        # Round element coordinates first (production emits integers).
        for de in det_elems:
            c = de.get("coordinates")
            de["coords_rounded"] = (
                [_round_val(v, cfg["round_mode"]) for v in c]
                if c is not None else None
            )
        res = compute_res(obj.get("cal_mode", ""), det_elems, h)
        results.append({
            "object_index": len(results),
            "cal_mode": obj.get("cal_mode"),
            "res": res,
            "elements": [{k: v for k, v in e.items() if k != "crossings"}
                         for e in det_elems],
        })

    return {
        "camera_sn": req.get("camera_sn", os.path.basename(cam_dir)),
        "req": req,
        "gt": gt,
        "pred": {"objects": results},
        "image_h": h, "image_w": w,
    }


def format_report(cam: Dict[str, Any], round_mode: str) -> List[str]:
    lines: List[str] = []
    req, gt, pred = cam["req"], cam["gt"], cam["pred"]
    lines.append(f"camera {cam['camera_sn']}  img={cam['image_w']}x{cam['image_h']}")
    gt_objs = gt.get("calc_resutls", [])
    stats = {"res_ok": 0, "res_n": 0, "elem_ok": 0, "elem_1px": 0, "elem_n": 0}
    for i, po in enumerate(pred["objects"]):
        go = gt_objs[i] if i < len(gt_objs) else {}
        res_p, res_g = po["res"], go.get("res")
        res_match = res_g is not None and abs(res_p - res_g) < 0.5
        stats["res_n"] += 1
        stats["res_ok"] += 1 if res_match else 0
        lines.append(
            f"  obj[{i}] {po['cal_mode']:<8} res pred={res_p:9.1f} gt={res_g} "
            f"[{'OK' if res_match else 'DIFF'}]"
        )
        for j, (pe, ge) in enumerate(
            zip(po["elements"], go.get("elements", []))
        ):
            cp = pe.get("coords_rounded")
            cg = ge.get("coordinates")
            stats["elem_n"] += 1
            if cp is None or cg is None:
                tag = "NONE"
            else:
                dmax = max(abs(a - b) for a, b in zip(cp, cg))
                if cp == cg:
                    tag = "OK "
                    stats["elem_ok"] += 1
                elif dmax <= 1:
                    tag = f"~{dmax}"
                    stats["elem_1px"] += 1
                else:
                    tag = f"DIFF({dmax})"
            lines.append(f"    elem[{j}] {pe['type']:<7} pred={cp} gt={cg} [{tag}]")
    lines.append(
        f"  == res {stats['res_ok']}/{stats['res_n']}  |  elem OK "
        f"{stats['elem_ok']} +~1px {stats['elem_1px']}/{stats['elem_n']}"
    )
    return lines


def run_debug_dir(debug_dir: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    cams = []
    root = Path(debug_dir)
    cam_root = root / "cameras"
    if not cam_root.exists():
        raise RuntimeError(f"no cameras dir under {debug_dir}")
    for cam_path in sorted(cam_root.iterdir()):
        if cam_path.is_dir() and os.path.exists(
                cam_path / "algorithm_request.json"):
            cams.append(reproduce_camera(str(cam_path), cfg))
    return {"manifest": json.load(open(root / "manifest.json", encoding="utf-8")),
            "cameras": cams}


def main() -> None:
    ap = argparse.ArgumentParser(description="reproduce protocol measurements")
    ap.add_argument("debug_dir", help="measurement-debug-* directory")
    ap.add_argument("--out", default=None, help="write JSON report")
    ap.add_argument("--params", default=None, help="tune.json to override CONFIG")
    args = ap.parse_args()

    cfg = dict(CONFIG)
    if args.params:
        with open(args.params, encoding="utf-8") as f:
            cfg.update(json.load(f))

    result = run_debug_dir(args.debug_dir, cfg)
    print("=" * 78)
    for cam in result["cameras"]:
        for line in format_report(cam, cfg["round_mode"]):
            print(line)
        print("-" * 78)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=1, default=str)
        print(f"report written: {args.out}")




if __name__ == "__main__":
    main()
