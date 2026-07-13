#!/usr/bin/env python3
"""
Verify stereo calibration (R,T-based) by projecting board dots
from each camera to the OTHER camera's image plane using stereo R,T.

Key insight: with proper stereo calibration (R, T), dots detected in
camera B can be projected onto camera A's image (and vice versa).
If the calibration is correct, the projected dots should overlap
with the directly detected dots.

Usage:
    python tests/verify_stereo_RT.py
    python tests/verify_stereo_RT.py --pair 3
"""

import os
import sys
import json
import argparse
from typing import Optional

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from measure.measure_calibration import CameraCalibration, StereoRigCalibration
from measure.viz import to_bgr


# =========================================================================
# Configuration
# =========================================================================

IMAGE_DIR = os.environ.get("IMAGE_DIR", os.path.expanduser("~/桌面/temp/圆形标定板/part1"))
CALIB_DIR = os.path.join(os.path.dirname(__file__), "output", "stereo_rectification")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output", "stereo_verify_RT")

# Use the stereo-calibrated rig (with R,T), fall back to original rig
RIG_FILE = "rig_stereo.npz"

BLOB_PARAMS = {
    "minArea": 200,
    "maxArea": 20000,
    "minCircularity": 0.3,
    "minConvexity": 0.3,
}

DOT_RADIUS = 12
LINE_THICKNESS = 2


# =========================================================================
# Visualization helpers
# =========================================================================

def _hsv_to_bgr(h, s, v):
    hsv = np.uint8([[[h / 2, s * 255, v * 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return tuple(int(x) for x in bgr[0, 0])


def draw_overlay(vis, pts_detected, pts_projected, diffs):
    """Draw detected (green filled) and projected (red cross) dots."""
    for i, (pd, pp, diff) in enumerate(zip(pts_detected, pts_projected, diffs)):
        xd, yd = int(round(pd[0])), int(round(pd[1]))
        xp, yp = int(round(pp[0])), int(round(pp[1]))

        # Detected: green filled circle
        cv2.circle(vis, (xd, yd), DOT_RADIUS, (0, 255, 0), -1)
        cv2.circle(vis, (xd, yd), DOT_RADIUS + 2, (0, 0, 0), 1)

        # Projected: red cross
        half = DOT_RADIUS + 2
        cv2.line(vis, (xp - half, yp), (xp + half, yp), (0, 0, 255), 2)
        cv2.line(vis, (xp, yp - half), (xp, yp + half), (0, 0, 255), 2)

        # Link for large errors
        if diff > 3.0:
            cv2.line(vis, (xd, yd), (xp, yp), (255, 255, 0), 1)

    return vis


def draw_error_heatmap(vis, pts, diffs, vmin, vmax):
    """Draw per-dot error heatmap (green=0, red=max error)."""
    for i, (pt, diff) in enumerate(zip(pts, diffs)):
        x, y = int(round(pt[0])), int(round(pt[1]))
        t = np.clip((diff - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
        hue = 120 * (1.0 - t)
        bgr = _hsv_to_bgr(hue, 1.0, 1.0)
        cv2.circle(vis, (x, y), DOT_RADIUS,
                   (int(bgr[0]), int(bgr[1]), int(bgr[2])), -1)
        cv2.circle(vis, (x, y), DOT_RADIUS + 2, (0, 0, 0), 1)
    return vis


def make_composite_summary(all_results, imgs_a, imgs_b, output_dir):
    """Create a summary grid: for each pair, show A-overlay + B-overlay."""
    n = len(all_results)
    if n == 0:
        return

    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    tile_h, tile_w = 350, 300
    pad = 8

    canvas = np.zeros(
        (nrows * (tile_h + pad) + pad,
         ncols * (tile_w * 2 + pad) + pad, 3),
        dtype=np.uint8,
    )
    canvas[:] = (30, 30, 30)

    for idx in range(n):
        r = idx // ncols
        c = idx % ncols
        pi = all_results[idx]["pair"] - 1

        img_a = to_bgr(imgs_a[pi])
        img_b = to_bgr(imgs_b[pi])

        # A overlay
        vis_a = img_a.copy()
        pts_a = np.array(all_results[idx]["pts_a_detected"])
        proj_a = np.array(all_results[idx]["pts_a_from_b_projected"])
        diffs_a = np.array(all_results[idx]["diff_per_dot_a"])
        draw_overlay(vis_a, pts_a, proj_a, diffs_a)

        # B overlay
        vis_b = img_b.copy()
        pts_b = np.array(all_results[idx]["pts_b_detected"])
        proj_b = np.array(all_results[idx]["pts_b_from_a_projected"])
        diffs_b = np.array(all_results[idx]["diff_per_dot_b"])
        draw_overlay(vis_b, pts_b, proj_b, diffs_b)

        def _fit(img, tw, th):
            h, w = img.shape[:2]
            s = min(th / h, tw / w)
            return cv2.resize(img, (int(w * s), int(h * s)),
                              interpolation=cv2.INTER_AREA)

        va = _fit(vis_a, tile_w, tile_h)
        vb = _fit(vis_b, tile_w, tile_h)

        y0 = r * (tile_h + pad) + pad
        x0 = c * (tile_w * 2 + pad) + pad
        canvas[y0:y0 + va.shape[0], x0:x0 + va.shape[1]] = va
        xb = x0 + tile_w
        canvas[y0:y0 + vb.shape[0], xb:xb + vb.shape[1]] = vb

        cv2.putText(canvas, f"P{all_results[idx]['pair']} A err={all_results[idx]['mean_error_px']:.2f}",
                    (x0 + 4, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1)
        cv2.putText(canvas, f"P{all_results[idx]['pair']} B",
                    (xb + 4, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1)

    path = os.path.join(output_dir, "summary_grid.png")
    cv2.imwrite(path, canvas)
    print(f"  Summary grid: {path}")
    return canvas


# =========================================================================
# Main
# =========================================================================

def load_image_pairs(img_dir):
    pairs = []
    entries = sorted([d for d in os.listdir(img_dir) if d.isdigit()], key=int)
    for entry in entries:
        path_a = os.path.join(img_dir, entry, "1.jpg")
        path_b = os.path.join(img_dir, entry, "2.jpg")
        if not os.path.exists(path_a) or not os.path.exists(path_b):
            continue
        a = cv2.imread(path_a, cv2.IMREAD_GRAYSCALE)
        b = cv2.imread(path_b, cv2.IMREAD_GRAYSCALE)
        pairs.append((int(entry), a, b))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Verify stereo R,T quality")
    parser.add_argument("--pair", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Stereo R,T Verification")
    print("=" * 60)

    # Load calibrations
    rig_path = os.path.join(CALIB_DIR, RIG_FILE)
    if not os.path.exists(rig_path):
        print(f"ERROR: Stereo rig file not found: {rig_path}")
        print("Run calibrate_stereo() first.")
        sys.exit(1)

    calib_a = CameraCalibration.load(os.path.join(CALIB_DIR, "calib_a.npz"))
    calib_b = CameraCalibration.load(os.path.join(CALIB_DIR, "calib_b.npz"))
    rig = StereoRigCalibration.load(rig_path, calib_a=calib_a, calib_b=calib_b)

    if rig.R is None:
        print("ERROR: Loaded rig has no stereo R,T.")
        sys.exit(1)

    print(f"  Camera A: {calib_a.grid_size} grid, "
          f"reproj err={calib_a.reprojection_error:.3f} px")
    print(f"  Camera B: {calib_b.grid_size} grid, "
          f"reproj err={calib_b.reprojection_error:.3f} px")
    print(f"  Stereo RMS: {rig.stereo_rms:.4f}")
    print(f"  Baseline: {np.linalg.norm(rig.T):.1f} mm")
    print(f"  R:\n{np.array2string(rig.R, precision=4, suppress_small=True)}")
    print(f"  T: [ {rig.T[0,0]:.2f}, {rig.T[1,0]:.2f}, {rig.T[2,0]:.2f} ]")
    print()

    # Load images
    all_pairs = load_image_pairs(IMAGE_DIR)
    print(f"[1] Loaded {len(all_pairs)} image pairs")

    if args.pair is not None:
        all_pairs = [p for p in all_pairs if p[0] == args.pair]
        if not all_pairs:
            print(f"ERROR: Pair {args.pair} not found")
            sys.exit(1)

    # ------------------------------------------------------------------
    # Verify each pair
    # ------------------------------------------------------------------
    all_results = []
    imgs_a_gray = []
    imgs_b_gray = []

    for pair_idx, img_a, img_b in all_pairs:
        print(f"\n[2] Pair {pair_idx}: verifying...")
        result = rig.verify_stereo_on_board(img_a, img_b)
        result["pair"] = pair_idx

        if "error" in result:
            print(f"  ERROR: {result['error']}")
            all_results.append(result)
            continue

        all_results.append(result)
        imgs_a_gray.append(img_a)
        imgs_b_gray.append(img_b)

        print(f"  {result['num_points']} dots")
        print(f"  mean={result['mean_error_px']:.3f} px  "
              f"median={result['median_error_px']:.3f} px  "
              f"max={result['max_error_px']:.3f} px  "
              f"std={result['std_error_px']:.3f} px")

        # Per-pair visualizations
        img_a_bgr = to_bgr(img_a)
        img_b_bgr = to_bgr(img_b)
        pts_a = np.array(result["pts_a_detected"])
        pts_b = np.array(result["pts_b_detected"])
        proj_a = np.array(result["pts_a_from_b_projected"])
        proj_b = np.array(result["pts_b_from_a_projected"])
        diffs_a = np.array(result["diff_per_dot_a"])
        diffs_b = np.array(result["diff_per_dot_b"])

        # Camera A with B's dots projected
        vis_a = img_a_bgr.copy()
        draw_overlay(vis_a, pts_a, proj_a, diffs_a)
        cv2.putText(vis_a,
                    f"Pair {pair_idx} | A det(green) vs B->A proj(red) | "
                    f"mean={result['mean_error_px']:.2f}px",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        pa = os.path.join(OUTPUT_DIR, f"pair{pair_idx:02d}_A_projected.png")
        cv2.imwrite(pa, vis_a)

        # Camera B with A's dots projected
        vis_b = img_b_bgr.copy()
        draw_overlay(vis_b, pts_b, proj_b, diffs_b)
        cv2.putText(vis_b,
                    f"Pair {pair_idx} | B det(green) vs A->B proj(red) | "
                    f"mean={result['mean_error_px']:.2f}px",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        pb = os.path.join(OUTPUT_DIR, f"pair{pair_idx:02d}_B_projected.png")
        cv2.imwrite(pb, vis_b)

        # Error heatmap on A
        heat_a = img_a_bgr.copy()
        all_diffs = np.concatenate([diffs_a, diffs_b])
        vmin, vmax = max(0.0, all_diffs.min()), all_diffs.max()
        draw_error_heatmap(heat_a, pts_a, diffs_a, vmin, vmax)
        bar_h, bar_w = 16, 300
        bar_x, bar_y = 20, img_a.shape[0] - bar_h - 30
        for col in range(bar_w):
            t = col / max(bar_w - 1, 1)
            hue = 120 * (1.0 - t)
            bgr = _hsv_to_bgr(hue, 1.0, 1.0)
            cv2.line(heat_a, (bar_x + col, bar_y),
                     (bar_x + col, bar_y + bar_h),
                     (int(bgr[0]), int(bgr[1]), int(bgr[2])), 1)
        cv2.rectangle(heat_a, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 1)
        cv2.putText(heat_a, f"{vmin:.2f}", (bar_x - 5, bar_y + bar_h + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.putText(heat_a, f"{vmax:.2f} px", (bar_x + bar_w - 30, bar_y + bar_h + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.putText(heat_a, f"Pair {pair_idx} error heatmap (A-side)",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        ph = os.path.join(OUTPUT_DIR, f"pair{pair_idx:02d}_heatmap.png")
        cv2.imwrite(ph, heat_a)

        print(f"    outputs: {os.path.basename(pa)}, {os.path.basename(pb)}, "
              f"{os.path.basename(ph)}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    valid = [r for r in all_results if "error" not in r]
    if not valid:
        print("No valid results.")
        return

    for r in valid:
        print(f"Pair {r['pair']:2d}: mean={r['mean_error_px']:.3f}px  "
              f"median={r['median_error_px']:.3f}px  "
              f"max={r['max_error_px']:.3f}px  "
              f"std={r['std_error_px']:.3f}px")

    means = [r["mean_error_px"] for r in valid]
    medians = [r["median_error_px"] for r in valid]
    maxes = [r["max_error_px"] for r in valid]

    print(f"\n--- Aggregate ({len(valid)} pairs) ---")
    print(f"  Mean of means:   {np.mean(means):.3f} px")
    print(f"  Mean of medians: {np.mean(medians):.3f} px")
    print(f"  Worst max:       {np.max(maxes):.3f} px")

    # Save JSON summary
    summary_path = os.path.join(OUTPUT_DIR, "verify_RT_summary.json")
    for r in valid:
        for k in list(r.keys()):
            if isinstance(r[k], np.ndarray):
                r[k] = r[k].tolist()
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(valid, f, indent=2, ensure_ascii=False)

    make_composite_summary(valid, imgs_a_gray, imgs_b_gray, OUTPUT_DIR)

    print(f"\nSummary JSON: {summary_path}")
    print(f"Visual outputs: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
