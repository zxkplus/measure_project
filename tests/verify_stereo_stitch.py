#!/usr/bin/env python3
"""
Verify stereo stitching accuracy by projecting detected circle centers
from both cameras onto the shared canvas and comparing their positions.

Usage:
    python tests/verify_stereo_stitch.py
    python tests/verify_stereo_stitch.py --pair 3  # verify a specific pair
    python tests/verify_stereo_stitch.py --all      # verify all pairs
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
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output", "stereo_verify")

BLOB_PARAMS = {
    "minArea": 200,
    "maxArea": 20000,
    "minThreshold": 1,
    "maxThreshold": 255,
    "thresholdStep": 5,
    "minCircularity": 0.3,
    "minConvexity": 0.3,
}

PIXEL_SIZE_MM = 0.05

# Colors (BGR)
CAM_A_COLOR = (0, 255, 0)    # green — camera A projected circles
CAM_B_COLOR = (0, 0, 255)    # red   — camera B projected circles
LINK_COLOR = (255, 255, 0)   # cyan  — lines connecting corresponding dots


# =========================================================================
# Core logic
# =========================================================================


def detect_circles(img: np.ndarray, grid_size: tuple) -> Optional[np.ndarray]:
    """Detect symmetric dot-grid circles. Returns (N, 2) float32 or None."""
    detector = CameraCalibration._make_blob_detector(BLOB_PARAMS)
    found, centers = cv2.findCirclesGrid(
        img, grid_size,
        cv2.CALIB_CB_SYMMETRIC_GRID,
        blobDetector=detector,
    )
    if not found:
        return None
    return centers.reshape(-1, 2).astype(np.float32)


def project_to_canvas(
    pts: np.ndarray,
    calib: CameraCalibration,
    H: np.ndarray,
    M_full: np.ndarray,
) -> np.ndarray:
    """
    Project image points to canvas pixel coordinates via:
    undistort → homography (image→board) → affine (board→canvas).

    Args:
        pts: (N, 2) float32 image pixel coordinates.
        calib: Calibrated camera (must have K, D).
        H: (3, 3) homography from image to board-plane (mm).
        M_full: (3, 3) composite affine for board-mm → canvas-pixels.

    Returns:
        (N, 2) canvas pixel coordinates.
    """
    h, w = calib.image_size

    # Get optimal new K for rectified image
    new_K, _ = cv2.getOptimalNewCameraMatrix(calib.K, calib.D, (w, h), 0.0)

    # Undistort the points
    # cv2.undistortPoints expects (N, 1, 2) and returns (N, 1, 2)
    pts_undist = cv2.undistortPoints(
        pts.reshape(-1, 1, 2), calib.K, calib.D, P=new_K
    ).reshape(-1, 2)

    # Adjust homography for rectified intrinsics
    H_adj = H @ calib.K @ np.linalg.inv(new_K)

    # Convert to homogeneous
    pts_h = np.hstack([pts_undist, np.ones((len(pts_undist), 1))]).T  # (3, N)

    # Board-plane mm coordinates
    board_pts = (H_adj @ pts_h)  # (3, N)
    board_pts = board_pts[:2] / (board_pts[2] + 1e-12)  # (2, N)

    # Canvas pixels
    canvas_pts = (M_full[:2, :2] @ board_pts + M_full[:2, 2:3])  # (2, N)
    return canvas_pts.T  # (N, 2)


def verify_pair(
    rig: StereoRigCalibration,
    img_a: np.ndarray,
    img_b: np.ndarray,
    pair_idx: int,
    output_dir: str,
) -> dict:
    """
    Verify stitching for a single image pair.

    Returns:
        Dict with error metrics.
    """
    cols, rows = rig.calib_a.grid_size

    # Detect circles in both images
    pts_a = detect_circles(img_a, (cols, rows))
    pts_b = detect_circles(img_b, (cols, rows))

    if pts_a is None:
        return {"pair": pair_idx, "error": "Camera A detection failed"}
    if pts_b is None:
        return {"pair": pair_idx, "error": "Camera B detection failed"}

    # Stitch first to establish canvas_size/canvas_affine for this pair
    stitch_result = rig.stitch(img_a, img_b, pixel_size_mm=PIXEL_SIZE_MM)
    stitched = stitch_result["stitched_image"]
    canvas_h, canvas_w = rig.canvas_size

    # Compute canvas transform from updated rig state
    M = rig.canvas_affine  # (2, 3)
    M_full = np.vstack([M, [[0, 0, 1]]])  # (3, 3)

    # Project circle centers from both cameras to canvas coords
    canvas_a = project_to_canvas(pts_a, rig.calib_a, rig.H_A, M_full)
    canvas_b = project_to_canvas(pts_b, rig.calib_b, rig.H_B, M_full)

    # Distance between corresponding points (same grid index)
    diffs = np.linalg.norm(canvas_a - canvas_b, axis=1)
    mean_err = float(np.mean(diffs))
    max_err = float(np.max(diffs))
    std_err = float(np.std(diffs))
    median_err = float(np.median(diffs))

    result = {
        "pair": pair_idx,
        "num_points": len(pts_a),
        "canvas_size": list(rig.canvas_size),
        "mean_error_px": mean_err,
        "median_error_px": median_err,
        "max_error_px": max_err,
        "std_error_px": std_err,
        "all_errors_px": diffs.tolist(),
    }

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    canvas_bg = to_bgr(stitched)

    # --- Visual 1: overlay both cameras' projected dots on stitched canvas ---
    overlay = canvas_bg.copy()

    dot_radius = max(6, int(min(canvas_h, canvas_w) * 0.005))
    line_thickness = max(1, dot_radius // 3)

    for i, (pa, pb, diff) in enumerate(zip(canvas_a, canvas_b, diffs)):
        xa, ya = int(round(pa[0])), int(round(pa[1]))
        xb, yb = int(round(pb[0])), int(round(pb[1]))

        # Camera A: green filled circle
        cv2.circle(overlay, (xa, ya), dot_radius, CAM_A_COLOR, -1)
        cv2.circle(overlay, (xa, ya), dot_radius + 1, (0, 0, 0), 1)

        # Camera B: red cross (offset slightly so both are visible if they differ)
        half = dot_radius + 2
        cv2.line(overlay, (xb - half, yb - half), (xb + half, yb + half), CAM_B_COLOR, 2)
        cv2.line(overlay, (xb - half, yb + half), (xb + half, yb - half), CAM_B_COLOR, 2)

        # Connecting line between corresponding dots
        if diff > 3.0:
            cv2.line(overlay, (xa, ya), (xb, yb), LINK_COLOR, 1)

    # Legend
    y_legend = 30
    cv2.putText(overlay, "Green=CamA  Red=CamB  Cyan line=error > 3px",
                (10, y_legend), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(overlay,
                f"Pair {pair_idx} | mean={mean_err:.2f}px  median={median_err:.2f}px  "
                f"max={max_err:.2f}px  std={std_err:.2f}px",
                (10, y_legend + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    overlay_path = os.path.join(output_dir, f"pair{pair_idx:02d}_overlay.png")
    cv2.imwrite(overlay_path, overlay)

    # --- Visual 2: side-by-side (cam A projected only | cam B projected only) ---
    side_a = canvas_bg.copy()
    side_b = canvas_bg.copy()

    for pa in canvas_a:
        xa, ya = int(round(pa[0])), int(round(pa[1]))
        cv2.circle(side_a, (xa, ya), dot_radius, CAM_A_COLOR, -1)
        cv2.circle(side_a, (xa, ya), dot_radius + 1, (0, 0, 0), 1)
    cv2.putText(side_a, f"Camera A only", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    for pb in canvas_b:
        xb, yb = int(round(pb[0])), int(round(pb[1]))
        cv2.circle(side_b, (xb, yb), dot_radius, CAM_B_COLOR, -1)
        cv2.circle(side_b, (xb, yb), dot_radius + 1, (0, 0, 0), 1)
    cv2.putText(side_b, f"Camera B only", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    side_by_side = np.hstack([side_a, side_b])
    side_path = os.path.join(output_dir, f"pair{pair_idx:02d}_side_by_side.png")
    cv2.imwrite(side_path, side_by_side)

    # --- Visual 3: error map (color-coded per-dot error) ---
    error_map = canvas_bg.copy()
    vmin, vmax = max(0.0, diffs.min()), diffs.max()
    if vmax - vmin < 1e-6:
        vmax = vmin + 1.0

    for i, (pa, diff) in enumerate(zip(canvas_a, diffs)):
        xa, ya = int(round(pa[0])), int(round(pa[1]))
        # Scale error to color range: 0 (green) → 1 (red)
        t = np.clip((diff - vmin) / (vmax - vmin), 0.0, 1.0)
        # HSV: hue from 120 (green) to 0 (red)
        hue = 120 * (1.0 - t)
        bgr = _hsv_to_bgr(hue, 1.0, 1.0)
        cv2.circle(error_map, (xa, ya), dot_radius, (int(bgr[0]), int(bgr[1]), int(bgr[2])), -1)
        cv2.circle(error_map, (xa, ya), dot_radius + 1, (0, 0, 0), 1)
        # Label error value on significant errors
        if diff > 2.0:
            cv2.putText(error_map, f"{diff:.1f}", (xa - 15, ya - dot_radius - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    # Color bar at bottom
    bar_h = 24
    bar_y = canvas_h - bar_h - 10
    bar_w = min(400, canvas_w - 40)
    bar_x = (canvas_w - bar_w) // 2
    for col in range(bar_w):
        t = col / max(bar_w - 1, 1)
        hue = 120 * (1.0 - t)
        bgr = _hsv_to_bgr(hue, 1.0, 1.0)
        cv2.line(error_map, (bar_x + col, bar_y),
                 (bar_x + col, bar_y + bar_h),
                 (int(bgr[0]), int(bgr[1]), int(bgr[2])), 1)
    cv2.rectangle(error_map, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (255, 255, 255), 1)
    cv2.putText(error_map, f"{vmin:.2f} px", (bar_x - 10, bar_y + bar_h + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    cv2.putText(error_map, f"{vmax:.2f} px", (bar_x + bar_w - 40, bar_y + bar_h + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    cv2.putText(error_map, "Error (CamA vs CamB)",
                (bar_x + bar_w // 2 - 80, bar_y + bar_h + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    error_path = os.path.join(output_dir, f"pair{pair_idx:02d}_error_map.png")
    cv2.imwrite(error_path, error_map)

    # --- Visual 4: zoomed-in ROI of overlapping dots ---
    # Find the region around the stitched dot cluster
    all_canvas = np.vstack([canvas_a, canvas_b])
    center_x = float(np.mean(all_canvas[:, 0]))
    center_y = float(np.mean(all_canvas[:, 1]))
    zoom_size = 500
    x0 = max(0, int(center_x - zoom_size // 2))
    y0 = max(0, int(center_y - zoom_size // 2))
    x1 = min(canvas_w, x0 + zoom_size)
    y1 = min(canvas_h, y0 + zoom_size)

    if x1 > x0 and y1 > y0:
        zoom = overlay[y0:y1, x0:x1].copy()
        cv2.putText(zoom, f"ZOOM: pair {pair_idx}", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        zoom_path = os.path.join(output_dir, f"pair{pair_idx:02d}_zoom.png")
        cv2.imwrite(zoom_path, zoom)
        result["zoom_region"] = [x0, y0, x1, y1]

    result["outputs"] = {
        "overlay": overlay_path,
        "side_by_side": side_path,
        "error_map": error_path,
    }
    if x1 > x0 and y1 > y0:
        result["outputs"]["zoom"] = zoom_path

    return result


def _hsv_to_bgr(h, s, v):
    """Convert HSV (h in 0-360, s,v in 0-1) to BGR (0-255)."""
    hsv = np.uint8([[[h / 2, s * 255, v * 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return tuple(int(x) for x in bgr[0, 0])


# =========================================================================
# Load image pairs
# =========================================================================


def load_image_pairs(img_dir: str) -> list:
    """Load all image pairs from numbered subdirectories."""
    pairs = []
    entries = sorted(
        [d for d in os.listdir(img_dir) if d.isdigit()],
        key=int,
    )
    for entry in entries:
        path_a = os.path.join(img_dir, entry, "1.jpg")
        path_b = os.path.join(img_dir, entry, "2.jpg")
        if not os.path.exists(path_a) or not os.path.exists(path_b):
            print(f"  Skipping {entry}: missing images")
            continue
        a = cv2.imread(path_a, cv2.IMREAD_GRAYSCALE)
        b = cv2.imread(path_b, cv2.IMREAD_GRAYSCALE)
        pairs.append((int(entry), a, b))
    return pairs


# =========================================================================
# Main
# =========================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Verify stereo stitching accuracy")
    parser.add_argument("--pair", type=int, default=None,
                        help="Verify a specific pair (1-indexed). Default: all pairs.")
    parser.add_argument("--all", action="store_true",
                        help="Verify all pairs (default if --pair not given).")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Stereo Stitch Verification")
    print("=" * 60)
    print(f"Image dir:  {IMAGE_DIR}")
    print(f"Calib dir:  {CALIB_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")
    print()

    # Load calibrations
    print("[1] Loading calibrations...")
    calib_a = CameraCalibration.load(os.path.join(CALIB_DIR, "calib_a.npz"))
    calib_b = CameraCalibration.load(os.path.join(CALIB_DIR, "calib_b.npz"))
    rig = StereoRigCalibration.load(
        os.path.join(CALIB_DIR, "rig.npz"),
        calib_a=calib_a, calib_b=calib_b,
    )
    print(f"  Camera A: {calib_a.grid_size} grid, "
          f"reproj err={calib_a.reprojection_error:.3f} px")
    print(f"  Camera B: {calib_b.grid_size} grid, "
          f"reproj err={calib_b.reprojection_error:.3f} px")
    print(f"  Canvas: {rig.canvas_size[1]}×{rig.canvas_size[0]} px")
    print()

    # Load image pairs
    print("[2] Loading image pairs...")
    all_pairs = load_image_pairs(IMAGE_DIR)
    print(f"  Loaded {len(all_pairs)} pairs")

    # Select pairs to verify
    if args.pair is not None:
        pairs = [(p[0], p[1], p[2]) for p in all_pairs if p[0] == args.pair]
        if not pairs:
            print(f"ERROR: Pair {args.pair} not found")
            sys.exit(1)
    else:
        pairs = all_pairs

    # ------------------------------------------------------------------
    # Verify each pair
    # ------------------------------------------------------------------
    all_results = []

    for pair_idx, img_a, img_b in pairs:
        print(f"[3] Verifying pair {pair_idx}...")
        result = verify_pair(rig, img_a, img_b, pair_idx, OUTPUT_DIR)
        all_results.append(result)

        if "error" in result:
            print(f"  ERROR: {result['error']}")
        else:
            print(f"  {result['num_points']} dots, "
                  f"mean={result['mean_error_px']:.3f} px, "
                  f"median={result['median_error_px']:.3f} px, "
                  f"max={result['max_error_px']:.3f} px, "
                  f"std={result['std_error_px']:.3f} px")
            for name, path in result.get("outputs", {}).items():
                print(f"    {name}: {path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    valid_results = [r for r in all_results if "error" not in r]

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if not valid_results:
        print("No valid results.")
        return

    for r in valid_results:
        print(f"\nPair {r['pair']:2d}: {r['num_points']} dots")
        print(f"  mean={r['mean_error_px']:.3f} px  "
              f"median={r['median_error_px']:.3f} px  "
              f"max={r['max_error_px']:.3f} px  "
              f"std={r['std_error_px']:.3f} px")

    # Aggregate
    all_means = [r["mean_error_px"] for r in valid_results]
    all_medians = [r["median_error_px"] for r in valid_results]
    all_maxes = [r["max_error_px"] for r in valid_results]

    print(f"\n--- Aggregate over {len(valid_results)} pairs ---")
    print(f"  Mean of means:   {np.mean(all_means):.3f} px")
    print(f"  Mean of medians: {np.mean(all_medians):.3f} px")
    print(f"  Worst max:       {np.max(all_maxes):.3f} px")

    # Write summary
    summary_path = os.path.join(OUTPUT_DIR, "verify_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved: {summary_path}")
    print(f"Visual outputs: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
