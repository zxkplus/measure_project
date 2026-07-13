#!/usr/bin/env python3
"""
Stereo rig calibration & rectification validation with real dot-grid board images.

Usage:
    python tests/test_stereo_rectification.py

    # Override image directory:
    IMAGE_DIR=/path/to/images python tests/test_stereo_rectification.py

The script expects image pairs in subdirectories 1..N, each containing
`1.jpg` (camera A) and `2.jpg` (camera B) that observe the same board.

Workflow:
    1. Single-camera calibration per camera (Zhang) on all poses.
    2. Stereo rig calibration on a chosen reference pair (homography per camera).
    3. Stitch every pair and save diagnostics.
"""

import os
import sys
import json
import textwrap
from typing import List, Tuple
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from measure.measure_calibration import CameraCalibration, StereoRigCalibration
from measure.viz import to_bgr


# =========================================================================
# Configuration
# =========================================================================

IMAGE_DIR = os.environ.get(
    "IMAGE_DIR",
    os.path.expanduser("~/桌面/temp/圆形标定板/part1"),
)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output", "stereo_rectification")

GRID_SIZE = (7, 7)
CIRCLE_SPACING_MM = float(os.environ.get("CIRCLE_SPACING_MM", 3.75))
PIXEL_SIZE_MM = float(os.environ.get("PIXEL_SIZE_MM", 0.05))
REFERENCE_PAIR = int(os.environ.get("REFERENCE_PAIR", 1))  # which pair to use for rig calib

# Blob detector tuned for large dots (radius ≈ 28-38 px → area ≈ 2460-4540)
BLOB_PARAMS = {
    "minArea": 200,
    "maxArea": 20000,
    "minThreshold": 1,
    "maxThreshold": 255,
    "thresholdStep": 5,
    "minCircularity": 0.3,
    "minConvexity": 0.3,
}


# =========================================================================
# Helpers
# =========================================================================

def load_image_pairs(img_dir: str) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Load all image pairs from numbered subdirectories.

    Returns:
        List of (img_a, img_b) pairs, sorted by directory number.
    """
    pairs: List[Tuple[np.ndarray, np.ndarray]] = []
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
        pairs.append((a, b))
    return pairs


def make_composite(images: List[np.ndarray], labels: List[str],
                   ncols: int = 3, tile_h: int = 400) -> np.ndarray:
    """Arrange images in a grid with labels for side-by-side comparison."""
    n = len(images)
    nrows = (n + ncols - 1) // ncols
    h_pad, w_pad = 4, 4

    # Compute tile widths scaled by aspect ratio
    aspect = images[0].shape[1] / max(images[0].shape[0], 1)
    tile_w = int(tile_h * aspect)

    canvas = np.zeros(
        (nrows * (tile_h + h_pad) + h_pad,
         ncols * (tile_w + w_pad) + w_pad, 3),
        dtype=np.uint8,
    )
    canvas[:] = (40, 40, 40)

    for idx, (img, lbl) in enumerate(zip(images, labels)):
        r = idx // ncols
        c = idx % ncols
        if len(img.shape) == 2:
            disp = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            disp = img.copy()
        # Resize to tile
        h, w = disp.shape[:2]
        scale = min(tile_h / h, tile_w / w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(disp, (new_w, new_h), interpolation=cv2.INTER_AREA)
        y0 = r * (tile_h + h_pad) + h_pad + (tile_h - new_h) // 2
        x0 = c * (tile_w + w_pad) + w_pad + (tile_w - new_w) // 2
        canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
        # Label
        cv2.putText(canvas, lbl, (x0 + 4, y0 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return canvas


# =========================================================================
# Main
# =========================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Stereo Rectification Validation")
    print("=" * 60)
    print(f"Image dir:      {IMAGE_DIR}")
    print(f"Output dir:     {OUTPUT_DIR}")
    print(f"Grid size:      {GRID_SIZE}")
    print(f"Circle spacing: {CIRCLE_SPACING_MM} mm (affects canvas scale only)")
    print(f"Pixel size:     {PIXEL_SIZE_MM} mm/px")
    print()

    # ------------------------------------------------------------------
    # 1. Load images
    # ------------------------------------------------------------------
    print("[1/5] Loading image pairs...")
    pairs = load_image_pairs(IMAGE_DIR)
    print(f"  Loaded {len(pairs)} pairs")
    if len(pairs) < 3:
        print("ERROR: Need at least 3 pairs for calibration")
        sys.exit(1)

    imgs_a = [p[0] for p in pairs]
    imgs_b = [p[1] for p in pairs]

    # Pre-load color versions for matched-circle visualization
    imgs_a_color = [cv2.imread(os.path.join(IMAGE_DIR, d, "1.jpg")) for d in
                    sorted([d for d in os.listdir(IMAGE_DIR) if d.isdigit()], key=int)
                    if os.path.exists(os.path.join(IMAGE_DIR, d, "1.jpg"))
                    and os.path.exists(os.path.join(IMAGE_DIR, d, "2.jpg"))]
    imgs_b_color = [cv2.imread(os.path.join(IMAGE_DIR, d, "2.jpg")) for d in
                    sorted([d for d in os.listdir(IMAGE_DIR) if d.isdigit()], key=int)
                    if os.path.exists(os.path.join(IMAGE_DIR, d, "1.jpg"))
                    and os.path.exists(os.path.join(IMAGE_DIR, d, "2.jpg"))]
    # ------------------------------------------------------------------
    # 2. Single-camera calibration
    # ------------------------------------------------------------------
    print("[2/5] Calibrating camera A...")
    calib_a = CameraCalibration(
        grid_size=GRID_SIZE,
        circle_spacing_mm=CIRCLE_SPACING_MM,
    )
    result_a = calib_a.calibrate(imgs_a, blob_detector_params=BLOB_PARAMS)
    print(f"  Detected: {result_a['num_images']}/{len(imgs_a)} images")
    print(f"  Reprojection error: {result_a['reprojection_error']:.4f} px")

    print("[2/5] Calibrating camera B...")
    calib_b = CameraCalibration(
        grid_size=GRID_SIZE,
        circle_spacing_mm=CIRCLE_SPACING_MM,
    )
    result_b = calib_b.calibrate(imgs_b, blob_detector_params=BLOB_PARAMS)
    print(f"  Detected: {result_b['num_images']}/{len(imgs_b)} images")
    print(f"  Reprojection error: {result_b['reprojection_error']:.4f} px")

    # Save intrinsics
    calib_a.save(os.path.join(OUTPUT_DIR, "calib_a.npz"))
    calib_b.save(os.path.join(OUTPUT_DIR, "calib_b.npz"))

    # ------------------------------------------------------------------
    # 3. Stereo rig calibration (homography per camera)
    # ------------------------------------------------------------------
    ref_idx = REFERENCE_PAIR - 1  # 0-indexed
    print(f"[3/5] Stereo rig calibration using pair {REFERENCE_PAIR}...")
    rig = StereoRigCalibration(calib_a, calib_b)
    rig_result = rig.calibrate(
        imgs_a[ref_idx], imgs_b[ref_idx],
        blob_detector_params=BLOB_PARAMS,
    )
    print(f"  Camera A homography reproj error: {rig_result['reprojection_error_a']:.4f} px")
    print(f"  Camera B homography reproj error: {rig_result['reprojection_error_b']:.4f} px")
    print(f"  Canvas size: {rig_result['canvas_size']}")

    rig.save(os.path.join(OUTPUT_DIR, "rig.npz"))

    # ------------------------------------------------------------------
    # 3b. Visualize matched circle centers on the reference pair
    # ------------------------------------------------------------------
    print("[3b/5] Drawing matched circles on reference pair...")
    matched_dir = os.path.join(OUTPUT_DIR, "matched_circles")
    os.makedirs(matched_dir, exist_ok=True)

    # Re-detect dots on reference pair
    cols, rows = GRID_SIZE
    flags = cv2.CALIB_CB_SYMMETRIC_GRID
    detector = CameraCalibration._make_blob_detector(BLOB_PARAMS)

    def _detect_dots(img):
        found, centers = cv2.findCirclesGrid(
            img, (cols, rows), flags=flags, blobDetector=detector)
        if not found:
            return None
        return centers.reshape(-1, 2).astype(np.float32)

    pts_a = _detect_dots(imgs_a[ref_idx])
    pts_b = _detect_dots(imgs_b[ref_idx])

    # Build model points (same logic as calibrate)
    spacing = CIRCLE_SPACING_MM
    model_pts = np.zeros((rows * cols, 2), dtype=np.float32)
    for j in range(rows):
        for i in range(cols):
            model_pts[j * cols + i] = (i * spacing, j * spacing)

    # Invert homographies to project model points back to image
    H_A_inv = np.linalg.inv(rig_result["H_A"])
    H_B_inv = np.linalg.inv(rig_result["H_B"])

    def _draw_circles(img_color, detected_pts, reproj_pts):
        """Draw detected (green filled) and reprojected (red cross) circle centers."""
        vis = img_color.copy()
        h, w = vis.shape[:2]
        radius = max(3, int(min(h, w) * 0.006))
        # Detected: green filled circle with index
        for i, pt in enumerate(detected_pts):
            cx, cy = int(round(pt[0])), int(round(pt[1]))
            cv2.circle(vis, (cx, cy), radius, (0, 255, 0), -1)
            cv2.circle(vis, (cx, cy), radius + 1, (0, 0, 0), 1)
            cv2.putText(vis, str(i), (cx + radius + 2, cy - radius - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        # Reprojected: red cross
        for i, pt in enumerate(reproj_pts):
            cx, cy = int(round(pt[0])), int(round(pt[1]))
            half = radius + 2
            cv2.line(vis, (cx - half, cy - half), (cx + half, cy + half),
                     (0, 0, 255), 1)
            cv2.line(vis, (cx - half, cy + half), (cx + half, cy - half),
                     (0, 0, 255), 1)
        # Legend
        cv2.putText(vis, "green=detected  red=reprojected", (8, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        return vis

    reproj_a = cv2.perspectiveTransform(
        model_pts.reshape(-1, 1, 2), H_A_inv).reshape(-1, 2)
    reproj_b = cv2.perspectiveTransform(
        model_pts.reshape(-1, 1, 2), H_B_inv).reshape(-1, 2)

    # ------------------------------------------------------------------
    # 4. Dot detection visualizations
    # ------------------------------------------------------------------
    print("[4/5] Generating dot-detection diagnostics...")
    detection_dir = os.path.join(OUTPUT_DIR, "detections")
    os.makedirs(detection_dir, exist_ok=True)

    for idx, (img_a, img_b) in enumerate(pairs):
        pos = idx + 1
        vis_a = calib_a.visualize(img_a, index=idx, show_axes=True, show_reprojection=True)
        vis_b = calib_b.visualize(img_b, index=idx, show_axes=True, show_reprojection=True)
        cv2.imwrite(os.path.join(detection_dir, f"pos{pos:02d}_camA.png"), vis_a)
        cv2.imwrite(os.path.join(detection_dir, f"pos{pos:02d}_camB.png"), vis_b)

    # Grid overview of detection results
    detection_composite = make_composite(
        [to_bgr(imgs_a[i]) for i in range(len(pairs))]
        + [to_bgr(imgs_b[i]) for i in range(len(pairs))],
        [f"A-{i+1}" for i in range(len(pairs))]
        + [f"B-{i+1}" for i in range(len(pairs))],
        ncols=3, tile_h=300,
    )
    cv2.imwrite(os.path.join(detection_dir, "grid_overview.png"), detection_composite)

    # ------------------------------------------------------------------
    # 5. Undistortion comparison
    # ------------------------------------------------------------------
    print("[4/5] Generating undistortion comparison...")
    undist_dir = os.path.join(OUTPUT_DIR, "undistortion")
    os.makedirs(undist_dir, exist_ok=True)

    for idx, (img_a, img_b) in enumerate(pairs):
        pos = idx + 1
        und_a = calib_a.undistort(img_a)
        und_b = calib_b.undistort(img_b)
        cv2.imwrite(os.path.join(undist_dir, f"pos{pos:02d}_undist_A.png"), und_a)
        cv2.imwrite(os.path.join(undist_dir, f"pos{pos:02d}_undist_B.png"), und_b)

        # Side-by-side comparison
        comp = np.hstack([to_bgr(und_a), to_bgr(und_b)])
        cv2.imwrite(os.path.join(undist_dir, f"pos{pos:02d}_side.png"), comp)


    # Save matched circles for reference pair
    if pts_a is not None and pts_b is not None:
        vis_a = _draw_circles(imgs_a_color[ref_idx], pts_a, reproj_a)
        vis_b = _draw_circles(imgs_b_color[ref_idx], pts_b, reproj_b)
        cv2.imwrite(os.path.join(matched_dir, f"ref_pair{REFERENCE_PAIR:02d}_camA.png"), vis_a)
        cv2.imwrite(os.path.join(matched_dir, f"ref_pair{REFERENCE_PAIR:02d}_camB.png"), vis_b)

        # Side-by-side composite
        h_a, w_a = vis_a.shape[:2]
        h_b, w_b = vis_b.shape[:2]
        h_max = max(h_a, h_b)
        vis_a_pad = cv2.copyMakeBorder(
            vis_a, 0, h_max - h_a, 0, 0, cv2.BORDER_CONSTANT, value=(40, 40, 40))
        vis_b_pad = cv2.copyMakeBorder(
            vis_b, 0, h_max - h_b, 0, 0, cv2.BORDER_CONSTANT, value=(40, 40, 40))
        side_by_side = np.hstack([vis_a_pad, vis_b_pad])
        cv2.imwrite(os.path.join(matched_dir, "side_by_side.png"), side_by_side)
        print(f"  Saved: {matched_dir}/ref_pair{REFERENCE_PAIR:02d}_camA.png")
        print(f"  Saved: {matched_dir}/ref_pair{REFERENCE_PAIR:02d}_camB.png")
        print(f"  Saved: {matched_dir}/side_by_side.png")

    # ------------------------------------------------------------------
    # 6. Stitch all pairs
    # ------------------------------------------------------------------
    print("[5/5] Stitching all image pairs...")
    stitch_dir = os.path.join(OUTPUT_DIR, "stitched")
    os.makedirs(stitch_dir, exist_ok=True)

    stitch_results = []
    for idx, (img_a, img_b) in enumerate(pairs):
        pos = idx + 1
        result = rig.stitch(img_a, img_b, pixel_size_mm=PIXEL_SIZE_MM)
        stitched = result["stitched_image"]
        mask = result["valid_mask"]

        # Save
        cv2.imwrite(os.path.join(stitch_dir, f"pos{pos:02d}_stitched.png"), stitched)
        cv2.imwrite(os.path.join(stitch_dir, f"pos{pos:02d}_mask.png"), mask)

        # Annotated composite: original A + B + stitched
        h_raw = max(img_a.shape[0], img_b.shape[0])
        scale = h_raw / max(stitched.shape[0], 1)
        stitched_disp = cv2.resize(
            to_bgr(stitched),
            (int(stitched.shape[1] * scale), h_raw),
            interpolation=cv2.INTER_AREA,
        )
        annotated = np.hstack([
            to_bgr(img_a),
            to_bgr(img_b),
            stitched_disp,
        ])
        cv2.imwrite(os.path.join(stitch_dir, f"pos{pos:02d}_annotated.png"), annotated)

        stitch_results.append({
            "pair": pos,
            "stitch_h": stitched.shape[0],
            "stitch_w": stitched.shape[1],
            "valid_ratio": float(mask.sum() / mask.size),
        })

    # Stitch overview grid
    stitched_images = [to_bgr(r["stitched_image"])
                       for r in [rig.stitch(a, b, pixel_size_mm=PIXEL_SIZE_MM)
                                 for a, b in pairs]]
    stitch_composite = make_composite(
        stitched_images,
        [f"Pair {i+1}" for i in range(len(pairs))],
        ncols=3, tile_h=300,
    )
    cv2.imwrite(os.path.join(stitch_dir, "all_stitched_overview.png"), stitch_composite)

    for sr in stitch_results:
        print(f"  Pair {sr['pair']:2d}: "
              f"canvas={sr['stitch_h']}×{sr['stitch_w']}, "
              f"valid={sr['valid_ratio']:.1%}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    summary = {
        "image_dir": IMAGE_DIR,
        "num_pairs": len(pairs),
        "grid_size": list(GRID_SIZE),
        "circle_spacing_mm": CIRCLE_SPACING_MM,
        "pixel_size_mm": PIXEL_SIZE_MM,
        "reference_pair": REFERENCE_PAIR,
        "camera_A": {
            "num_images_detected": result_a["num_images"],
            "reprojection_error_px": result_a["reprojection_error"],
            "per_image_errors_px": result_a["per_image_errors"],
            "image_size": result_a["image_size"],
            "K": result_a["K"].tolist() if result_a["K"] is not None else None,
        },
        "camera_B": {
            "num_images_detected": result_b["num_images"],
            "reprojection_error_px": result_b["reprojection_error"],
            "per_image_errors_px": result_b["per_image_errors"],
            "image_size": result_b["image_size"],
            "K": result_b["K"].tolist() if result_b["K"] is not None else None,
        },
        "rig": {
            "reprojection_error_A_px": rig_result["reprojection_error_a"],
            "reprojection_error_B_px": rig_result["reprojection_error_b"],
            "canvas_size": rig_result["canvas_size"],
            "origin_offset_mm": rig_result["origin_offset_mm"],
        },
        "stitch_results": stitch_results,
    }

    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Print final summary
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Camera A — reprojection error: {result_a['reprojection_error']:.4f} px")
    print(f"  K = {np.array2string(result_a['K'], precision=1, suppress_small=True)}")
    print(f"Camera B — reprojection error: {result_b['reprojection_error']:.4f} px")
    print(f"  K = {np.array2string(result_b['K'], precision=1, suppress_small=True)}")
    print(f"Rig — homography errors: "
          f"A={rig_result['reprojection_error_a']:.4f} px, "
          f"B={rig_result['reprojection_error_b']:.4f} px")
    print()
    print("Outputs:")
    print(f"  {OUTPUT_DIR}/")
    for sub in ["detections", "undistortion", "stitched"]:
        count = len(os.listdir(os.path.join(OUTPUT_DIR, sub)))
        print(f"    {sub}/ ({count} files)")
    print(f"    calib_a.npz, calib_b.npz, rig.npz, summary.json")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
