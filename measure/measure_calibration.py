"""
Camera Calibration & Stereo Perspective Stitching Module

Core principle:
1. Dot-grid calibration board -> single camera intrinsics/distortion (Zhang's method)
2. Two cameras observe the same board -> compute homography per camera
3. Undistort + warpPerspective both images onto a shared physical canvas (mm)

Coordinate conventions:
  - Image coordinates:       origin at top-left, row-down / col-right (OpenCV)
  - Board physical coords:   origin at first dot (top-left), X-right / Y-down, mm
  - Canvas pixel coords:     origin = board origin + optional offset, pixel = mm / pixel_size_mm

Usage:
    from measure_calibration import CameraCalibration, StereoRigCalibration

    # === Single camera calibration ===
    calib = CameraCalibration(grid_size=(11, 9), circle_spacing_mm=10.0)
    result = calib.calibrate(images)         # on 10+ images at different poses
    undistorted = calib.undistort(raw_img)   # remove lens distortion
    calib.save("calib.npz")

    # ... later ...
    calib = CameraCalibration.load("calib.npz")

    # === Stereo rig stitching ===
    rig = StereoRigCalibration(calib_a, calib_b)
    rig.calibrate(board_img_a, board_img_b)  # both cameras see the SAME board
    result = rig.stitch(img_a, img_b)         # produce physical-scale panorama
    rig.save("rig.npz")
"""

import json
import math
from dataclasses import dataclass

import numpy as np
import cv2
from typing import Tuple, List, Optional, Dict, Any
from measure.computation_types import (
    Observation,
    Space,
    contract_error,
    invalid,
    observed,
    require_finite,
    require_xy_array,
)
from measure.constants import EPS
from measure.viz import to_bgr, draw_text_shadow


# =========================================================================
# CameraCalibration — single-camera intrinsics + distortion
# =========================================================================

class CameraCalibration:
    """Single-camera intrinsic and distortion calibration via Zhang's method.

    Uses cv2.findCirclesGrid + cv2.calibrateCamera with a dot-grid board.

    Lifecycle:  __init__ → calibrate(images) → save() / load() → undistort(img)
    """

    def __init__(self,
                 grid_size: Tuple[int, int] = (11, 9),
                 circle_spacing_mm: float = 10.0):
        """
        Args:
            grid_size: Dot grid (cols, rows), e.g. (11, 9) = 11 cols × 9 rows.
            circle_spacing_mm: Centre-to-centre distance between adjacent dots (mm).
        """
        self.grid_size = grid_size
        self.circle_spacing_mm = circle_spacing_mm

        # Calibration result (populated by calibrate())
        self.K: Optional[np.ndarray] = None          # 3×3 intrinsic matrix
        self.D: Optional[np.ndarray] = None           # distortion coefficients
        self.rvecs: List[np.ndarray] = []             # per-image rotation vectors
        self.tvecs: List[np.ndarray] = []             # per-image translation vectors
        self.reprojection_error: float = 0.0          # mean reprojection error (px)
        self.per_image_errors: List[float] = []       # per-image errors
        self.image_size: Optional[Tuple[int, int]] = None  # (h, w)
        self.result: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_model_points(self) -> np.ndarray:
        """Build the 3D model points for the dot grid (Z=0 plane)."""
        cols, rows = self.grid_size
        pts = np.zeros((rows * cols, 3), dtype=np.float32)
        for j in range(rows):
            for i in range(cols):
                pts[j * cols + i] = (i * self.circle_spacing_mm,
                                     j * self.circle_spacing_mm,
                                     0.0)
        return pts

    @staticmethod
    def _make_blob_detector(params: Optional[Dict[str, Any]] = None) -> cv2.SimpleBlobDetector:
        """Create a SimpleBlobDetector tuned for dot-grid circles."""
        defaults = {
            "minThreshold": 1,
            "maxThreshold": 255,
            "thresholdStep": 5,
            "minArea": 5,
            "maxArea": 2000,
            "minCircularity": 0.3,
            "minConvexity": 0.3,
            "filterByInertia": False,   # not used — dot grids may have elliptical blobs
        }
        if params:
            defaults.update(params)

        p = cv2.SimpleBlobDetector_Params()
        p.minThreshold = defaults["minThreshold"]
        p.maxThreshold = defaults["maxThreshold"]
        p.thresholdStep = defaults["thresholdStep"]
        p.filterByArea = True
        p.minArea = defaults["minArea"]
        p.maxArea = defaults["maxArea"]
        p.filterByCircularity = True
        p.minCircularity = defaults["minCircularity"]
        p.filterByConvexity = True
        p.minConvexity = defaults["minConvexity"]
        p.filterByInertia = defaults.get("filterByInertia", False)
        if p.filterByInertia:
            p.minInertiaRatio = defaults.get("minInertiaRatio", 0.3)

        return cv2.SimpleBlobDetector_create(p)

    # ------------------------------------------------------------------
    # calibrate
    # ------------------------------------------------------------------

    def calibrate(self,
                  images: List[np.ndarray],
                  symmetric_grid: bool = True,
                  blob_detector_params: Optional[Dict[str, Any]] = None,
                  subpix_refine: bool = True) -> Dict[str, Any]:
        """Run Zhang calibration on a set of board images at different poses.

        Args:
            images: Grayscale board images (≥ 10 poses recommended).
            symmetric_grid: True for symmetric dot grid, False for asymmetric.
            blob_detector_params: Optional overrides for SimpleBlobDetector.
            subpix_refine: Enable cv2.cornerSubPix sub-pixel refinement.

        Returns:
            Dict with keys 'K', 'D', 'rvecs', 'tvecs',
            'reprojection_error', 'per_image_errors', 'num_images',
            'image_size'.

        Raises:
            ValueError: If fewer than 3 images have detectable grids.
        """
        model_points = self._build_model_points()

        # cv2.findCirclesGrid expects (cols, rows) and the flags
        flags = (cv2.CALIB_CB_SYMMETRIC_GRID if symmetric_grid
                 else cv2.CALIB_CB_ASYMMETRIC_GRID)

        detector = self._make_blob_detector(blob_detector_params)

        obj_points: List[np.ndarray] = []   # 3D model points
        img_points: List[np.ndarray] = []   # 2D image points
        detected_count = 0

        for img in images:
            h, w = img.shape[:2]
            found, centers = cv2.findCirclesGrid(
                img, self.grid_size, flags=flags, blobDetector=detector
            )
            if not found:
                print(f"Warning: findCirclesGrid failed on image "
                      f"{detected_count} (size={w}x{h})")
                continue

            if subpix_refine:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                            30, 0.001)
                cv2.cornerSubPix(img, centers, (5, 5), (-1, -1), criteria)

            obj_points.append(model_points)
            img_points.append(centers.reshape(-1, 2))
            detected_count += 1

        if detected_count < 3:
            raise ValueError(
                f"Need at least 3 images with detected grids, got {detected_count}"
            )

        # Run Zhang calibration
        ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, (w, h), None, None
        )

        # Per-image and mean reprojection errors
        total_error = 0.0
        per_image_errors: List[float] = []
        for i in range(len(obj_points)):
            projected, _ = cv2.projectPoints(
                obj_points[i], rvecs[i], tvecs[i], K, D
            )
            err = np.linalg.norm(img_points[i] - projected.reshape(-1, 2), axis=1)
            mean_err = float(np.mean(err))
            per_image_errors.append(mean_err)
            total_error += mean_err * len(obj_points[i])

        self.K = K
        self.D = D
        self.rvecs = rvecs
        self.tvecs = tvecs
        self.reprojection_error = float(total_error / sum(len(p) for p in obj_points))
        self.per_image_errors = per_image_errors
        self.image_size = (h, w)

        self.result = {
            "K": K,
            "D": D,
            "rvecs": rvecs,
            "tvecs": tvecs,
            "reprojection_error": self.reprojection_error,
            "per_image_errors": per_image_errors,
            "num_images": detected_count,
            "image_size": self.image_size,
        }
        return self.result

    # ------------------------------------------------------------------
    # undistort
    # ------------------------------------------------------------------

    def undistort(self, image: np.ndarray,
                  alpha: float = 0.0) -> np.ndarray:
        """Remove lens distortion from *image*.

        Args:
            image: Grayscale or BGR image.
            alpha: 0 = crop black borders, 1 = keep all pixels (may have borders).

        Returns:
            Undistorted image (same dtype / channels as input).

        Raises:
            RuntimeError: If calibrate() has not been called.
        """
        if self.K is None or self.D is None:
            raise RuntimeError("Camera not calibrated. Call calibrate() first.")

        h, w = image.shape[:2]
        new_K, _ = cv2.getOptimalNewCameraMatrix(
            self.K, self.D, (w, h), alpha
        )
        return cv2.undistort(image, self.K, self.D, None, new_K)

    # ------------------------------------------------------------------
    # visualize
    # ------------------------------------------------------------------

    def visualize(self, image: np.ndarray,
                  save_path: Optional[str] = None,
                  show_reprojection: bool = True,
                  show_axes: bool = True,
                  index: int = 0) -> np.ndarray:
        """Draw detected dots + optional reprojection / axes on a board image.

        Args:
            image: Grayscale board image.
            save_path: If not None, write the annotated image to this path.
            show_reprojection: Draw reprojected dot centres (red circles).
            show_axes: Draw coordinate axes via cv2.drawFrameAxes.
            index: Which rvec/tvec to use for projection (default 0).

        Returns:
            Annotated BGR image.
        """
        vis = to_bgr(image)
        h, w = image.shape[:2]

        # Detect circles
        detector = self._make_blob_detector()
        found, centers = cv2.findCirclesGrid(
            image, self.grid_size, cv2.CALIB_CB_SYMMETRIC_GRID, blobDetector=detector
        )

        if found:
            # Draw detected centres
            for ci, (cx, cy) in enumerate(centers.reshape(-1, 2)):
                cv2.circle(vis, (int(cx), int(cy)), 3, (0, 255, 0), -1)
                if self.grid_size[0] <= 15:  # only label when not too dense
                    cv2.putText(vis, str(ci), (int(cx) + 5, int(cy) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

            # Reprojection
            if show_reprojection and self.rvecs and index < len(self.rvecs):
                model_pts = self._build_model_points()
                projected, _ = cv2.projectPoints(
                    model_pts, self.rvecs[index], self.tvecs[index],
                    self.K, self.D
                )
                for (px, py) in projected.reshape(-1, 2):
                    cv2.circle(vis, (int(px), int(py)), 2, (0, 0, 255), -1)

                if self.per_image_errors and index < len(self.per_image_errors):
                    text = f"Reprojection error: {self.per_image_errors[index]:.3f} px"
                    draw_text_shadow(vis, text, (10, 25),
                                     color=(255, 255, 255), font_scale=0.5, thickness=1)

            # Axes
            if show_axes and self.K is not None and self.rvecs and index < len(self.rvecs):
                cv2.drawFrameAxes(vis, self.K, self.D,
                                  self.rvecs[index], self.tvecs[index],
                                  self.circle_spacing_mm * 2, thickness=2)

        draw_text_shadow(vis, f"Board #{index}  "
                              f"{'DETECTED' if found else 'NOT FOUND'}"
                              f"  ({self.grid_size[0]}x{self.grid_size[1]})",
                         (10, h - 15), color=(255, 255, 255),
                         font_scale=0.5, thickness=1)

        if save_path:
            cv2.imwrite(save_path, vis)

        return vis

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize calibration config + results to a JSON-compatible dict."""
        if self.K is None:
            return {
                "grid_size": list(self.grid_size),
                "circle_spacing_mm": self.circle_spacing_mm,
            }
        return {
            "grid_size": list(self.grid_size),
            "circle_spacing_mm": self.circle_spacing_mm,
            "K": self.K.tolist(),
            "D": self.D.tolist() if self.D is not None else None,
            "reprojection_error": self.reprojection_error,
            "per_image_errors": self.per_image_errors,
            "image_size": list(self.image_size) if self.image_size else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CameraCalibration":
        """Reconstruct a CameraCalibration from a dict."""
        obj = cls(
            grid_size=tuple(data["grid_size"]),
            circle_spacing_mm=data["circle_spacing_mm"],
        )
        if "K" in data and data["K"] is not None:
            obj.K = np.array(data["K"], dtype=np.float64)
            obj.D = (np.array(data["D"], dtype=np.float64)
                     if data["D"] is not None else None)
            obj.reprojection_error = data.get("reprojection_error", 0.0)
            obj.per_image_errors = data.get("per_image_errors", [])
            obj.image_size = (tuple(data["image_size"])
                              if data.get("image_size") else None)
        return obj

    def save(self, filepath: str) -> None:
        """Persist calibration to a .npz file."""
        data = self.to_dict()
        # Store K & D as raw arrays for precision, rest as JSON metadata
        save_kwargs = {
            "metadata_json": json.dumps(data),
        }
        if self.K is not None:
            save_kwargs["K"] = self.K
            save_kwargs["D"] = self.D
        np.savez_compressed(filepath, **save_kwargs)

    @classmethod
    def load(cls, filepath: str) -> "CameraCalibration":
        """Load a CameraCalibration from a .npz file."""
        packed = np.load(filepath, allow_pickle=False)
        if "metadata_json" in packed:
            data = json.loads(str(packed["metadata_json"]))
        else:
            # Legacy: data stored as flat arrays
            data = {
                "grid_size": packed["grid_size"].tolist(),
                "circle_spacing_mm": float(packed["circle_spacing_mm"]),
                "K": packed["K"].tolist() if "K" in packed else None,
                "D": packed["D"].tolist() if "D" in packed else None,
            }
        obj = cls.from_dict(data)
        # Overwrite K/D from array keys for precision
        if "K" in packed:
            obj.K = packed["K"]
            obj.D = packed["D"]
        return obj


# =========================================================================
# StereoRigCalibration — two-camera perspective stitching
# =========================================================================

class StereoRigCalibration:
    """Two-camera panoramic stitching via homography.

    Each camera's image is undistorted and then warped onto a shared
    physical coordinate canvas using a board-plane homography.

    Lifecycle:
        __init__(calib_a, calib_b) → calibrate(board_img_a, board_img_b)
        → (optional) set_origin(board_img, click_row, click_col)
        → save() / load() → stitch(img_a, img_b)
    """

    def __init__(self,
                 calib_a: CameraCalibration,
                 calib_b: CameraCalibration):
        """
        Args:
            calib_a: Calibrated camera A.
            calib_b: Calibrated camera B.
        """
        self.calib_a = calib_a
        self.calib_b = calib_b

        # Homographies (image → board plane, 3×3)
        self.H_A: Optional[np.ndarray] = None
        self.H_B: Optional[np.ndarray] = None
        # Stereo calibration: relative pose camera B -> camera A
        self.R: Optional[np.ndarray] = None       # 3x3 rotation B->A
        self.T: Optional[np.ndarray] = None       # 3x1 translation B->A
        self.E: Optional[np.ndarray] = None       # 3x3 essential matrix
        self.F: Optional[np.ndarray] = None       # 3x3 fundamental matrix
        self.stereo_rms: float = 0.0              # stereo calibration RMS error

        # Canvas transform (physical-mm → canvas-pixels, 2×3 affine)
        self.canvas_affine: Optional[np.ndarray] = None
        self.canvas_size: Optional[Tuple[int, int]] = None

        # Origin offset
        self.origin_offset_mm: Tuple[float, float] = (0.0, 0.0)

        # Control points for board detection
        self.board_points_a: Optional[np.ndarray] = None  # (N, 2) physical coords
        self.board_points_b: Optional[np.ndarray] = None

        # Stitching result
        self.result: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # calibrate — compute homographies for both cameras
    # ------------------------------------------------------------------

    def calibrate(self,
                  board_img_a: np.ndarray,
                  board_img_b: np.ndarray,
                  symmetric_grid: bool = True,
                  blob_detector_params: Optional[Dict[str, Any]] = None
                  ) -> Dict[str, Any]:
        """Compute homography per camera from a single shared board image pair.

        Both cameras must see the *same* board at the *same* time.
        The board defines the physical coordinate system.

        Args:
            board_img_a: Camera A's view of the board (grayscale).
            board_img_b: Camera B's view of the board (grayscale).
            symmetric_grid: True for symmetric dot grid.
            blob_detector_params: Optional blob detector overrides.

        Returns:
            Dict with 'H_A', 'H_B', 'num_points_a', 'num_points_b',
            'reprojection_error_a', 'reprojection_error_b',
            'canvas_size', 'pixel_size_mm', 'canvas_affine'.

        Raises:
            ValueError: If dot detection fails on either image.
        """
        cols, rows = self.calib_a.grid_size
        spacing = self.calib_a.circle_spacing_mm

        flags = (cv2.CALIB_CB_SYMMETRIC_GRID if symmetric_grid
                 else cv2.CALIB_CB_ASYMMETRIC_GRID)
        detector = CameraCalibration._make_blob_detector(blob_detector_params)

        # Build physical model points (Z=0 plane)
        model_pts = np.zeros((rows * cols, 2), dtype=np.float32)
        for j in range(rows):
            for i in range(cols):
                model_pts[j * cols + i] = (i * spacing, j * spacing)

        def _detect(img: np.ndarray) -> Optional[np.ndarray]:
            found, centers = cv2.findCirclesGrid(
                img, (cols, rows), flags=flags, blobDetector=detector
            )
            if not found:
                return None
            return centers.reshape(-1, 2).astype(np.float32)

        pts_a = _detect(board_img_a)
        pts_b = _detect(board_img_b)

        if pts_a is None:
            raise ValueError("Dot detection failed on camera A board image")
        if pts_b is None:
            raise ValueError("Dot detection failed on camera B board image")

        # Compute homographies
        self.H_A, mask_a = cv2.findHomography(pts_a, model_pts, method=0)
        self.H_B, mask_b = cv2.findHomography(pts_b, model_pts, method=0)

        # Reprojection errors
        def _reproj_error(H, src, dst):
            projected = cv2.perspectiveTransform(
                src.reshape(-1, 1, 2), H
            ).reshape(-1, 2)
            return float(np.mean(np.linalg.norm(projected - dst, axis=1)))

        err_a = _reproj_error(self.H_A, pts_a, model_pts)
        err_b = _reproj_error(self.H_B, pts_b, model_pts)

        self.board_points_a = model_pts.copy()
        self.board_points_b = model_pts.copy()

        # Compute canvas bounds (union of both camera projections onto board plane)
        self._compute_canvas(board_img_a, board_img_b)

        self.result = {
            "H_A": self.H_A,
            "H_B": self.H_B,
            "num_points_a": len(pts_a),
            "num_points_b": len(pts_b),
            "reprojection_error_a": err_a,
            "reprojection_error_b": err_b,
            "canvas_size": self.canvas_size,
            "canvas_affine": self.canvas_affine,
            "origin_offset_mm": self.origin_offset_mm,
        }
        return self.result

    # ------------------------------------------------------------------
    # _compute_canvas
    # ------------------------------------------------------------------

    def _compute_canvas(self,
                        img_a: np.ndarray,
                        img_b: np.ndarray,
                        pixel_size_mm: float = 0.05):
        """Determine canvas size + affine transform mapping board-mm → canvas-pixels."""
        def _project_corners(img: np.ndarray, H: np.ndarray) -> np.ndarray:
            h, w = img.shape[:2]
            corners = np.array([[0, 0], [w - 1, 0],
                                [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
            return cv2.perspectiveTransform(
                corners.reshape(-1, 1, 2), H
            ).reshape(-1, 2)

        pts_a = _project_corners(img_a, self.H_A)
        pts_b = _project_corners(img_b, self.H_B)
        all_pts = np.vstack([pts_a, pts_b])

        x_min = float(all_pts[:, 0].min())
        x_max = float(all_pts[:, 0].max())
        y_min = float(all_pts[:, 1].min())
        y_max = float(all_pts[:, 1].max())

        # Apply origin offset (shifts the board origin relative to canvas)
        dx_mm, dy_mm = self.origin_offset_mm
        x_min += dx_mm
        x_max += dx_mm
        y_min += dy_mm
        y_max += dy_mm

        canvas_w = int(np.ceil((x_max - x_min) / pixel_size_mm)) + 1
        canvas_h = int(np.ceil((y_max - y_min) / pixel_size_mm)) + 1
        self.canvas_size = (canvas_h, canvas_w)

        # Board-mm → canvas-pixel affine:  pxl_x = (mm_x - x_min) / pixel_size_mm
        self.canvas_affine = np.array([
            [1.0 / pixel_size_mm, 0.0,                -x_min / pixel_size_mm],
            [0.0,                 1.0 / pixel_size_mm, -y_min / pixel_size_mm],
        ], dtype=np.float64)

    # ------------------------------------------------------------------
    # set_origin
    # ------------------------------------------------------------------

    def set_origin(self,
                   board_img: np.ndarray,
                   click_row: float,
                   click_col: float,
                   camera: str = "A") -> Dict[str, Any]:
        """Set a custom physical origin by clicking a dot on the board image.

        The dot nearest to the click is found, and its physical offset
        is saved so subsequent stitch() calls use it as the canvas origin.

        Args:
            board_img: Grayscale board image from camera 'A' or 'B'.
            click_row, click_col: Pixel coordinates of the user's click.
            camera: Which camera the image is from ('A' or 'B').

        Returns:
            Dict with 'origin_point', 'origin_grid_index',
            'origin_physical_offset_mm'.
        """
        if camera not in ("A", "B"):
            raise ValueError(f"camera must be 'A' or 'B', got '{camera}'")

        calib = self.calib_a if camera == "A" else self.calib_b
        cols, rows = calib.grid_size
        spacing = calib.circle_spacing_mm

        # Detect dots
        detector = CameraCalibration._make_blob_detector()
        found, centers = cv2.findCirclesGrid(
            board_img, (cols, rows),
            cv2.CALIB_CB_SYMMETRIC_GRID, blobDetector=detector
        )
        if not found:
            raise ValueError("Dot detection failed — cannot set origin")

        centers = centers.reshape(-1, 2)
        dists = np.linalg.norm(centers - np.array([[click_col, click_row]]), axis=1)
        nearest_idx = int(np.argmin(dists))
        gi = nearest_idx % cols
        gj = nearest_idx // cols

        dx_mm = -gi * spacing
        dy_mm = -gj * spacing
        self.origin_offset_mm = (dx_mm, dy_mm)

        return {
            "origin_point": (float(centers[nearest_idx][1]),
                             float(centers[nearest_idx][0])),
            "origin_grid_index": (gj, gi),
            "origin_physical_offset_mm": (dx_mm, dy_mm),
        }

    # ------------------------------------------------------------------
    # stitch
    # ------------------------------------------------------------------
    # calibrate_stereo -- compute relative pose (R, T) between cameras
    # ------------------------------------------------------------------

    def _build_model_points_3d(self) -> np.ndarray:
        """Build 3D model points for the dot grid (N, 3), Z=0 plane."""
        cols, rows = self.calib_a.grid_size
        spacing = self.calib_a.circle_spacing_mm
        pts = np.zeros((rows * cols, 3), dtype=np.float32)
        for j in range(rows):
            for i in range(cols):
                pts[j * cols + i] = (i * spacing, j * spacing, 0.0)
        return pts

    def _detect_board_centers(self,
                              img: np.ndarray,
                              camera: str,
                              blob_params: Optional[Dict[str, Any]] = None,
                              ) -> Optional[np.ndarray]:
        """Detect dot-grid centres on a single board image."""
        calib = self.calib_a if camera == "A" else self.calib_b
        cols, rows = calib.grid_size
        flags = cv2.CALIB_CB_SYMMETRIC_GRID
        detector = CameraCalibration._make_blob_detector(blob_params)
        found, centers = cv2.findCirclesGrid(
            img, (cols, rows), flags=flags, blobDetector=detector,
        )
        if not found:
            return None
        # Sub-pixel refinement
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        cv2.cornerSubPix(img, centers, (5, 5), (-1, -1), criteria)
        return centers.reshape(-1, 2).astype(np.float32)

    def _solve_board_pose(self,
                          img: np.ndarray,
                          camera: str,
                          blob_params: Optional[Dict[str, Any]] = None,
                          ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """SolvePnP to get board pose (rvec, tvec) relative to camera."""
        calib = self.calib_a if camera == "A" else self.calib_b
        centers = self._detect_board_centers(img, camera, blob_params=blob_params)
        if centers is None:
            return None
        model_pts = self._build_model_points_3d()
        ret, rvec, tvec = cv2.solvePnP(
            model_pts, centers, calib.K, calib.D,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ret:
            return None
        return rvec, tvec

    def calibrate_stereo(self,
                         image_pairs: list,
                         fix_intrinsics: bool = True,
                         symmetric_grid: bool = True,
                         blob_detector_params: Optional[Dict[str, Any]] = None,
                         ) -> Dict[str, Any]:
        """Run full stereo calibration using multiple image pairs.

        Uses cv2.stereoCalibrate() with known per-camera intrinsics (K, D).
        Computes the relative pose R, T between camera B and camera A,
        as well as the essential (E) and fundamental (F) matrices.
        """
        calib_a = self.calib_a
        calib_b = self.calib_b
        cols, rows = calib_a.grid_size

        if calib_a.K is None or calib_b.K is None:
            raise RuntimeError("Both cameras must be individually calibrated first.")

        flags_detect = (cv2.CALIB_CB_SYMMETRIC_GRID if symmetric_grid
                        else cv2.CALIB_CB_ASYMMETRIC_GRID)
        detector = CameraCalibration._make_blob_detector(blob_detector_params)

        # Build model points once
        model_points = self._build_model_points_3d()

        obj_points: List[np.ndarray] = []    # 3D model points per pair
        img_points_a: List[np.ndarray] = []  # 2D image points camera A
        img_points_b: List[np.ndarray] = []  # 2D image points camera B

        for idx, (img_a, img_b) in enumerate(image_pairs):
            h, w = img_a.shape[:2]

            # Detect in camera A
            found_a, centers_a = cv2.findCirclesGrid(
                img_a, (cols, rows), flags=flags_detect, blobDetector=detector,
            )
            if not found_a:
                print(f"  Pair {idx + 1}: Camera A detection failed, skipping")
                continue
            # Sub-pixel refinement
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            cv2.cornerSubPix(img_a, centers_a, (5, 5), (-1, -1), criteria)

            # Detect in camera B
            found_b, centers_b = cv2.findCirclesGrid(
                img_b, (cols, rows), flags=flags_detect, blobDetector=detector,
            )
            if not found_b:
                print(f"  Pair {idx + 1}: Camera B detection failed, skipping")
                continue
            cv2.cornerSubPix(img_b, centers_b, (5, 5), (-1, -1), criteria)

            obj_points.append(model_points)
            img_points_a.append(centers_a.reshape(-1, 2).astype(np.float32))
            img_points_b.append(centers_b.reshape(-1, 2).astype(np.float32))

        if len(obj_points) < 3:
            raise ValueError(
                f"Need at least 3 pairs with detected boards, got {len(obj_points)}"
            )

        print(f"  Stereo calibration on {len(obj_points)} pairs, "
              f"image size {w}x{h}")

        # cv2.stereoCalibrate flags
        flags_stereo = cv2.CALIB_FIX_INTRINSIC if fix_intrinsics else 0

        stereo_result = cv2.stereoCalibrate(
            obj_points, img_points_a, img_points_b,
            calib_a.K, calib_a.D,
            calib_b.K, calib_b.D,
            (w, h),
            flags=flags_stereo,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
        )
        # Handle both 9- and 10-value return (OpenCV version dependent)
        if len(stereo_result) == 10:
            ret, K1, D1, K2, D2, R, T, E, F, per_view_errors = stereo_result
        else:
            ret, K1, D1, K2, D2, R, T, E, F = stereo_result
            per_view_errors = None

        self.R = R
        self.T = T
        self.E = E
        self.F = F
        self.stereo_rms = float(ret)

        # Optionally update K/D with refined values
        if not fix_intrinsics:
            calib_a.K = K1
            calib_a.D = D1
            calib_b.K = K2
            calib_b.D = D2

        # Per-pair stereo reprojection errors via solvePnP
        per_pair_errors: List[float] = []
        for i in range(len(obj_points)):
            # Camera A: board -> camera A via solvePnP
            ret_a, rv_a, tv_a = cv2.solvePnP(
                obj_points[i], img_points_a[i], K1, D1,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            p_a, _ = cv2.projectPoints(obj_points[i], rv_a, tv_a, K1, D1)
            err_a = np.mean(np.linalg.norm(
                img_points_a[i] - p_a.reshape(-1, 2), axis=1))
            # Camera B: board -> camera B via solvePnP
            ret_b, rv_b, tv_b = cv2.solvePnP(
                obj_points[i], img_points_b[i], K2, D2,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            p_b, _ = cv2.projectPoints(obj_points[i], rv_b, tv_b, K2, D2)
            err_b = np.mean(np.linalg.norm(
                img_points_b[i] - p_b.reshape(-1, 2), axis=1))
            per_pair_errors.append(float((err_a + err_b) / 2.0))

        self.result = {
            "R": self.R,
            "T": self.T,
            "E": self.E,
            "F": self.F,
            "stereo_rms": self.stereo_rms,
            "num_pairs": len(obj_points),
            "fix_intrinsics": fix_intrinsics,
            "per_pair_errors": per_pair_errors,
        }
        return self.result

    # ------------------------------------------------------------------
    # stereo_project_points -- project points between cameras via R,T
    # ------------------------------------------------------------------

    def stereo_transform_to_camera_A(self,
                                      pts_3d_in_B: np.ndarray,
                                      ) -> np.ndarray:
        """Transform 3D points from camera B's frame to camera A's frame.

        NOTE: OpenCV stereoCalibrate returns R,T as cam1(A)->cam2(B).
        X_B = R * X_A + T, so X_A = R^T * X_B - R^T * T
        """
        if self.R is None or self.T is None:
            raise RuntimeError("Stereo calibration not done.")
        return (self.R.T @ pts_3d_in_B.T - self.R.T @ self.T).T

    def project_to_camera_image(self,
                                pts_3d: np.ndarray,
                                camera: str,
                                rvec: Optional[np.ndarray] = None,
                                tvec: Optional[np.ndarray] = None,
                                ) -> np.ndarray:
        """Project 3D points onto a camera's image plane.

        Args:
            pts_3d: (N, 3) points. If rvec/tvec are None, points are in
                    camera coordinates (identity pose).
            camera: 'A' or 'B'.
        Returns:
            (N, 2) projected image coordinates.
        """
        calib = self.calib_a if camera == "A" else self.calib_b
        if rvec is None:
            rvec = np.zeros((3, 1), dtype=np.float64)
        if tvec is None:
            tvec = np.zeros((3, 1), dtype=np.float64)
        projected, _ = cv2.projectPoints(
            pts_3d.astype(np.float64), rvec, tvec, calib.K, calib.D,
        )
        return projected.reshape(-1, 2)

    def verify_stereo_on_board(self,
                                img_a: np.ndarray,
                                img_b: np.ndarray,
                                ) -> Dict[str, Any]:
        """Verify stereo calibration quality on a single pair of board images.

        For an image pair showing the SAME board:
        1. Detect board in both cameras -> get (rvec, tvec) per camera
        2. Camera B dots: transform via R,T -> camera A coords -> project to A's image
        3. Camera A dots: transform via R^T,-R^T*T -> camera B coords -> project to B's image
        4. Compare projected vs. detected centres

        Returns dict with error metrics and projected image coordinates.
        """
        if self.R is None or self.T is None:
            raise RuntimeError("Stereo calibration not done. Call calibrate_stereo() first.")

        model_pts = self._build_model_points_3d()

        # Detect board centres (use tuned blob params for dot-grid board)
        bp = {"minArea": 200, "maxArea": 20000, "minCircularity": 0.3, "minConvexity": 0.3}
        pts_a = self._detect_board_centers(img_a, "A", blob_params=bp)
        pts_b = self._detect_board_centers(img_b, "B", blob_params=bp)
        if pts_a is None or pts_b is None:
            return {"error": "Board detection failed"}

        calib_a, calib_b = self.calib_a, self.calib_b

        # solvePnP for each camera
        ret_a, rvec_a, tvec_a = cv2.solvePnP(
            model_pts, pts_a, calib_a.K, calib_a.D, flags=cv2.SOLVEPNP_ITERATIVE,
        )
        ret_b, rvec_b, tvec_b = cv2.solvePnP(
            model_pts, pts_b, calib_b.K, calib_b.D, flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ret_a or not ret_b:
            return {"error": "solvePnP failed"}

        R_a, _ = cv2.Rodrigues(rvec_a)
        R_b, _ = cv2.Rodrigues(rvec_b)

        # Camera B's dots in camera B frame
        pts_3d_in_B = (R_b @ model_pts.T + tvec_b).T  # (N, 3)
        # Transform to camera A frame via stereo R,T
        # NOTE: OpenCV stereoCalibrate returns R,T as cam1(A)->cam2(B).
        #   X_B = R * X_A + T, so X_A = R^T * X_B - R^T * T
        pts_3d_in_A_from_B = (self.R.T @ pts_3d_in_B.T - self.R.T @ self.T).T  # (N, 3)
        # Project to camera A's image
        proj_a_from_b = self.project_to_camera_image(pts_3d_in_A_from_B, "A")

        # Camera A's dots in camera A frame, projected (should match detection)
        pts_3d_in_A = (R_a @ model_pts.T + tvec_a).T
        proj_a = self.project_to_camera_image(pts_3d_in_A, "A")

        # Camera A -> camera B (forward stereo transform)
        # X_B = R * X_A + T
        pts_3d_in_B_from_A = (self.R @ pts_3d_in_A.T + self.T).T
        proj_b_from_a = self.project_to_camera_image(pts_3d_in_B_from_A, "B")

        # Camera B's own board dots reprojected
        proj_b = self.project_to_camera_image(pts_3d_in_B, "B")

        # Errors
        diff_a = np.linalg.norm(proj_a_from_b - pts_a, axis=1)
        diff_b = np.linalg.norm(proj_b_from_a - pts_b, axis=1)
        all_diffs = np.concatenate([diff_a, diff_b])

        return {
            "num_points": len(pts_a),
            "mean_error_px": float(np.mean(all_diffs)),
            "median_error_px": float(np.median(all_diffs)),
            "max_error_px": float(np.max(all_diffs)),
            "std_error_px": float(np.std(all_diffs)),
            "diff_per_dot_a": diff_a.tolist(),
            "diff_per_dot_b": diff_b.tolist(),
            "pts_a_detected": pts_a.tolist(),
            "pts_b_detected": pts_b.tolist(),
            "pts_a_from_b_projected": proj_a_from_b.tolist(),
            "pts_b_from_a_projected": proj_b_from_a.tolist(),
            "rvec_a": rvec_a.tolist(),
            "tvec_a": tvec_a.tolist(),
            "rvec_b": rvec_b.tolist(),
            "tvec_b": tvec_b.tolist(),
        }
    # ------------------------------------------------------------------

    def stitch(self,
               img_a: np.ndarray,
               img_b: np.ndarray,
               pixel_size_mm: float = 0.05,
               blend_width: int = 0) -> Dict[str, Any]:
        """Undistort and warp both images onto the shared physical canvas.

        Args:
            img_a, img_b: Grayscale input images from the two cameras.
            pixel_size_mm: Physical size of each output pixel (mm/px).
                           Default 0.05 = 50 μm/px.
            blend_width: Overlap blend width in pixels. 0 = no blending
                         (last-writer-wins).

        Returns:
            Dict with 'stitched_image', 'valid_mask', 'pixel_size_mm',
            'canvas_offset_mm', 'canvas_bounds_mm'.

        Raises:
            RuntimeError: If calibrate() has not been called.
        """
        if self.H_A is None or self.H_B is None:
            raise RuntimeError("Rig not calibrated. Call calibrate() first.")

        # (Re)compute canvas now that pixel_size_mm is known
        self._compute_canvas(img_a, img_b, pixel_size_mm)

        canvas_h, canvas_w = self.canvas_size
        M = self.canvas_affine  # 2×3: board-mm → canvas pixels

        # Per-camera composite warp: undistort → homography → canvas-affine
        # Build 3×3 canvas transform
        M_full = np.vstack([M, [[0, 0, 1]]])  # 3×3

        def _warp(img: np.ndarray, calib: CameraCalibration,
                  H: np.ndarray, dst_size: Tuple[int, int]) -> np.ndarray:
            """Undistort, then adjust homography for the rectified intrinsics,
            then warp to the canvas."""
            h, w = img.shape[:2]
            new_K, _ = cv2.getOptimalNewCameraMatrix(
                calib.K, calib.D, (w, h), 0.0
            )
            # Rectify lens distortion
            map1, map2 = cv2.initUndistortRectifyMap(
                calib.K, calib.D, None, new_K, (w, h), cv2.CV_32FC1
            )
            rectified = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
            # Adjust H for the rectified intrinsics: H' = new_K @ K⁻¹ @ H
            # (so H' maps rectified-image → board-plane)
            H_adj = new_K @ np.linalg.inv(calib.K) @ H
            # Composite: canvas = M_full @ H_adj
            T = M_full @ H_adj
            warped = cv2.warpPerspective(
                rectified, T, (dst_size[1], dst_size[0]),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            return warped

        warped_a = _warp(img_a, self.calib_a, self.H_A, (canvas_h, canvas_w))
        warped_b = _warp(img_b, self.calib_b, self.H_B, (canvas_h, canvas_w))

        # Stitch via simple compositing
        stitched = np.zeros_like(warped_a)
        mask_a = (warped_a > 0).astype(np.uint8)
        mask_b = (warped_b > 0).astype(np.uint8)
        overlap = mask_a & mask_b

        stitched[mask_a > 0] = warped_a[mask_a > 0]
        stitched[mask_b > 0] = warped_b[mask_b > 0]

        if blend_width > 0 and overlap.any():
            alpha = np.where(mask_a & mask_b, 0.5, 1.0).astype(np.float32)
            blended = (warped_a.astype(np.float32) * alpha +
                       warped_b.astype(np.float32) * (1.0 - alpha))
            stitched = blended.clip(0, 255).astype(np.uint8)

        valid_mask = (mask_a | mask_b) * 255

        self.result = {
            "stitched_image": stitched,
            "valid_mask": valid_mask,
            "pixel_size_mm": pixel_size_mm,
            "canvas_offset_mm": self.origin_offset_mm,
            "canvas_bounds_mm": {
                "x_min": -self.canvas_affine[0, 2] * pixel_size_mm,
                "x_max": (-self.canvas_affine[0, 2] + canvas_w) * pixel_size_mm,
                "y_min": -self.canvas_affine[1, 2] * pixel_size_mm,
                "y_max": (-self.canvas_affine[1, 2] + canvas_h) * pixel_size_mm,
            },
        }
        return self.result

    # ------------------------------------------------------------------
    # visualize
    # ------------------------------------------------------------------

    def visualize(self,
                  img_a: np.ndarray,
                  img_b: np.ndarray,
                  save_dir: Optional[str] = None) -> Dict[str, np.ndarray]:
        """Generate diagnostic images for the full calibration + stitch pipeline.

        Args:
            img_a, img_b: Grayscale input images.
            save_dir: If not None, write images to this directory.

        Returns:
            Dict mapping name → annotated BGR image.
        """
        out: Dict[str, np.ndarray] = {}

        # Board detection
        out["board_a"] = self.calib_a.visualize(img_a)
        out["board_b"] = self.calib_b.visualize(img_b)

        # Undistorted
        out["undistorted_a"] = to_bgr(self.calib_a.undistort(img_a))
        out["undistorted_b"] = to_bgr(self.calib_b.undistort(img_b))

        # Stitched panorama
        if self.H_A is not None:
            result = self.stitch(img_a, img_b)
            out["stitched"] = to_bgr(result["stitched_image"])

            # Overlay canvas mask
            mask_overlay = cv2.cvtColor(
                result["stitched_image"], cv2.COLOR_GRAY2BGR
            )
            mask_layer = np.zeros_like(mask_overlay)
            mask_layer[:, :, 2] = result["valid_mask"] // 2  # red tint on valid
            out["stitched_mask"] = cv2.addWeighted(mask_overlay, 0.7, mask_layer, 0.3, 0)

        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            for name, img in out.items():
                cv2.imwrite(os.path.join(save_dir, f"{name}.png"), img)

        return out

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize rig config + results to a JSON-compatible dict."""
        data: Dict[str, Any] = {
            "calib_a": self.calib_a.to_dict(),
            "calib_b": self.calib_b.to_dict(),
            "origin_offset_mm": list(self.origin_offset_mm),
        }
        if self.R is not None:
            data["R"] = self.R.tolist()
            data["T"] = self.T.tolist()
            data["E"] = self.E.tolist() if self.E is not None else None
            data["F"] = (self.F.tolist()
                         if isinstance(self.F, np.ndarray) else None)
            data["stereo_rms"] = self.stereo_rms
        if self.H_A is not None:
            data["H_A"] = self.H_A.tolist()
            data["H_B"] = self.H_B.tolist()
            data["canvas_size"] = list(self.canvas_size) if self.canvas_size else None
            data["canvas_affine"] = (self.canvas_affine.tolist()
                                     if self.canvas_affine is not None else None)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StereoRigCalibration":
        """Reconstruct a StereoRigCalibration from a dict.

        The nested CameraCalibration objects are rebuilt from their dicts.
        """
        calib_a = CameraCalibration.from_dict(data["calib_a"])
        calib_b = CameraCalibration.from_dict(data["calib_b"])
        obj = cls(calib_a, calib_b)

        obj.origin_offset_mm = tuple(data.get("origin_offset_mm", (0.0, 0.0)))
        if "R" in data and data["R"] is not None:
            obj.R = np.array(data["R"], dtype=np.float64)
            obj.T = np.array(data["T"], dtype=np.float64)
            obj.E = (np.array(data["E"], dtype=np.float64)
                     if data.get("E") is not None else None)
            obj.F = (np.array(data["F"], dtype=np.float64)
                     if data.get("F") is not None else None)
            obj.stereo_rms = data.get("stereo_rms", 0.0)
        if "H_A" in data and data["H_A"] is not None:
            obj.H_A = np.array(data["H_A"], dtype=np.float64)
            obj.H_B = np.array(data["H_B"], dtype=np.float64)
            obj.canvas_size = (tuple(data["canvas_size"])
                               if data.get("canvas_size") else None)
            obj.canvas_affine = (np.array(data["canvas_affine"], dtype=np.float64)
                                 if data.get("canvas_affine") else None)
        return obj

    def save(self, filepath: str) -> None:
        """Persist rig calibration to a .npz file."""
        data = self.to_dict()
        save_kwargs: Dict[str, Any] = {
            "metadata_json": json.dumps(data),
        }
        if self.R is not None:
            save_kwargs["R"] = self.R
            save_kwargs["T"] = self.T
            if self.E is not None:
                save_kwargs["E"] = self.E
            if self.F is not None:
                save_kwargs["F"] = self.F
        if self.H_A is not None:
            save_kwargs["H_A"] = self.H_A
            save_kwargs["H_B"] = self.H_B
        np.savez_compressed(filepath, **save_kwargs)

    @classmethod
    def load(cls, filepath: str,
             calib_a: Optional[CameraCalibration] = None,
             calib_b: Optional[CameraCalibration] = None) -> "StereoRigCalibration":
        """Load a StereoRigCalibration from a .npz file.

        If *calib_a* / *calib_b* are provided they override any
        embedded calibration data in the file.
        """
        packed = np.load(filepath, allow_pickle=False)
        if "metadata_json" in packed:
            data = json.loads(str(packed["metadata_json"]))
        else:
            # Legacy flat format
            data = {
                "calib_a": {"grid_size": [11, 9], "circle_spacing_mm": 10.0},
                "calib_b": {"grid_size": [11, 9], "circle_spacing_mm": 10.0},
            }
            if "H_A" in packed:
                data["H_A"] = packed["H_A"].tolist()
                data["H_B"] = packed["H_B"].tolist()

        # Override embedded calib if explicitly provided
        if calib_a is not None:
            data["calib_a"] = calib_a.to_dict()
        if calib_b is not None:
            data["calib_b"] = calib_b.to_dict()

        obj = cls.from_dict(data)
        # Overwrite matrices from raw arrays for precision
        if "R" in packed:
            obj.R = packed["R"]
            obj.T = packed["T"]
        if "E" in packed:
            obj.E = packed["E"]
            obj.F = packed["F"] if "F" in packed else None
        if "H_A" in packed:
            obj.H_A = packed["H_A"]
            obj.H_B = packed["H_B"]
        return obj


# ===========================================================================
# Typed calibration contract
# ===========================================================================
#
# ``CameraCalibration``/``StereoRigCalibration`` keep their teaching/serialisation
# responsibilities; these entries are the single numeric implementation used by
# the v1 GUI adapters and the v2 calibration operations.


@dataclass(frozen=True)
class CameraModel:
    """A finite pinhole camera with explicit frame and image size."""

    camera_id: str
    image_space: Space
    image_size_wh: Tuple[int, int]
    matrix: np.ndarray
    distortion: np.ndarray

    def __post_init__(self) -> None:
        if not self.camera_id:
            raise contract_error("contract.invalid_parameter", "camera_id must be non-empty",
                                 "camera.camera_id")
        width, height = self.image_size_wh
        if width <= 0 or height <= 0:
            raise contract_error("contract.invalid_parameter", "image size must be positive",
                                 "camera.image_size_wh")
        matrix = np.asarray(self.matrix, dtype=np.float64)
        if matrix.shape != (3, 3):
            raise contract_error("contract.invalid_shape", "camera matrix must be 3x3",
                                 "camera.matrix")
        distortion = np.asarray(self.distortion, dtype=np.float64).reshape(-1)
        require_finite((matrix, distortion), "camera.parameters")
        object.__setattr__(self, "matrix", matrix.copy())
        object.__setattr__(self, "distortion", distortion.copy())
        object.__setattr__(self, "image_size_wh", (int(width), int(height)))


@dataclass(frozen=True)
class Pose3D:
    """A rigid pose from one named frame to another."""

    source_frame: str
    destination_frame: str
    unit: str
    rotation: np.ndarray
    translation: np.ndarray

    def __post_init__(self) -> None:
        for name in ("source_frame", "destination_frame", "unit"):
            if not getattr(self, name):
                raise contract_error("contract.invalid_parameter", f"{name} must be non-empty",
                                     f"pose.{name}")
        rotation = np.asarray(self.rotation, dtype=np.float64)
        translation = np.asarray(self.translation, dtype=np.float64).reshape(-1)
        if rotation.shape != (3, 3) or translation.size != 3:
            raise contract_error("contract.invalid_shape",
                                 "pose needs a 3x3 rotation and a 3-vector translation",
                                 "pose")
        require_finite((rotation, translation), "pose")
        object.__setattr__(self, "rotation", rotation.copy())
        object.__setattr__(self, "translation", translation.copy())


@dataclass(frozen=True)
class BoardSpec:
    """A physical dot/chessboard grid."""

    columns: int
    rows: int
    spacing: float
    unit: str = "mm"
    symmetric: bool = True

    def __post_init__(self) -> None:
        if self.columns < 2 or self.rows < 2:
            raise contract_error("contract.invalid_parameter", "board needs at least 2x2 points",
                                 "board")
        require_finite(self.spacing, "board.spacing")
        if self.spacing <= 0.0:
            raise contract_error("contract.invalid_parameter", "spacing must be positive",
                                 "board.spacing")


@dataclass(frozen=True)
class DotDetectionOptions:
    """Blob-detection tuning for dot grids.

    Threshold fields default to OpenCV's own blob-detector defaults; callers
    that want a denser sweep pass a narrower range.
    """

    min_area: float = 5.0
    max_area: float = 2000.0
    min_circularity: float = 0.3
    min_convexity: float = 0.3
    #: ``dark`` dots on a light board, or ``light`` dots on a dark board.
    color: str = "dark"
    refine: bool = False
    min_threshold: float = 50.0
    max_threshold: float = 220.0
    threshold_step: float = 10.0

    def __post_init__(self) -> None:
        if self.color not in ("dark", "light"):
            raise contract_error("contract.invalid_parameter",
                                 f"unknown blob color {self.color!r}", "board.blob.color")
        require_finite((self.min_area, self.max_area, self.min_circularity,
                        self.min_convexity, self.min_threshold, self.max_threshold,
                        self.threshold_step), "board.blob")
        if self.max_area <= self.min_area:
            raise contract_error("contract.invalid_parameter",
                                 "max_area must be greater than min_area",
                                 "board.blob.max_area")
        if self.threshold_step <= 0.0 or self.min_threshold >= self.max_threshold:
            raise contract_error("contract.invalid_parameter",
                                 "blob thresholds must be ordered and stepped",
                                 "board.blob.min_threshold")


@dataclass(frozen=True)
class BoardObservation:
    """One board view with ordered image and object correspondences."""

    view_id: str
    image_space: Space
    object_space: Space
    image_size_wh: Tuple[int, int]
    image_points_xy: np.ndarray
    object_points_xyz: np.ndarray
    ordering: str = "row_major"

    def __post_init__(self) -> None:
        image_points = require_xy_array(self.image_points_xy, "board.image_points_xy")
        object_points = np.asarray(self.object_points_xyz, dtype=np.float64)
        if object_points.ndim != 2 or object_points.shape[1] != 3:
            raise contract_error("contract.invalid_shape",
                                 "object points must be (N, 3)", "board.object_points_xyz")
        if len(image_points) != len(object_points):
            raise contract_error("contract.invalid_shape",
                                 "image and object point counts must match", "board")
        require_finite(object_points, "board.object_points_xyz")
        if self.ordering not in ("row_major", "column_major"):
            raise contract_error("contract.invalid_parameter",
                                 f"unknown ordering {self.ordering!r}", "board.ordering")
        object.__setattr__(self, "image_points_xy", image_points)
        object.__setattr__(self, "object_points_xyz", object_points.copy())


@dataclass(frozen=True)
class IntrinsicOptions:
    """Pinhole calibration policy."""

    fix_aspect_ratio: bool = False
    fix_principal_point: bool = False
    rational_model: bool = False


@dataclass(frozen=True)
class StereoObservationPair:
    """One capture observed by both cameras."""

    capture_id: str
    a: BoardObservation
    b: BoardObservation


@dataclass(frozen=True)
class StereoOptions:
    """Stereo calibration policy."""

    fix_intrinsics: bool = True
    fix_principal_point: bool = False


@dataclass(frozen=True)
class IntrinsicResult:
    """Solved intrinsics with per-view poses and residuals."""

    camera: CameraModel
    poses: Tuple[Pose3D, ...]
    accepted_view_ids: Tuple[str, ...]
    rms_error_px: float
    per_view_errors_px: Tuple[float, ...]


@dataclass(frozen=True)
class StereoResult:
    """Relative pose between two calibrated cameras."""

    a_to_b: Pose3D
    baseline: float
    essential: np.ndarray
    rms_error_px: float
    accepted_capture_ids: Tuple[str, ...]


def validate_camera_model(camera: CameraModel) -> None:
    """Validate the numeric invariants of a pinhole model.

    Args:
        camera: Candidate camera model.

    Raises:
        ComputationContractError: On non-finite or degenerate intrinsics.
    """
    matrix = np.asarray(camera.matrix, dtype=np.float64)
    if float(matrix[0, 0]) <= 0.0 or float(matrix[1, 1]) <= 0.0:
        raise contract_error("calibration.camera_invalid", "focal lengths must be positive",
                             "camera.matrix")
    if float(matrix[2, 2]) == 0.0:
        raise contract_error("calibration.camera_invalid", "invalid homogeneous scale",
                             "camera.matrix")


@dataclass(frozen=True)
class Undistortion:
    """Corrected pixels plus the complete inverse map of one camera."""

    pixels: np.ndarray
    #: Corrected camera matrix used for both the pixels and the inverse map.
    camera_matrix: np.ndarray
    #: Valid destination ``(x, y, width, height)`` after cropping.
    valid_roi_xywh: Tuple[int, int, int, int]
    #: Destination-to-source column map.
    inverse_map_x: np.ndarray
    #: Destination-to-source row map.
    inverse_map_y: np.ndarray


def undistort_image(image: np.ndarray, camera: CameraModel, *,
                    alpha: float = 0.0) -> Undistortion:
    """Remove lens distortion and return the destination-to-source inverse map.

    Args:
        image: ``uint8`` image whose size matches ``camera.image_size_wh``.
        camera: Validated pinhole camera with distortion coefficients.
        alpha: ``0`` keeps only valid pixels, ``1`` keeps every source pixel.

    Returns:
        Corrected pixels, the corrected camera matrix, the valid destination
        ROI and the inverse map arrays.

    Raises:
        ComputationContractError: On an image size mismatch or an alpha outside
            ``[0, 1]``.
    """
    width, height = camera.image_size_wh
    if image.shape[:2] != (height, width):
        raise contract_error("contract.invalid_shape",
                             "image size must match the camera image size", "image")
    require_finite(alpha, "undistort.alpha")
    if not 0.0 <= float(alpha) <= 1.0:
        raise contract_error("contract.invalid_parameter", "alpha must be in [0, 1]", "alpha")
    matrix, distortion = camera.matrix, camera.distortion
    corrected_matrix, roi = cv2.getOptimalNewCameraMatrix(
        matrix, distortion, (width, height), float(alpha))
    pixels = cv2.undistort(image, matrix, distortion, None, corrected_matrix)
    map_x, map_y = cv2.initUndistortRectifyMap(
        matrix, distortion, None, corrected_matrix, (width, height), cv2.CV_32FC1)
    return Undistortion(pixels=pixels, camera_matrix=corrected_matrix,
                        valid_roi_xywh=tuple(int(value) for value in roi),
                        inverse_map_x=map_x, inverse_map_y=map_y)


def validate_pose(pose: Pose3D) -> None:
    """Validate that a pose is a proper rigid transform.

    Args:
        pose: Candidate pose.

    Raises:
        ComputationContractError: When the rotation is not orthonormal with
            determinant +1.
    """
    rotation = np.asarray(pose.rotation, dtype=np.float64)
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=1e-9, atol=1e-9):
        raise contract_error("calibration.pose_invalid", "rotation is not orthonormal", "pose")
    determinant = float(np.linalg.det(rotation))
    if determinant <= 0.0:
        raise contract_error("calibration.pose_invalid",
                             "rotation must not reflect (det <= 0)", "pose")


def board_object_points(spec: BoardSpec) -> np.ndarray:
    """Return the row-major physical ``(N, 3)`` coordinates of a board grid.

    OpenCV orders both grid patterns row-major, but an asymmetric grid staggers
    every odd row by one spacing, so its physical x is ``2 * column + row % 2``.

    Args:
        spec: Board geometry.

    Returns:
        An ``(N, 3)`` array of board coordinates in ``spec.unit``.
    """
    points = np.zeros((spec.columns * spec.rows, 3), dtype=np.float64)
    for index in range(len(points)):
        row, column = divmod(index, spec.columns)
        staggered = column if spec.symmetric else 2 * column + row % 2
        points[index] = (staggered * spec.spacing, row * spec.spacing, 0.0)
    return points


def detect_dot_grid(image: np.ndarray, spec: BoardSpec, options: DotDetectionOptions, *,
                    image_space: Space, object_frame: str,
                    view_id: str) -> Observation[BoardObservation]:
    """Detect an ordered dot grid and build its physical correspondences.

    Args:
        image: ``uint8`` image containing the grid; colour is reduced to luma.
        spec: Board geometry.
        options: Blob-detection tuning.
        image_space: Space of the image.
        object_frame: Name of the board frame.
        view_id: Caller capture identity.

    Returns:
        A valid board observation, or ``grid_not_found``.
    """
    if image.dtype != np.uint8:
        raise contract_error("contract.invalid_parameter", "image must be uint8", "image")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    parameters = cv2.SimpleBlobDetector_Params()
    parameters.minThreshold = float(options.min_threshold)
    parameters.maxThreshold = float(options.max_threshold)
    parameters.thresholdStep = float(options.threshold_step)
    parameters.filterByArea = True
    parameters.minArea = float(options.min_area)
    parameters.maxArea = float(options.max_area)
    parameters.filterByCircularity = True
    parameters.minCircularity = float(options.min_circularity)
    parameters.filterByConvexity = True
    parameters.minConvexity = float(options.min_convexity)
    parameters.filterByInertia = False
    parameters.filterByColor = True
    parameters.blobColor = 0 if options.color == "dark" else 255
    detector = cv2.SimpleBlobDetector_create(parameters)
    flags = cv2.CALIB_CB_SYMMETRIC_GRID if spec.symmetric else cv2.CALIB_CB_ASYMMETRIC_GRID
    found, centers = cv2.findCirclesGrid(gray, (spec.columns, spec.rows), flags=flags,
                                         blobDetector=detector)
    if not found or centers is None:
        return invalid(image_space, "grid_not_found",
                       {"board": spec, "refined": False})
    refined = False
    if options.refine:
        corners = np.asarray(centers, dtype=np.float32).reshape(-1, 1, 2)
        cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1),
                         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        centers = corners.reshape(-1, 2)
        refined = True
    image_points = np.asarray(centers, dtype=np.float64).reshape(-1, 2)
    observation = BoardObservation(
        view_id=view_id, image_space=image_space,
        object_space=Space(object_frame, spec.unit),
        image_size_wh=(image.shape[1], image.shape[0]),
        image_points_xy=image_points,
        object_points_xyz=board_object_points(spec)[:len(image_points)],
        ordering="row_major",
    )
    return observed(image_space, observation, {"board": spec, "refined": refined})


def _solve_intrinsics(views, camera_id, image_space, image_size_wh, options):
    """Run OpenCV's intrinsic solve on prepared board observations."""
    object_points = [np.asarray(view.object_points_xyz, dtype=np.float32) for view in views]
    image_points = [np.asarray(view.image_points_xy, dtype=np.float32) for view in views]
    flags = 0
    if options.fix_aspect_ratio:
        flags |= cv2.CALIB_FIX_ASPECT_RATIO
    if options.fix_principal_point:
        flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
    if options.rational_model:
        flags |= cv2.CALIB_RATIONAL_MODEL
    rms, matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, tuple(image_size_wh), None, None, flags=flags,
    )
    camera = CameraModel(camera_id=camera_id, image_space=image_space,
                         image_size_wh=image_size_wh, matrix=matrix,
                         distortion=np.asarray(distortion).reshape(-1))
    poses = tuple(
        Pose3D(source_frame="board", destination_frame=camera.image_space.frame,
               unit="mm", rotation=cv2.Rodrigues(rvec)[0], translation=np.asarray(tvec).reshape(3))
        for rvec, tvec in zip(rvecs, tvecs)
    )
    per_view = []
    for index, (view, rvec, tvec) in enumerate(zip(views, rvecs, tvecs)):
        projected, _ = cv2.projectPoints(
            np.asarray(view.object_points_xyz, dtype=np.float64), rvec, tvec,
            np.asarray(matrix, dtype=np.float64), np.asarray(distortion, dtype=np.float64),
        )
        error = np.linalg.norm(projected.reshape(-1, 2) - view.image_points_xy, axis=1)
        per_view.append(float(np.sqrt(np.mean(error ** 2))))
    return float(rms), camera, poses, tuple(per_view)


def fit_camera_intrinsics(views, *, camera_id: str, image_space: Space,
                          image_size_wh: Tuple[int, int],
                          options: Optional[IntrinsicOptions] = None
                          ) -> Observation[IntrinsicResult]:
    """Recover pinhole intrinsics from independent board views.

    Args:
        views: Board observations (at least three distinct views).
        camera_id: Identity recorded in the solved model.
        image_space: Space of every supplied view.
        image_size_wh: Image size in whole pixels.
        options: Solver policy.

    Returns:
        A valid result with camera, poses and residuals, or
        ``insufficient_views`` when fewer than three distinct views exist.
    """
    options = options or IntrinsicOptions()
    distinct = {}
    for view in views:
        distinct.setdefault((np.asarray(view.image_points_xy).round(9).tobytes(),
                             np.asarray(view.object_points_xyz).round(9).tobytes()), view)
    if len(distinct) < 3:
        return invalid(image_space, "insufficient_views")
    ordered = list(distinct.values())
    rms, camera, poses, per_view = _solve_intrinsics(
        ordered, camera_id, image_space, image_size_wh, options)
    result = IntrinsicResult(camera=camera, poses=poses,
                             accepted_view_ids=tuple(view.view_id for view in ordered),
                             rms_error_px=rms, per_view_errors_px=per_view)
    return observed(image_space, result, {"view_count": len(ordered)})


def fit_stereo_rig(pairs, camera_a: CameraModel, camera_b: CameraModel, *,
                   unit: str, options: Optional[StereoOptions] = None
                   ) -> Observation[StereoResult]:
    """Solve the relative pose between two calibrated cameras.

    Args:
        pairs: Paired observations; at least three distinct captures.
        camera_a: Camera A intrinsics (never mutated).
        camera_b: Camera B intrinsics (never mutated).
        unit: Physical unit of the supplied object points.
        options: Solver policy.

    Returns:
        A valid stereo result, or an invalid observation when too few distinct
        pairs are supplied.
    """
    options = options or StereoOptions()
    records = tuple(pairs)
    distinct = {}
    for pair in records:
        distinct.setdefault(pair.capture_id, pair)
    if len(distinct) < 3:
        return invalid(camera_a.image_space, "insufficient_pairs")
    ordered = list(distinct.values())
    object_points = [np.asarray(pair.a.object_points_xyz, dtype=np.float32)
                     for pair in ordered]
    points_a = [np.asarray(pair.a.image_points_xy, dtype=np.float32) for pair in ordered]
    points_b = [np.asarray(pair.b.image_points_xy, dtype=np.float32) for pair in ordered]
    flags = cv2.CALIB_FIX_INTRINSIC if options.fix_intrinsics else 0
    if options.fix_principal_point:
        flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
    rms, k_a, d_a, k_b, d_b, rotation, translation, essential, fundamental = cv2.stereoCalibrate(
        object_points, points_a, points_b,
        np.array(camera_a.matrix, dtype=np.float64),
        np.array(camera_a.distortion, dtype=np.float64),
        np.array(camera_b.matrix, dtype=np.float64),
        np.array(camera_b.distortion, dtype=np.float64),
        tuple(camera_a.image_size_wh), flags=flags,
    )
    translation = np.asarray(translation, dtype=np.float64).reshape(3)
    rotation = np.asarray(rotation, dtype=np.float64)
    pose = Pose3D(source_frame=camera_a.image_space.frame,
                  destination_frame=camera_b.image_space.frame, unit=unit,
                  rotation=rotation, translation=translation)
    result = StereoResult(
        a_to_b=pose, baseline=float(np.linalg.norm(translation)),
        essential=np.asarray(essential, dtype=np.float64),
        rms_error_px=float(rms),
        accepted_capture_ids=tuple(pair.capture_id for pair in ordered),
    )
    return observed(camera_a.image_space, result, {"pair_count": len(ordered)})
