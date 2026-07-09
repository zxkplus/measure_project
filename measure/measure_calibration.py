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
import numpy as np
import cv2
from typing import Tuple, List, Optional, Dict, Any
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
        # Overwrite H matrices from raw arrays for precision
        if "H_A" in packed:
            obj.H_A = packed["H_A"]
            obj.H_B = packed["H_B"]
        return obj
