"""
Template Matching 2D Point Measurement Module

Core principle:
1. Crop a square template centered at a user-clicked position on a reference image
2. Extract Canny edge features from the template (lighting-invariant representation)
3. On inspection images, match the edge template using normalized cross-correlation
4. Refine the match location to subpixel accuracy via quadratic interpolation
5. Compute Euclidean distance between two matched points

Usage:
    # Create templates from reference image
    pt_a = TemplatePoint(ref_img, click_row=200, click_col=150, template_size=80)
    pt_b = TemplatePoint(ref_img, click_row=200, click_col=350, template_size=80)

    # Save for later use
    pt_a.save("template_A.npz")
    pt_b.save("template_B.npz")

    # ... later, on a new image ...
    pt_a = TemplatePoint.from_file("template_A.npz")
    pt_b = TemplatePoint.from_file("template_B.npz")

    dm = DistanceMeasure(pt_a, pt_b)
    result = dm.measure(inspection_img)
    print(f"Distance: {result['distance']:.3f} px")
    vis = dm.visualize(inspection_img)
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any


class TemplatePoint:
    """
    Template-matching point measurement object.

    Crops a square template centered at a user-clicked position on a reference
    image, extracts Canny edge features, and matches them against inspection
    images using normalized cross-correlation.

    Usage:
        pt = TemplatePoint(ref_image, click_row=200, click_col=300, template_size=80)
        pt.save("template_A.npz")
        # ... later ...
        pt2 = TemplatePoint.from_file("template_A.npz")
        result = pt2.measure(inspection_image)
        vis = pt2.visualize(inspection_image)
    """

    def __init__(self,
                 reference_image: np.ndarray,
                 click_row: float,
                 click_col: float,
                 template_size: int = 80,
                 use_edges: bool = False,
                 canny_threshold1: float = 50.0,
                 canny_threshold2: float = 150.0,
                 match_score_threshold: float = 0.5,
                 use_subpixel: bool = True):
        """
        Initialize a template point from a reference image.

        Crops a template_size x template_size square centered at (click_row, click_col)
        from the reference image. When use_edges=True, extracts a Canny edge map;
        otherwise uses raw pixel intensity for template matching.

        Args:
            reference_image: Grayscale reference image (uint8)
            click_row: User-clicked row position (y-coordinate in image)
            click_col: User-clicked column position (x-coordinate in image)
            template_size: Square template side length in pixels (default 80)
            use_edges: If True, match on Canny edge maps (lighting-robust).
                       If False, match on raw pixel intensity (default False).
            canny_threshold1: Canny lower threshold (default 50, only when use_edges=True)
            canny_threshold2: Canny upper threshold (default 150, only when use_edges=True)
            match_score_threshold: Minimum NCC score for valid match (default 0.5)
            use_subpixel: Enable subpixel refinement of the correlation peak (default True)
        """
        self.click_row = click_row
        self.click_col = click_col
        self.template_size = template_size
        self.use_edges = use_edges
        self.canny_threshold1 = canny_threshold1
        self.canny_threshold2 = canny_threshold2
        self.match_score_threshold = match_score_threshold
        self.use_subpixel = use_subpixel

        # Crop the template region (clamped to image bounds)
        gray = self._to_gray(reference_image)
        self._actual_crop_bounds = self._compute_crop_bounds(gray.shape)
        r1, r2, c1, c2 = self._actual_crop_bounds
        crop = gray[r1:r2, c1:c2]

        # Guard: click position may be entirely outside the image
        if crop.size == 0:
            raise ValueError(
                f"Template crop is empty. Click position (row={click_row:.0f}, col={click_col:.0f}) "
                f"with template_size={template_size} is entirely outside the "
                f"image bounds (h={gray.shape[0]}, w={gray.shape[1]})."
            )

        # Compute the actual center of the crop in image coordinates
        # (may differ from click position if clamped at image boundary)
        self._crop_center_row = (r1 + r2) / 2.0
        self._crop_center_col = (c1 + c2) / 2.0
        self._crop_h, self._crop_w = crop.shape

        if self.use_edges:
            # Extract Canny edge template
            edge_result = self._extract_edges(crop)
            if edge_result is None or edge_result.size == 0:
                raise ValueError(
                    f"Edge extraction returned empty result for template at "
                    f"(row={click_row:.0f}, col={click_col:.0f}). "
                    f"Crop size is {self._crop_h}x{self._crop_w} px."
                )
            self.edge_template = edge_result

            # Validate template has sufficient edge content
            edge_ratio = np.count_nonzero(self.edge_template) / self.edge_template.size
            if edge_ratio < 0.001:
                print(f"Warning: Template has very few edges (edge pixel ratio={edge_ratio:.4f}). "
                      f"Matching may be unreliable. Consider selecting a point with more texture.")
        else:
            # Use raw pixel intensity — store the crop directly
            self.edge_template = crop.astype(np.float32)

        # Result storage
        self.result: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Core measurement
    # ------------------------------------------------------------------

    def measure(self,
                inspection_image: np.ndarray,
                search_region: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
        """
        Match the template against an inspection image.

        When use_edges=True, matches on Canny edge maps.
        When use_edges=False (default), matches on raw pixel intensity.

        Args:
            inspection_image: Grayscale inspection image (uint8)
            search_region: Optional search bounds as (r1, r2, c1, c2).
                           Restricts matching to this sub-region for speed
                           and to reduce false positives.

        Returns:
            Dict with keys:
                - 'matched_row': float, matched center row in inspection image
                - 'matched_col': float, matched center col in inspection image
                - 'dx': float, displacement in cols from reference click position
                - 'dy': float, displacement in rows from reference click position
                - 'match_score': float, NCC score in [-1, 1], higher = better
                - 'valid': bool, whether match_score >= match_score_threshold
                - 'int_peak_y': int, integer peak row in the heatmap
                - 'int_peak_x': int, integer peak col in the heatmap
        """
        gray = self._to_gray(inspection_image)

        # Validate image is large enough
        if gray.shape[0] < self._crop_h or gray.shape[1] < self._crop_w:
            raise ValueError(
                f"Inspection image size ({gray.shape[0]}x{gray.shape[1]}) "
                f"is smaller than template size ({self._crop_h}x{self._crop_w})"
            )

        # Prepare search image (apply Canny if use_edges, else use raw pixels)
        if search_region is not None:
            r1, r2, c1, c2 = search_region
            r1 = max(0, r1)
            r2 = min(gray.shape[0], r2)
            c1 = max(0, c1)
            c2 = min(gray.shape[1], c2)
            search_img = gray[r1:r2, c1:c2]
        else:
            r1, c1 = 0, 0
            search_img = gray

        if self.use_edges:
            search_img = self._extract_edges(search_img)
        else:
            search_img = search_img.astype(np.float32)

        # Template matching
        heatmap = cv2.matchTemplate(search_img, self.edge_template, cv2.TM_CCOEFF_NORMED)

        # Find integer peak
        _, max_val, _, max_loc = cv2.minMaxLoc(heatmap)
        int_peak_x = max_loc[0]  # col in heatmap
        int_peak_y = max_loc[1]  # row in heatmap

        # Subpixel refinement
        if self.use_subpixel:
            subpixel_x, subpixel_y = self._refine_subpixel_2d(heatmap, int_peak_y, int_peak_x)
        else:
            subpixel_x = float(int_peak_x)
            subpixel_y = float(int_peak_y)

        # Compute matched center in the full inspection image coordinates
        # The heatmap position gives the top-left corner of the template match
        offset_y = self._crop_h / 2.0
        offset_x = self._crop_w / 2.0
        matched_row = r1 + subpixel_y + offset_y
        matched_col = c1 + subpixel_x + offset_x

        # Compute displacement from original click position
        dy = matched_row - self.click_row
        dx = matched_col - self.click_col

        self.result = {
            'matched_row': matched_row,
            'matched_col': matched_col,
            'dx': dx,
            'dy': dy,
            'match_score': float(max_val),
            'valid': max_val >= self.match_score_threshold,
            'int_peak_y': int_peak_y,
            'int_peak_x': int_peak_x,
        }
        return self.result

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def visualize(self,
                  image: np.ndarray,
                  show_template_box: bool = True,
                  show_matched_point: bool = True,
                  show_labels: bool = True,
                  template_color: Tuple[int, int, int] = (0, 255, 0),
                  matched_color: Tuple[int, int, int] = (0, 0, 255),
                  line_thickness: int = 2,
                  point_radius: int = 6,
                  wait_time: int = -1) -> np.ndarray:
        """
        Visualize the template region and matched point on an image.

        Args:
            image: Input image (grayscale or BGR)
            show_template_box: Draw the template crop region
            show_matched_point: Draw the matched center crosshair
            show_labels: Show text labels with coordinates and score
            template_color: Color for the template box (B, G, R)
            matched_color: Color for the matched point (B, G, R)
            line_thickness: Line thickness for drawings
            point_radius: Radius of the matched point marker
            wait_time: OpenCV waitKey time in ms (-1 = no display)

        Returns:
            Annotated BGR image (copy)
        """
        vis_img = self._to_bgr(image)

        if show_template_box and hasattr(self, '_actual_crop_bounds'):
            self._draw_template_box(vis_img, template_color, line_thickness)

        if show_matched_point and self.result is not None:
            self._draw_matched_point(vis_img, matched_color, point_radius, show_labels)

        self._draw_info(vis_img)

        if wait_time >= 0:
            cv2.imshow("Template Point", vis_img)
            cv2.waitKey(wait_time)
            if wait_time > 0:
                cv2.destroyWindow("Template Point")

        return vis_img

    def _draw_template_box(self, img: np.ndarray,
                           color: Tuple[int, int, int],
                           thickness: int):
        """Draw the template crop region as a square centered on the click position."""
        half = self.template_size / 2.0
        r1 = int(self.click_row - half)
        r2 = int(self.click_row + half)
        c1 = int(self.click_col - half)
        c2 = int(self.click_col + half)

        # Double-draw for visibility (black outline + color)
        cv2.rectangle(img, (c1, r1), (c2, r2), (0, 0, 0), thickness + 2)
        cv2.rectangle(img, (c1, r1), (c2, r2), color, thickness)

        # Click position marker
        cv2.circle(img, (int(self.click_col), int(self.click_row)), 4, color, -1)

    def _draw_matched_point(self, img: np.ndarray,
                            color: Tuple[int, int, int],
                            radius: int,
                            show_labels: bool):
        """Draw the matched center as a crosshair."""
        r = self.result['matched_row']
        c = self.result['matched_col']

        # Crosshair
        cv2.line(img, (int(c) - radius, int(r)), (int(c) + radius, int(r)),
                 (0, 0, 0), 3)
        cv2.line(img, (int(c) - radius, int(r)), (int(c) + radius, int(r)),
                 color, 2)
        cv2.line(img, (int(c), int(r) - radius), (int(c), int(r) + radius),
                 (0, 0, 0), 3)
        cv2.line(img, (int(c), int(r) - radius), (int(c), int(r) + radius),
                 color, 2)

        if show_labels:
            score = self.result['match_score']
            label = f'({c:.1f}, {r:.1f}) score:{score:.3f}'
            cv2.putText(img, label, (int(c) + radius + 5, int(r) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
            cv2.putText(img, label, (int(c) + radius + 5, int(r) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    def _draw_info(self, img: np.ndarray):
        """Draw information text overlay."""
        y = 25
        mode_str = 'Edges' if self.use_edges else 'Raw'
        title = f'Template Point [{mode_str}]'
        cv2.putText(img, title, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(img, title, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        y += 22
        size_text = f'Template: {self._crop_h}x{self._crop_w} px'
        cv2.putText(img, size_text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
        cv2.putText(img, size_text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        if self.result is not None:
            y += 20
            score_text = f'Score: {self.result["match_score"]:.4f}'
            cv2.putText(img, score_text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
            cv2.putText(img, score_text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            y += 20
            valid_text = 'VALID' if self.result['valid'] else 'INVALID'
            vcolor = (0, 255, 0) if self.result['valid'] else (0, 0, 255)
            cv2.putText(img, valid_text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(img, valid_text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, vcolor, 1)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, filepath: str) -> None:
        """
        Serialize the template to a .npz file.

        Saves the edge template and all configuration parameters needed
        to reconstruct the TemplatePoint without the original reference image.

        Args:
            filepath: Path to the .npz output file
        """
        np.savez_compressed(
            filepath,
            edge_template=self.edge_template,
            click_row=self.click_row,
            click_col=self.click_col,
            template_size=self.template_size,
            use_edges=self.use_edges,
            canny_threshold1=self.canny_threshold1,
            canny_threshold2=self.canny_threshold2,
            match_score_threshold=self.match_score_threshold,
            use_subpixel=self.use_subpixel,
            crop_center_row=self._crop_center_row,
            crop_center_col=self._crop_center_col,
            crop_h=self._crop_h,
            crop_w=self._crop_w,
            actual_crop_bounds=np.array(self._actual_crop_bounds, dtype=np.int32),
        )

    @classmethod
    def from_file(cls, filepath: str) -> 'TemplatePoint':
        """
        Deserialize a TemplatePoint from a .npz file.

        Args:
            filepath: Path to the .npz file

        Returns:
            A fully initialized TemplatePoint ready for measure()
        """
        data = np.load(filepath, allow_pickle=False)
        # Bypass __init__ since we don't have a reference image
        obj = cls.__new__(cls)
        obj.edge_template = data['edge_template']
        obj.click_row = float(data['click_row'])
        obj.click_col = float(data['click_col'])
        obj.template_size = int(data['template_size'])
        # Backward compat: old .npz files don't have use_edges, default to True
        obj.use_edges = bool(data['use_edges']) if 'use_edges' in data else True
        obj.canny_threshold1 = float(data['canny_threshold1'])
        obj.canny_threshold2 = float(data['canny_threshold2'])
        obj.match_score_threshold = float(data['match_score_threshold'])
        obj.use_subpixel = bool(data['use_subpixel'])
        obj._crop_center_row = float(data['crop_center_row'])
        obj._crop_center_col = float(data['crop_center_col'])
        obj._crop_h = int(data['crop_h'])
        obj._crop_w = int(data['crop_w'])
        obj._actual_crop_bounds = tuple(data['actual_crop_bounds'].tolist())
        obj.result = None
        return obj

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale if it is BGR."""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    @staticmethod
    def _to_bgr(image: np.ndarray) -> np.ndarray:
        """Convert image to BGR if it is grayscale (returns a copy)."""
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image.copy()

    def _compute_crop_bounds(self, image_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """
        Compute the crop rectangle clamped to image bounds.

        Returns:
            (r1, r2, c1, c2): Row and column bounds of the crop
        """
        h, w = image_shape
        half = self.template_size / 2.0
        r1 = int(max(0, self.click_row - half))
        r2 = int(min(h, self.click_row + half))
        c1 = int(max(0, self.click_col - half))
        c2 = int(min(w, self.click_col + half))
        return (r1, r2, c1, c2)

    def _extract_edges(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Canny edge detection.

        Returns:
            Binary edge map (uint8, values 0 or 255)
        """
        return cv2.Canny(image, self.canny_threshold1, self.canny_threshold2)

    def _refine_subpixel_2d(self, heatmap: np.ndarray,
                             peak_y: int, peak_x: int) -> Tuple[float, float]:
        """
        Refine the correlation peak to subpixel accuracy using two independent
        1D quadratic fits along x and y through the peak.

        This mirrors the _refine_subpixel pattern from measure1D.py, extended
        to the correlation surface: one quadratic fit along the row direction
        through the peak, and one along the column direction.

        Args:
            heatmap: 2D correlation map from cv2.matchTemplate
            peak_y: Integer peak row index
            peak_x: Integer peak col index

        Returns:
            (subpixel_x, subpixel_y): Refined peak position in the heatmap
        """
        h, w = heatmap.shape

        # Refine along x (columns) — fit to heatmap[peak_y, peak_x-1 : peak_x+2]
        if 0 < peak_x < w - 1:
            x_vals = np.array([peak_x - 1, peak_x, peak_x + 1], dtype=np.float64)
            y_vals = heatmap[peak_y, peak_x - 1 : peak_x + 2].astype(np.float64)
            try:
                coeffs = np.polyfit(x_vals, y_vals, 2)
                a, b, c = coeffs
                if abs(a) > 1e-10:
                    subpixel_x = -b / (2.0 * a)
                else:
                    subpixel_x = float(peak_x)
                subpixel_x = max(x_vals[0], min(x_vals[-1], subpixel_x))
            except Exception:
                subpixel_x = float(peak_x)
        else:
            subpixel_x = float(peak_x)

        # Refine along y (rows) — fit to heatmap[peak_y-1 : peak_y+2, peak_x]
        if 0 < peak_y < h - 1:
            x_vals = np.array([peak_y - 1, peak_y, peak_y + 1], dtype=np.float64)
            y_vals = heatmap[peak_y - 1 : peak_y + 2, peak_x].astype(np.float64)
            try:
                coeffs = np.polyfit(x_vals, y_vals, 2)
                a, b, c = coeffs
                if abs(a) > 1e-10:
                    subpixel_y = -b / (2.0 * a)
                else:
                    subpixel_y = float(peak_y)
                subpixel_y = max(x_vals[0], min(x_vals[-1], subpixel_y))
            except Exception:
                subpixel_y = float(peak_y)
        else:
            subpixel_y = float(peak_y)

        return subpixel_x, subpixel_y


class DistanceMeasure:
    """
    Two-point distance measurement using template matching.

    Holds two TemplatePoint instances, matches both against an inspection image,
    and computes the Euclidean distance between their matched positions.

    Usage:
        dm = DistanceMeasure(point_a, point_b)
        result = dm.measure(inspection_image)
        print(f"Distance: {result['distance']:.3f} px")
        vis = dm.visualize(inspection_image)
    """

    def __init__(self, point_a: TemplatePoint, point_b: TemplatePoint):
        """
        Initialize a distance measure with two template points.

        Args:
            point_a: First TemplatePoint
            point_b: Second TemplatePoint
        """
        self.point_a = point_a
        self.point_b = point_b
        self.result: Optional[Dict[str, Any]] = None

    def measure(self,
                inspection_image: np.ndarray,
                search_region_a: Optional[Tuple[int, int, int, int]] = None,
                search_region_b: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
        """
        Match both templates against the inspection image and compute distance.

        Args:
            inspection_image: Grayscale inspection image (uint8)
            search_region_a: Optional search bounds for point A
            search_region_b: Optional search bounds for point B

        Returns:
            Dict with keys:
                - 'point_a': result dict from point_a.measure()
                - 'point_b': result dict from point_b.measure()
                - 'distance': float, Euclidean distance in pixels
                - 'valid': bool, True only when both points matched successfully
        """
        result_a = self.point_a.measure(inspection_image, search_region_a)
        result_b = self.point_b.measure(inspection_image, search_region_b)

        # Euclidean distance between matched centers
        d_row = result_b['matched_row'] - result_a['matched_row']
        d_col = result_b['matched_col'] - result_a['matched_col']
        distance = np.sqrt(d_row ** 2 + d_col ** 2)

        self.result = {
            'point_a': result_a,
            'point_b': result_b,
            'distance': float(distance),
            'valid': result_a['valid'] and result_b['valid'],
        }
        return self.result

    def visualize(self,
                  image: np.ndarray,
                  show_distance_line: bool = True,
                  distance_color: Tuple[int, int, int] = (255, 0, 255),
                  line_thickness: int = 2,
                  wait_time: int = -1,
                  **kwargs) -> np.ndarray:
        """
        Visualize both template points and the distance line between them.

        Args:
            image: Input image (grayscale or BGR)
            show_distance_line: Draw a line between the two matched points
            distance_color: Color for the distance line (B, G, R)
            line_thickness: Line thickness
            wait_time: OpenCV waitKey time in ms (-1 = no display)
            **kwargs: Forwarded to each TemplatePoint.visualize()
                      (e.g., show_matched_point, show_labels, template_color)

        Returns:
            Annotated BGR image (copy)
        """
        # Each point's visualize handles grayscale-to-BGR conversion internally,
        # so we just pass the image through sequentially
        vis_img = self.point_a.visualize(image, wait_time=-1, **kwargs)
        vis_img = self.point_b.visualize(vis_img, wait_time=-1, **kwargs)

        # Draw distance line
        if show_distance_line and self.result is not None:
            self._draw_distance_line(vis_img, distance_color, line_thickness)

        # Draw distance info
        self._draw_distance_info(vis_img)

        if wait_time >= 0:
            cv2.imshow("Distance Measure", vis_img)
            cv2.waitKey(wait_time)
            if wait_time > 0:
                cv2.destroyWindow("Distance Measure")

        return vis_img

    def _draw_distance_line(self, img: np.ndarray,
                            color: Tuple[int, int, int],
                            thickness: int):
        """Draw a line between the two matched points with distance label."""
        r1 = self.result['point_a']['matched_row']
        c1 = self.result['point_a']['matched_col']
        r2 = self.result['point_b']['matched_row']
        c2 = self.result['point_b']['matched_col']

        # Double-draw for visibility
        cv2.line(img, (int(c1), int(r1)), (int(c2), int(r2)),
                 (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.line(img, (int(c1), int(r1)), (int(c2), int(r2)),
                 color, thickness, cv2.LINE_AA)

        # Distance label at midpoint
        mid_r = int((r1 + r2) / 2)
        mid_c = int((c1 + c2) / 2)
        dist_text = f'{self.result["distance"]:.2f} px'
        cv2.putText(img, dist_text, (mid_c - 30, mid_r - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(img, dist_text, (mid_c - 30, mid_r - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def _draw_distance_info(self, img: np.ndarray):
        """Draw distance measurement summary text."""
        y = img.shape[0] - 40

        if self.result is not None:
            valid_text = 'VALID' if self.result['valid'] else 'PARTIAL/INVALID'
            vcolor = (0, 255, 0) if self.result['valid'] else (0, 165, 255)
            cv2.putText(img, valid_text, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
            cv2.putText(img, valid_text, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, vcolor, 1)
