"""
Tkinter Canvas-based image viewer with zoom, pan, and interactive rotated ROI drawing.

Supports three modes:
  - BROWSE: Pan and zoom the image
  - DRAW_ROI: Draw and edit a rotated rectangular ROI
  - VIEW_RESULT: View inspection results (read-only)
"""

from __future__ import annotations

import tkinter as tk
from enum import Enum, auto
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

from .utils import compute_rotated_box_corners, cv2_to_tk


class CanvasMode(Enum):
    BROWSE = auto()
    DRAW_ROI = auto()
    VIEW_RESULT = auto()


class ImageCanvas(tk.Frame):
    """
    Scrollable, zoomable image viewer with interactive rotated ROI drawing.

    Usage:
        canvas = ImageCanvas(parent)
        canvas.load_image(image)
        canvas.set_mode(CanvasMode.DRAW_ROI)
        canvas.on_roi_changed = lambda center, size, angle: print("ROI updated")
    """

    MIN_ZOOM = 0.05
    MAX_ZOOM = 20.0
    ZOOM_STEP = 1.1
    ZOOM_DEBOUNCE_MS = 80      # ms between zoom redraws (prevents UI freeze from rapid scrolling)
    HANDLE_RADIUS = 6          # pixel radius of control handles on canvas
    ROTATE_HANDLE_OFFSET = 30  # pixels from top edge midpoint on canvas

    def __init__(
        self,
        parent: tk.Widget,
        width: int = 800,
        height: int = 600,
        bg: str = "#2d2d2d",
        **kwargs,
    ):
        super().__init__(parent, **kwargs)

        self._width = width
        self._height = height

        # Image data
        self._image: Optional[np.ndarray] = None           # original (BGR or grayscale)
        self._display_image: Optional[np.ndarray] = None   # scaled for display

        # View transform: canvas_x = img_col * zoom + offset_x, canvas_y = img_row * zoom + offset_y
        self._zoom: float = 1.0
        self._offset_x: float = 0.0
        self._offset_y: float = 0.0

        # Zoom debounce
        self._zoom_after_id: Optional[str] = None  # tkinter after ID
        self._accumulated_factor: float = 1.0
        self._zoom_cursor_x: float = 0.0
        self._zoom_cursor_y: float = 0.0

        # ROI state
        self._mode: CanvasMode = CanvasMode.BROWSE
        self._roi_center: Optional[Tuple[float, float]] = None  # (row, col) in image coords
        self._roi_size: Tuple[float, float] = (100.0, 100.0)    # (height, width) in image coords
        self._roi_angle: float = 0.0                             # degrees
        self._roi_confirmed: bool = False

        # Interaction state
        self._drag_start: Optional[Tuple[float, float]] = None   # canvas coords
        self._drag_what: Optional[str] = None  # 'pan', 'move', 'resize_tl', ..., 'rotate'
        self._drawing_phase: int = 0  # 0=wait_center, 1=wait_size

        # Callbacks
        self.on_roi_changed: Optional[Callable] = None
        # Signature: (center_row, center_col, height, width, angle_deg) -> None
        self.on_roi_confirmed: Optional[Callable] = None
        # Signature: (center_row, center_col, height, width, angle_deg) -> None
        self.on_mode_changed: Optional[Callable] = None

        # Canvas items (IDs for update/delete)
        self._img_item: Optional[int] = None
        self._roi_items: dict = {}     # polygon, handles, center_cross, direction_arrow
        self._overlay_items: list = []  # result overlay items

        # Build UI
        self._build_ui(bg)

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _build_ui(self, bg: str):
        # Canvas
        self._canvas = tk.Canvas(
            self,
            width=self._width,
            height=self._height,
            bg=bg,
            highlightthickness=0,
            cursor="crosshair",
        )
        self._canvas.pack(fill=tk.BOTH, expand=True)

        # Bindings
        self._canvas.bind("<Button-1>", self._on_mouse_down)
        self._canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        self._canvas.bind("<Double-Button-1>", self._on_double_click)
        self._canvas.bind("<Button-4>", self._on_scroll_up)     # Linux scroll up
        self._canvas.bind("<Button-5>", self._on_scroll_down)   # Linux scroll down
        self._canvas.bind("<MouseWheel>", self._on_mousewheel)  # Windows/Mac
        self._canvas.bind("<Configure>", self._on_resize)
        self._canvas.bind("<Motion>", self._on_mouse_move)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_image(self, image: np.ndarray):
        """Load a new image (BGR uint8 or grayscale uint8)."""
        self._image = image
        self._zoom = self._compute_fit_zoom()
        self._offset_x = 0.0
        self._offset_y = 0.0
        self._redraw()
        self._update_title()

    def set_mode(self, mode: CanvasMode):
        """Switch interaction mode."""
        old = self._mode
        self._mode = mode

        if mode == CanvasMode.DRAW_ROI:
            self._canvas.config(cursor="crosshair")
            self._drawing_phase = 0
        elif mode == CanvasMode.BROWSE:
            self._canvas.config(cursor="")
        elif mode == CanvasMode.VIEW_RESULT:
            self._canvas.config(cursor="")

        if old != mode and self.on_mode_changed:
            self.on_mode_changed(mode)

    def get_mode(self) -> CanvasMode:
        return self._mode

    @property
    def image(self) -> Optional[np.ndarray]:
        return self._image

    @property
    def roi_center(self) -> Optional[Tuple[float, float]]:
        return self._roi_center

    @property
    def roi_size(self) -> Tuple[float, float]:
        return self._roi_size

    @property
    def roi_angle(self) -> float:
        return self._roi_angle

    @property
    def roi_confirmed(self) -> bool:
        return self._roi_confirmed

    def set_roi(self, center: Tuple[float, float], size: Tuple[float, float], angle_deg: float):
        """Programmatically set the ROI."""
        self._roi_center = center
        self._roi_size = size
        self._roi_angle = angle_deg
        self._roi_confirmed = False
        self._drawing_phase = 0  # reset to allow re-draw
        self._redraw_roi()
        self._notify_roi_changed()

    def confirm_roi(self):
        """Confirm the current ROI (locks it)."""
        if self._roi_center is not None:
            self._roi_confirmed = True
            self._redraw_roi()
            if self.on_roi_confirmed:
                self.on_roi_confirmed(
                    self._roi_center[0], self._roi_center[1],
                    self._roi_size[0], self._roi_size[1],
                    self._roi_angle,
                )
            self.set_mode(CanvasMode.BROWSE)

    def reset_roi(self):
        """Clear the ROI."""
        self._roi_center = None
        self._roi_size = (100.0, 100.0)
        self._roi_angle = 0.0
        self._roi_confirmed = False
        self._drawing_phase = 0
        self._clear_roi_items()
        self._notify_roi_changed()

    def get_view_state(self) -> dict:
        """Serialize zoom/pan state for project saving.

        Returns:
            dict with keys: zoom (float), offset_x (float), offset_y (float).
        """
        return {
            "zoom": self._zoom,
            "offset_x": self._offset_x,
            "offset_y": self._offset_y,
        }

    def set_view_state(self, state: dict):
        """Restore zoom/pan state from a saved project.

        Args:
            state: dict with zoom, offset_x, offset_y keys.
        """
        self._zoom = state.get("zoom", 1.0)
        self._offset_x = state.get("offset_x", 0.0)
        self._offset_y = state.get("offset_y", 0.0)
        self._redraw()

    def clear_overlays(self):
        """Remove all result overlay items."""
        for item_id in self._overlay_items:
            self._canvas.delete(item_id)
        self._overlay_items = []

    def draw_overlay_polygon(self, corners: np.ndarray, color: str = "lime",
                             width: int = 2, tag: str = ""):
        """Draw a polygon overlay (corners in image coords as (row, col))."""
        pts_flat = []
        for r, c in corners:
            cx, cy = self._img_to_canvas(r, c)
            pts_flat.extend([cx, cy])
        item = self._canvas.create_polygon(
            pts_flat, outline=color, fill="", width=width, tags=tag,
        )
        self._overlay_items.append(item)
        return item

    def draw_overlay_cross(self, row: float, col: float, color: str = "lime",
                           size: int = 8, width: int = 1, tag: str = ""):
        """Draw a cross marker at image position (row, col)."""
        cx, cy = self._img_to_canvas(row, col)
        item1 = self._canvas.create_line(
            cx - size, cy, cx + size, cy, fill=color, width=width, tags=tag,
        )
        item2 = self._canvas.create_line(
            cx, cy - size, cx, cy + size, fill=color, width=width, tags=tag,
        )
        self._overlay_items.extend([item1, item2])
        return [item1, item2]

    def draw_overlay_text(self, row: float, col: float, text: str,
                          color: str = "lime", tag: str = ""):
        """Draw text at image position (row, col)."""
        cx, cy = self._img_to_canvas(row, col)
        item = self._canvas.create_text(
            cx + 8, cy - 8, text=text, fill=color,
            anchor=tk.NW, font=("TkDefaultFont", 9), tags=tag,
        )
        self._overlay_items.append(item)
        return item

    def zoom_to_fit(self):
        """Zoom so the entire image fits in the canvas."""
        self._zoom = self._compute_fit_zoom()
        self._offset_x = 0.0
        self._offset_y = 0.0
        self._redraw()

    def zoom_to_100(self):
        """Zoom to 100% (1:1 pixel mapping)."""
        self._zoom = 1.0
        # Center the image
        if self._image is not None:
            h, w = self._image.shape[:2]
            canvas_w = self._canvas.winfo_width()
            canvas_h = self._canvas.winfo_height()
            self._offset_x = (canvas_w - w) / 2
            self._offset_y = (canvas_h - h) / 2
        self._redraw()

    # ------------------------------------------------------------------
    # Coordinate transforms
    # ------------------------------------------------------------------

    def _img_to_canvas(self, row: float, col: float) -> Tuple[float, float]:
        """Convert image (row, col) to canvas (x, y)."""
        return (col * self._zoom + self._offset_x,
                row * self._zoom + self._offset_y)

    def _canvas_to_img(self, cx: float, cy: float) -> Tuple[float, float]:
        """Convert canvas (x, y) to image (row, col)."""
        return ((cy - self._offset_y) / self._zoom,
                (cx - self._offset_x) / self._zoom)

    def _compute_fit_zoom(self) -> float:
        """Compute zoom to fit the entire image."""
        if self._image is None:
            return 1.0
        h, w = self._image.shape[:2]
        canvas_w = max(self._canvas.winfo_width(), 100)
        canvas_h = max(self._canvas.winfo_height(), 100)
        return min(canvas_w / w, canvas_h / h) * 0.92

    # ------------------------------------------------------------------
    # Redraw
    # ------------------------------------------------------------------

    def _redraw(self):
        """Full redraw: image + ROI."""
        if self._image is None:
            return

        # Scale image for display
        h, w = self._image.shape[:2]
        new_w = max(1, int(w * self._zoom))
        new_h = max(1, int(h * self._zoom))

        if new_w > 0 and new_h > 0:
            self._display_image = cv2.resize(self._image, (new_w, new_h),
                                            interpolation=cv2.INTER_NEAREST)

        # Convert to PhotoImage
        self._photo = cv2_to_tk(self._display_image)

        # Update or create image item
        if self._img_item is None:
            self._img_item = self._canvas.create_image(
                self._offset_x, self._offset_y,
                anchor=tk.NW, image=self._photo,
            )
        else:
            self._canvas.coords(self._img_item, self._offset_x, self._offset_y)
            self._canvas.itemconfig(self._img_item, image=self._photo)

        # Redraw ROI
        if self._roi_center is not None:
            self._redraw_roi()

        # Ensure image is below ROI and overlays
        self._canvas.tag_lower(self._img_item)

    def _redraw_roi(self):
        """Redraw the ROI overlay."""
        if self._roi_center is None:
            return

        self._clear_roi_items()

        row, col = self._roi_center
        h, w = self._roi_size
        angle = self._roi_angle

        # Set color based on state
        if self._roi_confirmed:
            color = "cyan"
            handle_fill = "cyan"
        elif self._mode == CanvasMode.DRAW_ROI:
            color = "lime"
            handle_fill = "lime"
        else:
            color = "#888888"
            handle_fill = "#888888"

        # Compute corners in image coords
        # compute_rotated_box_corners uses CCW+, but _roi_angle is CW+
        # (from arctan2 in image coords). Negate to make box visual match mouse.
        corners_img = compute_rotated_box_corners((row, col), (h, w), -angle)

        # Convert to canvas coords
        corners_canvas = []
        for r, c in corners_img:
            cx, cy = self._img_to_canvas(r, c)
            corners_canvas.extend([cx, cy])

        # Draw rotated rectangle
        poly_id = self._canvas.create_polygon(
            corners_canvas, outline=color, fill="", width=2, tags="roi",
        )
        self._roi_items["polygon"] = poly_id

        # Center cross
        cx_c, cy_c = self._img_to_canvas(row, col)
        cs = 8
        cross_id = self._canvas.create_line(
            cx_c - cs, cy_c, cx_c + cs, cy_c,
            cx_c, cy_c - cs, cx_c, cy_c + cs,
            fill=color, width=1, tags="roi",
        )
        self._roi_items["center_cross"] = cross_id

        # Corner handles
        handle_ids = []
        for i, (r, c) in enumerate(corners_img):
            hx, hy = self._img_to_canvas(r, c)
            r_val = self.HANDLE_RADIUS
            hid = self._canvas.create_oval(
                hx - r_val, hy - r_val, hx + r_val, hy + r_val,
                outline=color, fill=handle_fill, tags=("roi", f"handle_{i}"),
            )
            handle_ids.append(hid)
        self._roi_items["corner_handles"] = handle_ids

        # Rotation handle (top edge midpoint, extended outward)
        if not self._roi_confirmed:
            # Top edge in image coords: between corners_img[0] and corners_img[1]
            r_mid = (corners_img[0][0] + corners_img[1][0]) / 2
            c_mid = (corners_img[0][1] + corners_img[1][1]) / 2
            # Direction outward (perpendicular to top edge, away from center)
            dr = r_mid - row
            dc = c_mid - col
            dist = np.sqrt(dr**2 + dc**2)
            if dist > 0:
                dr /= dist
                dc /= dist
            r_handle_img = r_mid + dr * (self.ROTATE_HANDLE_OFFSET / max(self._zoom, 0.1))
            c_handle_img = c_mid + dc * (self.ROTATE_HANDLE_OFFSET / max(self._zoom, 0.1))

            hx, hy = self._img_to_canvas(r_handle_img, c_handle_img)
            mx, my = self._img_to_canvas(r_mid, c_mid)

            # Line from midpoint to handle
            line_id = self._canvas.create_line(
                mx, my, hx, hy, fill=color, width=1, tags="roi",
            )
            self._roi_items["rotate_line"] = line_id

            # Handle circle
            hr_val = self.HANDLE_RADIUS + 2
            rh_id = self._canvas.create_oval(
                hx - hr_val, hy - hr_val, hx + hr_val, hy + hr_val,
                outline="white", fill=color, tags=("roi", "handle_rotate"),
            )
            self._roi_items["rotate_handle"] = rh_id

    def _clear_roi_items(self):
        """Remove all ROI canvas items."""
        for item_id in self._roi_items.values():
            if isinstance(item_id, list):
                for i in item_id:
                    self._canvas.delete(i)
            else:
                self._canvas.delete(item_id)
        self._roi_items = {}

    def _update_title(self):
        """Update the window title for standalone use."""
        # No-op when embedded; used when ImageCanvas is the root window
        pass

    # ------------------------------------------------------------------
    # Mouse event handlers
    # ------------------------------------------------------------------

    def _on_mouse_down(self, event):
        """Handle mouse button press."""
        if self._mode == CanvasMode.BROWSE:
            self._drag_start = (event.x, event.y)
            self._drag_what = "pan"

        elif self._mode == CanvasMode.DRAW_ROI:
            # Check if clicking on a handle
            handle = self._hit_test_handle(event.x, event.y)
            if handle is not None and self._roi_center is not None:
                self._drag_start = (event.x, event.y)
                self._drag_what = handle
                return

            # Check if clicking inside the ROI box
            if self._is_inside_roi(event.x, event.y):
                self._drag_start = (event.x, event.y)
                self._drag_what = "move"
                return

            # Start or restart ROI drawing
            if self._drawing_phase == 0 or self._roi_center is None:
                # Set center
                row, col = self._canvas_to_img(event.x, event.y)
                if self._image is not None:
                    h, w = self._image.shape[:2]
                    row = max(0, min(row, h - 1))
                    col = max(0, min(col, w - 1))
                self._roi_center = (row, col)
                self._roi_size = (10.0, 10.0)
                self._roi_angle = 0.0
                self._roi_confirmed = False
                self._drawing_phase = 1
                self._drag_start = (event.x, event.y)
                self._drag_what = "resize_diag"
                self._notify_roi_changed()

            elif self._drawing_phase == 1:
                # Already have center+size, click to start new ROI
                row, col = self._canvas_to_img(event.x, event.y)
                if self._image is not None:
                    h, w = self._image.shape[:2]
                    row = max(0, min(row, h - 1))
                    col = max(0, min(col, w - 1))
                self._roi_center = (row, col)
                self._roi_size = (10.0, 10.0)
                self._roi_angle = 0.0
                self._roi_confirmed = False
                self._drag_start = (event.x, event.y)
                self._drag_what = "resize_diag"
                self._notify_roi_changed()

    def _on_mouse_drag(self, event):
        """Handle mouse drag."""
        if self._drag_start is None:
            return

        dx = event.x - self._drag_start[0]
        dy = event.y - self._drag_start[1]

        if self._drag_what == "pan":
            self._offset_x += dx
            self._offset_y += dy
            self._drag_start = (event.x, event.y)
            self._redraw()

        elif self._drag_what == "move" and self._roi_center is not None:
            row, col = self._canvas_to_img(event.x, event.y)
            self._roi_center = (row, col)
            self._drag_start = (event.x, event.y)
            self._redraw_roi()
            self._notify_roi_changed()

        elif self._drag_what == "resize_diag" and self._roi_center is not None:
            # Symmetric resize from center
            row, col = self._canvas_to_img(event.x, event.y)
            cy, cx = self._roi_center
            new_h = abs(row - cy) * 2
            new_w = abs(col - cx) * 2
            self._roi_size = (max(5.0, new_h), max(5.0, new_w))
            self._redraw_roi()
            self._notify_roi_changed()

        elif self._drag_what and self._drag_what.startswith("handle_") and self._roi_center is not None:
            if self._drag_what == "handle_rotate":
                # Rotate: angle from center to current mouse position
                cy, cx = self._roi_center
                row, col = self._canvas_to_img(event.x, event.y)
                # arctan2 gives +angle for CW in image coords (row↓).
                # compute_rotated_box_corners expects CCW+, so we negate there.
                # crop_and_straighten expects CCW+ and matches this arctan2
                # direction convention.
                angle = np.rad2deg(np.arctan2(col - cx, -(row - cy)))
                self._roi_angle = angle
                self._redraw_roi()
                self._notify_roi_changed()
            elif self._drag_what.startswith("handle_"):
                # Corner resize - resize the appropriate corner
                row, col = self._canvas_to_img(event.x, event.y)
                cy, cx = self._roi_center
                self._roi_size = (
                    max(5.0, abs(row - cy) * 2),
                    max(5.0, abs(col - cx) * 2),
                )
                self._redraw_roi()
                self._notify_roi_changed()

        elif self._drag_what == "rotate" and self._roi_center is not None:
            # Rotate via mouse direction from center
            cy, cx = self._roi_center
            row, col = self._canvas_to_img(event.x, event.y)
            angle = np.rad2deg(np.arctan2(col - cx, -(row - cy)))
            self._roi_angle = angle
            self._redraw_roi()
            self._notify_roi_changed()

    def _on_mouse_up(self, event):
        """Handle mouse button release."""
        if self._drag_what and self._drag_what != "pan":
            # Finished drawing/editing ROI
            pass
        self._drag_start = None
        self._drag_what = None

    def _on_double_click(self, event):
        """Double click: confirm ROI or zoom to fit."""
        if self._mode == CanvasMode.DRAW_ROI and self._roi_center is not None:
            self.confirm_roi()
        elif self._mode == CanvasMode.BROWSE:
            self.zoom_to_fit()

    def _on_scroll_up(self, event):
        self._zoom_at(event.x, event.y, self.ZOOM_STEP)

    def _on_scroll_down(self, event):
        self._zoom_at(event.x, event.y, 1.0 / self.ZOOM_STEP)

    def _on_mousewheel(self, event):
        # Windows/Mac mousewheel
        if event.delta > 0:
            self._zoom_at(event.x, event.y, self.ZOOM_STEP)
        else:
            self._zoom_at(event.x, event.y, 1.0 / self.ZOOM_STEP)

    def _on_mouse_move(self, event):
        """Track mouse position for status bar."""
        if self._image is not None:
            row, col = self._canvas_to_img(event.x, event.y)
            h, w = self._image.shape[:2]
            if 0 <= row < h and 0 <= col < w:
                # Could emit signal here for status bar
                pass

    def _on_resize(self, event):
        """Canvas resized."""
        pass

    # ------------------------------------------------------------------
    # Zoom
    # ------------------------------------------------------------------

    def _zoom_at(self, canvas_x: float, canvas_y: float, factor: float):
        """Zoom centered at a canvas position (debounced).

        The zoom transform (offset + zoom) is accumulated on each scroll event,
        but the expensive _redraw() (cv2.resize + cv2_to_tk) is deferred via
        tkinter ``after`` so that rapid mouse-wheel bursts produce only one
        redraw every ZOOM_DEBOUNCE_MS milliseconds.
        """
        new_zoom = self._zoom * self._accumulated_factor * factor
        if new_zoom < self.MIN_ZOOM or new_zoom > self.MAX_ZOOM:
            return

        # Accumulate zoom factor
        self._accumulated_factor *= factor

        # Save mouse position
        self._zoom_cursor_x = canvas_x
        self._zoom_cursor_y = canvas_y

        # Cancel previous pending redraw
        if self._zoom_after_id is not None:
            self.after_cancel(self._zoom_after_id)

        # Schedule debounced redraw
        self._zoom_after_id = self.after(self.ZOOM_DEBOUNCE_MS, self._apply_accumulated_zoom)

    def _apply_accumulated_zoom(self):
        """Apply accumulated zoom transforms and redraw once."""
        self._zoom_after_id = None

        factor = self._accumulated_factor
        if factor == 1.0:
            return
        self._accumulated_factor = 1.0

        # Check final zoom is in bounds
        new_zoom = self._zoom * factor
        if new_zoom < self.MIN_ZOOM:
            factor = self.MIN_ZOOM / self._zoom
        elif new_zoom > self.MAX_ZOOM:
            factor = self.MAX_ZOOM / self._zoom

        canvas_x = self._zoom_cursor_x
        canvas_y = self._zoom_cursor_y

        # Adjust offset to keep the point under cursor fixed
        self._offset_x = canvas_x - (canvas_x - self._offset_x) * factor
        self._offset_y = canvas_y - (canvas_y - self._offset_y) * factor
        self._zoom *= factor

        # Do the expensive redraw once
        self._redraw()

    # ------------------------------------------------------------------
    # Hit testing
    # ------------------------------------------------------------------

    def _hit_test_handle(self, cx: float, cy: float) -> Optional[str]:
        """Check if canvas point (cx, cy) is on any ROI handle."""
        if self._roi_center is None:
            return None

        # Check rotation handle
        if "rotate_handle" in self._roi_items:
            coords = self._canvas.coords(self._roi_items["rotate_handle"])
            if coords and len(coords) == 4:
                hx = (coords[0] + coords[2]) / 2
                hy = (coords[1] + coords[3]) / 2
                if abs(cx - hx) < self.HANDLE_RADIUS + 4 and abs(cy - hy) < self.HANDLE_RADIUS + 4:
                    return "handle_rotate"

        # Check corner handles
        corners_img = compute_rotated_box_corners(
            self._roi_center, self._roi_size, -self._roi_angle
        )
        for i, (r, c) in enumerate(corners_img):
            hx, hy = self._img_to_canvas(r, c)
            if abs(cx - hx) < self.HANDLE_RADIUS + 2 and abs(cy - hy) < self.HANDLE_RADIUS + 2:
                return f"handle_{i}"

        return None

    def _is_inside_roi(self, cx: float, cy: float) -> bool:
        """Check if canvas point is inside the ROI polygon."""
        if self._roi_center is None or "polygon" not in self._roi_items:
            return False

        # Create a small 1x1 rectangle and check overlap with polygon
        try:
            overlap = self._canvas.find_overlapping(cx - 1, cy - 1, cx + 1, cy + 1)
            poly_id = self._roi_items["polygon"]
            return poly_id in overlap
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _notify_roi_changed(self):
        """Fire the ROI changed callback."""
        if self.on_roi_changed and self._roi_center is not None:
            self.on_roi_changed(
                self._roi_center[0], self._roi_center[1],
                self._roi_size[0], self._roi_size[1],
                self._roi_angle,
            )
