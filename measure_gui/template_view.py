"""
Template preview view with interactive measurement tool placement.

Displays the straightened template image and allows users to:
  - Click/drag to place measurement tools (edge point, edge pair, fit line, fit circle)
  - Edit tool parameters via dialogs
  - See all placed tools overlaid on the template

Tools are defined in the straightened template coordinate space (origin at top-left).
"""

from __future__ import annotations

import math
import tkinter as tk
from enum import Enum, auto
from tkinter import ttk
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .dialogs import (
    ComposedMeasureDialog,
    EdgePairDialog,
    EdgePointDialog,
    FitCircleDialog,
    FitLineDialog,
    TemplateMatchPointDialog,
)
from .utils import cv2_to_tk


class TemplateTool(Enum):
    """Tools that can be interactively placed on the template."""
    SELECT = auto()              # Select/edit existing tools
    EDGE_POINT = auto()          # Place an edge point probe
    EDGE_PAIR = auto()           # Place an edge pair probe
    FIT_LINE = auto()            # Place a line fitting ROI
    FIT_CIRCLE = auto()          # Place a circle fitting ROI
    TEMPLATE_MATCH_POINT = auto()  # Place a template-matching point


class TemplateView(tk.Frame):
    """
    Template preview with interactive measurement tool placement.

    Usage:
        view = TemplateView(parent)
        view.load_template(template_image)
        view.set_tool(TemplateTool.EDGE_POINT)
        view.on_tool_added = lambda tool_type, label, params: ...
    """

    # Zoom constants (similar to ImageCanvas)
    MIN_ZOOM = 0.1
    MAX_ZOOM = 10.0
    ZOOM_STEP = 1.1
    ZOOM_DEBOUNCE_MS = 80      # ms between zoom redraws (prevents UI freeze from rapid scrolling)

    def __init__(self, parent: tk.Widget, width: int = 350, height: int = 350,
                 bg: str = "#1e1e1e", **kwargs):
        super().__init__(parent, **kwargs)

        self._width = width
        self._height = height
        self._template_image: Optional[np.ndarray] = None
        self._display_image: Optional[np.ndarray] = None

        # Current tool
        self._current_tool: TemplateTool = TemplateTool.SELECT

        # Placed measurement tools: list of {label, type, params, canvas_items}
        self._tools: List[Dict[str, Any]] = []

        # Interaction state
        self._drawing: bool = False
        self._draw_start: Optional[Tuple[float, float]] = None  # (row, col) in template coords

        # Zoom state (similar to ImageCanvas)
        # _base_scale: auto-fit scale (no zoom applied)
        # _zoom_factor: user zoom multiplier
        # Actual display scale = _base_scale * _zoom_factor
        self._scale: float = 1.0
        self._base_scale: Optional[float] = None
        self._zoom_factor: float = 1.0
        self._offset_x: float = 0.0
        self._offset_y: float = 0.0

        # Pan state (SELECT mode: click empty space to pan)
        self._panning: bool = False

        # Zoom debounce (prevents UI freeze from rapid scrolling)
        self._zoom_after_id: Optional[str] = None  # tkinter after ID
        self._accumulated_factor: float = 1.0
        self._zoom_cursor_x: float = 0.0
        self._zoom_cursor_y: float = 0.0

        # Callbacks
        self.on_tool_added: Optional[Callable] = None
        # (object_type: str, label: str, params: dict) -> None
        self.on_tool_removed: Optional[Callable] = None
        # (label: str) -> None
        self.on_tool_edited: Optional[Callable] = None
        # (label: str, params: dict) -> None

        # Tool count for auto-labeling
        self._tool_counters: Dict[TemplateTool, int] = {}

        # Build UI
        self._build_ui(bg)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self, bg: str):
        # Toolbar
        toolbar = ttk.Frame(self)
        toolbar.pack(fill=tk.X, padx=2, pady=2)

        self._btn_select = ttk.Button(toolbar, text="选择",
                                      command=lambda: self.set_tool(TemplateTool.SELECT))
        self._btn_select.pack(side=tk.LEFT, padx=1)

        self._btn_edge_pt = ttk.Button(toolbar, text="+边缘点",
                                       command=lambda: self.set_tool(TemplateTool.EDGE_POINT))
        self._btn_edge_pt.pack(side=tk.LEFT, padx=1)

        self._btn_edge_pair = ttk.Button(toolbar, text="+边缘对",
                                         command=lambda: self.set_tool(TemplateTool.EDGE_PAIR))
        self._btn_edge_pair.pack(side=tk.LEFT, padx=1)

        self._btn_fit_line = ttk.Button(toolbar, text="+拟合直线",
                                        command=lambda: self.set_tool(TemplateTool.FIT_LINE))
        self._btn_fit_line.pack(side=tk.LEFT, padx=1)

        self._btn_fit_circle = ttk.Button(toolbar, text="+拟合圆",
                                          command=lambda: self.set_tool(TemplateTool.FIT_CIRCLE))
        self._btn_fit_circle.pack(side=tk.LEFT, padx=1)

        self._btn_tmpl_match = ttk.Button(
            toolbar, text="+模板匹配点",
            command=lambda: self.set_tool(TemplateTool.TEMPLATE_MATCH_POINT))
        self._btn_tmpl_match.pack(side=tk.LEFT, padx=1)

        self._btn_delete = ttk.Button(toolbar, text="删除", command=self._delete_selected)
        self._btn_delete.pack(side=tk.LEFT, padx=1)

        # Tool indicator
        self._tool_label = ttk.Label(toolbar, text="选择", foreground="gray")
        self._tool_label.pack(side=tk.RIGHT, padx=5)

        # Canvas
        canvas_frame = ttk.Frame(self)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        self._canvas = tk.Canvas(
            canvas_frame, width=self._width, height=self._height,
            bg=bg, highlightthickness=1, highlightbackground="#555",
            cursor="crosshair",
        )
        self._canvas.pack(fill=tk.BOTH, expand=True)

        # Bindings
        self._canvas.bind("<Button-1>", self._on_click)
        self._canvas.bind("<Double-Button-1>", self._on_double_click)
        self._canvas.bind("<B1-Motion>", self._on_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_release)
        self._canvas.bind("<Motion>", self._on_motion)
        self._canvas.bind("<Button-4>", self._on_scroll_up)     # Linux scroll up
        self._canvas.bind("<Button-5>", self._on_scroll_down)   # Linux scroll down
        self._canvas.bind("<MouseWheel>", self._on_mousewheel)  # Windows/Mac

        # Status bar
        self._status = ttk.Label(self, text="加载模板后点击放置测量工具", foreground="gray")
        self._status.pack(fill=tk.X, padx=5)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_template(self, image: np.ndarray, clear_tools: bool = True):
        """Load a straightened template image.

        Args:
            image: Grayscale or BGR template image.
            clear_tools: If True (default), reset all tools and counters.
                         Set to False when restoring from a saved project
                         (tools will be restored separately via set_state).
        """
        self._template_image = image
        if clear_tools:
            self._tools = []
            self._tool_counters = {}
            # Reset zoom state so _redraw recalculates base_scale
            self._zoom_factor = 1.0
            self._base_scale = None
        self._redraw()

    @property
    def template_image(self) -> Optional[np.ndarray]:
        return self._template_image

    def set_tool(self, tool: TemplateTool):
        """Switch the active placement tool."""
        self._current_tool = tool
        self._drawing = False

        names = {
            TemplateTool.SELECT: "选择",
            TemplateTool.EDGE_POINT: "边缘点",
            TemplateTool.EDGE_PAIR: "边缘对",
            TemplateTool.FIT_LINE: "拟合直线",
            TemplateTool.FIT_CIRCLE: "拟合圆",
            TemplateTool.TEMPLATE_MATCH_POINT: "模板匹配点",
        }
        self._tool_label.config(text=names.get(tool, ""))

        # Highlight active button
        for btn, t in [
            (self._btn_select, TemplateTool.SELECT),
            (self._btn_edge_pt, TemplateTool.EDGE_POINT),
            (self._btn_edge_pair, TemplateTool.EDGE_PAIR),
            (self._btn_fit_line, TemplateTool.FIT_LINE),
            (self._btn_fit_circle, TemplateTool.FIT_CIRCLE),
            (self._btn_tmpl_match, TemplateTool.TEMPLATE_MATCH_POINT),
        ]:
            if t == tool:
                btn.state(["pressed"])
            else:
                btn.state(["!pressed"])

    def get_tools(self) -> List[Dict[str, Any]]:
        """Return all placed measurement tools."""
        return [
            {"object_type": t["object_type"], "label": t["label"], "params": t["params"]}
            for t in self._tools
        ]

    def remove_tool(self, label: str):
        """Remove a tool by label."""
        for i, t in enumerate(self._tools):
            if t["label"] == label:
                self._delete_tool_items(t)
                self._tools.pop(i)
                break
        self._redraw_tools()
        if self.on_tool_removed:
            self.on_tool_removed(label)

    def clear_tools(self):
        """Remove all tools."""
        for t in self._tools:
            self._delete_tool_items(t)
        self._tools = []
        self._tool_counters = {}
        self._redraw_tools()

    def get_state(self) -> dict:
        """Serialize complete template view state for project saving.

        Returns a dict with:
            scale, base_scale, zoom_factor, offset_x, offset_y — display transform
            tools — list of {object_type, label, params, _selected}
            tool_counters — dict[str, int] auto-label counters
        """
        tools_data = []
        for t in self._tools:
            tools_data.append({
                "object_type": t["object_type"],
                "label": t["label"],
                "params": dict(t["params"]),  # shallow copy; values are JSON-serializable
                "_selected": t.get("_selected", False),
            })
        return {
            "scale": self._scale,
            "base_scale": self._base_scale,
            "zoom_factor": self._zoom_factor,
            "offset_x": self._offset_x,
            "offset_y": self._offset_y,
            "tools": tools_data,
            "tool_counters": dict(self._tool_counters),
        }

    def set_state(self, state: dict) -> None:
        """Restore template view state from a saved project.

        Recreates all canvas tool overlays from the serialized state.
        Does NOT fire on_tool_added callbacks — the workflow already has
        the measurement definitions from the .npz load.
        """
        # Restore display params
        self._scale = state.get("scale", 1.0)
        self._base_scale = state.get("base_scale", None)
        self._zoom_factor = state.get("zoom_factor", 1.0)
        self._offset_x = state.get("offset_x", 0.0)
        self._offset_y = state.get("offset_y", 0.0)

        # Restore tool counters
        self._tool_counters = {}
        for prefix, count in state.get("tool_counters", {}).items():
            self._tool_counters[prefix] = count

        # Restore tools list (rebuilding canvas items via _redraw_tools)
        self._tools = []
        for tool_data in state.get("tools", []):
            tool = {
                "object_type": tool_data["object_type"],
                "label": tool_data["label"],
                "params": dict(tool_data["params"]),
                "canvas_items": [],
                "_selected": tool_data.get("_selected", False),
            }
            self._tools.append(tool)

        # Redraw all canvas content (image + tool overlays) at restored scale
        self._redraw()

    def set_status(self, text: str):
        self._status.config(text=text)

    # ------------------------------------------------------------------
    # Template coordinate transforms (zoom-aware)
    # ------------------------------------------------------------------

    def _tmpl_to_canvas(self, row: float, col: float) -> Tuple[float, float]:
        """Template coords to canvas coords."""
        return (col * self._scale + self._offset_x,
                row * self._scale + self._offset_y)

    def _canvas_to_tmpl(self, cx: float, cy: float) -> Tuple[float, float]:
        """Canvas coords to template coords."""
        return ((cy - self._offset_y) / self._scale,
                (cx - self._offset_x) / self._scale)

    # ------------------------------------------------------------------
    # Zoom (similar to ImageCanvas)
    # ------------------------------------------------------------------

    def _zoom_at(self, canvas_x: float, canvas_y: float, factor: float):
        """Zoom centered at a canvas position (debounced).

        The zoom transform (offset + zoom_factor) is accumulated on each scroll
        event, but the expensive _redraw() (cv2.resize + cv2_to_tk) is
        deferred so that rapid mouse-wheel bursts produce only one redraw
        every ZOOM_DEBOUNCE_MS milliseconds.
        """
        new_zoom = self._zoom_factor * self._accumulated_factor * factor
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

        # Clamp accumulated zoom factor to bounds
        new_zoom = self._zoom_factor * factor
        if new_zoom < self.MIN_ZOOM:
            factor = self.MIN_ZOOM / self._zoom_factor
        elif new_zoom > self.MAX_ZOOM:
            factor = self.MAX_ZOOM / self._zoom_factor

        canvas_x = self._zoom_cursor_x
        canvas_y = self._zoom_cursor_y

        # Adjust offset to keep the point under cursor fixed
        self._offset_x = canvas_x - (canvas_x - self._offset_x) * factor
        self._offset_y = canvas_y - (canvas_y - self._offset_y) * factor
        self._zoom_factor *= factor

        # Do the expensive redraw once
        self._redraw()

    def _on_scroll_up(self, event):
        """Linux scroll up — zoom in."""
        if self._template_image is not None:
            self._zoom_at(event.x, event.y, self.ZOOM_STEP)

    def _on_scroll_down(self, event):
        """Linux scroll down — zoom out."""
        if self._template_image is not None:
            self._zoom_at(event.x, event.y, 1.0 / self.ZOOM_STEP)

    def _on_mousewheel(self, event):
        """Windows/Mac mousewheel — zoom in/out."""
        if self._template_image is not None:
            if event.delta > 0:
                self._zoom_at(event.x, event.y, self.ZOOM_STEP)
            else:
                self._zoom_at(event.x, event.y, 1.0 / self.ZOOM_STEP)

    def zoom_to_fit(self):
        """Reset to auto-fit zoom (zoom_factor = 1.0)."""
        self._zoom_factor = 1.0
        if self._template_image is not None:
            h, w = self._template_image.shape[:2]
            canvas_w = self._canvas.winfo_width()
            canvas_h = self._canvas.winfo_height()
            if canvas_w <= 1:
                canvas_w = self._width
            if canvas_h <= 1:
                canvas_h = self._height
            self._base_scale = min(canvas_w / w, canvas_h / h) * 0.95
            self._offset_x = (canvas_w - w * self._base_scale) / 2
            self._offset_y = (canvas_h - h * self._base_scale) / 2
        self._redraw()

    def zoom_to_100(self):
        """Zoom to 1:1 (template pixels = screen pixels)."""
        if self._base_scale and self._base_scale > 0:
            self._zoom_factor = 1.0 / self._base_scale
        else:
            self._zoom_factor = 1.0
        self._redraw()

    # ------------------------------------------------------------------
    # Redraw
    # ------------------------------------------------------------------

    def _redraw(self):
        """Full redraw of template image."""
        self._canvas.delete("all")

        if self._template_image is None:
            return

        h, w = self._template_image.shape[:2]
        canvas_w = self._canvas.winfo_width()
        canvas_h = self._canvas.winfo_height()
        # winfo_width/height returns 1 when the widget isn't mapped yet —
        # 1 is truthy so a plain `or` fallback never fires.  Use > 1.
        if canvas_w <= 1:
            canvas_w = self._width
        if canvas_h <= 1:
            canvas_h = self._height

        # Calculate base (auto-fit) scale on first load or after zoom_to_fit
        if self._base_scale is None:
            self._base_scale = min(canvas_w / w, canvas_h / h) * 0.95
            self._offset_x = (canvas_w - w * self._base_scale) / 2
            self._offset_y = (canvas_h - h * self._base_scale) / 2

        # Actual display scale = base_scale * zoom_factor
        self._scale = self._base_scale * self._zoom_factor

        # Resize for display
        new_w = max(1, int(w * self._scale))
        new_h = max(1, int(h * self._scale))
        self._display_image = cv2.resize(self._template_image, (new_w, new_h),
                                        interpolation=cv2.INTER_NEAREST)

        # Convert and display
        if len(self._display_image.shape) == 2:
            disp = cv2.cvtColor(self._display_image, cv2.COLOR_GRAY2BGR)
        else:
            disp = self._display_image
        self._photo = cv2_to_tk(disp)
        self._canvas.create_image(self._offset_x, self._offset_y,
                                  anchor=tk.NW, image=self._photo)

        # Redraw tools
        self._redraw_tools()

    def _redraw_tools(self):
        """Redraw all measurement tool overlays."""
        # Delete existing tool items
        for t in self._tools:
            self._delete_tool_items(t)

        # Redraw
        for i, t in enumerate(self._tools):
            t["canvas_items"] = self._draw_tool_overlay(t, i)

    def _draw_tool_overlay(self, tool: Dict[str, Any], index: int) -> List[int]:
        """Draw a single tool overlay. Returns list of canvas item IDs."""
        items = []
        obj_type = tool["object_type"]
        params = tool["params"]
        color = "lime"
        highlight = "yellow" if tool.get("_selected") else color

        if obj_type == "EdgePoint":
            items = self._draw_edge_probe(params, highlight, params.get("transition", "all"))
        elif obj_type == "EdgePair":
            items = self._draw_edge_probe(params, highlight, params.get("transition", "negative"))
        elif obj_type == "FitLine":
            items = self._draw_fit_line_overlay(params, highlight)
        elif obj_type == "FitCircle":
            items = self._draw_fit_circle_overlay(params, highlight)
        elif obj_type == "TemplateMatchPoint":
            items = self._draw_template_match_overlay(params, highlight)

        # Label
        row = params.get("row", 0)
        col = params.get("col", 0)
        if obj_type == "FitLine":
            s = params.get("start", (0, 0))
            row, col = s[0], s[1]
        elif obj_type == "FitCircle":
            c = params.get("center", (0, 0))
            row, col = c[0], c[1]

        cx, cy = self._tmpl_to_canvas(row, col)
        lbl = self._canvas.create_text(
            cx + 8, cy - 10, text=f"{index}:{tool['label']}",
            fill=highlight, anchor=tk.NW, font=("TkDefaultFont", 8),
        )
        items.append(lbl)

        # Store tool index on items for hit testing
        for item_id in items:
            self._canvas.addtag_withtag(f"tool_{index}", item_id)

        return items

    def _draw_edge_probe(self, params: dict, color: str, transition: str) -> List[int]:
        """Draw an edge probe (rectangle ROI)."""
        items = []
        row, col = params["row"], params["col"]
        angle = params["angle"]
        length1 = params["length1"]
        length2 = params["length2"]

        # Direction vector of probe
        dr = -np.cos(angle)  # row direction
        dc = np.sin(angle)   # col direction

        # Perpendicular direction
        pr = dc
        pc = dr

        # 4 corners of the probe rectangle
        corners = [
            (row - dr * length1 - pr * length2, col - dc * length1 - pc * length2),
            (row - dr * length1 + pr * length2, col - dc * length1 + pc * length2),
            (row + dr * length1 + pr * length2, col + dc * length1 + pc * length2),
            (row + dr * length1 - pr * length2, col + dc * length1 - pc * length2),
        ]

        pts = []
        for r, c in corners:
            cx, cy = self._tmpl_to_canvas(r, c)
            pts.extend([cx, cy])

        rect = self._canvas.create_polygon(pts, outline=color, fill="", width=1)
        items.append(rect)

        # Direction arrow
        mid_cx, mid_cy = self._tmpl_to_canvas(row, col)
        end_r = row + dr * length1 * 0.8
        end_c = col + dc * length1 * 0.8
        end_cx, end_cy = self._tmpl_to_canvas(end_r, end_c)
        arrow = self._canvas.create_line(mid_cx, mid_cy, end_cx, end_cy,
                                        fill=color, width=1, arrow=tk.LAST)
        items.append(arrow)

        return items

    def _draw_fit_line_overlay(self, params: dict, color: str) -> List[int]:
        """Draw a fit line ROI with perpendicular measurement probes.

        The fit line tool places ``num_measures`` equally-spaced measurement
        rectangles perpendicular to the line segment.  Each rectangle runs
        edge detection and the collected points are fitted to a line.  This
        overlay draws the main line, endpoint markers, and small probe ticks
        that represent those measurement rectangles so the user can see at a
        glance how the tool will measure.
        """
        items = []
        start = params["start"]
        end = params["end"]
        num_measures = params.get("num_measures", 10)
        measure_length2 = params.get("measure_length2", 25.0)  # half-width perpendicular to line

        sx, sy = self._tmpl_to_canvas(start[0], start[1])
        ex, ey = self._tmpl_to_canvas(end[0], end[1])

        # Main line (dashed)
        line = self._canvas.create_line(sx, sy, ex, ey, fill=color, width=2,
                                        dash=(4, 2))
        items.append(line)

        # Endpoint markers
        r = 3
        s_dot = self._canvas.create_oval(sx - r, sy - r, sx + r, sy + r,
                                         fill=color, outline="")
        e_dot = self._canvas.create_oval(ex - r, ey - r, ex + r, ey + r,
                                         fill=color, outline="")
        items.extend([s_dot, e_dot])

        # --- Perpendicular measurement probes ---
        # Each probe is a short line segment perpendicular to the main line,
        # centred at equally-spaced sample points.  It represents the
        # measurement rectangle that will scan for edge points.
        if num_measures >= 2:
            sr, sc = start[0], start[1]
            er, ec = end[0], end[1]
            # Direction vector of the main line (row, col)
            line_dr = er - sr
            line_dc = ec - sc
            line_len = np.sqrt(line_dr ** 2 + line_dc ** 2)
            if line_len > 1e-6:
                # Unit vector along the line
                u_r = line_dr / line_len
                u_c = line_dc / line_len
                # Unit perpendicular vector (rotate 90° clockwise)
                n_r = u_c
                n_c = -u_r

                # Scale probe half-length from template to canvas pixels
                probe_half = measure_length2 * self._scale

                # Draw probes with the main colour + stipple pattern so they
                # are always visible regardless of the underlying image content.
                for i in range(num_measures):
                    # Parameter t ∈ [0, 1] along the line
                    t = i / (num_measures - 1)
                    pr = sr + t * line_dr
                    pc = sc + t * line_dc
                    p_cx, p_cy = self._tmpl_to_canvas(pr, pc)

                    # Perpendicular endpoints
                    c1x = p_cx - n_c * probe_half
                    c1y = p_cy - n_r * probe_half
                    c2x = p_cx + n_c * probe_half
                    c2y = p_cy + n_r * probe_half

                    tick = self._canvas.create_line(
                        c1x, c1y, c2x, c2y,
                        fill=color, width=1, stipple="gray25")
                    items.append(tick)

        return items

    def _draw_fit_circle_overlay(self, params: dict, color: str) -> List[int]:
        """Draw a fit circle ROI with search boundaries and measurement rectangles."""
        items = []
        center = params["center"]
        radius = params["radius"]
        measure_length1 = params.get("measure_length1", 20.0)
        measure_length2 = params.get("measure_length2", 10.0)
        num_measures = params.get("num_measures", 12)

        # radius_min / radius_max auto-derived from measure_length1
        r_min = params.get("radius_min")
        if r_min is None:
            r_min = radius - measure_length1
        r_max = params.get("radius_max")
        if r_max is None:
            r_max = radius + measure_length1

        cx, cy = self._tmpl_to_canvas(center[0], center[1])

        # --- Measurement rectangles (blue outlines + arrows) ---
        if num_measures > 0:
            self._draw_circle_measure_rectangles(
                items, center, radius, measure_length1, measure_length2,
                num_measures, params.get("start_phi", 0.0),
                params.get("end_phi", 2 * math.pi),
            )

        # --- Search radius boundaries (dashed) ---
        if r_min is not None and r_min > 0:
            r_min_px = max(1, int(r_min * self._scale))
            circ_min = self._canvas.create_oval(
                cx - r_min_px, cy - r_min_px, cx + r_min_px, cy + r_min_px,
                outline="cyan", width=1, dash=(6, 4))
            items.append(circ_min)

        if r_max is not None and r_max > 0 and r_max > (r_min or 0):
            r_max_px = max(1, int(r_max * self._scale))
            circ_max = self._canvas.create_oval(
                cx - r_max_px, cy - r_max_px, cx + r_max_px, cy + r_max_px,
                outline="orange", width=1, dash=(6, 4))
            items.append(circ_max)

        # --- Expected circle (solid) ---
        r = max(1, int(radius * self._scale))
        circ = self._canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                        outline=color, width=1)
        items.append(circ)

        # Center cross
        cs = 5
        cross_h = self._canvas.create_line(cx - cs, cy, cx + cs, cy, fill=color)
        cross_v = self._canvas.create_line(cx, cy - cs, cx, cy + cs, fill=color)
        items.extend([cross_h, cross_v])

        return items

    def _draw_circle_measure_rectangles(
        self, items: list, center: tuple, radius: float,
        length1: float, length2: float, num: int,
        start_phi: float, end_phi: float,
    ):
        """Draw measurement rectangles and radial arrows around a circle.

        Every rectangle is perpendicular to the circle edge (radially
        oriented) and every arrow points toward the centre.

        Image-coordinate convention (row ↓, col →):
          point on circle at angle φ:
            pr = cr + radius · sin(φ)    pc = cc + radius · cos(φ)
          radial direction toward centre:   (−sin(φ), −cos(φ))
          tangential direction (CCW):       ( cos(φ), −sin(φ))
        """
        cr, cc = center
        rect_color = "#4488FF"
        arrow_color = "#FF4444"

        if end_phi <= start_phi:
            phi_range = end_phi + 2 * math.pi - start_phi
        else:
            phi_range = end_phi - start_phi

        half_l1 = length1 / 2.0  # radial half-length
        half_l2 = length2 / 2.0  # tangential half-length

        # Local corners in (tangential, radial) = (t, r) where
        #  r > 0  means "toward centre",  r < 0  means "away from centre"
        corners_t_r = [
            (-half_l2, -half_l1),   # far  – tangentially left
            ( half_l2, -half_l1),   # far  – tangentially right
            ( half_l2,  half_l1),   # near – tangentially right
            (-half_l2,  half_l1),   # near – tangentially left
        ]

        for i in range(num):
            if num > 1:
                phi = start_phi + i / (num - 1) * phi_range
            else:
                phi = start_phi + phi_range / 2

            cos_p = math.cos(phi)
            sin_p = math.sin(phi)

            # point on the expected circle
            pr = cr + radius * sin_p
            pc = cc + radius * cos_p

            # ---- rotated rectangle ----
            canvas_corners = []
            for t, r in corners_t_r:
                rw = pr + t * cos_p - r * sin_p
                cw = pc - t * sin_p - r * cos_p
                cx_c, cy_c = self._tmpl_to_canvas(rw, cw)
                canvas_corners.extend([cx_c, cy_c])
            rect_item = self._canvas.create_polygon(
                *canvas_corners, outline=rect_color, fill="", width=1,
            )
            items.append(rect_item)

            # ---- arrow pointing toward centre ----
            # tail = little bit outward,  tip = little bit inward (→ centre)
            tail_row = pr + half_l1 * 0.6 * sin_p
            tail_col = pc + half_l1 * 0.6 * cos_p
            tip_row  = pr - half_l1 * 0.6 * sin_p
            tip_col  = pc - half_l1 * 0.6 * cos_p
            sx, sy = self._tmpl_to_canvas(tail_row, tail_col)
            tx, ty = self._tmpl_to_canvas(tip_row,  tip_col)
            arr = self._canvas.create_line(
                sx, sy, tx, ty, fill=arrow_color, width=1, arrow=tk.LAST,
            )
            items.append(arr)

    def _draw_template_match_overlay(self, params: dict, color: str) -> List[int]:
        """Draw a template match point region.

        The bounding box shows the template patch that will be extracted.
        Corner brackets face outward (┌┐ └┘) — a convention that signals
        "this region will be searched/matched elsewhere."  The crosshair
        marks the reference point.
        """
        items = []
        row = params["row"]
        col = params["col"]
        half = params.get("template_size", 40) / 2.0

        # Bounding box
        x1, y1 = self._tmpl_to_canvas(row - half, col - half)
        x2, y2 = self._tmpl_to_canvas(row + half, col + half)
        rect = self._canvas.create_rectangle(
            x1, y1, x2, y2, outline=color, width=1, dash=(4, 2))
        items.append(rect)

        # Outward-facing corner brackets (indicate "searched elsewhere")
        bracket_len = 6  # canvas pixels
        # ┌ top-left
        items.append(self._canvas.create_line(x1, y1, x1 - bracket_len, y1, fill=color))
        items.append(self._canvas.create_line(x1, y1, x1, y1 - bracket_len, fill=color))
        # ┐ top-right
        items.append(self._canvas.create_line(x2, y1, x2 + bracket_len, y1, fill=color))
        items.append(self._canvas.create_line(x2, y1, x2, y1 - bracket_len, fill=color))
        # └ bottom-left
        items.append(self._canvas.create_line(x1, y2, x1 - bracket_len, y2, fill=color))
        items.append(self._canvas.create_line(x1, y2, x1, y2 + bracket_len, fill=color))
        # ┘ bottom-right
        items.append(self._canvas.create_line(x2, y2, x2 + bracket_len, y2, fill=color))
        items.append(self._canvas.create_line(x2, y2, x2, y2 + bracket_len, fill=color))

        # Crosshair at center (reference point for matching)
        cx, cy = self._tmpl_to_canvas(row, col)
        cs = 6
        cross_h = self._canvas.create_line(cx - cs, cy, cx + cs, cy, fill=color)
        cross_v = self._canvas.create_line(cx, cy - cs, cx, cy + cs, fill=color)
        items.extend([cross_h, cross_v])

        return items

    def _delete_tool_items(self, tool: Dict[str, Any]):
        """Remove canvas items for a tool."""
        for item_id in tool.get("canvas_items", []):
            self._canvas.delete(item_id)
        tool["canvas_items"] = []

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def _on_click(self, event):
        """Handle mouse click on template canvas."""
        row, col = self._canvas_to_tmpl(event.x, event.y)

        # Clamp to template bounds
        if self._template_image is not None:
            h, w = self._template_image.shape[:2]
            row = max(0.0, min(float(row), float(h - 1)))
            col = max(0.0, min(float(col), float(w - 1)))

        if self._current_tool == TemplateTool.SELECT:
            # Check if clicking on an existing tool
            clicked_label, prev_selected = self._select_tool_at(event.x, event.y)
            if clicked_label is not None:
                if clicked_label == prev_selected:
                    # Same tool clicked again — open edit dialog
                    self._edit_selected_tool()
            else:
                # Clicked on empty space — start panning
                self._drag_start = (event.x, event.y)
                self._panning = True
            return

        elif self._current_tool in (TemplateTool.EDGE_POINT, TemplateTool.EDGE_PAIR,
                                     TemplateTool.FIT_LINE):
            # Click to set start, drag to set direction + length/endpoint
            self._draw_start = (row, col)
            self._drawing = True

        elif self._current_tool == TemplateTool.FIT_CIRCLE:
            # Click center, drag radius
            self._draw_start = (row, col)
            self._drawing = True

        elif self._current_tool == TemplateTool.TEMPLATE_MATCH_POINT:
            # Single click: place template-matching point immediately
            self._add_template_match_point(row, col)
            self.set_status("已添加模板匹配点")
            # Switch back to select
            self.after(100, lambda: self.set_tool(TemplateTool.SELECT))

    def _on_drag(self, event):
        """Handle mouse drag."""
        # Panning in SELECT mode
        if self._panning and self._drag_start is not None:
            dx = event.x - self._drag_start[0]
            dy = event.y - self._drag_start[1]
            self._offset_x += dx
            self._offset_y += dy
            self._drag_start = (event.x, event.y)
            self._redraw()
            return

        if not self._drawing or self._draw_start is None:
            return

        row, col = self._canvas_to_tmpl(event.x, event.y)

        if self._current_tool in (TemplateTool.EDGE_POINT, TemplateTool.EDGE_PAIR):
            # Show preview of probe rectangle
            start_r, start_c = self._draw_start
            dr = row - start_r
            dc = col - start_c
            length = np.sqrt(dr**2 + dc**2)
            angle = np.arctan2(dc, -dr)  # angle in radians

            # Preview (draw temporary rectangle)
            self._canvas.delete("preview")
            params = {
                "row": start_r, "col": start_c,
                "angle": angle, "length1": length, "length2": max(3.0, length * 0.1),
                "transition": "all",
            }
            items = self._draw_edge_probe(params, "cyan", "all")
            for item_id in items:
                self._canvas.addtag_withtag("preview", item_id)

        elif self._current_tool == TemplateTool.FIT_LINE:
            # Show preview line from start to current position
            start_r, start_c = self._draw_start
            self._canvas.delete("preview")
            params = {
                "start": self._draw_start,
                "end": (row, col),
                "num_measures": 10,
                "measure_length2": 25.0,
            }
            items = self._draw_fit_line_overlay(params, "cyan")
            for item_id in items:
                self._canvas.addtag_withtag("preview", item_id)

        elif self._current_tool == TemplateTool.FIT_CIRCLE:
            start_r, start_c = self._draw_start
            radius = np.sqrt((row - start_r)**2 + (col - start_c)**2)

            self._canvas.delete("preview")
            params = {
                "center": self._draw_start,
                "radius": radius,
                "radius_min": radius * 0.5,
                "radius_max": radius * 1.5,
            }
            items = self._draw_fit_circle_overlay(params, "cyan")
            for item_id in items:
                self._canvas.addtag_withtag("preview", item_id)

    def _on_release(self, event):
        """Handle mouse release."""
        # End panning
        if self._panning:
            self._panning = False
            self._drag_start = None
            return

        if not self._drawing:
            return

        self._canvas.delete("preview")
        row, col = self._canvas_to_tmpl(event.x, event.y)

        if self._current_tool == TemplateTool.EDGE_POINT:
            self._add_edge_tool("EdgePoint", row, col)
        elif self._current_tool == TemplateTool.EDGE_PAIR:
            self._add_edge_tool("EdgePair", row, col)
        elif self._current_tool == TemplateTool.FIT_LINE:
            self._add_line_tool(self._draw_start, (row, col))
        elif self._current_tool == TemplateTool.FIT_CIRCLE:
            self._add_circle_tool(row, col)

        self._drawing = False
        self._draw_start = None

    def _on_motion(self, event):
        """Mouse motion - update status."""
        if self._template_image is None:
            return
        row, col = self._canvas_to_tmpl(event.x, event.y)
        h, w = self._template_image.shape[:2]
        if 0 <= row < h and 0 <= col < w:
            self.set_status(f"({row:.1f}, {col:.1f}) 模板坐标")

    # ------------------------------------------------------------------
    # Tool creation
    # ------------------------------------------------------------------

    def _auto_label(self, prefix: str) -> str:
        """Generate a unique label."""
        if prefix not in self._tool_counters:
            self._tool_counters[prefix] = 0
        self._tool_counters[prefix] += 1
        existing = {t["label"] for t in self._tools}
        label = f"{prefix}_{self._tool_counters[prefix]}"
        while label in existing:
            self._tool_counters[prefix] += 1
            label = f"{prefix}_{self._tool_counters[prefix]}"
        return label

    def _add_edge_tool(self, obj_type: str, end_row: float, end_col: float):
        """Add an edge point or edge pair tool based on draw interaction."""
        start_r, start_c = self._draw_start
        dr = end_row - start_r
        dc = end_col - start_c
        length = np.sqrt(dr**2 + dc**2)
        angle = np.arctan2(dc, -dr)

        params = {
            "row": start_r,
            "col": start_c,
            "angle": angle,
            "length1": max(5.0, length),
            "length2": max(2.0, length * 0.1),
            "sigma": 1.0,
            "threshold": 30.0,
            "transition": "all" if obj_type == "EdgePoint" else "negative",
            "select": "first",
            "interpolation": "linear",
        }

        # Show dialog for algorithm params
        if obj_type == "EdgePoint":
            dlg_params = EdgePointDialog.ask(self, params={
                "sigma": 1.0, "threshold": 30.0,
                "transition": "all", "select": "first",
                "interpolation": "linear",
            })
        else:
            dlg_params = EdgePairDialog.ask(self, params={
                "sigma": 1.0, "threshold": 30.0,
                "transition": "negative", "select": "first",
                "interpolation": "linear",
            })

        if dlg_params is None:
            return  # Cancelled

        params.update(dlg_params)
        prefix = "ep" if obj_type == "EdgePoint" else "pair"
        label = self._auto_label(prefix)

        tool = {
            "object_type": obj_type,
            "label": label,
            "params": params,
            "canvas_items": [],
            "_selected": False,
        }
        self._tools.append(tool)
        self._redraw_tools()

        if self.on_tool_added:
            self.on_tool_added(obj_type, label, params)

        self.set_status(f"已添加 {label} ({obj_type})")

    def _add_line_tool(self, start: Tuple[float, float], end: Tuple[float, float]):
        """Add a fit line tool."""
        params = {
            "start": start,
            "end": end,
            "measure_length1": 10.0,
            "measure_length2": 25.0,
            "num_measures": 10,
            "sigma": 1.0,
            "threshold": 30.0,
            "transition": "all",
        }

        dlg_params = FitLineDialog.ask(self, params={
            "num_measures": 10, "measure_length1": 10.0,
            "measure_length2": 25.0, "sigma": 1.0,
            "threshold": 30.0, "transition": "all",
        })

        if dlg_params is None:
            return

        params.update(dlg_params)
        label = self._auto_label("line")

        tool = {
            "object_type": "FitLine",
            "label": label,
            "params": params,
            "canvas_items": [],
            "_selected": False,
        }
        self._tools.append(tool)
        self._redraw_tools()

        if self.on_tool_added:
            self.on_tool_added("FitLine", label, params)

        self.set_status(f"已添加 {label} (FitLine)")

    def _add_circle_tool(self, end_row: float, end_col: float):
        """Add a fit circle tool."""
        center = self._draw_start
        radius = np.sqrt((end_row - center[0])**2 + (end_col - center[1])**2)

        params = {
            "center": center,
            "radius": max(5.0, radius),
            "measure_length1": 20.0,
            "measure_length2": 10.0,
            "num_measures": 12,
            "sigma": 1.0,
            "threshold": 30.0,
            "transition": "all",
            "start_phi": 0.0,
            "end_phi": 2 * np.pi,
        }

        dlg_params = FitCircleDialog.ask(self, params={
            "measure_length1": 20.0, "measure_length2": 10.0,
            "num_measures": 12, "sigma": 1.0,
            "threshold": 30.0, "transition": "all",
        })

        if dlg_params is None:
            return

        params.update(dlg_params)
        label = self._auto_label("circle")

        tool = {
            "object_type": "FitCircle",
            "label": label,
            "params": params,
            "canvas_items": [],
            "_selected": False,
        }
        self._tools.append(tool)
        self._redraw_tools()

        if self.on_tool_added:
            self.on_tool_added("FitCircle", label, params)

        self.set_status(f"已添加 {label} (FitCircle)")

    def _add_template_match_point(self, row: float, col: float):
        """Add a template-matching point at the clicked position."""
        params = {
            "row": row,
            "col": col,
            "template_size": 40,
            "preprocessor_type": "raw",
            "match_score_threshold": 0.5,
            "angle_range_half": 15.0,
            "angle_step": 1.0,
            "use_subpixel": True,
        }

        dlg_params = TemplateMatchPointDialog.ask(self, params={
            "template_size": 40,
            "preprocessor_type": "raw",
            "match_score_threshold": 0.5,
            "angle_range_half": 15.0,
            "angle_step": 1.0,
            "use_subpixel": True,
        })

        if dlg_params is None:
            return

        params.update(dlg_params)
        label = self._auto_label("tmpl")

        tool = {
            "object_type": "TemplateMatchPoint",
            "label": label,
            "params": params,
            "canvas_items": [],
            "_selected": False,
        }
        self._tools.append(tool)
        self._redraw_tools()

        if self.on_tool_added:
            self.on_tool_added("TemplateMatchPoint", label, params)

        self.set_status(f"已添加 {label} (TemplateMatchPoint)")

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _select_tool_at(self, cx: float, cy: float) -> Tuple[Optional[str], Optional[str]]:
        """Select a tool at canvas position.

        Returns:
            (clicked_label, prev_selected_label) — both None if nothing clicked.
            If the same tool is clicked again, both will be the same label.
        """
        # Remember previously selected label
        prev_selected = None
        for t in self._tools:
            if t.get("_selected"):
                prev_selected = t["label"]
                break

        # Deselect all
        for t in self._tools:
            t["_selected"] = False

        # Find clicked tool
        clicked_label = None
        overlap = self._canvas.find_overlapping(cx - 5, cy - 5, cx + 5, cy + 5)
        for i, t in enumerate(self._tools):
            for item_id in t.get("canvas_items", []):
                if item_id in overlap:
                    t["_selected"] = True
                    clicked_label = t["label"]
                    self._redraw_tools()
                    self.set_status(f"已选中 {t['label']} — 再次点击可编辑参数")
                    break

        if clicked_label is None:
            self._redraw_tools()

        return clicked_label, prev_selected

    def _delete_selected(self):
        """Delete the currently selected tool."""
        for i, t in enumerate(self._tools):
            if t.get("_selected"):
                self.remove_tool(t["label"])
                self.set_status(f"已删除 {t['label']}")
                return

    def _edit_selected_tool(self):
        """Open the parameter dialog for the currently selected tool.

        The dialog is pre-filled with the tool's current params.
        If the user confirms, the tool's params and overlay are updated,
        and the on_tool_edited callback is fired.
        """
        selected = None
        for t in self._tools:
            if t.get("_selected"):
                selected = t
                break

        if selected is None:
            return

        obj_type = selected["object_type"]
        params = selected["params"]

        # Open the appropriate dialog with current params pre-filled
        dlg_params = None
        if obj_type == "EdgePoint":
            dlg_params = EdgePointDialog.ask(self, params=dict(params))
        elif obj_type == "EdgePair":
            dlg_params = EdgePairDialog.ask(self, params=dict(params))
        elif obj_type == "FitLine":
            dlg_params = FitLineDialog.ask(self, params=dict(params))
        elif obj_type == "FitCircle":
            dlg_params = FitCircleDialog.ask(self, params=dict(params))
        elif obj_type == "TemplateMatchPoint":
            dlg_params = TemplateMatchPointDialog.ask(self, params=dict(params))

        if dlg_params is None:
            return  # User cancelled

        # Update only the algorithm params (dialog returns subset of keys)
        params.update(dlg_params)
        self._redraw_tools()

        if self.on_tool_edited:
            self.on_tool_edited(selected["label"], params)

        self.set_status(f"已更新 {selected['label']} 的参数")

    def _on_double_click(self, event):
        """Handle double-click on template canvas — edit tool at position."""
        if self._current_tool == TemplateTool.SELECT:
            clicked_label, _ = self._select_tool_at(event.x, event.y)
            if clicked_label is not None:
                self._edit_selected_tool()
