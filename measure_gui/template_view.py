"""
Template preview view with interactive measurement tool placement.

Displays the straightened template image and allows users to:
  - Click/drag to place measurement tools (edge point, edge pair, fit line, fit circle)
  - Edit tool parameters via dialogs
  - See all placed tools overlaid on the template

Tools are defined in the straightened template coordinate space (origin at top-left).
"""

from __future__ import annotations

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
        self._draw_phase: int = 0  # 0=wait_start, 1=wait_end (for line)

        # Display scale (template usually small, may need to upscale)
        self._scale: float = 1.0
        self._offset_x: float = 0.0
        self._offset_y: float = 0.0

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
        self._canvas.bind("<B1-Motion>", self._on_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_release)
        self._canvas.bind("<Motion>", self._on_motion)

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
        self._redraw()

    @property
    def template_image(self) -> Optional[np.ndarray]:
        return self._template_image

    def set_tool(self, tool: TemplateTool):
        """Switch the active placement tool."""
        self._current_tool = tool
        self._drawing = False
        self._draw_phase = 0

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
            scale, offset_x, offset_y — display transform
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

        # Redraw all tool overlays on the canvas
        self._redraw_tools()

    def set_status(self, text: str):
        self._status.config(text=text)

    # ------------------------------------------------------------------
    # Template coordinate transforms (no zoom — fixed scale to fit canvas)
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
    # Redraw
    # ------------------------------------------------------------------

    def _redraw(self):
        """Full redraw of template image."""
        self._canvas.delete("all")

        if self._template_image is None:
            return

        h, w = self._template_image.shape[:2]
        canvas_w = self._canvas.winfo_width() or self._width
        canvas_h = self._canvas.winfo_height() or self._height

        # Scale to fit canvas
        self._scale = min(canvas_w / w, canvas_h / h) * 0.95
        self._offset_x = (canvas_w - w * self._scale) / 2
        self._offset_y = (canvas_h - h * self._scale) / 2

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
        """Draw a fit line ROI."""
        items = []
        start = params["start"]
        end = params["end"]

        sx, sy = self._tmpl_to_canvas(start[0], start[1])
        ex, ey = self._tmpl_to_canvas(end[0], end[1])

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

        return items

    def _draw_fit_circle_overlay(self, params: dict, color: str) -> List[int]:
        """Draw a fit circle ROI."""
        items = []
        center = params["center"]
        radius = params["radius"]

        cx, cy = self._tmpl_to_canvas(center[0], center[1])
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

    def _draw_template_match_overlay(self, params: dict, color: str) -> List[int]:
        """Draw a template match point region (square + crosshair)."""
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

        # Crosshair at center
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
            self._select_tool_at(event.x, event.y)
            return

        elif self._current_tool in (TemplateTool.EDGE_POINT, TemplateTool.EDGE_PAIR):
            # Click to set center, drag to set direction + length
            self._draw_start = (row, col)
            self._drawing = True

        elif self._current_tool == TemplateTool.FIT_LINE:
            if self._draw_phase == 0:
                self._draw_start = (row, col)
                self._drawing = True
                self._draw_phase = 1
                self.set_status(f"点击终点... 起点=({row:.0f},{col:.0f})")
            elif self._draw_phase == 1:
                # Complete the line
                start = self._draw_start
                end = (row, col)
                self._add_line_tool(start, end)
                self._draw_phase = 0
                self._drawing = False
                self.set_status("已添加拟合直线")

        elif self._current_tool == TemplateTool.FIT_CIRCLE:
            # Click center, drag radius
            self._draw_start = (row, col)
            self._drawing = True

        elif self._current_tool == TemplateTool.TEMPLATE_MATCH_POINT:
            # Single click: place template-matching point immediately
            self._add_template_match_point(row, col)
            self.set_status("已添加模板匹配点")
            # Switch back to select
            self._after(100, lambda: self.set_tool(TemplateTool.SELECT))

    def _on_drag(self, event):
        """Handle mouse drag."""
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
        if not self._drawing:
            return

        self._canvas.delete("preview")
        row, col = self._canvas_to_tmpl(event.x, event.y)

        if self._current_tool == TemplateTool.EDGE_POINT:
            self._add_edge_tool("EdgePoint", row, col)
        elif self._current_tool == TemplateTool.EDGE_PAIR:
            self._add_edge_tool("EdgePair", row, col)
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
            "radius_min": max(2.0, radius * 0.5),
            "radius_max": max(10.0, radius * 1.5),
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
            "radius_min": params["radius_min"],
            "radius_max": params["radius_max"],
            "num_measures": 12, "measure_length1": 20.0,
            "measure_length2": 10.0, "sigma": 1.0,
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
        from .dialogs import TemplateMatchPointDialog

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

    def _select_tool_at(self, cx: float, cy: float):
        """Select a tool at canvas position."""
        # Deselect all
        for t in self._tools:
            t["_selected"] = False

        # Find clicked tool
        overlap = self._canvas.find_overlapping(cx - 5, cy - 5, cx + 5, cy + 5)
        for i, t in enumerate(self._tools):
            for item_id in t.get("canvas_items", []):
                if item_id in overlap:
                    t["_selected"] = True
                    self._redraw_tools()
                    self.set_status(f"已选中 {t['label']} — 点击[删除]移除或拖拽移动")
                    return

        self._redraw_tools()

    def _delete_selected(self):
        """Delete the currently selected tool."""
        for i, t in enumerate(self._tools):
            if t.get("_selected"):
                self.remove_tool(t["label"])
                self.set_status(f"已删除 {t['label']}")
                return
