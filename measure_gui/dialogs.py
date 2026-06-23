"""
Parameter dialogs for measurement tool configuration.

Each dialog is a tk.Toplevel that allows editing algorithm parameters
(sigma, threshold, transition, etc.) for a specific measurement tool.

The coordinate parameters (row, col, angle, length1, length2, etc.) are
typically set via interactive drawing on the template view and passed in,
while algorithm parameters are edited here.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Callable, Dict, Optional

import numpy as np


class BaseParamDialog(tk.Toplevel):
    """Base class for parameter dialogs."""

    def __init__(self, parent: tk.Widget, title: str = "Parameters",
                 params: Optional[Dict[str, Any]] = None):
        super().__init__(parent)
        self.title(title)
        self.result: Optional[Dict[str, Any]] = None
        self._params = params or {}
        self._vars: Dict[str, tk.Variable] = {}

        self.transient(parent)
        self.grab_set()

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)

        # Center on parent
        self.update_idletasks()
        if parent.winfo_rootx() > 0:
            x = parent.winfo_rootx() + parent.winfo_width() // 2 - self.winfo_width() // 2
            y = parent.winfo_rooty() + parent.winfo_height() // 2 - self.winfo_height() // 2
            self.geometry(f"+{x}+{y}")

    def _build_ui(self):
        # Subclasses override
        pass

    def _add_float_entry(self, parent: tk.Widget, label: str, key: str,
                         default: float, row: int, from_val: float = 0.0,
                         to_val: float = 1000.0):
        """Add a labeled float entry with a spinbox."""
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        var = tk.DoubleVar(value=self._params.get(key, default))
        spin = ttk.Spinbox(parent, textvariable=var, from_=from_val, to=to_val,
                          increment=0.1, width=12)
        spin.grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
        self._vars[key] = var

    def _add_int_entry(self, parent: tk.Widget, label: str, key: str,
                       default: int, row: int, from_val: int = 0, to_val: int = 1000):
        """Add a labeled integer spinbox."""
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        var = tk.IntVar(value=self._params.get(key, default))
        spin = ttk.Spinbox(parent, textvariable=var, from_=from_val, to=to_val,
                          increment=1, width=12)
        spin.grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
        self._vars[key] = var

    def _add_combo(self, parent: tk.Widget, label: str, key: str,
                   values: list, default: str, row: int):
        """Add a labeled combobox."""
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        var = tk.StringVar(value=self._params.get(key, default))
        combo = ttk.Combobox(parent, textvariable=var, values=values, state="readonly", width=15)
        combo.grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
        self._vars[key] = var

    def _add_buttons(self, parent: tk.Widget, row: int):
        """Add OK/Cancel buttons."""
        btn_frame = ttk.Frame(parent)
        btn_frame.grid(row=row, column=0, columnspan=2, pady=10)

        ttk.Button(btn_frame, text="确定", command=self._on_ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="取消", command=self._on_cancel).pack(side=tk.LEFT, padx=5)

    def _on_ok(self):
        self.result = {key: var.get() for key, var in self._vars.items()}
        self.destroy()

    def _on_cancel(self):
        self.result = None
        self.destroy()

    @classmethod
    def ask(cls, parent: tk.Widget, params: Optional[Dict[str, Any]] = None,
            **kwargs) -> Optional[Dict[str, Any]]:
        """Convenience: create dialog, wait, return result dict or None."""
        dlg = cls(parent, params=params, **kwargs)
        dlg.wait_window()
        return dlg.result


class EdgePointDialog(BaseParamDialog):
    """Dialog for EdgePointObject parameters."""

    def __init__(self, parent, params=None):
        super().__init__(parent, "边缘点参数 (Edge Point)", params)

    def _build_ui(self):
        main = ttk.Frame(self, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        row = 0
        ttk.Label(main, text="算法参数", font=("TkDefaultFont", 10, "bold")).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
        row += 1

        self._add_float_entry(main, "Sigma (平滑)", "sigma", 1.0, row, 0.0, 100.0); row += 1
        self._add_float_entry(main, "Threshold (阈值)", "threshold", 30.0, row, 0.0, 255.0); row += 1
        self._add_combo(main, "Transition (边缘方向)", "transition",
                       ["all", "positive", "negative"], "all", row); row += 1
        self._add_combo(main, "Select (选择)", "select",
                       ["first", "last", "all"], "first", row); row += 1
        self._add_combo(main, "Interpolation (插值)", "interpolation",
                       ["linear", "cubic", "nearest"], "linear", row); row += 1

        self._add_buttons(main, row + 1)


class EdgePairDialog(BaseParamDialog):
    """Dialog for EdgePairObject parameters."""

    def __init__(self, parent, params=None):
        super().__init__(parent, "边缘对参数 (Edge Pair)", params)

    def _build_ui(self):
        main = ttk.Frame(self, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        row = 0
        ttk.Label(main, text="算法参数", font=("TkDefaultFont", 10, "bold")).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
        row += 1

        self._add_float_entry(main, "Sigma (平滑)", "sigma", 1.0, row, 0.0, 100.0); row += 1
        self._add_float_entry(main, "Threshold (阈值)", "threshold", 30.0, row, 0.0, 255.0); row += 1
        self._add_combo(main, "Transition (边缘对方向)", "transition",
                       ["negative", "positive", "all"], "negative", row); row += 1
        self._add_combo(main, "Select (选择)", "select",
                       ["first", "last", "all"], "first", row); row += 1
        self._add_combo(main, "Interpolation (插值)", "interpolation",
                       ["linear", "cubic", "nearest"], "linear", row); row += 1

        self._add_buttons(main, row + 1)


class FitLineDialog(BaseParamDialog):
    """Dialog for FitLineObject parameters."""

    def __init__(self, parent, params=None):
        super().__init__(parent, "拟合直线参数 (Fit Line)", params)

    def _build_ui(self):
        main = ttk.Frame(self, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        row = 0
        ttk.Label(main, text="算法参数", font=("TkDefaultFont", 10, "bold")).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
        row += 1

        self._add_int_entry(main, "Num Measures (测量线数)", "num_measures", 10, row, 2, 200); row += 1
        self._add_float_entry(main, "Measure Length1 (半长)", "measure_length1", 10.0, row, 0.1, 1000.0); row += 1
        self._add_float_entry(main, "Measure Length2 (半宽)", "measure_length2", 25.0, row, 0.1, 1000.0); row += 1
        self._add_float_entry(main, "Sigma (平滑)", "sigma", 1.0, row, 0.0, 100.0); row += 1
        self._add_float_entry(main, "Threshold (阈值)", "threshold", 30.0, row, 0.0, 255.0); row += 1
        self._add_combo(main, "Transition (边缘方向)", "transition",
                       ["all", "positive", "negative"], "all", row); row += 1

        self._add_buttons(main, row + 1)


class FitCircleDialog(BaseParamDialog):
    """Dialog for FitCircleObject parameters."""

    def __init__(self, parent, params=None):
        super().__init__(parent, "拟合圆参数 (Fit Circle)", params)

    def _build_ui(self):
        main = ttk.Frame(self, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        row = 0
        ttk.Label(main, text="算法参数", font=("TkDefaultFont", 10, "bold")).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
        row += 1

        self._add_float_entry(main, "Radius Min (最小半径)", "radius_min", 80.0, row, 0.0, 10000.0); row += 1
        self._add_float_entry(main, "Radius Max (最大半径)", "radius_max", 120.0, row, 0.0, 10000.0); row += 1
        self._add_int_entry(main, "Num Measures (测量线数)", "num_measures", 12, row, 3, 360); row += 1
        self._add_float_entry(main, "Measure Length1 (径向半长)", "measure_length1", 20.0, row, 0.1, 1000.0); row += 1
        self._add_float_entry(main, "Measure Length2 (切向半宽)", "measure_length2", 10.0, row, 0.1, 1000.0); row += 1
        self._add_float_entry(main, "Sigma (平滑)", "sigma", 1.0, row, 0.0, 100.0); row += 1
        self._add_float_entry(main, "Threshold (阈值)", "threshold", 30.0, row, 0.0, 255.0); row += 1
        self._add_combo(main, "Transition (边缘方向)", "transition",
                       ["all", "positive", "negative"], "all", row); row += 1

        self._add_buttons(main, row + 1)


class ComposedMeasureDialog(BaseParamDialog):
    """Dialog for composed measurement (distance, angle) — just label selection."""

    def __init__(self, parent, title: str, input_labels: list,
                 param_keys: list, params=None):
        self._input_labels = input_labels
        self._param_keys = param_keys
        super().__init__(parent, title, params)

    def _build_ui(self):
        main = ttk.Frame(self, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        row = 0
        for i, key in enumerate(self._param_keys):
            ttk.Label(main, text=f"{key}:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
            var = tk.StringVar(value=self._params.get(key, ""))
            combo = ttk.Combobox(main, textvariable=var, values=self._input_labels,
                                state="readonly", width=20)
            combo.grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
            self._vars[key] = var
            row += 1

        self._add_buttons(main, row + 1)


class TemplateMatchPointDialog(BaseParamDialog):
    """Dialog for template-matching point measurement parameters."""

    def __init__(self, parent, params=None):
        defaults = {
            "template_size": 40,
            "preprocessor_type": "raw",
            "match_score_threshold": 0.5,
            "angle_range_half": 15.0,
            "angle_step": 1.0,
            "use_subpixel": True,
        }
        if params:
            defaults.update(params)
        super().__init__(parent, "模板匹配点参数", defaults)

    def _build_ui(self):
        main = ttk.Frame(self, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        row = 0
        ttk.Label(main, text="算法参数", font=("TkDefaultFont", 10, "bold")).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
        row += 1

        self._add_int_entry(main, "模板尺寸 (px)", "template_size", 40, row, 10, 200); row += 1
        self._add_combo(main, "预处理类型", "preprocessor_type",
                       ["raw", "canny", "sobel", "clahe", "threshold"],
                       "raw", row); row += 1
        self._add_float_entry(main, "匹配分数阈值", "match_score_threshold", 0.5, row, 0.1, 1.0); row += 1
        self._add_float_entry(main, "角度搜索范围 (±度)", "angle_range_half", 15.0, row, 0.0, 180.0); row += 1
        self._add_float_entry(main, "角度步长 (度)", "angle_step", 1.0, row, 0.1, 10.0); row += 1

        # Subpixel checkbox (no _add_checkbox helper)
        ttk.Label(main, text="亚像素精度:").grid(
            row=row, column=0, sticky=tk.W, padx=5, pady=2)
        var = tk.BooleanVar(value=self._params.get("use_subpixel", True))
        cb = ttk.Checkbutton(main, variable=var)
        cb.grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
        self._vars["use_subpixel"] = var
        row += 1

        self._add_buttons(main, row + 1)
