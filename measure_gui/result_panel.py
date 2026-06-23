"""
Bottom result panel for displaying multi-target measurement results.

Contains:
  - Target list (ttk.Treeview) showing all detected targets
  - Measurement results table for the selected target
  - Summary text box showing all results (requirement Step 5)
  - Export functionality
"""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable, Dict, List, Optional

from .multi_target import TargetResult


class ResultPanel(ttk.Frame):
    """
    Bottom panel for measurement result display.

    Callbacks:
        on_target_selected: (target: TargetResult) -> None
    """

    def __init__(self, parent: tk.Widget, **kwargs):
        super().__init__(parent, **kwargs)

        # Callbacks
        self.on_target_selected: Optional[Callable] = None

        # Data
        self._targets: List[TargetResult] = []

        self._build_ui()

    def _build_ui(self):
        # Top control bar
        ctrl = ttk.Frame(self)
        ctrl.pack(fill=tk.X, padx=4, pady=2)

        self._status_label = ttk.Label(ctrl, text="就绪", foreground="gray")
        self._status_label.pack(side=tk.LEFT)

        ttk.Button(ctrl, text="导出 CSV", command=self._export_csv).pack(side=tk.RIGHT, padx=2)

        # Main content: horizontal PanedWindow
        pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=4, pady=2)

        # Left: target list
        left_frame = ttk.LabelFrame(pane, text="目标列表", padding=3)
        pane.add(left_frame, weight=1)

        target_columns = ("id", "score", "rotation", "center", "status")
        self._target_tree = ttk.Treeview(
            left_frame, columns=target_columns, show="headings", height=6,
        )
        self._target_tree.heading("id", text="ID")
        self._target_tree.heading("score", text="分数")
        self._target_tree.heading("rotation", text="旋转°")
        self._target_tree.heading("center", text="中心位置")
        self._target_tree.heading("status", text="状态")

        self._target_tree.column("id", width=40, anchor=tk.CENTER)
        self._target_tree.column("score", width=60, anchor=tk.CENTER)
        self._target_tree.column("rotation", width=60, anchor=tk.CENTER)
        self._target_tree.column("center", width=140)
        self._target_tree.column("status", width=50, anchor=tk.CENTER)

        target_scroll = ttk.Scrollbar(left_frame, orient=tk.VERTICAL,
                                      command=self._target_tree.yview)
        self._target_tree.configure(yscrollcommand=target_scroll.set)
        self._target_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        target_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self._target_tree.bind("<<TreeviewSelect>>", self._on_target_select)

        # Right: measurement results + summary
        right_frame = ttk.Frame(pane)
        pane.add(right_frame, weight=2)

        # Measurement table
        meas_frame = ttk.LabelFrame(right_frame, text="选中目标测量结果", padding=3)
        meas_frame.pack(fill=tk.BOTH, expand=True)

        meas_columns = ("label", "type", "value", "error", "status")
        self._meas_tree = ttk.Treeview(
            meas_frame, columns=meas_columns, show="headings", height=6,
        )
        self._meas_tree.heading("label", text="标签")
        self._meas_tree.heading("type", text="类型")
        self._meas_tree.heading("value", text="值")
        self._meas_tree.heading("error", text="误差")
        self._meas_tree.heading("status", text="状态")

        self._meas_tree.column("label", width=100)
        self._meas_tree.column("type", width=70)
        self._meas_tree.column("value", width=150)
        self._meas_tree.column("error", width=60)
        self._meas_tree.column("status", width=50, anchor=tk.CENTER)

        meas_scroll = ttk.Scrollbar(meas_frame, orient=tk.VERTICAL,
                                    command=self._meas_tree.yview)
        self._meas_tree.configure(yscrollcommand=meas_scroll.set)
        self._meas_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        meas_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Summary text
        summ_frame = ttk.LabelFrame(right_frame, text="测量结果汇总", padding=3)
        summ_frame.pack(fill=tk.BOTH, expand=True)

        self._summary_text = tk.Text(summ_frame, height=8, wrap=tk.WORD,
                                     font=("TkFixedFont", 9))
        summ_scroll = ttk.Scrollbar(summ_frame, orient=tk.VERTICAL,
                                    command=self._summary_text.yview)
        self._summary_text.configure(yscrollcommand=summ_scroll.set)
        self._summary_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        summ_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_results(self, targets: List[TargetResult], summary: str = ""):
        """Set all measurement results."""
        self._targets = targets
        self._clear_target_list()
        self._clear_meas_table()
        self._summary_text.delete("1.0", tk.END)

        valid_count = sum(1 for t in targets if t.valid)
        self._status_label.config(
            text=f"检测到 {len(targets)} 个目标, {valid_count} 个有效",
            foreground="green" if valid_count > 0 else "red",
        )

        for t in targets:
            status_str = "✓" if t.valid else "✗"
            self._target_tree.insert(
                "", tk.END, iid=str(t.id),
                values=(
                    t.id, f"{t.score:.4f}", f"{t.rotation_deg:.1f}",
                    f"({t.center_row:.1f}, {t.center_col:.1f})", status_str,
                ),
                tags=("valid" if t.valid else "invalid",),
            )

        # Color coding
        self._target_tree.tag_configure("valid", foreground="green")
        self._target_tree.tag_configure("invalid", foreground="red")

        if summary:
            self._summary_text.insert("1.0", summary)
        elif targets:
            self._summary_text.insert("1.0", "选择目标查看详情")

    def clear_results(self):
        """Clear all results."""
        self._targets = []
        self._clear_target_list()
        self._clear_meas_table()
        self._summary_text.delete("1.0", tk.END)
        self._status_label.config(text="就绪", foreground="gray")

    def get_summary_text(self) -> str:
        """Get the summary text content."""
        return self._summary_text.get("1.0", tk.END)

    def set_summary_text(self, text: str):
        """Set the summary text content."""
        self._summary_text.delete("1.0", tk.END)
        self._summary_text.insert("1.0", text)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_target_select(self, event):
        """Update measurement table when a target is selected."""
        selection = self._target_tree.selection()
        if not selection:
            return

        target_id = int(selection[0])
        target = None
        for t in self._targets:
            if t.id == target_id:
                target = t
                break

        if target is None:
            return

        self._clear_meas_table()

        for label, result in target.measurements.items():
            if label == "_error":
                continue

            if isinstance(result, dict):
                rtype = result.get("type", "?")
                valid = result.get("valid", False)
                status_str = "✓" if valid else "✗"

                if rtype == "point" and valid:
                    value_str = f"({result['row']:.2f}, {result['col']:.2f})"
                elif rtype == "line" and valid:
                    value_str = (
                        f"({result['start_row']:.1f},{result['start_col']:.1f})"
                        f"→({result['end_row']:.1f},{result['end_col']:.1f})"
                    )
                elif rtype == "circle" and valid:
                    value_str = (
                        f"({result['center_row']:.1f},{result['center_col']:.1f}) "
                        f"r={result['radius']:.2f}"
                    )
                elif rtype in ("distance", "angle") and valid:
                    value_str = f"{result.get('value', result.get('value_deg', 0)):.3f}"
                else:
                    value_str = "-"
            else:
                rtype = getattr(result, "type", "?")
                valid = getattr(result, "valid", False)
                status_str = "✓" if valid else "✗"
                value_str = str(result) if valid else "-"

            error_str = ""
            if isinstance(result, dict):
                error_str = str(result.get("meta", {}).get("mean_error", ""))[:8]
            elif hasattr(result, "meta"):
                error_str = str(result.meta.get("mean_error", ""))[:8]

            self._meas_tree.insert(
                "", tk.END,
                values=(label, rtype, value_str, error_str, status_str),
            )

        # Notify
        if self.on_target_selected:
            self.on_target_selected(target)

    def _clear_target_list(self):
        for item in self._target_tree.get_children():
            self._target_tree.delete(item)

    def _clear_meas_table(self):
        for item in self._meas_tree.get_children():
            self._meas_tree.delete(item)

    def _export_csv(self):
        """Export results to a CSV file."""
        if not self._targets:
            messagebox.showinfo("提示", "没有可导出的结果")
            return

        filepath = filedialog.asksaveasfilename(
            parent=self,
            title="导出测量结果",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not filepath:
            return

        try:
            import csv

            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Target ID", "Score", "Rotation (deg)", "Scale",
                    "Center Row", "Center Col", "Valid",
                    "Measurement", "Type", "Value", "Status",
                ])

                for t in self._targets:
                    base = [t.id, t.score, t.rotation_deg, t.scale,
                            t.center_row, t.center_col, t.valid]
                    if not t.measurements:
                        writer.writerow(base + ["", "", "", ""])
                    else:
                        for label, result in t.measurements.items():
                            if label == "_error":
                                continue
                            if isinstance(result, dict):
                                rtype = result.get("type", "")
                                if rtype == "point":
                                    val = f"({result.get('row',0):.2f},{result.get('col',0):.2f})"
                                elif rtype == "distance":
                                    val = f"{result.get('value', 0):.3f}"
                                elif rtype == "angle":
                                    val = f"{result.get('value_deg', 0):.3f}°"
                                else:
                                    val = str(result)
                                status_v = "✓" if result.get("valid") else "✗"
                            else:
                                rtype = getattr(result, "type", "")
                                val = str(result)
                                status_v = "✓" if getattr(result, "valid", False) else "✗"
                            writer.writerow(base + [label, rtype, val, status_v])

            messagebox.showinfo("成功", f"结果已导出到:\n{filepath}")
        except Exception as e:
            messagebox.showerror("导出失败", str(e))
