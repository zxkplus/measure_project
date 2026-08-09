"""Measurement SDK packages.

The legacy development SDK is loaded lazily so importing the production v2
package never imports GUI factories or display-oriented modules.
"""

from __future__ import annotations

from typing import Any

__all__ = ["MeasureProject", "SessionReplay"]


def __getattr__(name: str) -> Any:
    """Load legacy SDK exports only when callers request them."""
    if name == "MeasureProject":
        from measure_api.project import MeasureProject

        return MeasureProject
    if name == "SessionReplay":
        from measure_api.replay import SessionReplay

        return SessionReplay
    raise AttributeError(name)
