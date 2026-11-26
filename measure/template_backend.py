"""Optional template-matching backends.

The legacy backend calls OpenCV directly. ``tm_ref`` is the pure Python
reference implementation used to validate pose/search semantics before the C++
backend arrives. Selection is deployment-wide through
``MEASURE_TEMPLATE_BACKEND`` and defaults to ``legacy`` so existing behaviour is
unchanged until a caller opts in.
"""

from __future__ import annotations

import importlib
import os
from types import ModuleType
from typing import Any, Dict, Optional

import cv2
import numpy as np

_ENV_VAR = "MEASURE_TEMPLATE_BACKEND"
_VALID = ("legacy", "tm_ref", "auto")

_TM_REF: Optional[ModuleType] = None
_TM_REF_ERROR: Optional[str] = None


def _load_tm_ref() -> Optional[ModuleType]:
    """Import ``tm_ref`` once and remember the failure reason."""

    global _TM_REF, _TM_REF_ERROR
    if _TM_REF is not None:
        return _TM_REF
    if _TM_REF_ERROR is not None:
        return None
    try:
        module = importlib.import_module("tm_ref")
    except Exception as error:  # pragma: no cover - depends on deployment
        _TM_REF_ERROR = f"{type(error).__name__}: {error}"
        return None
    _TM_REF = module
    return module


def selected_backend() -> str:
    """Return the configured backend name, normalised to a known value."""

    value = os.environ.get(_ENV_VAR, "legacy").strip().lower()
    return value if value in _VALID else "legacy"


def resolved_backend() -> str:
    """Resolve ``auto`` and validate an explicit ``tm_ref`` request."""

    selected = selected_backend()
    if selected == "tm_ref":
        if _load_tm_ref() is None:
            raise RuntimeError(
                f"MEASURE_TEMPLATE_BACKEND=tm_ref but tm_ref is not importable: "
                f"{_TM_REF_ERROR}"
            )
        return "tm_ref"
    if selected == "auto":
        return "tm_ref" if _load_tm_ref() is not None else "legacy"
    return "legacy"


def available() -> bool:
    """Return whether the optional reference backend can be imported."""

    return _load_tm_ref() is not None


def build_info() -> Dict[str, Any]:
    """Return backend identity and version information for diagnostics."""

    module = _load_tm_ref()
    return {
        "selected": selected_backend(),
        "resolved": resolved_backend(),
        "tm_ref_available": module is not None,
        "tm_ref_version": getattr(module, "__version__", None) if module else None,
        "tm_ref_error": _TM_REF_ERROR,
    }


def _legacy_correlate(
    search: np.ndarray,
    template: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Reference correlation path, kept identical to the former helper."""

    area = np.ascontiguousarray(search, dtype=np.float32)
    variant = np.ascontiguousarray(template, dtype=np.float32)
    if mask is None:
        scores = cv2.matchTemplate(area, variant, cv2.TM_CCOEFF_NORMED)
    else:
        scores = cv2.matchTemplate(
            area,
            variant,
            cv2.TM_CCOEFF_NORMED,
            mask=np.ascontiguousarray(mask, dtype=np.float32),
        )
    return np.nan_to_num(scores, nan=-1.0, posinf=-1.0, neginf=-1.0)


def correlate(
    search: np.ndarray,
    template: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return the NCC score map using the configured backend."""

    backend = resolved_backend()
    if backend == "tm_ref":
        module = _load_tm_ref()
        assert module is not None
        return module.correlate(search, template, mask)
    return _legacy_correlate(search, template, mask)