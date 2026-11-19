"""Guard: v2 must not implement numeric algorithms itself.

Everything algorithmic belongs to ``measure/``; the API layer maps typed
requests onto those entry points. This test walks the AST of every v2 module
and fails on a call into a numeric library routine, so a later change cannot
quietly reintroduce a second implementation.

An entry may be allowed with a written reason in ``ALLOWLIST``; the intent is
for that map to stay empty.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List, Set

import pytest

V2_ROOT = Path(__file__).resolve().parents[2] / "measure_api" / "v2"

#: NumPy routines that implement an algorithm rather than a data move.
FORBIDDEN_NUMPY = {
    "np.linalg.inv", "np.linalg.lstsq", "np.linalg.norm", "np.linalg.det",
    "np.linalg.matrix_rank", "np.linalg.solve", "np.linalg.eig", "np.linalg.svd",
    "np.gradient", "np.polyfit", "np.arctan2",
    # Unit conversions (np.deg2rad/np.rad2deg) are deliberately not listed:
    # they are the same as math.radians and carry no algorithm.
}

#: OpenCV routines that implement an algorithm rather than encode a file.
FORBIDDEN_CV2 = {
    "cv2.matchTemplate", "cv2.calibrateCamera", "cv2.stereoCalibrate",
    "cv2.undistort", "cv2.undistortPoints", "cv2.undistortPointsIter",
    "cv2.initUndistortRectifyMap", "cv2.projectPoints", "cv2.solvePnP",
    "cv2.Rodrigues", "cv2.findCirclesGrid", "cv2.findChessboardCorners",
    "cv2.cornerSubPix", "cv2.estimateAffinePartial2D", "cv2.estimateAffine2D",
    "cv2.findHomography", "cv2.decomposeHomographyMat", "cv2.warpAffine",
    "cv2.warpPerspective", "cv2.remap", "cv2.GaussianBlur", "cv2.minMaxLoc",
    "cv2.meanStdDev", "cv2.rotatedRectangleIntersection", "cv2.contourArea",
    "cv2.normalize", "cv2.getOptimalNewCameraMatrix",
}

#: ``file -> {dotted name: reason}``; a tolerant allowlist that must stay empty.
ALLOWLIST: Dict[str, Dict[str, str]] = {}


def _dotted(node: ast.AST) -> str:
    """Return the dotted name of an attribute chain, or an empty string."""
    parts: List[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return ""
    parts.append(current.id)
    return ".".join(reversed(parts))


def _calls(path: Path) -> Set[str]:
    """Return every dotted call name used by one module."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _dotted(node.func)
            if name:
                found.add(name)
    return found


def _modules() -> List[Path]:
    """Return every v2 module."""
    return sorted(V2_ROOT.rglob("*.py"))


def test_v2_does_not_call_numeric_algorithms():
    """Every numeric algorithm call lives in measure/, not in measure_api/v2."""
    violations: List[str] = []
    for path in _modules():
        relative = str(path.relative_to(V2_ROOT))
        allowed = ALLOWLIST.get(relative, {})
        for name in sorted(_calls(path)):
            if name in allowed:
                continue
            if name in FORBIDDEN_NUMPY or name in FORBIDDEN_CV2:
                violations.append(f"{relative}: {name}")
    assert not violations, (
        "v2 must delegate numeric work to measure/:\n  " + "\n  ".join(violations)
    )


@pytest.mark.parametrize("name", sorted(FORBIDDEN_NUMPY | FORBIDDEN_CV2))
def test_guard_can_actually_see_forbidden_calls(name):
    """The scanner must detect a synthetic module that calls a forbidden name."""
    source = f"import numpy as np\nimport cv2\n\ndef f(value):\n    return {name}(value)\n"
    tree = ast.parse(source)
    calls = {_dotted(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
    assert name in calls