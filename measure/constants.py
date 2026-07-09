"""
Shared constants for the measurement package.

Centralises magic numbers and common colour / font defaults that were
previously inlined across ~25+ call sites.
"""

from pathlib import Path

import cv2

# ---------------------------------------------------------------------------
# Numerical tolerance
# ---------------------------------------------------------------------------
EPS: float = 1e-10

# ---------------------------------------------------------------------------
# Common BGR colours  (B, G, R)
# ---------------------------------------------------------------------------
BLACK: tuple[int, int, int] = (0, 0, 0)
WHITE: tuple[int, int, int] = (255, 255, 255)
GRAY: tuple[int, int, int] = (128, 128, 128)
RED: tuple[int, int, int] = (0, 0, 255)
GREEN: tuple[int, int, int] = (0, 255, 0)
BLUE: tuple[int, int, int] = (255, 0, 0)
CYAN: tuple[int, int, int] = (255, 255, 0)
MAGENTA: tuple[int, int, int] = (255, 0, 255)
YELLOW: tuple[int, int, int] = (0, 255, 255)
ORANGE: tuple[int, int, int] = (0, 165, 255)

# ---------------------------------------------------------------------------
# Drawing defaults
# ---------------------------------------------------------------------------
DEFAULT_FONT: int = cv2.FONT_HERSHEY_SIMPLEX
DEFAULT_FONT_SCALE: float = 0.5
DEFAULT_THICKNESS: int = 1
DEFAULT_LINE_TYPE: int = cv2.LINE_AA
DEFAULT_MARKER_TYPE: int = cv2.MARKER_CROSS

# ---------------------------------------------------------------------------
# Display / output control
# ---------------------------------------------------------------------------
DISPLAY_MODE: str = "show"  # 'show' | 'save' | 'both'

# Root directory for saved images when using save/both mode
OUTPUT_DIR: Path = Path.cwd() / "tests" / "output"


def set_display_mode(mode: str) -> None:
    global DISPLAY_MODE
    if mode not in ("show", "save", "both"):
        raise ValueError(f"Invalid display mode: {mode!r}")
    DISPLAY_MODE = mode


def set_output_dir(path: str | Path) -> None:
    global OUTPUT_DIR
    OUTPUT_DIR = Path(path)
