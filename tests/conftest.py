"""
Shared pytest configuration and fixtures for the measurement test suite.

Provides:
  * ``--headless`` CLI option (and ``HEADLESS`` env var) to suppress all
    ``cv2.imshow`` / ``cv2.waitKey`` calls.
  * ``display_or_save()`` — single entry point that ALWAYS saves the image
    to ``tests/output/<test_name>/`` and optionally shows it.
  * ``test_output_dir`` fixture — creates a per-test output directory.
"""

import os
from pathlib import Path

import cv2
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Headless mode
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    parser.addoption(
        "--headless",
        action="store_true",
        default=True,
        help="Suppress all cv2.imshow / cv2.waitKey calls (default: True).",
    )
    parser.addoption(
        "--show",
        action="store_true",
        default=False,
        help="Show cv2 windows during tests.",
    )


@pytest.fixture(scope="session", autouse=True)
def _apply_headless(request) -> None:
    """Monkey-patch cv2 by default; use --show to see windows."""
    show = request.config.getoption("--show")
    if not show:
        from measurement import _apply_headless_patch
        _apply_headless_patch()


@pytest.fixture(scope="session")
def headless(request) -> bool:
    return not request.config.getoption("--show")


# ---------------------------------------------------------------------------
# Output directory fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def test_output_dir(request) -> Path:
    """Create a per-test output directory under tests/output/."""
    test_name = request.node.name.replace("[", "_").replace("]", "_").replace("/", "_")
    out_dir = Path(__file__).parent / "output" / test_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


# ---------------------------------------------------------------------------
# Unified display-or-save helper
# ---------------------------------------------------------------------------

def display_or_save(
    image: np.ndarray,
    filename: str,
    output_dir: Path,
    *,
    headless: bool = True,
    window_name: str = "debug",
    wait_time: int = 0,
) -> None:
    """Save *image* to *output_dir* / *filename*.

    If *headless* is ``False``, also show it via ``cv2.imshow`` and wait
    for *wait_time* ms.
    """
    path = output_dir / filename
    cv2.imwrite(str(path), image)
    if not headless:
        cv2.imshow(window_name, image)
        cv2.waitKey(wait_time)
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Shared data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sample_data_dir() -> Path:
    return Path(__file__).parent.parent / "data" / "sample"
