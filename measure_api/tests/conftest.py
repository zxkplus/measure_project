"""
Shared test fixtures for Measure API tests.
"""

from __future__ import annotations

import os
import tempfile
from typing import Generator, Tuple

import cv2
import numpy as np
import pytest


@pytest.fixture(scope="function")
def temp_dir() -> Generator[str, None, None]:
    """Create a temporary directory for test output."""
    with tempfile.TemporaryDirectory(prefix="measure_api_test_") as tmp:
        yield tmp


@pytest.fixture(scope="function")
def sample_image() -> np.ndarray:
    """
    Create a synthetic 200x200 grayscale image with a bright circle.
    Useful for testing measurement objects.
    """
    img = np.zeros((200, 200), dtype=np.uint8) + 128
    cv2.circle(img, (100, 100), 60, 200, -1)  # bright circle
    cv2.circle(img, (100, 100), 40, 50, -1)   # dark inner circle
    # Add some noise
    noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
    img = cv2.addWeighted(img, 1.0, noise, 0.3, 0)
    return img


@pytest.fixture(scope="function")
def sample_reference_path(temp_dir: str, sample_image: np.ndarray) -> str:
    """Save sample_image as reference.png in temp_dir."""
    path = os.path.join(temp_dir, "reference.png")
    cv2.imwrite(path, sample_image)
    return path


@pytest.fixture(scope="function")
def sample_inspection_path(temp_dir: str, sample_image: np.ndarray) -> str:
    """Save sample_image as inspection.png in temp_dir."""
    path = os.path.join(temp_dir, "inspection.png")
    cv2.imwrite(path, sample_image)
    return path


@pytest.fixture(scope="function")
def setup_config() -> Generator[None, None, None]:
    """Set up a minimal in-memory config for testing."""
    from measure_api.config import Config
    Config.reset()
    cfg = Config.load_from_dict({
        "log": {
            "directory": "/tmp/measure_api_test_logs",
            "level": "DEBUG",
            "console_output": False,
            "backup_days": 1,
        },
        "server": {
            "host": "127.0.0.1",
            "port": 0,
            "max_sessions": 10,
        },
        "call_records": {
            "enabled": False,
        },
    })
    yield
    Config.reset()
