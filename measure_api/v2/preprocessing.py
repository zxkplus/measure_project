"""Reusable preprocessing definitions shared by teaching and matching."""

from __future__ import annotations

from typing import Annotated, Literal, Union

import cv2
import numpy as np
from pydantic import Field, model_validator

from measure_api.v2.models import StrictModel


class RawPreprocessing(StrictModel):
    """Preserve source intensities as floating numeric samples."""

    type: Literal["raw"] = "raw"


class CannyPreprocessing(StrictModel):
    """Detect binary edges with explicit hysteresis thresholds."""

    type: Literal["canny"] = "canny"
    low_threshold: float = Field(default=50, ge=0, le=255)
    high_threshold: float = Field(default=150, ge=0, le=255)

    @model_validator(mode="after")
    def check_thresholds(self) -> CannyPreprocessing:
        """Require monotonic hysteresis thresholds.

        Returns:
            This Canny preset after checking ordered hysteresis thresholds.
        """
        if self.high_threshold < self.low_threshold:
            raise ValueError("High threshold must be at least the low threshold")
        return self


class SobelPreprocessing(StrictModel):
    """Keep gradient magnitude as float32 rather than clipping it to 255."""

    type: Literal["sobel"] = "sobel"
    kernel_size: Literal[1, 3, 5, 7] = 3


class ClahePreprocessing(StrictModel):
    """Improve local grayscale contrast with bounded amplification."""

    type: Literal["clahe"] = "clahe"
    clip_limit: float = Field(default=2.0, gt=0, le=100)
    tile_grid_size: int = Field(default=8, ge=1, le=64)


class ThresholdPreprocessing(StrictModel):
    """Apply an explicit global threshold mode."""

    type: Literal["threshold"] = "threshold"
    threshold: float = Field(default=127, ge=0, le=255)
    mode: Literal["binary", "binary_inv", "truncate", "tozero", "tozero_inv"] = "binary"


Preprocessing = Annotated[Union[
    RawPreprocessing, CannyPreprocessing, SobelPreprocessing, ClahePreprocessing,
    ThresholdPreprocessing,
], Field(discriminator="type")]


def to_gray(image: np.ndarray) -> np.ndarray:
    """Convert BGR input once, leaving already gray input unchanged.

    Args:
        image: Decoded source pixels; gray arrays remain in their original sample frame.

    Returns:
        Gray pixel array; a BGR array is converted using the declared OpenCV convention.
    """
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_preprocessing(image: np.ndarray, options: Preprocessing) -> np.ndarray:
    """Apply the same definition to reference and inspection image data.

    Args:
        image: Decoded source pixels; gray arrays remain in their original sample frame.
        options: Typed preprocessing preset applied consistently to both image inputs.

    Returns:
        Numeric processed pixels, preserving the preset-specific sample dtype.
    """
    if options.type == "raw":
        return image.astype(np.float32)
    gray = to_gray(image)
    if options.type == "canny":
        return cv2.Canny(gray, options.low_threshold, options.high_threshold)
    if options.type == "sobel":
        dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=options.kernel_size)
        dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=options.kernel_size)
        return cv2.magnitude(dx, dy)
    if options.type == "clahe":
        clahe = cv2.createCLAHE(
            options.clip_limit, (options.tile_grid_size, options.tile_grid_size),
        )
        return clahe.apply(gray).astype(np.float32)
    modes = {
        "binary": cv2.THRESH_BINARY, "binary_inv": cv2.THRESH_BINARY_INV,
        "truncate": cv2.THRESH_TRUNC, "tozero": cv2.THRESH_TOZERO,
        "tozero_inv": cv2.THRESH_TOZERO_INV,
    }
    return cv2.threshold(gray, options.threshold, 255, modes[options.mode])[1]


def display_preview(image: np.ndarray) -> np.ndarray:
    """Produce an independent 8-bit display representation of numeric samples.

    Args:
        image: Decoded source pixels; gray arrays remain in their original sample frame.

    Returns:
        Independent 8-bit visualization of the numeric input sample array.
    """
    if image.dtype == np.uint8:
        return image.copy()
    if image.size == 0:
        return image.astype(np.uint8)
    low, high = float(image.min()), float(image.max())
    if 0 <= low <= high <= 255:
        return np.clip(image, 0, 255).astype(np.uint8)
    if high == low:
        return np.zeros(image.shape, dtype=np.uint8)
    return np.clip((image - low) * (255 / (high - low)), 0, 255).astype(np.uint8)
