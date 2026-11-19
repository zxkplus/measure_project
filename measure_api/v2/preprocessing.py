"""Reusable preprocessing definitions shared by teaching and matching."""

from __future__ import annotations

from measure_api.v2.timing import timed, span

from typing import Annotated, Literal, Union

import cv2
import numpy as np
from pydantic import Field, model_validator

from measure.preprocessing import (
    CannySpec,
    ClaheSpec,
    RawSpec,
    SobelSpec,
    ThresholdSpec,
    preprocess_pixels,
)
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


@timed("preprocessing")
def apply_preprocessing(image: np.ndarray, options: Preprocessing) -> np.ndarray:
    """Apply the same definition to reference and inspection image data.

    Args:
        image: Decoded source pixels; gray arrays remain in their original sample frame.
        options: Typed preprocessing preset applied consistently to both image inputs.

    Returns:
        Numeric processed pixels, preserving the preset-specific sample dtype.
    """
    if options.type == "raw":
        return preprocess_pixels(image, RawSpec(channel_mode="preserve"))
    if options.type == "canny":
        return preprocess_pixels(
            image,
            CannySpec(threshold1=options.low_threshold,
                      threshold2=options.high_threshold))
    if options.type == "sobel":
        return preprocess_pixels(
            image,
            SobelSpec(kernel_size=options.kernel_size, magnitude=True))
    if options.type == "clahe":
        return preprocess_pixels(
            image,
            ClaheSpec(clip_limit=options.clip_limit,
                      tile_grid_size=options.tile_grid_size))
    return preprocess_pixels(
        image, ThresholdSpec(threshold=options.threshold, mode=options.mode))


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
