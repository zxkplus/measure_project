"""Framework-free pixel preprocessing shared by teaching, matching and APIs.

Every spec returns owned pixels: the input array is never modified and the
output never shares memory with it.  Channel handling is explicit — a
"preserve" raw preset keeps colour and cannot silently enter grayscale matching,
while the grayscale presets document their BGR-to-luma conversion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union

import cv2
import numpy as np

from measure.computation_types import Array, contract_error, require_finite

THRESHOLD_MODES = {
    "binary": cv2.THRESH_BINARY,
    "binary_inv": cv2.THRESH_BINARY_INV,
    "truncate": cv2.THRESH_TRUNC,
    "tozero": cv2.THRESH_TOZERO,
    "tozero_inv": cv2.THRESH_TOZERO_INV,
}


@dataclass(frozen=True)
class RawSpec:
    """No enhancement; ``gray`` converts BGR to luma, ``preserve`` keeps channels."""

    channel_mode: str = "gray"

    def __post_init__(self) -> None:
        if self.channel_mode not in ("gray", "preserve"):
            raise contract_error("contract.invalid_parameter",
                                 f"unknown channel mode {self.channel_mode!r}",
                                 "preprocessing.channel_mode")


@dataclass(frozen=True)
class ThresholdSpec:
    """Global threshold with an explicit OpenCV mode."""

    threshold: float = 128.0
    mode: str = "binary"

    def __post_init__(self) -> None:
        require_finite(self.threshold, "threshold.threshold")
        if self.mode not in THRESHOLD_MODES:
            raise contract_error("contract.invalid_parameter",
                                 f"unknown threshold mode {self.mode!r}",
                                 "preprocessing.mode")


@dataclass(frozen=True)
class SobelSpec:
    """Sobel gradient magnitude kept in ``float32`` (no 8-bit saturation)."""

    kernel_size: int = 3
    order_x: int = 1
    order_y: int = 0
    magnitude: bool = False

    def __post_init__(self) -> None:
        if self.kernel_size not in (1, 3, 5, 7):
            raise contract_error("contract.invalid_parameter",
                                 "kernel_size must be 1, 3, 5 or 7",
                                 "preprocessing.kernel_size")
        if self.order_x + self.order_y <= 0:
            raise contract_error("contract.invalid_parameter",
                                 "at least one derivative order must be non-zero",
                                 "preprocessing.order")


@dataclass(frozen=True)
class CannySpec:
    """Canny edges as binary ``uint8``."""

    threshold1: float = 50.0
    threshold2: float = 150.0

    def __post_init__(self) -> None:
        require_finite((self.threshold1, self.threshold2), "canny.thresholds")


@dataclass(frozen=True)
class ClaheSpec:
    """Contrast-limited adaptive equalisation in ``float32``."""

    clip_limit: float = 2.0
    tile_grid_size: int = 8

    def __post_init__(self) -> None:
        require_finite(self.clip_limit, "clahe.clip_limit")
        if self.clip_limit <= 0.0:
            raise contract_error("contract.invalid_parameter", "clip_limit must be positive",
                                 "preprocessing.clip_limit")
        if self.tile_grid_size < 1:
            raise contract_error("contract.invalid_parameter", "tile_grid_size must be positive",
                                 "preprocessing.tile_grid_size")


PreprocessingSpec = Union[RawSpec, ThresholdSpec, SobelSpec, CannySpec, ClaheSpec]


def _require_image(image: Array) -> Array:
    """Validate the accepted pixel layouts and return the array."""
    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3):
        raise contract_error("contract.invalid_shape", "image must be 2D or 3D", "image")
    if image.dtype != np.uint8:
        raise contract_error("contract.invalid_parameter", "image must be uint8", "image")
    if image.ndim == 3 and image.shape[2] not in (1, 3):
        raise contract_error("contract.invalid_shape",
                             "image must be GRAY8 or BGR8", "image")
    if image.size == 0:
        raise contract_error("contract.invalid_shape", "image must not be empty", "image")
    return image


def _gray(image: Array) -> Array:
    """Return an owned grayscale ``uint8`` view of a 2D or BGR image."""
    if image.ndim == 2:
        return image
    if image.shape[2] == 1:
        return image[:, :, 0]
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def to_display_uint8(pixels: Array) -> Array:
    """Scale arbitrary numeric pixels into an 8-bit display array.

    Args:
        pixels: Numeric pixels of any finite dtype; a ``uint8`` input is
            returned unchanged.

    Returns:
        The pixels as ``uint8``, min-max normalised when needed.
    """
    array = np.ascontiguousarray(pixels)
    if array.dtype == np.uint8:
        return array
    return cv2.normalize(array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def preprocess_pixels(image: Array, spec: PreprocessingSpec) -> Array:
    """Apply one preprocessing spec and return owned pixels.

    Args:
        image: ``uint8`` GRAY8 or BGR8 image.
        spec: One of the declared preprocessing specs.

    Returns:
        A new array whose dtype matches the spec: ``float32`` for raw, Sobel and
        CLAHE, ``uint8`` for threshold and Canny.

    Raises:
        ComputationContractError: On unsupported image layout or spec type.
    """
    values = _require_image(image)
    if isinstance(spec, RawSpec):
        if spec.channel_mode == "preserve":
            if values.ndim == 2:
                return np.array(values, dtype=np.float32, copy=True)
            return np.array(values, dtype=np.float32, copy=True)
        return np.array(_gray(values), dtype=np.float32, copy=True)
    if isinstance(spec, ThresholdSpec):
        gray = _gray(values)
        _, result = cv2.threshold(gray, float(spec.threshold), 255, THRESHOLD_MODES[spec.mode])
        return np.array(result, dtype=np.uint8, copy=True)
    if isinstance(spec, SobelSpec):
        gray = _gray(values)
        result = cv2.Sobel(gray, cv2.CV_32F, spec.order_x, spec.order_y,
                           ksize=spec.kernel_size)
        if spec.magnitude:
            other = cv2.Sobel(gray, cv2.CV_32F, spec.order_y, spec.order_x,
                              ksize=spec.kernel_size)
            result = cv2.magnitude(result, other)
        return np.array(result, dtype=np.float32, copy=True)
    if isinstance(spec, CannySpec):
        gray = _gray(values)
        result = cv2.Canny(gray, float(spec.threshold1), float(spec.threshold2))
        return np.array(result, dtype=np.uint8, copy=True)
    if isinstance(spec, ClaheSpec):
        gray = _gray(values)
        clahe = cv2.createCLAHE(clipLimit=float(spec.clip_limit),
                                tileGridSize=(int(spec.tile_grid_size),
                                              int(spec.tile_grid_size)))
        return np.array(clahe.apply(gray), dtype=np.float32, copy=True)
    raise contract_error("contract.invalid_parameter",
                         f"unsupported preprocessing spec {type(spec).__name__!r}", "spec")