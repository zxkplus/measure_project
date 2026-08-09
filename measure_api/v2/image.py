"""Fixed-layout POSIX shared-memory JPEG reader for the v2 engine."""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import resource_tracker, shared_memory
from typing import Any, Callable, Dict, Mapping, Optional

import cv2
import numpy as np

from measure_api.v2.application import EngineError

SLOT_COUNT = 30
SLOT_SIZE = 20 * 1024 * 1024
POOL_SIZE = SLOT_COUNT * SLOT_SIZE
DEFAULT_POOL_NAME = "measure_v2_img_shm"


@dataclass(frozen=True)
class ImageDescriptor:
    """The shared JPEG request fields common to both measurement engines."""

    uuid: str
    camera_sn: str
    type: str
    height: int
    width: int
    channel: int
    jpeg_size: int
    bin_index: int
    timestamp: Any

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ImageDescriptor":
        """Parse and validate the fixed v2 image request structure."""
        expected = {
            "uuid",
            "camera_sn",
            "type",
            "height",
            "width",
            "channel",
            "jpeg_size",
            "bin_index",
            "timestamp",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise EngineError(
                "image.descriptor_invalid",
                "request",
                "Image descriptor does not match the v2 contract",
                details={
                    "missing": sorted(expected - set(value)),
                    "unknown": sorted(set(value) - expected),
                }
                if isinstance(value, Mapping)
                else {},
            )
        integer_fields = ("height", "width", "channel", "jpeg_size", "bin_index")
        if any(
            isinstance(value[name], bool) or not isinstance(value[name], int)
            for name in integer_fields
        ):
            raise EngineError(
                "image.descriptor_invalid",
                "request",
                "Image descriptor contains invalid field types",
            )
        try:
            descriptor = cls(
                uuid=str(value["uuid"]),
                camera_sn=str(value["camera_sn"]),
                type=str(value["type"]),
                height=int(value["height"]),
                width=int(value["width"]),
                channel=int(value["channel"]),
                jpeg_size=int(value["jpeg_size"]),
                bin_index=int(value["bin_index"]),
                timestamp=value["timestamp"],
            )
        except (TypeError, ValueError) as exc:
            raise EngineError(
                "image.descriptor_invalid",
                "request",
                "Image descriptor contains invalid field types",
            ) from exc
        if descriptor.type != "jpg" or descriptor.channel != 1:
            raise EngineError(
                "image.format_unsupported",
                "image",
                "Only JPEG channel 1 is supported",
            )
        if not descriptor.uuid or not descriptor.camera_sn:
            raise EngineError(
                "image.descriptor_invalid",
                "request",
                "uuid and camera_sn are required",
            )
        if not 0 <= descriptor.bin_index < SLOT_COUNT:
            raise EngineError("image.descriptor_invalid", "request", "bin_index is out of range")
        if not 0 < descriptor.jpeg_size <= SLOT_SIZE:
            raise EngineError("image.descriptor_invalid", "request", "jpeg_size is out of range")
        return descriptor

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable descriptor."""
        return {
            "uuid": self.uuid,
            "camera_sn": self.camera_sn,
            "type": self.type,
            "height": self.height,
            "width": self.width,
            "channel": self.channel,
            "jpeg_size": self.jpeg_size,
            "bin_index": self.bin_index,
            "timestamp": self.timestamp,
        }


class SharedMemoryImageReader:
    """Decode a private GRAY8 snapshot from the engine's fixed image pool."""

    def __init__(
        self,
        name: str = DEFAULT_POOL_NAME,
        *,
        opener: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.name = name
        self._opener = opener or shared_memory.SharedMemory

    def snapshot(self, value: Mapping[str, Any]) -> np.ndarray:
        """Read, decode, copy, and release one temporary image reference."""
        descriptor = ImageDescriptor.from_mapping(value)
        segment = self._open()
        try:
            self._validate_pool(segment)
            offset = descriptor.bin_index * SLOT_SIZE
            encoded = bytes(segment.buf[offset : offset + descriptor.jpeg_size])
        finally:
            segment.close()
        image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise EngineError("image.decode_failed", "image", "JPEG image could not be decoded")
        return np.ascontiguousarray(image, dtype=np.uint8).copy()

    def ensure_pool(self) -> None:
        """Attach or create the fixed pool and fail fast on size mismatch."""
        segment = self._open()
        try:
            self._validate_pool(segment)
        finally:
            segment.close()

    @staticmethod
    def _validate_pool(segment: Any) -> None:
        if segment.size != POOL_SIZE or len(segment.buf) != POOL_SIZE:
            raise EngineError(
                "image.pool_size_mismatch",
                "image",
                "Shared-memory image pool has an unexpected size",
                details={"expected": POOL_SIZE, "actual": segment.size},
            )

    def _open(self) -> Any:
        try:
            return self._without_unlink_tracking(
                self._opener(name=self.name, create=False)
            )
        except FileNotFoundError:
            try:
                return self._without_unlink_tracking(
                    self._opener(name=self.name, create=True, size=POOL_SIZE)
                )
            except OSError as exc:
                raise EngineError(
                    "image.pool_unavailable",
                    "image",
                    "Shared-memory image pool is unavailable",
                    retryable=True,
                ) from exc
        except OSError as exc:
            raise EngineError(
                "image.pool_unavailable",
                "image",
                "Shared-memory image pool is unavailable",
                retryable=True,
            ) from exc

    def _without_unlink_tracking(self, segment: Any) -> Any:
        if self._opener is shared_memory.SharedMemory:
            resource_tracker.unregister(segment._name, "shared_memory")
        return segment
