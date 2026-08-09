"""Contract tests for the v2 shared-memory image reader."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from measure_api.v2.application import EngineError
from measure_api.v2.image import POOL_SIZE, SLOT_SIZE, SharedMemoryImageReader


class _SparseBuffer:
    def __init__(self, offset: int, payload: bytes, total_size: int = POOL_SIZE) -> None:
        self.offset = offset
        self.payload = payload
        self.total_size = total_size

    def __len__(self) -> int:
        return self.total_size

    def __getitem__(self, key: slice) -> bytes:
        assert isinstance(key, slice)
        assert key.start == self.offset
        return self.payload[: key.stop - key.start]


class _FakeSharedMemory:
    def __init__(self, payload: bytes, bin_index: int, size: int = POOL_SIZE) -> None:
        self.size = size
        self.buf = _SparseBuffer(bin_index * SLOT_SIZE, payload, size)
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _descriptor(jpeg: bytes, bin_index: int = 7) -> dict:
    return {
        "uuid": "frame-1",
        "camera_sn": "camera-a",
        "type": "jpg",
        "height": 12,
        "width": 16,
        "channel": 1,
        "jpeg_size": len(jpeg),
        "bin_index": bin_index,
        "timestamp": 1786243200,
    }


def test_snapshot_decodes_private_contiguous_gray_image() -> None:
    source = np.zeros((12, 16), dtype=np.uint8)
    source[:, 8:] = 255
    encoded, jpeg = cv2.imencode(".jpg", source)
    assert encoded
    shared = _FakeSharedMemory(jpeg.tobytes(), bin_index=7)
    reader = SharedMemoryImageReader("measure_v2_img_shm", opener=lambda **_: shared)

    snapshot = reader.snapshot(_descriptor(jpeg.tobytes()))
    shared.buf.payload = b"\x00" * len(jpeg)

    assert snapshot.shape == source.shape
    assert snapshot.dtype == np.uint8
    assert snapshot.flags.c_contiguous
    assert int(snapshot[:, 8:].mean()) > 240
    assert shared.closed


def test_snapshot_rejects_existing_pool_with_wrong_size() -> None:
    source = np.zeros((4, 4), dtype=np.uint8)
    _, jpeg = cv2.imencode(".jpg", source)
    shared = _FakeSharedMemory(jpeg.tobytes(), bin_index=1, size=POOL_SIZE - 1)
    reader = SharedMemoryImageReader("measure_v2_img_shm", opener=lambda **_: shared)

    with pytest.raises(EngineError) as error:
        reader.snapshot(_descriptor(jpeg.tobytes(), bin_index=1))

    assert error.value.error["code"] == "image.pool_size_mismatch"
