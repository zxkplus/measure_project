"""Task-local pixel reuse preserves input validation, space isolation and memory bounds."""

import cv2
import numpy as np
import pytest

from measure_api.v2.errors import ServiceError
from measure_api.v2.files import OperationImages, SharedFiles
from measure_api.v2.models import FileReference, ServiceSettings


def files_for(tmp_path, limit=4096):
    settings = ServiceSettings(shared_roots={"exchange": tmp_path},
                               state_directory=tmp_path / "state", cache_max_bytes=limit)
    return SharedFiles(settings)


def test_repeat_image_and_gray_reads_share_readonly_arrays_but_recheck_content(tmp_path):
    pixels = np.full((20, 30, 3), 90, np.uint8)
    cv2.imwrite(str(tmp_path / "image.png"), pixels)
    files = files_for(tmp_path)
    reference = FileReference(root="exchange", path="image.png")
    first, verified = files.read_image(reference)
    second, _ = files.read_image(verified)
    assert first is second
    assert not first.flags.writeable
    gray = files.read_gray(verified)
    assert gray is files.read_gray(verified)
    assert files.cache_stats["gray_conversions"] == 1
    with pytest.raises(ValueError):
        first[0, 0] = 0
    cv2.imwrite(str(tmp_path / "image.png"), np.full_like(pixels, 123))
    with pytest.raises(ServiceError) as caught:
        files.read_image(verified)
    assert caught.value.code == "file.digest_mismatch"
    changed, _ = files.read_image(reference)
    assert np.all(changed == 123)


def test_shared_task_cache_distinguishes_different_image_contents(tmp_path):
    for name, value in [("first.png", 23), ("second.png", 77)]:
        cv2.imwrite(str(tmp_path / name), np.full((16, 16), value, np.uint8))
    first = files_for(tmp_path)
    second = SharedFiles(first.settings, image_cache=first.image_cache)
    a, _ = first.read_image(FileReference(root="exchange", path="first.png"))
    b, _ = second.read_image(FileReference(root="exchange", path="second.png"))
    assert np.all(a == 23) and np.all(b == 77)
    reused, _ = second.read_image(FileReference(root="exchange", path="first.png"))
    assert reused is a


def test_task_image_cache_evicts_and_never_exceeds_its_budget():
    cache = OperationImages(1024)
    cache.put("first", np.zeros((16, 32), np.uint8))
    cache.put("second", np.ones((16, 32), np.uint8))
    cache.put("third", np.zeros((16, 32), np.uint8))
    assert cache.get("first") is None
    assert cache.bytes == 1024
    cache.put("large", np.zeros((2048,), np.uint8))
    assert cache.get("large") is None
    assert cache.bytes <= 1024
