"""External-codec boundary regression for decoded shared-file cache upgrades."""

import cv2
import numpy as np

from measure_api.v2.files import SharedFiles
from measure_api.v2.models import FileReference, ServiceSettings


def test_decoder_version_change_recomputes_pixels_in_existing_state_directory(
    tmp_path, monkeypatch
):
    pixels = np.zeros((20, 30), np.uint8)
    cv2.imwrite(str(tmp_path / "source.png"), pixels)
    settings = ServiceSettings(
        shared_roots={"exchange": tmp_path}, state_directory=tmp_path / "state"
    )
    reference = FileReference(root="exchange", path="source.png")
    before, identity = SharedFiles(settings).read_image(reference)
    assert before[0, 0] == 0
    # Simulate an external codec upgrade with different output for the same encoded bytes.
    monkeypatch.setattr(cv2, "__version__", "new-external-codec")
    monkeypatch.setattr(cv2, "imdecode", lambda data, flags: np.full((20, 30), 77, np.uint8))
    after, changed = SharedFiles(settings).read_image(reference)
    assert changed.sha256 == identity.sha256
    assert after[0, 0] == 77
