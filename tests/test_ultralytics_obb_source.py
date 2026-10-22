"""Tests for the ultralytics OBB adapter (``measure_obb.ultralytics_source``).

The adapter is the boundary between ultralytics' ``(x, y)`` tensors and the
engine's ``(row, col)`` arrays, so these tests exercise the conversion against
ultralytics' own tensor types rather than numpy stand-ins — a plain array would
hide a ``.detach().cpu()`` mistake.

No model file is needed: the ``Results`` object is built directly. Real inference
is out of scope here (it needs weights and a GPU), but the field contract this
relies on — ``obb.xyxyxyxy`` / ``cls`` / ``conf`` as tensors — was verified against
a genuine forward pass with ``yolov8n-obb``.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch",
                            reason="adapter tests need ultralytics' tensor types")
pytest.importorskip("ultralytics", reason="ultralytics is not installed")

from measure.obb_source import ObbObservation, ObbSource  # noqa: E402
from measure_obb.ultralytics_source import UltralyticsObbSource  # noqa: E402

CY, CX, H, W = 300.0, 400.0, 140.0, 200.0


def quad_xy(angle_deg: float = 0.0, *, cy=CY, cx=CX, h=H, w=W):
    """Corners in ultralytics' ``(x, y)`` order, rotated in image coordinates.

    Image ``y`` grows downwards, while the engine's angle convention is
    "visually counter-clockwise positive"; this helper therefore negates the sine
    term.  Calibrated against ``measure.rectify.compute_rotated_box_corners``,
    which is the authoritative definition.
    """
    theta = math.radians(angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    out = []
    for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2),
                   (w / 2, h / 2), (-w / 2, h / 2)):
        out.append([cx + dx * cos_t + dy * sin_t,
                    cy - dx * sin_t + dy * cos_t])
    return out


def fake_result(quads, cls, conf):
    """A minimal stand-in for one ultralytics ``Results``.

    Uses real tensors so the adapter's tensor handling is exercised.
    """
    return SimpleNamespace(obb=SimpleNamespace(
        xyxyxyxy=torch.tensor(quads, dtype=torch.float32).reshape(-1, 4, 2),
        cls=torch.tensor(cls, dtype=torch.float32),
        conf=torch.tensor(conf, dtype=torch.float32),
    ))


def make_source(class_names=None, **overrides):
    """An adapter without loading a model, for conversion-only tests."""
    source = UltralyticsObbSource.__new__(UltralyticsObbSource)
    source._class_names = class_names if class_names is not None else {
        0: "oritent", 1: "bottle_center"}
    source.name = "ultralytics-obb"
    source.requires_roi = False
    source.model_path = overrides.get("model_path", "/tmp/model.pt")
    source.conf = overrides.get("conf", 0.25)
    source.imgsz = overrides.get("imgsz", 640)
    source.device = overrides.get("device")
    source.task = "obb"
    source.model_sha256 = overrides.get("sha256", "a" * 64)
    return source


class TestCoordinateConversion:
    """``(x, y)`` from ultralytics, ``(row, col)`` in the engine."""

    def test_centre_and_size_are_transposed(self):
        obs = make_source()._to_observations(
            fake_result([quad_xy()], [0], [0.9]))[0]
        assert obs.center == pytest.approx((CY, CX))
        # size is (height, width): rows first, as crop_and_straighten expects.
        assert obs.size == pytest.approx((H, W))

    @pytest.mark.parametrize("angle", [0.0, 30.0, -45.0, 75.0, -89.0])
    def test_angle_round_trips_through_image_coordinates(self, angle):
        """The sign convention is the easiest thing to get backwards."""
        obs = make_source()._to_observations(
            fake_result([quad_xy(angle)], [0], [0.9]))[0]
        assert obs.angle_deg == pytest.approx(angle, abs=1e-3)

    def test_metadata_is_preserved(self):
        obs = make_source()._to_observations(
            fake_result([quad_xy()], [1], [0.42]))[0]
        assert obs.class_id == 1
        assert obs.class_name == "bottle_center"
        assert obs.score == pytest.approx(0.42, abs=1e-6)

    def test_multiple_detections(self):
        obs = make_source()._to_observations(
            fake_result([quad_xy(), quad_xy(30.0)], [0, 1], [0.9, 0.5]))
        assert len(obs) == 2
        assert [o.class_id for o in obs] == [0, 1]
        assert obs[1].angle_deg == pytest.approx(30.0, abs=1e-3)

    def test_unknown_class_has_no_name(self):
        obs = make_source({0: "oritent"})._to_observations(
            fake_result([quad_xy()], [7], [0.5]))[0]
        assert obs.class_id == 7 and obs.class_name is None


class TestEmptyAndMalformedOutput:
    def test_no_detections(self):
        assert make_source()._to_observations(fake_result([], [], [])) == []

    def test_missing_obb_attribute(self):
        assert make_source()._to_observations(SimpleNamespace()) == []

    def test_none_xyxyxyxy(self):
        result = SimpleNamespace(obb=SimpleNamespace(xyxyxyxy=None))
        assert make_source()._to_observations(result) == []


class TestImagePreparation:
    @pytest.mark.parametrize("shape,expected", [
        ((100, 200), (100, 200, 3)),          # grayscale -> 3 channels
        ((100, 200, 3), (100, 200, 3)),       # BGR passthrough
        ((100, 200, 4), (100, 200, 3)),       # BGRA drops alpha
    ])
    def test_accepted_shapes(self, shape, expected):
        prepared = UltralyticsObbSource._prepare(np.zeros(shape, np.uint8))
        assert prepared.shape == expected

    def test_rejects_nonsense_shape(self):
        with pytest.raises(ValueError, match="2-D grayscale or 3-channel"):
            UltralyticsObbSource._prepare(np.zeros((5, 6, 7, 8), np.uint8))

    def test_grayscale_channels_are_identical(self):
        """Ultralytics expects three luma channels, not zeros or duplicates."""
        gray = np.arange(12, dtype=np.uint8).reshape(3, 4)
        prepared = UltralyticsObbSource._prepare(gray)
        for channel in range(3):
            assert np.array_equal(prepared[:, :, channel], gray)


class TestProtocolConformance:
    def test_satisfies_the_protocol(self):
        assert isinstance(make_source(), ObbSource)

    def test_never_requires_an_roi(self):
        """The whole point of a detector: it finds its own boxes."""
        assert make_source().requires_roi is False


class TestSerialization:
    def test_records_identity_and_settings(self):
        payload = make_source(conf=0.4, imgsz=1024).serialize()
        assert payload["name"] == "ultralytics-obb"
        assert payload["conf"] == pytest.approx(0.4)
        assert payload["imgsz"] == 1024
        assert payload["sha256"] == "a" * 64
        assert payload["class_names"] == {"0": "oritent", "1": "bottle_center"}

    def test_verify_passes_on_an_unchanged_model(self):
        source = make_source(sha256="b" * 64, imgsz=640)
        assert source.verify_matches(source.serialize()) is None

    def test_verify_flags_a_changed_digest(self):
        """A retrained weight under the same filename must not pass silently."""
        source = make_source(sha256="b" * 64)
        ref = source.serialize()
        ref["sha256"] = "c" * 64
        message = source.verify_matches(ref)
        assert message is not None and "digest changed" in message

    def test_verify_flags_a_changed_inference_size(self):
        source = make_source(imgsz=640)
        ref = source.serialize()
        ref["imgsz"] = 1024
        message = source.verify_matches(ref)
        assert message is not None and "inference size changed" in message

    def test_verify_flags_a_moved_model(self):
        source = make_source(model_path="/models/new.pt")
        ref = source.serialize()
        ref["path"] = "/models/old.pt"
        message = source.verify_matches(ref)
        assert message is not None and "path changed" in message

    def test_verify_tolerates_a_partial_reference(self):
        """An older manifest may predate some fields; that is not a mismatch."""
        assert make_source().verify_matches({}) is None


class TestConstructionGuards:
    def test_missing_file_raises_before_loading(self):
        with pytest.raises(FileNotFoundError, match="OBB model not found"):
            UltralyticsObbSource("/nonexistent/model.pt")

    def test_engine_import_does_not_pull_in_torch(self):
        """measure/ must stay importable where torch is absent.

        Checked by inspecting the engine's own module source rather than sys.modules,
        because the test process already imported torch for these tests.
        """
        import measure.alignment as alignment
        import measure.obb_source as obb_source
        for module in (alignment, obb_source):
            source = open(module.__file__, encoding="utf-8").read()
            for forbidden in ("import torch", "from torch", "import ultralytics"):
                assert forbidden not in source, \
                    f"{module.__name__} must not import {forbidden!r}"