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
from pathlib import Path
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

# ---------------------------------------------------------------------------
# Real-model integration
# ---------------------------------------------------------------------------

_WEIGHTS = Path("/tmp/yolov8n-obb.pt")
_SAMPLE = Path(__file__).parent / "fixtures" / "obb_sample.jpg"

needs_real_model = pytest.mark.skipif(
    not _WEIGHTS.is_file() or not _SAMPLE.is_file(),
    reason="real OBB weights or sample image not present "
           "(see tests/fixtures/README.md)")


@needs_real_model
class TestRealModelIntegration:
    """Drives an actual YOLO-OBB model over an actual aerial photograph.

    Worth the runtime because stub-based tests cannot catch the class of defect
    this found: a float32 orthogonality residual (median 1.2e-6 over 166 real
    detections) exceeding the 1e-6 tolerance that seemed reasonable on paper,
    which rejected 57% of genuine detections.
    """

    @pytest.fixture(scope="class")
    def source(self):
        import cv2
        return UltralyticsObbSource(str(_WEIGHTS), conf=0.5, imgsz=1024), cv2

    def test_detects_and_converts_real_boxes(self, source):
        detector, cv2 = source
        image = cv2.imread(str(_SAMPLE))
        assert image is not None, "sample image failed to load"
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        observations = detector.detect(gray)
        assert len(observations) > 10, "expected many detections on this image"

        # Every observation must carry geometry the engine can crop with.
        for obs in observations:
            assert obs.height > 0 and obs.width > 0
            assert -90.0 <= obs.angle_deg < 90.0
            assert obs.class_name in detector.class_names.values()

    def test_grayscale_input_is_accepted_and_close_to_colour(self, source):
        """Grayscale works, but is NOT bit-identical to true colour input.

        The GUI reads reference images as grayscale, so the adapter replicates the
        single channel to three.  That is not the same tensor a real BGR image
        produces: measured on this sample, 169 detections vs 166, with scores
        differing in the third decimal (0.8581 vs 0.8640) throughout rather than
        only at the low-confidence tail.

        Documented instead of asserted away, because it is a real property callers
        should know: teaching on a grayscale reference and inspecting a colour one
        will not reproduce the same boxes exactly.  The test pins the weaker
        guarantee that actually holds — same order of magnitude, and the same
        dominant classes.
        """
        detector, cv2 = source
        image = cv2.imread(str(_SAMPLE))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        from_gray = detector.detect(gray)
        from_bgr = detector.detect(image)

        assert from_gray and from_bgr
        # Counts may differ slightly; they must not differ wildly.
        ratio = len(from_gray) / len(from_bgr)
        assert 0.5 < ratio < 2.0, f"grayscale gave {len(from_gray)}, colour {len(from_bgr)}"

        # The high-confidence structure must be the same.
        top_gray = sorted((o.score for o in from_gray), reverse=True)[:5]
        top_bgr = sorted((o.score for o in from_bgr), reverse=True)[:5]
        for a, b in zip(top_gray, top_bgr):
            assert a == pytest.approx(b, abs=0.05)

        # Both must agree that ships dominate.
        def dominant(obs):
            from collections import Counter
            return Counter(o.class_name for o in obs).most_common(1)[0][0]
        assert dominant(from_gray) == dominant(from_bgr) == "ship"

    def test_detects_multiple_classes(self, source):
        """The sample has ships and harbour, which is what makes it useful for
        exercising anchor-class selection in the GUI."""
        detector, cv2 = source
        image = cv2.cvtColor(cv2.imread(str(_SAMPLE)), cv2.COLOR_BGR2GRAY)
        classes = {o.class_id for o in detector.detect(image)}
        assert len(classes) > 1, f"expected several classes, got {classes}"

    def test_angles_are_continuous_not_quantised(self, source):
        """OBB's advantage over template matching: a real angle, not a grid step.

        The template path sweeps a discrete angle grid, which is the root of the
        12-27 degree errors documented in the divergence notes.
        """
        detector, cv2 = source
        image = cv2.cvtColor(cv2.imread(str(_SAMPLE)), cv2.COLOR_BGR2GRAY)
        angles = sorted(o.angle_deg for o in detector.detect(image))
        # Neighbouring detections should not all land on the same few values.
        distinct = {round(a, 1) for a in angles}
        assert len(distinct) > 10, f"angles look quantised: {sorted(distinct)[:8]}"

    def test_full_engine_chain_produces_targets(self, source):
        """adapter -> ObbAlignment -> engine, producing posed targets."""
        import cv2
        from measure.alignment import ObbAlignment
        from measure.multi_target_workflow import MultiTargetWorkflow

        detector, _ = source
        image = cv2.cvtColor(cv2.imread(str(_SAMPLE)), cv2.COLOR_BGR2GRAY)
        ship_class = next(k for k, v in detector.class_names.items() if v == "ship")

        wf = MultiTargetWorkflow()
        strategy = ObbAlignment()
        wf._alignment = strategy
        strategy.teach(image, source=detector, anchor_class_id=ship_class,
                       min_score=0.7, max_targets=3, nominal_size=(80.0, 240.0))

        results = wf.measure(image)
        assert len(results) == 3
        for target in results:
            assert target.score is not None and target.score >= 0.7
            # The pose must be the detector's, in inspection coordinates.
            assert target.center_row > 0 and target.center_col > 0
            assert -90.0 <= target.rotation_deg < 90.0
