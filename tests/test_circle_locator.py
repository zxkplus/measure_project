"""
Tests for the robust circle locator used by the pick calibration.

The locator is the measurement front end, so these tests focus on the
properties that matter for calibration: sub-pixel centre accuracy, a
confidence score that actually rejects non-circles, tolerance to partial
occlusion and image borders, and deterministic serialisation.
"""

from __future__ import annotations

import numpy as np
import cv2
import pytest

from measure import CircleLocator, estimate_noise_sigma, find_circle_center


# ---------------------------------------------------------------------------
# Synthetic image helpers
# ---------------------------------------------------------------------------


def make_disc(
    width: int = 640,
    height: int = 480,
    center=(320.0, 240.0),
    radius: float = 40.0,
    inner: float = 45.0,
    outer: float = 205.0,
    noise: float = 0.0,
    blur: float = 0.8,
    seed: int = 1,
) -> np.ndarray:
    """Render an anti-aliased filled disc with optional noise."""
    yy, xx = np.mgrid[0:height, 0:width]
    distance = np.hypot(xx - center[0], yy - center[1])
    alpha = np.clip(0.5 + (radius - distance), 0.0, 1.0)
    image = outer + (inner - outer) * alpha
    if blur > 0:
        image = cv2.GaussianBlur(image, (0, 0), blur)
    if noise > 0:
        image = image + np.random.default_rng(seed).normal(0, noise, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


def make_square(
    width: int = 640,
    height: int = 480,
    center=(320, 240),
    side: int = 90,
    inner: int = 20,
    outer: int = 205,
) -> np.ndarray:
    """Render a filled square distractor."""
    image = np.full((height, width), outer, dtype=np.uint8)
    half = side // 2
    cv2.rectangle(
        image,
        (center[0] - half, center[1] - half),
        (center[0] + half, center[1] + half),
        inner,
        -1,
    )
    return cv2.GaussianBlur(image, (0, 0), 0.8)


def make_ring(
    width: int = 640,
    height: int = 480,
    center=(300, 250),
    radius: int = 60,
    thickness: int = 12,
) -> np.ndarray:
    """Render a dark ring on a light background."""
    image = np.full((height, width), 200, dtype=np.uint8)
    cv2.circle(image, center, radius, 30, thickness)
    return cv2.GaussianBlur(image, (0, 0), 0.8)


@pytest.fixture
def locator() -> CircleLocator:
    return CircleLocator(radius_range=(12, 120), polarity="dark")


# ---------------------------------------------------------------------------
# Accuracy and confidence
# ---------------------------------------------------------------------------


class TestCircleAccuracy:
    def test_clean_disc_subpixel_accuracy(self, locator):
        truth = (321.4, 238.7)
        detection = locator.find(make_disc(center=truth))
        assert detection.valid, detection.reason
        assert detection.confidence > 0.85
        error = np.hypot(
            detection.center[0] - truth[0], detection.center[1] - truth[1]
        )
        assert error < 0.15, f"centre error {error:.3f}px"
        assert abs(detection.radius - 40.0) < 0.5
        assert detection.support > 0.9
        assert detection.reason == "ok"

    def test_noisy_disc_remains_accurate(self, locator):
        truth = (280.3, 300.6)
        detection = locator.find(make_disc(center=truth, radius=35, noise=4.0))
        assert detection.valid
        error = np.hypot(
            detection.center[0] - truth[0], detection.center[1] - truth[1]
        )
        assert error < 0.35, f"centre error {error:.3f}px"
        assert detection.rms_error < 1.0

    def test_light_disc_on_dark_background(self):
        locator = CircleLocator(radius_range=(20, 120), polarity="auto")
        truth = (250.3, 300.2)
        yy, xx = np.mgrid[0:480, 0:640]
        distance = np.hypot(xx - truth[0], yy - truth[1])
        alpha = np.clip(0.5 + (55.0 - distance), 0.0, 1.0)
        image = 30 + (210 - 30) * alpha
        image = np.clip(cv2.GaussianBlur(image, (0, 0), 0.8), 0, 255).astype(np.uint8)
        detection = locator.find(image)
        assert detection.valid, detection.reason
        error = np.hypot(
            detection.center[0] - truth[0], detection.center[1] - truth[1]
        )
        assert error < 0.3

    @pytest.mark.parametrize("radius", [8, 12, 150])
    def test_radius_extremes(self, radius):
        locator = CircleLocator(radius_range=(6, 200), polarity="dark")
        truth = (321.4, 238.7)
        detection = locator.find(make_disc(center=truth, radius=radius, noise=2.0))
        assert detection.valid, detection.reason
        error = np.hypot(
            detection.center[0] - truth[0], detection.center[1] - truth[1]
        )
        assert error < 0.3, f"r={radius} error={error:.3f}"

    def test_module_level_helper(self):
        detection = find_circle_center(
            make_disc(), radius_range=(20, 80), polarity="dark"
        )
        assert detection.valid
        assert abs(detection.radius - 40.0) < 0.5

    def test_noise_estimate_close_to_truth(self):
        rng = np.random.default_rng(7)
        image = np.clip(128 + rng.normal(0, 5.0, (480, 640)), 0, 255).astype(np.uint8)
        sigma = estimate_noise_sigma(image[20:-20, 20:-20])
        assert 4.4 < sigma < 5.6, sigma


class TestConfidenceGates:
    def test_square_is_rejected(self, locator):
        detection = locator.find(make_square())
        assert not detection.valid
        assert detection.reason != "ok"
        assert detection.confidence < 0.35

    def test_random_texture_has_no_circle(self):
        locator = CircleLocator(radius_range=(10, 100), polarity="dark")
        rng = np.random.default_rng(11)
        image = np.clip(rng.normal(128, 30, (480, 640)), 0, 255).astype(np.uint8)
        detection = locator.find(image)
        assert not detection.valid, detection.reason

    def test_low_contrast_circle_still_valid(self, locator):
        image = make_disc(inner=110, outer=130, noise=1.0, seed=3)
        detection = locator.find(image)
        assert detection.valid, detection.reason
        assert detection.contrast >= locator.min_contrast
        assert detection.confidence > 0.5

    def test_partial_occlusion(self, locator):
        truth = (320.0, 240.0)
        image = make_disc(center=truth, radius=45, noise=2.0)
        cv2.rectangle(image, (320, 150), (440, 330), 205, -1)
        detection = locator.find(image)
        assert detection.valid, detection.reason
        error = np.hypot(
            detection.center[0] - truth[0], detection.center[1] - truth[1]
        )
        assert error < 0.6, f"centre error {error:.3f}px"
        assert detection.support < 0.85

    def test_circle_touching_image_border(self, locator):
        truth = (10.0, 240.0)
        image = make_disc(center=truth, radius=45, noise=1.0, seed=5)
        detection = locator.find(image)
        assert detection.valid, detection.reason
        error = np.hypot(
            detection.center[0] - truth[0], detection.center[1] - truth[1]
        )
        assert error < 0.6

    def test_ring_centre_is_located(self, locator):
        detection = locator.find(make_ring())
        assert detection.valid, detection.reason
        error = np.hypot(detection.center[0] - 300, detection.center[1] - 250)
        assert error < 0.6


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------


class TestSelection:
    def test_radius_range_selects_the_small_circle(self):
        image = make_disc(center=(180, 160), radius=22, noise=1.0)
        second = make_disc(center=(430, 320), radius=60, noise=1.0, seed=2)
        image = np.minimum(image, second)
        locator = CircleLocator(radius_range=(15, 30), polarity="dark")
        detection = locator.find(image)
        assert detection.valid
        assert np.hypot(detection.center[0] - 180, detection.center[1] - 160) < 1.0

    def test_find_all_is_sorted_and_complete(self, locator):
        image = make_disc(center=(180, 160), radius=22, noise=1.0)
        second = make_disc(center=(430, 320), radius=60, noise=1.0, seed=2)
        image = np.minimum(image, second)
        detections = locator.find_all(image)
        assert len(detections) >= 2
        confidences = [item.confidence for item in detections]
        assert confidences == sorted(confidences, reverse=True)
        centres = sorted((round(item.center[0]), round(item.center[1]))
                         for item in detections if item.valid)
        assert (180, 160) in centres or any(
            abs(cx - 180) <= 1 and abs(cy - 160) <= 1 for cx, cy in centres
        )

    def test_roi_restricts_search(self, locator):
        image = make_disc(center=(100, 100), radius=20)
        second = make_disc(center=(500, 380), radius=30, noise=1.0, seed=2)
        image = np.minimum(image, second)
        detection = locator.find(image, roi=(430, 300, 610, 470))
        assert detection.valid
        assert np.hypot(detection.center[0] - 500, detection.center[1] - 380) < 1.0

    def test_expected_prior_wins(self, locator):
        image = make_disc(center=(100, 100), radius=20)
        second = make_disc(center=(500, 380), radius=30, noise=1.0, seed=2)
        image = np.minimum(image, second)
        detection = locator.find(
            image, expected_center=(100, 100), expected_radius=20
        )
        assert detection.valid
        assert np.hypot(detection.center[0] - 100, detection.center[1] - 100) < 1.0

    def test_invalid_polarity_raises(self):
        locator = CircleLocator(polarity="sideways")
        with pytest.raises(ValueError):
            locator.find(make_disc())


# ---------------------------------------------------------------------------
# Serialisation and visualisation
# ---------------------------------------------------------------------------


class TestPersistenceAndVisualisation:
    def test_configuration_roundtrip(self):
        locator = CircleLocator(
            radius_range=(9, 77),
            polarity="light",
            min_confidence=0.42,
            num_rays=120,
            hough_mode="never",
        )
        restored = CircleLocator.from_dict(locator.to_dict())
        assert restored == locator

    def test_detection_dict_is_json_ready(self):
        detection = CircleLocator().find(make_disc())
        data = detection.as_dict(include_points=False)
        assert data["valid"] is True
        assert len(data["center_uv"]) == 2
        assert isinstance(data["bbox"], list)
        assert "edge_points" not in data

    def test_visualize_returns_bgr_canvas(self, test_output_dir):
        detection = CircleLocator().find(make_disc())
        canvas = CircleLocator().visualize(make_disc(), detection, draw_edges=True)
        assert canvas.shape == (480, 640, 3)
        assert canvas.dtype == np.uint8
        path = test_output_dir / "circle_detection.png"
        assert cv2.imwrite(str(path), canvas)
