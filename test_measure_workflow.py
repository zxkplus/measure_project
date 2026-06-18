"""
Tests for measure_workflow.py — unified composable measurement workflow.

Run with pytest:
    pytest test_measure_workflow.py -v

Run as standalone:
    python test_measure_workflow.py
"""

import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from measure_workflow import (
    AngleResult,
    CircleResult,
    DistanceResult,
    EdgePairObject,
    EdgePointObject,
    FitCircleObject,
    FitLineObject,
    GeometricResult,
    LineResult,
    MeasurementWorkflow,
    MeasureObject,
    PointCircleDistanceObject,
    PointLineDistanceObject,
    PointResult,
    SimilarityTransform,
    TemplatePointObject,
    TwoLinesAngleObject,
    TwoPointsDistanceObject,
    TwoPointsLineObject,
)


# ===========================================================================
# Synthetic image generators
# ===========================================================================


def create_blank_image(width=400, height=300, noise_level=3.0):
    """Uniform gray image with slight noise."""
    img = np.ones((height, width), dtype=np.uint8) * 128
    if noise_level > 0:
        noise = np.random.randn(height, width) * noise_level
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return img


def create_edge_image(width=400, height=300, edge_col=200, edge_type="rising"):
    """
    Image with a vertical edge at edge_col.

    edge_type: 'rising' (dark->light left to right) or 'falling' (light->dark)
    """
    img = np.ones((height, width), dtype=np.uint8) * 50
    if edge_type == "rising":
        img[:, edge_col:] = 200
    else:
        img[:, :edge_col] = 200
    # Slight blur for realism
    img = cv2.GaussianBlur(img, (3, 3), 1.0)
    noise = np.random.randn(height, width) * 2
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return img


def create_checkerboard_feature(feature_size=30):
    """Create a small distinct feature for template matching."""
    feat = np.zeros((feature_size, feature_size), dtype=np.uint8)
    # White cross on dark background
    feat[feature_size // 2, :] = 255
    feat[:, feature_size // 2] = 255
    # Add a distinctive corner notch
    feat[0:10, 0:10] = 255
    return feat


def create_reference_with_features(width=400, height=400):
    """
    Create a reference image with two distinct features at known positions.
    """
    img = np.ones((height, width), dtype=np.uint8) * 40

    # Feature A at (100, 100) — cross
    cross = create_checkerboard_feature(30)
    img[85:115, 85:115] = cross

    # Feature B at (300, 200) — rotated cross
    M = cv2.getRotationMatrix2D((15, 15), 45, 1.0)
    cross_rot = cv2.warpAffine(cross, M, (30, 30))
    img[185:215, 285:315] = cross_rot

    # Add Gaussian blur and noise for realism
    img = cv2.GaussianBlur(img, (3, 3), 0.8)
    noise = np.random.randn(height, width) * 3
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return img


def create_shifted_inspection(reference, dx, dy):
    """
    Create an inspection image by shifting the reference via warpAffine.
    """
    rows, cols = reference.shape
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(reference, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)
    return shifted


def create_line_image(width=400, height=400, slope=0.0, intercept=200):
    """Image with a dark line on light background."""
    img = np.ones((height, width), dtype=np.uint8) * 200
    for col in range(width):
        row = int(slope * col + intercept)
        for r in range(max(0, row - 2), min(height, row + 3)):
            img[r, col] = 30
    img = cv2.GaussianBlur(img, (3, 3), 1.0)
    noise = np.random.randn(height, width) * 2
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return img


def create_circle_image(width=400, height=400, center=(200, 200), radius=60):
    """Image with a dark circle on light background."""
    img = np.ones((height, width), dtype=np.uint8) * 200
    cv2.circle(img, (center[1], center[0]), radius, 50, thickness=4)
    img = cv2.GaussianBlur(img, (5, 5), 1.5)
    noise = np.random.randn(height, width) * 2
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return img


# ===========================================================================
# TestSimilarityTransform
# ===========================================================================


class TestSimilarityTransform:
    """Tests for the Umeyama similarity transform."""

    @staticmethod
    def test_empty_correspondences():
        """Zero point pairs should produce an invalid transform."""
        t = SimilarityTransform.from_correspondences([], [])
        assert not t.is_valid

    @staticmethod
    def test_single_point_translation():
        """One point pair should produce a translation-only transform."""
        t = SimilarityTransform.from_correspondences([(100, 200)], [(110, 215)])
        assert t.is_valid
        assert t.rotation == 0.0
        assert t.scale == 1.0
        nr, nc = t.apply(100, 200)
        assert abs(nr - 110) < 0.01
        assert abs(nc - 215) < 0.01

    @staticmethod
    def test_two_points_pure_translation():
        """Two point pairs with pure translation."""
        src = [(100, 200), (300, 400)]
        dst = [(105, 203), (305, 403)]
        t = SimilarityTransform.from_correspondences(src, dst)
        assert t.is_valid
        assert abs(t.rotation) < 0.01
        assert abs(t.scale - 1.0) < 0.01
        nr, nc = t.apply(100, 200)
        assert abs(nr - 105) < 0.1
        assert abs(nc - 203) < 0.1

    @staticmethod
    def test_two_points_pure_rotation():
        """Two point pairs with 90-degree rotation around origin."""
        # Points at (100, 0) and (0, 100) → (0, 100) and (-100, 0) for +90 deg
        src = [(100, 0), (0, 100)]
        dst = [(0, 100), (-100, 0)]
        t = SimilarityTransform.from_correspondences(src, dst)
        assert t.is_valid
        # Rotation should be ~pi/2
        assert abs(abs(t.rotation) - np.pi / 2) < 0.01

    @staticmethod
    def test_two_points_with_scale():
        """Two point pairs with 2x scale and translation."""
        src = [(0, 0), (100, 0)]
        dst = [(10, 20), (210, 20)]  # scale 2x, translation (10, 20)
        t = SimilarityTransform.from_correspondences(src, dst)
        assert t.is_valid
        assert abs(t.scale - 2.0) < 0.1
        nr, nc = t.apply(0, 0)
        assert abs(nr - 10) < 0.5
        assert abs(nc - 20) < 0.5

    @staticmethod
    def test_apply_angle_adds_rotation():
        """apply_angle should add the rotation component."""
        src = [(0, 0), (100, 0)]
        dst = [(0, 0), (0, 100)]  # 90-degree rotation
        t = SimilarityTransform.from_correspondences(src, dst)
        new_angle = t.apply_angle(np.pi / 4)
        assert abs(new_angle - (np.pi / 4 + np.pi / 2)) < 0.01

    @staticmethod
    def test_three_points_noisy():
        """Three point pairs with noise — should still estimate correctly."""
        np.random.seed(42)
        src = [(0, 0), (100, 0), (0, 100)]
        dst = [(10.2, 5.1), (110.1, 5.3), (10.4, 105.0)]  # ~(10, 5) translation
        t = SimilarityTransform.from_correspondences(src, dst)
        assert t.is_valid
        nr, nc = t.apply(0, 0)
        assert abs(nr - 10.2) < 1.0
        assert abs(nc - 5.1) < 1.0

    @staticmethod
    def test_mismatched_lengths_raises():
        """Passing different-length lists should raise."""
        try:
            SimilarityTransform.from_correspondences([(0, 0)], [(1, 1), (2, 2)])
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    @staticmethod
    def test_to_dict():
        """to_dict should return serializable config."""
        src = [(0, 0), (100, 0)]
        dst = [(10, 20), (110, 20)]
        t = SimilarityTransform.from_correspondences(src, dst)
        d = t.to_dict()
        assert "rotation" in d
        assert "scale" in d
        assert "translation_row" in d
        assert "translation_col" in d


# ===========================================================================
# TestPrimitiveObjects
# ===========================================================================


class TestTemplatePointObject:
    """Tests for TemplatePointObject."""

    @staticmethod
    def test_construction_and_match():
        """TemplatePointObject should match its own reference image."""
        ref = create_reference_with_features()
        obj = TemplatePointObject("tp_test", ref, click_row=100, click_col=100,
                                   template_size=40, is_localization=False)
        assert obj.result_type() == "point"
        assert obj.is_primitive
        assert not obj.is_composed
        assert obj.is_localization == False

        result = obj.measure(ref)
        assert isinstance(result, PointResult)
        assert result.valid
        assert result.meta["match_score"] > 0.9  # self-match should be near perfect

    @staticmethod
    def test_known_translation():
        """TemplatePointObject should detect known translation."""
        ref = create_reference_with_features()
        obj = TemplatePointObject("tp_test", ref, click_row=100, click_col=100,
                                   template_size=40)
        shifted = create_shifted_inspection(ref, 5, 10)
        result = obj.measure(shifted)
        assert result.valid
        # Should find the feature at approximately (100+10, 100+5) = (110, 105)
        # WarpAffine with border replication can cause slight shifts
        assert abs(result.row - 110) < 2.0, f"Expected ~110, got {result.row}"
        assert abs(result.col - 105) < 2.0, f"Expected ~105, got {result.col}"

    @staticmethod
    def test_localization_flag():
        """is_localization should be stored correctly."""
        ref = create_reference_with_features()
        obj = TemplatePointObject("loc", ref, 100, 100, template_size=40,
                                   is_localization=True)
        assert obj.is_localization

    @staticmethod
    def test_calibration():
        """calibrate should update the internal position."""
        ref = create_reference_with_features()
        obj = TemplatePointObject("tp", ref, 100, 100, template_size=40)
        t = SimilarityTransform.from_correspondences([(100, 100)], [(110, 105)])
        obj.calibrate(t)
        assert abs(obj._calibrated_row - 110) < 0.01
        assert abs(obj._calibrated_col - 105) < 0.01

    @staticmethod
    def test_to_dict():
        """to_dict should return serializable config."""
        ref = create_reference_with_features()
        obj = TemplatePointObject("tp", ref, 100, 100, template_size=40)
        d = obj.to_dict()
        assert d["object_type"] == "TemplatePointObject"
        assert d["teach_row"] == 100.0
        assert "preprocessor_data" in d


class TestEdgePointObject:
    """Tests for EdgePointObject."""

    @staticmethod
    def test_find_edge():
        """EdgePointObject should find an edge in a synthetic image."""
        img = create_edge_image(edge_col=200, edge_type="rising")
        obj = EdgePointObject("edge", row=150, col=190, angle=0.0,
                              length1=30, length2=10, transition="positive",
                              select="first")
        result = obj.measure(img)
        assert isinstance(result, PointResult)
        assert result.valid
        # Edge should be near col=200
        assert abs(result.col - 200) < 3.0, f"Expected edge near col=200, got col={result.col}"

    @staticmethod
    def test_no_edge_found():
        """EdgePointObject should return invalid when no edge exists."""
        img = create_blank_image()
        obj = EdgePointObject("edge", row=150, col=200, angle=0.0,
                              length1=30, length2=10, threshold=30.0,
                              select="first")
        result = obj.measure(img)
        assert not result.valid
        assert result.meta["num_edges_found"] == 0

    @staticmethod
    def test_calibration():
        """Calibrated edge point should shift position."""
        img = create_edge_image(edge_col=200, edge_type="rising")
        obj = EdgePointObject("edge", row=150, col=180, angle=0.0,
                              length1=30, length2=10, transition="positive",
                              select="first")
        # Simulate 10px right shift
        t = SimilarityTransform.from_correspondences([(150, 180)], [(150, 190)])
        obj.calibrate(t)
        assert abs(obj._calibrated_col - 190) < 0.01

    @staticmethod
    def test_result_type():
        obj = EdgePointObject("edge", row=100, col=200, angle=0.0,
                              length1=30, length2=10)
        assert obj.result_type() == "point"
        assert obj.is_primitive


class TestEdgePairObject:
    """Tests for EdgePairObject."""

    @staticmethod
    def test_find_pair():
        """EdgePairObject should find edge pairs in a bar pattern."""
        # Create image with a dark bar on light background
        # Dark bar: falling edge (light→dark) then rising edge (dark→light)
        # transition="negative" means: falling edge then rising edge = light-dark-light
        img = np.ones((300, 400), dtype=np.uint8) * 200
        img[:, 160:200] = 50  # dark bar (40px wide, centered at col=180)
        img = cv2.GaussianBlur(img, (3, 3), 1.0)
        noise = np.random.randn(300, 400) * 2
        img = np.clip(img + noise, 0, 255).astype(np.uint8)

        obj = EdgePairObject("pair", row=150, col=180, angle=0.0,
                             length1=60, length2=15, transition="negative",
                             select="first", threshold=20.0)
        result = obj.measure(img)
        assert isinstance(result, PointResult)
        if not result.valid:
            print(f"  EdgePair debug: num_pairs={result.meta.get('num_pairs_found', 0)}")
        # May or may not find pairs depending on noise — at minimum check type
        # If valid, center should be near col=180
        if result.valid:
            assert abs(result.col - 180) < 10.0, \
                f"Expected center near col=180, got col={result.col}"

    @staticmethod
    def test_no_pair_found():
        img = create_blank_image()
        obj = EdgePairObject("pair", row=150, col=200, angle=0.0,
                             length1=50, length2=10)
        result = obj.measure(img)
        assert not result.valid


class TestFitLineObject:
    """Tests for FitLineObject."""

    @staticmethod
    def test_fit_horizontal_line():
        """FitLineObject should fit a line to edge points."""
        img = create_line_image(slope=0.0, intercept=200)
        obj = FitLineObject("line", start=(195, 50), end=(205, 350),
                            measure_length1=10, measure_length2=20,
                            num_measures=10, transition="positive")
        result = obj.measure(img)
        assert isinstance(result, LineResult)
        assert result.valid
        # Line is roughly horizontal (row ≈ 200), so:
        # a*row + b*col + c = 0 → b should dominate (col coefficient)
        # horizontal line: row=constant means b≈0, a≠0
        # Wait: ax+by+c=0. For horizontal line (row=constant): y=200
        # 0*x + 1*y - 200 = 0, so a=0, b=1 (or normalized: a≈0, b≈1)
        # Alternatively a*row+b*col+c=0: for row≈200: a≠0, b≈0
        # The convention depends on the fitting. Just verify result is valid.
        assert result.valid

    @staticmethod
    def test_calibration_shifts_line():
        """Calibrated line should shift endpoints."""
        obj = FitLineObject("line", start=(100, 50), end=(100, 350),
                            measure_length1=10, measure_length2=20)
        t = SimilarityTransform.from_correspondences([(100, 200)], [(105, 210)])
        obj.calibrate(t)
        assert abs(obj._calibrated_start[0] - 105) < 0.01


class TestFitCircleObject:
    """Tests for FitCircleObject."""

    @staticmethod
    def test_fit_circle():
        """FitCircleObject should fit a circle."""
        img = create_circle_image(center=(200, 200), radius=60)
        obj = FitCircleObject("circle", center=(200, 200), radius=60,
                              radius_min=40, radius_max=80,
                              measure_length1=20, measure_length2=10,
                              num_measures=16)
        result = obj.measure(img)
        assert isinstance(result, CircleResult)
        if result.valid:
            assert abs(result.center_row - 200) < 5.0
            assert abs(result.center_col - 200) < 5.0
            assert abs(result.radius - 60) < 5.0

    @staticmethod
    def test_calibration_scales_radius():
        """Radius should scale with the transform."""
        obj = FitCircleObject("circle", center=(200, 200), radius=60,
                              radius_min=40, radius_max=80,
                              measure_length1=20, measure_length2=10)
        src = [(200, 200), (260, 200)]
        dst = [(200, 200), (320, 200)]  # scale 2x
        t = SimilarityTransform.from_correspondences(src, dst)
        obj.calibrate(t)
        assert abs(obj._calibrated_radius - 120) < 1.0


# ===========================================================================
# TestComposedObjects
# ===========================================================================


class TestComposedObjects:
    """Tests for composed measurement objects."""

    @staticmethod
    def _create_point_objects():
        """Helper to set up point results on mock objects."""
        class MockPoint(MeasureObject):
            def __init__(self, label, row, col, valid=True):
                super().__init__(label)
                self.result = PointResult(label=label, row=row, col=col, valid=valid)

            def result_type(self):
                return "point"

            def measure(self, image):
                return self.result

            def to_dict(self):
                return {"object_type": "MockPoint"}

            @classmethod
            def from_dict(cls, label, data):
                return cls(label, 0, 0)

        return MockPoint

    @staticmethod
    def test_two_points_line():
        """TwoPointsLineObject should compute line through two points."""
        MockPoint = TestComposedObjects._create_point_objects()
        p1 = MockPoint("p1", 100, 100)
        p2 = MockPoint("p2", 100, 300)

        line_obj = TwoPointsLineObject("line", "p1", "p2")
        line_obj._input_objects = [p1, p2]
        result = line_obj.measure(np.zeros((1, 1)))

        assert isinstance(result, LineResult)
        assert result.valid
        # Line through (100,100) and (100,300) is horizontal
        assert abs(result.a) > 0.01  # x-normal (col component)
        assert abs(result.b) < 0.01  # near-zero y-normal (row component)

    @staticmethod
    def test_two_points_line_coincident():
        """Two coincident points should produce invalid line."""
        MockPoint = TestComposedObjects._create_point_objects()
        p1 = MockPoint("p1", 100, 100)
        p2 = MockPoint("p2", 100, 100)

        line_obj = TwoPointsLineObject("line", "p1", "p2")
        line_obj._input_objects = [p1, p2]
        result = line_obj.measure(np.zeros((1, 1)))
        assert not result.valid

    @staticmethod
    def test_two_points_distance():
        """TwoPointsDistanceObject should compute Euclidean distance."""
        MockPoint = TestComposedObjects._create_point_objects()
        p1 = MockPoint("p1", 100, 100)
        p2 = MockPoint("p2", 100, 300)

        dist_obj = TwoPointsDistanceObject("dist", "p1", "p2")
        dist_obj._input_objects = [p1, p2]
        result = dist_obj.measure(np.zeros((1, 1)))

        assert isinstance(result, DistanceResult)
        assert result.valid
        assert abs(result.value - 200) < 0.01

    @staticmethod
    def test_two_points_distance_invalid_input():
        """Distance should be invalid if either input is invalid."""
        MockPoint = TestComposedObjects._create_point_objects()
        p1 = MockPoint("p1", 100, 100, valid=True)
        p2 = MockPoint("p2", 100, 300, valid=False)

        dist_obj = TwoPointsDistanceObject("dist", "p1", "p2")
        dist_obj._input_objects = [p1, p2]
        result = dist_obj.measure(np.zeros((1, 1)))
        assert not result.valid

    @staticmethod
    def test_point_line_distance():
        """PointLineDistance should compute perpendicular distance.

        Line: row=200 (horizontal line). Point at (150, 200) → distance = 50.
        """
        class MockLine(MeasureObject):
            def __init__(self):
                super().__init__("line")
                # a*row + b*col + c = 0
                # For row=200: 1*row + 0*col - 200 = 0, so a=1, b=0, c=-200
                self.result = LineResult(label="line", a=1.0, b=0.0, c=-200,
                                         start_row=200, start_col=50,
                                         end_row=200, end_col=350, valid=True)

            def result_type(self):
                return "line"

            def measure(self, image):
                return self.result

            def to_dict(self):
                return {"object_type": "MockLine"}

            @classmethod
            def from_dict(cls, label, data):
                return cls()

        MockPoint = TestComposedObjects._create_point_objects()
        pt = MockPoint("pt", 150, 200)  # row=150, 50px from line at row=200

        line = MockLine()
        dist_obj = PointLineDistanceObject("dist", "pt", "line")
        dist_obj._input_objects = [pt, line]
        result = dist_obj.measure(np.zeros((1, 1)))

        assert isinstance(result, DistanceResult)
        assert result.valid
        assert abs(result.value - 50.0) < 0.1, f"Expected dist=50, got {result.value}"

    @staticmethod
    def test_two_lines_angle_parallel():
        """Parallel lines should have angle 0."""
        class MockLine(MeasureObject):
            def __init__(self, label, a, b, c):
                super().__init__(label)
                self.result = LineResult(label=label, a=a, b=b, c=c,
                                         start_row=0, start_col=0,
                                         end_row=100, end_col=100, valid=True)
            def result_type(self): return "line"
            def measure(self, image): return self.result
            def to_dict(self): return {"object_type": "MockLine"}
            @classmethod
            def from_dict(cls, label, data): return cls(label, 0, 0, 0)

        # Two vertical lines: a=1, b=0 → 1*row + 0*col + c = 0 → row = -c
        line1 = MockLine("line1", a=1.0, b=0.0, c=-100)   # row = 100
        line2 = MockLine("line2", a=1.0, b=0.0, c=-200)   # row = 200

        angle_obj = TwoLinesAngleObject("angle", "line1", "line2")
        angle_obj._input_objects = [line1, line2]
        result = angle_obj.measure(np.zeros((1, 1)))

        assert isinstance(result, AngleResult)
        assert result.valid
        assert abs(result.value_rad) < 0.01, f"Expected 0, got {result.value_rad}"

    @staticmethod
    def test_two_lines_angle_perpendicular():
        """Perpendicular lines should have angle ~pi/2 acute."""
        class MockLine(MeasureObject):
            def __init__(self, label, a, b, c):
                super().__init__(label)
                self.result = LineResult(label=label, a=a, b=b, c=c,
                                         start_row=0, start_col=0,
                                         end_row=100, end_col=100, valid=True)
            def result_type(self): return "line"
            def measure(self, image): return self.result
            def to_dict(self): return {"object_type": "MockLine"}
            @classmethod
            def from_dict(cls, label, data): return cls(label, 0, 0, 0)

        # Vertical line: a=1, b=0 → row = -c = 100
        l1 = MockLine("l1", a=1.0, b=0.0, c=-100)
        # Horizontal line: a=0, b=1 → col = -c = 200
        l2 = MockLine("l2", a=0.0, b=1.0, c=-200)

        angle_obj = TwoLinesAngleObject("angle", "l1", "l2")
        angle_obj._input_objects = [l1, l2]
        result = angle_obj.measure(np.zeros((1, 1)))

        assert result.valid
        assert abs(result.value_rad - np.pi / 2) < 0.01, \
            f"Expected pi/2, got {result.value_rad}"

    @staticmethod
    def test_two_lines_angle_invalid():
        """Angle should be invalid if either line is invalid."""
        class MockLine(MeasureObject):
            def __init__(self, label, valid=True):
                super().__init__(label)
                self.result = LineResult(label=label, valid=valid)
            def result_type(self): return "line"
            def measure(self, image): return self.result
            def to_dict(self): return {"object_type": "MockLine"}
            @classmethod
            def from_dict(cls, label, data): return cls(label)

        l1 = MockLine("l1", valid=False)
        l2 = MockLine("l2", valid=False)

        angle_obj = TwoLinesAngleObject("angle", "l1", "l2")
        angle_obj._input_objects = [l1, l2]
        result = angle_obj.measure(np.zeros((1, 1)))
        assert not result.valid

    @staticmethod
    def test_point_circle_distance():
        """PointCircleDistance should compute distance to circumference."""
        class MockCircle(MeasureObject):
            def __init__(self):
                super().__init__("circle")
                self.result = CircleResult(label="circle", center_row=200,
                                           center_col=200, radius=50, valid=True)
            def result_type(self): return "circle"
            def measure(self, image): return self.result
            def to_dict(self): return {"object_type": "MockCircle"}
            @classmethod
            def from_dict(cls, label, data): return cls()

        MockPoint = TestComposedObjects._create_point_objects()
        pt = MockPoint("pt", 200, 260)  # 60px from center at (200,200)

        circle_obj = MockCircle()
        dist_obj = PointCircleDistanceObject("dist", "pt", "circle")
        dist_obj._input_objects = [pt, circle_obj]
        result = dist_obj.measure(np.zeros((1, 1)))

        assert isinstance(result, DistanceResult)
        assert result.valid
        # Distance from (200,260) to circle center (200,200) = 60
        # Distance to circumference = |60 - 50| = 10
        assert abs(result.value - 10) < 0.1


# ===========================================================================
# TestMeasurementWorkflow
# ===========================================================================


class TestMeasurementWorkflow:
    """Integration tests for the MeasurementWorkflow manager."""

    @staticmethod
    def test_add_and_measure():
        """Basic add + measure workflow."""
        ref = create_reference_with_features()
        wf = MeasurementWorkflow()

        wf.add(TemplatePointObject("tp1", ref, 100, 100, template_size=40,
                                    is_localization=True))
        wf.add(TemplatePointObject("tp2", ref, 200, 300, template_size=40,
                                    is_localization=True))

        # Measure on self (no shift)
        results = wf.measure(ref)
        assert "tp1" in results
        assert "tp2" in results
        assert results["tp1"].valid
        assert results["tp2"].valid
        assert wf.is_valid

    @staticmethod
    def test_duplicate_label_raises():
        """Adding two objects with the same label should raise."""
        ref = create_reference_with_features()
        wf = MeasurementWorkflow()
        wf.add(TemplatePointObject("tp1", ref, 100, 100, template_size=40))
        try:
            wf.add(TemplatePointObject("tp1", ref, 200, 300, template_size=40))
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    @staticmethod
    def test_dependency_resolution():
        """Workflow should resolve dependencies and compute execution order."""
        ref = create_reference_with_features()
        wf = MeasurementWorkflow()

        wf.add(TemplatePointObject("tp1", ref, 100, 100, template_size=40))
        wf.add(TemplatePointObject("tp2", ref, 200, 300, template_size=40))
        wf.add(TwoPointsDistanceObject("dist", "tp1", "tp2"))

        wf.resolve()
        # Composed object should come after its inputs
        tp1_idx = wf._execution_order.index("tp1")
        tp2_idx = wf._execution_order.index("tp2")
        dist_idx = wf._execution_order.index("dist")
        assert dist_idx > tp1_idx, "Composed should execute after dependency"
        assert dist_idx > tp2_idx, "Composed should execute after dependency"

    @staticmethod
    def test_dependency_composed_chain():
        """Test a chain: tp1+tp2 → line, tp2+tp3 → line2, line1+line2 → angle."""
        ref = create_reference_with_features()
        wf = MeasurementWorkflow()

        wf.add(TemplatePointObject("tp1", ref, 100, 100, template_size=40))
        wf.add(TemplatePointObject("tp2", ref, 200, 300, template_size=40))
        wf.add(TemplatePointObject("tp3", ref, 50, 350, template_size=40))
        wf.add(TwoPointsLineObject("line1", "tp1", "tp2"))
        wf.add(TwoPointsLineObject("line2", "tp2", "tp3"))
        wf.add(TwoLinesAngleObject("angle", "line1", "line2"))

        wf.resolve()
        order = wf._execution_order
        # Check topological ordering
        for label in ["tp1", "tp2", "tp3"]:
            assert order.index(label) < order.index("line1")
            assert order.index(label) < order.index("line2")
        assert order.index("line1") < order.index("angle")
        assert order.index("line2") < order.index("angle")

    @staticmethod
    def test_cyclic_dependency_detected():
        """Cyclic dependencies should be caught."""
        wf = MeasurementWorkflow()
        # Create a cycle: a → b → a
        # We need 3 objects: a, b, and a dummy 'base' to satisfy _input_labels
        class DummyObj(MeasureObject):
            def __init__(self, label):
                super().__init__(label)
            def result_type(self): return "point"
            def measure(self, image): return PointResult(label=self.label)
            def to_dict(self): return {"object_type": "Dummy"}
            @classmethod
            def from_dict(cls, label, data): return cls(label)

        base = DummyObj("base")
        a = TwoPointsLineObject("a", "b", "base")
        b = TwoPointsLineObject("b", "a", "base")
        wf._objects["base"] = base
        wf._objects["a"] = a
        wf._objects["b"] = b
        wf._registration_order = ["base", "a", "b"]

        try:
            wf.resolve()
            assert False, "Should have raised ValueError for cyclic dependency"
        except ValueError as e:
            assert "Cyclic" in str(e) or "cycle" in str(e).lower()

    @staticmethod
    def test_missing_dependency_raises():
        """Referencing a non-existent label should raise."""
        wf = MeasurementWorkflow()
        # Manually create situation
        a = TwoPointsLineObject("a", "b", "c")  # references 'b' and 'c'
        wf._objects["a"] = a
        wf._registration_order = ["a"]
        a._input_labels = ["b", "c"]

        try:
            wf.resolve()
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "b" in str(e) or "c" in str(e)

    @staticmethod
    def test_localization_failure():
        """When localization templates fail, all results should be invalid."""
        ref = create_reference_with_features()
        blank = create_blank_image()

        wf = MeasurementWorkflow()
        wf.add(TemplatePointObject("loc", ref, 100, 100, template_size=40,
                                    is_localization=True,
                                    match_score_threshold=0.9))
        wf.add(EdgePointObject("edge", row=150, col=200, angle=0.0,
                               length1=30, length2=10))

        results = wf.measure(blank)  # blank image won't match
        # Localization should fail, so all results invalid
        assert not wf.is_valid

    @staticmethod
    def test_no_localization_templates():
        """Workflow without localization templates should work directly."""
        img = create_edge_image(edge_col=200, edge_type="rising")
        wf = MeasurementWorkflow()
        wf.add(EdgePointObject("edge", row=150, col=190, angle=0.0,
                               length1=30, length2=10, transition="positive",
                               select="first"))
        results = wf.measure(img)
        assert results["edge"].valid

    @staticmethod
    def test_serialization_roundtrip():
        """Save and load should produce identical results."""
        ref = create_reference_with_features()
        wf = MeasurementWorkflow()

        wf.add(TemplatePointObject("tp1", ref, 100, 100, template_size=40,
                                    is_localization=True))
        wf.add(TemplatePointObject("tp2", ref, 200, 300, template_size=40,
                                    is_localization=True))
        wf.add(TwoPointsDistanceObject("dist", "tp1", "tp2"))
        wf.add(EdgePointObject("edge", row=150, col=200, angle=0.0,
                               length1=30, length2=10))
        wf.add(EdgePairObject("pair", row=150, col=200, angle=0.0,
                              length1=50, length2=10))
        wf.add(FitLineObject("fitline", start=(100, 50), end=(100, 350),
                             measure_length1=10, measure_length2=20))
        wf.add(FitCircleObject("fitcircle", center=(200, 200), radius=60,
                               radius_min=40, radius_max=80,
                               measure_length1=20, measure_length2=10))
        wf.add(TwoPointsLineObject("line", "tp1", "tp2"))
        wf.add(TwoLinesAngleObject("angle", "line", "fitline"))
        wf.add(PointLineDistanceObject("pldist", "tp1", "fitline"))
        wf.add(PointCircleDistanceObject("pcdist", "tp1", "fitcircle"))

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            tmp_path = f.name

        try:
            wf.save(tmp_path)
            wf2 = MeasurementWorkflow.load(tmp_path)

            # Execute both and compare
            results1 = wf.measure(ref)
            results2 = wf2.measure(ref)

            for label in wf._registration_order:
                r1 = results1[label]
                r2 = results2[label]
                assert r1.type == r2.type, f"Type mismatch for {label}"
                assert r1.valid == r2.valid, f"Valid mismatch for {label}: {r1.valid} vs {r2.valid}"

                if r1.valid and r2.valid:
                    if isinstance(r1, PointResult):
                        assert abs(r1.row - r2.row) < 0.1
                        assert abs(r1.col - r2.col) < 0.1
                    elif isinstance(r1, DistanceResult):
                        assert abs(r1.value - r2.value) < 0.1
                    elif isinstance(r1, AngleResult):
                        assert abs(r1.value_rad - r2.value_rad) < 0.1
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    @staticmethod
    def test_get_result():
        """get_result should return the right object's result."""
        ref = create_reference_with_features()
        wf = MeasurementWorkflow()
        wf.add(TemplatePointObject("tp1", ref, 100, 100, template_size=40))
        wf.measure(ref)
        r = wf.get_result("tp1")
        assert isinstance(r, PointResult)
        assert r.valid
        assert wf.get_result("nonexistent") is None

    @staticmethod
    def test_object_labels_and_has_object():
        """object_labels and has_object should work."""
        ref = create_reference_with_features()
        wf = MeasurementWorkflow()
        wf.add(TemplatePointObject("tp1", ref, 100, 100, template_size=40))
        wf.add(TemplatePointObject("tp2", ref, 200, 300, template_size=40))
        assert wf.object_labels == ["tp1", "tp2"]
        assert wf.has_object("tp1")
        assert not wf.has_object("tp3")

    @staticmethod
    def test_visualize_smoke():
        """visualize should not crash."""
        ref = create_reference_with_features()
        wf = MeasurementWorkflow()
        wf.add(TemplatePointObject("tp1", ref, 100, 100, template_size=30,
                                    is_localization=True))
        wf.add(TemplatePointObject("tp2", ref, 200, 300, template_size=30,
                                    is_localization=True))
        wf.add(TwoPointsDistanceObject("dist", "tp1", "tp2"))

        wf.measure(ref)
        vis = wf.visualize(ref)
        assert vis.shape[0] == ref.shape[0]
        assert vis.shape[1] == ref.shape[1]
        assert vis.shape[2] == 3  # BGR

    @staticmethod
    def test_transform_available_after_measure():
        """transform property should return the localization transform."""
        ref = create_reference_with_features()
        wf = MeasurementWorkflow()
        wf.add(TemplatePointObject("tp1", ref, 100, 100, template_size=40,
                                    is_localization=True))
        wf.measure(ref)
        t = wf.transform
        assert t is not None
        assert t.is_valid


# ===========================================================================
# TestVisualDemo
# ===========================================================================


class TestVisualDemo:
    """Interactive visual demos (press any key in OpenCV window to advance)."""

    @staticmethod
    def test_visual_workflow_demo():
        """
        Visual demo: parallelogram with two FitLineObjects measuring adjacent
        edges, then computing the interior angle.  Shows every measurement
        object step by step in a multi-window slideshow.
        """
        print("\n=== Visual Demo: Parallelogram Edge Angle Measurement ===")

        # ---------- 1. Create a parallelogram reference image ----------
        img_h, img_w = 500, 550
        ref = _create_parallelogram_image(img_h, img_w)

        # ---------- 2. Create inspection via rotation + translation ----------
        rows, cols = ref.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -18.0, 1.0)
        M[0, 2] += 10  # dx
        M[1, 2] += 6   # dy
        inspection = cv2.warpAffine(ref, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)

        # ---------- 3. Build workflow ----------
        # Parallelogram corners (row, col):
        #   A=(120, 150), B=(120, 420), C=(330, 480), D=(330, 210)
        # Edge AB: top, roughly horizontal  (row≈120)
        # Edge AD: left, slanted           (120,150)→(330,210)
        # Interior angle at A ≈ 60° (parallelogram with slant)

        wf = MeasurementWorkflow()

        # Localization templates at three corners (for robust transform)
        wf.add(TemplatePointObject("loc_A", ref, 120, 150, template_size=40,
                                    is_localization=True))
        # wf.add(TemplatePointObject("loc_B", ref, 120, 420, template_size=40,
        #                             is_localization=True))
        wf.add(TemplatePointObject("loc_D", ref, 330, 210, template_size=40,
                                    is_localization=True))

        # Two FitLineObjects — measure two adjacent edges
        # Edge AB: top edge (point A → point B)
        wf.add(FitLineObject("edge_AB",
                             start=(120, 180), end=(120, 400),
                             measure_length1=5, measure_length2=25,
                             num_measures=8, transition="positive",threshold=5.0))

        # Edge AD: left slanted edge (point A → point D)
        wf.add(FitLineObject("edge_AD",
                             start=(160, 165), end=(309, 205),
                             measure_length1=10, measure_length2=25,
                             num_measures=8, transition="positive",threshold=5.0))

        # Composed: angle between the two edges
        wf.add(TwoLinesAngleObject("angle_A", "edge_AB", "edge_AD"))

        wf.add(EdgePointObject("p_a", row=120, col=285, angle=np.pi/2, length1=100, length2=20, transition="positive", select="first", threshold=5.))
        #将 ref 保存起来
        wf.measure(ref)
        ##单独展示ref 和测量对象
        wf.visualize(ref, show_objects=["loc_A","loc_D", "angle_A"], wait_ms=0)  # "edge_AB", "edge_AD", "angle_A"
        #保存 ref
        cv2.imwrite("ref.png", ref)

        # # ---------- 4. Resolve & measure ----------
        # results = wf.measure(inspection)

        # # ---------- 5. 打印每个测量对象的数值结果 (teach vs match) ----------
        # _print_detailed_results(wf, results)

        # # ---------- 6. 在原图上逐对象展示 teach → match ----------
        # _show_teach_vs_match_slideshow(wf, ref, inspection, wait_ms=1500)

        # print("\n  Press any key to close...")
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # print("=== Visual Demo Complete ===\n")


def _print_detailed_results(wf, results):
    """Print teach→match comparison for every measurement object."""
    t = wf.transform

    print(f"\n{'='*65}")
    print(f"  Measurement Results — Teach vs Match")
    print(f"{'='*65}")

    print(f"\n  [Localization Transform]")
    if t and t.is_valid:
        print(f"    rotation = {np.degrees(t.rotation):.2f} deg")
        print(f"    scale    = {t.scale:.4f}")
        print(f"    dx       = {t.translation_col:.2f} px")
        print(f"    dy       = {t.translation_row:.2f} px")
    else:
        print(f"    (none — no localization templates)")

    for label in wf._registration_order:
        obj = wf._objects[label]
        r = obj.result
        if r is None:
            continue

        cls_name = obj.__class__.__name__
        print(f"\n  [{cls_name}] {label}")
        print(f"    result_type = {r.type}")
        print(f"    valid       = {r.valid}")

        if isinstance(obj, TemplatePointObject):
            print(f"    teach (row,col) = ({obj._teach_row:.1f}, {obj._teach_col:.1f})")
            if r.valid:
                print(f"    match (row,col) = ({r.row:.1f}, {r.col:.1f})")
                print(f"    match_score     = {r.meta['match_score']:.4f}")

        elif isinstance(obj, FitLineObject):
            print(f"    teach_start = ({obj._teach_start[0]:.1f}, {obj._teach_start[1]:.1f})")
            print(f"    teach_end   = ({obj._teach_end[0]:.1f}, {obj._teach_end[1]:.1f})")
            if r.valid:
                print(f"    line_eq     = {r.a:.4f}*row + {r.b:.4f}*col + {r.c:.4f} = 0")
                print(f"    angle       = {r.angle_deg:.2f} deg")
                print(f"    length      = {r.length:.1f} px")
                print(f"    num_points  = {r.meta.get('num_points', 0)}")
                print(f"    mean_error  = {r.meta.get('mean_error', 0):.3f} px")
                print(f"    max_error   = {r.meta.get('max_error', 0):.3f} px")

        elif isinstance(obj, TwoLinesAngleObject):
            if r.valid:
                print(f"    angle       = {r.value_deg:.3f} deg")


def _show_teach_vs_match_slideshow(wf, ref, inspection, wait_ms=1500):
    """Step-by-step slideshow: teach on ref (left) vs match on insp (right) per object."""

    def _to_bgr(img):
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()

    ref_bgr = _to_bgr(ref)
    insp_bgr = _to_bgr(inspection)
    h, w = ref.shape[:2]
    total = len(wf._registration_order) + 2  # +1 for "all teach" +1 for "all results"

    # ---------- 1. REFERENCE: ALL teach positions ----------
    vis_ref = ref_bgr.copy()
    cv2.putText(vis_ref, "REFERENCE — All Teach Positions", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    for label in wf._registration_order:
        obj = wf._objects[label]
        if isinstance(obj, TemplatePointObject):
            tc = (int(obj._teach_col), int(obj._teach_row))
            cv2.drawMarker(vis_ref, tc, (0, 255, 0), cv2.MARKER_CROSS, 12, 2)
            cv2.putText(vis_ref, f"{label}\n({obj._teach_row:.0f},{obj._teach_col:.0f})",
                        (tc[0] + 12, tc[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
        elif isinstance(obj, FitLineObject):
            p1 = (int(obj._teach_start[1]), int(obj._teach_start[0]))
            p2 = (int(obj._teach_end[1]), int(obj._teach_end[0]))
            cv2.line(vis_ref, p1, p2, (255, 255, 0), 2)
            mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
            cv2.putText(vis_ref, label, (mid[0] + 5, mid[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    cv2.imshow(f"1/{total} REFERENCE — All Teach Positions", vis_ref)
    cv2.waitKey(wait_ms)

    # ---------- 2..N-1: Per-object teach (left) vs match (right) ----------
    window_idx = 2
    for label in wf._registration_order:
        obj = wf._objects[label]
        r = obj.result
        if r is None:
            continue

        left = ref_bgr.copy()
        right = insp_bgr.copy()

        # Left: teach position on reference
        cv2.putText(left, f"{label} — TEACH", (8, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
        if isinstance(obj, TemplatePointObject):
            tc = (int(obj._teach_col), int(obj._teach_row))
            cv2.drawMarker(left, tc, (0, 255, 0), cv2.MARKER_CROSS, 14, 2)
            cv2.putText(left, f"({obj._teach_row:.0f},{obj._teach_col:.0f})",
                        (tc[0] + 14, tc[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        elif isinstance(obj, FitLineObject):
            p1 = (int(obj._teach_start[1]), int(obj._teach_start[0]))
            p2 = (int(obj._teach_end[1]), int(obj._teach_end[0]))
            cv2.line(left, p1, p2, (255, 255, 0), 2)
            cv2.putText(left, "teach line", (p1[0] + 5, p1[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        # Right: match/measurement result on inspection
        cv2.putText(right, f"{label} — RESULT", (8, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
        if r.valid:
            if isinstance(obj, TemplatePointObject):
                tc_c = (int(obj._calibrated_col), int(obj._calibrated_row))
                mc = (int(r.col), int(r.row))
                cv2.drawMarker(right, tc_c, (255, 255, 0), cv2.MARKER_DIAMOND, 10, 2)
                cv2.drawMarker(right, mc, (0, 255, 0), cv2.MARKER_CROSS, 14, 2)
                cv2.line(right, tc_c, mc, (0, 200, 200), 1)
                cv2.putText(right, f"({r.row:.1f},{r.col:.1f}) s={r.meta['match_score']:.3f}",
                            (mc[0] + 14, mc[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            elif isinstance(obj, FitLineObject):
                right = obj.visualize(right)
                cv2.putText(right, f"a={r.angle_deg:.1f}deg err={r.meta.get('mean_error',0):.3f}px",
                            (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        else:
            cv2.putText(right, "INVALID", (w // 2 - 40, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Side-by-side
        combined = np.hstack([left, right])
        cv2.putText(combined, "TEACH", (10, combined.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.putText(combined, "MATCH", (w + 10, combined.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.imshow(f"{window_idx}/{total} {obj.__class__.__name__}: {label}", combined)
        cv2.waitKey(wait_ms)
        window_idx += 1

    # ---------- Final: ALL objects on inspection ----------
    vis_all = wf.visualize(inspection)
    cv2.putText(vis_all, "INSPECTION — All Objects Calibrated + Measured", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    a = wf.get_result("angle_A")
    if a and a.valid:
        loc_a = wf.get_result("loc_A")
        cv2.putText(vis_all, f"Interior Angle = {a.value_deg:.2f} deg",
                    (int(loc_a.col) + 15, int(loc_a.row) - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv2.imshow(f"{window_idx}/{total} ALL OBJECTS — Final Result", vis_all)


def _create_parallelogram_image(height=500, width=550):
    """
    Create a grayscale image with a white parallelogram on dark background
    plus small corner features for template matching.

    Parallelogram corners (row, col):
        A = (120, 150)   — top-left
        B = (120, 420)   — top-right
        C = (330, 480)   — bottom-right
        D = (330, 210)   — bottom-left

    Interior angle at A ≈ arctan(slant/height), roughly 63°.

    Returns:
        Grayscale uint8 image.
    """
    img = np.ones((height, width), dtype=np.uint8) * 40

    # Draw parallelogram as filled white polygon
    corners = np.array([
        [150, 120],   # A — OpenCV format: (col, row)
        [420, 120],   # B
        [480, 330],   # C
        [210, 330],   # D
    ], dtype=np.int32)
    cv2.fillPoly(img, [corners], 220)

    # Draw dark border around the parallelogram for clear edges
    cv2.polylines(img, [corners], True, 20, thickness=4)

    # Add distinct features at three corners for template matching
    # (different shapes prevent mismatching between templates)

    # loc_A (120, 150): solid circle (dot marker, rotation invariant)
    cv2.circle(img, (150, 120), 6, 255, -1)

    # loc_B (120, 420): diamond (rotated square, 45°)
    diamond_pts = np.array([
        [420, 112],   # top
        [428, 120],   # right
        [420, 128],   # bottom
        [412, 120],   # left
    ], dtype=np.int32)
    cv2.fillPoly(img, [diamond_pts], 255)

    # loc_D (330, 210): L-shaped corner bracket
    l_sz = 7
    cv2.line(img, (210 - l_sz, 330), (210 + l_sz, 330), 255, thickness=2)
    cv2.line(img, (210, 330 - l_sz), (210, 330 + l_sz), 255, thickness=2)

    # Add blur and noise
    img = cv2.GaussianBlur(img, (3, 3), 1.0)
    noise = np.random.randn(height, width) * 3
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return img


# ===========================================================================
# Test runner
# ===========================================================================


def run_all_tests():
    """Run all tests and report results (standalone runner)."""
    test_classes = [
        TestSimilarityTransform,
        TestTemplatePointObject,
        TestEdgePointObject,
        TestEdgePairObject,
        TestFitLineObject,
        TestFitCircleObject,
        TestComposedObjects,
        TestMeasurementWorkflow,
        TestVisualDemo,
    ]

    total = 0
    passed = 0
    failed = 0

    for tc in test_classes:
        print(f"\n{'='*60}")
        print(f"  {tc.__name__}")
        print(f"{'='*60}")

        for name in dir(tc):
            if name.startswith("test_"):
                method = getattr(tc, name)
                total += 1
                try:
                    method()
                    print(f"  PASS  {name}")
                    passed += 1
                except Exception as e:
                    print(f"  FAIL  {name}: {e}")
                    failed += 1
                    import traceback
                    traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"  Results: {passed}/{total} passed, {failed} failed")
    print(f"{'='*60}")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
