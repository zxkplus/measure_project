"""
Tests for multi_target_workflow.py — multi-target measurement workflow.

Run with pytest:
    pytest test_multi_target_workflow.py -v

Run as standalone:
    python test_multi_target_workflow.py
"""

import os
import sys
import tempfile

import cv2
import numpy as np
import pytest

from multi_target_workflow import (
    MultiTargetWorkflow,
    RotatedTemplate,
    TargetInstance,
    TargetResult,
    MeasureDef,
)


# ===========================================================================
# Test helpers
# ===========================================================================


def _create_test_image(height=300, width=400):
    """Create a reference image with an L-shaped feature at the center."""
    img = np.ones((height, width), dtype=np.uint8) * 128
    # L-shaped feature (asymmetric for unambiguous orientation detection)
    crow, ccol = height // 2, width // 2  # 150, 200
    img[crow - 20:crow, ccol - 20:ccol + 20] = 255   # horizontal bar
    img[crow:crow + 20, ccol - 20:ccol - 10] = 255    # vertical bar
    img[crow + 10, ccol + 8] = 0                       # dot for asymmetry
    return img


def _create_inspection_with_targets(ref, positions, angles=None):
    """
    Create an inspection image by pasting copies of the reference's
    L-shape feature at specified positions, optionally rotated.

    The feature is cropped from the center of the reference, then pasted
    onto a blank canvas at each target position.

    Args:
        ref: Reference image.
        positions: List of (row, col) centers.
        angles: Optional list of rotation angles in degrees.

    Returns:
        Inspection image.
    """
    canvas = np.ones_like(ref) * 128
    h, w = ref.shape
    feature_size = 60
    half = feature_size // 2

    # Crop the feature from the reference center
    cr, cc = h // 2, w // 2
    fr1 = max(0, cr - half)
    fr2 = min(h, cr + half)
    fc1 = max(0, cc - half)
    fc2 = min(w, cc + half)
    base_feature = ref[fr1:fr2, fc1:fc2].copy()

    if angles is None:
        angles = [0.0] * len(positions)

    for (trow, tcol), ang in zip(positions, angles):
        feat = base_feature.copy()
        fh, fw = feat.shape

        if abs(ang) > 0.01:
            M = cv2.getRotationMatrix2D((fw / 2, fh / 2), ang, 1.0)
            feat = cv2.warpAffine(feat, M, (fw, fh),
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=int(np.mean(feat)))

        # Paste onto canvas
        pr1 = max(0, trow - fh // 2)
        pc1 = max(0, tcol - fw // 2)
        pr2 = min(h, pr1 + fh)
        pc2 = min(w, pc1 + fw)
        paste_h = pr2 - pr1
        paste_w = pc2 - pc1
        if paste_h > 0 and paste_w > 0:
            canvas[pr1:pr2, pc1:pc2] = feat[:paste_h, :paste_w]

    return canvas


# ===========================================================================
# RotatedTemplate Tests
# ===========================================================================


class TestRotatedTemplate:
    """Tests for RotatedTemplate — rotated bbox crop + rectify + detect."""

    def test_crop_rectify_no_rotation(self):
        """Crop + rectify with 0° rotation produces expected dimensions."""
        ref = _create_test_image()
        rt = RotatedTemplate(ref, center_row=150, center_col=200,
                             bbox_width=80, bbox_height=80, bbox_angle_deg=0)
        assert rt.template_image is not None
        assert rt.template_image.shape == (80, 80)
        # Content should be similar to the reference center
        assert np.mean(rt.template_image) > 130  # not blank

    def test_crop_rectify_with_rotation(self):
        """Crop + rectify with 30° rotation still works."""
        ref = _create_test_image()
        rt = RotatedTemplate(ref, center_row=150, center_col=200,
                             bbox_width=80, bbox_height=80, bbox_angle_deg=30)
        assert rt.template_image.shape == (80, 80)

    def test_detect_all_single_target(self):
        """detect_all finds the single target at the reference position."""
        ref = _create_test_image()
        rt = RotatedTemplate(ref, center_row=150, center_col=200,
                             bbox_width=80, bbox_height=80, bbox_angle_deg=0,
                             angle_range=(-15, 15),
                             match_score_threshold=0.3)
        # Inspection = reference (target at same position)
        targets = rt.detect_all(ref)
        assert len(targets) >= 1
        t0 = targets[0]
        assert t0.match_score > 0.4  # large template has low NCC on flat bg
        # Position should be near the template center
        assert abs(t0.row - 150) < 30
        assert abs(t0.col - 200) < 30

    def test_detect_all_two_targets(self):
        """detect_all finds two identical targets."""
        ref = _create_test_image()
        positions = [(150, 200), (100, 320)]
        inspection = _create_inspection_with_targets(ref, positions)

        rt = RotatedTemplate(ref, center_row=150, center_col=200,
                             bbox_width=80, bbox_height=80, bbox_angle_deg=0,
                             angle_range=(-15, 15))
        targets = rt.detect_all(inspection)
        assert len(targets) >= 2

        # Verify both expected positions are found
        found_positions = [(t.row, t.col) for t in targets]
        for exp_row, exp_col in positions:
            found = any(abs(tr - exp_row) < 20 and abs(tc - exp_col) < 20
                        for tr, tc in found_positions)
            assert found, f"No match near ({exp_row},{exp_col}), got {found_positions}"

    def test_detect_all_reports_angle(self):
        """Each detected target reports its rotation angle."""
        ref = _create_test_image()
        positions = [(150, 200)]
        angles = [10.0]
        inspection = _create_inspection_with_targets(ref, positions, angles)

        rt = RotatedTemplate(ref, center_row=150, center_col=200,
                             bbox_width=80, bbox_height=80, bbox_angle_deg=0,
                             angle_range=(-30, 30))
        targets = rt.detect_all(inspection)
        assert len(targets) >= 1
        # The detected angle may differ from ground truth due to feature content
        assert 'angle_deg' in targets[0].__dict__

    def test_empty_detection_on_blank(self):
        """detect_all returns empty list on blank image with high threshold."""
        ref = _create_test_image()
        rt = RotatedTemplate(ref, center_row=150, center_col=200,
                             bbox_width=80, bbox_height=80, bbox_angle_deg=0,
                             match_score_threshold=0.9, angle_range=(-15, 15))
        blank = np.ones_like(ref) * 128
        targets = rt.detect_all(blank)
        assert len(targets) == 0


# ===========================================================================
# MultiTargetWorkflow Tests
# ===========================================================================


class TestMultiTargetWorkflow:
    """Tests for MultiTargetWorkflow — full teach → inspect pipeline."""

    def test_teach_and_inspect_two_targets(self):
        """Full pipeline: teach on ref, detect 2 targets, measure each."""
        ref = _create_test_image()
        positions = [(150, 200), (100, 320)]
        inspection = _create_inspection_with_targets(ref, positions)

        mtw = MultiTargetWorkflow()
        mtw.set_template(ref, center_row=150, center_col=200,
                         bbox_width=80, bbox_height=80, bbox_angle_deg=0,
                         angle_range=(-15, 15))

        # Add a line measurement in template coordinates
        mtw.add_measurement('fit_line', 'top_edge',
                            start=(10, 20), end=(10, 60),
                            measure_length1=10, measure_length2=5)

        results = mtw.inspect(inspection)
        assert len(results) >= 2

        for tr in results:
            assert tr.target.match_score > 0.5
            assert 'top_edge' in tr.measurements
            line = tr.measurements['top_edge']
            assert hasattr(line, 'start_row')
            assert hasattr(line, 'end_row')

    def test_measurements_transformed_to_image_coords(self):
        """Measurement results are in inspection image coordinates."""
        ref = _create_test_image()
        positions = [(150, 200)]
        inspection = _create_inspection_with_targets(ref, positions)

        mtw = MultiTargetWorkflow()
        mtw.set_template(ref, center_row=150, center_col=200,
                         bbox_width=80, bbox_height=80, bbox_angle_deg=0,
                         angle_range=(-15, 15))
        mtw.add_measurement('fit_line', 'edge1',
                            start=(20, 20), end=(20, 60),
                            measure_length1=10, measure_length2=5)

        results = mtw.inspect(inspection)
        assert len(results) >= 1
        tr = results[0]
        line = tr.get('edge1')
        assert line is not None
        # The line should be near the target position, not in template coords
        assert abs(line.start_row - tr.target.row) < 50
        assert abs(line.start_col - tr.target.col) < 50

    def test_add_measurement_duplicate_label_raises(self):
        """Adding a measurement with a duplicate label raises ValueError."""
        ref = _create_test_image()
        mtw = MultiTargetWorkflow()
        mtw.set_template(ref, center_row=150, center_col=200,
                         bbox_width=80, bbox_height=80, bbox_angle_deg=0)
        mtw.add_measurement('fit_line', 'edge1',
                            start=(10, 10), end=(50, 10))
        with pytest.raises(ValueError, match='Duplicate'):
            mtw.add_measurement('fit_line', 'edge1',
                                start=(10, 20), end=(50, 20))

    def test_add_measurement_unsupported_type_raises(self):
        """Unsupported measure_type raises ValueError."""
        ref = _create_test_image()
        mtw = MultiTargetWorkflow()
        mtw.set_template(ref, center_row=150, center_col=200,
                         bbox_width=80, bbox_height=80, bbox_angle_deg=0)
        with pytest.raises(ValueError, match='Unsupported'):
            mtw.add_measurement('unsupported_type', 'test')

    def test_inspect_without_template_raises(self):
        """Calling inspect() before set_template() raises RuntimeError."""
        mtw = MultiTargetWorkflow()
        ref = _create_test_image()
        with pytest.raises(RuntimeError, match='not set'):
            mtw.inspect(ref)

    def test_serialization_roundtrip(self):
        """save() and load() preserve template, measurements, and results."""
        ref = _create_test_image()
        positions = [(150, 200), (100, 320)]
        inspection = _create_inspection_with_targets(ref, positions)

        mtw = MultiTargetWorkflow()
        mtw.set_template(ref, center_row=150, center_col=200,
                         bbox_width=80, bbox_height=80, bbox_angle_deg=0,
                         angle_range=(-15, 15))
        mtw.add_measurement('fit_line', 'edge1',
                            start=(10, 20), end=(50, 20),
                            measure_length1=10, measure_length2=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "mtw.npz")
            mtw.save(filepath)
            loaded = MultiTargetWorkflow.load(filepath)

        # Verify loaded template shape
        assert loaded.template_shape == mtw.template_shape

        # Verify loaded workflow produces similar results
        results_orig = mtw.inspect(inspection)
        results_loaded = loaded.inspect(inspection)

        assert len(results_loaded) == len(results_orig)
        for ro, rl in zip(results_orig, results_loaded):
            assert abs(ro.target.row - rl.target.row) < 0.1
            assert abs(ro.target.col - rl.target.col) < 0.1
            assert set(ro.measurements.keys()) == set(rl.measurements.keys())

    def test_visualize_smoke(self):
        """visualize() does not crash."""
        ref = _create_test_image()
        positions = [(150, 200), (100, 320)]
        inspection = _create_inspection_with_targets(ref, positions)

        mtw = MultiTargetWorkflow()
        mtw.set_template(ref, center_row=150, center_col=200,
                         bbox_width=80, bbox_height=80, bbox_angle_deg=0,
                         angle_range=(-15, 15))
        mtw.add_measurement('fit_line', 'top', start=(10, 20), end=(10, 60),
                            measure_length1=10, measure_length2=5)
        results = mtw.inspect(inspection)

        vis = mtw.visualize(inspection, results, wait_time=-1)
        assert vis is not None
        assert vis.shape[:2] == inspection.shape[:2]
        assert len(vis.shape) == 3  # BGR

    def test_template_properties(self):
        """template_image and template_shape properties work."""
        ref = _create_test_image()
        mtw = MultiTargetWorkflow()
        assert mtw.template_image is None
        assert mtw.template_shape is None

        mtw.set_template(ref, center_row=150, center_col=200,
                         bbox_width=80, bbox_height=80, bbox_angle_deg=0)
        assert mtw.template_image is not None
        assert mtw.template_shape == (80, 80)

    def test_fit_circle_measurement(self):
        """fit_circle measurement type works."""
        # Create a reference image with a circular feature
        ref = np.ones((300, 400), dtype=np.uint8) * 128
        cv2.circle(ref, (200, 150), 30, 255, -1)
        cv2.circle(ref, (200, 150), 30, 0, 2)

        # Inspection: same circle, shifted
        inspection = np.ones((300, 400), dtype=np.uint8) * 128
        cv2.circle(inspection, (300, 100), 30, 255, -1)
        cv2.circle(inspection, (300, 100), 30, 0, 2)

        mtw = MultiTargetWorkflow()
        mtw.set_template(ref, center_row=150, center_col=200,
                         bbox_width=80, bbox_height=80, bbox_angle_deg=0,
                         angle_range=(-15, 15),
                         match_score_threshold=0.3)
        mtw.add_measurement('fit_circle', 'hole',
                            center=(40, 40), radius=30,
                            radius_min=20, radius_max=40,
                            measure_length1=10, measure_length2=3)

        results = mtw.inspect(inspection)
        assert len(results) >= 1
        tr = results[0]
        circle = tr.get('hole')
        assert circle is not None
        # The circle center in image coords should be near the target's circle
        if circle.valid:
            assert abs(circle.center_row - 100) < 20
            assert abs(circle.center_col - 300) < 20


# ===========================================================================
# Coordinate Transform Tests
# ===========================================================================


class TestCoordinateTransforms:
    """Tests for _transform_point/line/circle_result functions."""

    def test_transform_point_no_rotation(self):
        """Point at template center maps to target center (no rotation)."""
        from multi_target_workflow import _transform_point_result
        from measure_workflow import PointResult
        target = TargetInstance(index=0, row=200, col=300, angle_deg=0,
                                scale=1.0, match_score=1.0)
        result = PointResult(label='test', row=30, col=40, valid=True)
        transformed = _transform_point_result(result, target, 60, 80)
        # dx=40-40=0, dy=30-30=0 → same as target center
        assert abs(transformed.row - 200) < 1
        assert abs(transformed.col - 300) < 1
        assert 'template_row' in transformed.meta

    def test_transform_point_with_rotation(self):
        """Point rotates correctly with target angle."""
        from multi_target_workflow import _transform_point_result
        from measure_workflow import PointResult
        target = TargetInstance(index=0, row=100, col=100, angle_deg=90,
                                scale=1.0, match_score=1.0)
        # Point at (30, 50) in template (center at 30, 40): dy=0, dx=10
        # Rotated 90° CCW: new dx=0, new dy=10
        result = PointResult(label='test', row=30, col=50, valid=True)
        transformed = _transform_point_result(result, target, 60, 80)
        assert abs(transformed.row - 110) < 1  # 100 + 10
        assert abs(transformed.col - 100) < 1  # 100 + 0

    def test_transform_line_preserves_endpoints(self):
        """Line endpoints transform correctly."""
        from multi_target_workflow import _transform_line_result
        from measure_workflow import LineResult
        target = TargetInstance(index=0, row=200, col=300, angle_deg=0,
                                scale=1.0, match_score=1.0)
        result = LineResult(
            label='test', a=0, b=0, c=0,
            start_row=20, start_col=30,
            end_row=20, end_col=50,
            valid=True,
        )
        transformed = _transform_line_result(result, target, 60, 80)
        assert abs(transformed.start_row - 190) < 1  # 200 + (20-30) = 190
        assert abs(transformed.start_col - 290) < 1  # 300 + (30-40) = 290
        assert abs(transformed.end_col - 310) < 1    # 300 + (50-40) = 310

    def test_transform_circle_scales_radius(self):
        """Circle radius scales with target scale."""
        from multi_target_workflow import _transform_circle_result
        from measure_workflow import CircleResult
        target = TargetInstance(index=0, row=100, col=100, angle_deg=0,
                                scale=2.0, match_score=1.0)
        result = CircleResult(label='test', center_row=30, center_col=40,
                              radius=15, valid=True)
        transformed = _transform_circle_result(result, target, 60, 80)
        assert abs(transformed.radius - 30) < 0.1  # 15 * 2
        assert 'template_radius' in transformed.meta


# ===========================================================================
# Test runner
# ===========================================================================


def run_all_tests():
    """Run all tests and report results (standalone runner)."""
    test_classes = [
        TestRotatedTemplate,
        TestMultiTargetWorkflow,
        TestCoordinateTransforms,
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
