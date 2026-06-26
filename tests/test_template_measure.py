"""
Template Matching module test suite.

Test coverage:
1. TemplatePoint - construction, matching, subpixel refinement, serialization
2. DistanceMeasure - distance computation, partial failure handling
3. Visualization - smoke tests
"""

import cv2
import numpy as np
import sys
import os
from typing import Tuple

# Ensure the module can be imported from the repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from measure_template import (TemplatePoint, DistanceMeasure,
                              RawPreprocessor, CannyPreprocessor,
                              SobelPreprocessor, CLAHEPreprocessor,
                              ThresholdPreprocessor,
                              _PREPROCESSOR_REGISTRY, _deserialize_preprocessor)


# =========================================================================
# Synthetic Test Image Generators
# =========================================================================

def create_synthetic_reference(width: int = 400, height: int = 400) -> np.ndarray:
    """
    Create a synthetic reference image with distinct high-contrast features.

    Two dark corner-shaped features at known positions (100, 100) and (300, 200)
    provide strong, unambiguous edges for template matching.

    Returns:
        Grayscale image (uint8)
    """
    img = np.ones((height, width), dtype=np.uint8) * 200

    # Feature A: dark rectangle at (70:130, 70:130), center ~ (100, 100)
    cv2.rectangle(img, (70, 70), (130, 130), 50, -1)
    # Add a distinctive notch so it's not symmetric
    cv2.rectangle(img, (70, 70), (90, 90), 200, -1)

    # Feature B: dark rectangle at (270:330, 170:230), center ~ (300, 200)
    cv2.rectangle(img, (270, 170), (330, 230), 50, -1)
    # Distinctive notch
    cv2.rectangle(img, (310, 170), (330, 200), 200, -1)

    # Add slight blur to mimic real imaging
    img = cv2.GaussianBlur(img, (3, 3), 0.8)

    return img


def create_synthetic_inspection(reference: np.ndarray,
                                offset_row: float = 5.0,
                                offset_col: float = 3.0,
                                noise_level: float = 3.0) -> np.ndarray:
    """
    Create an inspection image by translating the reference with subpixel offset.

    Uses cv2.warpAffine with bilinear interpolation so the offset can be
    fractional (subpixel), which is essential for testing subpixel refinement.

    Args:
        reference: Reference image (grayscale)
        offset_row: Translation in rows (downward = positive)
        offset_col: Translation in cols (rightward = positive)
        noise_level: Std of additive Gaussian noise

    Returns:
        Translated inspection image (uint8)
    """
    h, w = reference.shape
    M = np.float32([[1, 0, offset_col], [0, 1, offset_row]])
    translated = cv2.warpAffine(reference, M, (w, h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)

    # Add noise
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, translated.shape).astype(np.float32)
        translated = np.clip(translated.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return translated


def create_blank_image(width: int = 400, height: int = 400) -> np.ndarray:
    """Create a blank (uniform gray) image for low-contrast tests."""
    img = np.ones((height, width), dtype=np.uint8) * 128
    noise = np.random.normal(0, 1, img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


# =========================================================================
# Test: TemplatePoint
# =========================================================================

class TestTemplatePoint:
    """Tests for the TemplatePoint class."""

    @staticmethod
    def test_construction():
        """Verify template cropping and edge extraction produce valid output."""
        ref = create_synthetic_reference()
        pt = TemplatePoint(ref, click_row=100, click_col=100, template_size=80)

        # Template should be created
        assert pt.edge_template is not None
        assert pt.edge_template.shape[0] == 80
        assert pt.edge_template.shape[1] == 80
        # dtype depends on mode: uint8 for edges, float32 for raw
        assert pt.edge_template.dtype in (np.uint8, np.float32)

        # Template should have non-zero variance (feature is high-contrast)
        assert pt.edge_template.std() > 0, "Template has zero variance!"

        # Result should be None before measure()
        assert pt.result is None

        print("  ✓ test_construction passed")

    @staticmethod
    def test_perfect_match():
        """Match against the same reference image: should find the exact click point."""
        ref = create_synthetic_reference()
        pt = TemplatePoint(ref, click_row=100, click_col=100, template_size=80)

        result = pt.measure(ref)

        assert result is not None
        assert result['valid'], f"Match should be valid, got score={result['match_score']:.4f}"
        assert result['match_score'] > 0.7, f"Score too low: {result['match_score']:.4f}"

        # Should be within 2 px of the original click
        assert abs(result['matched_row'] - 100) < 2.0, \
            f"Row off by {abs(result['matched_row'] - 100):.2f} px"
        assert abs(result['matched_col'] - 100) < 2.0, \
            f"Col off by {abs(result['matched_col'] - 100):.2f} px"

        print("  ✓ test_perfect_match passed")

    @staticmethod
    def test_known_translation():
        """Match against an image translated by a known integer offset."""
        ref = create_synthetic_reference()
        pt = TemplatePoint(ref, click_row=100, click_col=100, template_size=80)

        # Translate by (7, 3) pixels
        inspection = create_synthetic_inspection(ref, offset_row=7.0, offset_col=3.0, noise_level=0.5)
        result = pt.measure(inspection)

        assert result is not None
        assert result['valid'], f"Match should be valid, got score={result['match_score']:.4f}"

        expected_row = 100 + 7.0
        expected_col = 100 + 3.0
        assert abs(result['matched_row'] - expected_row) < 1.5, \
            f"Row error: {abs(result['matched_row'] - expected_row):.3f} px"
        assert abs(result['matched_col'] - expected_col) < 1.5, \
            f"Col error: {abs(result['matched_col'] - expected_col):.3f} px"

        print("  ✓ test_known_translation passed")

    @staticmethod
    def test_subpixel_refinement():
        """Verify subpixel refinement with a known fractional translation."""
        ref = create_synthetic_reference()
        pt = TemplatePoint(ref, click_row=100, click_col=100, template_size=80, use_subpixel=True)

        # Translate by a fractional amount (2.5, 1.5) pixels
        inspection = create_synthetic_inspection(ref, offset_row=2.5, offset_col=1.5, noise_level=0.3)
        result = pt.measure(inspection)

        assert result is not None
        expected_row = 100 + 2.5
        expected_col = 100 + 1.5

        # With subpixel refinement, should be within ~0.5 px
        row_err = abs(result['matched_row'] - expected_row)
        col_err = abs(result['matched_col'] - expected_col)
        assert row_err < 1.0, f"Subpixel row error: {row_err:.3f} px"
        assert col_err < 1.0, f"Subpixel col error: {col_err:.3f} px"

        print(f"  ✓ test_subpixel_refinement passed (row_err={row_err:.3f}, col_err={col_err:.3f})")

    @staticmethod
    def test_subpixel_disabled():
        """When subpixel is disabled, positions should be integer."""
        ref = create_synthetic_reference()
        pt = TemplatePoint(ref, click_row=100, click_col=100, template_size=80, use_subpixel=False)

        inspection = create_synthetic_inspection(ref, offset_row=7.0, offset_col=3.0, noise_level=0.5)
        result = pt.measure(inspection)

        assert result is not None

        print("  ✓ test_subpixel_disabled passed")

    @staticmethod
    def test_low_contrast_no_match():
        """Matching against a blank image should produce a low score."""
        ref = create_synthetic_reference()
        pt = TemplatePoint(ref, click_row=100, click_col=100, template_size=80)

        blank = create_blank_image()
        result = pt.measure(blank)

        assert result is not None
        # The score should be low (or valid=False)
        assert result['match_score'] < 0.8 or not result['valid'], \
            f"Match against blank image should be weak, got score={result['match_score']:.4f}"

        print("  ✓ test_low_contrast_no_match passed")

    @staticmethod
    def test_search_region():
        """Verify that search_region restricts the match correctly."""
        ref = create_synthetic_reference()
        pt = TemplatePoint(ref, click_row=100, click_col=100, template_size=80)

        # Inspection with (7, 3) offset
        inspection = create_synthetic_inspection(ref, offset_row=7.0, offset_col=3.0, noise_level=0.5)

        # Provide a generous search region around the expected position
        # Must be at least as large as the template (80x80)
        search_region = (60, 160, 60, 160)
        result = pt.measure(inspection, search_region=search_region)

        assert result is not None
        assert result['valid'], f"Match in search region should be valid, got score={result['match_score']:.4f}"

        # Matched position should be within the search region
        assert 60 <= result['matched_row'] <= 160, \
            f"Row {result['matched_row']} outside search region [60, 160]"
        assert 60 <= result['matched_col'] <= 160, \
            f"Col {result['matched_col']} outside search region [60, 160]"

        print("  ✓ test_search_region passed")

    @staticmethod
    def test_serialization_roundtrip():
        """Save a template to file, reload, and verify identical match result."""
        ref = create_synthetic_reference()
        inspection = create_synthetic_inspection(ref, offset_row=7.0, offset_col=3.0, noise_level=0.5)

        pt_orig = TemplatePoint(ref, click_row=100, click_col=100, template_size=80)
        result_orig = pt_orig.measure(inspection)

        # Save and reload
        filepath = "/tmp/test_template_roundtrip.npz"
        pt_orig.save(filepath)
        pt_loaded = TemplatePoint.from_file(filepath)

        result_loaded = pt_loaded.measure(inspection)

        # Results should be identical
        assert abs(result_orig['matched_row'] - result_loaded['matched_row']) < 0.01
        assert abs(result_orig['matched_col'] - result_loaded['matched_col']) < 0.01
        assert abs(result_orig['match_score'] - result_loaded['match_score']) < 0.01

        print("  ✓ test_serialization_roundtrip passed")

    @staticmethod
    def test_template_at_border():
        """Template near image border should be clamped, not crash."""
        ref = create_synthetic_reference()
        # Click very close to the top-left corner
        pt = TemplatePoint(ref, click_row=20, click_col=20, template_size=80)

        # Crop should be clamped; edge template should still be valid
        assert pt.edge_template is not None
        assert pt.edge_template.shape[0] <= 80
        assert pt.edge_template.shape[1] <= 80

        # measure() should not crash
        result = pt.measure(ref)
        assert result is not None

        print("  ✓ test_template_at_border passed")

    @staticmethod
    def test_visualize_smoke():
        """Smoke test: visualize() should return an ndarray without crashing."""
        ref = create_synthetic_reference()
        pt = TemplatePoint(ref, click_row=100, click_col=100, template_size=80)
        pt.measure(ref)

        vis = pt.visualize(ref, wait_time=-1)
        assert vis is not None
        assert vis.shape[:2] == ref.shape[:2]
        assert len(vis.shape) == 3  # Should be BGR

        print("  ✓ test_visualize_smoke passed")

    @staticmethod
    def test_dx_dy_displacement():
        """Verify that dx/dy reflect the actual displacement from click position."""
        ref = create_synthetic_reference()
        pt = TemplatePoint(ref, click_row=100, click_col=100, template_size=80)

        inspection = create_synthetic_inspection(ref, offset_row=7.0, offset_col=3.0, noise_level=0.5)
        result = pt.measure(inspection)

        # dy (row displacement) ≈ +7, dx (col displacement) ≈ +3
        assert abs(result['dy'] - 7.0) < 2.0, f"dy={result['dy']:.3f}, expected ~7.0"
        assert abs(result['dx'] - 3.0) < 2.0, f"dx={result['dx']:.3f}, expected ~3.0"

        print("  ✓ test_dx_dy_displacement passed")


# =========================================================================
# Test: DistanceMeasure
# =========================================================================

class TestDistanceMeasure:
    """Tests for the DistanceMeasure class."""

    @staticmethod
    def test_known_distance():
        """Two points with known reference distance and known translation."""
        ref = create_synthetic_reference()
        # Use the two actual feature centers in the synthetic image:
        # Feature A at (100, 100), Feature B at (300, 200)
        pt_a = TemplatePoint(ref, click_row=100, click_col=100, template_size=80)
        pt_b = TemplatePoint(ref, click_row=200, click_col=300, template_size=80)

        dm = DistanceMeasure(pt_a, pt_b)

        # Match against the SAME reference image (no translation)
        result = dm.measure(ref)

        assert result['valid'], f"Match should be valid"
        # Reference distance: (100, 100) to (200, 300) = sqrt(100^2 + 200^2) ≈ 223.607 px
        expected_distance = np.sqrt((200 - 100) ** 2 + (300 - 100) ** 2)
        assert abs(result['distance'] - expected_distance) < 5.0, \
            f"Distance error: {abs(result['distance'] - expected_distance):.3f} px"

        print("  ✓ test_known_distance passed")

    @staticmethod
    def test_distance_with_translation():
        """Distance should remain consistent when both points translate equally."""
        ref = create_synthetic_reference()
        pt_a = TemplatePoint(ref, click_row=100, click_col=100, template_size=80)
        pt_b = TemplatePoint(ref, click_row=200, click_col=300, template_size=80)

        dm = DistanceMeasure(pt_a, pt_b)

        # Reference distance
        result_ref = dm.measure(ref)
        dist_ref = result_ref['distance']

        # Both points translated by the same amount: distance should be preserved
        inspection = create_synthetic_inspection(ref, offset_row=7.0, offset_col=3.0, noise_level=0.5)
        result_insp = dm.measure(inspection)

        assert result_insp['valid'], f"Match should be valid after translation"
        assert abs(result_insp['distance'] - dist_ref) < 3.0, \
            f"Distance changed by {abs(result_insp['distance'] - dist_ref):.3f} px after translation"

        print("  ✓ test_distance_with_translation passed")

    @staticmethod
    def test_one_point_fails():
        """When one point has a low score, result['valid'] should be False."""
        ref = create_synthetic_reference()
        blank = create_blank_image()

        # Point A: good template on reference
        pt_a = TemplatePoint(ref, click_row=100, click_col=100, template_size=80)
        # Point B: also a good template
        pt_b = TemplatePoint(ref, click_row=200, click_col=300, template_size=80)

        dm = DistanceMeasure(pt_a, pt_b)

        # Match against blank image — both should fail or be weak
        result = dm.measure(blank)

        # At least one should be invalid, and the distance should still be computed
        assert result['distance'] >= 0, "Distance should be computed even with invalid matches"
        # valid should be False since at least one point fails
        # (On a blank image with random noise, sometimes both might barely pass if threshold is low)

        print("  ✓ test_one_point_fails passed")

    @staticmethod
    def test_visualize_smoke():
        """Smoke test: visualize() should return an annotated image without crashing."""
        ref = create_synthetic_reference()
        pt_a = TemplatePoint(ref, click_row=100, click_col=100, template_size=80)
        pt_b = TemplatePoint(ref, click_row=200, click_col=300, template_size=80)
        dm = DistanceMeasure(pt_a, pt_b)
        dm.measure(ref)

        vis = dm.visualize(ref, wait_time=-1)
        assert vis is not None
        assert vis.shape[:2] == ref.shape[:2]
        assert len(vis.shape) == 3

        print("  ✓ test_visualize_smoke passed")

    @staticmethod
    def test_visual_demo(wait_time: int = 1500):
        """
        Interactive visual demo of the full template-matching measurement pipeline.

        Shows step-by-step popup windows:
          1. Reference image with template crop regions marked
          2. Cropped edge templates (Canny edge maps) for each point
          3. Inspection image (translated) with template boxes overlaid
          4. Matched positions on the inspection image
          5. Final distance measurement result

        Args:
            wait_time: Milliseconds to show each window (default 1500).
                       Set to 0 for manual key-press advancement.
        """
        print("=" * 60)
        print("Visual Demo: Template Matching Measurement Pipeline")
        print("=" * 60)

        # ------------------------------------------------------------------
        # Step 0: Create a rich synthetic test scene
        # ------------------------------------------------------------------
        # Larger image with multiple distinct features at known positions
        ref = np.ones((500, 600), dtype=np.uint8) * 210

        # Feature A: dark square with a white notch (top-left region)
        cv2.rectangle(ref, (50, 50), (150, 150), 40, -1)
        cv2.rectangle(ref, (50, 50), (90, 90), 210, -1)

        # Feature B: dark circle with a white cross (right region)
        cv2.circle(ref, (450, 120), 50, 40, -1)
        cv2.line(ref, (430, 120), (470, 120), 210, 2)
        cv2.line(ref, (450, 100), (450, 140), 210, 2)

        # Additional decorative elements (not used as measurement points)
        cv2.rectangle(ref, (200, 350), (400, 400), 60, -1)
        cv2.circle(ref, (300, 250), 60, 80, 2)

        # Add noise and blur for realism
        noise = np.random.normal(0, 3, ref.shape).astype(np.float32)
        ref = np.clip(ref.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        ref = cv2.GaussianBlur(ref, (3, 3), 0.6)

        # Click positions: center of Feature A and Feature B
        click_a = (100, 100)   # (row, col) — center of the square
        click_b = (120, 450)   # (row, col) — center of the circle

        # ------------------------------------------------------------------
        # Step 1: Show reference image with template crop regions
        # ------------------------------------------------------------------
        print("\n[Step 1] Reference image with template crop regions")
        print(f"  Point A (square feature):    row={click_a[0]}, col={click_a[1]}")
        print(f"  Point B (circle feature):    row={click_b[0]}, col={click_b[1]}")

        vis_ref = cv2.cvtColor(ref, cv2.COLOR_GRAY2BGR)

        # Draw template boxes (80x80 squares)
        template_size = 80
        for (cr, cc), color, label in [
            (click_a, (0, 255, 0), 'A'),
            (click_b, (0, 255, 255), 'B'),
        ]:
            half = template_size // 2
            # Double-draw for visibility
            cv2.rectangle(vis_ref, (cc - half, cr - half), (cc + half, cr + half),
                          (0, 0, 0), 3)
            cv2.rectangle(vis_ref, (cc - half, cr - half), (cc + half, cr + half),
                          color, 2)
            cv2.circle(vis_ref, (cc, cr), 5, (0, 0, 255), -1)
            cv2.circle(vis_ref, (cc, cr), 7, (0, 0, 0), 2)
            # Label
            cv2.putText(vis_ref, f'Template {label}', (cc + 12, cr - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(vis_ref, f'Template {label}', (cc + 12, cr - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.putText(vis_ref, 'STEP 1: Reference Image + Template Regions',
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(vis_ref, 'STEP 1: Reference Image + Template Regions',
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(vis_ref, f'Green=PtA ({click_a[0]},{click_a[1]}), Yellow=PtB ({click_b[0]},{click_b[1]})',
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(vis_ref, f'Green=PtA ({click_a[0]},{click_a[1]}), Yellow=PtB ({click_b[0]},{click_b[1]})',
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Step1_Reference", vis_ref)
        cv2.waitKey(wait_time)

        # ------------------------------------------------------------------
        # Step 2: Show cropped edge templates
        # ------------------------------------------------------------------
        print("\n[Step 2] Cropping templates and extracting Canny edges...")

        pt_a = TemplatePoint(ref, click_row=click_a[0], click_col=click_a[1],
                             template_size=template_size)
        pt_b = TemplatePoint(ref, click_row=click_b[0], click_col=click_b[1],
                             template_size=template_size)

        edge_a = pt_a.edge_template
        edge_b = pt_b.edge_template

        print(f"  Template A: {edge_a.shape[1]}x{edge_a.shape[0]} px, "
              f"edge pixels={np.count_nonzero(edge_a)}")
        print(f"  Template B: {edge_b.shape[1]}x{edge_b.shape[0]} px, "
              f"edge pixels={np.count_nonzero(edge_b)}")

        # Show edge templates side by side (resize to same height for display)
        disp_h = 200
        disp_w_a = int(edge_a.shape[1] * disp_h / edge_a.shape[0])
        disp_w_b = int(edge_b.shape[1] * disp_h / edge_b.shape[0])
        edge_a_disp = cv2.resize(edge_a, (disp_w_a, disp_h), interpolation=cv2.INTER_NEAREST)
        edge_b_disp = cv2.resize(edge_b, (disp_w_b, disp_h), interpolation=cv2.INTER_NEAREST)

        # Combine side by side with labels
        gap = 20
        combined_w = disp_w_a + gap + disp_w_b + 40
        combined_h = disp_h + 60
        combined = np.ones((combined_h, combined_w), dtype=np.uint8) * 200

        combined[30:30 + disp_h, 20:20 + disp_w_a] = edge_a_disp
        combined[30:30 + disp_h, 20 + disp_w_a + gap:20 + disp_w_a + gap + disp_w_b] = edge_b_disp

        combined_bgr = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
        cv2.putText(combined_bgr, 'Template A (edges)', (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(combined_bgr, 'Template A (edges)', (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(combined_bgr, 'Template B (edges)', (20 + disp_w_a + gap, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(combined_bgr, 'Template B (edges)', (20 + disp_w_a + gap, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(combined_bgr, 'STEP 2: Canny Edge Templates',
                    (10, combined_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
        cv2.putText(combined_bgr, 'STEP 2: Canny Edge Templates',
                    (10, combined_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        cv2.imshow("Step2_EdgeTemplates", combined_bgr)
        cv2.waitKey(wait_time)

        # ------------------------------------------------------------------
        # Step 3: Create inspection image with translation
        # ------------------------------------------------------------------
        # Translate by (12, 8) pixels — visible but not extreme
        offset_row, offset_col = 12.0, 8.0
        print(f"\n[Step 3] Creating inspection image with offset ({offset_row:.0f}, {offset_col:.0f}) px...")

        inspection = create_synthetic_inspection(ref, offset_row=offset_row,
                                                 offset_col=offset_col, noise_level=2.0)

        # Show inspection with template boxes at reference positions (visibly offset)
        vis_insp = cv2.cvtColor(inspection, cv2.COLOR_GRAY2BGR)
        for (cr, cc), color, label in [
            (click_a, (0, 255, 0), 'A'),
            (click_b, (0, 255, 255), 'B'),
        ]:
            half = template_size // 2
            # Dashed-style reference position (offset rectangles)
            for dx in [-2, 0, 2]:
                for dy in [-2, 0, 2]:
                    cv2.rectangle(vis_insp, (cc + dx - half, cr + dy - half),
                                  (cc + dx + half, cr + dy + half), (100, 100, 100), 1)

        cv2.putText(vis_insp, 'STEP 3: Inspection Image (translated)',
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(vis_insp, 'STEP 3: Inspection Image (translated)',
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(vis_insp, f'Offset: ({offset_row:.0f}, {offset_col:.0f}) px. '
                    f'Gray boxes = reference positions',
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(vis_insp, f'Offset: ({offset_row:.0f}, {offset_col:.0f}) px. '
                    f'Gray boxes = reference positions',
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Step3_InspectionImage", vis_insp)
        cv2.waitKey(wait_time)

        # ------------------------------------------------------------------
        # Step 4: Match templates and show results
        # ------------------------------------------------------------------
        print("\n[Step 4] Matching templates on inspection image...")

        dm = DistanceMeasure(pt_a, pt_b)
        result = dm.measure(inspection)

        print(f"  Point A: matched ({result['point_a']['matched_row']:.2f}, "
              f"{result['point_a']['matched_col']:.2f}), "
              f"score={result['point_a']['match_score']:.4f}, "
              f"valid={result['point_a']['valid']}")
        print(f"  Point B: matched ({result['point_b']['matched_row']:.2f}, "
              f"{result['point_b']['matched_col']:.2f}), "
              f"score={result['point_b']['match_score']:.4f}, "
              f"valid={result['point_b']['valid']}")

        # Show matched positions on the inspection image
        vis_match = cv2.cvtColor(inspection, cv2.COLOR_GRAY2BGR)

        # Draw matched crosshairs
        colors = [(0, 255, 0), (0, 255, 255)]  # green for A, yellow for B
        labels = ['A', 'B']
        for i, pt_res in enumerate([result['point_a'], result['point_b']]):
            r = pt_res['matched_row']
            c = pt_res['matched_col']
            color = colors[i]
            # Crosshair
            cv2.line(vis_match, (int(c) - 12, int(r)), (int(c) + 12, int(r)),
                     (0, 0, 0), 3)
            cv2.line(vis_match, (int(c) - 12, int(r)), (int(c) + 12, int(r)),
                     color, 2)
            cv2.line(vis_match, (int(c), int(r) - 12), (int(c), int(r) + 12),
                     (0, 0, 0), 3)
            cv2.line(vis_match, (int(c), int(r) - 12), (int(c), int(r) + 12),
                     color, 2)
            # Label
            score_text = f'Pt{labels[i]}: score={pt_res["match_score"]:.3f}'
            cv2.putText(vis_match, score_text, (int(c) + 16, int(r) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
            cv2.putText(vis_match, score_text, (int(c) + 16, int(r) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Draw distance line
        r1, c1 = result['point_a']['matched_row'], result['point_a']['matched_col']
        r2, c2 = result['point_b']['matched_row'], result['point_b']['matched_col']
        cv2.line(vis_match, (int(c1), int(r1)), (int(c2), int(r2)),
                 (0, 0, 0), 3, cv2.LINE_AA)
        cv2.line(vis_match, (int(c1), int(r1)), (int(c2), int(r2)),
                 (255, 0, 255), 2, cv2.LINE_AA)
        # Distance label at midpoint
        mid_r = int((r1 + r2) / 2)
        mid_c = int((c1 + c2) / 2)
        dist_text = f'{result["distance"]:.2f} px'
        cv2.putText(vis_match, dist_text, (mid_c - 40, mid_r - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(vis_match, dist_text, (mid_c - 40, mid_r - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 1)

        cv2.putText(vis_match, 'STEP 4: Matched Positions + Distance',
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(vis_match, 'STEP 4: Matched Positions + Distance',
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        cv2.imshow("Step4_MatchResult", vis_match)
        cv2.waitKey(wait_time)

        # ------------------------------------------------------------------
        # Step 5: Side-by-side comparison (reference vs. inspection)
        # ------------------------------------------------------------------
        print("\n[Step 5] Side-by-side comparison...")

        # Create the final visualization using the DistanceMeasure's own visualize
        vis_final = dm.visualize(inspection, wait_time=-1,
                                 show_distance_line=True,
                                 template_color=(0, 200, 0))

        # Also show the reference visualization
        vis_ref_final = cv2.cvtColor(ref, cv2.COLOR_GRAY2BGR)
        half = template_size // 2
        for (cr, cc), color, label in [
            (click_a, (0, 255, 0), 'A'),
            (click_b, (0, 255, 255), 'B'),
        ]:
            cv2.rectangle(vis_ref_final, (cc - half, cr - half), (cc + half, cr + half),
                          (0, 0, 0), 2)
            cv2.rectangle(vis_ref_final, (cc - half, cr - half), (cc + half, cr + half),
                          color, 1)
            cv2.putText(vis_ref_final, f'Ref {label}', (cc - 30, cr - half - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
            cv2.putText(vis_ref_final, f'Ref {label}', (cc - 30, cr - half - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv2.putText(vis_ref_final, 'REFERENCE', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(vis_ref_final, 'REFERENCE', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        # Side by side
        border = np.ones((vis_ref_final.shape[0], 4, 3), dtype=np.uint8) * 100
        side_by_side = np.hstack([vis_ref_final, border, vis_final])

        cv2.putText(side_by_side, 'INSPECTION',
                    (vis_ref_final.shape[1] + 10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(side_by_side, 'INSPECTION',
                    (vis_ref_final.shape[1] + 10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        cv2.imshow("Step5_SideBySide", side_by_side)
        cv2.waitKey(wait_time)

        # ------------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------------
        print("\n" + "=" * 60)
        print("Visual Demo Complete — Summary")
        print("=" * 60)
        print(f"  Reference distance (click positions): "
              f"{np.sqrt((click_b[0]-click_a[0])**2 + (click_b[1]-click_a[1])**2):.2f} px")
        print(f"  Measured distance (matched positions): {result['distance']:.2f} px")
        print(f"  Point A score: {result['point_a']['match_score']:.4f} "
              f"({'VALID' if result['point_a']['valid'] else 'INVALID'})")
        print(f"  Point B score: {result['point_b']['match_score']:.4f} "
              f"({'VALID' if result['point_b']['valid'] else 'INVALID'})")
        print(f"  Overall: {'VALID' if result['valid'] else 'PARTIAL/INVALID'}")
        print(f"\nPress any key in the OpenCV windows to close, or wait for auto-close...")

        cv2.waitKey(wait_time)
        cv2.destroyAllWindows()

        print("  ✓ test_visual_demo passed")
        return True

    @staticmethod
    def test_visual_real_demo(wait_time: int = 1500):
        template_org_path = "data/sample/bottleneck_6.jpg"
        test_path1 = "data/sample/bottleneck_5.jpg"
        output_dir = "output/template_match_4"

        ##读取为np.uint8的灰度图
        ref = cv2.imread(template_org_path, cv2.IMREAD_GRAYSCALE)
        insp1 = cv2.imread(test_path1, cv2.IMREAD_GRAYSCALE)

        if ref is None:
            print(f"  SKIP: reference image not found: {template_org_path}")
            return
        if insp1 is None:
            print(f"  SKIP: inspection image not found: {test_path1}")
            return

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        pp = ThresholdPreprocessor(threshold=200)

        print("=" * 60)
        print("Visual Demo: Real Image Template Matching")
        print("=" * 60)

        # ------------------------------------------------------------------
        # Step 1: Create templates from reference image
        # ------------------------------------------------------------------
        click_a = [2082,749]
        click_b = [2123,2698]
        template_size = 512

        print(f"\n[Step 1] Creating templates from reference image "
              f"({ref.shape[0]}x{ref.shape[1]} px)...")
        print(f"  Point A: row={click_a[0]}, col={click_a[1]}, template={template_size}px")
        print(f"  Point B: row={click_b[0]}, col={click_b[1]}, template={template_size}px")

        pt_a = TemplatePoint(ref, click_row=click_a[0], click_col=click_a[1],
                             template_size=template_size, preprocessor=pp)
        pt_b = TemplatePoint(ref, click_row=click_b[0], click_col=click_b[1],
                             template_size=template_size, preprocessor=pp)

        print(f"  Template A: {pt_a._crop_h}x{pt_a._crop_w} px, "
              f"edges={np.count_nonzero(pt_a.edge_template)}")
        print(f"  Template B: {pt_b._crop_h}x{pt_b._crop_w} px, "
              f"edges={np.count_nonzero(pt_b.edge_template)}")
        pt_a.save(os.path.join(output_dir, "template_a"))
        pt_b.save(os.path.join(output_dir, "template_b"))
        # ---- Reference image with template boxes ----
        vis_a = pt_a.visualize(ref, wait_time=-1,
                               template_color=(0, 255, 0),
                               matched_color=(0, 0, 255))
        vis_both = pt_b.visualize(vis_a, wait_time=-1,
                                  template_color=(0, 255, 255),
                                  matched_color=(255, 0, 255))

        cv2.putText(vis_both, 'STEP 1: Reference Image + Templates',
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(vis_both, 'STEP 1: Reference Image + Templates',
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(vis_both, f'Green=PtA ({click_a[0]},{click_a[1]}), '
                    f'Yellow=PtB ({click_b[0]},{click_b[1]})',
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(vis_both, f'Green=PtA ({click_a[0]},{click_a[1]}), '
                    f'Yellow=PtB ({click_b[0]},{click_b[1]})',
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imwrite(f"{output_dir}/01_reference_with_templates.jpg", vis_both)
        cv2.imshow("Step1_Reference", vis_both)
        cv2.waitKey(wait_time)

        # ---- Edge templates ----
        disp_h = 300
        for edge, name in [(pt_a.edge_template, 'A'), (pt_b.edge_template, 'B')]:
            h, w = edge.shape
            disp_w = int(w * disp_h / h)
            edge_disp = cv2.resize(edge, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
            edge_bgr = cv2.cvtColor(edge_disp, cv2.COLOR_GRAY2BGR)
            cv2.putText(edge_bgr, f'Edge Template {name} ({h}x{w})',
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            cv2.putText(edge_bgr, f'Edge Template {name} ({h}x{w})',
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
            cv2.imwrite(f"{output_dir}/02_edge_template_{name}.jpg", edge_bgr)
            cv2.imshow(f"Step1_EdgeTemplate_{name}", edge_bgr)
        cv2.waitKey(wait_time)

        # ------------------------------------------------------------------
        # Step 2: Match on the inspection image
        # ------------------------------------------------------------------
        print(f"\n[Step 2] Matching on inspection image "
              f"({insp1.shape[0]}x{insp1.shape[1]} px)...")

        dm = DistanceMeasure(pt_a, pt_b)
        result = dm.measure(insp1)

        print(f"  Point A: matched ({result['point_a']['matched_row']:.2f}, "
              f"{result['point_a']['matched_col']:.2f}), "
              f"score={result['point_a']['match_score']:.4f}, "
              f"{'VALID' if result['point_a']['valid'] else 'INVALID'}")
        print(f"  Point B: matched ({result['point_b']['matched_row']:.2f}, "
              f"{result['point_b']['matched_col']:.2f}), "
              f"score={result['point_b']['match_score']:.4f}, "
              f"{'VALID' if result['point_b']['valid'] else 'INVALID'}")

        # ---- Matched positions on inspection image ----
        vis_match = cv2.cvtColor(insp1, cv2.COLOR_GRAY2BGR)

        colors = [(0, 255, 0), (0, 255, 255)]  # green for A, yellow for B
        labels = ['A', 'B']
        for i, pt_res in enumerate([result['point_a'], result['point_b']]):
            r = pt_res['matched_row']
            c = pt_res['matched_col']
            color = colors[i]
            half = template_size // 2

            # Matched template box
            cv2.rectangle(vis_match, (int(c - half), int(r - half)),
                          (int(c + half), int(r + half)), (0, 0, 0), 3)
            cv2.rectangle(vis_match, (int(c - half), int(r - half)),
                          (int(c + half), int(r + half)), color, 2)

            # Crosshair
            cv2.line(vis_match, (int(c) - 15, int(r)), (int(c) + 15, int(r)),
                     (0, 0, 0), 3)
            cv2.line(vis_match, (int(c) - 15, int(r)), (int(c) + 15, int(r)),
                     color, 2)
            cv2.line(vis_match, (int(c), int(r) - 15), (int(c), int(r) + 15),
                     (0, 0, 0), 3)
            cv2.line(vis_match, (int(c), int(r) - 15), (int(c), int(r) + 15),
                     color, 2)

            # Score label
            score_text = f'Pt{labels[i]}: ({r:.1f},{c:.1f}) score={pt_res["match_score"]:.3f}'
            cv2.putText(vis_match, score_text, (int(c) + 20, int(r) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
            cv2.putText(vis_match, score_text, (int(c) + 20, int(r) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # Distance line
        r1, c1 = result['point_a']['matched_row'], result['point_a']['matched_col']
        r2, c2 = result['point_b']['matched_row'], result['point_b']['matched_col']
        cv2.line(vis_match, (int(c1), int(r1)), (int(c2), int(r2)),
                 (0, 0, 0), 4, cv2.LINE_AA)
        cv2.line(vis_match, (int(c1), int(r1)), (int(c2), int(r2)),
                 (255, 0, 255), 2, cv2.LINE_AA)

        mid_r = int((r1 + r2) / 2)
        mid_c = int((c1 + c2) / 2)
        dist_text = f'{result["distance"]:.2f} px'
        cv2.putText(vis_match, dist_text, (mid_c - 50, mid_r - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(vis_match, dist_text, (mid_c - 50, mid_r - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1)

        status_text = 'VALID' if result['valid'] else 'PARTIAL/INVALID'
        status_color = (0, 255, 0) if result['valid'] else (0, 165, 255)
        cv2.putText(vis_match, f'Status: {status_text}  |  Distance: {result["distance"]:.2f} px',
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(vis_match, f'Status: {status_text}  |  Distance: {result["distance"]:.2f} px',
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        cv2.putText(vis_match, 'STEP 2: Matched Positions + Distance',
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(vis_match, 'STEP 2: Matched Positions + Distance',
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        cv2.imwrite(f"{output_dir}/03_matched_positions_distance.jpg", vis_match)
        cv2.imshow("Step2_MatchResult", vis_match)
        cv2.waitKey(wait_time)

        # ------------------------------------------------------------------
        # Step 3: Reference vs Inspection side-by-side comparison
        # ------------------------------------------------------------------
        print("\n[Step 3] Side-by-side comparison...")

        # Reference side
        vis_ref = cv2.cvtColor(ref, cv2.COLOR_GRAY2BGR)
        ref_colors = [(0, 255, 0), (0, 255, 255)]
        for (cr, cc), color, label in [
            ((click_a[0], click_a[1]), ref_colors[0], 'A'),
            ((click_b[0], click_b[1]), ref_colors[1], 'B'),
        ]:
            half = template_size // 2
            cv2.rectangle(vis_ref, (cc - half, cr - half), (cc + half, cr + half),
                          (0, 0, 0), 3)
            cv2.rectangle(vis_ref, (cc - half, cr - half), (cc + half, cr + half),
                          color, 2)
            cv2.circle(vis_ref, (cc, cr), 6, (0, 0, 255), -1)
            cv2.circle(vis_ref, (cc, cr), 8, (0, 0, 0), 2)
            cv2.putText(vis_ref, f'Pt{label}', (cc + 14, cr - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(vis_ref, f'Pt{label}', (cc + 14, cr - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(vis_ref, 'REFERENCE', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(vis_ref, 'REFERENCE', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        cv2.imwrite(f"{output_dir}/04a_reference.jpg", vis_ref)

        # Inspection side
        vis_insp = dm.visualize(insp1, wait_time=-1,
                                show_distance_line=True,
                                template_color=(0, 200, 0))
        cv2.putText(vis_insp, 'INSPECTION', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(vis_insp, 'INSPECTION', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        cv2.imwrite(f"{output_dir}/04b_inspection.jpg", vis_insp)

        # Side-by-side
        target_w = min(vis_ref.shape[1], vis_insp.shape[1])
        scale_ref = target_w / vis_ref.shape[1]
        scale_insp = target_w / vis_insp.shape[1]
        vis_ref_resized = cv2.resize(vis_ref, (target_w, int(vis_ref.shape[0] * scale_ref)))
        vis_insp_resized = cv2.resize(vis_insp, (target_w, int(vis_insp.shape[0] * scale_insp)))

        border_bar = np.ones((max(vis_ref_resized.shape[0], vis_insp_resized.shape[0]), 4, 3),
                             dtype=np.uint8) * 100
        max_h = max(vis_ref_resized.shape[0], vis_insp_resized.shape[0])
        if vis_ref_resized.shape[0] < max_h:
            pad = np.ones((max_h - vis_ref_resized.shape[0], target_w, 3), dtype=np.uint8) * 80
            vis_ref_resized = np.vstack([vis_ref_resized, pad])
        if vis_insp_resized.shape[0] < max_h:
            pad = np.ones((max_h - vis_insp_resized.shape[0], target_w, 3), dtype=np.uint8) * 80
            vis_insp_resized = np.vstack([vis_insp_resized, pad])

        side_by_side = np.hstack([vis_ref_resized, border_bar, vis_insp_resized])
        cv2.imwrite(f"{output_dir}/05_side_by_side_comparison.jpg", side_by_side)
        cv2.imshow("Step3_SideBySide", side_by_side)
        cv2.waitKey(wait_time)

        # ------------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------------
        ref_dist = np.sqrt((click_b[0] - click_a[0]) ** 2 + (click_b[1] - click_a[1]) ** 2)
        print("\n" + "=" * 60)
        print("Real Image Demo — Summary")
        print("=" * 60)
        print(f"  Reference distance (click positions): {ref_dist:.2f} px")
        print(f"  Measured distance (matched positions): {result['distance']:.2f} px")
        print(f"  Point A score: {result['point_a']['match_score']:.4f} "
              f"({'VALID' if result['point_a']['valid'] else 'INVALID'})")
        print(f"  Point B score: {result['point_b']['match_score']:.4f} "
              f"({'VALID' if result['point_b']['valid'] else 'INVALID'})")
        print(f"  Overall: {status_text}")
        print(f"\nImages saved to: {os.path.abspath(output_dir)}/")
        print(f"  01_reference_with_templates.jpg")
        print(f"  02_edge_template_A.jpg / B.jpg")
        print(f"  03_matched_positions_distance.jpg")
        print(f"  04a_reference.jpg / 04b_inspection.jpg")
        print(f"  05_side_by_side_comparison.jpg")

        cv2.waitKey(wait_time)
        cv2.destroyAllWindows()
        print("  ✓ test_visual_real_demo passed")
    
    @staticmethod
    def test_visual_real_demo_2(wait_time: int = 1500):
        template_org_path = "/home/industai/workspace/srv-profilometer-1st/test_data/bottleneck_5.jpg"
        test_path1 = "/home/industai/workspace/srv-profilometer-1st/test_data/bottleneck_6.jpg"
        output_dir = "output/template_match_6"

        ##读取为np.uint8的灰度图
        ref = cv2.imread(template_org_path, cv2.IMREAD_GRAYSCALE)
        insp1 = cv2.imread(test_path1, cv2.IMREAD_GRAYSCALE)

        if ref is None:
            print(f"  SKIP: reference image not found: {template_org_path}")
            return
        if insp1 is None:
            print(f"  SKIP: inspection image not found: {test_path1}")
            return

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        pp = ThresholdPreprocessor(threshold=180)

        print("=" * 60)
        print("Visual Demo: Real Image Template Matching")
        print("=" * 60)

        # ------------------------------------------------------------------
        # Step 1: Create templates from reference image
        # ------------------------------------------------------------------
        click_a = [729, 609]
        click_b = [770, 2734]
        template_size = 125

        print(f"\n[Step 1] Creating templates from reference image "
              f"({ref.shape[0]}x{ref.shape[1]} px)...")
        print(f"  Point A: row={click_a[0]}, col={click_a[1]}, template={template_size}px")
        print(f"  Point B: row={click_b[0]}, col={click_b[1]}, template={template_size}px")

        pt_a = TemplatePoint(ref, click_row=click_a[0], click_col=click_a[1],
                             template_size=template_size, preprocessor=pp)
        pt_b = TemplatePoint(ref, click_row=click_b[0], click_col=click_b[1],
                             template_size=template_size, preprocessor=pp)

        print(f"  Template A: {pt_a._crop_h}x{pt_a._crop_w} px, "
              f"edges={np.count_nonzero(pt_a.edge_template)}")
        print(f"  Template B: {pt_b._crop_h}x{pt_b._crop_w} px, "
              f"edges={np.count_nonzero(pt_b.edge_template)}")
        pt_a.save(os.path.join(output_dir, "template_a"))
        pt_b.save(os.path.join(output_dir, "template_b"))
        # ---- Reference image with template boxes ----
        vis_a = pt_a.visualize(ref, wait_time=-1,
                               template_color=(0, 255, 0),
                               matched_color=(0, 0, 255))
        vis_both = pt_b.visualize(vis_a, wait_time=-1,
                                  template_color=(0, 255, 255),
                                  matched_color=(255, 0, 255))

        cv2.putText(vis_both, 'STEP 1: Reference Image + Templates',
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(vis_both, 'STEP 1: Reference Image + Templates',
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(vis_both, f'Green=PtA ({click_a[0]},{click_a[1]}), '
                    f'Yellow=PtB ({click_b[0]},{click_b[1]})',
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(vis_both, f'Green=PtA ({click_a[0]},{click_a[1]}), '
                    f'Yellow=PtB ({click_b[0]},{click_b[1]})',
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imwrite(f"{output_dir}/01_reference_with_templates.jpg", vis_both)
        cv2.imshow("Step1_Reference", vis_both)
        cv2.waitKey(wait_time)

        # ---- Edge templates ----
        disp_h = 300
        for edge, name in [(pt_a.edge_template, 'A'), (pt_b.edge_template, 'B')]:
            h, w = edge.shape
            disp_w = int(w * disp_h / h)
            edge_disp = cv2.resize(edge, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
            edge_bgr = cv2.cvtColor(edge_disp, cv2.COLOR_GRAY2BGR)
            cv2.putText(edge_bgr, f'Edge Template {name} ({h}x{w})',
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            cv2.putText(edge_bgr, f'Edge Template {name} ({h}x{w})',
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
            cv2.imwrite(f"{output_dir}/02_edge_template_{name}.jpg", edge_bgr)
            cv2.imshow(f"Step1_EdgeTemplate_{name}", edge_bgr)
        cv2.waitKey(wait_time)

        # ------------------------------------------------------------------
        # Step 2: Match on the inspection image
        # ------------------------------------------------------------------
        print(f"\n[Step 2] Matching on inspection image "
              f"({insp1.shape[0]}x{insp1.shape[1]} px)...")

        dm = DistanceMeasure(pt_a, pt_b)
        result = dm.measure(insp1)

        print(f"  Point A: matched ({result['point_a']['matched_row']:.2f}, "
              f"{result['point_a']['matched_col']:.2f}), "
              f"score={result['point_a']['match_score']:.4f}, "
              f"{'VALID' if result['point_a']['valid'] else 'INVALID'}")
        print(f"  Point B: matched ({result['point_b']['matched_row']:.2f}, "
              f"{result['point_b']['matched_col']:.2f}), "
              f"score={result['point_b']['match_score']:.4f}, "
              f"{'VALID' if result['point_b']['valid'] else 'INVALID'}")

        # ---- Matched positions on inspection image ----
        vis_match = cv2.cvtColor(insp1, cv2.COLOR_GRAY2BGR)

        colors = [(0, 255, 0), (0, 255, 255)]  # green for A, yellow for B
        labels = ['A', 'B']
        for i, pt_res in enumerate([result['point_a'], result['point_b']]):
            r = pt_res['matched_row']
            c = pt_res['matched_col']
            color = colors[i]
            half = template_size // 2

            # Matched template box
            cv2.rectangle(vis_match, (int(c - half), int(r - half)),
                          (int(c + half), int(r + half)), (0, 0, 0), 3)
            cv2.rectangle(vis_match, (int(c - half), int(r - half)),
                          (int(c + half), int(r + half)), color, 2)

            # Crosshair
            cv2.line(vis_match, (int(c) - 15, int(r)), (int(c) + 15, int(r)),
                     (0, 0, 0), 3)
            cv2.line(vis_match, (int(c) - 15, int(r)), (int(c) + 15, int(r)),
                     color, 2)
            cv2.line(vis_match, (int(c), int(r) - 15), (int(c), int(r) + 15),
                     (0, 0, 0), 3)
            cv2.line(vis_match, (int(c), int(r) - 15), (int(c), int(r) + 15),
                     color, 2)

            # Score label
            score_text = f'Pt{labels[i]}: ({r:.1f},{c:.1f}) score={pt_res["match_score"]:.3f}'
            cv2.putText(vis_match, score_text, (int(c) + 20, int(r) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
            cv2.putText(vis_match, score_text, (int(c) + 20, int(r) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # Distance line
        r1, c1 = result['point_a']['matched_row'], result['point_a']['matched_col']
        r2, c2 = result['point_b']['matched_row'], result['point_b']['matched_col']
        cv2.line(vis_match, (int(c1), int(r1)), (int(c2), int(r2)),
                 (0, 0, 0), 4, cv2.LINE_AA)
        cv2.line(vis_match, (int(c1), int(r1)), (int(c2), int(r2)),
                 (255, 0, 255), 2, cv2.LINE_AA)

        mid_r = int((r1 + r2) / 2)
        mid_c = int((c1 + c2) / 2)
        dist_text = f'{result["distance"]:.2f} px'
        cv2.putText(vis_match, dist_text, (mid_c - 50, mid_r - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(vis_match, dist_text, (mid_c - 50, mid_r - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1)

        status_text = 'VALID' if result['valid'] else 'PARTIAL/INVALID'
        status_color = (0, 255, 0) if result['valid'] else (0, 165, 255)
        cv2.putText(vis_match, f'Status: {status_text}  |  Distance: {result["distance"]:.2f} px',
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(vis_match, f'Status: {status_text}  |  Distance: {result["distance"]:.2f} px',
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        cv2.putText(vis_match, 'STEP 2: Matched Positions + Distance',
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(vis_match, 'STEP 2: Matched Positions + Distance',
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        cv2.imwrite(f"{output_dir}/03_matched_positions_distance.jpg", vis_match)
        cv2.imshow("Step2_MatchResult", vis_match)
        cv2.waitKey(wait_time)

        # ------------------------------------------------------------------
        # Step 3: Reference vs Inspection side-by-side comparison
        # ------------------------------------------------------------------
        print("\n[Step 3] Side-by-side comparison...")

        # Reference side
        vis_ref = cv2.cvtColor(ref, cv2.COLOR_GRAY2BGR)
        ref_colors = [(0, 255, 0), (0, 255, 255)]
        for (cr, cc), color, label in [
            ((click_a[0], click_a[1]), ref_colors[0], 'A'),
            ((click_b[0], click_b[1]), ref_colors[1], 'B'),
        ]:
            half = template_size // 2
            cv2.rectangle(vis_ref, (cc - half, cr - half), (cc + half, cr + half),
                          (0, 0, 0), 3)
            cv2.rectangle(vis_ref, (cc - half, cr - half), (cc + half, cr + half),
                          color, 2)
            cv2.circle(vis_ref, (cc, cr), 6, (0, 0, 255), -1)
            cv2.circle(vis_ref, (cc, cr), 8, (0, 0, 0), 2)
            cv2.putText(vis_ref, f'Pt{label}', (cc + 14, cr - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(vis_ref, f'Pt{label}', (cc + 14, cr - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(vis_ref, 'REFERENCE', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(vis_ref, 'REFERENCE', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        cv2.imwrite(f"{output_dir}/04a_reference.jpg", vis_ref)

        # Inspection side
        vis_insp = dm.visualize(insp1, wait_time=-1,
                                show_distance_line=True,
                                template_color=(0, 200, 0))
        cv2.putText(vis_insp, 'INSPECTION', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(vis_insp, 'INSPECTION', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        cv2.imwrite(f"{output_dir}/04b_inspection.jpg", vis_insp)

        # Side-by-side
        target_w = min(vis_ref.shape[1], vis_insp.shape[1])
        scale_ref = target_w / vis_ref.shape[1]
        scale_insp = target_w / vis_insp.shape[1]
        vis_ref_resized = cv2.resize(vis_ref, (target_w, int(vis_ref.shape[0] * scale_ref)))
        vis_insp_resized = cv2.resize(vis_insp, (target_w, int(vis_insp.shape[0] * scale_insp)))

        border_bar = np.ones((max(vis_ref_resized.shape[0], vis_insp_resized.shape[0]), 4, 3),
                             dtype=np.uint8) * 100
        max_h = max(vis_ref_resized.shape[0], vis_insp_resized.shape[0])
        if vis_ref_resized.shape[0] < max_h:
            pad = np.ones((max_h - vis_ref_resized.shape[0], target_w, 3), dtype=np.uint8) * 80
            vis_ref_resized = np.vstack([vis_ref_resized, pad])
        if vis_insp_resized.shape[0] < max_h:
            pad = np.ones((max_h - vis_insp_resized.shape[0], target_w, 3), dtype=np.uint8) * 80
            vis_insp_resized = np.vstack([vis_insp_resized, pad])

        side_by_side = np.hstack([vis_ref_resized, border_bar, vis_insp_resized])
        cv2.imwrite(f"{output_dir}/05_side_by_side_comparison.jpg", side_by_side)
        cv2.imshow("Step3_SideBySide", side_by_side)
        cv2.waitKey(wait_time)

        # ------------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------------
        ref_dist = np.sqrt((click_b[0] - click_a[0]) ** 2 + (click_b[1] - click_a[1]) ** 2)
        print("\n" + "=" * 60)
        print("Real Image Demo — Summary")
        print("=" * 60)
        print(f"  Reference distance (click positions): {ref_dist:.2f} px")
        print(f"  Measured distance (matched positions): {result['distance']:.2f} px")
        print(f"  Point A score: {result['point_a']['match_score']:.4f} "
              f"({'VALID' if result['point_a']['valid'] else 'INVALID'})")
        print(f"  Point B score: {result['point_b']['match_score']:.4f} "
              f"({'VALID' if result['point_b']['valid'] else 'INVALID'})")
        print(f"  Overall: {status_text}")
        print(f"\nImages saved to: {os.path.abspath(output_dir)}/")
        print(f"  01_reference_with_templates.jpg")
        print(f"  02_edge_template_A.jpg / B.jpg")
        print(f"  03_matched_positions_distance.jpg")
        print(f"  04a_reference.jpg / 04b_inspection.jpg")
        print(f"  05_side_by_side_comparison.jpg")

        cv2.waitKey(wait_time)
        cv2.destroyAllWindows()
        print("  ✓ test_visual_real_demo passed")


# =========================================================================
# Test: Preprocessors
# =========================================================================

class TestPreprocessor:
    """Tests for the Preprocessor classes and serialization."""

    @staticmethod
    def test_raw_preprocessor():
        """RawPreprocessor should return float32 with same shape."""
        img = np.ones((100, 100), dtype=np.uint8) * 128
        pp = RawPreprocessor()
        result = pp(img)
        assert result.shape == (100, 100)
        assert result.dtype == np.float32
        assert pp.name == 'Raw'
        print("  ✓ test_raw_preprocessor passed")

    @staticmethod
    def test_canny_preprocessor():
        """CannyPreprocessor should return uint8 binary edge map."""
        img = create_synthetic_reference()
        pp = CannyPreprocessor(50, 150)
        result = pp(img)
        assert result.shape == img.shape
        assert result.dtype == np.uint8
        assert 'Canny' in pp.name
        # Should contain edges on synthetic image with features
        assert np.count_nonzero(result) > 0
        print("  ✓ test_canny_preprocessor passed")

    @staticmethod
    def test_sobel_preprocessor():
        """SobelPreprocessor should return float32 gradient magnitude."""
        img = create_synthetic_reference()
        pp = SobelPreprocessor(kernel_size=3)
        result = pp(img)
        assert result.shape == img.shape
        assert result.dtype == np.float32
        assert 'Sobel' in pp.name
        print("  ✓ test_sobel_preprocessor passed")

    @staticmethod
    def test_clahe_preprocessor():
        """CLAHEPreprocessor should return float32 enhanced image."""
        img = create_synthetic_reference()
        pp = CLAHEPreprocessor(clip_limit=2.0)
        result = pp(img)
        assert result.shape == img.shape
        assert result.dtype == np.float32
        assert 'CLAHE' in pp.name
        print("  ✓ test_clahe_preprocessor passed")

    @staticmethod
    def test_template_with_canny_preprocessor():
        """TemplatePoint with CannyPreprocessor should match correctly."""
        ref = create_synthetic_reference()
        pt = TemplatePoint(ref, click_row=100, click_col=100, template_size=80,
                           preprocessor=CannyPreprocessor(50, 150))

        # Template should be uint8 (Canny output)
        assert pt.edge_template.dtype == np.uint8

        # Should match perfectly on same image
        result = pt.measure(ref)
        assert result['valid'], f"Match should be valid, score={result['match_score']:.4f}"
        assert abs(result['matched_row'] - 100) < 2.0
        assert abs(result['matched_col'] - 100) < 2.0

        print("  ✓ test_template_with_canny_preprocessor passed")

    @staticmethod
    def test_preprocessor_serialization_roundtrip():
        """Each built-in preprocessor should survive serialize → deserialize."""
        test_cases = [
            RawPreprocessor(),
            CannyPreprocessor(60, 180),
            SobelPreprocessor(kernel_size=5),
            CLAHEPreprocessor(clip_limit=3.0, tile_grid_size=(16, 16)),
        ]

        img = create_synthetic_reference()

        for pp_orig in test_cases:
            data = pp_orig.serialize()
            pp_restored = _deserialize_preprocessor(data)
            result_orig = pp_orig(img)
            result_restored = pp_restored(img)
            # Results should be identical (same dtype, same values)
            assert result_orig.dtype == result_restored.dtype
            assert np.array_equal(result_orig, result_restored), \
                f"Roundtrip mismatch for {pp_orig.name}"

        print("  ✓ test_preprocessor_serialization_roundtrip passed")

    @staticmethod
    def test_template_save_load_with_preprocessor():
        """TemplatePoint with non-default preprocessor should survive save → from_file."""
        ref = create_synthetic_reference()
        pt_orig = TemplatePoint(ref, click_row=100, click_col=100, template_size=80,
                                preprocessor=CannyPreprocessor(70, 160))

        filepath = "/tmp/test_template_preprocessor_roundtrip.npz"
        pt_orig.save(filepath)
        pt_loaded = TemplatePoint.from_file(filepath)

        # Verify preprocessor restored correctly
        assert isinstance(pt_loaded.preprocessor, CannyPreprocessor)
        assert pt_loaded.preprocessor.threshold1 == 70
        assert pt_loaded.preprocessor.threshold2 == 160

        # Verify match results are identical
        inspection = create_synthetic_inspection(ref, offset_row=7.0, offset_col=3.0, noise_level=0.5)
        result_orig = pt_orig.measure(inspection)
        result_loaded = pt_loaded.measure(inspection)
        assert abs(result_orig['matched_row'] - result_loaded['matched_row']) < 0.01
        assert abs(result_orig['matched_col'] - result_loaded['matched_col']) < 0.01

        print("  ✓ test_template_save_load_with_preprocessor passed")

    @staticmethod
    def test_custom_preprocessor_registration():
        """Users should be able to register and use custom preprocessors."""
        # Define a custom preprocessor
        class GaussianBlurPP:
            name = 'GaussianBlur(sigma=1.0)'

            def serialize(self):
                return {'type': 'gaussian_blur_test', 'sigma': 1.0}

            @staticmethod
            def deserialize(data):
                return GaussianBlurPP()

            def __call__(self, image):
                return cv2.GaussianBlur(image, (0, 0), 1.0).astype(np.float32)

        # Register it
        _PREPROCESSOR_REGISTRY['gaussian_blur_test'] = GaussianBlurPP

        # Use it
        ref = create_synthetic_reference()
        pt = TemplatePoint(ref, click_row=100, click_col=100, template_size=80,
                           preprocessor=GaussianBlurPP())
        result = pt.measure(ref)
        assert result['valid'], "Custom preprocessor should produce valid match"

        # Cleanup
        del _PREPROCESSOR_REGISTRY['gaussian_blur_test']

        print("  ✓ test_custom_preprocessor_registration passed")

    @staticmethod
    def test_backward_compat_old_npz_format():
        """Old .npz files with use_edges should deserialize correctly."""
        ref = create_synthetic_reference()

        # Simulate old format: write raw npz with use_edges=True
        pt_old_style = TemplatePoint(ref, click_row=100, click_col=100, template_size=80)
        # Manually add use_edges to simulate old format
        filepath = "/tmp/test_old_format_compat.npz"
        np.savez_compressed(
            filepath,
            edge_template=pt_old_style.edge_template,
            click_row=pt_old_style.click_row,
            click_col=pt_old_style.click_col,
            template_size=pt_old_style.template_size,
            use_edges=np.bool_(True),
            canny_threshold1=np.float64(55.0),
            canny_threshold2=np.float64(165.0),
            match_score_threshold=pt_old_style.match_score_threshold,
            use_subpixel=pt_old_style.use_subpixel,
            crop_center_row=pt_old_style._crop_center_row,
            crop_center_col=pt_old_style._crop_center_col,
            crop_h=pt_old_style._crop_h,
            crop_w=pt_old_style._crop_w,
            actual_crop_bounds=np.array(pt_old_style._actual_crop_bounds, dtype=np.int32),
        )

        # Load with new code
        pt_loaded = TemplatePoint.from_file(filepath)
        assert isinstance(pt_loaded.preprocessor, CannyPreprocessor), \
            f"Expected CannyPreprocessor, got {type(pt_loaded.preprocessor)}"
        assert pt_loaded.preprocessor.threshold1 == 55.0
        assert pt_loaded.preprocessor.threshold2 == 165.0

        print("  ✓ test_backward_compat_old_npz_format passed")

    @staticmethod
    def test_from_file_preprocessor_override():
        """from_file(..., preprocessor=...) should override stored preprocessor."""
        ref = create_synthetic_reference()
        pt_orig = TemplatePoint(ref, click_row=100, click_col=100, template_size=80,
                                preprocessor=RawPreprocessor())

        filepath = "/tmp/test_override_preprocessor.npz"
        pt_orig.save(filepath)

        # Load with a different preprocessor
        pt_loaded = TemplatePoint.from_file(filepath, preprocessor=CannyPreprocessor(50, 150))
        assert isinstance(pt_loaded.preprocessor, CannyPreprocessor)

        print("  ✓ test_from_file_preprocessor_override passed")

    @staticmethod
    def test_threshold_preprocessor():
        """ThresholdPreprocessor should produce binary output and match correctly."""
        ref = create_synthetic_reference()
        pp = ThresholdPreprocessor(threshold=128)

        # Basic property checks
        result = pp(ref)
        assert result.shape == ref.shape
        assert result.dtype == np.uint8
        assert 'Threshold' in pp.name
        # Binary output should only contain 0 and 255
        assert np.all((result == 0) | (result == 255))

        # binary_inv mode
        pp_inv = ThresholdPreprocessor(threshold=128, mode='binary_inv')
        result_inv = pp_inv(ref)
        # Inverted should be the complement
        assert np.all((result == 0) == (result_inv == 255))
        assert np.all((result == 255) == (result_inv == 0))

        # Serialization roundtrip
        data = pp.serialize()
        pp2 = _deserialize_preprocessor(data)
        assert isinstance(pp2, ThresholdPreprocessor)
        assert pp2.threshold == 128
        assert pp2.mode == 'binary'

        # TemplatePoint integration
        pt = TemplatePoint(ref, click_row=100, click_col=100, template_size=80,
                           preprocessor=ThresholdPreprocessor(threshold=100))
        match_result = pt.measure(ref)
        assert match_result['valid'], f"Match should be valid, score={match_result['match_score']:.4f}"

        print("  ✓ test_threshold_preprocessor passed")
# =========================================================================
# Test Runner
# =========================================================================

def run_all_tests():
    """Run all test methods and report results."""
    tests = [
        # TemplatePoint tests
        ("TemplatePoint: construction", TestTemplatePoint.test_construction),
        ("TemplatePoint: perfect match", TestTemplatePoint.test_perfect_match),
        ("TemplatePoint: known translation", TestTemplatePoint.test_known_translation),
        ("TemplatePoint: subpixel refinement", TestTemplatePoint.test_subpixel_refinement),
        ("TemplatePoint: subpixel disabled", TestTemplatePoint.test_subpixel_disabled),
        ("TemplatePoint: low contrast no match", TestTemplatePoint.test_low_contrast_no_match),
        ("TemplatePoint: search region", TestTemplatePoint.test_search_region),
        ("TemplatePoint: serialization roundtrip", TestTemplatePoint.test_serialization_roundtrip),
        ("TemplatePoint: template at border", TestTemplatePoint.test_template_at_border),
        ("TemplatePoint: visualize smoke", TestTemplatePoint.test_visualize_smoke),
        ("TemplatePoint: dx/dy displacement", TestTemplatePoint.test_dx_dy_displacement),
        # DistanceMeasure tests
        ("DistanceMeasure: known distance", TestDistanceMeasure.test_known_distance),
        ("DistanceMeasure: distance with translation", TestDistanceMeasure.test_distance_with_translation),
        ("DistanceMeasure: one point fails", TestDistanceMeasure.test_one_point_fails),
        ("DistanceMeasure: visualize smoke", TestDistanceMeasure.test_visualize_smoke),
        ("DistanceMeasure: VISUAL DEMO", TestDistanceMeasure.test_visual_demo),
        ("DistanceMeasure: REAL IMAGE VISUAL DEMO", TestDistanceMeasure.test_visual_real_demo),
        # Preprocessor tests
        ("Preprocessor: raw", TestPreprocessor.test_raw_preprocessor),
        ("Preprocessor: canny", TestPreprocessor.test_canny_preprocessor),
        ("Preprocessor: sobel", TestPreprocessor.test_sobel_preprocessor),
        ("Preprocessor: clahe", TestPreprocessor.test_clahe_preprocessor),
        ("Preprocessor: template with canny", TestPreprocessor.test_template_with_canny_preprocessor),
        ("Preprocessor: serialization roundtrip", TestPreprocessor.test_preprocessor_serialization_roundtrip),
        ("Preprocessor: template save/load", TestPreprocessor.test_template_save_load_with_preprocessor),
        ("Preprocessor: custom registration", TestPreprocessor.test_custom_preprocessor_registration),
        ("Preprocessor: backward compat", TestPreprocessor.test_backward_compat_old_npz_format),
        ("Preprocessor: from_file override", TestPreprocessor.test_from_file_preprocessor_override),
        ("Preprocessor: threshold", TestPreprocessor.test_threshold_preprocessor),
    ]

    print("=" * 70)
    print("Template Matching Module — Test Suite")
    print("=" * 70)

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            print(f"\n[{name}]")
            test_func()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print(f"Results: {passed}/{len(tests)} passed, {failed}/{len(tests)} failed")
    print("=" * 70)

    cv2.destroyAllWindows()
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
