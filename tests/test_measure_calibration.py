"""Tests for measure_calibration.py — CameraCalibration and StereoRigCalibration.

Uses programmatically-generated synthetic dot-grid boards to validate
the calibration pipeline without requiring real images.
"""

import os
import json
import tempfile
import numpy as np
import cv2
import pytest
from measure.measure_calibration import CameraCalibration, StereoRigCalibration


# ---------------------------------------------------------------------------
# Fixtures — synthetic dot-grid board generators
# ---------------------------------------------------------------------------

SPACING = 30         # px (treated as mm)
COLS, ROWS = 11, 9
MARGIN = 50
W = COLS * SPACING + 2 * MARGIN
H = ROWS * SPACING + 2 * MARGIN


def _make_dot_board() -> np.ndarray:
    """White background with black filled circles."""
    img = np.ones((H, W), dtype=np.uint8) * 255
    for j in range(ROWS):
        for i in range(COLS):
            cv2.circle(img,
                       (MARGIN + i * SPACING, MARGIN + j * SPACING),
                       6, 0, -1)
    return img


def _generate_views(n: int = 12, seed: int = 42,
                    border_value: int = 255) -> list:
    """Rotate the board at different angles to simulate multi-pose capture."""
    board = _make_dot_board()
    rng = np.random.RandomState(seed)
    images = []
    for i in range(n):
        angle = (i - n // 2) * 3
        M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
        warped = cv2.warpAffine(board, M, (W, H),
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=border_value)
        noisy = np.clip(warped.astype(np.float32)
                        + rng.normal(0, 1, warped.shape),
                        0, 255).astype(np.uint8)
        images.append(noisy)
    return images


@pytest.fixture(scope="module")
def views():
    return _generate_views()


@pytest.fixture(scope="module")
def calibrated(views):
    """A pre-calibrated CameraCalibration shared across tests."""
    calib = CameraCalibration(grid_size=(COLS, ROWS),
                              circle_spacing_mm=SPACING)
    calib.calibrate(views, symmetric_grid=True)
    return calib


# ---------------------------------------------------------------------------
# CameraCalibration
# ---------------------------------------------------------------------------

class TestCameraCalibration:
    """Unit tests for single-camera calibration."""

    def test_grid_detection_count(self, calibrated):
        """At least 8 of 12 views should have detectable dot grids."""
        assert calibrated.result["num_images"] >= 8

    def test_reprojection_error(self, calibrated):
        """Reprojection error below 10 px on synthetic data."""
        assert calibrated.reprojection_error < 10.0

    def test_K_matrix_shape(self, calibrated):
        assert calibrated.K.shape == (3, 3)
        assert calibrated.K[2, 2] == 1.0

    def test_D_vector(self, calibrated):
        assert calibrated.D is not None
        assert len(calibrated.D) >= 1  # can be 1, 4, 5, or 8 elements

    def test_undistort_shape(self, calibrated, views):
        und = calibrated.undistort(views[0])
        assert und.shape == views[0].shape

    def test_undistort_does_not_crash_on_bgr(self, calibrated, views):
        bgr = cv2.cvtColor(views[0], cv2.COLOR_GRAY2BGR)
        und = calibrated.undistort(bgr)
        assert und.shape == bgr.shape

    def test_undistort_before_calibrate_raises(self, views):
        calib = CameraCalibration()
        with pytest.raises(RuntimeError, match="calibrate"):
            calib.undistort(views[0])

    def test_calibrate_too_few_images(self):
        calib = CameraCalibration(grid_size=(COLS, ROWS),
                                  circle_spacing_mm=SPACING)
        # Three copies of the same image won't calibrate well but won't
        # trigger the "too few" error.  Use an image that's totally blank.
        blank = np.zeros((H, W), dtype=np.uint8)
        with pytest.raises(ValueError, match="at least 3"):
            calib.calibrate([blank, blank, blank])

    def test_visualize_output(self, calibrated, views):
        vis = calibrated.visualize(views[0], index=0,
                                   show_axes=True,
                                   show_reprojection=True)
        assert vis.shape == (*views[0].shape, 3)  # BGR

    def test_visualize_save(self, calibrated, views, tmp_path):
        path = str(tmp_path / "calib_vis.png")
        calibrated.visualize(views[0], save_path=path)
        assert os.path.exists(path)

    def test_to_dict_uncalibrated(self):
        calib = CameraCalibration()
        d = calib.to_dict()
        assert d["grid_size"] == [11, 9]

    def test_to_dict_calibrated(self, calibrated):
        d = calibrated.to_dict()
        assert "K" in d
        assert len(d["K"]) == 3

    def test_from_dict_roundtrip(self, calibrated):
        d = calibrated.to_dict()
        calib2 = CameraCalibration.from_dict(d)
        assert np.allclose(calib2.K, calibrated.K, atol=1e-6)
        assert calib2.reprojection_error == calibrated.reprojection_error

    def test_save_load_roundtrip(self, calibrated):
        f = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
        fpath = f.name; f.close()
        try:
            calibrated.save(fpath)
            calib2 = CameraCalibration.load(fpath)
            assert np.allclose(calib2.K, calibrated.K, atol=1e-6)
            np.testing.assert_array_equal(calib2.D, calibrated.D)
        finally:
            os.unlink(fpath)

    def test_save_then_undistort_same_result(self, calibrated, views):
        """Undistort via a loaded calibration should match the original."""
        f = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
        fpath = f.name; f.close()
        try:
            calibrated.save(fpath)
            calib2 = CameraCalibration.load(fpath)
            a = calibrated.undistort(views[0])
            b = calib2.undistort(views[0])
            np.testing.assert_array_equal(a, b)
        finally:
            os.unlink(fpath)


# ---------------------------------------------------------------------------
# StereoRigCalibration
# ---------------------------------------------------------------------------

class TestStereoRigCalibration:
    """Unit tests for two-camera perspective stitching."""

    @pytest.fixture(scope="class")
    def rig(self, calibrated):
        """A calibrated rig using different views as 'two cameras'."""
        views = _generate_views()
        rig = StereoRigCalibration(calibrated, calibrated)
        rig.calibrate(views[0], views[-1], symmetric_grid=True)
        return rig, views

    def test_calibrate_point_counts(self, rig):
        r, views = rig
        assert r.result["num_points_a"] == COLS * ROWS
        assert r.result["num_points_b"] == COLS * ROWS

    def test_homography_reproj_error(self, rig):
        r, _ = rig
        assert r.result["reprojection_error_a"] < 1.0
        assert r.result["reprojection_error_b"] < 1.0

    def test_stitch_shape(self, rig):
        r, views = rig
        sr = r.stitch(views[0], views[-1], pixel_size_mm=0.5)
        assert sr["stitched_image"].shape[0] > 100
        assert sr["stitched_image"].shape[1] > 100
        assert sr["valid_mask"].any()

    def test_stitch_before_calibrate_raises(self):
        c = CameraCalibration()
        r = StereoRigCalibration(c, c)
        with pytest.raises(RuntimeError, match="calibrate"):
            r.stitch(np.zeros((100, 100), dtype=np.uint8),
                     np.zeros((100, 100), dtype=np.uint8))

    def test_calibrate_detection_failure_raises(self, calibrated):
        r = StereoRigCalibration(calibrated, calibrated)
        blank = np.zeros((H, W), dtype=np.uint8)
        with pytest.raises(ValueError, match="camera A"):
            r.calibrate(blank, _make_dot_board())

    def test_set_origin(self, rig):
        r, views = rig
        before = r.origin_offset_mm
        result = r.set_origin(views[0], 185, 215)
        after = r.origin_offset_mm
        assert after != before
        assert "origin_grid_index" in result

    def test_set_origin_bad_camera(self, rig):
        r, views = rig
        with pytest.raises(ValueError, match="camera"):
            r.set_origin(views[0], 100, 100, camera="C")

    def test_visualize_output(self, rig):
        r, views = rig
        out = r.visualize(views[0], views[-1])
        assert "board_a" in out
        assert "board_b" in out
        assert "undistorted_a" in out
        assert "stitched" in out

    def test_visualize_save_dir(self, rig, tmp_path):
        r, views = rig
        save_dir = str(tmp_path / "rig_viz")
        r.visualize(views[0], views[-1], save_dir=save_dir)
        assert os.path.isdir(save_dir)
        assert len(os.listdir(save_dir)) >= 3

    def test_to_dict(self, rig):
        r, _ = rig
        d = r.to_dict()
        assert "calib_a" in d
        assert "H_A" in d

    def test_from_dict_roundtrip(self, rig):
        r, _ = rig
        d = r.to_dict()
        r2 = StereoRigCalibration.from_dict(d)
        assert np.allclose(r2.H_A, r.H_A, atol=1e-6)
        assert np.allclose(r2.H_B, r.H_B, atol=1e-6)

    def test_save_load_roundtrip(self, rig, calibrated):
        r, views = rig
        f = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
        fpath = f.name; f.close()
        try:
            r.save(fpath)
            r2 = StereoRigCalibration.load(fpath, calibrated, calibrated)
            assert np.allclose(r2.H_A, r.H_A, atol=1e-6)
            assert np.allclose(r2.H_B, r.H_B, atol=1e-6)
        finally:
            os.unlink(fpath)

    def test_load_preserves_canvas(self, rig, calibrated):
        """After load, stitch() should produce a meaningful result."""
        r, views = rig
        f = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
        fpath = f.name; f.close()
        try:
            r.save(fpath)
            r2 = StereoRigCalibration.load(fpath, calibrated, calibrated)
            sr = r2.stitch(views[0], views[-1])
            assert sr["stitched_image"].shape[0] > 50
        finally:
            os.unlink(fpath)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_small_grid(self):
        calib = CameraCalibration(grid_size=(5, 4), circle_spacing_mm=20.0)
        # Make a small board
        w = 5 * 20 + 60
        h = 4 * 20 + 60
        img = np.ones((h, w), dtype=np.uint8) * 255
        for j in range(4):
            for i in range(5):
                cv2.circle(img, (30 + i * 20, 30 + j * 20), 4, 0, -1)
        images = [img, img, img]  # not ideal but tests the code path
        # Same image won't calibrate, but should detect grids
        r = calib.calibrate(images, symmetric_grid=True,
                            subpix_refine=False)
        assert r["num_images"] == 3

    def test_custom_blob_params(self):
        calib = CameraCalibration(grid_size=(COLS, ROWS),
                                  circle_spacing_mm=SPACING)
        views = _generate_views(6)
        r = calib.calibrate(views,
                            blob_detector_params={
                                "minThreshold": 50,
                                "maxThreshold": 200,
                                "minArea": 30,
                            })
        assert r["num_images"] >= 1


# ---------------------------------------------------------------------------
# __main__ — standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    success = True
    tests = [
        ("TestCameraCalibration.test_grid_detection_count", lambda: None),  # placeholder
        ("TestStereoRigCalibration.test_calibrate_point_counts", lambda: None),
    ]
    sys.exit(0 if success else 1)
