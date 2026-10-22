"""Tests for the OBB localisation primitives (``measure.obb_source``).

The geometry here decides which way a patch is rotated for every downstream
measurement, so these tests are written against ``measure.rectify``'s
authoritative corner helper rather than against a reimplementation of it.  No ML
dependency is involved: an OBB detector is only ever seen through the
``ObbSource`` protocol.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from measure.obb_source import (
    ObbObservation,
    ObbSource,
    select_observations,
)
from measure.rectify import compute_rotated_box_corners, crop_and_straighten

CY, CX, H, W = 200.0, 250.0, 80.0, 140.0


def corners_at(angle_deg: float) -> np.ndarray:
    """Canonical ``(row, col)`` corners, straight from the shared helper."""
    return compute_rotated_box_corners((CY, CX), (H, W), angle_deg)


class TestFromCornersRecoversGeometry:
    """Round-trip: the box that generated the corners must come back out."""

    @pytest.mark.parametrize("angle", [0.0, 15.0, -30.0, 44.0, 89.0, -89.0])
    def test_angle_size_and_centre_round_trip(self, angle):
        obs = ObbObservation.from_corners(corners_at(angle))
        assert obs.angle_deg == pytest.approx(angle, abs=1e-9)
        assert obs.height == pytest.approx(H, abs=1e-9)
        assert obs.width == pytest.approx(W, abs=1e-9)
        assert obs.row == pytest.approx(CY, abs=1e-9)
        assert obs.col == pytest.approx(CX, abs=1e-9)

    def test_size_is_height_width_for_the_crop_helper(self):
        obs = ObbObservation.from_corners(corners_at(20.0))
        # crop_and_straighten takes (height, width); a swap here would rotate
        # the patch ninety degrees and silently misplace every tool.
        assert obs.size == pytest.approx((H, W))
        assert obs.center == pytest.approx((CY, CX))

    def test_angle_is_normalised_into_half_open_range(self):
        for angle in (0.0, 90.0, 180.0, -90.0, 270.0, -270.0, 45.0):
            obs = ObbObservation.from_corners(corners_at(angle))
            assert -90.0 <= obs.angle_deg < 90.0

    @pytest.mark.parametrize("transform", ["reverse", "roll2", "reverse_roll1"])
    def test_vertex_order_does_not_change_the_result(self, transform):
        """A box has no canonical starting corner; all windings agree."""
        base = corners_at(15.0)
        variants = {
            "reverse": base[::-1],
            "roll2": np.roll(base, 2, axis=0),
            "reverse_roll1": np.roll(base[::-1], 1, axis=0),
        }
        obs = ObbObservation.from_corners(variants[transform])
        assert obs.angle_deg == pytest.approx(15.0, abs=1e-9)
        assert (obs.height, obs.width) == pytest.approx((H, W), abs=1e-9)

    def test_corners_agree_with_the_reported_angle(self):
        """Emitting canonical corners keeps `corners` and `angle_deg` consistent.

        A box given by its short edge is the same box as one given by its long
        edge; both must serialise identically or equal boxes compare unequal.
        """
        obs = ObbObservation.from_corners(corners_at(30.0))
        assert obs.corners is not None
        rederived = ObbObservation.from_corners(obs.corners)
        assert rederived.angle_deg == pytest.approx(obs.angle_deg, abs=1e-9)
        assert rederived.size == pytest.approx(obs.size, abs=1e-9)


class TestFromCornersRejectsBadInput:
    def test_wrong_shape(self):
        with pytest.raises(ValueError, match=r"\(4, 2\)"):
            ObbObservation.from_corners(np.zeros((3, 2)))

    def test_non_rectangle(self):
        with pytest.raises(ValueError, match="do not form a rectangle"):
            ObbObservation.from_corners(
                np.array([[0, 0], [10, 0], [10, 5], [0, 10]], dtype=float))

    def test_non_perpendicular_edges(self):
        # A parallelogram with equal-length sides but a 60-degree corner.  The
        # closure check passes, so only the perpendicularity test can catch it.
        pts = np.array([[0, 0], [10, 0], [15, 8.660254], [5, 8.660254]], float)
        with pytest.raises(ValueError, match="instead of 90"):
            ObbObservation.from_corners(pts)

    def test_degenerate_box(self):
        with pytest.raises(ValueError, match="degenerate"):
            ObbObservation.from_corners(np.zeros((4, 2)))


class TestFromXywhr:
    """The other common detector output form, in radians."""

    def test_matches_from_corners(self):
        from_xywhr = ObbObservation.from_xywhr(CY, CX, W, H, math.radians(20.0))
        from_corners = ObbObservation.from_corners(corners_at(20.0))
        assert from_xywhr.angle_deg == pytest.approx(from_corners.angle_deg, abs=1e-9)
        assert from_xywhr.size == pytest.approx(from_corners.size, abs=1e-9)
        assert (from_xywhr.row, from_xywhr.col) == pytest.approx((CY, CX))

    def test_swapping_width_and_height_keeps_the_same_box(self):
        a = ObbObservation.from_xywhr(CY, CX, W, H, math.radians(20.0))
        b = ObbObservation.from_xywhr(CY, CX, H, W, math.radians(20.0))
        assert a.size == pytest.approx(b.size)
        # The canonical angle refers to the long axis, so the second form must
        # have been rotated a quarter turn to describe the same rectangle.
        assert abs(a.angle_deg - b.angle_deg) == pytest.approx(90.0, abs=1e-9)

    def test_radians_not_degrees(self):
        """A degree value passed as radians must not accidentally look right."""
        wrong = ObbObservation.from_xywhr(CY, CX, W, H, 20.0)   # 20 rad, not 20 deg
        right = ObbObservation.from_xywhr(CY, CX, W, H, math.radians(20.0))
        assert wrong.angle_deg != pytest.approx(right.angle_deg, abs=1.0)


class TestRectificationAlignment:
    """The observation must drive crop_and_straighten correctly.

    This is the property that matters operationally: a wrong sign or a height/
    width swap would still "work" while measuring the wrong part of the image.
    """

    @pytest.mark.parametrize("angle", [0.0, 30.0, -45.0, 89.0])
    def test_patch_has_the_box_dimensions(self, angle):
        obs = ObbObservation.from_corners(corners_at(angle))
        patch, _ = crop_and_straighten(
            np.zeros((400, 500), np.uint8), obs.center, obs.size, obs.angle_deg)
        # int() truncation in warpAffine means a value like 79.9999 yields 79,
        # so allow one pixel of slack rather than demanding exact equality.
        assert abs(patch.shape[0] - obs.height) <= 1
        assert abs(patch.shape[1] - obs.width) <= 1

    @pytest.mark.parametrize("angle", [0.0, 25.0, -40.0])
    def test_patch_centre_maps_back_onto_the_box_centre(self, angle):
        """The transform pair must agree, whichever way the matrix is named.

        ``crop_and_straighten`` documents M_inv as patch->original while
        ``map_point_via_affine`` takes it as original->patch and inverts
        internally; the engine relies on the latter reading.  Testing through the
        public helper exercises the direction that actually ships, so a naming
        slip cannot hide a real coordinate error.
        """
        from measure.rectify import map_point_via_affine

        obs = ObbObservation.from_corners(corners_at(angle))
        patch, M_inv = crop_and_straighten(
            np.zeros((400, 500), np.uint8), obs.center, obs.size, obs.angle_deg)
        h, w = patch.shape[:2]
        back = map_point_via_affine((h / 2.0, w / 2.0), M_inv[:2, :])
        assert back[0] == pytest.approx(obs.row, abs=0.51)
        assert back[1] == pytest.approx(obs.col, abs=0.51)

    @pytest.mark.parametrize("angle", [0.0, 25.0, -40.0])
    def test_tools_defined_at_the_centre_stay_at_the_centre(self, angle):
        """Round trip: box centre -> patch centre -> back to the box centre.

        A sign error in the angle would rotate the patch but leave its centre
        fixed, so this alone cannot prove orientation; it does prove the box was
        cropped where it was found, which is the failure that misplaces every
        measurement at once.
        """
        from measure.rectify import map_point_via_affine

        obs = ObbObservation.from_corners(corners_at(angle))
        patch, M_inv = crop_and_straighten(
            np.zeros((400, 500), np.uint8), obs.center, obs.size, obs.angle_deg)
        h, w = patch.shape[:2]
        for patch_pt in [(h / 2.0, w / 2.0), (5.0, 5.0), (h - 6.0, w - 6.0)]:
            orig = map_point_via_affine(patch_pt, M_inv[:2, :])
            # Those image points must lie inside the detected box.
            delta = np.array(orig) - np.array(obs.center)
            assert np.max(np.abs(delta)) <= max(obs.height, obs.width)


class TestSelectionPolicy:
    def make(self, class_id, score):
        return ObbObservation.from_corners(
            corners_at(0.0), class_id=class_id, score=score)

    def test_filters_by_class(self):
        obs = [self.make(0, 0.9), self.make(1, 0.95), self.make(0, 0.8)]
        kept = select_observations(obs, class_id=0)
        assert [o.score for o in kept] == [0.9, 0.8]

    def test_filters_by_score(self):
        obs = [self.make(0, 0.9), self.make(0, 0.4)]
        assert len(select_observations(obs, min_score=0.5)) == 1

    def test_orders_by_descending_score(self):
        obs = [self.make(0, 0.3), self.make(0, 0.9), self.make(0, 0.6)]
        assert [o.score for o in select_observations(obs)] == [0.9, 0.6, 0.3]

    def test_unscored_observations_sort_last_but_survive(self):
        """A detector reporting no confidence must not be silently dropped."""
        scored = self.make(0, 0.5)
        unscored = ObbObservation.from_corners(corners_at(0.0), class_id=0)
        kept = select_observations([unscored, scored])
        assert kept[0] is scored and kept[1] is unscored
        assert len(select_observations([unscored], min_score=0.0)) == 0

    def test_max_targets_caps_after_ordering(self):
        obs = [self.make(0, 0.3), self.make(0, 0.9)]
        kept = select_observations(obs, max_targets=1)
        assert len(kept) == 1 and kept[0].score == pytest.approx(0.9)

    def test_zero_max_targets_means_unlimited(self):
        obs = [self.make(0, 0.3), self.make(0, 0.9)]
        assert len(select_observations(obs, max_targets=0)) == 2


class TestSerialization:
    def test_to_dict_is_json_safe(self):
        obs = ObbObservation.from_corners(
            corners_at(10.0), class_id=2, score=0.75, class_name="cap")
        payload = json.dumps(obs.to_dict(), allow_nan=False)
        restored = json.loads(payload)
        assert restored["class_name"] == "cap"
        assert restored["score"] == pytest.approx(0.75)
        assert len(restored["corners"]) == 4

    def test_none_score_survives_as_null(self):
        obs = ObbObservation.from_corners(corners_at(0.0))
        assert json.loads(json.dumps(obs.to_dict()))["score"] is None


class TestProtocol:
    def test_engine_only_needs_detect_and_serialize(self):
        """"The engine must be able to drive any detector without ML imports."""

        class FakeSource:
            name = "fake"
            requires_roi = False

            def detect(self, image):
                return [ObbObservation.from_corners(corners_at(0.0))]

            def serialize(self):
                return {"name": self.name}

        assert isinstance(FakeSource(), ObbSource)

    def test_requires_roi_flag_is_part_of_the_contract(self):
        """Stage-1 configuration differs per source, so it must be declarable."""
        class NeedsRoi:
            name = "roi"
            requires_roi = True

            def detect(self, image):
                return []

            def serialize(self):
                return {}

        assert NeedsRoi().requires_roi is True