"""End-to-end guards for ``TargetResult.rotation_deg`` being truly absolute.

Stage 1 reports rotation *relative to the taught pose*, and the two locator
families encode that quantity with opposite signs.  A single ``taught - reported``
expression therefore could not serve both, and for template matching it silently
reported the negated deviation: exact at the taught pose (where everything used
to be verified) and wrong by *twice* the deviation once the part rotated.

These tests rotate a known part by a known amount and compare against ground
truth, so they fail on the old behaviour instead of merely pinning whatever the
code happens to do.  A tolerance of a couple of degrees is used because the
template search quantises angles (``angle_step``); the bug produced 10 degrees of
error for a 5 degree rotation, an order of magnitude above that noise floor.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from measure.alignment import (
    FixedRoiAlignment,
    NoneAlignment,
    ObbAlignment,
)
from measure.multi_target_workflow import (
    MultiTargetWorkflow,
    _compute_target_box_corners,
)
from measure.obb_source import ObbObservation
from measure.rectify import compute_rotated_box_corners

IMG_H, IMG_W = 300, 400
CENTER = (150.0, 200.0)
BOX = (120.0, 200.0)          # (height, width): the bar plus a dark margin
BAR = (120, 120, 280, 180)    # row_start, col_start, row_end, col_end
TOL_DEG = 2.5


def base_image() -> np.ndarray:
    """An elongated bright bar with an off-centre marker (orientation is unambiguous)."""
    img = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    img[BAR[0]:BAR[2], BAR[1]:BAR[3]] = 200
    # Marker breaks the bar's 180-degree symmetry so a mirrored match is rejected.
    img[135:165, 135:165] = 90
    return img


def rotated(angle_deg: float) -> np.ndarray:
    """The bar rotated ``angle_deg`` degrees counter-clockwise on screen."""
    matrix = np.array(
        cv2.getRotationMatrix2D((IMG_W / 2.0, IMG_H / 2.0), angle_deg, 1.0),
        dtype=np.float64,
    )
    return cv2.warpAffine(base_image(), matrix, (IMG_W, IMG_H))


def tau(teach_angle_deg: float) -> MultiTargetWorkflow:
    """A template model taught on the bar tilted ``teach_angle_deg``, with no tools.

    Localisation-only is the configuration most exposed to this bug: the pose is
    the entire deliverable, so a mirrored angle is the whole result.
    """
    workflow = MultiTargetWorkflow()
    workflow.teach_template(
        rotated(teach_angle_deg),
        center=CENTER,
        size=BOX,
        angle_deg=teach_angle_deg,
        match_score_threshold=0.5,
        angle_range=(-40.0, 40.0),
        angle_step=0.25,
        max_matches=1,
    )
    return workflow


class TestTemplateReportsAbsoluteOrientation:
    @pytest.mark.parametrize("teach", [0.0, 15.0, -20.0])
    @pytest.mark.parametrize("offset", [-12.0, -5.0, 0.0, 5.0, 12.0])
    def test_reported_angle_matches_ground_truth(self, teach, offset):
        truth = teach + offset
        target = tau(teach).measure(rotated(truth))[0]
        assert target.rotation_deg == pytest.approx(truth, abs=TOL_DEG), (
            f"taught at {teach}, part actually at {truth}: "
            f"reported {target.rotation_deg:.2f} "
            f"(old behaviour gave {teach - (truth - teach):.2f})"
        )

    def test_the_old_double_deviation_signature_is_gone(self):
        """A 5-degree rotation used to be reported as 10 degrees the other way."""
        target = tau(0.0).measure(rotated(5.0))[0]
        assert abs(target.rotation_deg - 5.0) < TOL_DEG
        assert target.rotation_deg > 0, "sign must follow the actual rotation"

    def test_drawn_box_follows_the_part(self):
        """The visualisation uses ``rotation_deg`` directly, so it must agree too."""
        target = tau(0.0).measure(rotated(10.0))[0]
        drawn = _compute_target_box_corners(target, BOX)
        expected = compute_rotated_box_corners(
            (target.center_row, target.center_col), BOX, 10.0)
        assert np.allclose(drawn, expected, atol=4.0)

    def test_taught_pose_is_exact(self):
        """The case every previous verification used — it must stay exact."""
        target = tau(15.0).measure(rotated(15.0))[0]
        assert target.rotation_deg == pytest.approx(15.0, abs=0.5)


class TestObbConventionSurvives:
    """``detect()`` negates the delta on purpose; the inverse must undo it."""

    @pytest.mark.parametrize("instance_angle", [-25.0, -5.0, 0.0, 10.0, 24.0])
    def test_instance_angle_is_recovered(self, instance_angle):
        anchor_angle = 10.0
        strategy = ObbAlignment()
        strategy.teach(
            np.zeros((IMG_H, IMG_W), dtype=np.uint8),
            source=_ScriptedSource([
                ObbObservation.from_corners(
                    compute_rotated_box_corners(CENTER, BOX, instance_angle),
                    class_id=0, score=0.9)]),
            anchor_observation=ObbObservation.from_corners(
                compute_rotated_box_corners(CENTER, BOX, anchor_angle),
                class_id=0, score=0.9),
        )
        match = strategy.detect(np.zeros((IMG_H, IMG_W), dtype=np.uint8))[0]
        assert strategy.absolute_rotation(match["best_rotation_deg"]) == pytest.approx(
            instance_angle)

    def test_base_and_obb_conversions_disagree_by_design(self):
        """Guard against 'simplifying' both into one expression again."""
        from measure.alignment import SingleBoxAlignment

        template = SingleBoxAlignment()
        template._box_angle_deg = 10.0
        detector = ObbAlignment()
        detector._box_angle_deg = 10.0
        assert template.absolute_rotation(5.0) == pytest.approx(15.0)
        assert detector.absolute_rotation(5.0) == pytest.approx(5.0)


class TestSearchFreeModes:
    """No search means no rotation: the taught angle is the answer, verbatim."""

    @pytest.mark.parametrize("cls", [NoneAlignment, FixedRoiAlignment])
    def test_reports_the_taught_angle(self, cls):
        workflow = MultiTargetWorkflow()
        workflow.alignment = cls()
        size = (IMG_H, IMG_W) if cls is NoneAlignment else BOX
        workflow.teach_without_search(base_image(), CENTER, size, 7.5)
        target = workflow.measure(base_image())[0]
        assert target.rotation_deg == pytest.approx(7.5)


class _ScriptedSource:
    """Detector stub: fixed output, no ML runtime."""

    name = "scripted"
    requires_roi = False

    def __init__(self, observations):
        self._observations = list(observations)

    def detect(self, image):
        return list(self._observations)

    def serialize(self):
        return {"name": self.name, "scripted": True}
