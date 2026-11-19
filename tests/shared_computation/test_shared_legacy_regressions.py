"""Existing public workflows expose defects before the new entry points are implemented."""

from typing import Any, Dict

import numpy as np
import pytest

from measure.alignment import NoneAlignment
from measure.measure_workflow import (
    GeometricResult,
    LineResult,
    MeasureObject,
    MeasurementWorkflow,
    PointLineDistanceObject,
    PointResult,
    SimilarityTransform,
    TwoPointsDistanceObject,
    TwoPointsLineObject,
)
from measure.multi_target_workflow import MultiTargetWorkflow


class ConstantGeometry(MeasureObject):
    """Provide a known observation through the public measurement extension contract."""

    def __init__(self, geometry: GeometricResult):
        super().__init__(geometry.label)
        self.geometry = geometry

    def measure(self, image: np.ndarray) -> GeometricResult:
        self.result = self.geometry
        return self.result

    def result_type(self) -> str:
        return self.geometry.type

    def to_dict(self) -> Dict[str, Any]:
        return {"geometry": self.geometry}

    @classmethod
    def from_dict(cls, label: str, data: Dict[str, Any]) -> "ConstantGeometry":
        return cls(data["geometry"])


class FramePoint(MeasureObject):
    """Read a frame marker, with zero representing a failed acquisition."""

    def measure(self, image: np.ndarray) -> PointResult:
        marker = int(image[0, 0])
        if marker == 0:
            raise RuntimeError("Current frame has no point observation")
        self.result = PointResult(label=self.label, row=float(marker), col=5.0)
        return self.result

    def result_type(self) -> str:
        return "point"

    def to_dict(self) -> Dict[str, Any]:
        return {}

    @classmethod
    def from_dict(cls, label: str, data: Dict[str, Any]) -> "FramePoint":
        return cls(label)


def test_existing_workflow_preserves_three_four_five_point_distance():
    workflow = MeasurementWorkflow()
    workflow.add(ConstantGeometry(PointResult(label="a", row=0.0, col=0.0)))
    workflow.add(ConstantGeometry(PointResult(label="b", row=4.0, col=3.0)))
    workflow.add(TwoPointsDistanceObject("distance", "a", "b"))
    result = workflow.measure(np.zeros((2, 2), np.uint8))
    assert result["distance"].valid and result["distance"].value == pytest.approx(5.0)


def test_d01_existing_workflow_uses_xy_coefficients_for_point_line_distance():
    workflow = MeasurementWorkflow()
    workflow.add(ConstantGeometry(PointResult(label="point", row=13.0, col=5.0)))
    workflow.add(ConstantGeometry(LineResult(
        label="line", a=0.0, b=1.0, c=-10.0,
        start_row=10.0, start_col=0.0, end_row=10.0, end_col=20.0,
    )))
    workflow.add(PointLineDistanceObject("distance", "point", "line"))
    result = workflow.measure(np.zeros((2, 2), np.uint8))
    assert result["distance"].valid and result["distance"].value == pytest.approx(3.0)


def test_d01_existing_two_point_line_satisfies_its_endpoints_in_xy_order():
    workflow = MeasurementWorkflow()
    workflow.add(ConstantGeometry(PointResult(label="a", row=10.0, col=0.0)))
    workflow.add(ConstantGeometry(PointResult(label="b", row=10.0, col=20.0)))
    workflow.add(TwoPointsLineObject("line", "a", "b"))
    line = workflow.measure(np.zeros((2, 2), np.uint8))["line"]
    assert line.valid
    assert line.a * line.start_col + line.b * line.start_row + line.c == pytest.approx(0.0)
    assert line.a * line.end_col + line.b * line.end_row + line.c == pytest.approx(0.0)


def test_d01_existing_line_angle_agrees_with_horizontal_xy_endpoints():
    line = LineResult(
        a=0.0, b=1.0, c=-10.0,
        start_row=10.0, start_col=0.0, end_row=10.0, end_col=20.0,
    )
    assert line.angle_rad == pytest.approx(0.0)
    assert line.angle_deg == pytest.approx(0.0)


def test_d04_existing_similarity_rejects_repeated_source_points():
    result = SimilarityTransform.from_correspondences(
        [(1.0, 2.0), (1.0, 2.0)], [(3.0, 4.0), (5.0, 6.0)],
    )
    assert not result.is_valid


@pytest.mark.parametrize("check", ["point", "dependent", "snapshot"])
def test_d05_failed_frame_never_reuses_successful_frame_results_and_can_recover(check):
    workflow = MeasurementWorkflow()
    workflow.add(FramePoint("moving"))
    workflow.add(ConstantGeometry(PointResult(label="fixed", row=10.0, col=5.0)))
    workflow.add(TwoPointsDistanceObject("distance", "moving", "fixed"))

    first = workflow.measure(np.full((2, 2), 13, np.uint8))
    failed = workflow.measure(np.zeros((2, 2), np.uint8))
    failed_public_point = workflow.get_result("moving")
    failed_public_distance = workflow.get_result("distance")
    failed_is_valid = workflow.is_valid
    recovered = workflow.measure(np.full((2, 2), 17, np.uint8))

    assert first["distance"].value == pytest.approx(3.0)
    assert failed["fixed"].valid
    assert recovered["moving"].valid and recovered["distance"].value == pytest.approx(7.0)
    assert workflow.is_valid
    if check == "point":
        assert not failed["moving"].valid
        assert failed_public_point is not first["moving"]
    elif check == "dependent":
        assert not failed["distance"].valid and not failed_public_distance.valid
    else:
        assert not failed_is_valid


@pytest.mark.parametrize("check", ["independent", "typed_results"])
def test_d06_bad_probe_is_rejected_or_isolated_from_independent_tools(check):
    image = np.zeros((80, 100), np.uint8)
    image[:, 50:] = 255
    workflow = MultiTargetWorkflow()
    workflow.alignment = NoneAlignment()
    workflow.teach_without_search(image, center=(40.0, 50.0), size=(80.0, 100.0))
    probe = dict(row=40.0, col=50.0, angle=0.0, length1=20.0, length2=3.0,
                 sigma=1.0, threshold=30.0)

    # The legacy API accepts this one-column probe, then np.gradient raises at runtime.
    # Rejecting it before registration is also acceptable after validation is centralized.
    try:
        workflow.add_measurement("EdgePoint", "bad", **dict(probe, length1=0.5))
    except ValueError:
        assert all(item["label"] != "bad" for item in workflow.measurement_defs)
    workflow.add_measurement("EdgePoint", "independent", **probe)
    failed = workflow.measure(image)[0]
    workflow.remove_measurement("bad")
    recovered = workflow.measure(image)[0]

    assert recovered.measurements["independent"]["valid"]
    if check == "independent":
        assert "independent" in failed.measurements
        assert failed.measurements["independent"]["valid"]
    else:
        assert all(isinstance(value, dict) and "valid" in value and "type" in value
                   for value in failed.measurements.values())
