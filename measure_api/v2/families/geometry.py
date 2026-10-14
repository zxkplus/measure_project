"""Frame-safe atomic geometry composition and supplied-point fits."""

from typing import List, Optional

import numpy as np
from pydantic import Field

from measure.geometric_fits import fit_circle_points, fit_line_points
from measure_api.v2.atomic_types import (
    Circle, FitDiagnostics, Line, Observation, PositionedPoint, Scalar,
)
from measure_api.v2.errors import ServiceError
from measure_api.v2.models import CalculationRequest, Point
from measure_api.v2.registry import CalculationContext, Operation


class PointFitRequest(CalculationRequest):
    points: List[Point] = Field(max_length=100000)
    frame: str = Field(default='source', min_length=1)
    unit: str = Field(default='px', min_length=1)


class LineFitResult(Observation):
    line: Line
    samples: List[Point] = Field(default_factory=list)
    diagnostics: FitDiagnostics


class CircleFitResult(Observation):
    circle: Circle
    samples: List[Point] = Field(default_factory=list)
    diagnostics: FitDiagnostics
    sampled_diameter_spread: Optional[float] = None


def fit_diagnostics(data: dict, requested: int = 0, successful: int = 0) -> FitDiagnostics:
    """Report directly computed residual metrics, never an inferred RMS.

    Args:
        data: Numeric-core fit result with measured residual and probe-quality fields.
        requested: Number of probes requested before filtering unsuccessful measurements.
        successful: Number of probes that produced a selected usable observation.

    Returns:
        Measured residual quality, fit status and requested/successful probe counts.
    """
    return FitDiagnostics(
        sample_count=data['num_points'], requested_probe_count=requested,
        successful_probe_count=successful, residuals=data['residuals'],
        mean_absolute_error=data['mean_error'], max_absolute_error=data['max_error'],
        rms_error=data['rms_error'],
    )


def fit_line(request: PointFitRequest, context: CalculationContext) -> LineFitResult:
    """Fit distinct supplied xy points; degeneracy is an invalid observation.

    Args:
        request: Typed ordered xy samples with their coordinate frame and length unit.
        context: Isolated operation identity, resource limits and verified shared-file
            access.

    Returns:
        Observed normalized line, selected samples and independently computed residual
        quality.
    """
    try:
        data = fit_line_points([(p.x, p.y) for p in request.points])
    except ValueError as exc:
        reason = str(exc)
        return LineFitResult(
            valid=False, reason=reason, frame=request.frame, unit=request.unit,
            line=Line(valid=False, reason=reason, frame=request.frame, unit=request.unit),
            samples=request.points, diagnostics=FitDiagnostics(sample_count=len(request.points)),
        )
    a, b, c = data['params']
    return LineFitResult(
        valid=True, frame=request.frame, unit=request.unit, samples=request.points,
        line=Line(valid=True, a=a, b=b, c=c, frame=request.frame, unit=request.unit,
                  start=Point(x=data['start'][1], y=data['start'][0]),
                  end=Point(x=data['end'][1], y=data['end'][0])),
        diagnostics=fit_diagnostics(data),
    )


def fit_circle(request: PointFitRequest, context: CalculationContext) -> CircleFitResult:
    """Fit a circle, rejecting collinear points and nonpositive radii.

    Args:
        request: Typed ordered xy samples with their coordinate frame and length unit.
        context: Isolated operation identity, resource limits and verified shared-file
            access.

    Returns:
        Observed center/radius, selected samples and independently computed fit quality.
    """
    try:
        data = fit_circle_points([(p.x, p.y) for p in request.points])
    except ValueError as exc:
        reason = str(exc)
        return CircleFitResult(
            valid=False, reason=reason, frame=request.frame, unit=request.unit,
            circle=Circle(valid=False, reason=reason, frame=request.frame, unit=request.unit),
            samples=request.points, diagnostics=FitDiagnostics(sample_count=len(request.points)),
        )
    return CircleFitResult(
        valid=True, frame=request.frame, unit=request.unit, samples=request.points,
        circle=Circle(valid=True, center=Point(x=data['center'][1], y=data['center'][0]),
                      radius=data['radius'], frame=request.frame, unit=request.unit),
        sampled_diameter_spread=data['ellipticity'], diagnostics=fit_diagnostics(data),
    )


def compatible(*values: Observation) -> bool:
    """Require matching frame and units even for supplied invalid observations.

    Args:
        values: Observations whose coordinate frames and units must agree.

    Returns:
        True when all observations are valid; incompatible frames or units raise an error.
    """
    if len({v.frame for v in values}) > 1 or len({v.unit for v in values}) > 1:
        raise ServiceError('geometry.frame_mismatch', 'Geometry frames and units must match')
    return all(v.valid for v in values)


class PointLineRequest(CalculationRequest):
    point: PositionedPoint
    line: Line


def point_line_distance(request: PointLineRequest, context: CalculationContext) -> Scalar:
    """Measure perpendicular distance using the public xy line convention.

    Args:
        request: Typed observed point and normalized line with compatible frame and units.
        context: Isolated operation identity, resource limits and verified shared-file
            access.

    Returns:
        Observed scalar with validity, frame, length/angle unit and an invalid-observation
        reason.
    """
    valid = compatible(request.point, request.line) and request.point.point is not None
    point, line = request.point.point, request.line
    return Scalar(valid=valid, reason=None if valid else 'invalid_input',
                  frame=line.frame, unit=line.unit,
                  value=abs(line.a * point.x + line.b * point.y + line.c) if valid else None)


OPERATIONS = [
    Operation('/geometry/lines:fit', PointFitRequest, LineFitResult, fit_line,
              'Fit an infinite line from supplied xy points', 'geometry', {'line': 'line'}),
    Operation('/geometry/circles:fit', PointFitRequest, CircleFitResult, fit_circle,
              'Fit a circle from supplied xy points', 'geometry', {'circle': 'circle'}),
    Operation('/geometry/point-line-distances', PointLineRequest, Scalar, point_line_distance,
              'Measure perpendicular point-to-line distance', 'geometry', {'value': 'number'}),
]


class TwoPointRequest(CalculationRequest):
    point_a: PositionedPoint
    point_b: PositionedPoint


def two_point_line(request: TwoPointRequest, context: CalculationContext) -> LineFitResult:
    """Construct a normalized xy line through two compatible observed points.

    Args:
        request: Typed two observed points with compatible coordinate frames and units.
        context: Isolated operation identity, resource limits and verified shared-file
            access.

    Returns:
        Observed normalized line, selected samples and independently computed residual
        quality.
    """
    first, second = request.point_a, request.point_b
    valid = compatible(first, second) and first.point is not None and second.point is not None
    if not valid:
        line = Line(valid=False, reason='invalid_input', frame=first.frame, unit=first.unit)
        return LineFitResult(valid=False, reason='invalid_input', frame=first.frame,
                             unit=first.unit, line=line, diagnostics=FitDiagnostics(sample_count=0))
    return fit_line(PointFitRequest(points=[first.point, second.point],
                                    frame=first.frame, unit=first.unit), context)


def point_distance(request: TwoPointRequest, context: CalculationContext) -> Scalar:
    """Measure the Euclidean distance between two compatible observed points.

    Args:
        request: Typed two observed points with compatible coordinate frames and units.
        context: Isolated operation identity, resource limits and verified shared-file
            access.

    Returns:
        Observed scalar with validity, frame, length/angle unit and an invalid-observation
        reason.
    """
    first, second = request.point_a, request.point_b
    valid = compatible(first, second) and first.point is not None and second.point is not None
    return Scalar(valid=valid, reason=None if valid else 'invalid_input',
                  frame=first.frame, unit=first.unit, value=float(np.hypot(
                      second.point.x - first.point.x, second.point.y - first.point.y))
                  if valid else None)


class LineAngleRequest(CalculationRequest):
    line_a: Line
    line_b: Line


def line_angle(request: LineAngleRequest, context: CalculationContext) -> Scalar:
    """Return the smaller unsigned angle between infinite lines in degrees.

    Args:
        request: Typed two observed lines whose smaller unsigned angle is requested.
        context: Isolated operation identity, resource limits and verified shared-file
            access.

    Returns:
        Observed scalar with validity, frame, length/angle unit and an invalid-observation
        reason.
    """
    first, second = request.line_a, request.line_b
    valid = compatible(first, second)
    value = None
    if valid:
        cosine = np.clip(abs(first.a * second.a + first.b * second.b), 0, 1)
        value = float(np.rad2deg(np.arccos(cosine)))
    return Scalar(valid=valid, reason=None if valid else 'invalid_input',
                  frame=first.frame, unit='deg', value=value)


class PointCircleRequest(CalculationRequest):
    point: PositionedPoint
    circle: Circle


def point_circle_distance(request: PointCircleRequest, context: CalculationContext) -> Scalar:
    """Measure the shortest distance to the circumference, including inside points.

    Args:
        request: Typed observed point and circle for shortest circumference distance.
        context: Isolated operation identity, resource limits and verified shared-file
            access.

    Returns:
        Observed scalar with validity, frame, length/angle unit and an invalid-observation
        reason.
    """
    point, circle = request.point, request.circle
    valid = compatible(point, circle) and point.point is not None
    value = None
    if valid:
        value = abs(float(np.hypot(point.point.x - circle.center.x,
                                   point.point.y - circle.center.y)) - circle.radius)
    return Scalar(valid=valid, reason=None if valid else 'invalid_input',
                  frame=circle.frame, unit=circle.unit, value=value)


class CircleDimensionsRequest(CalculationRequest):
    circle: Circle
    samples: List[PositionedPoint] = Field(default_factory=list, max_length=100000)


class CircleDimensionsResult(Observation):
    radius: Scalar
    diameter: Scalar
    sampled_diameter_spread: Scalar


def circle_dimensions(
    request: CircleDimensionsRequest, context: CalculationContext,
) -> CircleDimensionsResult:
    """Report 2*(max sampled radius - min sampled radius), not ellipse roundness.

    Args:
        request: Typed observed circle and sampled radii for diameter and radial variation.
        context: Isolated operation identity, resource limits and verified shared-file
            access.

    Returns:
        Circle diameter and observed radial variation with matching frame and unit.
    """
    circle = request.circle
    valid = compatible(circle, *request.samples)
    common = {'frame': circle.frame, 'unit': circle.unit}
    radii = []
    if valid:
        radii = [float(np.hypot(p.point.x - circle.center.x, p.point.y - circle.center.y))
                 for p in request.samples if p.point is not None]
    radius = Scalar(valid=circle.valid, value=circle.radius, **common)
    diameter = Scalar(valid=circle.valid, value=2 * circle.radius if circle.valid else None,
                      **common)
    enough = valid and len(radii) >= 2
    spread = Scalar(valid=enough, value=2 * (max(radii) - min(radii)) if enough else None,
                    reason=None if enough else 'insufficient_valid_samples', **common)
    return CircleDimensionsResult(valid=circle.valid, radius=radius, diameter=diameter,
                                  sampled_diameter_spread=spread, **common)


OPERATIONS.extend([
    Operation('/geometry/two-point-lines', TwoPointRequest, LineFitResult, two_point_line,
              'Construct a line through two points', 'geometry',
              {'line': 'line'}),
    Operation('/geometry/point-distances', TwoPointRequest, Scalar, point_distance,
              'Measure two-point distance', 'geometry', {'value': 'number'}),
    Operation('/geometry/line-angles', LineAngleRequest, Scalar, line_angle,
              'Measure smaller unsigned line angle', 'geometry', {'value': 'number'}),
    Operation('/geometry/point-circle-distances', PointCircleRequest, Scalar, point_circle_distance,
              'Measure shortest point-to-circumference distance', 'geometry', {'value': 'number'}),
    Operation('/geometry/circle-dimensions', CircleDimensionsRequest, CircleDimensionsResult,
              circle_dimensions, 'Calculate radius, diameter and sampled radial spread', 'geometry',
              {'radius': 'scalar', 'diameter': 'scalar', 'sampled_diameter_spread': 'scalar'}),
])
