"""Geometry composition over typed observations.

Every entry point takes :class:`~measure.computation_types.Observation` values,
checks frames and units **before** validity, and returns a new observation in
the same coordinate space (angles use ``rad``).  An invalid input never
produces a numeric zero: it propagates as ``invalid_input`` so a downstream
consumer cannot mistake "not measured" for "measured as zero".

This module owns the single implementation of the distance/angle formulas that
used to be duplicated in ``measure_workflow``, ``multi_target_workflow`` and the
v2 geometry family.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from measure.computation_types import (
    Array,
    CircleValue,
    LineValue,
    Observation,
    Space,
    XY,
    contract_error,
    invalid,
    observed,
    require_space_match,
    require_xy_array,
)

RADIAN_UNIT = "rad"


def _require_common_space(*observations: Observation) -> Space:
    """Return the shared space, rejecting mixed frames before checking validity.

    Args:
        observations: Observations that must share frame and unit.

    Returns:
        The first observation's space.

    Raises:
        ComputationContractError: If any observation uses a different frame or unit.
    """
    first = observations[0].space
    for index, observation in enumerate(observations[1:], start=1):
        require_space_match(first, observation.space, f"observation[{index}].space")
    return first


def circle_center(circle: Observation[CircleValue]) -> Observation[XY]:
    """Convert a circle observation into its centre point.

    Args:
        circle: Circle observation in any supported space.

    Returns:
        A valid point observation in the same space, or the circle's original
        failure reason when the input is invalid.
    """
    if not circle.valid:
        return invalid(circle.space, circle.reason or "invalid_input", dict(circle.diagnostics))
    value = circle.value
    return observed(circle.space, (float(value.center[0]), float(value.center[1])),
                    dict(circle.diagnostics))


def line_through_points(
    point_a: Observation[XY], point_b: Observation[XY],
) -> Observation[LineValue]:
    """Build the exact line through two points, keeping the real endpoints.

    Args:
        point_a: First point observation.
        point_b: Second point observation.

    Returns:
        A line whose coefficients are canonical xy and whose endpoints are the
        supplied points ordered along ``d = (-b, a)``, or ``coincident_points``.
    """
    space = _require_common_space(point_a, point_b)
    if not (point_a.valid and point_b.valid):
        return invalid(space, "invalid_input")
    ax, ay = point_a.value
    bx, by = point_b.value
    dx, dy = bx - ax, by - ay
    norm = math.hypot(dx, dy)
    if norm <= 0.0:
        return invalid(space, "coincident_points")
    a, b = -dy / norm, dx / norm
    c = -(a * ax + b * ay)
    if a < 0.0 or (a == 0.0 and b < 0.0):
        a, b, c = -a, -b, -c
    direction_x, direction_y = -b, a
    projection_a = ax * direction_x + ay * direction_y
    projection_b = bx * direction_x + by * direction_y
    if projection_a <= projection_b:
        start, end = (float(ax), float(ay)), (float(bx), float(by))
    else:
        start, end = (float(bx), float(by)), (float(ax), float(ay))
    value = LineValue(coefficients=(a, b, c), start=start, end=end)
    return observed(space, value)


def point_distance(
    point_a: Observation[XY], point_b: Observation[XY],
) -> Observation[float]:
    """Measure the Euclidean distance between two points in one space.

    Args:
        point_a: First point observation.
        point_b: Second point observation.

    Returns:
        Distance observation in the shared length unit, or ``invalid_input``.
    """
    space = _require_common_space(point_a, point_b)
    if not (point_a.valid and point_b.valid):
        return invalid(space, "invalid_input")
    ax, ay = point_a.value
    bx, by = point_b.value
    return observed(space, math.hypot(bx - ax, by - ay))


def point_line_distance(
    point: Observation[XY], line: Observation[LineValue],
) -> Observation[float]:
    """Measure the perpendicular distance from a point to an infinite line.

    Args:
        point: Point observation.
        line: Line observation in the same space.

    Returns:
        Non-negative distance in the shared length unit, or ``invalid_input``.
    """
    space = _require_common_space(point, line)
    if not (point.valid and line.valid):
        return invalid(space, "invalid_input")
    x, y = point.value
    a, b, c = line.value.coefficients
    return observed(space, abs(a * x + b * y + c))


def line_angle(
    line_a: Observation[LineValue], line_b: Observation[LineValue],
) -> Observation[float]:
    """Measure the smaller unsigned angle between two infinite lines.

    Args:
        line_a: First line observation.
        line_b: Second line observation in the same frame.

    Returns:
        Angle in ``[0, pi/2]`` with unit ``rad``, or ``invalid_input``.
    """
    space = _require_common_space(line_a, line_b)
    if not (line_a.valid and line_b.valid):
        return invalid(Space(space.frame, RADIAN_UNIT), "invalid_input")
    a1, b1 = line_a.value.coefficients[:2]
    a2, b2 = line_b.value.coefficients[:2]
    cosine = min(1.0, max(0.0, abs(a1 * a2 + b1 * b2)))
    return observed(Space(space.frame, RADIAN_UNIT), math.acos(cosine))


def point_circle_distance(
    point: Observation[XY], circle: Observation[CircleValue],
) -> Observation[float]:
    """Measure the shortest distance from a point to a circle's circumference.

    Args:
        point: Point observation.
        circle: Circle observation in the same space.

    Returns:
        Non-negative distance to the circumference (not to the centre), or
        ``invalid_input``.
    """
    space = _require_common_space(point, circle)
    if not (point.valid and circle.valid):
        return invalid(space, "invalid_input")
    x, y = point.value
    cx, cy = circle.value.center
    return observed(space, abs(math.hypot(x - cx, y - cy) - circle.value.radius))


def sampled_diameter_spread(
    circle: Observation[CircleValue], samples: Observation[Array],
) -> Observation[float]:
    """Compute ``2 * (max sampled radius - min sampled radius)``.

    This is a sampled radial spread, not an ellipse axis or a standard
    roundness metric, and it must not be reported when fewer than two samples
    exist.

    Args:
        circle: Circle observation providing centre and radius.
        samples: Observation whose payload is a finite ``(N, 2)`` sample array.

    Returns:
        Spread observation in the shared length unit, or an invalid observation
        with ``insufficient_samples`` / ``invalid_input``.
    """
    space = _require_common_space(circle, samples)
    if not (circle.valid and samples.valid):
        return invalid(space, "invalid_input")
    points = require_xy_array(samples.value, "samples.value")
    if len(points) < 2:
        return invalid(space, "insufficient_samples")
    cx, cy = circle.value.center
    radii = np.linalg.norm(points - np.array([cx, cy], dtype=np.float64), axis=1)
    return observed(space, float(2.0 * (radii.max() - radii.min())))