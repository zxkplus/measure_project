"""Exact geometry transforms and sample-preserving metric mapping.

Two different jobs live here and must not be confused:

* :func:`transform_geometry` maps a **typed geometry value** exactly.  A
  similarity scales a circle's radius; an affine or projective map that would
  turn it into an ellipse is reported as ``unsupported_geometry_mapping``
  instead of silently keeping a scalar radius.
* :func:`map_metric_samples` maps the **actual selected samples** and refits the
  declared model in the destination space, so residuals and radii are true
  millimetres rather than converted pixels.

Display rendering is separate again: :func:`map_geometry_for_display` samples a
curve for drawing and labels the result ``display_curve``; those points are not
measurement samples.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple, Union

import cv2
import numpy as np

from measure.computation_types import (
    Array,
    CircleValue,
    ComputationContractError,
    LineValue,
    Mapping2D,
    Observation,
    PolylineValue,
    RotatedBox,
    Space,
    XY,
    contract_error,
    invalid,
    observed,
    require_finite,
    require_space_match,
    require_xy_array,
)
from measure.geometric_fits import (
    CircleFitDiagnostics,
    FitDiagnostics,
    fit_circle_observation,
    fit_line_observation,
)

DISPLAY_APPROXIMATION = "display_curve"
MIN_HOMOGENEOUS = 1e-12


@dataclass(frozen=True)
class MatrixStep:
    """One 3x3 matrix step inside a :class:`MappingChain`."""

    mapping: Mapping2D

    @property
    def source(self) -> Space:
        """Return the step's input space."""
        return self.mapping.source

    @property
    def destination(self) -> Space:
        """Return the step's output space."""
        return self.mapping.destination


@dataclass(frozen=True)
class LensStep:
    """One pinhole lens step, optionally with radial/tangential distortion."""

    source: Space
    destination: Space
    camera_matrix: Array
    distortion: Array
    direction: str = "undistort"

    def __post_init__(self) -> None:
        if self.direction not in ("undistort", "distort"):
            raise contract_error("contract.invalid_parameter",
                                 f"unknown lens direction {self.direction!r}",
                                 "lens.direction")
        matrix = np.asarray(self.camera_matrix)
        if matrix.shape != (3, 3) or matrix.dtype.kind not in "iuf":
            raise contract_error("contract.invalid_shape", "camera_matrix must be 3x3 numeric",
                                 "lens.camera_matrix")
        if not bool(np.isfinite(matrix).all()) or float(matrix[0, 0]) == 0.0 or float(
            matrix[1, 1]
        ) == 0.0:
            raise contract_error("contract.invalid_parameter",
                                 "camera_matrix must be finite with non-zero focal length",
                                 "lens.camera_matrix")
        distortion = np.asarray(self.distortion, dtype=np.float64).reshape(-1)
        if not bool(np.isfinite(distortion).all()):
            raise contract_error("contract.nonfinite", "distortion must be finite",
                                 "lens.distortion")
        object.__setattr__(self, "camera_matrix", np.array(matrix, dtype=np.float64, copy=True))
        object.__setattr__(self, "distortion", distortion)


Step = Union[MatrixStep, LensStep]


@dataclass(frozen=True)
class MappingChain:
    """An ordered chain whose adjacent steps must agree on the intermediate space."""

    steps: Tuple[Step, ...]

    def __post_init__(self) -> None:
        steps = tuple(self.steps)
        if not steps:
            raise contract_error("contract.invalid_parameter",
                                 "a mapping chain needs at least one step", "chain.steps")
        for index, step in enumerate(steps):
            if not isinstance(step, (MatrixStep, LensStep)):
                raise contract_error("contract.invalid_parameter",
                                     f"unsupported chain step {type(step).__name__!r}",
                                     f"chain.steps[{index}]")
            if index:
                require_space_match(steps[index - 1].destination, step.source,
                                    f"chain.steps[{index}].source")
        object.__setattr__(self, "steps", steps)

    @property
    def source(self) -> Space:
        """Return the chain's first input space."""
        return self.steps[0].source

    @property
    def destination(self) -> Space:
        """Return the chain's last output space."""
        return self.steps[-1].destination


@dataclass(frozen=True)
class MetricSamples:
    """Mapped samples plus the model that was refitted in the target space."""

    model: str
    samples_xy: Array
    geometry: Optional[Union[LineValue, CircleValue]] = None

    def validate(self) -> None:
        """Re-check finiteness of the mapped samples."""
        require_finite(self.samples_xy, "metric.samples_xy")


def _apply_matrix(points: Array, matrix: Array) -> Array:
    """Apply one homogeneous matrix, rejecting points on the horizon."""
    homogeneous = np.column_stack([points, np.ones(len(points))])
    mapped = homogeneous @ matrix.T
    denominator = mapped[:, 2]
    if not bool(np.isfinite(mapped).all()):
        raise _HorizonError("mapped coordinates are not finite")
    if len(points) and bool((np.abs(denominator) <= MIN_HOMOGENEOUS).any()):
        raise _HorizonError("point maps onto the projective horizon")
    if len(points) > 1:
        sign = np.sign(denominator)
        if bool((sign[:-1] * sign[1:] < 0.0).any()):
            raise _HorizonError("adjacent points cross the projective horizon")
    return mapped[:, :2] / denominator[:, None]


class _HorizonError(ValueError):
    """Internal signal that a projective mapping cannot represent the input."""


def _apply_lens(points: Array, step: LensStep) -> Array:
    """Apply one pinhole lens step in pixel coordinates.

    Both directions defer to OpenCV so every distortion model the project
    accepts (4, 5, 8, 12 and 14 coefficients, including rational and tilted
    sensor) is honoured. The undistort direction keeps OpenCV's default
    iteration criteria deliberately: that solver slack is what the historical
    display and measurement paths were pinned against.
    """
    matrix = np.asarray(step.camera_matrix, dtype=np.float64)
    distortion = np.asarray(step.distortion, dtype=np.float64).reshape(-1)
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float64)
    if step.direction == "undistort":
        result = cv2.undistortPoints(
            points.reshape(-1, 1, 2).astype(np.float64), matrix, distortion,
            None, matrix,
        )
        return result.reshape(-1, 2).astype(np.float64)
    fx, fy = float(matrix[0, 0]), float(matrix[1, 1])
    cx, cy = float(matrix[0, 2]), float(matrix[1, 2])
    normalized = np.column_stack([(points[:, 0] - cx) / fx,
                                  (points[:, 1] - cy) / fy])
    rays = np.column_stack([normalized, np.ones(len(normalized))]).astype(np.float64)
    projected, _ = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), matrix, distortion)
    return projected.reshape(-1, 2).astype(np.float64)


def _map_points(points: Array, chain: MappingChain) -> Array:
    """Run ``(N, 2)`` points through every chain step in order."""
    current = np.array(points, dtype=np.float64, copy=True)
    for step in chain.steps:
        if isinstance(step, MatrixStep):
            current = _apply_matrix(current, step.mapping.matrix)
        else:
            current = _apply_lens(current, step)
    return current


def map_sample_points(points: Array, chain: MappingChain) -> Observation[Array]:
    """Map selected sample points through a chain without touching the input.

    Args:
        points: Finite ``(N, 2)`` points in ``chain.source``; an empty array is
            allowed and keeps its shape.
        chain: Ordered mapping steps.

    Returns:
        A valid observation of the mapped points in ``chain.destination``, or an
        invalid one when a projective step cannot represent the input.
    """
    values = require_xy_array(points, "points") if len(np.asarray(points)) else \
        np.empty((0, 2), dtype=np.float64)
    try:
        mapped = _map_points(values, chain)
    except _HorizonError as error:
        return invalid(chain.destination, "unsupported_geometry_mapping",
                       {"detail": str(error)})
    return observed(chain.destination, mapped)


def _linear_part(matrix: Array) -> Array:
    return np.asarray(matrix, dtype=np.float64)[:2, :2]


def _similarity_scale(matrix: Array) -> Optional[float]:
    """Return the uniform scale of a similarity, or ``None`` when not similar."""
    linear = _linear_part(matrix)
    scale_squared = float(np.mean(np.sum(linear ** 2, axis=0)))
    if scale_squared <= 0.0:
        return None
    if not np.allclose(linear.T @ linear, scale_squared * np.eye(2), rtol=1e-9, atol=1e-12):
        return None
    if float(np.linalg.det(linear)) <= 0.0:
        return None
    return math.sqrt(scale_squared)


def _map_line(value: LineValue, mapping: Mapping2D) -> LineValue:
    """Transform an infinite line exactly with ``H^-T``."""
    coefficients = np.asarray(value.coefficients, dtype=np.float64)
    transformed = np.linalg.inv(np.asarray(mapping.matrix, dtype=np.float64)).T @ coefficients
    start = end = None
    if value.start is not None:
        mapped = _apply_matrix(np.asarray([value.start, value.end], dtype=np.float64),
                               mapping.matrix)
        start = (float(mapped[0][0]), float(mapped[0][1]))
        end = (float(mapped[1][0]), float(mapped[1][1]))
    return LineValue(coefficients=tuple(float(item) for item in transformed),
                     start=start, end=end)


def _map_circle(value: CircleValue, mapping: Mapping2D) -> CircleValue:
    """Scale a circle exactly, refusing shapes that are no longer circles."""
    scale = _similarity_scale(mapping.matrix)
    if scale is None:
        raise _Unsupported("a circle stays a circle only under a similarity")
    center = _apply_matrix(np.asarray([value.center], dtype=np.float64), mapping.matrix)[0]
    return CircleValue(center=(float(center[0]), float(center[1])),
                       radius=float(value.radius) * scale)


def _rotation_angle(matrix: Array) -> float:
    linear = _linear_part(matrix)
    return math.atan2(float(linear[1, 0]), float(linear[0, 0]))


def _map_box(value: RotatedBox, mapping: Mapping2D) -> RotatedBox:
    """Map a rotated box, scaling its size and adding the mapping rotation."""
    scale = _similarity_scale(mapping.matrix)
    if scale is None:
        raise _Unsupported("a rotated box keeps its shape only under a similarity")
    center = _apply_matrix(np.asarray([value.center], dtype=np.float64), mapping.matrix)[0]
    return RotatedBox(
        center=(float(center[0]), float(center[1])),
        size_wh=(value.size_wh[0] * scale, value.size_wh[1] * scale),
        angle_rad=value.angle_rad + _rotation_angle(mapping.matrix),
    )


class _Unsupported(ValueError):
    """Internal signal that a transform cannot represent the geometry exactly."""


def transform_geometry(
    geometry: Observation[Any], mapping: Mapping2D,
) -> Observation[Any]:
    """Transform one typed geometry value exactly.

    Args:
        geometry: Observation whose space equals ``mapping.source``.
        mapping: Explicit source-to-destination mapping.

    Returns:
        The transformed observation in ``mapping.destination``; an invalid input
        propagates its reason, and an inexactly representable result returns
        ``unsupported_geometry_mapping``.
    """
    require_space_match(geometry.space, mapping.source, "mapping.source")
    if not geometry.valid:
        return invalid(mapping.destination, geometry.reason or "invalid_input",
                       dict(geometry.diagnostics))
    value = geometry.value
    try:
        if isinstance(value, LineValue):
            mapped = _map_line(value, mapping)
        elif isinstance(value, CircleValue):
            mapped = _map_circle(value, mapping)
        elif isinstance(value, RotatedBox):
            mapped = _map_box(value, mapping)
        elif isinstance(value, PolylineValue):
            points = _apply_matrix(np.asarray(value.points_xy, dtype=np.float64),
                                   mapping.matrix)
            mapped = PolylineValue(points_xy=points, closed=value.closed)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            mapped_points = _apply_matrix(np.asarray([value], dtype=np.float64),
                                          mapping.matrix)
            mapped = (float(mapped_points[0][0]), float(mapped_points[0][1]))
        else:
            raise _Unsupported(f"no exact transform for {type(value).__name__}")
    except _Unsupported as error:
        return invalid(mapping.destination, "unsupported_geometry_mapping",
                       {"detail": str(error)})
    except _HorizonError as error:
        return invalid(mapping.destination, "unsupported_geometry_mapping",
                       {"detail": str(error)})
    return observed(mapping.destination, mapped, dict(geometry.diagnostics))


def map_geometry_for_display(
    geometry: Observation[Any], chain: MappingChain, curve_samples: int = 64,
) -> Observation[PolylineValue]:
    """Approximate one geometry as a display polyline in the chain's destination.

    Args:
        geometry: Valid circle observation in ``chain.source``.
        chain: Mapping steps to apply.
        curve_samples: Number of samples used to draw the closed curve.

    Returns:
        A closed polyline observation; diagnostics mark it ``display_curve`` so
        it can never be mistaken for measurement samples.
    """
    require_space_match(geometry.space, chain.source, "chain.source")
    if not geometry.valid:
        return invalid(chain.destination, geometry.reason or "invalid_input")
    if curve_samples < 3:
        raise contract_error("contract.invalid_parameter",
                             "curve_samples must be at least three", "curve_samples")
    value = geometry.value
    if not isinstance(value, CircleValue):
        raise contract_error("contract.invalid_parameter",
                             "display rendering supports circles only", "geometry")
    angles = np.linspace(0.0, 2.0 * math.pi, int(curve_samples), endpoint=False)
    samples = np.column_stack([
        value.center[0] + value.radius * np.cos(angles),
        value.center[1] + value.radius * np.sin(angles),
    ])
    try:
        mapped = _map_points(samples, chain)
    except _HorizonError as error:
        return invalid(chain.destination, "unsupported_geometry_mapping",
                       {"detail": str(error)})
    polyline = PolylineValue(points_xy=mapped, closed=True)
    return observed(chain.destination, polyline, {"approximation": DISPLAY_APPROXIMATION})


def map_metric_samples(
    samples: Observation[Array],
    chain: MappingChain,
    target_model: str = "points",
) -> Observation[MetricSamples]:
    """Map selected samples and refit the declared model in the target space.

    Args:
        samples: Valid observation whose payload is ``(N, 2)`` in ``chain.source``.
        chain: Mapping steps from the sample space to the metric space.
        target_model: ``points`` (no fit), ``line`` or ``circle``.

    Returns:
        A valid observation carrying the mapped samples and the refitted model,
        or an invalid observation when mapping or fitting fails.
    """
    if target_model not in ("points", "line", "circle"):
        raise contract_error("contract.invalid_parameter",
                             f"unknown target model {target_model!r}", "target_model")
    require_space_match(samples.space, chain.source, "chain.source")
    if not samples.valid:
        return invalid(chain.destination, samples.reason or "invalid_input")
    mapped = map_sample_points(samples.value, chain)
    if not mapped.valid:
        return mapped
    if target_model == "points":
        value = MetricSamples(model="points", samples_xy=mapped.value, geometry=None)
        return observed(chain.destination, value, dict(samples.diagnostics))
    if target_model == "circle":
        fit = fit_circle_observation(mapped.value, space=chain.destination)
    else:
        fit = fit_line_observation(mapped.value, space=chain.destination)
    if not fit.valid:
        return invalid(chain.destination, fit.reason or "solver_failed", dict(fit.diagnostics))
    value = MetricSamples(model=target_model, samples_xy=mapped.value, geometry=fit.value)
    diagnostics = dict(fit.diagnostics)
    diagnostics["approximation"] = "refit_from_samples"
    return observed(chain.destination, value, diagnostics)