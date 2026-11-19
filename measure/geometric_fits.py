"""GUI-independent stable point fits in image x/y coordinates.

Two entry points share one solver per model:

* ``fit_line_points`` / ``fit_circle_points`` keep the historical dictionary
  shape (row/column fields, legacy key names) for existing callers;
* ``fit_line_observation`` / ``fit_circle_observation`` return the canonical
  :class:`~measure.computation_types.Observation` with typed payloads and
  explicit fit diagnostics.

Both paths run the *same* numerical code, so a coefficient, residual or RMS
value can never disagree between the two representations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from measure.computation_types import (
    Array,
    canonical_line_coefficients,
    CircleValue,
    ComputationContractError,
    LineValue,
    Observation,
    Space,
    XY,
    invalid,
    observed,
    require_xy_array,
)


@dataclass(frozen=True)
class FitDiagnostics:
    """Actual samples, signed residuals and probe accounting of one fit.

    Pure point fits leave the probe counts as ``None`` (unknown) rather than
    zero, so a later coverage calculation cannot mistake "not a probe fit" for
    "every probe failed".
    """

    samples_xy: Array
    sample_count: int
    residuals: Optional[Array]
    mean_absolute_error: Optional[float]
    max_absolute_error: Optional[float]
    rms_error: Optional[float]
    residual_space: Space
    requested_probe_count: Optional[int] = None
    successful_probe_count: Optional[int] = None
    sample_probe_indices: Optional[Array] = None

    @property
    def fit_rms(self) -> Optional[float]:
        """Return the RMS residual (legacy alias)."""
        return self.rms_error


@dataclass(frozen=True)
class CircleFitDiagnostics(FitDiagnostics):
    """Circle fit diagnostics including the sampled radial spread."""

    min_radius: Optional[float] = None
    max_radius: Optional[float] = None
    sampled_diameter_spread: Optional[float] = None
    min_radius_point: Optional[XY] = None
    max_radius_point: Optional[XY] = None


def _points(points: Sequence[Sequence[float]], minimum: int) -> np.ndarray:
    """Validate legacy fit input, raising plain ``ValueError`` on domain failure."""
    data = np.asarray(points, dtype=float)
    if data.ndim != 2 or data.shape[1] != 2 or len(data) < minimum:
        raise ValueError('insufficient_samples')
    if not np.isfinite(data).all():
        raise ValueError('nonfinite_samples')
    return data


def _line_solution(data: np.ndarray) -> Dict[str, Any]:
    """Solve one orthogonal line fit in canonical xy form.

    Args:
        data: Finite ``(N, 2)`` points with at least two distinct positions.

    Returns:
        Coefficients (canonical sign), direction, extended endpoints and
        signed residuals computed with the canonical coefficients.

    Raises:
        ValueError: ``coincident_points`` when the samples collapse.
    """
    center = data.mean(axis=0)
    centered = data - center
    _, singular, vectors = np.linalg.svd(centered, full_matrices=False)
    if singular[0] <= 1e-12:
        raise ValueError('coincident_points')
    normal = vectors[-1]
    a, b, c = canonical_line_coefficients(
        float(normal[0]), float(normal[1]), float(-np.dot(normal, center)))
    direction = np.array([-b, a])
    projection = centered @ direction
    extent = float(np.ptp(projection)) * 0.1
    low, high = float(projection.min()) - extent, float(projection.max()) + extent
    start = center + low * direction
    end = center + high * direction
    residuals = data @ np.array([a, b]) + c
    return {
        'coefficients': (a, b, c),
        'center': center,
        'direction': direction,
        'start': start,
        'end': end,
        'residuals': residuals,
        'mean_error': float(np.abs(residuals).mean()),
        'max_error': float(np.abs(residuals).max()),
        'rms_error': float(np.sqrt(np.mean(residuals ** 2))),
    }


def _circle_solution(data: np.ndarray) -> Dict[str, Any]:
    """Solve one translation-conditioned algebraic circle fit.

    Args:
        data: Finite ``(N, 2)`` points with at least three noncollinear positions.

    Returns:
        Centre, radius, radial residuals and sampled spread extremes.

    Raises:
        ValueError: ``collinear_points`` or ``degenerate_circle``.
    """
    origin = data.mean(axis=0)
    centered = data - origin
    if np.linalg.matrix_rank(centered) < 2:
        raise ValueError('collinear_points')
    matrix = np.column_stack([2 * centered, np.ones(len(data))])
    params, _, rank, _ = np.linalg.lstsq(matrix, np.sum(centered ** 2, axis=1), rcond=None)
    if rank < 3:
        raise ValueError('degenerate_circle')
    radius_squared = float(np.dot(params[:2], params[:2]) + params[2])
    if radius_squared <= 1e-24:
        raise ValueError('degenerate_circle')
    center = params[:2] + origin
    radius = float(np.sqrt(radius_squared))
    radii = np.linalg.norm(data - center, axis=1)
    residuals = radii - radius
    high, low = int(radii.argmax()), int(radii.argmin())
    return {
        'center': center,
        'radius': radius,
        'residuals': residuals,
        'mean_error': float(np.abs(residuals).mean()),
        'max_error': float(np.abs(residuals).max()),
        'rms_error': float(np.sqrt(np.mean(residuals ** 2))),
        'min_radius': float(radii[low]),
        'max_radius': float(radii[high]),
        'spread': float(2.0 * np.ptp(radii)),
        'min_radius_point': (float(data[low][0]), float(data[low][1])),
        'max_radius_point': (float(data[high][0]), float(data[high][1])),
    }


def fit_line_points(points: Sequence[Sequence[float]]) -> Dict[str, Any]:
    """Fit an orthogonal line and report actual sample residuals.

    Args:
        points: Image (x, y) points, with at least two distinct positions.

    Returns:
        Normalized coefficients, row/column legacy endpoints and residual metrics.
    """
    data = _points(points, 2)
    fit = _line_solution(data)
    a, b, c = fit['coefficients']
    center = fit['center']
    start, end = fit['start'], fit['end']
    return {
        'params': (a, b, c), 'point': tuple(center[::-1]),
        'direction': fit['direction'].tolist(),
        'start': (float(start[1]), float(start[0])),
        'end': (float(end[1]), float(end[0])),
        'angle': float(np.arctan2(fit['direction'][1], fit['direction'][0])),
        'num_points': len(data),
        'mean_error': fit['mean_error'],
        'max_error': fit['max_error'],
        'rms_error': fit['rms_error'],
        'residuals': fit['residuals'].tolist(),
    }


def fit_circle_points(points: Sequence[Sequence[float]]) -> Dict[str, Any]:
    """Fit a nondegenerate circle with translation-conditioned algebraic least squares.

    Args:
        points: At least three noncollinear image (x, y) points.

    Returns:
        Circle, actual radial residuals and sampled diameter spread.
    """
    data = _points(points, 3)
    fit = _circle_solution(data)
    center = fit['center']
    return {
        'center': tuple(center[::-1]), 'radius': fit['radius'], 'num_points': len(data),
        'mean_error': fit['mean_error'],
        'max_error': fit['max_error'],
        'rms_error': fit['rms_error'],
        'residuals': fit['residuals'].tolist(),
        'ellipticity': fit['spread'],
        'max_radius': fit['max_radius'], 'min_radius': fit['min_radius'],
        'max_radius_point': tuple(fit['max_radius_point'][::-1]),
        'min_radius_point': tuple(fit['min_radius_point'][::-1]),
    }


def fit_line_observation(points_xy: Array, *, space: Space) -> Observation[LineValue]:
    """Fit a line and return it as a typed observation.

    Args:
        points_xy: Finite ``(N, 2)`` sample points in ``space``.
        space: Frame and unit of the samples.

    Returns:
        A valid observation with canonical coefficients and endpoints, or an
        invalid observation with ``insufficient_samples`` / ``coincident_points``.

    Raises:
        ComputationContractError: If the samples are malformed or not finite.
    """
    data = require_xy_array(points_xy, "points_xy")
    if len(data) < 2:
        return invalid(space, "insufficient_samples", {"fit": _line_diagnostics(data, space)})
    try:
        fit = _line_solution(data)
    except ValueError as error:
        return invalid(space, str(error), {"fit": _line_diagnostics(data, space)})
    value = LineValue(
        coefficients=fit['coefficients'],
        start=(float(fit['start'][0]), float(fit['start'][1])),
        end=(float(fit['end'][0]), float(fit['end'][1])),
    )
    return observed(space, value, {"fit": _line_diagnostics(data, space, fit)})


def _line_diagnostics(
    data: np.ndarray, space: Space, fit: Optional[Dict[str, Any]] = None,
) -> FitDiagnostics:
    """Assemble line diagnostics with real residuals when the fit succeeded."""
    residuals = fit['residuals'] if fit else None
    return FitDiagnostics(
        samples_xy=np.array(data, dtype=np.float64, copy=True),
        sample_count=len(data),
        residuals=None if residuals is None else np.array(residuals, dtype=np.float64, copy=True),
        mean_absolute_error=fit['mean_error'] if fit else None,
        max_absolute_error=fit['max_error'] if fit else None,
        rms_error=fit['rms_error'] if fit else None,
        residual_space=space,
    )


def fit_circle_observation(points_xy: Array, *, space: Space) -> Observation[CircleValue]:
    """Fit a circle and return it as a typed observation.

    Args:
        points_xy: Finite ``(N, 2)`` sample points in ``space``.
        space: Frame and unit of the samples.

    Returns:
        A valid observation with centre/radius, or an invalid observation with
        ``insufficient_samples`` / ``collinear_points`` / ``degenerate_circle``.

    Raises:
        ComputationContractError: If the samples are malformed or not finite.
    """
    data = require_xy_array(points_xy, "points_xy")
    if len(data) < 3:
        return invalid(space, "insufficient_samples", {"fit": _circle_diagnostics(data, space)})
    try:
        fit = _circle_solution(data)
    except ValueError as error:
        return invalid(space, str(error), {"fit": _circle_diagnostics(data, space)})
    value = CircleValue(
        center=(float(fit['center'][0]), float(fit['center'][1])),
        radius=float(fit['radius']),
    )
    return observed(space, value, {"fit": _circle_diagnostics(data, space, fit)})


def _circle_diagnostics(
    data: np.ndarray, space: Space, fit: Optional[Dict[str, Any]] = None,
) -> CircleFitDiagnostics:
    """Assemble circle diagnostics with radial residuals when the fit succeeded."""
    residuals = fit['residuals'] if fit else None
    return CircleFitDiagnostics(
        samples_xy=np.array(data, dtype=np.float64, copy=True),
        sample_count=len(data),
        residuals=None if residuals is None else np.array(residuals, dtype=np.float64, copy=True),
        mean_absolute_error=fit['mean_error'] if fit else None,
        max_absolute_error=fit['max_error'] if fit else None,
        rms_error=fit['rms_error'] if fit else None,
        residual_space=space,
        min_radius=fit['min_radius'] if fit else None,
        max_radius=fit['max_radius'] if fit else None,
        sampled_diameter_spread=fit['spread'] if fit else None,
        min_radius_point=fit['min_radius_point'] if fit else None,
        max_radius_point=fit['max_radius_point'] if fit else None,
    )