"""GUI-independent stable point fits in image x/y coordinates."""

from typing import Any, Dict, Sequence

import numpy as np


def _points(points: Sequence[Sequence[float]], minimum: int) -> np.ndarray:
    data = np.asarray(points, dtype=float)
    if data.ndim != 2 or data.shape[1] != 2 or len(data) < minimum:
        raise ValueError('insufficient_samples')
    if not np.isfinite(data).all():
        raise ValueError('nonfinite_samples')
    return data


def fit_line_points(points: Sequence[Sequence[float]]) -> Dict[str, Any]:
    """Fit an orthogonal line and report actual sample residuals.

    Args:
        points: Image (x, y) points, with at least two distinct positions.

    Returns:
        Normalized coefficients, row/column legacy endpoints and residual metrics.
    """
    data = _points(points, 2)
    center = data.mean(axis=0)
    centered = data - center
    _, singular, vectors = np.linalg.svd(centered, full_matrices=False)
    if singular[0] <= 1e-12:
        raise ValueError('coincident_points')
    normal = vectors[-1]
    a, b = normal
    c = -np.dot(normal, center)
    direction = np.array([-b, a])
    projection = centered @ direction
    extent = np.ptp(projection) * 0.1
    start = center + (projection.min() - extent) * direction
    end = center + (projection.max() + extent) * direction
    residuals = data @ normal + c
    return {
        'params': (float(a), float(b), float(c)), 'point': tuple(center[::-1]),
        'direction': direction.tolist(), 'start': tuple(start[::-1]), 'end': tuple(end[::-1]),
        'angle': float(np.arctan2(direction[1], direction[0])), 'num_points': len(data),
        'mean_error': float(np.abs(residuals).mean()),
        'max_error': float(np.abs(residuals).max()),
        'rms_error': float(np.sqrt(np.mean(residuals ** 2))), 'residuals': residuals.tolist(),
    }


def fit_circle_points(points: Sequence[Sequence[float]]) -> Dict[str, Any]:
    """Fit a nondegenerate circle with translation-conditioned algebraic least squares.

    Args:
        points: At least three noncollinear image (x, y) points.

    Returns:
        Circle, actual radial residuals and sampled diameter spread.
    """
    data = _points(points, 3)
    origin = data.mean(axis=0)
    centered = data - origin
    if np.linalg.matrix_rank(centered) < 2:
        raise ValueError('collinear_points')
    matrix = np.column_stack([2 * centered, np.ones(len(data))])
    params, _, rank, _ = np.linalg.lstsq(matrix, np.sum(centered ** 2, axis=1), rcond=None)
    if rank < 3:
        raise ValueError('degenerate_circle')
    radius_squared = np.dot(params[:2], params[:2]) + params[2]
    if radius_squared <= 1e-24:
        raise ValueError('degenerate_circle')
    center = params[:2] + origin
    radius = np.sqrt(radius_squared)
    radii = np.linalg.norm(data - center, axis=1)
    residuals = radii - radius
    high, low = int(radii.argmax()), int(radii.argmin())
    return {
        'center': tuple(center[::-1]), 'radius': float(radius), 'num_points': len(data),
        'mean_error': float(np.abs(residuals).mean()),
        'max_error': float(np.abs(residuals).max()),
        'rms_error': float(np.sqrt(np.mean(residuals ** 2))), 'residuals': residuals.tolist(),
        'ellipticity': float(2 * np.ptp(radii)), 'max_radius': float(radii[high]),
        'min_radius': float(radii[low]), 'max_radius_point': tuple(data[high]),
        'min_radius_point': tuple(data[low]),
    }
