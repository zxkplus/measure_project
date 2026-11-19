"""Sample curves through actual projective and OpenCV lens coordinate mappings."""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np

from measure.computation_types import Space, make_mapping
from measure.geometry_mapping import (
    LensStep, MappingChain, MatrixStep, map_sample_points,
)
from measure_api.v2.errors import ServiceError
from measure_api.v2.families.alignment import validated_matrix
from measure_api.v2.models import CoordinateMapping


def perspective(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Project finite xy samples, rejecting points on the projective horizon.

    Args:
        points: Finite xy samples in the declared input frame.
        matrix: Declared homogeneous coordinate transform.

    Returns:
        Finite projected xy samples.
    """
    values = np.column_stack([points, np.ones(len(points))]) @ matrix.T
    if np.any(np.abs(values[:, 2]) <= 1e-12):
        raise ServiceError(
            "visualization.point_at_infinity", "Curve crosses the projective horizon"
        )
    return values[:, :2] / values[:, 2:3]


def _lens_chain(
    mapping: CoordinateMapping, camera: np.ndarray, distortion: np.ndarray,
    corrected_to_physical: np.ndarray, physical_to_corrected: np.ndarray, forward: bool,
):
    """Build the ordered lens + plane steps for one declared direction.

    Args:
        mapping: Declared nonlinear mapping carrying the frame and unit names.
        camera: Pinhole camera matrix.
        distortion: Distortion coefficients in OpenCV order.
        corrected_to_physical: Plane transform applied after undistortion.
        physical_to_corrected: Its declared inverse.
        forward: Whether the chain runs observation to display frame.

    Returns:
        A mapping chain that maps ``(N, 2)`` samples into the destination frame.
    """
    observation = Space(mapping.source_frame, "px")
    corrected = Space(mapping.source_frame + ":undistorted", "px")
    display = Space(mapping.destination_frame, mapping.unit)
    lens = LensStep(source=observation, destination=corrected, camera_matrix=camera,
                    distortion=distortion, direction="undistort")
    plane_out = MatrixStep(make_mapping(corrected_to_physical, source=corrected,
                                        destination=display, kind="projective"))
    plane_in = MatrixStep(make_mapping(physical_to_corrected, source=display,
                                       destination=corrected, kind="projective"))
    distort = LensStep(source=corrected, destination=observation, camera_matrix=camera,
                       distortion=distortion, direction="distort")
    return MappingChain((lens, plane_out) if forward else (plane_in, distort))


def mapping_function(
    mapping: CoordinateMapping, frame: str, destination: str
) -> Tuple[Callable, object, str]:
    """Choose the declared direction and units, retaining explicit nonlinear lens models.

    Args:
        mapping: Explicit forward mapping and its inverse or lens model.
        frame: Declared observation coordinate frame.
        destination: Requested display coordinate frame.

    Returns:
        The transform, optional analytic matrix and destination units.
    """
    forward = frame == mapping.source_frame and destination == mapping.destination_frame
    reverse = frame == mapping.destination_frame and destination == mapping.source_frame
    if not forward and not reverse:
        raise ServiceError(
            "visualization.frame_mismatch",
            "Mapping does not connect observation and display frames",
        )
    unit = mapping.unit if forward else "px"
    if mapping.type != "nonlinear":
        matrix = validated_matrix(mapping)
        if reverse:
            matrix = np.array(mapping.inverse_matrix, np.float64)
        return lambda points: perspective(points, matrix), matrix, unit
    detail = mapping.details
    if detail.get("model") != "opencv_pinhole_plane":
        raise ServiceError(
            "visualization.nonlinear_unsupported",
            "Unsupported nonlinear display mapping model",
            details={"model": detail.get("model")},
        )
    try:
        k = np.array(detail["camera_matrix"], np.float64)
        distortion = np.array(detail["distortion"], np.float64)
        matrix = np.array(detail["corrected_to_physical"], np.float64)
        inverse = np.array(detail["physical_to_corrected"], np.float64)
        if (
            k.shape != (3, 3)
            or matrix.shape != (3, 3)
            or inverse.shape != (3, 3)
            or abs(np.linalg.det(k)) < 1e-12
            or not np.allclose(matrix @ inverse, np.eye(3), atol=1e-6)
            or distortion.size not in {4, 5, 8, 12, 14}
            or not all(np.isfinite(v).all() for v in [k, distortion, matrix, inverse])
        ):
            raise ValueError("Invalid lens mapping")
    except (KeyError, ValueError, TypeError) as error:
        raise ServiceError(
            "visualization.mapping_invalid", "Invalid nonlinear lens mapping"
        ) from error

    chain = _lens_chain(mapping, k, distortion, matrix, inverse, forward)

    def transform(points: np.ndarray) -> np.ndarray:
        # The lens model, its direction and the projective plane step are owned
        # by the shared mapping layer; only the frame/unit choice stays here.
        mapped = map_sample_points(np.asarray(points, np.float64), chain)
        if not mapped.valid:
            raise ServiceError(
                "visualization.point_at_infinity", "Curve crosses the projective horizon"
            )
        return np.asarray(mapped.value, np.float64)

    return transform, None, unit


def map_geometry(
    shape: str,
    geometry: dict,
    mapping: CoordinateMapping,
    frame: str,
    destination: str,
    curve_samples: int,
) -> tuple:
    """Keep analytic geometry only when its mapping preserves that geometry.

    Args:
        shape: Public primitive geometry type.
        geometry: Actual unrounded primitive coordinates.
        mapping: Explicit forward mapping and its inverse or lens model.
        frame: Declared observation coordinate frame.
        destination: Requested display coordinate frame.
        curve_samples: Bounded sample count for curved geometry.

    Returns:
        Primitive type, mapped coordinates and destination units.
    """
    transform, matrix, unit = mapping_function(mapping, frame, destination)

    def point(value: dict) -> dict:
        mapped = transform(np.array([[value["x"], value["y"]]], np.float64))[0]
        if not np.isfinite(mapped).all():
            raise ServiceError("visualization.nonfinite", "Mapped geometry is not finite")
        return {"x": float(mapped[0]), "y": float(mapped[1])}

    if shape == "point":
        return shape, point(geometry), unit
    if shape == "annotation":
        return shape, geometry, unit
    if shape == "line" and matrix is not None:
        coefficients = np.linalg.inv(matrix).T @ [geometry["a"], geometry["b"], geometry["c"]]
        norm = np.linalg.norm(coefficients[:2])
        if norm <= 1e-12:
            raise ServiceError("visualization.line_at_infinity", "Mapped line lies at infinity")
        coefficients /= norm
        value = dict(zip(["a", "b", "c"], map(float, coefficients)))
        for key in ["start", "end"]:
            value[key] = point(geometry[key]) if geometry.get(key) else None
        return shape, value, unit
    if shape == "circle":
        center, radius = geometry["center"], geometry["radius"]
        if matrix is not None:
            gram = matrix[:2, :2].T @ matrix[:2, :2]
            if np.allclose(matrix[2], [0, 0, 1], atol=1e-10) and np.allclose(
                gram, np.eye(2) * gram[0, 0], atol=1e-10
            ):
                return (
                    shape,
                    {"center": point(center), "radius": radius * float(np.sqrt(gram[0, 0]))},
                    unit,
                )
        angles = np.linspace(0, 2 * np.pi, curve_samples, endpoint=False)
        samples = np.column_stack(
            [center["x"] + radius * np.cos(angles), center["y"] + radius * np.sin(angles)]
        )
        if matrix is not None:
            # A closed projective curve is invalid if its continuous circumference hits infinity.
            normal = matrix[2]
            offset = normal[0] * center["x"] + normal[1] * center["y"] + normal[2]
            if abs(offset) <= radius * np.linalg.norm(normal[:2]) + 1e-12:
                raise ServiceError(
                    "visualization.point_at_infinity", "Circle crosses the projective horizon"
                )
        points = transform(samples)
        if not np.isfinite(points).all():
            raise ServiceError("visualization.nonfinite", "Mapped curve is not finite")
        values = [{"x": float(x), "y": float(y)} for x, y in points]
        return "polyline", {"points": values, "closed": True, "source_type": "circle"}, unit
    if shape == "line":
        if not geometry.get("start") or not geometry.get("end"):
            raise ServiceError(
                "visualization.line_samples_required", "Nonlinear line rendering requires endpoints"
            )
        start, end = geometry["start"], geometry["end"]
        samples = np.column_stack(
            [
                np.linspace(start["x"], end["x"], curve_samples),
                np.linspace(start["y"], end["y"], curve_samples),
            ]
        )
        values = [{"x": float(x), "y": float(y)} for x, y in transform(samples)]
        return "polyline", {"points": values, "closed": False, "source_type": "line"}, unit
    return (
        "polyline",
        {"points": [point(p) for p in geometry["points"]], "closed": geometry.get("closed", True)},
        unit,
    )
