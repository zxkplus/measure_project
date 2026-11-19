"""Framework-free computation contracts shared by every API generation.

This module owns the value types that cross module boundaries: coordinate
spaces, observations, geometry payloads and explicit 2-D mappings.  It holds
no business algorithm and deliberately imports neither Pydantic nor any HTTP
framework, so ``measure/`` can be called from the GUI, the v1 session server,
the v2 service and the recipe runner alike.

Conventions (see ``measure_api/v2/SHARED_COMPUTATION_IMPLEMENTATION.md`` §2):

* points are ``(x, y) = (col, row)``; arrays are ``(N, 2)`` ``float64``;
* lines are normalised ``a*x + b*y + c = 0`` with ``hypot(a, b) == 1`` and a
  deterministic sign (``a > 0``, or ``b > 0`` when ``a == 0``);
* image angles are radians from ``atan2(dy, dx)`` with x right and y down;
* mappings are column-vector ``p_destination ~ H @ p_source`` and always carry
  an engine-computed inverse.

Failures are split in two: malformed input raises
:class:`ComputationContractError`, while a well-formed input that cannot yield
a trustworthy result returns an invalid :class:`Observation` with a stable
``reason`` code.  Neither path logs, writes files nor touches global state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Generic, Mapping, Optional, Tuple, TypeVar

import numpy as np

Array = np.ndarray
XY = Tuple[float, float]
T = TypeVar("T")

#: Tolerance used by the line/endpoint consistency checks.
ENDPOINT_TOLERANCE = 1e-7
#: A mapping is singular when the normalised extreme singular values collapse.
SINGULAR_RATIO = 1e-12
#: Similarity matrices must satisfy ``A.T @ A == s**2 * I`` within this bound.
SIMILARITY_RTOL = 1e-9
SIMILARITY_ATOL = 1e-12

#: Coefficients below this magnitude are treated as zero when choosing the sign.
SIGN_TOLERANCE = 1e-12

VALID_UNITS = ("px", "mm", "cm", "m", "um")
PHYSICAL_UNITS = ("mm", "cm", "m", "um")


class ComputationContractError(ValueError):
    """A malformed or self-contradictory computation input.

    The error carries a stable machine code and an optional dotted field path.
    It intentionally has no HTTP status: adapters decide how to expose it.
    """

    def __init__(self, code: str, message: str, *, field: Optional[str] = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.field = field

    def __str__(self) -> str:
        if self.field:
            return f"{self.code}: {self.message} (field {self.field})"
        return f"{self.code}: {self.message}"


def contract_error(code: str, message: str, field: Optional[str] = None) -> ComputationContractError:
    """Build a contract error uniformly.

    Args:
        code: Stable machine-readable code such as ``contract.nonfinite``.
        message: Human-readable explanation.
        field: Optional dotted path of the offending field.

    Returns:
        The constructed error, ready to raise.
    """
    return ComputationContractError(code, message, field=field)


def require_finite(value: Any, field_name: str = "value") -> None:
    """Reject non-finite scalars, sequences and numeric arrays.

    Args:
        value: Scalar, tuple/list of scalars or numeric ndarray.
        field_name: Dotted path used in the raised error.

    Raises:
        ComputationContractError: If a numeric component is NaN or infinite, or
            if an array is not numeric.
    """
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)):
        return
    if isinstance(value, (int, float, np.integer, np.floating)):
        if not math.isfinite(float(value)):
            raise contract_error("contract.nonfinite", f"{field_name} is not finite", field_name)
        return
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "iuf":
            raise contract_error(
                "contract.invalid_shape", f"{field_name} must be a numeric array", field_name,
            )
        if not bool(np.isfinite(value).all()):
            raise contract_error("contract.nonfinite", f"{field_name} contains non-finite values",
                                 field_name)
        return
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            require_finite(item, f"{field_name}[{index}]")
        return
    validate = getattr(value, "validate", None)
    if callable(validate):
        validate()


def require_finite_xy(point: Any, field_name: str = "point") -> XY:
    """Validate and return a finite ``(x, y)`` pair.

    Args:
        point: Two-item sequence of numbers.
        field_name: Dotted path used in the raised error.

    Returns:
        The pair as a tuple of floats.

    Raises:
        ComputationContractError: If the value is not a finite two-item pair.
    """
    if not isinstance(point, (tuple, list, np.ndarray)) or len(point) != 2:
        raise contract_error("contract.invalid_shape", f"{field_name} must be an (x, y) pair",
                             field_name)
    x, y = point
    require_finite(x, f"{field_name}.x")
    require_finite(y, f"{field_name}.y")
    return (float(x), float(y))


def require_xy_array(points: Any, field_name: str = "points") -> Array:
    """Validate a finite ``(N, 2)`` point array.

    Args:
        points: Array-like with shape ``(N, 2)``.
        field_name: Dotted path used in the raised error.

    Returns:
        A ``float64`` copy owned by this call, preserving input order.

    Raises:
        ComputationContractError: If the shape is not ``(N, 2)`` or values are
            not finite.
    """
    array = np.asarray(points)
    if array.ndim != 2 or array.shape[1] != 2:
        raise contract_error("contract.invalid_shape", f"{field_name} must have shape (N, 2)",
                             field_name)
    if array.dtype.kind not in "iuf":
        raise contract_error("contract.invalid_shape", f"{field_name} must be numeric", field_name)
    values = np.array(array, dtype=np.float64, copy=True)
    if not bool(np.isfinite(values).all()):
        raise contract_error("contract.nonfinite", f"{field_name} contains non-finite values",
                             field_name)
    return values


def require_space_match(left: "Space", right: "Space", field_name: str = "space") -> None:
    """Require two observations to share frame and unit.

    The check runs before validity is inspected: an invalid observation cannot
    hide a frame mismatch.

    Args:
        left: First space.
        right: Second space.
        field_name: Dotted path used in the raised error.

    Raises:
        ComputationContractError: If frame or unit differs.
    """
    if left.frame != right.frame or left.unit != right.unit:
        raise contract_error(
            "contract.space_mismatch",
            f"incompatible spaces: {left.frame}/{left.unit} vs {right.frame}/{right.unit}",
            field_name,
        )


@dataclass(frozen=True)
class Space:
    """A named coordinate frame with its length unit.

    The identity is supplied by the caller; the library never guesses a camera
    or target frame from the data.
    """

    frame: str
    unit: str

    def __post_init__(self) -> None:
        if not isinstance(self.frame, str) or not self.frame.strip():
            raise contract_error("contract.invalid_parameter", "frame must be a non-empty string",
                                 "space.frame")
        if not isinstance(self.unit, str) or not self.unit.strip():
            raise contract_error("contract.invalid_parameter", "unit must be a non-empty string",
                                 "space.unit")


@dataclass(frozen=True)
class PixelWindow:
    """An integer half-open pixel window sliced as ``[y:y+h, x:x+w]``."""

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        for name in ("x", "y", "width", "height"):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
                raise contract_error("contract.invalid_parameter", f"{name} must be an integer",
                                     f"window.{name}")
        if self.x < 0 or self.y < 0:
            raise contract_error("contract.invalid_parameter", "window origin must be non-negative",
                                 "window.origin")
        if self.width <= 0 or self.height <= 0:
            raise contract_error("contract.invalid_parameter", "window size must be positive",
                                 "window.size")

    @property
    def bounds(self) -> Tuple[int, int, int, int]:
        """Return ``(x0, y0, x1, y1)`` as half-open bounds."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)


@dataclass(frozen=True)
class Observation(Generic[T]):
    """One typed result whose validity is independent of execution success.

    ``valid=True`` requires a finite payload and no reason; ``valid=False``
    requires no payload and a stable non-empty reason code.  Diagnostics may
    still describe what was sampled before the failure.
    """

    space: Space
    valid: bool
    value: Optional[T]
    reason: Optional[str] = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.space, Space):
            raise contract_error("contract.invalid_parameter", "observation requires a Space",
                                 "observation.space")
        if not isinstance(self.valid, (bool, np.bool_)):
            raise contract_error("contract.invalid_parameter", "valid must be a boolean",
                                 "observation.valid")
        if self.valid:
            if self.value is None:
                raise contract_error("contract.invalid_observation",
                                     "valid observation requires a payload", "observation.value")
            if self.reason is not None:
                raise contract_error("contract.invalid_observation",
                                     "valid observation cannot carry a reason",
                                     "observation.reason")
            require_finite(self.value, "observation.value")
        else:
            if self.value is not None:
                raise contract_error("contract.invalid_observation",
                                     "invalid observation cannot carry a payload",
                                     "observation.value")
            if not isinstance(self.reason, str) or not self.reason.strip():
                raise contract_error("contract.invalid_observation",
                                     "invalid observation requires a reason code",
                                     "observation.reason")

    @property
    def frame(self) -> str:
        """Return the frame name of this observation."""
        return self.space.frame

    @property
    def unit(self) -> str:
        """Return the length unit of this observation."""
        return self.space.unit


def observed(
    space: Space, value: T, diagnostics: Optional[Mapping[str, Any]] = None,
) -> Observation[T]:
    """Build a valid observation.

    Args:
        space: Frame and unit of the payload.
        value: Finite payload.
        diagnostics: Optional sampling metadata kept alongside the value.

    Returns:
        A valid :class:`Observation`.
    """
    return Observation(space=space, valid=True, value=value, diagnostics=diagnostics or {})


def invalid(
    space: Space, reason: str, diagnostics: Optional[Mapping[str, Any]] = None,
) -> Observation[Any]:
    """Build an invalid observation with a stable reason code.

    Args:
        space: Frame and unit the observation would have used.
        reason: Stable machine-readable failure code.
        diagnostics: Optional sampling metadata kept from before the failure.

    Returns:
        An invalid :class:`Observation` carrying no payload.
    """
    return Observation(space=space, valid=False, value=None, reason=reason,
                       diagnostics=diagnostics or {})


def canonical_line_coefficients(a: float, b: float, c: float) -> Tuple[float, float, float]:
    """Return a unit-normal line with a deterministic sign.

    A coefficient that is zero up to floating-point noise counts as zero, so a
    fitted horizontal line always normalises to ``(0, +1, c)`` instead of
    flipping between ``(eps, -1, ...)`` and ``(-eps, +1, ...)``.

    Args:
        a: Normal coefficient on x.
        b: Normal coefficient on y.
        c: Constant term.

    Returns:
        Coefficients with ``hypot(a, b) == 1`` and ``a > 0`` (or ``b > 0`` when
        ``a`` is zero within :data:`SIGN_TOLERANCE`).
    """
    norm = math.hypot(a, b)
    if norm <= 0.0 or not math.isfinite(norm):
        raise contract_error("contract.degenerate_line", "line normal must be non-zero",
                             "line.coefficients")
    a, b, c = a / norm, b / norm, c / norm
    if a < -SIGN_TOLERANCE or (abs(a) <= SIGN_TOLERANCE and b < 0.0):
        a, b, c = -a, -b, -c
    if abs(a) <= SIGN_TOLERANCE:
        a = 0.0
    return a, b, c


@dataclass(frozen=True)
class LineValue:
    """A normalised infinite line with optional consistent display endpoints."""

    coefficients: Tuple[float, float, float]
    start: Optional[XY] = None
    end: Optional[XY] = None

    def __post_init__(self) -> None:
        coefficients = tuple(self.coefficients)
        if len(coefficients) != 3:
            raise contract_error("contract.invalid_shape",
                                 "line coefficients must have three items",
                                 "line.coefficients")
        a, b, c = (float(value) for value in coefficients)
        require_finite((a, b, c), "line.coefficients")
        a, b, c = canonical_line_coefficients(a, b, c)
        if (self.start is None) != (self.end is None):
            raise contract_error("contract.invalid_parameter",
                                 "line endpoints must be provided together", "line.endpoints")
        if self.start is not None:
            start = require_finite_xy(self.start, "line.start")
            end = require_finite_xy(self.end, "line.end")
            for name, point in (("start", start), ("end", end)):
                tolerance = ENDPOINT_TOLERANCE * max(1.0, math.hypot(point[0], point[1]))
                residual = abs(a * point[0] + b * point[1] + c)
                if residual > tolerance:
                    raise contract_error(
                        "contract.inconsistent_endpoint",
                        f"line {name} does not satisfy the line equation",
                        f"line.{name}",
                    )
            object.__setattr__(self, "start", start)
            object.__setattr__(self, "end", end)
        object.__setattr__(self, "coefficients", (a, b, c))

    @property
    def angle_rad(self) -> float:
        """Return the undirected direction angle in ``[0, pi)``."""
        a, b = self.coefficients[0], self.coefficients[1]
        angle = math.atan2(a, -b)
        if angle < 0.0:
            angle += math.pi
        if angle >= math.pi:
            angle -= math.pi
        return angle

    def distance_to(self, point: XY) -> float:
        """Return the perpendicular distance from a point to this line.

        Args:
            point: Finite ``(x, y)`` pair in the same space as the line.

        Returns:
            Absolute perpendicular distance in the line's unit.
        """
        x, y = require_finite_xy(point)
        a, b, c = self.coefficients
        return abs(a * x + b * y + c)

    def validate(self) -> None:
        """Re-run the construction checks (used by nested payload validation)."""
        LineValue(self.coefficients, self.start, self.end)


@dataclass(frozen=True)
class CircleValue:
    """A circle with a finite centre and a strictly positive radius."""

    center: XY
    radius: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", require_finite_xy(self.center, "circle.center"))
        require_finite(self.radius, "circle.radius")
        radius = float(self.radius)
        if radius <= 0.0:
            raise contract_error("contract.invalid_parameter",
                                 "circle radius must be positive", "circle.radius")
        object.__setattr__(self, "radius", radius)

    def validate(self) -> None:
        """Re-run the construction checks (used by nested payload validation)."""
        CircleValue(self.center, self.radius)


@dataclass(frozen=True)
class RotatedBox:
    """A continuous rotated rectangle in the local frame of its centre."""

    center: XY
    size_wh: Tuple[float, float]
    angle_rad: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", require_finite_xy(self.center, "box.center"))
        width, height = self.size_wh
        require_finite((width, height), "box.size_wh")
        if float(width) <= 0.0 or float(height) <= 0.0:
            raise contract_error("contract.invalid_parameter", "box size must be positive",
                                 "box.size_wh")
        require_finite(self.angle_rad, "box.angle_rad")
        object.__setattr__(self, "size_wh", (float(width), float(height)))
        object.__setattr__(self, "angle_rad", float(self.angle_rad))

    def validate(self) -> None:
        """Re-run the construction checks (used by nested payload validation)."""
        RotatedBox(self.center, self.size_wh, self.angle_rad)


@dataclass(frozen=True)
class PolylineValue:
    """An ordered point sequence, optionally closed."""

    points_xy: Array
    closed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "points_xy", require_xy_array(self.points_xy, "polyline.points_xy"))
        object.__setattr__(self, "closed", bool(self.closed))

    def validate(self) -> None:
        """Re-run the construction checks (used by nested payload validation)."""
        PolylineValue(self.points_xy, self.closed)


@dataclass(frozen=True)
class Mapping2D:
    """An explicit source-to-destination 2-D mapping with an engine inverse."""

    source: Space
    destination: Space
    kind: str
    matrix: Array
    inverse_matrix: Array

    def __post_init__(self) -> None:
        if self.kind not in ("similarity", "affine", "projective"):
            raise contract_error("contract.invalid_parameter", f"unknown mapping kind {self.kind!r}",
                                 "mapping.kind")
        if not isinstance(self.source, Space) or not isinstance(self.destination, Space):
            raise contract_error("contract.invalid_parameter", "mapping requires two Spaces",
                                 "mapping.space")
        for name in ("matrix", "inverse_matrix"):
            value = np.asarray(getattr(self, name))
            if value.shape != (3, 3):
                raise contract_error("contract.invalid_shape", f"{name} must be 3x3",
                                     f"mapping.{name}")
            if value.dtype.kind not in "iuf" or not bool(np.isfinite(value).all()):
                raise contract_error("contract.nonfinite", f"{name} must be finite numeric",
                                     f"mapping.{name}")
            object.__setattr__(self, name, np.array(value, dtype=np.float64, copy=True))

    @property
    def forward(self) -> Array:
        """Return the source-to-destination matrix."""
        return self.matrix

    def apply(self, points: Any) -> Array:
        """Map ``(N, 2)`` points from source to destination.

        Args:
            points: Array-like of shape ``(N, 2)``.

        Returns:
            Mapped points as ``(N, 2)`` ``float64``.
        """
        values = require_xy_array(points, "mapping.points")
        homogeneous = np.column_stack([values, np.ones(len(values))])
        mapped = homogeneous @ self.matrix.T
        return mapped[:, :2] / mapped[:, 2:3]

    def apply_inverse(self, points: Any) -> Array:
        """Map ``(N, 2)`` points from destination back to source.

        Args:
            points: Array-like of shape ``(N, 2)``.

        Returns:
            Mapped points as ``(N, 2)`` ``float64``.
        """
        values = require_xy_array(points, "mapping.points")
        homogeneous = np.column_stack([values, np.ones(len(values))])
        mapped = homogeneous @ self.inverse_matrix.T
        return mapped[:, :2] / mapped[:, 2:3]

    def validate(self) -> None:
        """Re-run the construction checks (used by nested payload validation)."""
        validate_mapping_matrix(self.matrix, self.kind)


def _normalised_singular_range(matrix: Array) -> Tuple[float, float]:
    scale = float(np.max(np.abs(matrix)))
    if scale <= 0.0 or not math.isfinite(scale):
        raise contract_error("contract.degenerate_mapping", "mapping matrix is degenerate",
                             "mapping.matrix")
    singular = np.linalg.svd(matrix / scale, compute_uv=False)
    return float(singular.min()), float(singular.max())


def validate_mapping_matrix(matrix: Array, kind: str) -> None:
    """Validate a 3x3 matrix against the claimed mapping kind.

    Args:
        matrix: Candidate 3x3 matrix.
        kind: ``similarity``, ``affine`` or ``projective``.

    Raises:
        ComputationContractError: On wrong shape, non-finite values, singularity
            or a claim the matrix does not satisfy.
    """
    values = np.asarray(matrix)
    if values.shape != (3, 3):
        raise contract_error("contract.invalid_shape", "mapping matrix must be 3x3",
                             "mapping.matrix")
    if values.dtype.kind not in "iuf" or not bool(np.isfinite(values).all()):
        raise contract_error("contract.nonfinite", "mapping matrix must be finite numeric",
                             "mapping.matrix")
    smallest, largest = _normalised_singular_range(values)
    if smallest <= SINGULAR_RATIO or largest <= 0.0:
        raise contract_error("contract.degenerate_mapping", "mapping matrix is singular",
                             "mapping.matrix")
    last_row = values[2] / max(abs(float(values[2, 2])), 1e-30)
    if kind in ("affine", "similarity"):
        if not np.allclose(last_row, [0.0, 0.0, 1.0], rtol=1e-9, atol=1e-9):
            raise contract_error("contract.false_mapping_kind",
                                 f"matrix is not an {kind} mapping", "mapping.kind")
    if kind == "similarity":
        linear = values[:2, :2]
        scale_squared = float(np.mean(np.sum(linear ** 2, axis=0)))
        if scale_squared <= 0.0:
            raise contract_error("contract.degenerate_mapping", "similarity scale must be positive",
                                 "mapping.matrix")
        expected = scale_squared * np.eye(2)
        if not np.allclose(linear.T @ linear, expected, rtol=SIMILARITY_RTOL, atol=SIMILARITY_ATOL):
            raise contract_error("contract.false_mapping_kind",
                                 "matrix is not a similarity mapping", "mapping.kind")
        if float(np.linalg.det(linear)) <= 0.0:
            raise contract_error("contract.false_mapping_kind",
                                 "similarity mapping cannot reflect", "mapping.kind")


def make_mapping(matrix: Array, *, source: Space, destination: Space, kind: str) -> Mapping2D:
    """Build a :class:`Mapping2D` with an engine-computed inverse.

    Args:
        matrix: 3x3 source-to-destination matrix (column-vector convention).
        source: Space of the mapped input.
        destination: Space of the mapped output.
        kind: Declared ``similarity``, ``affine`` or ``projective``.

    Returns:
        A validated mapping whose inverse is computed here, never supplied by
        the caller.

    Raises:
        ComputationContractError: If the matrix is malformed, singular, or does
            not satisfy the declared kind.
    """
    validate_mapping_matrix(matrix, kind)
    values = np.array(matrix, dtype=np.float64, copy=True)
    inverse = np.linalg.inv(values)
    if not bool(np.isfinite(inverse).all()):
        raise contract_error("contract.degenerate_mapping", "mapping inverse is not finite",
                             "mapping.inverse_matrix")
    identity = values @ inverse
    if not np.allclose(identity, np.eye(3), rtol=1e-9, atol=1e-12):
        raise contract_error("contract.degenerate_mapping", "mapping inverse is inconsistent",
                             "mapping.inverse_matrix")
    return Mapping2D(source=source, destination=destination, kind=kind,
                     matrix=values, inverse_matrix=inverse)