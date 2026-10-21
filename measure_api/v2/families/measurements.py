"""Independent image probes with explicit pixel coordinates and observation validity."""

import io
from typing import List, Literal, Optional

import cv2
import numpy as np
from pydantic import Field, model_validator
from scipy.ndimage import gaussian_filter1d

from measure.measure1D import Halcon1DMeasure
from measure_api.v2.atomic_types import FitDiagnostics, Line, Observation, PositionedPoint, Scalar
from measure_api.v2.errors import ServiceError
from measure_api.v2.families.geometry import (
    CircleFitResult, LineFitResult, PointFitRequest, fit_circle, fit_line,
)
from measure_api.v2.files import ArtifactWriter
from measure_api.v2.models import (
    CalculationRequest, FileDescriptor, FileReference, OutputSpec, Point, StrictModel,
)
from measure_api.v2.registry import CalculationContext, Operation


class Probe(StrictModel):
    """Search half-length along clockwise orientation; averaging perpendicular to it."""

    center: Point
    search_half_length: float = Field(ge=2, le=16384)
    average_half_width: float = Field(ge=0.5, le=16384)
    angle_degrees: float = 0
    interpolation: Literal['nearest', 'linear', 'cubic'] = 'linear'
    border_policy: Literal['reject', 'replicate'] = 'reject'


class ProbeRequest(CalculationRequest):
    """Inspect one image probe without persisting measurement configuration."""

    image: FileReference
    frame: str = Field(default='source', min_length=1)
    probe: Probe
    sigma: float = Field(default=1, ge=0, le=100)
    threshold: float = Field(default=30, ge=0, le=255)
    threshold_unit: Literal['gray', 'fraction'] = 'gray'
    polarity: Literal['positive', 'negative', 'all'] = 'all'
    selection: Literal['first', 'last', 'all'] = 'all'

    def gray_threshold(self) -> float:
        """Convert the explicit threshold unit into gray-level derivative amplitude.

        Returns:
            Derivative threshold in gray levels after explicit normalized-unit conversion.
        """
        if self.threshold_unit == 'fraction':
            if self.threshold > 1:
                raise ServiceError(
                    'measurement.threshold_invalid', 'Fraction threshold exceeds one',
                )
            return self.threshold * 255
        return self.threshold


class Edge(StrictModel):
    """Selected subpixel boundary and signed derivative amplitude."""

    point: Point
    amplitude: float
    frame: str = 'source'
    unit: str = 'px'


class EdgeResult(Observation):
    point: Optional[PositionedPoint] = None
    edges: List[Edge] = Field(default_factory=list)
    distances: List[float] = Field(default_factory=list)


class ProfileRequest(ProbeRequest):
    """Optionally exchange large numeric curves through shared output files."""

    output: Optional[OutputSpec] = None
    inline_samples: bool = True

    @model_validator(mode='after')
    def require_curve_destination(self) -> 'ProfileRequest':
        """Require a file destination when the caller suppresses inline curves.

        Returns:
            This profile request after authorizing file-backed curve output.
        """
        if not self.inline_samples and self.output is None:
            raise ValueError('Noninline curves require an output directory')
        return self


class ProfileResult(Observation):
    gray: List[float]
    smoothed: List[float]
    gradient: List[float]
    sample_points: List[Point]
    threshold_gray: float
    artifacts: List[FileDescriptor] = Field(default_factory=list)
    manifest: Optional[FileDescriptor] = None


def make_probe(probe: Probe, image: np.ndarray, context: CalculationContext) -> Halcon1DMeasure:
    """Authorize sample area and adapt clockwise xy geometry to the numeric core.

    Args:
        probe: Clockwise source-image probe geometry and requested sampling dimensions.
        image: Decoded source pixels; gray arrays remain in their original sample frame.
        context: Isolated operation identity, resource limits and verified shared-file
            access.

    Returns:
        Authorized numeric-core probe configured in row/column coordinates.
    """
    if int(2 * probe.search_half_length) * int(2 * probe.average_half_width) > (
        context.settings.max_image_pixels
    ):
        raise ServiceError('image.too_large', 'Probe exceeds configured pixel limit')
    angle = np.deg2rad(probe.angle_degrees)
    instrument = Halcon1DMeasure(
        probe.center.y, probe.center.x, angle,
        probe.search_half_length, probe.average_half_width, probe.interpolation,
    )
    if probe.border_policy == 'reject':
        width, height = int(2 * probe.search_half_length), int(2 * probe.average_half_width)
        corners = np.array([[0, 0, 1], [width - 1, 0, 1],
                            [0, height - 1, 1], [width - 1, height - 1, 1]])
        original = corners @ instrument.inverse_matrix.T
        if (original[:, 0].min() < -1e-7 or original[:, 1].min() < -1e-7
                or original[:, 0].max() > image.shape[1] - 1 + 1e-7
                or original[:, 1].max() > image.shape[0] - 1 + 1e-7):
            raise ServiceError('image.roi_outside', 'Probe includes pixels outside source image')
    return instrument


def read_gray(request: ProbeRequest, context: CalculationContext) -> np.ndarray:
    """Read verified source pixels without changing their measurement frame.

    Args:
        request: Typed inspect one image probe without persisting measurement configuration.
        context: Isolated operation identity, resource limits and verified shared-file
            access.

    Returns:
        Verified 8-bit gray pixels in the measurement source frame.
    """
    image, _ = context.files.read_image(request.image)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def extract_profile(request: ProfileRequest, context: CalculationContext) -> ProfileResult:
    """Expose gray profile, smoothing, derivative and source sample coordinates.

    Args:
        request: Typed optionally exchange large numeric curves through shared output files.
        context: Isolated operation identity, resource limits and verified shared-file
            access.

    Returns:
        Gray/smoothed/derivative curves and source sample coordinates or their numeric
        files.
    """
    threshold = request.gray_threshold()
    image = read_gray(request, context)
    probe = make_probe(request.probe, image, context)
    gray = probe.get_profile(image).astype(float)
    smoothed = gaussian_filter1d(gray, request.sigma) if request.sigma > 0 else gray
    points = np.column_stack([np.arange(len(gray)),
                              np.full(len(gray), request.probe.average_half_width),
                              np.ones(len(gray))]) @ probe.inverse_matrix.T
    gradient = np.gradient(smoothed)
    artifacts, manifest = [], None
    if request.output is not None:
        payload = io.BytesIO()
        np.savez_compressed(payload, gray=gray, smoothed=smoothed,
                            gradient=gradient, sample_points=points)
        artifacts, manifest = ArtifactWriter(
            context.files, request.output, context.operation_id,
        ).commit([('profiles.npz', 'curves', 'npz', payload.getvalue())])
    return ProfileResult(
        valid=True, frame=request.frame,
        gray=gray.tolist() if request.inline_samples else [],
        smoothed=smoothed.tolist() if request.inline_samples else [],
        gradient=gradient.tolist() if request.inline_samples else [],
        threshold_gray=threshold,
        sample_points=[Point(x=x, y=y) for x, y in points] if request.inline_samples else [],
        artifacts=artifacts, manifest=manifest,
    )


def measure_edges(request: ProbeRequest, context: CalculationContext) -> EdgeResult:
    """Detect signed subpixel boundaries and select after edge detection.

    Args:
        request: Typed inspect one image probe without persisting measurement configuration.
        context: Isolated operation identity, resource limits and verified shared-file
            access.

    Returns:
        Selected subpixel edge observations and consecutive edge distances.
    """
    image = read_gray(request, context)
    probe = make_probe(request.probe, image, context)
    rows, cols, amplitudes, distances = probe.measure_pos(
        image, request.sigma, request.gray_threshold(), request.polarity, request.selection,
    )
    edges = [Edge(point=Point(x=x, y=y), amplitude=a, frame=request.frame)
             for y, x, a in zip(rows, cols, amplitudes)]
    return EdgeResult(valid=bool(edges), reason=None if edges else 'no_edges',
                      frame=request.frame, edges=edges, distances=list(distances),
                      point=PositionedPoint(valid=True, point=edges[0].point, frame=request.frame)
                      if len(edges) == 1 else None)


OPERATIONS = [
    Operation('/edge-profiles', ProfileRequest, ProfileResult, extract_profile,
              'Inspect averaged gray and gradient profiles', 'measurements',
              {'gray': 'number_array', 'smoothed': 'number_array',
               'gradient': 'number_array', 'sample_points': 'xy_array',
               'artifacts': 'file_array'}),
    Operation('/measurements/edge-points', ProbeRequest, EdgeResult, measure_edges,
              'Detect selectable subpixel edges', 'measurements',
              {'edges': 'edge_array', 'point': 'point'}),
]


class PairRequest(ProbeRequest):
    selection: Literal['first', 'last', 'all', 'endpoints'] = 'all'


class EdgePair(StrictModel):
    edge_a: Edge
    edge_b: Edge
    center: PositionedPoint
    width: Scalar


class PairResult(Observation):
    pairs: List[EdgePair] = Field(default_factory=list)
    edge_a: Optional[PositionedPoint] = None
    edge_b: Optional[PositionedPoint] = None
    center: Optional[PositionedPoint] = None
    width: Optional[Scalar] = None
    center_distances: List[float] = Field(default_factory=list)


def measure_pairs(request: PairRequest, context: CalculationContext) -> PairResult:
    """Pair complete detected boundaries before first/last selection.

    Args:
        request: Typed source image, edge probe, polarity and complete-pair selection
            policy.
        context: Isolated operation identity, resource limits and verified shared-file
            access.

    Returns:
        Complete subpixel edge pairs, centers, widths and named single-pair output ports.
    """
    image = read_gray(request, context)
    probe = make_probe(request.probe, image, context)
    data = probe.measure_pairs(image, request.sigma, request.gray_threshold(),
                               request.polarity, request.selection)
    pairs = []
    for ya, xa, aa, yb, xb, ab, yc, xc, width in zip(*data[:9]):
        pairs.append(EdgePair(
            edge_a=Edge(point=Point(x=xa, y=ya), amplitude=aa, frame=request.frame),
            edge_b=Edge(point=Point(x=xb, y=yb), amplitude=ab, frame=request.frame),
            center=PositionedPoint(valid=True, point=Point(x=xc, y=yc), frame=request.frame),
            width=Scalar(valid=True, value=width, frame=request.frame),
        ))
    single = pairs[0] if len(pairs) == 1 else None
    return PairResult(
        valid=bool(pairs), reason=None if pairs else 'no_pairs', frame=request.frame,
        pairs=pairs, edge_a=PositionedPoint(valid=True, point=single.edge_a.point,
                                           frame=request.frame) if single else None,
        edge_b=PositionedPoint(valid=True, point=single.edge_b.point, frame=request.frame)
        if single else None, center=single.center if single else None,
        width=single.width if single else None,
        center_distances=list(data[9]) if len(pairs) > 1 else [],
    )


OPERATIONS.append(Operation(
    '/measurements/edge-pairs', PairRequest, PairResult, measure_pairs,
    'Detect typed paired boundaries, center and width', 'measurements',
    {'pairs': 'edge_pair_array', 'edge_a': 'point', 'edge_b': 'point',
     'center': 'point', 'width': 'scalar'},
))


class ImageFitRequest(CalculationRequest):
    """Common image-fitting controls, with explicit search and averaging dimensions."""

    image: FileReference
    frame: str = Field(default='source', min_length=1)
    search_half_length: float = Field(default=15, ge=2, le=16384)
    average_half_width: float = Field(default=5, ge=0.5, le=16384)
    probe_count: int = Field(default=20, ge=2, le=10000)
    sigma: float = Field(default=1, ge=0, le=100)
    threshold: float = Field(default=30, ge=0, le=255)
    threshold_unit: Literal['gray', 'fraction'] = 'gray'
    polarity: Literal['positive', 'negative', 'all'] = 'all'
    selection: Literal['nearest', 'first', 'last', 'all'] = 'nearest'
    interpolation: Literal['nearest', 'linear', 'cubic'] = 'linear'
    border_policy: Literal['reject', 'replicate'] = 'reject'


class ImageLineRequest(ImageFitRequest):
    start: Point
    end: Point


class ImageCircleRequest(ImageFitRequest):
    center: Point
    radius: float = Field(gt=0)
    radius_min: Optional[float] = Field(default=None, gt=0)
    radius_max: Optional[float] = Field(default=None, gt=0)
    start_angle_degrees: float = 0
    end_angle_degrees: float = 360
    probe_count: int = Field(default=24, ge=3, le=10000)


class ImageLineResult(LineFitResult):
    edges: List[Edge] = Field(default_factory=list)


class ImageCircleResult(CircleFitResult):
    edges: List[Edge] = Field(default_factory=list)
    start_angle_degrees: float
    end_angle_degrees: float


def _probe_edges(
    request: ImageFitRequest, center: Point, angle: float,
    image: np.ndarray, context: CalculationContext,
) -> List[Edge]:
    probe = Probe(center=center, angle_degrees=angle,
                  search_half_length=request.search_half_length,
                  average_half_width=request.average_half_width,
                  interpolation=request.interpolation, border_policy=request.border_policy)
    instrument = make_probe(probe, image, context)
    threshold = request.threshold
    if request.threshold_unit == 'fraction':
        if threshold > 1:
            raise ServiceError('measurement.threshold_invalid', 'Fraction threshold exceeds one')
        threshold *= 255
    ys, xs, amplitudes, _ = instrument.measure_pos(
        image, request.sigma, threshold, request.polarity, 'all',
    )
    return [Edge(point=Point(x=x, y=y), amplitude=a, frame=request.frame)
            for y, x, a in zip(ys, xs, amplitudes)]


def measure_line(request: ImageLineRequest, context: CalculationContext) -> ImageLineResult:
    """Probe perpendicular to an expected line and fit selected observations in source xy.

    Args:
        request: Typed expected source line and perpendicular probe/fitting parameters.
        context: Isolated operation identity, resource limits and verified shared-file
            access.

    Returns:
        Selected perpendicular probe samples and the resulting source-frame line fit.
    """
    start, end = request.start, request.end
    vector = np.array([end.x - start.x, end.y - start.y])
    if np.linalg.norm(vector) <= 1e-12:
        return ImageLineResult(
            valid=False, reason='coincident_expected_points', frame=request.frame,
            line=Line(valid=False, reason='coincident_expected_points', frame=request.frame),
            diagnostics=FitDiagnostics(sample_count=0, requested_probe_count=request.probe_count),
        )
    angle = np.rad2deg(np.arctan2(vector[1], vector[0])) + 90
    image = read_gray(request, context)
    edges = []
    successful = 0
    for fraction in np.linspace(0, 1, request.probe_count):
        center = Point(x=start.x + fraction * vector[0], y=start.y + fraction * vector[1])
        candidates = _probe_edges(request, center, angle, image, context)
        if candidates:
            successful += 1
            if request.selection == 'nearest':
                candidates = [min(candidates, key=lambda e: (
                    (e.point.x - center.x) ** 2 + (e.point.y - center.y) ** 2))]
            elif request.selection == 'first':
                candidates = candidates[:1]
            elif request.selection == 'last':
                candidates = candidates[-1:]
            edges.extend(candidates)
    fitted = fit_line(
        PointFitRequest(points=[e.point for e in edges], frame=request.frame), context,
    )
    fitted.diagnostics.requested_probe_count = request.probe_count
    fitted.diagnostics.successful_probe_count = successful
    return ImageLineResult(**fitted.model_dump(), edges=edges)


def measure_circle(request: ImageCircleRequest, context: CalculationContext) -> ImageCircleResult:
    """Probe inward around a circle or clockwise arc and filter actual observed radii.

    Args:
        request: Typed expected circle or arc and radial probe/fitting parameters.
        context: Isolated operation identity, resource limits and verified shared-file
            access.

    Returns:
        Selected radial probe samples and the resulting source-frame circle fit.
    """
    if (request.radius_min is not None and request.radius_max is not None
            and request.radius_min > request.radius_max):
        raise ServiceError('measurement.radius_limits_invalid', 'Minimum radius exceeds maximum')
    span = (request.end_angle_degrees - request.start_angle_degrees) % 360
    if span == 0:
        span = 360
    image = read_gray(request, context)
    edges = []
    successful = 0
    angles = request.start_angle_degrees + np.linspace(0, span, request.probe_count, endpoint=False)
    for angle in angles:
        phi = np.deg2rad(angle)
        center = Point(x=request.center.x + request.radius * np.cos(phi),
                       y=request.center.y + request.radius * np.sin(phi))
        candidates = _probe_edges(request, center, angle + 180, image, context)
        candidates = [e for e in candidates if (
            (request.radius_min is None or np.hypot(e.point.x - request.center.x,
                                                    e.point.y - request.center.y)
             >= request.radius_min)
            and (request.radius_max is None or np.hypot(e.point.x - request.center.x,
                                                        e.point.y - request.center.y)
                 <= request.radius_max))]
        if candidates:
            successful += 1
            if request.selection == 'nearest':
                candidates = [min(candidates, key=lambda e: abs(np.hypot(
                    e.point.x - request.center.x, e.point.y - request.center.y) - request.radius))]
            elif request.selection == 'first':
                candidates = candidates[:1]
            elif request.selection == 'last':
                candidates = candidates[-1:]
            edges.extend(candidates)
    fitted = fit_circle(
        PointFitRequest(points=[e.point for e in edges], frame=request.frame), context,
    )
    fitted.diagnostics.requested_probe_count = request.probe_count
    fitted.diagnostics.successful_probe_count = successful
    return ImageCircleResult(**fitted.model_dump(), edges=edges,
                             start_angle_degrees=request.start_angle_degrees,
                             end_angle_degrees=request.start_angle_degrees + span)


OPERATIONS.extend([
    Operation('/measurements/lines', ImageLineRequest, ImageLineResult, measure_line,
              'Probe and fit source-image lines', 'measurements',
              {'line': 'line', 'edges': 'edge_array'}),
    Operation('/measurements/circles', ImageCircleRequest, ImageCircleResult, measure_circle,
              'Probe and fit source-image circles and arcs', 'measurements',
              {'circle': 'circle', 'edges': 'edge_array', 'sampled_diameter_spread': 'number'}),
])
