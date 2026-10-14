"""Explicit similarity alignment and typed geometry frame conversion."""

from __future__ import annotations

from typing import Annotated, List, Literal, Optional, Union

import cv2
import numpy as np
from pydantic import Field, model_validator

from measure_api.v2.errors import ServiceError
from measure_api.v2.models import (
    CalculationRequest, CoordinateMapping, FileDescriptor, FileReference, OutputSpec,
    Point, StrictModel,
)
from measure_api.v2.registry import CalculationContext, Operation
from measure_api.v2.families.templates import coordinate_mapping


class SimilarityEstimateRequest(CalculationRequest):
    """Known xy correspondences with optional robust similarity estimation."""

    source_points: List[Point] = Field(min_length=2, max_length=10000)
    destination_points: List[Point] = Field(min_length=2, max_length=10000)
    source_frame: str = Field(default="source", min_length=1)
    destination_frame: str = Field(default="destination", min_length=1)
    estimate_scale: bool = True
    robust: bool = False
    ransac_threshold: float = Field(default=3, gt=0)
    max_residual: float = Field(default=3, gt=0)
    minimum_inliers: int = Field(default=2, ge=2)

    @model_validator(mode="after")
    def paired_points(self) -> SimilarityEstimateRequest:
        """Ensure each source correspondence has exactly one destination.

        Returns:
            This estimate request after checking paired correspondence counts.
        """
        if len(self.source_points) != len(self.destination_points):
            raise ValueError("Source and destination point counts must agree")
        if self.minimum_inliers > len(self.source_points):
            raise ValueError("Minimum inliers exceeds supplied correspondences")
        return self


class SimilarityEstimateResult(StrictModel):
    """Observation validity and actual fit diagnostics in destination pixels."""

    valid: bool
    reason: Optional[str] = None
    mapping: Optional[CoordinateMapping] = None
    scale: Optional[float] = None
    angle_degrees: Optional[float] = None
    residuals: List[float] = Field(default_factory=list)
    inliers: List[bool] = Field(default_factory=list)
    rms_residual: Optional[float] = None


def _fit_similarity(source: np.ndarray, destination: np.ndarray, scale: bool) -> np.ndarray:
    xmean, ymean = source.mean(axis=0), destination.mean(axis=0)
    x, y = source - xmean, destination - ymean
    variance = np.sum(x*x)
    if variance < 1e-10 or np.sum(y*y) < 1e-10:
        raise ValueError("Degenerate correspondences")
    u, singular, vt = np.linalg.svd(x.T @ y)
    correction = np.diag([1, np.linalg.det(vt.T @ u.T)])
    rotation = vt.T @ correction @ u.T
    factor = np.sum(singular * np.diag(correction)) / variance if scale else 1.0
    if factor <= 1e-10:
        raise ValueError("Degenerate scale")
    matrix = np.eye(3)
    matrix[:2, :2] = factor * rotation
    matrix[:2, 2] = ymean - matrix[:2, :2] @ xmean
    return matrix


def estimate_similarity(
    request: SimilarityEstimateRequest, context: CalculationContext,
) -> SimilarityEstimateResult:
    """Estimate an orientation-preserving transform and enforce residual policy.

    Args:
        request: Paired xy correspondences and robust/residual policy.
        context: Calculation identity; no files are read by this operation.

    Returns:
        Valid similarity and diagnostics, or an invalid observation with reason.
    """
    source = np.array([[p.x, p.y] for p in request.source_points], np.float64)
    destination = np.array([[p.x, p.y] for p in request.destination_points], np.float64)
    inliers = np.ones(len(source), bool)
    try:
        if request.robust:
            matrix, mask = cv2.estimateAffinePartial2D(
                source, destination, method=cv2.RANSAC,
                ransacReprojThreshold=request.ransac_threshold,
                maxIters=2000, confidence=0.99,
            )
            if matrix is None or mask is None:
                raise ValueError("No robust model")
            inliers = mask.reshape(-1).astype(bool)
        if inliers.sum() < request.minimum_inliers:
            return SimilarityEstimateResult(valid=False, reason="insufficient_inliers")
        matrix = _fit_similarity(source[inliers], destination[inliers], request.estimate_scale)
        observed = source @ matrix[:2, :2].T + matrix[:2, 2]
        residuals = np.linalg.norm(observed - destination, axis=1)
        accepted = inliers & (residuals <= request.max_residual)
        if accepted.sum() < request.minimum_inliers:
            return SimilarityEstimateResult(valid=False, reason="excessive_residual",
                                             residuals=residuals.tolist(),
                                             inliers=accepted.tolist())
        if not request.robust and not accepted.all():
            return SimilarityEstimateResult(valid=False, reason="excessive_residual",
                                             residuals=residuals.tolist(),
                                             inliers=accepted.tolist())
        factor = np.linalg.norm(matrix[:2, 0])
        angle = np.rad2deg(np.arctan2(matrix[1, 0], matrix[0, 0]))
        return SimilarityEstimateResult(
            valid=True, mapping=coordinate_mapping(
                matrix, request.source_frame, request.destination_frame,
            ), scale=factor, angle_degrees=angle, residuals=residuals.tolist(),
            inliers=accepted.tolist(), rms_residual=float(np.sqrt(np.mean(residuals[accepted]**2))),
        )
    except (ValueError, np.linalg.LinAlgError):
        return SimilarityEstimateResult(valid=False, reason="degenerate_correspondences")


class PointGeometry(StrictModel):
    type: Literal["point"] = "point"
    point: Point


class LineGeometry(StrictModel):
    type: Literal["line"] = "line"
    a: float
    b: float
    c: float

    @model_validator(mode="after")
    def nonzero_normal(self) -> LineGeometry:
        if self.a**2 + self.b**2 < 1e-20:
            raise ValueError("Line normal must be nonzero")
        return self


class CircleGeometry(StrictModel):
    type: Literal["circle"] = "circle"
    center: Point
    radius: float = Field(gt=0)


class RectangleGeometry(StrictModel):
    type: Literal["rectangle"] = "rectangle"
    center: Point
    width: float = Field(gt=0)
    height: float = Field(gt=0)
    angle_degrees: float = 0


class PolylineGeometry(StrictModel):
    type: Literal["polyline"] = "polyline"
    points: List[Point] = Field(min_length=2)


Geometry = Annotated[Union[PointGeometry, LineGeometry, CircleGeometry,
                           RectangleGeometry, PolylineGeometry], Field(discriminator="type")]


class GeometryTransformRequest(CalculationRequest):
    """Convert typed observations between declared pixel frames."""

    mapping: CoordinateMapping
    geometry: List[Geometry] = Field(min_length=1, max_length=10000)


class GeometryTransformResult(StrictModel):
    geometry: List[Geometry]
    frame: str
    unit: str
    mapping: CoordinateMapping


def validated_matrix(mapping: CoordinateMapping, similarity: bool = False) -> np.ndarray:
    """Verify matrix direction, inverse and metrology-compatible shape.

    Args:
        mapping: Declared forward/inverse coordinate mapping to authorize.
        similarity: Whether to reject affine/projective transforms for metrology.

    Returns:
        Validated forward 3-by-3 mapping suitable for the requested metrology.
    """
    if mapping.type == "nonlinear" or mapping.matrix is None or mapping.inverse_matrix is None:
        raise ServiceError(
            "transform.matrix_required", "Operation requires an explicit invertible matrix",
        )
    matrix = np.array(mapping.matrix, np.float64)
    inverse = np.array(mapping.inverse_matrix, np.float64)
    if (matrix.shape != (3, 3) or inverse.shape != (3, 3)
            or abs(np.linalg.det(matrix)) < 1e-12
            or not np.allclose(matrix @ inverse, np.eye(3), atol=1e-6)):
        raise ServiceError("transform.invalid", "Mapping matrix or inverse is inconsistent")
    if similarity:
        gram = matrix[:2, :2].T @ matrix[:2, :2]
        if (not np.allclose(matrix[2], [0, 0, 1], atol=1e-7)
                or not np.allclose(gram, np.eye(2) * gram[0, 0], atol=1e-7)
                or np.linalg.det(matrix[:2, :2]) <= 0):
            raise ServiceError(
                "transform.similarity_required", "Geometry requires a similarity mapping",
            )
    return matrix


def _map_point(point: Point, matrix: np.ndarray) -> Point:
    mapped = matrix @ [point.x, point.y, 1]
    if abs(mapped[2]) < 1e-12:
        raise ServiceError("transform.point_at_infinity", "Point has no finite mapped position")
    return Point(x=mapped[0]/mapped[2], y=mapped[1]/mapped[2])


def transform_geometry(
    request: GeometryTransformRequest, context: CalculationContext,
) -> GeometryTransformResult:
    """Transform circles/rectangles under shape-preserving similarities.

    Args:
        request: Typed geometry and an explicit invertible frame mapping.
        context: Calculation identity; no shared assets are needed.

    Returns:
        Converted geometry labeled with its destination frame and unit.
    """
    matrix = validated_matrix(request.mapping)
    results = []
    for geometry in request.geometry:
        if geometry.type == "point":
            results.append(PointGeometry(point=_map_point(geometry.point, matrix)))
        elif geometry.type == "polyline":
            results.append(PolylineGeometry(
                points=[_map_point(p, matrix) for p in geometry.points],
            ))
        elif geometry.type == "line":
            coefficients = np.linalg.inv(matrix).T @ [geometry.a, geometry.b, geometry.c]
            norm = np.linalg.norm(coefficients[:2])
            if norm < 1e-12:
                raise ServiceError("transform.line_at_infinity", "Line has no finite mapped normal")
            results.append(LineGeometry(a=coefficients[0]/norm, b=coefficients[1]/norm,
                                        c=coefficients[2]/norm))
        else:
            validated_matrix(request.mapping, similarity=True)
            factor = np.linalg.norm(matrix[:2, 0])
            center = _map_point(geometry.center, matrix)
            if geometry.type == "circle":
                results.append(CircleGeometry(center=center, radius=geometry.radius * factor))
            else:
                angle = np.rad2deg(np.arctan2(matrix[1, 0], matrix[0, 0]))
                results.append(RectangleGeometry(center=center, width=geometry.width * factor,
                                                  height=geometry.height * factor,
                                                  angle_degrees=geometry.angle_degrees + angle))
    return GeometryTransformResult(geometry=results, frame=request.mapping.destination_frame,
                                   unit=request.mapping.unit, mapping=request.mapping)


OPERATIONS = [
    Operation("/transforms/similarity-estimates",
              SimilarityEstimateRequest, SimilarityEstimateResult,
              estimate_similarity, "Estimate similarity from correspondences", "alignment",
              {"mapping": "transform"}),
    Operation("/geometry/transforms", GeometryTransformRequest, GeometryTransformResult,
              transform_geometry, "Map typed geometry between frames", "alignment",
              {"geometry": "geometry"}),
]


class RigidAlignmentRequest(CalculationRequest):
    """Normalize one isolated target using an inspection-to-aligned similarity."""

    image: FileReference
    initial_mapping: CoordinateMapping
    width: int = Field(ge=1, le=32768)
    height: int = Field(ge=1, le=32768)
    output: OutputSpec
    interpolation: Literal["nearest", "linear", "cubic"] = "linear"
    border_policy: Literal["constant", "replicate", "reject"] = "constant"
    border_value: int = Field(default=0, ge=0, le=255)


class ControlTemplate(StrictModel):
    """A portable local feature at a desired aligned-frame position."""

    template: FileReference
    reference_point: Point
    search_radius: int = Field(default=10, ge=1, le=32768)
    score_threshold: float = Field(default=0.7, ge=0, le=1)


class ControlAlignmentRequest(RigidAlignmentRequest):
    """Refine a single target from controls confined to its own initial patch."""

    controls: List[ControlTemplate] = Field(min_length=2, max_length=1000)
    minimum_controls: int = Field(default=2, ge=2)
    robust: bool = True
    ransac_threshold: float = Field(default=3, gt=0)
    max_residual: float = Field(default=3, gt=0)
    estimate_scale: bool = True
    failure_policy: Literal["fail", "rigid_fallback"] = "fail"

    @model_validator(mode="after")
    def check_controls(self) -> ControlAlignmentRequest:
        if self.minimum_controls > len(self.controls):
            raise ValueError("Minimum controls exceeds supplied controls")
        for control in self.controls:
            if not (0 <= control.reference_point.x < self.width
                    and 0 <= control.reference_point.y < self.height):
                raise ValueError("Control reference point must lie in the target patch")
        return self


class ControlObservation(StrictModel):
    index: int
    valid: bool
    reference_point: Point
    observed_point: Optional[Point] = None
    score: Optional[float] = None
    inlier: bool = False
    residual: Optional[float] = None


class AlignmentResult(StrictModel):
    """Aligned image and inverse metrology mapping into original inspection pixels."""

    valid: bool
    reason: Optional[str] = None
    used_fallback: bool = False
    source: FileReference
    image: Optional[FileReference] = None
    mapping: Optional[CoordinateMapping] = None
    metric_mapping: Optional[CoordinateMapping] = None
    correction: Optional[SimilarityEstimateResult] = None
    controls: List[ControlObservation] = Field(default_factory=list)
    artifacts: List[FileDescriptor] = Field(default_factory=list)
    manifest: Optional[FileDescriptor] = None


def _warp_alignment(image: np.ndarray, matrix: np.ndarray,
                    request: RigidAlignmentRequest, context: CalculationContext) -> np.ndarray:
    if request.width * request.height > context.settings.max_image_pixels:
        raise ServiceError("alignment.too_large", "Aligned image exceeds configured pixel limit")
    if request.border_policy == "reject":
        inverse = np.linalg.inv(matrix)
        corners = np.array([[0, 0, 1], [request.width-1, 0, 1],
                            [0, request.height-1, 1], [request.width-1, request.height-1, 1]])
        points = (inverse @ corners.T).T
        if (points[:, 0].min() < -1e-6 or points[:, 1].min() < -1e-6
                or points[:, 0].max() > image.shape[1]-1+1e-6
                or points[:, 1].max() > image.shape[0]-1+1e-6):
            raise ServiceError("alignment.outside_image", "Aligned patch includes source border")
    interpolation = {"nearest": cv2.INTER_NEAREST, "linear": cv2.INTER_LINEAR,
                     "cubic": cv2.INTER_CUBIC}[request.interpolation]
    border = cv2.BORDER_REPLICATE if request.border_policy == "replicate" else cv2.BORDER_CONSTANT
    return cv2.warpAffine(image, matrix[:2], (request.width, request.height), flags=interpolation,
                          borderMode=border, borderValue=(request.border_value,)*3)


def _commit_alignment(image: np.ndarray, source: FileReference, matrix: np.ndarray,
                      request: RigidAlignmentRequest, context: CalculationContext,
                      **observations: object) -> AlignmentResult:
    from measure_api.v2.files import ArtifactWriter
    from measure_api.v2.families.images import encode_png

    pixels = _warp_alignment(image, matrix, request, context)
    writer = ArtifactWriter(context.files, request.output, context.operation_id)
    artifacts, manifest = writer.commit([
        ("aligned.png", "aligned_image", "png", encode_png(pixels)),
    ])
    mapping = coordinate_mapping(matrix, request.initial_mapping.source_frame,
                                 request.initial_mapping.destination_frame)
    metric = coordinate_mapping(np.linalg.inv(matrix), mapping.destination_frame,
                                mapping.source_frame)
    metric.details = {"purpose": "Recover actual dimensions in original inspection pixels"}
    return AlignmentResult(valid=True, source=source, image=artifacts[0].file,
                            mapping=mapping, metric_mapping=metric,
                            artifacts=artifacts, manifest=manifest, **observations)


def align_rigid(
    request: RigidAlignmentRequest, context: CalculationContext,
) -> AlignmentResult:
    """Warp original pixels once and preserve metric-frame scale in the inverse.

    Args:
        request: Original image, inspection-to-aligned mapping and output controls.
        context: Shared-file boundaries and operation artifact identity.

    Returns:
        Aligned pixels and forward/inverse metrology mappings.
    """
    matrix = validated_matrix(request.initial_mapping, similarity=True)
    image, source = context.files.read_image(request.image)
    return _commit_alignment(image, source, matrix, request, context)


def align_controls(
    request: ControlAlignmentRequest, context: CalculationContext,
) -> AlignmentResult:
    """Estimate correction Q, then apply inverse(Q) to initial mapping P.

    Args:
        request: One target patch, portable controls and explicit failure policy.
        context: Shared-file boundaries and operation artifact identity.

    Returns:
        Corrected alignment, marked rigid fallback, or invalid refinement.
    """
    from measure_api.v2.preprocessing import apply_preprocessing
    from measure_api.v2.families.templates import TemplateSearch, _scan, load_template

    initial = validated_matrix(request.initial_mapping, similarity=True)
    image, source = context.files.read_image(request.image)
    patch = _warp_alignment(image, initial, request, context)
    observations = []
    matched_indices = []
    reference_points, observed_points = [], []
    for index, control in enumerate(request.controls):
        pixels, metadata, _ = load_template(control.template, context)
        h, w = pixels.shape[:2]
        radius = control.search_radius
        x, y = control.reference_point.x, control.reference_point.y
        x0, y0 = max(0, int(x-radius-w/2)), max(0, int(y-radius-h/2))
        x1, y1 = min(request.width, int(x+radius+w/2)+1), min(request.height, int(y+radius+h/2)+1)
        local = apply_preprocessing(patch[y0:y1, x0:x1], metadata.preprocessing)
        if local.ndim != pixels.ndim:
            raise ServiceError(
                "template.channels_mismatch", "Control and inspection channels differ",
            )
        search = TemplateSearch(score_threshold=control.score_threshold, max_matches=1)
        candidates = _scan(local, pixels, search, 0, 1, origin=(x0, y0))
        observation = ControlObservation(
            index=index, valid=False, reference_point=control.reference_point,
        )
        if candidates:
            score, _, _, matrix = max(candidates, key=lambda item: item[0])
            anchor = np.array(metadata.source_to_support.matrix) @ [
                metadata.anchor.x, metadata.anchor.y, 1,
            ]
            position = matrix @ anchor
            observed = Point(x=position[0], y=position[1])
            # The feature's anchor must also remain within the requested local control radius.
            if abs(observed.x-x) <= radius and abs(observed.y-y) <= radius:
                observation.valid = True
                observation.observed_point, observation.score = observed, score
                matched_indices.append(index)
                reference_points.append(control.reference_point)
                observed_points.append(observed)
        observations.append(observation)
    correction = None
    reason = "insufficient_controls"
    if len(reference_points) >= request.minimum_controls:
        correction = estimate_similarity(SimilarityEstimateRequest(
            source_points=reference_points, destination_points=observed_points,
            source_frame=request.initial_mapping.destination_frame,
            destination_frame="observed_control_patch", robust=request.robust,
            estimate_scale=request.estimate_scale, ransac_threshold=request.ransac_threshold,
            max_residual=request.max_residual, minimum_inliers=request.minimum_controls,
        ), context)
        for offset, index in enumerate(matched_indices):
            observations[index].inlier = (
                bool(correction.inliers[offset]) if correction.inliers else False
            )
            observations[index].residual = (
                correction.residuals[offset] if correction.residuals else None
            )
        if correction.valid:
            final = np.array(correction.mapping.inverse_matrix) @ initial
            return _commit_alignment(image, source, final, request, context,
                                     controls=observations, correction=correction)
        reason = correction.reason
    if request.failure_policy == "rigid_fallback":
        return _commit_alignment(image, source, initial, request, context,
                                 used_fallback=True, reason=reason,
                                 controls=observations, correction=correction)
    return AlignmentResult(valid=False, reason=reason, source=source,
                            controls=observations, correction=correction)


OPERATIONS.extend([
    Operation("/alignments/rigid", RigidAlignmentRequest, AlignmentResult,
              align_rigid, "Align an isolated target and preserve its metric mapping", "alignment",
              {"image": "image", "mapping": "transform", "metric_mapping": "transform"}),
    Operation("/alignments/control-points", ControlAlignmentRequest, AlignmentResult,
              align_controls, "Refine one target using isolated local controls", "alignment",
              {"image": "image", "mapping": "transform", "metric_mapping": "transform"}),
])
