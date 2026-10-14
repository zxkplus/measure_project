"""Portable template teaching and stateless pixel localization."""

from __future__ import annotations

import io
import zipfile
from typing import List, Literal, Optional, Tuple

import cv2
import numpy as np
from pydantic import Field, model_validator

from measure_api.v2.errors import ServiceError
from measure_api.v2.files import ArtifactWriter
from measure_api.v2.models import (
    CalculationRequest, CoordinateMapping, FileDescriptor, FileReference,
    OutputSpec, Point, StrictModel,
)
from measure_api.v2.preprocessing import (
    Preprocessing, RawPreprocessing, apply_preprocessing, display_preview,
)
from measure_api.v2.registry import CalculationContext, Operation
from measure_api.v2.families.images import RotatedROI, encode_png, roi_to_patch_matrix


class TemplateMetadata(StrictModel):
    """Versioned support and provenance of already processed numeric pixels."""

    schema_version: Literal["1.0"] = "1.0"
    format: Literal["measure-template"] = "measure-template"
    support_mode: Literal["tight_roi", "square_context"]
    support: RotatedROI
    anchor: Point
    preprocessing: Preprocessing
    source: FileReference
    source_to_support: CoordinateMapping
    dtype: Literal["float32", "uint8"]
    channels: Literal[1, 3]


class TemplateBuildRequest(CalculationRequest):
    """Teach a support rectangle and arbitrary reference-image anchor."""

    image: FileReference
    roi: RotatedROI
    anchor: Optional[Point] = None
    support_mode: Literal["tight_roi", "square_context"] = "tight_roi"
    preprocessing: Preprocessing = Field(default_factory=RawPreprocessing)
    output: OutputSpec


class TemplateBuildResult(StrictModel):
    """Portable numeric template and independent display preview."""

    metadata: TemplateMetadata
    template: FileReference
    artifacts: List[FileDescriptor]
    manifest: FileDescriptor


def coordinate_mapping(
    matrix: np.ndarray, source: str, destination: str,
) -> CoordinateMapping:
    """Declare forward and inverse similarity mappings in pixel-center frames.

    Args:
        matrix: Finite 3-by-3 forward transform in homogeneous xy coordinates.
        source: Coordinate-frame name in which the forward mapping receives its input.
        destination: Coordinate-frame name assigned to the transformed image or patch.

    Returns:
        Forward and inverse pixel-center similarity transforms with frame names.
    """
    return CoordinateMapping(source_frame=source, destination_frame=destination,
                             type="similarity", matrix=matrix.tolist(),
                             inverse_matrix=np.linalg.inv(matrix).tolist())


def _has_texture(pixels: np.ndarray) -> bool:
    _, deviation = cv2.meanStdDev(pixels)
    return bool(np.sum(deviation**2) > 1e-12)


def build_template(
    request: TemplateBuildRequest, context: CalculationContext,
) -> TemplateBuildResult:
    """Rectify support and atomically publish numeric pixels and metadata.

    Args:
        request: Supplied teaching image, support, preprocessing and output location.
        context: Verified shared-file boundaries and isolated operation identity.

    Returns:
        Portable template reference, metadata and committed artifact descriptors.
    """
    image, source = context.files.read_image(request.image)
    roi = request.roi
    if request.support_mode == "square_context":
        size = max(roi.width, roi.height)
        roi = roi.model_copy(update={"width": size, "height": size, "angle_degrees": 0})
    if roi.width * roi.height > context.settings.max_image_pixels:
        raise ServiceError("template.too_large", "Template support exceeds pixel limit")
    matrix = roi_to_patch_matrix(roi)
    corners = np.array([[0, 0, 1], [roi.width - 1, 0, 1],
                        [0, roi.height - 1, 1], [roi.width - 1, roi.height - 1, 1]])
    positions = (np.linalg.inv(matrix) @ corners.T).T
    if (positions[:, 0].min() < -1e-6 or positions[:, 1].min() < -1e-6
            or positions[:, 0].max() > image.shape[1] - 1 + 1e-6
            or positions[:, 1].max() > image.shape[0] - 1 + 1e-6):
        raise ServiceError("template.roi_outside", "Support must lie inside source image")
    processed = apply_preprocessing(image, request.preprocessing)
    pixels = cv2.warpAffine(processed, matrix[:2], (roi.width, roi.height))
    if not _has_texture(pixels):
        raise ServiceError("template.uninformative", "Constant templates cannot localize targets")
    metadata = TemplateMetadata(
        support_mode=request.support_mode, support=roi,
        anchor=request.anchor or roi.center, preprocessing=request.preprocessing,
        source=source, source_to_support=coordinate_mapping(matrix, "reference", "support"),
        dtype=str(pixels.dtype), channels=1 if pixels.ndim == 2 else pixels.shape[2],
    )
    payload = io.BytesIO()
    np.savez_compressed(payload, pixels=pixels,
                        metadata=np.array(metadata.model_dump_json()))
    writer = ArtifactWriter(context.files, request.output, context.operation_id)
    artifacts, manifest = writer.commit([
        ("template.npz", "template", "npz", payload.getvalue()),
        ("preview.png", "preview", "png", encode_png(display_preview(pixels))),
    ])
    return TemplateBuildResult(
        metadata=metadata, template=artifacts[0].file, artifacts=artifacts, manifest=manifest,
    )


class TemplateInspectionRequest(CalculationRequest):
    """Inspect a supplied portable asset without engine registration."""

    template: FileReference


class TemplateInspectionResult(StrictModel):
    """Validated actual template metadata and file identity."""

    metadata: TemplateMetadata
    template: FileReference


def load_template(
    reference: FileReference, context: CalculationContext,
) -> Tuple[np.ndarray, TemplateMetadata, FileReference]:
    """Validate safe non-pickle payloads and actual bounded pixels.

    Args:
        reference: Upstream-owned portable asset file reference.
        context: Verified shared-file access and deployment size limits.

    Returns:
        Processed pixels, validated metadata and verified content identity.
    """
    data, verified = context.files.read_bytes(reference)
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            if set(archive.namelist()) != {"pixels.npy", "metadata.npy"}:
                raise ValueError("Unexpected members")
            expanded_size = sum(item.file_size for item in archive.infolist())
            if expanded_size > context.settings.max_input_bytes:
                raise ValueError("Uncompressed template exceeds configured limit")
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            metadata = TemplateMetadata.model_validate_json(str(archive["metadata"].item()))
            pixels = archive["pixels"].copy()
        height, width = pixels.shape[:2]
        channels = 1 if pixels.ndim == 2 else pixels.shape[2] if pixels.ndim == 3 else 0
        if (width != metadata.support.width or height != metadata.support.height
                or channels != metadata.channels or str(pixels.dtype) != metadata.dtype
                or width * height > context.settings.max_image_pixels
                or not np.isfinite(pixels).all() or not _has_texture(pixels)):
            raise ValueError("Invalid template pixels")
        matrix = np.array(metadata.source_to_support.matrix)
        inverse = np.array(metadata.source_to_support.inverse_matrix)
        if (matrix.shape != (3, 3) or inverse.shape != (3, 3)
                or not np.allclose(matrix @ inverse, np.eye(3), atol=1e-6)
                or not np.allclose(matrix[2], [0, 0, 1], atol=1e-7)):
            raise ValueError("Invalid template mapping")
        gram = matrix[:2, :2].T @ matrix[:2, :2]
        if (not np.allclose(gram, np.eye(2), atol=1e-6)
                or np.linalg.det(matrix[:2, :2]) <= 0):
            raise ValueError("Template support mapping must be a rigid rectification")
    except (ValueError, TypeError, KeyError, IndexError, OSError, zipfile.BadZipFile) as error:
        raise ServiceError("template.invalid", "Invalid or incompatible template asset") from error
    return pixels, metadata, verified


def inspect_template(
    request: TemplateInspectionRequest, context: CalculationContext,
) -> TemplateInspectionResult:
    """Inspect actual portable metadata and pixels.

    Args:
        request: Typed inspect a supplied portable asset without engine registration.
        context: Isolated operation identity, resource limits and verified shared-file
            access.

    Returns:
        Validated actual template metadata and file identity.
    """
    _, metadata, verified = load_template(request.template, context)
    return TemplateInspectionResult(metadata=metadata, template=verified)


class SearchRegion(StrictModel):
    """Axis-aligned inspection search window, in full image pixels."""

    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(ge=1)
    height: int = Field(ge=1)


class TemplateSearch(StrictModel):
    """NCC localization controls, angles clockwise and scales dimensionless."""

    score_threshold: float = Field(default=0.7, ge=0, le=1)
    max_matches: int = Field(default=1, ge=0, le=1000)
    overlap: float = Field(default=0.3, ge=0, lt=1)
    region: Optional[SearchRegion] = None
    subpixel: bool = True
    angle_min: float = Field(default=0, ge=-360, le=360)
    angle_max: float = Field(default=0, ge=-360, le=360)
    angle_step: float = Field(default=1, gt=0, le=360)
    scale_min: float = Field(default=1, gt=0, le=10)
    scale_max: float = Field(default=1, gt=0, le=10)
    scale_step: float = Field(default=0.02, gt=0, le=10)
    coarse_fine: bool = False
    coarse_score_threshold: float = Field(default=0.3, ge=0, le=1)
    coarse_angle_step: float = Field(default=5, gt=0, le=360)
    coarse_scale_step: float = Field(default=0.1, gt=0, le=10)
    pyramid_levels: int = Field(default=0, ge=0, le=8)
    pyramid_max_template_size: int = Field(default=400, ge=8, le=32768)

    @model_validator(mode="after")
    def validate_search(self) -> TemplateSearch:
        """Reject reversed or unbounded pose grids before execution.

        Returns:
            This search definition after bounding its angle/scale pose grid.
        """
        if self.angle_min > self.angle_max or self.scale_min > self.scale_max:
            raise ValueError("Pose search ranges must be increasing")
        count = ((self.angle_max - self.angle_min) / self.angle_step + 2) * (
            (self.scale_max - self.scale_min) / self.scale_step + 2
        )
        if count > 20000:
            raise ValueError("Pose grid exceeds 20000 variants")
        if self.coarse_fine and (self.coarse_angle_step < self.angle_step
                                 or self.coarse_scale_step < self.scale_step):
            raise ValueError("Coarse steps must be at least fine steps")
        return self


class TemplateMatchRequest(TemplateInspectionRequest):
    """Find independent target poses in one committed inspection image."""

    image: FileReference
    search: TemplateSearch = Field(default_factory=TemplateSearch)


class TemplateMatch(StrictModel):
    """A target pose with reference-to-inspection and support-to-inspection maps."""

    target_id: str
    center: Point
    anchor: Point
    angle_degrees: float
    scale: float
    score: float
    mapping: CoordinateMapping
    support_mapping: CoordinateMapping


class TemplateMatchResult(StrictModel):
    """Fresh match observations; failed localization contains no stale geometry."""

    valid: bool
    reason: Optional[str] = None
    image: FileReference
    template: FileReference
    matches: List[TemplateMatch]
    diagnostics: dict = Field(default_factory=dict)


def _peaks(scores: np.ndarray, options: TemplateSearch, width: int, height: int) -> list:
    work = np.nan_to_num(scores, nan=-1.0, posinf=-1.0, neginf=-1.0).copy()
    result = []
    limit = options.max_matches or 1000
    while len(result) < limit:
        _, score, _, position = cv2.minMaxLoc(work)
        if score < options.score_threshold:
            break
        x, y = position
        dx = dy = 0.0
        if options.subpixel:
            for axis in (0, 1):
                p = y if axis == 0 else x
                if 0 < p < work.shape[axis] - 1:
                    before = scores[y - 1, x] if axis == 0 else scores[y, x - 1]
                    after = scores[y + 1, x] if axis == 0 else scores[y, x + 1]
                    denom = float(before - 2 * score + after)
                    offset = float((before - after) / (2 * denom)) if abs(denom) > 1e-8 else 0
                    if np.isfinite(offset) and abs(offset) <= 0.5:
                        if axis == 0:
                            dy = offset
                        else:
                            dx = offset
        result.append((x + dx, y + dy, float(np.clip(score, -1, 1))))
        half_w = max(1, int(width * (1 - options.overlap)))
        half_h = max(1, int(height * (1 - options.overlap)))
        work[max(0, y - half_h):y + half_h + 1, max(0, x - half_w):x + half_w + 1] = -1
    return result


def _grid(low: float, high: float, step: float) -> list:
    values = np.arange(low, high + step * 1e-6, step).tolist()
    if not values or abs(values[-1] - high) > 1e-6:
        values.append(high)
    return values


def _variant(pixels: np.ndarray, angle: float, scale: float, limit: tuple) -> tuple:
    """Render a bounded rotated support and mask with its exact pixel mapping."""
    h, w = pixels.shape[:2]
    theta = np.deg2rad(angle)
    c, s = np.cos(theta) * scale, np.sin(theta) * scale
    linear = np.array([[c, -s], [s, c]])
    corners = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
    warped = corners @ linear.T
    low, high = np.floor(warped.min(axis=0)), np.ceil(warped.max(axis=0))
    width, height = np.maximum(1, (high - low + 1).astype(int))
    if width > limit[0] or height > limit[1]:
        return None, None, None
    matrix = np.eye(3)
    matrix[:2, :2], matrix[:2, 2] = linear, -low
    variant = cv2.warpAffine(pixels.astype(np.float32), matrix[:2], (width, height))
    mask = cv2.warpAffine(np.ones((h, w), np.uint8), matrix[:2], (width, height),
                          flags=cv2.INTER_NEAREST)
    return variant, mask, matrix


def _scan(search: np.ndarray, pixels: np.ndarray, options: TemplateSearch,
          angle: float, scale: float, origin: tuple = (0, 0),
          threshold: Optional[float] = None) -> list:
    if not _has_texture(search):
        return []
    variant, mask, matrix = _variant(pixels, angle, scale, (search.shape[1], search.shape[0]))
    if variant is None:
        return []
    h, w = variant.shape[:2]
    if not _has_texture(variant):
        return []
    scores = cv2.matchTemplate(search.astype(np.float32), variant,
                               cv2.TM_CCOEFF_NORMED, mask=mask)
    # Masked NCC can divide by zero on constant backgrounds. They are invalid candidates.
    scores[~np.isfinite(scores)] = -1
    settings = options.model_copy(update={"score_threshold": (
        options.score_threshold if threshold is None else threshold
    ), "max_matches": 1000})
    result = []
    for x, y, score in _peaks(scores, settings, w, h):
        pose = matrix.copy()
        pose[:2, 2] += [x + origin[0], y + origin[1]]
        result.append((score, angle, scale, pose))
    return result


def _suppress(candidates: list, pixels: np.ndarray, options: TemplateSearch) -> list:
    h, w = pixels.shape[:2]
    kept = []
    for candidate in sorted(candidates, key=lambda item: item[0], reverse=True):
        score, angle, scale, matrix = candidate
        center = matrix @ [(w - 1) / 2, (h - 1) / 2, 1]
        rect = ((float(center[0]), float(center[1])), (w * scale, h * scale), angle)
        overlapping = False
        for previous in kept:
            _, pa, ps, pm = previous
            pc = pm @ [(w - 1) / 2, (h - 1) / 2, 1]
            other = ((float(pc[0]), float(pc[1])), (w * ps, h * ps), pa)
            _, polygon = cv2.rotatedRectangleIntersection(rect, other)
            area = 0 if polygon is None else cv2.contourArea(polygon)
            if area / min(w * h * scale**2, w * h * ps**2) > options.overlap:
                overlapping = True
                break
        if not overlapping:
            kept.append(candidate)
            if len(kept) >= (options.max_matches or 1000):
                break
    return kept


def match_template(
    request: TemplateMatchRequest, context: CalculationContext,
) -> TemplateMatchResult:
    """Search poses and refine every pyramid target at full resolution.

    Args:
        request: Supplied template/image and explicit pose/search controls.
        context: Verified shared-file boundaries for this calculation.

    Returns:
        Fresh matches with anchors, scores and invertible frame mappings.
    """
    pixels, metadata, template = load_template(request.template, context)
    image, image_ref = context.files.read_image(request.image)
    processed = apply_preprocessing(image, metadata.preprocessing)
    options = request.search
    region = options.region or SearchRegion(
        x=0, y=0, width=image.shape[1], height=image.shape[0],
    )
    if region.x + region.width > image.shape[1] or region.y + region.height > image.shape[0]:
        raise ServiceError("template.search_outside", "Search region lies outside image")
    search = processed[region.y:region.y + region.height, region.x:region.x + region.width]
    if search.ndim != pixels.ndim or (search.ndim == 3 and search.shape[2] != pixels.shape[2]):
        raise ServiceError("template.channels_mismatch", "Template and image channels must agree")
    h, w = pixels.shape[:2]
    factor = min(2.0 ** -options.pyramid_levels,
                 options.pyramid_max_template_size / max(h, w))
    factor = min(1.0, max(factor, min(1.0, 6.0 / min(h, w))))
    coarse_pixels = pixels
    coarse_search = search
    if factor < 1:
        # Keep exact isotropic center mapping through warpAffine instead of rounded resize axes.
        matrix = np.array([[factor, 0, (factor - 1) / 2], [0, factor, (factor - 1) / 2]])
        # Anti-alias both images before decimation so odd-pixel translations retain correlation.
        sigma = 0.5 / factor
        smooth_pixels = cv2.GaussianBlur(pixels, (0, 0), sigma)
        smooth_search = cv2.GaussianBlur(search, (0, 0), sigma)
        coarse_pixels = cv2.warpAffine(
            smooth_pixels, matrix, (round(w*factor), round(h*factor)),
        )
        coarse_search = cv2.warpAffine(
            smooth_search, matrix,
            (round(search.shape[1]*factor), round(search.shape[0]*factor)),
        )
    astep = options.coarse_angle_step if options.coarse_fine else options.angle_step
    sstep = options.coarse_scale_step if options.coarse_fine else options.scale_step
    candidates = []
    coarse_threshold = min(options.score_threshold, options.coarse_score_threshold) if (
        factor < 1 or options.coarse_fine
    ) else options.score_threshold
    for angle in _grid(options.angle_min, options.angle_max, astep):
        for scale in _grid(options.scale_min, options.scale_max, sstep):
            candidates.extend(_scan(coarse_search, coarse_pixels, options, angle, scale,
                                    threshold=coarse_threshold))
    if factor < 1:
        resize = np.array([[factor, 0, (factor - 1)/2],
                           [0, factor, (factor - 1)/2], [0, 0, 1]])
        candidates = [(score, angle, scale, np.linalg.inv(resize) @ matrix @ resize)
                      for score, angle, scale, matrix in candidates]
    if options.coarse_fine or factor < 1:
        # Coarse candidates locate target neighborhoods; none is lost by a global top-five limit.
        coarse_options = options.model_copy(update={"max_matches": 0})
        neighborhoods = _suppress(candidates, pixels, coarse_options)
        refined = []
        for _, angle, scale, matrix in neighborhoods:
            center = matrix @ [(w-1)/2, (h-1)/2, 1]
            radius = int(np.ceil(max(w, h) * options.scale_max + 4/factor))
            x0, y0 = max(0, int(center[0])-radius), max(0, int(center[1])-radius)
            local = search[y0:min(search.shape[0], int(center[1])+radius+1),
                           x0:min(search.shape[1], int(center[0])+radius+1)]
            angles = (_grid(max(options.angle_min, angle-astep),
                            min(options.angle_max, angle+astep), options.angle_step)
                      if options.coarse_fine else [angle])
            scales = (_grid(max(options.scale_min, scale-sstep),
                            min(options.scale_max, scale+sstep), options.scale_step)
                      if options.coarse_fine else [scale])
            for fine_angle in angles:
                for fine_scale in scales:
                    refined.extend(_scan(local, pixels, options, fine_angle, fine_scale, (x0, y0)))
        candidates = refined
    candidates = _suppress(candidates, pixels, options)
    matches = []
    for index, (score, angle, scale, matrix) in enumerate(candidates):
        matrix = matrix.copy()
        matrix[:2, 2] += [region.x, region.y]
        full = matrix @ np.array(metadata.source_to_support.matrix)
        anchor = full @ [metadata.anchor.x, metadata.anchor.y, 1]
        center = matrix @ [(w - 1)/2, (h - 1)/2, 1]
        matches.append(TemplateMatch(
            target_id=str(index), center=Point(x=center[0], y=center[1]),
            anchor=Point(x=anchor[0], y=anchor[1]), angle_degrees=angle, scale=scale, score=score,
            mapping=coordinate_mapping(full, "reference", "inspection"),
            support_mapping=coordinate_mapping(matrix, "support", "inspection"),
        ))
    return TemplateMatchResult(
        valid=bool(matches), reason=None if matches else "no_match", image=image_ref,
        template=template, matches=matches,
        diagnostics={"pyramid_scale": factor, "preprocessing": metadata.preprocessing.type,
                     "metric_frame": "inspection", "normalization_recovers_size": True},
    )


class TemplatePointRequest(TemplateMatchRequest):
    """Locate the taught anchor as one local measurement feature."""

    frame: str = Field(default="source", min_length=1)

    @model_validator(mode="after")
    def single_point(self) -> TemplatePointRequest:
        """Reject ambiguous multi-target point measurements.

        Returns:
            This point request after requiring one unambiguous template match.
        """
        if self.search.max_matches != 1:
            raise ValueError("Template point measurement requires max_matches=1")
        return self


class TemplateFeaturePoint(StrictModel):
    """Point port uses the same observation shape as atomic geometry ports."""

    valid: bool
    reason: Optional[str] = None
    frame: str = "source"
    unit: str = "px"
    point: Optional[Point] = None


class TemplatePointLocalizationDiagnostics(StrictModel):
    """Preserve the original localization pose in the measurement sampling frame."""

    score: float
    angle_degrees: float
    scale: float = Field(gt=0)
    sampling_frame: str = Field(min_length=1)
    match: TemplateMatch


class TemplatePointResult(StrictModel):
    """Typed local point result for atomic measurement and dependency ports."""

    valid: bool
    reason: Optional[str] = None
    point: TemplateFeaturePoint
    score: Optional[float] = None
    frame: str = "source"
    unit: str = Field(default="px", min_length=1)
    match: Optional[TemplateMatch] = None
    localization_diagnostics: Optional[TemplatePointLocalizationDiagnostics] = None


def measure_template_point(
    request: TemplatePointRequest, context: CalculationContext,
) -> TemplatePointResult:
    """Measure the anchor using exactly the same stateless localization pipeline.

    Args:
        request: Typed locate the taught anchor as one local measurement feature.
        context: Isolated operation identity, resource limits and verified shared-file
            access.

    Returns:
        Typed local point result for atomic measurement and dependency ports.
    """
    result = match_template(request, context)
    if not result.valid:
        return TemplatePointResult(
            valid=False, reason=result.reason, frame=request.frame,
            point=TemplateFeaturePoint(valid=False, reason=result.reason, frame=request.frame),
        )
    match = result.matches[0]
    return TemplatePointResult(
        valid=True, frame=request.frame,
        point=TemplateFeaturePoint(valid=True, point=match.anchor, frame=request.frame),
        score=match.score, match=match,
    )


OPERATIONS = [
    Operation("/template-builds", TemplateBuildRequest, TemplateBuildResult, build_template,
              "Teach a portable template", "templates", {"template": "template_asset"}),
    Operation("/template-inspections", TemplateInspectionRequest, TemplateInspectionResult,
              inspect_template, "Inspect a portable template", "templates",
              {"metadata": "template_metadata"}),
    Operation("/template-matches", TemplateMatchRequest, TemplateMatchResult, match_template,
              "Locate target instances", "templates", {"matches": "target_poses"}),
]

OPERATIONS.append(Operation(
    "/measurements/template-points", TemplatePointRequest, TemplatePointResult,
    measure_template_point, "Measure a local template anchor", "templates", {"point": "point"},
))
