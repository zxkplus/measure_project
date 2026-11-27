"""Portable template teaching and stateless pixel localization."""

from __future__ import annotations

import io
import os
import threading
import zipfile
from collections import OrderedDict
from typing import Any, List, Literal, Optional, Tuple

import cv2
import numpy as np
from pydantic import Field, model_validator

from measure.computation_types import (
    Space, invert_matrix, make_mapping, validate_mapping_matrix,
)
from measure.geometry_mapping import similarity_scale
from measure.rectify import ResampleOptions, patch_source_corners, warp_mapped
from measure.template_search import (
    PreparedSupport, SearchOptions, has_texture, search_support,
)
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
from measure_api.v2.timing import counter, timed
from measure_api.v2.families.images import RotatedROI, encode_png, roi_to_patch_matrix


#: Bounded process-local cache of already validated template pixels.
#:
#: Worker processes are reused for many operations and every localization call
#: reloads the identical portable asset: unzip, ``np.load``, ``np.isfinite`` over
#: up to 6 Mpx and a rigid-mapping check.  Entry identity is
#: ``(path, modification time, size)``; a rewritten asset changes both, and
#: ``MEASURE_V2_TEMPLATE_CACHE=0`` disables the cache entirely.
_TEMPLATE_CACHE: "OrderedDict[Tuple[str, int, int], Tuple[np.ndarray, Any, Any]]" = (
    OrderedDict())
_TEMPLATE_CACHE_BYTES = 128 * 1024 * 1024
_TEMPLATE_CACHE_LOCK = threading.Lock()
_TEMPLATE_CACHE_STATE = {"bytes": 0}


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
                             inverse_matrix=invert_matrix(matrix).tolist())


def _has_texture(pixels: np.ndarray) -> bool:
    return has_texture(pixels)


@timed("template_build")
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
    placed = make_mapping(matrix, source=Space("source", "px"),
                          destination=Space("support", "px"), kind="affine")
    positions = patch_source_corners(placed, (roi.width, roi.height))
    if (positions[:, 0].min() < -1e-6 or positions[:, 1].min() < -1e-6
            or positions[:, 0].max() > image.shape[1] - 1 + 1e-6
            or positions[:, 1].max() > image.shape[0] - 1 + 1e-6):
        raise ServiceError("template.roi_outside", "Support must lie inside source image")
    processed = apply_preprocessing(image, request.preprocessing)
    pixels = warp_mapped(processed, placed, (roi.width, roi.height),
                         options=ResampleOptions(compute_valid_mask=False)).pixels
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


def _template_cache_enabled() -> bool:
    """Report whether validated template pixels may be reused in this process.

    Returns:
        ``False`` when ``MEASURE_V2_TEMPLATE_CACHE`` names a disabled value.
    """
    value = os.getenv("MEASURE_V2_TEMPLATE_CACHE", "1").strip().lower()
    return value not in {"0", "false", "off", "no"}


def _template_cache_key(reference: FileReference, context: CalculationContext):
    """Return the cache identity of one template file, or ``None`` when unreadable."""
    try:
        path = context.files.resolve(reference)
        status = path.stat()
    except (OSError, ServiceError):
        return None
    return (str(path), int(status.st_mtime_ns), int(status.st_size))


def _template_cache_get(key):
    """Return cached ``(pixels, metadata, verified)`` for ``key``, refreshing its age."""
    with _TEMPLATE_CACHE_LOCK:
        entry = _TEMPLATE_CACHE.get(key)
        if entry is not None:
            _TEMPLATE_CACHE.move_to_end(key)
        return entry


def _template_cache_drop(key) -> None:
    """Forget one entry, keeping the byte budget accurate."""
    with _TEMPLATE_CACHE_LOCK:
        entry = _TEMPLATE_CACHE.pop(key, None)
        if entry is not None:
            _TEMPLATE_CACHE_STATE["bytes"] -= int(entry[0].nbytes)


def _template_cache_put(key, pixels: np.ndarray, metadata: TemplateMetadata,
                        verified: FileReference) -> None:
    """Publish one validated template while staying inside the byte budget."""
    size = int(pixels.nbytes)
    if size > _TEMPLATE_CACHE_BYTES:
        return
    with _TEMPLATE_CACHE_LOCK:
        previous = _TEMPLATE_CACHE.pop(key, None)
        if previous is not None:
            _TEMPLATE_CACHE_STATE["bytes"] -= int(previous[0].nbytes)
        while _TEMPLATE_CACHE and (
            _TEMPLATE_CACHE_STATE["bytes"] + size > _TEMPLATE_CACHE_BYTES
        ):
            _, evicted = _TEMPLATE_CACHE.popitem(last=False)
            _TEMPLATE_CACHE_STATE["bytes"] -= int(evicted[0].nbytes)
        _TEMPLATE_CACHE[key] = (pixels, metadata, verified)
        _TEMPLATE_CACHE_STATE["bytes"] += size


@timed("template_load")
def load_template(
    reference: FileReference, context: CalculationContext,
) -> Tuple[np.ndarray, TemplateMetadata, FileReference]:
    """Validate safe non-pickle payloads and actual bounded pixels.

    A template already validated in this worker process for the same file
    identity is returned without re-reading, re-hashing, decompressing or
    re-scanning several million pixels.

    Args:
        reference: Upstream-owned portable asset file reference.
        context: Verified shared-file access and deployment size limits.

    Returns:
        Processed pixels, validated metadata and verified content identity.
    """
    key = _template_cache_key(reference, context) if _template_cache_enabled() else None
    if key is not None:
        cached = _template_cache_get(key)
        if cached is not None:
            pixels, metadata, verified = cached
            expected = reference.sha256
            if expected is None or expected.lower() == str(verified.sha256).lower():
                counter("template_cache_hit")
                return pixels, metadata, verified
            _template_cache_drop(key)
    if key is not None:
        counter("template_cache_miss")
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
        if inverse.shape != (3, 3) or not np.allclose(matrix @ inverse, np.eye(3), atol=1e-6):
            raise ValueError("Invalid template mapping")
        # The support mapping must be a rigid (unit-scale) rectification.
        validate_mapping_matrix(matrix, "similarity")
        if abs(float(similarity_scale(matrix)) - 1.0) > 1e-6:
            raise ValueError("Template support mapping must be a rigid rectification")
    except (ValueError, TypeError, KeyError, IndexError, OSError, zipfile.BadZipFile) as error:
        raise ServiceError("template.invalid", "Invalid or incompatible template asset") from error
    # Cached pixels are shared by reference: never allow a downstream stage to mutate them.
    pixels.setflags(write=False)
    if key is not None:
        _template_cache_put(key, pixels, metadata, verified)
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


def default_fine_policy() -> str:
    """Return the deployment default for pyramid fine-search locality.

    Returns:
        ``MEASURE_V2_TEMPLATE_FINE_POLICY`` when it names a known policy, otherwise
        ``"neighborhood"``. Set it to ``"whole"`` to restore the pre-change
        full-search-area rescans for every request that omits the field.
    """
    value = os.getenv("MEASURE_V2_TEMPLATE_FINE_POLICY", "").strip().lower()
    return value if value in {"neighborhood", "whole"} else "neighborhood"


def default_fine_candidates() -> Optional[int]:
    """Return the deployment default for how many neighborhoods a fine search refines.

    The coarse stage keeps every peak (``max_matches=0``) so a true target is
    never dropped by a global top-five cut, and the fine stage then rescans all
    of them at full resolution.  Bounding that set is a substantial saving on a
    symmetric part searched over 360 degrees, but it is a cost/accuracy
    trade-off that must be validated per model: equal-looking coarse peaks can
    rank within a hundredth of each other, and a cut can then report a
    different valid rotation than the full rescan (observed on the imported
    GUI-template acceptance case, where a 180 degree pose flipped -145 to
    -140).

    Returns:
        ``MEASURE_V2_TEMPLATE_FINE_CANDIDATES`` when it names a positive bound,
        otherwise ``None`` (refine every neighborhood, the historical behaviour).
    """
    value = os.getenv("MEASURE_V2_TEMPLATE_FINE_CANDIDATES", "").strip().lower()
    if value.isdigit() and int(value) > 0:
        return int(value)
    return None


#: Pose-grid concurrency used when the operator configures nothing.
DEFAULT_SEARCH_THREADS = 16
#: Hard ceiling on pose-grid concurrency, however configured.
MAX_SEARCH_THREADS = 16


def search_threads() -> int:
    """Return how many pose variants one operation may evaluate concurrently.

    Pose variants share the search window and differ only by support rotation,
    so splitting them across threads is a pure speed-up: recorded values are
    identical at any thread count, which is what the pose-grid determinism test
    pins. The unconfigured default is parallel because the alternative was
    leaving most of the machine idle - single-threaded workers gain ~4.8x at
    16 pose threads on a rotation search.

    Returns:
        ``MEASURE_V2_SEARCH_THREADS`` when it parses, bounded to
        ``1..MAX_SEARCH_THREADS``. Unset or unparseable falls back to
        ``DEFAULT_SEARCH_THREADS``, lowered to the machine's CPU count when that
        is smaller, and never below ``1``. Each compute worker gets this many
        pose threads, so keep ``max_workers * search_threads()`` inside the
        physical core count.
    """
    configured = os.getenv("MEASURE_V2_SEARCH_THREADS", "").strip()
    if configured:
        try:
            return max(1, min(int(configured), MAX_SEARCH_THREADS))
        except ValueError:
            pass  # An unusable value is treated as unconfigured.
    machines = os.cpu_count() or MAX_SEARCH_THREADS
    return max(1, min(DEFAULT_SEARCH_THREADS, machines))


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
    # Public field names are part of the v2 request contract and stay as they
    # were published; the engine calls the same quantities decimate_exponent and
    # coarse_template_side. The mapping happens here and nowhere else.
    pyramid_levels: int = Field(default=0, ge=0, le=8)
    pyramid_max_template_size: int = Field(default=128, ge=8, le=32768)
    # None resolves to MEASURE_V2_TEMPLATE_FINE_POLICY (default "neighborhood").
    fine_search_policy: Optional[Literal["neighborhood", "whole"]] = None
    fine_search_expand_px: int = Field(default=24, ge=0, le=4096)
    # None refines every coarse neighborhood (the pre-change cost); a positive
    # bound is a cost/accuracy trade-off validated per model.
    fine_candidates: Optional[int] = Field(default=None, ge=1, le=1000)
    # "dense" keeps the request's angle grid; "hierarchical" samples the same
    # window coarsely and refines only around its best pose.
    fine_angle_policy: Literal["dense", "hierarchical"] = "dense"
    # Halcon's Metric: "use_polarity" requires the part to keep its contrast,
    # "ignore_global_polarity" also accepts a globally inverted one.
    metric: Literal["use_polarity", "ignore_global_polarity"] = "use_polarity"
    # Which matcher runs. "ncc" correlates preprocessed intensity; "shape"
    # correlates edge gradient directions, which is much cheaper when the
    # support nearly fills the frame. The two report scores on different scales,
    # so a threshold calibrated for one does not transfer to the other.
    algorithm: Literal["ncc", "shape"] = "ncc"
    # Shape-only: gradient magnitude below which a pixel is not an edge point.
    # 0 derives the floor from the support's own magnitude distribution.
    min_contrast: float = Field(default=0.0, ge=0, le=255)

    def resolved_fine_policy(self) -> str:
        """Return the effective fine-search policy for this request.

        Returns:
            ``"neighborhood"`` (bounded local rescans) or ``"whole"`` (legacy
            full-search-area rescans).
        """
        return self.fine_search_policy or default_fine_policy()

    def resolved_fine_candidates(self) -> Optional[int]:
        """Return how many coarse neighborhoods the fine stage may refine.

        Every coarse neighborhood was previously rescanned at full resolution
        even when ``max_matches`` asked for one target: a symmetric part under a
        360 degree search produced six windows and 126 full-resolution
        correlations for a single result.

        Returns:
            An upper bound on refined neighborhoods, or ``None`` to refine all.
        """
        if self.max_matches == 0:
            return None
        if self.fine_candidates is not None:
            return self.fine_candidates
        return default_fine_candidates()

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


def _engine_support(pixels: np.ndarray, metadata: TemplateMetadata) -> PreparedSupport:
    """Express a loaded portable template as the engine's immutable support.

    A template asset already holds the processed support pixels and the
    reference-to-support mapping, so the engine takes that support directly
    instead of re-cropping a reference image. The anchor has to move into support
    coordinates: the asset stores it where it was taught (the reference frame),
    while every reported pose is relative to the support.

    Matching runs on one intensity plane, so a colour support is reduced to luma
    here exactly as the engine reduces the inspection image.

    Args:
        pixels: Processed support pixels from the asset.
        metadata: Support provenance, anchor and preprocessing description.

    Returns:
        A prepared support in pixel-centre coordinates.
    """
    support = pixels if pixels.ndim == 2 else cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
    height, width = support.shape[:2]
    matrix = np.array(metadata.source_to_support.matrix, dtype=np.float64)
    anchor = matrix @ np.array([metadata.anchor.x, metadata.anchor.y, 1.0])
    return PreparedSupport(
        pixels=np.ascontiguousarray(support, dtype=np.float32),
        valid_mask=np.ones((height, width), dtype=bool),
        anchor_support=(float(anchor[0]), float(anchor[1])),
        support_to_reference=None,
        reference_space=None,
        support_space=Space("support", "px"),
        mask=None,
        origin="pixel_center",
    )


def _engine_options(options: TemplateSearch, *, algorithm: str,
                    radius_px: float) -> SearchOptions:
    """Translate the published search controls into the engine's policy object.

    This is the only place the two vocabularies meet: degrees to radians,
    ``pyramid_levels`` to ``decimate_exponent``, ``overlap`` to ``max_overlap``.
    Both matchers share it, so no control can apply to one algorithm and not the
    other.

    Args:
        options: Request search controls, in degrees.
        algorithm: ``"ncc"`` or ``"shape"``.
        radius_px: Half-side of a full-resolution refinement window. A value as
            large as the search area expresses the legacy "whole" policy.

    Returns:
        The engine policy for this request.
    """
    return SearchOptions(
        score_threshold=float(options.score_threshold),
        max_matches=int(options.max_matches),
        max_overlap=float(options.overlap),
        subpixel=bool(options.subpixel),
        coarse_fine=bool(options.coarse_fine),
        coarse_score_threshold=float(options.coarse_score_threshold),
        coarse_angle_step_rad=float(np.deg2rad(options.coarse_angle_step)),
        coarse_scale_step=float(options.coarse_scale_step),
        angle_min_rad=float(np.deg2rad(options.angle_min)),
        angle_max_rad=float(np.deg2rad(options.angle_max)),
        angle_step_rad=float(np.deg2rad(options.angle_step)),
        scale_min=float(options.scale_min),
        scale_max=float(options.scale_max),
        scale_step=float(options.scale_step),
        decimate_exponent=int(options.pyramid_levels),
        coarse_template_side=int(options.pyramid_max_template_size),
        fine_search_radius_px=float(radius_px),
        fine_candidates=options.resolved_fine_candidates(),
        algorithm=algorithm,
        min_contrast=float(options.min_contrast),
        metric=str(options.metric),
        search_threads=search_threads(),
    )


def locate_support(pixels: np.ndarray, metadata: TemplateMetadata,
                   search: np.ndarray, options: TemplateSearch):
    """Run the shared search for either matcher and report what it planned.

    Args:
        pixels: Rectified support pixels.
        metadata: Support provenance, anchor and preprocessing description.
        search: Preprocessed search area (the request's region crop).
        options: Request search controls.

    Returns:
        ``(observation, policy, ignored_fields)`` where ``policy`` is the
        effective fine-search policy and ``ignored_fields`` lists the request
        controls the shared orchestration cannot honour.
    """
    algorithm = "shape" if options.algorithm == "shape" else "ncc"
    policy = options.resolved_fine_policy()
    # The refinement window must still hold the support whole, so "whole" is a
    # radius as large as the search area. The engine then reports the window it
    # really used rather than one this layer only assumed.
    radius = (float(max(search.shape)) if policy == "whole"
              else float(options.fine_search_expand_px))
    found = search_support(
        np.ascontiguousarray(search), _engine_support(pixels, metadata),
        _engine_options(options, algorithm=algorithm, radius_px=radius),
        image_space=Space("inspection", "px"))
    counter("ncc_evaluations",
            float(found.diagnostics.get("ncc_evaluations", 0) or 0))
    counter("fine_neighborhoods",
            float(found.diagnostics.get("fine_neighborhoods_available", 0) or 0))
    ignored: List[str] = []
    if policy != "neighborhood":
        ignored.append("fine_search_policy")
    if options.fine_angle_policy != "dense":
        ignored.append("fine_angle_policy")
    return found, policy, ignored


def _engine_diagnostics(found, options: TemplateSearch, *, algorithm: str,
                        factor: float, policy: str, ignored: List[str],
                        preprocessing: str) -> dict:
    """Publish the shared orchestration's own account of the search.

    The plan is translated, never re-derived: every count and every window here
    was reported by the stage that did the work, so a diagnostic cannot claim a
    scan that did not happen.

    Args:
        found: The engine's observation carrying its diagnostics.
        options: Request search controls.
        algorithm: ``"ncc"`` or ``"shape"``.
        factor: Decimation factor implied by the pyramid controls.
        policy: Effective fine-search policy.
        ignored: Request controls the shared orchestration could not honour.
        preprocessing: Preprocessing preset name from the template asset.

    Returns:
        The response diagnostics for this match.
    """
    report = dict(found.diagnostics)
    decimated = factor < 1.0 - 1e-12
    if algorithm == "shape":
        resample = "shape"
    else:
        resample = "blur" if decimated else "none"
    fine_positions = int(report.get("fine_positions", 0) or 0)
    radius = report.get("fine_search_radius")
    plan = {
        "resample": resample,
        "coarse_poses": int(report.get("coarse_poses", 0) or 0),
        "fine_poses_per_neighborhood": int(
            report.get("fine_poses_per_neighborhood", 0) or 0),
        "neighborhoods_available": int(
            report.get("fine_neighborhoods_available", fine_positions) or 0),
        "neighborhoods_refined": int(
            report.get("fine_neighborhoods", fine_positions) or 0),
        "fine_candidates": options.fine_candidates,
        "fine_angle_policy": options.fine_angle_policy,
        # The edge-direction search evaluates poses sequentially, so reporting
        # the configured pose-thread count there would claim parallelism that
        # does not happen.
        "threads": 1 if algorithm == "shape" else search_threads(),
        "ignored_fields": list(ignored),
    }
    diagnostics = {
        "pyramid_scale": float(factor),
        "preprocessing": preprocessing,
        "metric_frame": "inspection",
        "normalization_recovers_size": True,
        "algorithm": algorithm,
        "search_plan": plan,
        "fine_search": {
            "policy": policy,
            # The "whole" policy has no neighbourhood extent to report: its
            # radius is simply "as wide as the search area", and the windows
            # below already say how far that reached.
            "neighborhood_radius": (
                None if policy == "whole" or radius is None else float(radius)),
            "expand_px": options.fine_search_expand_px,
            "windows": list(report.get("fine_windows", ()) or ()),
            "skipped_poses": [],
            "bounded_outside": bool(report.get("bounded_outside", False)),
        },
        "verify_level": int(report.get("verify_level", 0) or 0),
        "unavailable_variant_count": int(
            report.get("unavailable_variant_count", 0) or 0),
        "ambiguous": bool(report.get("ambiguous", False)),
    }
    for key in ("rival_count", "runner_up_score", "runner_up_distance_px",
                "shape_scale_fallback", "shape_support_too_large"):
        if key in report:
            diagnostics[key] = report[key]
    return diagnostics


def match_template(
    request: TemplateMatchRequest, context: CalculationContext,
) -> TemplateMatchResult:
    """Locate the support with the engine's shared orchestration.

    Both matchers and both fine-search policies go through the same call, so the
    pose grid, the pyramid, the neighbourhood refinement and the suppression
    policy have one implementation and cannot drift between algorithms.

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
    found, policy, ignored = locate_support(pixels, metadata, search, options)
    # The decimation factor is published from the engine's own report rather than
    # recomputed here: two formulas for one quantity is exactly the drift this
    # layer is meant to remove. No decimation means exactly 1.0.
    factor = float(found.diagnostics.get("pyramid_factor", 1.0) or 1.0)
    diagnostics = _engine_diagnostics(
        found, options, algorithm=("shape" if options.algorithm == "shape" else "ncc"),
        factor=factor, policy=policy, ignored=ignored,
        preprocessing=metadata.preprocessing.type)
    height, width = pixels.shape[:2]
    matches: List[TemplateMatch] = []
    if found.value is not None:
        for index, match in enumerate(found.value.matches):
            matrix = np.array(match.support_to_image.matrix, dtype=np.float64)
            matrix[:2, 2] += [region.x, region.y]
            full = matrix @ np.array(metadata.source_to_support.matrix, dtype=np.float64)
            anchor = full @ [metadata.anchor.x, metadata.anchor.y, 1.0]
            center = matrix @ [(width - 1) / 2, (height - 1) / 2, 1.0]
            matches.append(TemplateMatch(
                target_id=str(index),
                center=Point(x=float(center[0]), y=float(center[1])),
                anchor=Point(x=float(anchor[0]), y=float(anchor[1])),
                angle_degrees=float(np.rad2deg(match.angle_rad)),
                scale=float(match.scale), score=float(match.score),
                mapping=coordinate_mapping(full, "reference", "inspection"),
                support_mapping=coordinate_mapping(matrix, "support", "inspection"),
            ))
    return TemplateMatchResult(
        valid=bool(matches), reason=None if matches else (found.reason or "no_match"),
        image=image_ref, template=template, matches=matches, diagnostics=diagnostics)


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
