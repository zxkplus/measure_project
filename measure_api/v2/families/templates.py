"""Portable template teaching and stateless pixel localization."""

from __future__ import annotations

import io
import os
import threading
import zipfile
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Literal, Optional, Tuple

import numpy as np
from pydantic import Field, model_validator

from measure.computation_types import (
    Space, invert_matrix, make_mapping, validate_mapping_matrix,
)
from measure.geometry_mapping import similarity_scale
from measure.rectify import ResampleOptions, patch_source_corners, warp_mapped
from measure.template_search import (
    SearchOptions, correlate, decimate_matrix, decimate_pixels, has_texture,
    PreparedSupport, keep_disjoint, peaks, refine_peak_xy, render_variant,
    rotated_box_overlap, variant_size,
)
from measure.template_search import _pyramid_factor as engine_factor
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
from measure_api.v2.timing import counter, current, measure, span, timed
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


def _grid(low: float, high: float, step: float) -> list:
    values = np.arange(low, high + step * 1e-6, step).tolist()
    if not values or abs(values[-1] - high) > 1e-6:
        values.append(high)
    return values


def _variant(pixels: np.ndarray, angle: float, scale: float,
             feasible: tuple) -> Tuple[Optional[np.ndarray], Optional[np.ndarray],
                                      Optional[np.ndarray], Tuple[float, ...]]:
    """Render a rotated/scaled support and mask with its exact pixel mapping.

    Args:
        pixels: Rectified support pixels.
        angle: Clockwise pose angle in degrees.
        scale: Dimensionless pose scale.
        feasible: Largest variant width/height the search area can accommodate.

    Returns:
        Variant, mask and support-to-search pose matrix, plus the rendered size.
        When the rotated support would not fit the search area, all data is
        ``None`` and only the required size is returned so callers can widen
        their window instead of clipping the variant into false peaks.
    """
    angle_rad = float(np.deg2rad(angle))
    # The shared layer owns the rotation geometry; ``pixel_center`` selects the
    # pixel-index convention this API has always reported.
    width, height = variant_size(pixels, angle_rad, scale, origin="pixel_center")
    size = (width, height)
    if width > feasible[0] or height > feasible[1]:
        return None, None, None, size
    with measure("variant_render"):
        rendered = render_variant(pixels.astype(np.float32), angle_rad, scale,
                                  origin="pixel_center")
    mask = rendered.mask
    if mask is None:
        mask = np.ones(rendered.pixels.shape[:2], np.uint8)
    return rendered.pixels, mask, rendered.matrix, size


def _scan(search: np.ndarray, pixels: np.ndarray, options: TemplateSearch,
          angle: float, scale: float, origin: tuple = (0, 0),
          threshold: Optional[float] = None,
          skipped: Optional[list] = None) -> list:
    """Correlate one pose variant over a search area.

    Args:
        search: Numeric search-area pixels.
        pixels: Rectified support pixels.
        options: Search controls (threshold, overlap, subpixel).
        angle: Pose angle in degrees.
        scale: Pose scale.
        origin: ``(x, y)`` of the search area's top-left pixel in its parent frame.
        threshold: Optional coarse/neighborhood threshold override.
        skipped: Optional list receiving poses whose rotated support exceeds the
            search area; never silently clipped.

    Returns:
        Candidates as ``(score, angle, scale, support-to-parent pose matrix)``.
    """
    if not _has_texture(search):
        return []
    variant, mask, matrix, size = _variant(
        pixels, angle, scale, (search.shape[1], search.shape[0]))
    if variant is None:
        if skipped is not None:
            skipped.append({"angle": angle, "scale": scale, "required": list(size)})
        return []
    h, w = variant.shape[:2]
    if not _has_texture(variant):
        return []
    # A full support mask is ordinary NCC; OpenCV's masked path is much slower.
    if np.all(mask):
        mask = None
    # One contiguous float32 window is reused for every pose: converting it per
    # evaluation copied the whole window (tens of megabytes on full-resolution
    # rescans) once for each angle in the grid.
    area = np.ascontiguousarray(search, dtype=np.float32)
    with measure("ncc"):
        scores = correlate(area, variant, mask, metric=options.metric)
    counter("ncc_evaluations")
    counter("ncc_positions", float(scores.size))
    counter("ncc_template_pixels", float(w * h))
    counter("ncc_masked" if mask is not None else "ncc_unmasked")
    settings = options.model_copy(update={"score_threshold": (
        options.score_threshold if threshold is None else threshold
    ), "max_matches": 1000})
    result = []
    # The shared primitive owns peak selection and the greedy suppression box;
    # subpixel refinement and pose re-attachment stay here.
    for x, y, score in peaks(scores, settings.score_threshold,
                             limit=settings.max_matches,
                             suppression_box=(float(w), float(h)),
                             overlap=settings.overlap):
        if settings.subpixel:
            x, y = refine_peak_xy(scores, int(x), int(y), centre=score)
        pose = matrix.copy()
        pose[:2, 2] += [x + origin[0], y + origin[1]]
        result.append((score, angle, scale, pose))
    return result


#: Upper bound on the working set a threaded pose grid may hold at once.
_POSE_GRID_MEMORY_BYTES = 768 * 1024 * 1024


def _scan_grid(search: np.ndarray, pixels: np.ndarray, options: TemplateSearch,
               poses: List[Tuple[float, float]], origin: Tuple[int, int] = (0, 0),
               threshold: Optional[float] = None) -> Tuple[list, list]:
    """Correlate independent pose variants over one search area.

    Pose variants share the search window and differ only by support rotation,
    so they can be evaluated concurrently without changing the result: blocks
    are merged in pose order and the final selection sorts by score.

    Args:
        search: Search-area pixels.
        pixels: Rectified support pixels.
        options: Search controls.
        poses: ``(angle, scale)`` variants in output priority order.
        origin: Search-area origin in the parent frame.
        threshold: Optional score threshold override for every variant.

    Returns:
        ``(candidates, skipped)`` merged in pose order.
    """
    recorder = current()

    def evaluate(target: list, misses: list, angle: float, scale: float) -> None:
        if recorder is None:
            target.extend(_scan(search, pixels, options, angle, scale, origin,
                                threshold=threshold, skipped=misses))
            return
        # A worker thread starts with an empty context: re-install the operation
        # recorder so pose spans and NCC counters survive the fan-out.
        with recorder.activate():
            target.extend(_scan(search, pixels, options, angle, scale, origin,
                                threshold=threshold, skipped=misses))

    if len(poses) > 1:
        threads = search_threads()
        working_set = int(search.nbytes) + int(pixels.nbytes)
        if threads > 1 and working_set > 0:
            threads = max(1, min(threads, _POSE_GRID_MEMORY_BYTES // working_set))
        if threads > 1:
            counter("pose_batches_parallel")
            blocks: List[List] = [[] for _ in poses]
            misses: List[List] = [[] for _ in poses]
            with ThreadPoolExecutor(max_workers=threads, thread_name_prefix="pose-grid") as pool:
                futures = [pool.submit(evaluate, blocks[index], misses[index], angle, scale)
                           for index, (angle, scale) in enumerate(poses)]
                for future in futures:
                    future.result()
            return ([candidate for block in blocks for candidate in block],
                    [entry for group in misses for entry in group])
    found: List = []
    skipped: List = []
    for angle, scale in poses:
        evaluate(found, skipped, angle, scale)
    return found, skipped


@timed("match_suppression")
def _suppress(candidates: list, pixels: np.ndarray, options: TemplateSearch) -> list:
    """Select disjoint instances from pose candidates with the shared policy.

    The overlap geometry is the same rotated-rectangle measure the engine uses;
    only the record layout differs here (``(score, angle, scale, matrix)``).
    """
    h, w = pixels.shape[:2]

    def center_of(candidate) -> list:
        matrix = candidate[3]
        return list(matrix @ [(w - 1) / 2, (h - 1) / 2, 1])

    def overlap_of(candidate, previous) -> float:
        _, angle, scale, _ = candidate
        _, previous_angle, previous_scale, _ = previous
        center = center_of(candidate)
        previous_center = center_of(previous)
        return rotated_box_overlap(
            (float(center[0]), float(center[1])), (w * scale, h * scale), angle,
            (float(previous_center[0]), float(previous_center[1])),
            (w * previous_scale, h * previous_scale), previous_angle,
        )

    return keep_disjoint(overlap_of, candidates, options.overlap,
                         options.max_matches or 1000)


def _fine_radius(pixels: np.ndarray, options: TemplateSearch) -> int:
    """Return the half-size of a coarse target's full-resolution fine window.

    The radius covers the half of the support's largest centered extent (the
    corners of a centered box reach ``diagonal / 2`` under any rotation) plus
    ``fine_search_expand_px`` for coarse-centre quantization
    (``0.5 / pyramid_scale`` px) and rotation/translation coupling. It depends
    on the support, never on the whole image.

    Args:
        pixels: Rectified support pixels.
        options: Search controls.

    Returns:
        Inclusive half-side in inspection pixels, at least one pixel.
    """
    h, w = pixels.shape[:2]
    diagonal = float(np.hypot(w, h)) * options.scale_max
    return max(1, int(np.ceil(diagonal / 2.0)) + options.fine_search_expand_px)


def _local_window(search: np.ndarray, center: tuple, radius: int) -> np.ndarray:
    """Clip an axis-aligned ``center +/- radius`` window to the search area.

    Args:
        search: Full search-area pixels.
        center: Window center ``(x, y)`` in search-area coordinates.
        radius: Inclusive half-side.

    Returns:
        Window as ``(x0, y0, x1, y1)`` with ``x1``/``y1`` exclusive; the
        center is rounded to the nearest pixel so a side is at most
        ``2 * radius + 1`` unless the search area clips it.

    Raises:
        ServiceError: ``template.fine_window_too_small`` when the clipped window
            cannot hold any pose variant.
    """
    height, width = search.shape[:2]
    cx = int(round(center[0]))
    cy = int(round(center[1]))
    x0 = max(0, min(cx - radius, width - 1))
    y0 = max(0, min(cy - radius, height - 1))
    x1 = min(width, cx + radius + 1)
    y1 = min(height, cy + radius + 1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        raise ServiceError(
            "template.fine_window_too_small",
            "Fine-search neighborhood cannot hold the rotated support",
        )
    return np.array([x0, y0, x1, y1], dtype=int)


def _fine_window(search: np.ndarray, pixels: np.ndarray, center: tuple,
                 angles: list, scales: list, options: TemplateSearch,
                 skipped_poses: Optional[list] = None) -> tuple:
    """Rescan one coarse neighborhood at full resolution.

    The window is the coarse center +/- ``_fine_radius`` clipped to the search
    area. A pose variant wider or taller than the window cannot be masked
    faithfully, so it is not scanned and is reported through ``skipped_poses``
    instead of being clipped into false peaks. Poses that fit are refined.

    Args:
        search: Full search-area pixels.
        pixels: Rectified support pixels.
        center: Coarse pose center in search-area coordinates.
        angles: Pose angles to refine.
        scales: Pose scales to refine.
        options: Search controls.
        skipped_poses: Optional list receiving poses that no window could hold.

    Returns:
        ``(candidates, window, touches_border)`` with the window as
        ``(x0, y0, x1, y1)`` and a flag for windows clipped by the search area.
    """
    window = _local_window(search, center, _fine_radius(pixels, options))
    x0, y0, x1, y1 = (int(value) for value in window)
    area = search[y0:y1, x0:x1]
    poses = [(angle, scale) for angle in angles for scale in scales]
    refined, skipped = [], []
    if options.fine_angle_policy == "hierarchical" and len(poses) > 4:
        stride = max(2, int(round(len(poses) ** 0.5)))
        refined, skipped = _scan_grid(area, pixels, options, poses[::stride], (x0, y0))
        if refined:
            best = max(refined, key=lambda item: item[0])[1]
            step = options.angle_step * stride
            low = max(min(angles), best - step)
            high = min(max(angles), best + step)
            window_poses = [(angle, scale) for angle in _grid(low, high, options.angle_step)
                            for scale in scales]
            counter("fine_poses_refined", float(len(window_poses)))
            dense, dense_skipped = _scan_grid(area, pixels, options, window_poses, (x0, y0))
            refined = refined + dense
            skipped = skipped + dense_skipped
    else:
        refined, skipped = _scan_grid(area, pixels, options, poses, (x0, y0))
    if skipped:
        if not refined:
            # Nothing fits this neighborhood; surface it rather than a silent no_match.
            raise ServiceError(
                "template.fine_window_too_small",
                "Fine-search neighborhood cannot hold the rotated support",
            )
        if skipped_poses is not None:
            skipped_poses.extend(skipped)
    touches_border = x0 == 0 or y0 == 0 or x1 == search.shape[1] or y1 == search.shape[0]
    return refined, (x0, y0, x1, y1), touches_border


#: Decimated supports are pure functions of (template pixels, factor); a worker
#: process reuses them across operations without changing a single sample.
_COARSE_TEMPLATE_CACHE: "OrderedDict[Tuple[Any, float], np.ndarray]" = OrderedDict()
_COARSE_TEMPLATE_LIMIT = 8


def _coarse_template_cached(key, factor: float) -> Optional[np.ndarray]:
    """Return a previously decimated support for ``(template key, factor)``."""
    if key is None:
        return None
    entry = _COARSE_TEMPLATE_CACHE.get((key, factor))
    if entry is not None:
        _COARSE_TEMPLATE_CACHE.move_to_end((key, factor))
    return entry


def _coarse_template_store(key, factor: float, pixels: np.ndarray) -> None:
    """Remember one decimated support inside the bounded process cache."""
    if key is None:
        return
    _COARSE_TEMPLATE_CACHE[(key, factor)] = pixels
    while len(_COARSE_TEMPLATE_CACHE) > _COARSE_TEMPLATE_LIMIT:
        _COARSE_TEMPLATE_CACHE.popitem(last=False)


def _decimate(pixels: np.ndarray, search: np.ndarray, factor: float,
              template_key: Any = None) -> tuple:
    """Downscale the support and the search area for the coarse pose scan.

    Both images are low-passed with ``sigma = 0.5 / factor`` before decimation,
    which is what keeps odd-pixel translations correlate-able.  ``INTER_AREA``
    averaging removes this pass at a third of the cost but aliases the phase of
    high-frequency supports: on the acceptance corpus the correlation peak fell
    from 0.65 to 0.27 and coarse detection failed, so the anti-aliased path
    stays.  The *support* half of the work is a pure function of the template
    and the factor, so it is cached across calls.

    Args:
        pixels: Rectified support pixels.
        search: Search-area pixels.
        factor: Decimation factor in ``(0, 1)``.
        template_key: Process-local template identity for the decimated-support
            cache, or ``None`` to always recompute.

    Returns:
        ``(coarse_pixels, coarse_search, support_map, search_map, scheme)``
        where each 3x3 map takes full-resolution coordinates to that coarse
        frame.
    """
    sigma = 0.5 / factor
    mapping = decimate_matrix(factor)
    coarse_pixels = _coarse_template_cached(template_key, factor)
    if coarse_pixels is None:
        with measure("pyramid_blur"):
            coarse_pixels, _ = decimate_pixels(pixels, factor, sigma=sigma)
        _coarse_template_store(template_key, factor, coarse_pixels)
    else:
        counter("pyramid_template_cache_hit")
    with measure("pyramid_blur"):
        coarse_search, _ = decimate_pixels(search, factor, sigma=sigma)
    return coarse_pixels, coarse_search, mapping, mapping, "blur"


def request_pyramid_factor(pixels: np.ndarray, options: TemplateSearch) -> float:
    """Return the decimation factor for one request, translated for the engine.

    The v2 request spells these two quantities ``pyramid_levels`` and
    ``pyramid_max_template_size`` (published names, kept for compatibility);
    the engine calls them ``decimate_exponent`` and ``coarse_template_side``.
    This is the only place the two spellings meet, so the formula itself keeps a
    single owner in the engine.

    Args:
        pixels: Rectified support pixels; only its shape is read.
        options: The request's search controls.

    Returns:
        Decimation factor in ``(0, 1]``.
    """
    engine_options = SearchOptions(
        decimate_exponent=int(options.pyramid_levels),
        coarse_template_side=int(options.pyramid_max_template_size))
    height, width = pixels.shape[:2]
    probe = PreparedSupport(
        pixels=np.zeros((height, width), dtype=np.float32),
        valid_mask=np.ones((height, width), dtype=bool),
        anchor_support=(width / 2.0, height / 2.0),
        support_to_reference=None, reference_space=None, support_space=None,
        mask=None,
    )
    return engine_factor(probe, engine_options)


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
    factor = request_pyramid_factor(pixels, options)
    coarse_pixels = pixels
    coarse_search = search
    support_map = np.eye(3)
    search_map = np.eye(3)
    resample = "none"
    if factor < 1:
        with span("pyramid_prepare"):
            (coarse_pixels, coarse_search,
             support_map, search_map, resample) = _decimate(
                pixels, search, factor,
                _template_cache_key(request.template, context)
                if _template_cache_enabled() else None)
    astep = options.coarse_angle_step if options.coarse_fine else options.angle_step
    sstep = options.coarse_scale_step if options.coarse_fine else options.scale_step
    coarse_poses = [(angle, scale)
                    for angle in _grid(options.angle_min, options.angle_max, astep)
                    for scale in _grid(options.scale_min, options.scale_max, sstep)]
    counter("coarse_poses", float(len(coarse_poses)))
    coarse_threshold = min(options.score_threshold, options.coarse_score_threshold) if (
        factor < 1 or options.coarse_fine
    ) else options.score_threshold
    with span("coarse_search"):
        candidates, _ = _scan_grid(coarse_search, coarse_pixels, options, coarse_poses,
                                   threshold=coarse_threshold)
    if factor < 1:
        candidates = [(score, angle, scale,
                       invert_matrix(search_map) @ matrix @ support_map)
                      for score, angle, scale, matrix in candidates]
    fine_windows = []
    skipped_poses = []
    bounded_outside = False
    refined_neighborhoods = 0
    available_neighborhoods = 0
    fine_poses_per_neighborhood = 0
    fine_candidates: Optional[int] = None
    if options.coarse_fine or factor < 1:
        # Coarse candidates locate target neighborhoods; the requested match count
        # bounds how many of them deserve a full-resolution rescan.
        coarse_options = options.model_copy(update={"max_matches": 0})
        neighborhoods = _suppress(candidates, pixels, coarse_options)
        available_neighborhoods = len(neighborhoods)
        fine_candidates = options.resolved_fine_candidates()
        if fine_candidates is not None and available_neighborhoods > fine_candidates:
            neighborhoods = neighborhoods[:fine_candidates]
        refined_neighborhoods = len(neighborhoods)
        counter("fine_neighborhoods", float(available_neighborhoods))
        counter("fine_neighborhoods_refined", float(refined_neighborhoods))
        policy = options.resolved_fine_policy()
        refined = []
        whole_area = np.ascontiguousarray(search) if policy == "whole" else None
        with span("fine_search"):
            for _, angle, scale, matrix in neighborhoods:
                center = matrix @ [(w-1)/2, (h-1)/2, 1]
                angles = (_grid(max(options.angle_min, angle-astep),
                                min(options.angle_max, angle+astep), options.angle_step)
                          if options.coarse_fine else [angle])
                scales = (_grid(max(options.scale_min, scale-sstep),
                                min(options.scale_max, scale+sstep), options.scale_step)
                          if options.coarse_fine else [scale])
                poses = [(fine_angle, fine_scale)
                         for fine_angle in angles for fine_scale in scales]
                fine_poses_per_neighborhood = len(poses)
                counter("fine_poses", float(len(poses)))
                if policy == "whole":
                    # Legacy: rescan the whole search area; the pose center only labels it.
                    if not fine_windows:
                        fine_windows.append({
                            "angle": angle, "x": 0, "y": 0,
                            "width": int(search.shape[1]), "height": int(search.shape[0]),
                        })
                    found, missed = _scan_grid(whole_area, pixels, options, poses, (0, 0))
                    refined.extend(found)
                    skipped_poses.extend(missed)
                    # Variants larger than the search area are clipped here (legacy behaviour).
                    bounded_outside = bounded_outside or bool(missed)
                    continue
                local, window, touches_border = _fine_window(
                    search, pixels, center, angles, scales, options, skipped_poses)
                refined.extend(local)
                bounded_outside = bounded_outside or touches_border
                fine_windows.append({
                    "angle": angle, "x": window[0], "y": window[1],
                    "width": window[2] - window[0], "height": window[3] - window[1],
                })
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
    fine_policy = options.resolved_fine_policy()
    refined_locally = fine_policy == "neighborhood" and bool(fine_windows)
    radius = _fine_radius(pixels, options) if refined_locally else None
    return TemplateMatchResult(
        valid=bool(matches), reason=None if matches else "no_match", image=image_ref,
        template=template, matches=matches,
        diagnostics={"pyramid_scale": factor, "preprocessing": metadata.preprocessing.type,
                     "metric_frame": "inspection", "normalization_recovers_size": True,
                     "search_plan": {
                         "resample": resample,
                         "coarse_poses": len(coarse_poses),
                         "fine_poses_per_neighborhood": fine_poses_per_neighborhood,
                         "neighborhoods_available": available_neighborhoods,
                         "neighborhoods_refined": refined_neighborhoods,
                         "fine_candidates": fine_candidates,
                         "fine_angle_policy": options.fine_angle_policy,
                         "threads": search_threads(),
                     },
                     "fine_search": {"policy": fine_policy, "neighborhood_radius": radius,
                                     "expand_px": options.fine_search_expand_px,
                                     "windows": fine_windows,
                                     "skipped_poses": skipped_poses,
                                     "bounded_outside": bounded_outside}},
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
