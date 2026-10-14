"""Image calculations exposed without registration or GUI runtime."""

import io
from typing import List, Literal

import cv2
import numpy as np
from pydantic import Field

from measure_api.v2.errors import ServiceError
from measure_api.v2.files import ArtifactWriter, image_metadata
from measure_api.v2.models import (
    CalculationRequest, CoordinateMapping, FileDescriptor, FileReference, ImageMetadata,
    OutputSpec, Point, StrictModel,
)
from measure_api.v2.preprocessing import (
    Preprocessing, RawPreprocessing, apply_preprocessing, display_preview,
)
from measure_api.v2.registry import CalculationContext, Operation


class ImageInspectionRequest(CalculationRequest):
    """Inspect one committed shared image."""

    image: FileReference


class ImageInspectionResult(StrictModel):
    """Actual pixel metadata and verified content identity."""

    metadata: ImageMetadata
    source: FileReference


def inspect_image(
    request: ImageInspectionRequest, context: CalculationContext,
) -> ImageInspectionResult:
    """Decode actual pixels and report metadata plus verified file identity.

    Args:
        request: Typed inspect one committed shared image.
        context: Isolated operation identity, resource limits and verified shared-file
            access.

    Returns:
        Actual pixel metadata and verified content identity.
    """
    image, verified = context.files.read_image(request.image)
    return ImageInspectionResult(metadata=image_metadata(image), source=verified)


OPERATIONS = [Operation(
    "/image-inspections", ImageInspectionRequest, ImageInspectionResult, inspect_image,
    "Inspect a committed shared image", "images", {"metadata": "image_metadata"},
)]


class ImagePreviewRequest(ImageInspectionRequest):
    """Make a bounded 8-bit display raster in a requested output directory."""

    output: OutputSpec
    max_width: int = Field(default=1024, ge=1, le=32768)
    max_height: int = Field(default=1024, ge=1, le=32768)


class ImageOutputResult(StrictModel):
    """Image data descriptors plus frame provenance and a completion marker."""

    metadata: ImageMetadata
    source: FileReference
    mapping: CoordinateMapping
    artifacts: List[FileDescriptor]
    manifest: FileDescriptor


def matrix_mapping(
    matrix: np.ndarray, destination: str, kind: str = "similarity",
) -> CoordinateMapping:
    """Describe an invertible source-to-destination image transformation.

    Args:
        matrix: Finite 3-by-3 forward transform in homogeneous xy coordinates.
        destination: Coordinate-frame name assigned to the transformed image or patch.
        kind: Supported transformation or legacy measurement kind to interpret.

    Returns:
        Explicit source-to-destination mapping of image pixel centers.
    """
    return CoordinateMapping(
        source_frame="source", destination_frame=destination, type=kind,
        matrix=matrix.tolist(), inverse_matrix=np.linalg.inv(matrix).tolist(),
    )


def encode_png(image: np.ndarray) -> bytes:
    """Encode a lossless display image in a computation worker.

    Args:
        image: Display-ready gray or BGR pixel array to encode losslessly.

    Returns:
        Lossless PNG bytes; an encoding failure raises a structured service error.
    """
    success, data = cv2.imencode(".png", image)
    if not success:
        raise ServiceError("image.encoding_failed", "Could not encode output image")
    return data.tobytes()


def preview_image(
    request: ImagePreviewRequest, context: CalculationContext,
) -> ImageOutputResult:
    """Preserve aspect ratio while emitting a preview with pixel-center mapping.

    Args:
        request: Typed make a bounded 8-bit display raster in a requested output directory.
        context: Isolated operation identity, resource limits and verified shared-file
            access.

    Returns:
        Image data descriptors plus frame provenance and a completion marker.
    """
    image, verified = context.files.read_image(request.image)
    height, width = image.shape[:2]
    scale = min(1.0, request.max_width / width, request.max_height / height)
    new_width, new_height = max(1, round(width * scale)), max(1, round(height * scale))
    preview = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    sx, sy = new_width / width, new_height / height
    matrix = np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]])
    writer = ArtifactWriter(context.files, request.output, context.operation_id)
    artifacts, manifest = writer.commit(
        [("preview.png", "preview", "png", encode_png(preview))],
    )
    return ImageOutputResult(
        metadata=image_metadata(preview, "preview"), source=verified,
        mapping=matrix_mapping(matrix, "preview", "affine"),
        artifacts=artifacts, manifest=manifest,
    )


OPERATIONS.append(Operation(
    "/image-previews", ImagePreviewRequest, ImageOutputResult, preview_image,
    "Create a bounded display preview", "images", {"image": "image"},
))


class ImagePreprocessingRequest(ImageInspectionRequest):
    """Produce typed numeric pixels and an independent display preview."""

    preprocessing: Preprocessing = Field(default_factory=RawPreprocessing)
    output: OutputSpec


def preprocess_image(
    request: ImagePreprocessingRequest, context: CalculationContext,
) -> ImageOutputResult:
    """Retain exact numeric preprocessing data in a safe NumPy payload.

    Args:
        request: Typed produce typed numeric pixels and an independent display preview.
        context: Isolated operation identity, resource limits and verified shared-file
            access.

    Returns:
        Image data descriptors plus frame provenance and a completion marker.
    """
    image, verified = context.files.read_image(request.image)
    processed = apply_preprocessing(image, request.preprocessing)
    data = io.BytesIO()
    np.save(data, processed, allow_pickle=False)
    writer = ArtifactWriter(context.files, request.output, context.operation_id)
    artifacts, manifest = writer.commit([
        ("numeric.npy", "numeric", "npy", data.getvalue()),
        ("preview.png", "preview", "png", encode_png(display_preview(processed))),
    ])
    metadata = ImageMetadata(
        width=processed.shape[1], height=processed.shape[0],
        channels=1 if processed.ndim == 2 else processed.shape[2], dtype=str(processed.dtype),
        pixel_format="FLOAT32" if processed.dtype == np.float32 else "GRAY8",
        frame="preprocessed",
    )
    return ImageOutputResult(
        metadata=metadata, source=verified,
        mapping=matrix_mapping(np.eye(3), "preprocessed", "identity"),
        artifacts=artifacts, manifest=manifest,
    )


OPERATIONS.append(Operation(
    "/image-preprocessings", ImagePreprocessingRequest, ImageOutputResult, preprocess_image,
    "Preprocess numeric pixels and create a separate display preview", "images",
    {"numeric": "numeric_image", "preview": "image"},
))


class RotatedROI(StrictModel):
    """Full output dimensions; clockwise angle from the source image x axis."""

    center: Point
    width: int = Field(ge=1, le=32768)
    height: int = Field(ge=1, le=32768)
    angle_degrees: float = Field(default=0, ge=-360, le=360)


class ImageCropRequest(ImageInspectionRequest):
    """Straighten a rotated rectangle, preserving mapping to source pixels."""

    roi: RotatedROI
    output: OutputSpec
    interpolation: Literal["nearest", "linear", "cubic"] = "linear"
    border_policy: Literal["reject", "constant", "replicate"] = "reject"
    border_value: int = Field(default=0, ge=0, le=255)


def roi_to_patch_matrix(roi: RotatedROI) -> np.ndarray:
    """Map clockwise ROI coordinates to a patch using exact pixel-center geometry.

    Args:
        roi: Source center, integer patch dimensions, and clockwise angle in degrees.

    Returns:
        Invertible 3-by-3 source-to-patch transform centered at (size - 1) / 2.
    """
    angle = np.deg2rad(roi.angle_degrees)
    cos, sin = np.cos(angle), np.sin(angle)
    cx, cy = (roi.width - 1) / 2, (roi.height - 1) / 2
    return np.array([
        [cos, sin, cx - cos * roi.center.x - sin * roi.center.y],
        [-sin, cos, cy + sin * roi.center.x - cos * roi.center.y],
        [0, 0, 1],
    ])


def crop_image(
    request: ImageCropRequest, context: CalculationContext,
) -> ImageOutputResult:
    """Crop with explicit clockwise pose and exact pixel-center provenance.

    Args:
        request: Typed straighten a rotated rectangle, preserving mapping to source pixels.
        context: Isolated operation identity, resource limits and verified shared-file
            access.

    Returns:
        Image data descriptors plus frame provenance and a completion marker.
    """
    image, verified = context.files.read_image(request.image)
    roi = request.roi
    if roi.width * roi.height > context.settings.max_image_pixels:
        raise ServiceError("image.too_large", "Crop exceeds configured pixel limit")
    matrix = roi_to_patch_matrix(roi)
    inverse = np.linalg.inv(matrix)
    corners = np.array([[0, 0, 1], [roi.width - 1, 0, 1],
                        [0, roi.height - 1, 1], [roi.width - 1, roi.height - 1, 1]])
    source = (inverse @ corners.T).T
    if request.border_policy == "reject":
        tolerance = 1e-8
        if (source[:, 0].min() < -tolerance or source[:, 1].min() < -tolerance
                or source[:, 0].max() > image.shape[1] - 1 + tolerance
                or source[:, 1].max() > image.shape[0] - 1 + tolerance):
            raise ServiceError(
                "image.roi_outside", "Rotated crop includes pixels outside the source",
            )
    modes = {"nearest": cv2.INTER_NEAREST, "linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC}
    border_mode = (
        cv2.BORDER_REPLICATE if request.border_policy == "replicate" else cv2.BORDER_CONSTANT
    )
    cropped = cv2.warpAffine(
        image, matrix[:2], (roi.width, roi.height), flags=modes[request.interpolation],
        borderMode=border_mode, borderValue=(request.border_value,) * 3,
    )
    writer = ArtifactWriter(context.files, request.output, context.operation_id)
    artifacts, manifest = writer.commit([
        ("crop.png", "crop", "png", encode_png(cropped)),
    ])
    return ImageOutputResult(
        metadata=image_metadata(cropped, "crop"), source=verified,
        mapping=matrix_mapping(matrix, "crop"), artifacts=artifacts, manifest=manifest,
    )


OPERATIONS.append(Operation(
    "/image-crops", ImageCropRequest, ImageOutputResult, crop_image,
    "Straighten a rotated full-size ROI", "images", {"image": "image", "mapping": "transform"},
))
