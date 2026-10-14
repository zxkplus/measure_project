"""Supplied observations render independently of the originating operation record."""

import json

from measure_api.v2.errors import ServiceError
from measure_api.v2.families.images import encode_png
from measure_api.v2.files import ArtifactWriter
from measure_api.v2.registry import CalculationContext, Operation
from measure_api.v2.result_diagnostics import diagnostic_payloads
from measure_api.v2.result_models import (
    ResultExportRequest,
    ResultExportResult,
    VisualizationRequest,
    VisualizationResult,
)
from measure_api.v2.result_reports import encode_csv, report_rows
from measure_api.v2.result_rendering import overlays, rasterize
from measure_api.v2.result_support import prepare_result


def visualize(request: VisualizationRequest, context: CalculationContext) -> VisualizationResult:
    """Return normal vector overlays and optional committed raster pixels.

    Args:
        request: Typed supplied observations and calculation controls.
        context: Isolated calculation resources and output identity.

    Returns:
        Vector overlays and any completed shared artifact descriptors.
    """
    vectors = overlays(request)
    payloads = []
    if request.raster:
        if request.image is None:
            raise ServiceError(
                "visualization.image_required", "Raster rendering requires a source image"
            )
        image, _ = context.files.read_image(request.image)
        payloads.append(
            ("overlay.png", "raster_overlay", "png", encode_png(rasterize(image, vectors)))
        )
    if request.diagnostics:
        payloads.extend(diagnostic_payloads(request, context))
    artifacts, manifest = [], None
    if payloads:
        artifacts, manifest = ArtifactWriter(
            context.files, request.output, context.operation_id
        ).commit(payloads)
    return VisualizationResult(
        display_frame=request.display_frame,
        overlays=vectors,
        artifacts=artifacts,
        manifest=manifest,
    )


OPERATIONS = [
    Operation(
        "/visualizations",
        VisualizationRequest,
        VisualizationResult,
        visualize,
        "Render supplied observations as selectable vectors and optional raster",
        "results",
        prepare=prepare_result,
    )
]


def export_result(request: ResultExportRequest, context: CalculationContext) -> ResultExportResult:
    """Commit all requested report formats with one final completeness manifest.

    Args:
        request: Typed supplied observations and calculation controls.
        context: Isolated calculation resources and output identity.

    Returns:
        Completed report descriptors and their final manifest.
    """
    rows = report_rows(request.result)
    payloads = []
    for format_name in request.formats:
        content = (
            encode_csv(rows)
            if format_name == "csv"
            else json.dumps(
                request.result.model_dump(mode="json"), ensure_ascii=False, allow_nan=False
            ).encode("utf-8")
        )
        payloads.append(("result." + format_name, "result_report", format_name, content))
    artifacts, manifest = ArtifactWriter(
        context.files, request.output, context.operation_id
    ).commit(payloads)
    return ResultExportResult(row_count=len(rows), artifacts=artifacts, manifest=manifest)


OPERATIONS.append(
    Operation(
        "/result-exports",
        ResultExportRequest,
        ResultExportResult,
        export_result,
        "Export supplied results as precision-preserving CSV and JSON",
        "results",
        prepare=prepare_result,
    )
)
