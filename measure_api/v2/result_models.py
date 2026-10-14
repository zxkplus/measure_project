"""Supplied observations, independently selectable overlays and shared reports."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import Field, model_validator

from measure_api.v2.models import (
    CalculationRequest,
    CoordinateMapping,
    FileDescriptor,
    FileReference,
    OutputSpec,
    StrictModel,
)
from measure_api.v2.recipe_models import RunBatchResult, RunResult


class AtomicSuppliedResult(StrictModel):
    """An atomic output validated against the registered public result schema."""

    schema_version: Literal["1.0"] = "1.0"
    operation: str
    space: Literal["pixel", "physical"] = "pixel"
    value: Dict[str, Any]
    target_id: str = "target-1"
    node_id: str = "observation"
    label: Optional[str] = None
    evaluation_status: Literal["pass", "fail", "not_evaluated"] = "not_evaluated"


SuppliedResult = Union[RunResult, RunBatchResult, AtomicSuppliedResult]


class SuppliedResultRequest(CalculationRequest):
    """Supply observations inline or in a versioned shared JSON file."""

    result: Optional[SuppliedResult] = None
    result_file: Optional[FileReference] = None
    source_result: Optional[FileReference] = None
    assets: List[FileReference] = Field(default_factory=list)

    @model_validator(mode="after")
    def one_result(self) -> SuppliedResultRequest:
        """Reject ambiguous or missing supplied records.

        Returns:
            This supplied-result request with exactly one inline or file-backed record.
        """
        if (self.result is None) == (self.result_file is None):
            raise ValueError("Provide exactly one result or result_file")
        return self


class VisualizationRequest(SuppliedResultRequest):
    """Stateless selection and visibility controls belong to this request only."""

    image: Optional[FileReference] = None
    display_frame: str = "source"
    space: Literal["pixel", "physical"] = "pixel"
    mapping: Optional[CoordinateMapping] = None
    visible_nodes: Optional[List[str]] = Field(default=None, max_length=10000)
    selected_nodes: List[str] = Field(default_factory=list, max_length=10000)
    raster: bool = False
    diagnostics: bool = False
    curve_samples: int = Field(default=128, ge=16, le=4096)
    output: Optional[OutputSpec] = None

    @model_validator(mode="after")
    def output_destination(self) -> VisualizationRequest:
        """Large outputs require an upstream-owned destination.

        Returns:
            This visualization request with a destination for requested large outputs.
        """
        if (self.raster or self.diagnostics) and self.output is None:
            raise ValueError("Raster/diagnostic files require output.directory")
        return self


class OverlayPrimitive(StrictModel):
    """One stable vector primitive with upstream-controlled display state."""

    id: str
    item_id: Optional[str] = None
    target_id: str
    node_id: str
    port: str
    type: Literal["point", "line", "circle", "rectangle", "polyline", "annotation"]
    geometry: Dict[str, Any]
    frame: str
    unit: str
    label: Optional[str] = None
    valid: bool
    visible: bool
    selected: bool
    style: Dict[str, Any] = Field(default_factory=lambda: {"color": "#00ff00", "width": 1})


class VisualizationResult(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    display_frame: str
    overlays: List[OverlayPrimitive]
    artifacts: List[FileDescriptor] = Field(default_factory=list)
    manifest: Optional[FileDescriptor] = None


class ResultExportRequest(SuppliedResultRequest):
    """Export machine-readable values; numerical precision stays unchanged."""

    formats: List[Literal["csv", "json"]] = Field(default_factory=lambda: ["json"], min_length=1)
    output: OutputSpec

    @model_validator(mode="after")
    def unique_formats(self) -> ResultExportRequest:
        """Reject duplicate export names.

        Returns:
            This export request after ensuring each output format occurs once.
        """
        if len(set(self.formats)) != len(self.formats):
            raise ValueError("Export formats must be unique")
        return self


class ResultExportResult(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    row_count: int
    artifacts: List[FileDescriptor]
    manifest: FileDescriptor
