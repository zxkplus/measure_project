"""Versioned upstream-owned recipes and typed calculation results."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import Field, model_validator

from measure_api.v2.calibration_assets import CalibrationApplicability, CalibrationInput
from measure_api.v2.families.alignment import ControlTemplate
from measure_api.v2.families.images import RotatedROI
from measure_api.v2.families.templates import TemplateSearch
from measure_api.v2.models import (
    CalculationRequest,
    CoordinateMapping,
    FileDescriptor,
    FileReference,
    OutputSpec,
    StrictModel,
)
from measure_api.v2.versions import RuntimeIdentity


class Binding(StrictModel):
    """A named typed port in one target; arrays require an explicit binding list."""

    node: str = Field(min_length=1)
    port: str = Field(min_length=1)
    space: Literal["pixel", "physical"] = "pixel"


class RecipeNode(StrictModel):
    id: str = Field(min_length=1, max_length=128)
    operation: str
    label: Optional[str] = None
    enabled: bool = True
    required: bool = True
    parameters: Dict[str, Any] = Field(default_factory=dict)
    inputs: Dict[str, Union[Binding, List[Binding]]] = Field(default_factory=dict)
    space: Literal["pixel", "physical"] = "pixel"


class Localization(StrictModel):
    """One template pose search or a fixed reference ROI, never implicit selection."""

    template: Optional[FileReference] = None
    search: TemplateSearch = Field(default_factory=TemplateSearch)
    fixed_roi: Optional[RotatedROI] = None
    max_targets: int = Field(default=1000, ge=1, le=1000)

    @model_validator(mode="after")
    def one_source(self) -> Localization:
        if (self.template is None) == (self.fixed_roi is None):
            raise ValueError("Localization requires exactly one template or fixed_roi")
        return self


class TargetAlignment(StrictModel):
    width: int = Field(ge=1, le=32768)
    height: int = Field(ge=1, le=32768)
    reference_to_patch: CoordinateMapping
    controls: List[ControlTemplate] = Field(default_factory=list, max_length=1000)
    minimum_controls: int = Field(default=2, ge=2)
    robust: bool = True
    max_residual: float = Field(default=3, gt=0)
    estimate_scale: bool = True
    failure_policy: Literal["fail", "rigid_fallback"] = "fail"
    border_policy: Literal["constant", "replicate", "reject"] = "constant"


class MetricSetup(StrictModel):
    calibration: CalibrationInput
    applicability: CalibrationApplicability
    physical_frame: str = Field(default="physical", min_length=1)


class ImageStage(StrictModel):
    """A declared correction or stitch operation with upstream-supplied parameters."""

    id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    operation: Literal["image-undistortions", "image-stitches"]
    parameters: Dict[str, Any]


class TargetPolicy(StrictModel):
    minimum: Optional[int] = Field(default=None, ge=0, le=1000)
    maximum: Optional[int] = Field(default=None, ge=0, le=1000)
    no_targets: Literal["not_evaluated", "fail", "pass"] = "not_evaluated"

    @model_validator(mode="after")
    def ordered(self) -> TargetPolicy:
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("Minimum target count exceeds maximum")
        return self


class Recipe(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    reference: Optional[FileReference] = None
    source_frame: str = Field(default="source", min_length=1)
    stages: List[ImageStage] = Field(default_factory=list, max_length=8)
    localization: Optional[Localization] = None
    alignment: Optional[TargetAlignment] = None
    metric: Optional[MetricSetup] = None
    nodes: List[RecipeNode] = Field(min_length=1, max_length=1000)
    required_outputs: List[Binding] = Field(default_factory=list, max_length=1000)
    target_policy: TargetPolicy = Field(default_factory=TargetPolicy)


class RecipeRequest(CalculationRequest):
    recipe: Optional[Recipe] = None
    recipe_file: Optional[FileReference] = None
    source_recipe: Optional[FileReference] = None
    assets: List[FileReference] = Field(default_factory=list)

    @model_validator(mode="after")
    def one_recipe(self) -> RecipeRequest:
        if (self.recipe is None) == (self.recipe_file is None):
            raise ValueError("Provide exactly one recipe or recipe_file")
        return self


class RunRequest(RecipeRequest):
    image: FileReference
    output: Optional[OutputSpec] = None
    retain_stage_images: bool = False

    @model_validator(mode="after")
    def retention_destination(self) -> RunRequest:
        if self.retain_stage_images and self.output is None:
            raise ValueError("Retaining stage images requires output.directory")
        return self


class PreviewRequest(RunRequest):
    selected_nodes: List[str] = Field(min_length=1, max_length=1000)
    upstream_revision: str
    preview_token: str


class RunBatchItem(StrictModel):
    id: str = Field(min_length=1)
    image: Optional[FileReference]
    error: Optional[Dict[str, Any]] = None
    assets: List[FileReference] = Field(default_factory=list)
    recipe: Optional[Recipe] = None
    recipe_file: Optional[FileReference] = None

    @model_validator(mode="after")
    def unambiguous(self) -> RunBatchItem:
        if self.recipe is not None and self.recipe_file is not None:
            raise ValueError("Batch item cannot supply both recipe forms")
        return self


class RunBatchRequest(RecipeRequest):
    items: List[RunBatchItem] = Field(min_length=1, max_length=1000)
    output: Optional[OutputSpec] = None

    @model_validator(mode="after")
    def unique_items(self) -> RunBatchRequest:
        if len({item.id for item in self.items}) != len(self.items):
            raise ValueError("Batch item IDs must be unique")
        return self


class RecipeValidationResult(StrictModel):
    valid: bool
    recipe_id: str
    recipe_version: str
    configuration_sha256: str
    execution_order: List[str]
    issues: List[Dict[str, Any]] = Field(default_factory=list)
    resolved_recipe: Recipe
    assets: List[FileReference] = Field(default_factory=list)


class NodeOutcome(StrictModel):
    id: str
    operation: str
    state: Literal["succeeded", "failed", "blocked"]
    valid: bool
    result: Optional[Dict[str, Any]] = None
    pixel_result: Optional[Dict[str, Any]] = None
    physical_result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None


class TargetOutcome(StrictModel):
    target_id: str
    valid: bool
    evaluation_status: Literal["pass", "fail", "not_evaluated"]
    mapping: Optional[CoordinateMapping] = None
    metric_mapping: Optional[CoordinateMapping] = None
    physical_mapping: Optional[CoordinateMapping] = None
    alignment: Optional[Dict[str, Any]] = None
    localization: Optional[Dict[str, Any]] = None
    nodes: List[NodeOutcome] = Field(default_factory=list)
    reason: Optional[str] = None


class RunResult(RecipeValidationResult):
    runtime_identity: Optional[RuntimeIdentity] = None
    sources: List[FileReference] = Field(default_factory=list)
    measurement_image: Optional[FileReference] = None
    image_stages: List[Dict[str, Any]] = Field(default_factory=list)
    status: Literal["completed", "no_targets", "partial_failure"]
    evaluation_status: Literal["pass", "fail", "not_evaluated"]
    targets: List[TargetOutcome] = Field(default_factory=list)
    stage_timings: Dict[str, float] = Field(default_factory=dict)
    upstream_revision: Optional[str] = None
    preview_token: Optional[str] = None
    artifacts: List[FileDescriptor] = Field(default_factory=list)
    manifest: Optional[FileDescriptor] = None


class BatchOutcome(StrictModel):
    id: str
    state: Literal["succeeded", "failed"]
    valid: bool
    result: Optional[RunResult] = None
    error: Optional[Dict[str, Any]] = None


class RunBatchResult(StrictModel):
    valid: bool
    items: List[BatchOutcome]
    artifacts: List[FileDescriptor] = Field(default_factory=list)
    manifest: Optional[FileDescriptor] = None
