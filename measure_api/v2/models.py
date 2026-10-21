"""Shared public types for the stateless FastAPI calculation contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Generic, List, Literal, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    """Reject undeclared fields and non-finite numeric values."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class FileReference(StrictModel):
    """A root-relative reference, or an allowed absolute pathname."""

    root: Optional[str] = None
    path: str = Field(min_length=1)
    sha256: Optional[str] = Field(default=None, pattern=r"^[a-fA-F0-9]{64}$")
    provenance: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def check_reference(self) -> FileReference:
        """Reject ambiguous or nonportable reference forms.

        Returns:
            This reference after rejecting ambiguous root/absolute-path combinations.
        """
        if "\x00" in self.path:
            raise ValueError("File path contains a NUL character")
        if self.root is None and not Path(self.path).is_absolute():
            raise ValueError("Relative paths require a shared root alias")
        if self.root is not None and Path(self.path).is_absolute():
            raise ValueError("Root alias and absolute pathname are mutually exclusive")
        return self


class OutputSpec(StrictModel):
    """Directory for an isolated, committed operation artifact set."""

    directory: FileReference


class CalculationRequest(StrictModel):
    """Common scheduling controls; waiting never changes the operation deadline."""

    mode: Literal["sync", "async"] = "sync"
    wait_ms: int = Field(default=10000, ge=0, le=60000)
    timeout_ms: int = Field(default=60000, ge=1, le=3600000)
    idempotency_key: Optional[str] = Field(default=None, min_length=1, max_length=200)


class Point(StrictModel):
    """Image position with x right and y down, measured at pixel centers."""

    x: float
    y: float


class ImageMetadata(StrictModel):
    """Decoded pixel layout and its measurement frame."""

    width: int
    height: int
    channels: int
    dtype: str
    pixel_format: Literal["GRAY8", "BGR8", "RGB8", "FLOAT32"]
    frame: str = "source"
    unit: Literal["px"] = "px"


class FileDescriptor(StrictModel):
    """A complete immutable shared artifact."""

    kind: str
    format: str
    schema_version: str = "1.0"
    file: FileReference
    sha256: str
    size_bytes: int


class ErrorDetail(StrictModel):
    """Stable error information suitable for upstream diagnostics."""

    code: str
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)
    field_path: Optional[str] = None
    trace_id: str
    retryable: bool = False


class ErrorResponse(StrictModel):
    """HTTP error envelope, optionally identifying an accepted failed operation."""

    error: ErrorDetail
    operation_id: Optional[str] = None


ResultT = TypeVar("ResultT")


class OperationResponse(StrictModel, Generic[ResultT]):
    """Queryable operation state; result observations are family-specific."""

    operation_id: str
    operation: str
    state: Literal[
        "queued", "running", "cancelling", "succeeded", "failed", "cancelled", "timed_out",
    ]
    trace_id: str
    result: Optional[ResultT] = None
    error: Optional[ErrorDetail] = None
    created_at: str
    completed_at: Optional[str] = None
    diagnostics: Dict[str, Any] = Field(default_factory=dict)


class ServiceSettings(StrictModel):
    """Injected deployment boundaries, independent of upstream asset ownership."""

    shared_roots: Dict[str, Path]
    writable_roots: Optional[List[str]] = None
    max_workers: int = Field(default=2, ge=1, le=64)
    max_pending: int = Field(default=16, ge=1, le=10000)
    max_input_bytes: int = Field(default=256 * 1024 * 1024, ge=1)
    max_image_pixels: int = Field(default=64 * 1024 * 1024, ge=1)
    state_directory: Optional[Path] = None
    operation_retention_seconds: float = Field(default=86400, gt=0)
    expired_retention_seconds: float = Field(default=86400, gt=0)
    max_retained_operations: int = Field(default=10000, ge=1)
    max_worker_memory_bytes: int = Field(default=2 * 1024 * 1024 * 1024, ge=16 * 1024 * 1024)
    cache_max_bytes: int = Field(default=256 * 1024 * 1024, ge=0)
    cache_max_entries: int = Field(default=64, ge=0)
    shutdown_wait_seconds: float = Field(default=5.0, ge=0, le=60)

    @field_validator("shared_roots")
    @classmethod
    def normalize_roots(cls, value: Dict[str, Path]) -> Dict[str, Path]:
        """Make aliases resolve identically in the HTTP and compute processes.

        Args:
            value: Supplied value to normalize under the public schema and coordinate
                convention.

        Returns:
            Absolute root paths preserving each configured shared-directory alias.
        """
        if not value or any(not key for key in value):
            raise ValueError("At least one nonempty shared root alias is required")
        return {key: path.resolve() for key, path in value.items()}

    @model_validator(mode="after")
    def check_writable_roots(self) -> ServiceSettings:
        """Only configured aliases can authorize output writes.

        Returns:
            This deployment after verifying each writable alias is configured.
        """
        if self.writable_roots is not None:
            if set(self.writable_roots) - set(self.shared_roots):
                raise ValueError("Writable roots must be configured shared root aliases")
        return self


class CoordinateMapping(StrictModel):
    """Explicit source-to-destination mapping of image pixel centers."""

    source_frame: str
    destination_frame: str
    type: Literal["identity", "similarity", "affine", "projective", "nonlinear"]
    direction: Literal["source_to_destination"] = "source_to_destination"
    matrix: Optional[List[List[float]]] = None
    inverse_matrix: Optional[List[List[float]]] = None
    unit: str = "px"
    details: Dict[str, Any] = Field(default_factory=dict)
