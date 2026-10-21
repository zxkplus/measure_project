"""Portable calibration assets and extensible validation shared by calculations."""

from __future__ import annotations

import hashlib
import importlib
import json
import pkgutil
from typing import Any, Dict, List, Literal, Optional, Type, Union

import numpy as np
from pydantic import Field, ValidationError, model_validator

from measure_api.v2.errors import ServiceError
from measure_api.v2.models import FileReference, StrictModel
from measure_api.v2.registry import CalculationContext

PhysicalUnit = Literal["mm", "cm", "m", "um"]
UNIT_MM = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "um": 0.001}


class ImageSize(StrictModel):
    """Applicable pixel dimensions in width/height order."""

    width: int = Field(gt=0, le=32768)
    height: int = Field(gt=0, le=32768)


class CalibrationApplicability(StrictModel):
    """Explicit imaging setup and coordinate frame used during calibration."""

    camera_id: str = Field(min_length=1)
    setup_id: str = Field(min_length=1)
    image_size: ImageSize
    frame: str = Field(default="source", min_length=1)


class CalibrationAsset(StrictModel):
    """Common portable envelope; each family validates its typed parameters."""

    schema_version: Literal["1.0"] = "1.0"
    kind: str
    unit: PhysicalUnit
    applicability: CalibrationApplicability
    parameters: Any


class ScalarParameters(StrictModel):
    """Uniform scale for a telecentric or explicitly orthographic setup."""

    scale_per_px: float = Field(gt=0)
    spacing_px: float = Field(gt=0)
    square_size: float = Field(gt=0)
    assumption: Literal["uniform_orthographic"] = "uniform_orthographic"

    @model_validator(mode="after")
    def consistent_scale(self) -> ScalarParameters:
        """Reject conflicting scale and physical-board provenance.

        Returns:
            This scalar calibration with scale consistent with its physical-board spacing.
        """
        if not np.isclose(self.scale_per_px, self.square_size / self.spacing_px, rtol=1e-8,
                          atol=0):
            raise ValueError("scale_per_px must equal square_size divided by spacing_px")
        return self


class ScalarCalibration(CalibrationAsset):
    kind: Literal["scalar"] = "scalar"
    parameters: ScalarParameters


class CameraParameters(StrictModel):
    """OpenCV pinhole intrinsics; no physical scene scale is implied."""

    camera_matrix: List[List[float]]
    distortion: List[float]
    reprojection_error_px: float = Field(default=0, ge=0)

    @model_validator(mode="after")
    def check_camera(self) -> CameraParameters:
        matrix = np.array(self.camera_matrix, dtype=float)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError("camera_matrix must be a finite 3 by 3 matrix")
        if (matrix[0, 0] <= 0 or matrix[1, 1] <= 0
                or not np.allclose(matrix[2], [0, 0, 1])
                or abs(matrix[0, 1]) > 1e-10 or abs(matrix[1, 0]) > 1e-10):
            raise ValueError("camera_matrix must be a positive pinhole intrinsic matrix")
        if len(self.distortion) not in {4, 5, 8, 12, 14}:
            raise ValueError("distortion requires 4, 5, 8, 12 or 14 coefficients")
        return self


class CameraCalibration(CalibrationAsset):
    kind: Literal["camera"] = "camera"
    parameters: CameraParameters


class CalibrationInput(StrictModel):
    """An upstream value or a verified shared JSON asset, never registration."""

    value: Optional[Union[ScalarCalibration, CameraCalibration, CalibrationAsset]] = None
    file: Optional[FileReference] = None

    @model_validator(mode="after")
    def one_source(self) -> CalibrationInput:
        if (self.value is None) == (self.file is None):
            raise ValueError("Provide exactly one calibration value or file")
        return self


def calibration_bytes(asset: CalibrationAsset) -> bytes:
    """Canonical non-pickle asset representation for portable content identity.

    Args:
        asset: Validated calibration value, including its declared setup and frame.

    Returns:
        Canonical UTF-8 JSON bytes for the complete portable calibration.
    """
    return json.dumps(asset.model_dump(mode="json"), sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def calibration_digest(asset: CalibrationAsset) -> str:
    """Content identity independent of upstream pathname and service state.

    Args:
        asset: Validated calibration value, including its declared setup and frame.

    Returns:
        SHA-256 digest of the canonical asset bytes, independent of pathname.
    """
    return hashlib.sha256(calibration_bytes(asset)).hexdigest()


def validate_calibration(value: Any) -> CalibrationAsset:
    """Validate known asset kinds and discover additional family-owned schemas.

    Family modules expose CALIBRATION_ASSETS mapping kinds to public model classes.
    This permits plane/stereo extensions without editing shared utility routes.

    Args:
        value: Inline calibration object whose versioned kind/schema is checked.

    Returns:
        Common portable envelope; each family validates its typed parameters.
    """
    from measure_api.v2 import families

    models: Dict[str, Type[CalibrationAsset]] = {
        "scalar": ScalarCalibration, "camera": CameraCalibration,
    }
    for item in pkgutil.iter_modules(families.__path__):
        module = importlib.import_module(f"{families.__name__}.{item.name}")
        models.update(getattr(module, "CALIBRATION_ASSETS", {}))
    data = value.model_dump(mode="json") if isinstance(value, CalibrationAsset) else value
    if not isinstance(data, dict) or data.get("kind") not in models:
        raise ServiceError("calibration.unsupported_kind", "Unsupported calibration asset kind")
    try:
        return models[data["kind"]].model_validate(data)
    except (ValidationError, ValueError) as error:
        raise ServiceError("calibration.invalid_asset", "Invalid calibration asset",
                           details={"reason": str(error)}) from error


def load_calibration(
    source: CalibrationInput, context: CalculationContext,
) -> CalibrationAsset:
    """Load a pinned upstream JSON asset or validate the supplied typed value.

    Args:
        source: Inline typed calibration or pinned upstream JSON file reference.
        context: Isolated operation identity, resource limits and verified shared-file
            access.

    Returns:
        Common portable envelope; each family validates its typed parameters.
    """
    if source.value is not None:
        return validate_calibration(source.value)
    data, _ = context.files.read_bytes(source.file)
    try:
        value = json.loads(data)
    except (ValueError, UnicodeError) as error:
        raise ServiceError("calibration.invalid_json", "Calibration must be UTF-8 JSON") from error
    return validate_calibration(value)


def check_applicability(
    asset: CalibrationAsset, expected: CalibrationApplicability,
) -> None:
    """Prevent using correct mathematics with another imaging setup or frame.

    Args:
        asset: Validated calibration value, including its declared setup and frame.
        expected: Required image, camera, setup and coordinate-frame applicability.
    """
    if asset.applicability != expected:
        raise ServiceError("calibration.applicability_mismatch",
                           "Calibration does not apply to this image/setup/frame",
                           details={"expected": asset.applicability.model_dump(),
                                    "received": expected.model_dump()})
