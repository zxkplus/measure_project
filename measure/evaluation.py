"""Product acceptance and measured quality, kept independent of observation validity.

Three concepts stay separate on purpose:

* execution status — did the calculation run;
* observation validity — is this measurement trustworthy;
* evaluation status — does a trustworthy measurement meet the limits.

An invalid observation never becomes a zero measurement, and a failed product
limit never invalidates the observation that produced it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Tuple

import numpy as np

from measure.computation_types import (
    Observation,
    Space,
    XY,
    contract_error,
    invalid,
    observed,
    require_finite,
    require_finite_xy,
)

INVALID_POLICIES = ("not_evaluated", "fail", "error", "ignore")
STATUS_PASS = "pass"
STATUS_FAIL = "fail"
STATUS_NOT_EVALUATED = "not_evaluated"


@dataclass(frozen=True)
class LimitsSpec:
    """Inclusive limits plus the explicit policy for invalid observations."""

    minimum: Optional[float] = None
    maximum: Optional[float] = None
    unit: str = "px"
    invalid_policy: str = "not_evaluated"

    def __post_init__(self) -> None:
        if self.minimum is None and self.maximum is None:
            raise contract_error("contract.invalid_parameter",
                                 "at least one limit is required", "limits")
        for name in ("minimum", "maximum"):
            value = getattr(self, name)
            if value is not None:
                require_finite(value, f"limits.{name}")
        if (self.minimum is not None and self.maximum is not None
                and self.minimum > self.maximum):
            raise contract_error("contract.invalid_parameter",
                                 "minimum exceeds maximum", "limits.minimum")
        if not isinstance(self.unit, str) or not self.unit:
            raise contract_error("contract.invalid_parameter", "unit must be non-empty",
                                 "limits.unit")
        if self.invalid_policy not in INVALID_POLICIES:
            raise contract_error("contract.invalid_parameter",
                                 f"unknown invalid_policy {self.invalid_policy!r}",
                                 "limits.invalid_policy")


@dataclass(frozen=True)
class EvaluationResult:
    """A measurement together with its product verdict."""

    measurement: Observation[float]
    evaluation_status: str
    calculation: str
    input_count: int
    valid_count: int
    minimum: Optional[float]
    maximum: Optional[float]
    invalid_policy: str

    @property
    def valid(self) -> bool:
        """Return whether the underlying measurement is usable."""
        return self.measurement.valid


@dataclass(frozen=True)
class QualitySummary:
    """What is actually known about one measurement's sampling quality."""

    sample_count: Optional[int] = None
    rms_error: Optional[float] = None
    requested_probe_count: Optional[int] = None
    successful_probe_count: Optional[int] = None
    coverage_ratio: Optional[float] = None
    amplitude_mean: Optional[float] = None
    amplitude_abs_min: Optional[float] = None


def _check_unit(space: Space, limits: LimitsSpec, field_name: str = "limits.unit") -> None:
    """Require the observation and limit units to match before judging."""
    if space.unit != limits.unit:
        raise contract_error("evaluation.unit_mismatch",
                             f"limits unit {limits.unit!r} does not match observation "
                             f"unit {space.unit!r}", field_name)


def _status_for(value: Optional[float], space: Space, limits: LimitsSpec,
                calculation: str, input_count: int, valid_count: int,
                reason: Optional[str]) -> EvaluationResult:
    """Build the verdict for one aggregate value."""
    if value is None:
        if limits.invalid_policy == "error":
            raise contract_error("evaluation.invalid_input",
                                 reason or "input observations are invalid", "measurement")
        status = STATUS_FAIL if limits.invalid_policy == "fail" else STATUS_NOT_EVALUATED
        measurement = invalid(space, reason or "invalid_input")
    else:
        passed = ((limits.minimum is None or value >= limits.minimum)
                  and (limits.maximum is None or value <= limits.maximum))
        status = STATUS_PASS if passed else STATUS_FAIL
        measurement = observed(space, value)
    return EvaluationResult(
        measurement=measurement, evaluation_status=status, calculation=calculation,
        input_count=input_count, valid_count=valid_count,
        minimum=limits.minimum, maximum=limits.maximum,
        invalid_policy=limits.invalid_policy,
    )


def evaluate_range(value: Observation[float], limits: LimitsSpec) -> EvaluationResult:
    """Evaluate one observed scalar against inclusive limits.

    Args:
        value: Scalar observation.
        limits: Inclusive limits and invalid policy.

    Returns:
        The unchanged measurement plus its verdict.
    """
    _check_unit(value.space, limits)
    usable = value.valid and value.value is not None
    return _status_for(float(value.value) if usable else None, value.space, limits,
                       "inclusive_range", 1, int(usable),
                       None if usable else (value.reason or "invalid_input"))


def diameter_variation(
    values: Sequence[Observation[float]], limits: LimitsSpec,
) -> EvaluationResult:
    """Evaluate ``max - min`` across supplied diameters.

    Args:
        values: Diameter observations; frames may differ, units may not.
        limits: Inclusive limits and invalid policy.

    Returns:
        The variation measurement plus verdict; fewer than two usable values is
        never reported as zero.
    """
    observations = tuple(values)
    for observation in observations:
        _check_unit(observation.space, limits)
    usable = [float(item.value) for item in observations
              if item.valid and item.value is not None]
    count = len(observations)
    complete = len(usable) == count
    if limits.invalid_policy == "ignore":
        complete = len(usable) >= 2
    value = (max(usable) - min(usable)) if complete and len(usable) >= 2 else None
    frames = {item.space.frame for item in observations}
    frame = next(iter(frames)) if len(frames) == 1 else "aggregate"
    reason = ("insufficient_valid_samples" if complete and len(usable) < 2
              else "invalid_input")
    return _status_for(value, Space(frame, limits.unit), limits,
                       "max_diameter_minus_min_diameter", count, len(usable), reason)


def centerline_deviation(
    points: Sequence[Observation[XY]], limits: LimitsSpec, *, space: Space,
) -> EvaluationResult:
    """Evaluate the horizontal range ``max(x) - min(x)`` of supplied centres.

    Args:
        points: Centre observations in one frame.
        limits: Inclusive limits and invalid policy.
        space: Frame and unit the deviation is reported in.

    Returns:
        The deviation measurement plus verdict.
    """
    observations = tuple(points)
    for observation in observations:
        if observation.space.frame != space.frame or observation.space.unit != space.unit:
            raise contract_error("contract.space_mismatch",
                                 "centerline observations must share one frame",
                                 "points")
    usable = [require_finite_xy(item.value, "points")[0] for item in observations
              if item.valid and item.value is not None]
    count = len(observations)
    complete = len(usable) == count
    if limits.invalid_policy == "ignore":
        complete = len(usable) >= 1
    value = (max(usable) - min(usable)) if complete and usable else None
    reason = "invalid_input" if not usable else "insufficient_valid_samples"
    return _status_for(value, space, limits, "max_center_x_minus_min_center_x",
                       count, len(usable), reason)


def _count(value: Any) -> int:
    """Normalise a scalar-or-array count to an integer total."""
    if value is None:
        return 0
    if isinstance(value, (list, tuple, np.ndarray)):
        return int(np.sum(value))
    return int(value)


def _summary_from_meta(meta: Any) -> QualitySummary:
    """Build a summary from a legacy result ``meta`` dictionary.

    Args:
        meta: Mapping produced by the measurement workflow.

    Returns:
        A summary whose unknown entries stay ``None``.
    """
    values = meta or {}
    samples = _count(values.get("num_points", values.get("num_edges")))
    requested = _count(values.get("num_measures"))
    coverage = (samples / requested) if requested > 0 else None
    rms = values.get("rms", values.get("fit_rms"))
    mean = values.get("amplitude_mean", values.get("edge_amplitude"))
    minimum = values.get("amplitude_min")
    return QualitySummary(
        sample_count=samples,
        rms_error=None if rms is None else float(rms),
        requested_probe_count=requested,
        successful_probe_count=samples,
        coverage_ratio=coverage,
        amplitude_mean=None if mean is None else float(mean),
        amplitude_abs_min=None if minimum is None else abs(float(minimum)),
    )


def summarize_quality(*, fit: Any = None, probes: Optional[Sequence[Any]] = None,
                      amplitudes: Optional[Any] = None,
                      meta: Any = None) -> QualitySummary:
    """Summarise what is actually known about one measurement.

    Args:
        fit: Optional fit diagnostics carrying residuals and sample counts.
        probes: Optional probe records with explicit per-probe status.
        amplitudes: Optional edge amplitudes used for signal statistics.

    Returns:
        A summary whose unknowns stay ``None`` instead of becoming zero.

    Raises:
        ComputationContractError: When only one of ``probes``/``amplitudes`` is given.
    """
    if meta is not None:
        return _summary_from_meta(meta)
    if (probes is None) != (amplitudes is None):
        raise contract_error("contract.invalid_parameter",
                             "probes and amplitudes must be supplied together",
                             "quality.amplitudes")
    if probes is not None:
        records = tuple(probes)
        requested = len(records)
        successful = sum(1 for record in records
                         if getattr(record, "status", "") == "selected")
        values = np.asarray(amplitudes, dtype=np.float64).reshape(-1)
        coverage = (successful / requested) if requested else None
        return QualitySummary(
            sample_count=int(values.size),
            rms_error=None,
            requested_probe_count=requested,
            successful_probe_count=successful,
            coverage_ratio=coverage,
            amplitude_mean=float(values.mean()) if values.size else None,
            amplitude_abs_min=float(np.abs(values).min()) if values.size else None,
        )
    if fit is not None:
        requested = getattr(fit, "requested_probe_count", None)
        successful = getattr(fit, "successful_probe_count", None)
        coverage = None
        if requested:
            coverage = (successful or 0) / requested
        return QualitySummary(
            sample_count=getattr(fit, "sample_count", None),
            rms_error=getattr(fit, "rms_error", None),
            requested_probe_count=requested,
            successful_probe_count=successful,
            coverage_ratio=coverage,
        )
    return QualitySummary()