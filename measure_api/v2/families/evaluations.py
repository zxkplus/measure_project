"""Supplied-value product acceptance, independent of computational completion."""

from typing import List, Literal, Optional

from pydantic import Field, model_validator

from measure.computation_types import (
    ComputationContractError,
    Observation,
    Space,
    invalid,
    observed,
)
from measure.evaluation import (
    LimitsSpec,
    centerline_deviation as shared_centerline_deviation,
    diameter_variation as shared_diameter_variation,
    evaluate_range as shared_evaluate_range,
)
from measure_api.v2.atomic_types import PositionedPoint, Scalar
from measure_api.v2.errors import ServiceError
from measure_api.v2.families.geometry import compatible
from measure_api.v2.models import CalculationRequest
from measure_api.v2.registry import CalculationContext, Operation


class LimitsRequest(CalculationRequest):
    """Inclusive limits and explicit policy for missing or invalid observations."""

    minimum: Optional[float] = None
    maximum: Optional[float] = None
    unit: str = Field(default='px', min_length=1)
    invalid_policy: Literal['not_evaluated', 'fail', 'error', 'ignore'] = 'not_evaluated'

    @model_validator(mode='after')
    def check_limits(self) -> 'LimitsRequest':
        """Require a meaningful interval and reject inverted limits.

        Returns:
            This limits request after rejecting an empty or inverted interval.
        """
        if self.minimum is None and self.maximum is None:
            raise ValueError('At least one inclusive limit is required')
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError('Minimum exceeds maximum')
        return self


class RangeRequest(LimitsRequest):
    value: Scalar


class VariationRequest(LimitsRequest):
    values: List[Scalar] = Field(max_length=100000)


class DeviationRequest(LimitsRequest):
    points: List[PositionedPoint] = Field(max_length=100000)


class EvaluationResult(Scalar):
    evaluation_status: Literal['pass', 'fail', 'not_evaluated']
    calculation: str
    input_count: int
    valid_count: int
    minimum: Optional[float]
    maximum: Optional[float]
    invalid_policy: str


def _limits(request: LimitsRequest) -> LimitsSpec:
    """Translate the HTTP limit model into the shared evaluation spec."""
    return LimitsSpec(minimum=request.minimum, maximum=request.maximum,
                      unit=request.unit, invalid_policy=request.invalid_policy)


def _observation(scalar) -> Observation:
    """Wrap one HTTP scalar as a shared observation."""
    space = Space(scalar.frame, scalar.unit)
    if scalar.valid and scalar.value is not None:
        return observed(space, float(scalar.value))
    return invalid(space, 'invalid_input')


def _from_shared(result, calculation: Optional[str] = None) -> EvaluationResult:
    """Translate a shared evaluation result into the HTTP response model."""
    measurement = result.measurement
    return EvaluationResult(
        valid=measurement.valid,
        reason=measurement.reason,
        value=measurement.value,
        frame=measurement.space.frame,
        unit=measurement.space.unit,
        evaluation_status=result.evaluation_status,
        calculation=calculation or result.calculation,
        input_count=result.input_count,
        valid_count=result.valid_count,
        minimum=result.minimum,
        maximum=result.maximum,
        invalid_policy=result.invalid_policy,
    )


def _translate(call):
    """Run a shared evaluation, mapping contract errors onto service errors."""
    try:
        return call()
    except ComputationContractError as error:
        raise ServiceError(error.code, error.message) from error


def _evaluate(
    request: LimitsRequest, value: Optional[float], frame: str,
    input_count: int, valid_count: int, calculation: str, reason: Optional[str] = None,
) -> EvaluationResult:
    valid = value is not None
    if not valid and request.invalid_policy == 'error':
        raise ServiceError('evaluation.invalid_input', reason or 'Input observations are invalid')
    status = 'not_evaluated'
    if valid:
        passed = ((request.minimum is None or value >= request.minimum)
                  and (request.maximum is None or value <= request.maximum))
        status = 'pass' if passed else 'fail'
    elif request.invalid_policy == 'fail':
        status = 'fail'
    return EvaluationResult(
        valid=valid, reason=None if valid else reason or 'invalid_input',
        value=value, frame=frame, unit=request.unit, evaluation_status=status,
        calculation=calculation, input_count=input_count, valid_count=valid_count,
        minimum=request.minimum, maximum=request.maximum, invalid_policy=request.invalid_policy,
    )


def evaluate_range(request: RangeRequest, context: CalculationContext) -> EvaluationResult:
    """Evaluate an observed scalar against inclusive limits in the same unit.

    Args:
        request: Typed observed scalar and inclusive minimum/maximum limits in matching
            units.
        context: Isolated operation identity, resource limits and verified shared-file
            access.

    Returns:
        Observed value and pass/fail/not-evaluated outcome with explicit limits and units.
    """
    if request.value.unit != request.unit:
        raise ServiceError('evaluation.unit_mismatch', 'Limit and observation units must match')
    return _from_shared(_translate(
        lambda: shared_evaluate_range(_observation(request.value), _limits(request))))


def evaluate_variation(
    request: VariationRequest, context: CalculationContext,
) -> EvaluationResult:
    """Compute maximum supplied diameter minus minimum supplied diameter.

    Args:
        request: Typed observed diameters and allowed maximum-minus-minimum variation.
        context: Isolated operation identity, resource limits and verified shared-file
            access.

    Returns:
        Observed value and pass/fail/not-evaluated outcome with explicit limits and units.
    """
    if any(v.unit != request.unit for v in request.values):
        raise ServiceError('evaluation.unit_mismatch', 'Diameter and limit units must match')
    return _from_shared(_translate(lambda: shared_diameter_variation(
        [_observation(value) for value in request.values], _limits(request))))


def evaluate_deviation(
    request: DeviationRequest, context: CalculationContext,
) -> EvaluationResult:
    """Compute maximum center x minus minimum center x without fitting a new axis.

    Args:
        request: Typed observed centers and allowed horizontal center deviation.
        context: Isolated operation identity, resource limits and verified shared-file
            access.

    Returns:
        Observed value and pass/fail/not-evaluated outcome with explicit limits and units.
    """
    if any(p.unit != request.unit for p in request.points):
        raise ServiceError('evaluation.unit_mismatch', 'Center and limit units must match')
    compatible(*request.points)
    usable = [p for p in request.points if p.valid and p.point is not None]
    frames = {p.frame for p in request.points}
    frame = next(iter(frames)) if len(frames) == 1 else (
        request.points[0].frame if request.points else 'source')
    # Two centres are required for a meaningful horizontal range; the shared
    # aggregate deliberately accepts one under the ignore policy, so the
    # stricter supplier contract is enforced here.
    if len(usable) < 2:
        return _evaluate(request, None, frame, len(request.points), len(usable),
                         'max_center_x_minus_min_center_x', 'insufficient_valid_samples')
    space = Space(frame, request.unit)
    observations = [observed(space, (p.point.x, p.point.y)) if p.point is not None
                    else invalid(space, 'invalid_input') for p in request.points]
    return _from_shared(_translate(lambda: shared_centerline_deviation(
        observations, _limits(request), space=space)))


OPERATIONS = [
    Operation('/evaluations/ranges', RangeRequest, EvaluationResult, evaluate_range,
              'Evaluate inclusive scalar limits', 'evaluations',
              {'value': 'number', 'evaluation_status': 'evaluation_status'}),
    Operation('/evaluations/diameter-variation', VariationRequest, EvaluationResult,
              evaluate_variation, 'Calculate supplied diameter variation', 'evaluations',
              {'value': 'number', 'evaluation_status': 'evaluation_status'}),
    Operation('/evaluations/centerline-deviation', DeviationRequest, EvaluationResult,
              evaluate_deviation, 'Calculate centerline x deviation', 'evaluations',
              {'value': 'number', 'evaluation_status': 'evaluation_status'}),
]
