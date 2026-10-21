"""Registry-based independent measurements sharing one verified source image."""

import io
from typing import Any, Dict, List, Literal, Optional

import numpy as np

from pydantic import Field, SerializeAsAny, model_validator

from measure_api.v2.atomic_types import Observation
from measure_api.v2.errors import ServiceError
from measure_api.v2.families.measurements import ProfileRequest
from measure_api.v2.files import ArtifactWriter
from measure_api.v2.models import (
    CalculationRequest, ErrorDetail, FileDescriptor, FileReference, OutputSpec, StrictModel,
)
from measure_api.v2.registry import CalculationContext, Operation, operations


def _operation(identifier: str) -> Operation:
    if not isinstance(identifier, str):
        raise ValueError('Operation identifier must be a string')
    identifier = identifier.strip('/').replace('/', '.').replace(':', '_')
    operation = operations().get(identifier)
    if operation is None or not operation.path.startswith(('/measurements/', '/edge-profiles')):
        raise ValueError('Batch items require a registered atomic image measurement operation')
    return operation


class BatchMeasurement(StrictModel):
    """Dynamic registered request; typed fields retain shared-file verification."""

    id: str = Field(min_length=1, max_length=200)
    operation: str
    parameters: SerializeAsAny[CalculationRequest]


class MeasurementBatchRequest(CalculationRequest):
    """Individual parameters omit image; this shared image is injected and verified once."""

    image: FileReference
    output: Optional[OutputSpec] = None
    measurements: List[BatchMeasurement] = Field(min_length=1, max_length=1000)

    @model_validator(mode='before')
    @classmethod
    def resolve_requests(cls, value: Any) -> Any:
        """Resolve dynamic requests before recursive shared-file preparation.

        Args:
            value: Supplied value to normalize under the public schema and coordinate
                convention.

        Returns:
            Batch input with each dynamic item parsed by its registered request schema.
        """
        if not isinstance(value, dict):
            return value
        measurements = value.get('measurements')
        if not isinstance(measurements, list):
            return value
        output = value.get('output')
        identifiers = set()
        resolved = []
        for item in measurements:
            if not isinstance(item, dict):
                resolved.append(item)
                continue
            identifier = item.get('id')
            if not isinstance(identifier, str):
                raise ValueError('Measurement identifiers must be strings')
            if identifier in identifiers:
                raise ValueError('Measurement identifiers must be unique')
            identifiers.add(identifier)
            operation = _operation(item.get('operation', ''))
            parameters = item.get('parameters', {})
            if isinstance(parameters, CalculationRequest):
                parameters = parameters.model_dump()
            if not isinstance(parameters, dict):
                raise ValueError('Measurement parameters must be an object')
            if 'image' in parameters and parameters['image'] != value.get('image'):
                raise ValueError('All independent measurements must share the batch image')
            if issubclass(operation.request_model, ProfileRequest):
                supplied_output = parameters.get('output')
                if supplied_output is not None:
                    if output is not None and supplied_output != output:
                        raise ValueError('Batch artifacts require one common output directory')
                    output = supplied_output
                if output is not None:
                    parameters = {**parameters, 'output': output}
            request = operation.request_model.model_validate({
                **parameters, 'image': value.get('image'),
            })
            resolved.append({**item, 'operation': operation.operation_id, 'parameters': request})
        return {**value, 'measurements': resolved, 'output': output}


class BatchMeasurementResult(StrictModel):
    id: str
    operation: str
    state: Literal['succeeded', 'failed']
    valid: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[ErrorDetail] = None


class MeasurementBatchResult(Observation):
    measurements: List[BatchMeasurementResult]
    artifacts: List[FileDescriptor] = Field(default_factory=list)
    manifest: Optional[FileDescriptor] = None


def execute_batch(
    request: MeasurementBatchRequest, context: CalculationContext,
) -> MeasurementBatchResult:
    """Keep successful measurements even when an independent item fails.

    Args:
        request: Typed individual parameters omit image; this shared image is injected and
            verified once.
        context: Isolated operation identity, resource limits and verified shared-file
            access.

    Returns:
        Independent successful/failed measurement items and optional joint artifacts.
    """
    results = []
    payloads, curve_items = [], []
    for index, item in enumerate(request.measurements):
        operation = _operation(item.operation)
        try:
            parameters = item.parameters
            if isinstance(parameters, ProfileRequest):
                parameters = parameters.model_copy(update={'output': None, 'inline_samples': True})
            result = operation.result_model.model_validate(operation.execute(parameters, context))
            data = result.model_dump(mode='json')
            if isinstance(item.parameters, ProfileRequest) and request.output is not None:
                buffer = io.BytesIO()
                np.savez_compressed(
                    buffer, gray=np.array(data['gray']), smoothed=np.array(data['smoothed']),
                    gradient=np.array(data['gradient']),
                    sample_points=np.array([[p['x'], p['y']] for p in data['sample_points']]),
                )
                payloads.append((f'profile-{index:04d}.npz', 'curves', 'npz', buffer.getvalue()))
                curve_items.append(len(results))
                if not item.parameters.inline_samples:
                    for field in ['gray', 'smoothed', 'gradient', 'sample_points']:
                        data[field] = []
            results.append(BatchMeasurementResult(
                id=item.id, operation=item.operation, state='succeeded',
                valid=data.get('valid', True), result=data,
            ))
        except ServiceError as exc:
            results.append(BatchMeasurementResult(
                id=item.id, operation=item.operation, state='failed', valid=False,
                error=ErrorDetail(**exc.payload(context.trace_id)),
            ))
        except Exception as exc:
            results.append(BatchMeasurementResult(
                id=item.id, operation=item.operation, state='failed', valid=False,
                error=ErrorDetail(
                    code='calculation.failed',
                    message='Independent measurement failed unexpectedly',
                    details={'exception_type': type(exc).__name__}, trace_id=context.trace_id,
                ),
            ))
    artifacts, manifest = [], None
    if payloads:
        artifacts, manifest = ArtifactWriter(
            context.files, request.output, context.operation_id,
        ).commit(payloads)
        for index, descriptor in zip(curve_items, artifacts):
            data = results[index].result
            data['artifacts'] = [descriptor.model_dump(mode='json')]
            data['manifest'] = manifest.model_dump(mode='json')
    valid = all(item.valid for item in results)
    return MeasurementBatchResult(valid=valid, reason=None if valid else 'partial_failure',
                                  measurements=results, artifacts=artifacts, manifest=manifest)


OPERATIONS = [Operation(
    '/measurement-batches', MeasurementBatchRequest, MeasurementBatchResult, execute_batch,
    'Execute independent atomic measurements sharing one image', 'measurements',
    {'measurements': 'measurement_result_array', 'artifacts': 'file_array'},
)]
