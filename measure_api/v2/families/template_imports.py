"""One-way import of GUI manifest templates into native stateless V2 assets."""

from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import PurePosixPath
import stat
import zipfile
from typing import List

import numpy as np
from pydantic import Field

from measure_api.v2.errors import ServiceError
from measure_api.v2.files import ArtifactWriter
from measure_api.v2.families.images import RotatedROI, encode_png
from measure_api.v2.families.templates import (
    TemplateBuildRequest, TemplateMetadata, TemplateSearch, build_template, load_template,
)
from measure_api.v2.models import (
    CalculationRequest, FileDescriptor, FileReference, OutputSpec, Point, StrictModel,
)
from measure_api.v2.portable_recipes import output_reference
from measure_api.v2.legacy_conversion import legacy_preprocessing
from measure_api.v2.registry import CalculationContext, Operation


class TemplateImportRequest(CalculationRequest):
    """Import a bounded GUI model ZIP, without sessions or measurement nodes."""

    package: FileReference
    output: OutputSpec


class TemplateImportResult(StrictModel):
    """Native template, unchanged reference pixels and explicit search controls."""

    source_package: FileReference
    reference: FileReference
    template: FileReference
    metadata: TemplateMetadata
    search: TemplateSearch
    warnings: List[str] = Field(default_factory=list)
    artifacts: List[FileDescriptor]
    manifest: FileDescriptor


def _members(data: bytes, context: CalculationContext) -> dict:
    """Bound archive expansion and permit one common GUI export directory."""
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        entries = archive.infolist()
        if len(entries) > 64 or len({item.filename for item in entries}) != len(entries):
            raise ValueError("Duplicate or excessive model members")
        limit = min(context.settings.max_input_bytes,
                    context.settings.max_worker_memory_bytes // 4)
        if sum(item.file_size for item in entries) > limit:
            raise ValueError("Expanded model exceeds configured limit")
        names = []
        for item in entries:
            path = PurePosixPath(item.filename)
            if (path.is_absolute() or '..' in path.parts or '\\' in item.filename
                    or ':' in item.filename or '\x00' in item.filename
                    or stat.S_ISLNK(item.external_attr >> 16) or item.flag_bits & 1):
                raise ValueError("Unsafe model member")
            if not item.is_dir() and not item.filename.startswith('__MACOSX/'):
                names.append(item.filename)
        manifests = [name for name in names if PurePosixPath(name).name == 'manifest.json']
        if len(manifests) != 1:
            raise ValueError("Expected exactly one manifest.json")
        parent = PurePosixPath(manifests[0]).parent
        allowed = {'manifest.json', 'reference.npy', 'workflow.npz', 'reference.png',
                   'inspection.png', 'project.json', '.DS_Store'}
        members = {}
        for name in names:
            path = PurePosixPath(name)
            if path.parent != parent or path.name not in allowed:
                raise ValueError("Unsupported model member: " + name)
            if path.name in {'manifest.json', 'reference.npy'}:
                members[path.name] = archive.read(name)
        return members


def _reference(data: bytes, context: CalculationContext) -> np.ndarray:
    """Check the NPY header before allocating, and never permit object arrays."""
    stream = io.BytesIO(data)
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError("Unsupported reference NPY version")
    if (dtype != np.uint8 or len(shape) != 2 or min(shape) < 1
            or math.prod(shape) > context.settings.max_image_pixels
            or math.prod(shape) != len(data) - stream.tell()):
        raise ValueError("Reference must be a bounded GRAY8 numeric array")
    return np.load(io.BytesIO(data), allow_pickle=False)


def _configuration(manifest: dict, image: np.ndarray) -> tuple:
    """Map only supported manifest semantics and report a changed matching engine."""
    allowed = {'alignment', 'contract_version', 'created_at', 'derived_fingerprint',
               'files', 'matching', 'model_id', 'objects', 'physical_scale_mm',
               'reference_image'}
    if set(manifest) - allowed or manifest.get('contract_version') != '3.1':
        raise ValueError("Only GUI manifest contract 3.1 is supported")
    if manifest.get('objects') != [] or manifest.get('physical_scale_mm') is not None:
        raise ValueError("Import supports direction-only models without measurement objects")
    info = manifest['reference_image']
    if info['file'] != 'reference.npy' or info['shape'] != list(image.shape):
        raise ValueError("Reference identity or shape disagrees with manifest")
    alignment = manifest['alignment']
    if set(alignment) - {'mode', 'box_angle_deg', 'box_center', 'box_size',
                         'preprocessor', 'roi_state', 'template_size'}:
        raise ValueError("Unsupported alignment fields")
    if alignment['mode'] != 'single_box':
        raise ValueError("Only single_box direction templates are supported")
    row, col = alignment['box_center']
    height, width = alignment['box_size']
    roi = RotatedROI(center=Point(x=col, y=row), width=round(width), height=round(height),
                     angle_degrees=-alignment['box_angle_deg'])
    pp = legacy_preprocessing(alignment['preprocessor'])
    matching = manifest['matching']
    # The pyramid keys were renamed in the GUI manifest; assets taught before the
    # rename still carry the old spelling, so both are accepted here.
    if set(matching) - {'angle_range', 'angle_step', 'coarse_angle_step', 'coarse_fine',
                       'match_score_threshold', 'max_matches', 'overlap', 'decimate_exponent',
                       'coarse_template_side', 'pyramid_decimate',
                       'pyramid_max_template_size', 'scale_invariant', 'scale_range',
                       'scale_step'}:
        raise ValueError("Unsupported matching fields")
    if matching.get('scale_invariant', False) or matching.get('max_matches', 1) != 1:
        raise ValueError("Direction import requires fixed scale and one target")
    low, high = matching['angle_range']
    search = TemplateSearch(
        score_threshold=matching['match_score_threshold'], max_matches=1,
        overlap=matching.get('overlap', 0.3), angle_min=-high, angle_max=-low,
        angle_step=matching['angle_step'], coarse_fine=matching.get('coarse_fine', True),
        coarse_angle_step=matching.get('coarse_angle_step', 5),
        pyramid_levels=int(matching.get('decimate_exponent',
                                        matching.get('pyramid_decimate', 0))),
        pyramid_max_template_size=int(matching.get('coarse_template_side',
                                                   matching.get('pyramid_max_template_size', 128))),
        scale_min=1, scale_max=1,
    )
    return roi, pp, search


def import_template(request: TemplateImportRequest,
                    context: CalculationContext) -> TemplateImportResult:
    """Rebuild a GUI direction model with the existing native V2 matcher.

    Args:
        request: Explicit immutable model ZIP and parent output namespace.
        context: Bounded worker files and cancellable temporary stages.

    Returns:
        Jointly committed native template, reference and preview descriptors.
    """
    data, package = context.files.read_bytes(request.package)
    try:
        members = _members(data, context)
        manifest = json.loads(members['manifest.json'])
        raw = members['reference.npy']
        actual = hashlib.sha256(raw).hexdigest()
        if (manifest['reference_image']['sha256'] != actual
                or manifest['files'].get('reference.npy') != actual):
            raise ValueError("Reference content disagrees with manifest identity")
        image = _reference(raw, context)
        roi, pp, search = _configuration(manifest, image)
    except (ValueError, TypeError, KeyError, OSError, zipfile.BadZipFile) as error:
        raise ServiceError('template_import.invalid', str(error)) from error
    png = encode_png(image)
    stage, output = context.stage('reference')
    inputs, _ = ArtifactWriter(stage.files, output, stage.operation_id).commit([
        ('reference.png', 'reference', 'png', png),
    ])
    build_context, build_output = context.stage('template')
    # Both stages share the same private stage root; no GUI or v1 loader is invoked.
    built = build_template(TemplateBuildRequest(
        image=inputs[0].file, roi=roi, preprocessing=pp, output=build_output,
    ), build_context)
    pixels, metadata, _ = load_template(built.template, build_context)
    reference = output_reference(request.output, 'reference.png', png, context)
    metadata = metadata.model_copy(update={'source': reference})
    buffer = io.BytesIO()
    np.savez_compressed(buffer, pixels=pixels, metadata=np.array(metadata.model_dump_json()))
    preview, _ = build_context.files.read_bytes(built.artifacts[1].file)
    artifacts, final = ArtifactWriter(context.files, request.output, context.operation_id).commit([
        ('reference.png', 'reference', 'png', png),
        ('template.npz', 'template', 'npz', buffer.getvalue()),
        ('preview.png', 'preview', 'png', preview),
        ('source-manifest.json', 'source', 'json', members['manifest.json']),
    ])
    return TemplateImportResult(
        source_package=package, reference=artifacts[0].file, template=artifacts[1].file,
        metadata=metadata, search=search, artifacts=artifacts, manifest=final,
        warnings=['Rebuilt with the V2 matching engine; scores and pyramid sampling differ '
                  'from v1. Revalidate reference angle, quality threshold and cycle time.'],
    )


OPERATIONS = [Operation('/template-imports', TemplateImportRequest, TemplateImportResult,
                        import_template, 'Import a GUI manifest direction template',
                        'templates', {'template': 'template_asset'})]
