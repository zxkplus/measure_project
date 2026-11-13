"""Public HTTP acceptance for GUI manifest direction-template migration."""

import hashlib
import io
import json
import zipfile

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from measure_api.v2.app import create_app
from measure_api.v2.models import ServiceSettings


def ref(path):
    return {'root': 'exchange', 'path': path}


def model_package(*, mutate=None, extra=None, prefix=''):
    noise = np.random.default_rng(42).integers(0, 255, (100, 120), np.uint8)
    image = cv2.GaussianBlur(noise, (5, 5), 1.1)
    buffer = io.BytesIO()
    np.save(buffer, image)
    raw = buffer.getvalue()
    digest = hashlib.sha256(raw).hexdigest()
    manifest = {
        'contract_version': '3.1', 'objects': [], 'physical_scale_mm': None,
        'files': {'reference.npy': digest},
        'reference_image': {'file': 'reference.npy', 'sha256': digest, 'shape': [100, 120]},
        'alignment': {'mode': 'single_box', 'box_center': [50, 60],
                      'box_size': [24.2, 32.1], 'box_angle_deg': -35,
                      'preprocessor': {'type': 'clahe', 'clip_limit': 2,
                                       'tile_grid_size': [8, 8]}},
        'matching': {'angle_range': [-180, 180], 'angle_step': 1,
                     'coarse_angle_step': 5, 'coarse_fine': True,
                     'match_score_threshold': 0.5, 'max_matches': 1,
                     'pyramid_decimate': 1, 'pyramid_max_template_size': 50},
    }
    if mutate:
        mutate(manifest)
    package = io.BytesIO()
    with zipfile.ZipFile(package, 'w', zipfile.ZIP_DEFLATED) as archive:
        if prefix:
            archive.writestr(prefix, '')
        archive.writestr(prefix + 'manifest.json', json.dumps(manifest))
        archive.writestr(prefix + 'reference.npy', raw)
        # A derived cache is intentionally irrelevant to import.
        archive.writestr(prefix + 'workflow.npz', b'ignored derived cache')
        for name, content in (extra or {}).items():
            archive.writestr(name, content)
    return image, package.getvalue()


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(ServiceSettings(shared_roots={'exchange': tmp_path},
                                               max_workers=1))) as value:
        yield value


def call(client, route, body):
    response = client.post('/api/v2' + route, json={**body, 'wait_ms': 60000})
    assert response.status_code == 200, response.text
    return response.json()['result']


def test_import_rotated_gui_template_and_match_known_pose(client, tmp_path):
    image, package = model_package(prefix='model/')
    (tmp_path / 'model.zip').write_bytes(package)
    result = call(client, '/template-imports', {
        'package': ref('model.zip'), 'output': {'directory': ref('imported')},
    })
    assert result['metadata']['support']['angle_degrees'] == 35
    assert result['metadata']['support']['width'] == 32
    assert result['metadata']['support']['height'] == 24
    assert result['search']['coarse_fine']
    assert result['search']['pyramid_levels'] == 1
    assert result['warnings']
    assert np.array_equal(cv2.imread(str(tmp_path / result['reference']['path']), 0), image)
    inspected = call(client, '/template-inspections', {'template': result['template']})
    assert inspected['metadata']['source'] == result['reference']
    observations = []
    for angle in (0, 30, -30, 180):
        transform = cv2.getRotationMatrix2D((60, 50), angle, 1)
        inspection = cv2.warpAffine(image, transform, (120, 100))
        cv2.imwrite(str(tmp_path / 'inspection.png'), inspection)
        matched = call(client, '/template-matches', {
            'image': ref('inspection.png'), 'template': result['template'],
            'search': result['search'],
        })
        assert matched['valid']
        observations.append(matched['matches'][0]['angle_degrees'])
    assert observations == pytest.approx([35, 5, 65, -145], abs=1)


@pytest.mark.parametrize('mutate,extra', [
    (lambda m: m.update(contract_version='3.0'), None),
    (lambda m: m.update(objects=[{'type': 'circle'}]), None),
    (lambda m: m['alignment'].update(mode='obb'), None),
    (lambda m: m['matching'].update(unknown=True), None),
    (lambda m: m['reference_image'].update(shape=[1, 1]), None),
    (lambda m: m['files'].update({'reference.npy': '0' * 64}), None),
    (None, {'../escape': b'x'}),
])
def test_reject_invalid_models_without_committed_output(client, tmp_path, mutate, extra):
    _, package = model_package(mutate=mutate, extra=extra)
    (tmp_path / 'bad.zip').write_bytes(package)
    response = client.post('/api/v2/template-imports', json={
        'package': ref('bad.zip'), 'output': {'directory': ref('rejected')}, 'wait_ms': 60000,
    })
    assert response.status_code == 422, response.text
    assert response.json()['error']['code'] == 'template_import.invalid'
    assert not list((tmp_path / 'rejected').glob('*/manifest.json'))
