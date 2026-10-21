"""Public HTTP acceptance for stateless atomic measurement capabilities."""

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from measure_api.v2.app import create_app
from measure_api.v2.models import ServiceSettings


@pytest.fixture
def client(tmp_path):
    image = np.zeros((120, 240), np.uint8)
    image[:, 40:60] = 255
    image[:, 130:160] = 255
    cv2.imwrite(str(tmp_path / 'stripes.png'), image)
    with TestClient(create_app(ServiceSettings(shared_roots={'exchange': tmp_path}))) as value:
        yield value


def result(client, path, payload):
    response = client.post('/api/v2' + path, json=payload)
    assert response.status_code == 200, response.text
    assert response.json()['state'] == 'succeeded'
    return response.json()['result']


IMAGE = {'root': 'exchange', 'path': 'stripes.png'}
PROBE = {'center': {'x': 110, 'y': 60}, 'search_half_length': 110,
         'average_half_width': 5}


def test_profiles_explain_selected_subpixel_edges(client):
    profile = result(client, '/edge-profiles', {'image': IMAGE, 'probe': PROBE})
    assert len(profile['gray']) == 220
    assert profile['gray'][45] == 255
    assert profile['gray'][70] == 0
    assert len(profile['smoothed']) == len(profile['gradient']) == 220
    edges = result(client, '/measurements/edge-points', {
        'image': IMAGE, 'probe': PROBE, 'polarity': 'positive', 'selection': 'all',
    })
    assert edges['valid']
    assert [edge['point']['x'] for edge in edges['edges']] == pytest.approx([39.5, 129.5])
    assert all(edge['amplitude'] > 0 for edge in edges['edges'])


@pytest.mark.parametrize('selection,width,center', [('first', 20, 49.5), ('last', 30, 144.5)])
def test_pairs_select_after_pairing_all_boundaries(client, selection, width, center):
    measured = result(client, '/measurements/edge-pairs', {
        'image': IMAGE, 'probe': PROBE, 'polarity': 'positive', 'selection': selection,
    })
    assert measured['valid']
    assert len(measured['pairs']) == 1
    assert measured['width']['value'] == pytest.approx(width)
    assert measured['center']['point']['x'] == pytest.approx(center)
    assert measured['pairs'][0]['edge_a']['amplitude'] > 0
    assert measured['pairs'][0]['edge_b']['amplitude'] < 0


def test_supplied_points_fit_and_compose_with_xy_convention(client):
    fitted = result(client, '/geometry/lines:fit', {
        'points': [{'x': 40, 'y': 99.5}, {'x': 200, 'y': 99.5}, {'x': 360, 'y': 99.5}],
    })
    assert fitted['line']['valid']
    distance = result(client, '/geometry/point-line-distances', {
        'point': {'valid': True, 'point': {'x': 200, 'y': 140}}, 'line': fitted['line'],
    })
    assert distance['value'] == pytest.approx(40.5)
    assert fitted['diagnostics']['rms_error'] == pytest.approx(0, abs=1e-10)
    circle = result(client, '/geometry/circles:fit', {
        'points': [{'x': 50, 'y': 40}, {'x': 40, 'y': 50},
                   {'x': 30, 'y': 40}, {'x': 40, 'y': 30}],
    })
    assert circle['circle']['center'] == pytest.approx({'x': 40, 'y': 40})
    assert circle['circle']['radius'] == pytest.approx(10)
    invalid = result(client, '/geometry/circles:fit', {
        'points': [{'x': 0, 'y': 0}, {'x': 1, 'y': 1}, {'x': 2, 'y': 2}],
    })
    assert not invalid['valid']
    assert invalid['circle']['radius'] is None


def test_image_fits_preserve_source_coordinates_and_fresh_invalid_results(client, tmp_path):
    image = np.zeros((300, 400), np.uint8)
    image[100:] = 255
    cv2.imwrite(str(tmp_path / 'line.png'), image)
    request = {'image': {'root': 'exchange', 'path': 'line.png'},
               'start': {'x': 40, 'y': 100}, 'end': {'x': 360, 'y': 100},
               'search_half_length': 15, 'average_half_width': 5,
               'probe_count': 10, 'polarity': 'positive'}
    fitted = result(client, '/measurements/lines', request)
    assert fitted['valid']
    assert fitted['diagnostics']['successful_probe_count'] == 10
    distance = result(client, '/geometry/point-line-distances', {
        'point': {'valid': True, 'point': {'x': 200, 'y': 140}}, 'line': fitted['line'],
    })
    assert distance['value'] == pytest.approx(40.5, abs=0.2)
    cv2.imwrite(str(tmp_path / 'line.png'), np.zeros_like(image))
    missing = result(client, '/measurements/lines', request)
    assert not missing['valid']
    assert missing['line']['a'] is None
    assert missing['diagnostics']['successful_probe_count'] == 0


def test_circle_and_arc_probes_report_real_residuals(client, tmp_path):
    image = np.zeros((200, 200), np.uint8)
    cv2.circle(image, (100, 100), 50, 255, -1)
    cv2.imwrite(str(tmp_path / 'circle.png'), image)
    measured = result(client, '/measurements/circles', {
        'image': {'root': 'exchange', 'path': 'circle.png'},
        'center': {'x': 100, 'y': 100}, 'radius': 50,
        'radius_min': 45, 'radius_max': 55, 'probe_count': 24,
        'search_half_length': 10, 'average_half_width': 2,
        'start_angle_degrees': 0, 'end_angle_degrees': 180,
    })
    assert measured['valid']
    assert measured['circle']['radius'] == pytest.approx(50, abs=0.7)
    assert measured['circle']['center']['x'] == pytest.approx(100, abs=0.5)
    assert measured['diagnostics']['successful_probe_count'] == 24
    assert measured['diagnostics']['rms_error'] < 0.5
    assert len(measured['edges']) == 24


def point(x, y, frame='source', unit='px'):
    return {'valid': True, 'point': {'x': x, 'y': y}, 'frame': frame, 'unit': unit}


def test_geometry_composition_and_frame_checks(client):
    line = result(client, '/geometry/two-point-lines', {
        'point_a': point(0, 10), 'point_b': point(20, 10),
    })['line']
    assert line['valid']
    assert abs(line['b']) == pytest.approx(1)
    assert result(client, '/geometry/point-distances', {
        'point_a': point(0, 0), 'point_b': point(3, 4),
    })['value'] == 5
    vertical = result(client, '/geometry/two-point-lines', {
        'point_a': point(3, 0), 'point_b': point(3, 20),
    })['line']
    angle = result(client, '/geometry/line-angles', {'line_a': line, 'line_b': vertical})
    assert angle['value'] == 90
    circle = {'valid': True, 'center': {'x': 40, 'y': 40}, 'radius': 10}
    assert result(client, '/geometry/point-circle-distances', {
        'point': point(40, 40), 'circle': circle,
    })['value'] == 10
    dimensions = result(client, '/geometry/circle-dimensions', {
        'circle': circle, 'samples': [point(50, 40), point(52, 40)],
    })
    assert dimensions['radius']['value'] == 10
    assert dimensions['diameter']['value'] == 20
    assert dimensions['sampled_diameter_spread']['value'] == 4
    invalid = result(client, '/geometry/two-point-lines', {
        'point_a': point(2, 3), 'point_b': point(2, 3),
    })
    assert not invalid['valid'] and invalid['line']['a'] is None
    mismatch = client.post('/api/v2/geometry/point-line-distances', json={
        'point': point(3, 16, frame='unrelated'), 'line': line,
    })
    assert mismatch.status_code == 422
    assert mismatch.json()['error']['code'] == 'geometry.frame_mismatch'


def scalar(value, valid=True, unit='px'):
    return {'valid': valid, 'value': value, 'unit': unit}


def test_business_limits_are_distinct_from_observation_validity(client):
    for value in [10, 20]:
        measured = result(client, '/evaluations/ranges', {
            'value': scalar(value), 'minimum': 10, 'maximum': 20, 'unit': 'px',
        })
        assert measured['valid'] and measured['evaluation_status'] == 'pass'
    assert result(client, '/evaluations/ranges', {
        'value': scalar(21), 'minimum': 10, 'maximum': 20,
    })['evaluation_status'] == 'fail'
    invalid = result(client, '/evaluations/ranges', {
        'value': scalar(None, False), 'maximum': 20,
    })
    assert not invalid['valid'] and invalid['evaluation_status'] == 'not_evaluated'
    assert result(client, '/evaluations/ranges', {
        'value': scalar(None, False), 'maximum': 20, 'invalid_policy': 'fail',
    })['evaluation_status'] == 'fail'
    variation = result(client, '/evaluations/diameter-variation', {
        'values': [scalar(10), scalar(14), scalar(12)], 'maximum': 4,
    })
    assert variation['value'] == 4 and variation['evaluation_status'] == 'pass'
    deviation = result(client, '/evaluations/centerline-deviation', {
        'points': [point(50, 10), point(48, 20), point(52, 30)], 'maximum': 3,
    })
    assert deviation['value'] == 4 and deviation['evaluation_status'] == 'fail'
    mismatch = client.post('/api/v2/evaluations/ranges', json={
        'value': scalar(10, unit='mm'), 'maximum': 20, 'unit': 'px',
    })
    assert mismatch.status_code == 422


def test_batch_shares_image_and_preserves_per_item_failures(client):
    batch = result(client, '/measurement-batches', {
        'image': IMAGE, 'measurements': [
            {'id': 'edges', 'operation': 'measurements.edge-points',
             'parameters': {'probe': PROBE, 'polarity': 'positive'}},
            {'id': 'width', 'operation': 'measurements.edge-pairs',
             'parameters': {'probe': PROBE, 'polarity': 'positive', 'selection': 'last'}},
            {'id': 'outside', 'operation': 'measurements.edge-points',
             'parameters': {'probe': dict(PROBE, center={'x': -100, 'y': 60})}},
        ],
    })
    assert [item['id'] for item in batch['measurements']] == ['edges', 'width', 'outside']
    assert batch['measurements'][0]['result']['valid']
    assert batch['measurements'][1]['result']['width']['value'] == 30
    assert batch['measurements'][2]['state'] == 'failed'
    assert batch['measurements'][2]['error']['code'] == 'image.roi_outside'
    assert not batch['valid']


def test_circle_radius_limits_apply_before_first_edge_selection(client, tmp_path):
    image = np.zeros((220, 220), np.uint8)
    for radius, level in [(80, 255), (60, 0), (40, 255)]:
        cv2.circle(image, (110, 110), radius, level, -1)
    cv2.imwrite(str(tmp_path / 'rings.png'), image)
    circle = result(client, '/measurements/circles', {
        'image': {'root': 'exchange', 'path': 'rings.png'},
        'center': {'x': 110, 'y': 110}, 'radius': 60, 'radius_min': 55, 'radius_max': 65,
        'search_half_length': 30, 'average_half_width': 2,
        'selection': 'first', 'probe_count': 20,
    })
    assert circle['valid']
    assert circle['circle']['radius'] == pytest.approx(60, abs=0.7)


def test_single_edge_and_pair_ports_compose_as_observed_points(client):
    edge = result(client, '/measurements/edge-points', {
        'image': IMAGE, 'probe': PROBE, 'selection': 'first', 'polarity': 'positive',
    })
    assert result(client, '/geometry/point-distances', {
        'point_a': edge['point'], 'point_b': point(39.5, 70),
    })['value'] == 10
    pair = result(client, '/measurements/edge-pairs', {
        'image': IMAGE, 'probe': PROBE, 'selection': 'last', 'polarity': 'positive',
    })
    assert result(client, '/geometry/point-distances', {
        'point_a': pair['edge_a'], 'point_b': pair['edge_b'],
    })['value'] == 30
    multiple = result(client, '/measurements/edge-pairs', {
        'image': IMAGE, 'probe': PROBE, 'polarity': 'positive', 'selection': 'all',
    })
    assert len(multiple['pairs']) == 2
    assert multiple['width'] is None


def test_diameter_variation_compares_lengths_from_separate_image_frames(client):
    first = dict(scalar(12), frame='camera_a')
    second = dict(scalar(15), frame='camera_b')
    variation = result(client, '/evaluations/diameter-variation', {
        'values': [first, second], 'maximum': 3,
    })
    assert variation['value'] == 3
    assert variation['evaluation_status'] == 'pass'


def test_profile_curves_can_be_committed_as_shared_numeric_files(client, tmp_path):
    measured = result(client, '/edge-profiles', {
        'image': IMAGE, 'probe': PROBE, 'inline_samples': False,
        'output': {'directory': {'root': 'exchange', 'path': 'curves'}},
    })
    assert measured['gray'] == []
    descriptor = measured['artifacts'][0]
    with np.load(tmp_path / descriptor['file']['path'], allow_pickle=False) as arrays:
        assert arrays['gray'][45] == 255
        assert arrays['sample_points'][0].tolist() == [0, 60]
        assert arrays['gradient'].shape == (220,)
    assert (tmp_path / measured['manifest']['file']['path']).is_file()


@pytest.mark.parametrize('measurements', [None, [{'id': ['bad'], 'operation': None}],
                                         [{'id': 'bad', 'operation': 7}]])
def test_malformed_dynamic_batch_requests_return_structured_validation(client, measurements):
    response = client.post('/api/v2/measurement-batches', json={
        'image': IMAGE, 'measurements': measurements,
    })
    assert response.status_code == 422
    assert response.json()['error']['code'] == 'request.validation'


def test_batch_profiles_share_one_complete_parent_manifest(client, tmp_path):
    measured = result(client, '/measurement-batches', {
        'image': IMAGE, 'output': {'directory': {'root': 'exchange', 'path': 'batch-curves'}},
        'measurements': [
            {'id': 'a', 'operation': 'edge-profiles',
             'parameters': {'probe': PROBE, 'inline_samples': False}},
            {'id': 'b', 'operation': 'edge-profiles',
             'parameters': {'probe': PROBE, 'inline_samples': False, 'sigma': 2}},
        ],
    })
    assert measured['valid']
    assert len(measured['artifacts']) == 2
    assert len(list((tmp_path / 'batch-curves').rglob('manifest.json'))) == 1
    for item in measured['measurements']:
        assert item['result']['gray'] == []
        assert item['result']['manifest'] == measured['manifest']
        with np.load(tmp_path / item['result']['artifacts'][0]['file']['path']) as arrays:
            assert arrays['gray'][45] == 255


def test_coincident_expected_line_reports_zero_observed_samples(client):
    measured = result(client, '/measurements/lines', {
        'image': IMAGE, 'start': {'x': 40, 'y': 60}, 'end': {'x': 40, 'y': 60},
        'probe_count': 10,
    })
    assert not measured['valid']
    assert measured['diagnostics']['sample_count'] == 0
    assert measured['diagnostics']['requested_probe_count'] == 10
    assert measured['diagnostics']['successful_probe_count'] == 0


def test_invalid_profile_threshold_never_publishes_a_completion_manifest(client, tmp_path):
    response = client.post('/api/v2/edge-profiles', json={
        'image': IMAGE, 'probe': PROBE, 'threshold': 2, 'threshold_unit': 'fraction',
        'output': {'directory': {'root': 'exchange', 'path': 'invalid-curves'}},
    })
    assert response.status_code == 422
    assert not list((tmp_path / 'invalid-curves').rglob('manifest.json'))


def test_batch_template_points_use_real_asset_and_validate_nested_file_identity(client, tmp_path):
    patch = np.random.default_rng(8).integers(20, 230, (20, 30), dtype=np.uint8)
    reference = np.zeros((100, 140), np.uint8)
    reference[10:30, 10:40] = patch
    cv2.imwrite(str(tmp_path / 'teach.png'), reference)
    template = result(client, '/template-builds', {
        'image': {'root': 'exchange', 'path': 'teach.png'},
        'roi': {'center': {'x': 24.5, 'y': 19.5}, 'width': 30, 'height': 20},
        'output': {'directory': {'root': 'exchange', 'path': 'templates'}},
    })['template']
    inspection = np.zeros((100, 140), np.uint8)
    inspection[60:80, 70:100] = patch
    cv2.imwrite(str(tmp_path / 'inspect.png'), inspection)
    request = {'image': {'root': 'exchange', 'path': 'inspect.png'}, 'measurements': [
        {'id': 'feature', 'operation': 'measurements.template-points',
         'parameters': {'template': template, 'search': {'score_threshold': 0.95}}},
    ]}
    batch = result(client, '/measurement-batches', request)
    assert batch['valid']
    located = batch['measurements'][0]['result']['point']
    assert located['point'] == pytest.approx({'x': 84.5, 'y': 69.5}, abs=0.1)
    assert result(client, '/geometry/point-distances', {
        'point_a': located, 'point_b': point(84.5, 79.5, frame=located['frame']),
    })['value'] == pytest.approx(10, abs=0.1)
    bad_template = dict(template, sha256='0' * 64)
    request['measurements'][0]['parameters']['template'] = bad_template
    response = client.post('/api/v2/measurement-batches', json=request)
    assert response.status_code == 409
    assert response.json()['error']['code'] == 'file.digest_mismatch'


def test_batch_contains_numeric_overflow_failure_without_losing_other_observations(client):
    measured = result(client, '/measurement-batches', {
        'image': IMAGE, 'measurements': [
            {'id': 'edge', 'operation': 'measurements.edge-points',
             'parameters': {'probe': PROBE, 'selection': 'first'}},
            {'id': 'overflow', 'operation': 'measurements.lines', 'parameters': {
                'start': {'x': 1e308, 'y': 60}, 'end': {'x': -1e308, 'y': 60},
            }},
        ],
    })
    assert measured['measurements'][0]['result']['valid']
    assert measured['measurements'][1]['state'] == 'failed'
    assert measured['measurements'][1]['error']['code'] == 'calculation.failed'
    assert not measured['valid']
