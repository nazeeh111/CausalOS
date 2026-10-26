from copy import deepcopy

from fastapi.testclient import TestClient

from causalos.api import app

client = TestClient(app)


def test_example_profile_and_assumption_gate():
    example = client.get('/api/examples/confounded').json()
    assert len(example['data']) == example['profile']['rows'] == 800
    assert len(example['profile']['cohorts']) == 2
    question = deepcopy(example['question'])
    question['assumptions']['exchangeability'] = False
    response = client.post('/api/analyze', json={'data': example['data'], 'question': question})
    assert response.status_code == 200
    assert response.json()['status'] == 'blocked'
    assert response.json()['estimates'] == []


def test_declared_bidirected_confounding_cannot_be_estimated():
    example = client.get('/api/examples/confounded').json()
    question = deepcopy(example['question'])
    question['graph']['bidirected'] = [['T', 'Y']]
    response = client.post('/api/analyze', json={'data': example['data'], 'question': question})
    assert response.status_code == 200
    assert response.json()['status'] == 'blocked'
    assert response.json()['estimates'] == []
    assert 'confounding' in response.json()['identification']['reason']


def test_design_and_report_flow():
    initial = client.get('/api/design').json()
    action = initial['recommendations']['selected']
    response = client.post('/api/design/step', json={'state': initial, 'action_id': action, 'seed': 42})
    assert response.status_code == 200
    after = response.json()
    assert len(after['history']) == 1
    assert after['posterior'] != initial['posterior']
    assert after['budget_remaining'] < initial['budget_remaining']
    assert after['evidence_status'] == 'simulation'
    assert 'true_model' not in after
    report = client.post('/api/report', json={'design': after})
    assert report.status_code == 200
    assert 'text/html' in report.headers['content-type']
    assert 'simulated' in report.text


def test_report_retains_source_study_provenance():
    example = client.get('/api/examples/nsw').json()
    response = client.post('/api/report?format=json', json={
        'data': example['data'], 'question': example['question'], 'metadata': example['metadata'],
    })
    report = response.json()
    assert report['source_metadata']['provenance']['original_study']
    assert report['source_metadata']['kind'] == 'real_experimental'
    assert report['dataset']['rows'] == 445


def test_bad_input_has_useful_status():
    assert client.post('/api/profile', json={'data': []}).status_code == 422
    assert client.post('/api/profile', json={'data': [{'nested': {'x': 1}}]}).status_code == 422
    assert client.get('/api/examples/unknown').status_code == 422
    state = client.get('/api/design').json()
    assert client.post('/api/design/step', json={'state': state, 'action_id': 'unknown'}).status_code == 422


def test_application_assets_are_served():
    assert 'CAUSALOS LAB' in client.get('/').text
    assert client.get('/app.js').status_code == 200
    assert client.get('/styles.css').status_code == 200
    assert client.get('/app.js').headers['cache-control'] == 'no-cache'


def test_family_challenge_and_counterfactual_are_live():
    state = client.post('/api/evidence/reset', json={'seed': 42, 'scenario': 'weak_confounding'}).json()
    assert state['scenario'] == 'validity-study'
    assert len(state['posterior']) == 8
    action = state['recommendations']['selected']
    after = client.post('/api/evidence/step', json={
        'state': state, 'action_id': action, 'seed': 42, 'scenario': 'weak_confounding',
    })
    assert after.status_code == 200
    assert len(after.json()['history']) == 1
    original = client.get('/api/design').json()
    cf = client.post('/api/counterfactual', json={
        'state': original, 'observed': {'A': .4, 'Y': -.2, 'B': 0, 'Z': 0}, 'target': 'A', 'value': 1,
    })
    assert cf.status_code == 200
    assert len(cf.json()['models']) == 4
