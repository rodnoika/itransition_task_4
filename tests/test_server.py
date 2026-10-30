import pytest
from fastapi.testclient import TestClient

from test_assistant import FakeModel


@pytest.fixture
def client(repository, monkeypatch):
    monkeypatch.setenv('ALLOWED_STORE_IDS', '1')
    import server
    monkeypatch.setattr(server, 'load_repository', lambda: repository)
    monkeypatch.setattr(server, 'load_planner', lambda: FakeModel([('query_data', {'metric': 'users'})]))
    with TestClient(server.app, base_url='http://localhost') as session:
        yield session


def headers():
    return {'Origin': 'http://localhost'}


def test_dashboard_routes_work_without_login(client):
    for path in ['/', '/store/1', '/api/store/1', '/api/stores']:
        response = client.get(path)
        assert response.status_code == 200
        assert 'www-authenticate' not in response.headers
    assert client.post('/api/assistant', json={'message': 'users?'}, headers=headers()).status_code == 200


def test_store_allowlist_cannot_be_overridden_by_browser(client):
    assert client.get('/api/store/2', headers=headers()).status_code == 403
    assert [row['store_id'] for row in client.get('/api/stores', headers=headers()).json()] == [1]
    response = client.post('/api/view', headers=headers(), json={'state': {'store_id': 2}})
    assert response.status_code == 403


def test_origin_and_schema_are_enforced(client):
    bad = headers() | {'Origin': 'https://attacker.example'}
    assert client.post('/api/assistant', headers=bad, json={'message': 'users?'}).status_code == 403
    assert client.post('/api/assistant', headers=headers(), json={'message': 'x', 'sql': 'SELECT 1'}).status_code == 422


def test_view_filters_update_html_metrics_chart_and_mask_rows(client):
    response = client.post('/api/view', headers=headers(), json={'state': {'tab': 'users', 'filters': {'email_prefixes': ['s']}}})
    assert response.status_code == 200
    result = response.json()
    assert result['preview']['users']['unique'] == 1
    assert result['dashboard']['total_revenue_usd'] == 10.01
    assert result['dashboard']['chart']['revenue_usd'] == [10.01]
    assert 'Sam@example.com' not in response.text
    assert '10.01' in result['dashboard_html']
    assert 'http-equiv' not in result['dashboard_html']


def test_empty_filter_is_valid_and_does_not_change_other_browser(client):
    response = client.post('/api/view', headers=headers(), json={'state': {'filters': {'email_prefixes': ['z']}}})
    assert response.status_code == 200
    assert response.json()['dashboard']['order_count'] == 0
    assert client.get('/api/store/1', headers=headers()).json()['order_count'] == 3


def test_assistant_result_is_grounded_and_no_cache(client):
    response = client.post('/api/assistant', headers=headers(), json={'message': 'How many users?'})
    assert response.status_code == 200
    assert response.json()['results'][0]['data']['rows'][0]['value'] == 3
    assert response.headers['cache-control'] == 'no-store'


def test_vercel_does_not_require_login(client, monkeypatch):
    monkeypatch.setenv('VERCEL', '1')
    response = client.get('/api/stores')
    assert response.status_code == 200
    assert 'www-authenticate' not in response.headers


def test_origin_is_detected_from_current_https_domain(client):
    import server
    with TestClient(server.app, base_url='https://dashboard.example') as session:
        response = session.post('/api/view', headers={'Origin': 'https://dashboard.example'}, json={'state': {'store_id': 1}})
        assert response.status_code == 200
        assert response.json()['preview']['users']['unique'] == 3
        assert session.post('/api/view', headers={'Origin': 'https://another.example'}, json={'state': {'store_id': 1}}).status_code == 403


def test_post_still_requires_origin_and_json(client):
    assert client.post('/api/view', json={'state': {'store_id': 1}}).status_code == 403
    assert client.post('/api/view', headers=headers(), content='state=1').status_code == 403


def test_html_contains_no_raw_script_breakout(client, repository):
    repository.stores[1].books.loc[0, 'title'] = '</script><script>alert(1)</script>'
    response = client.post('/api/view', headers=headers(), json={'state': {'tab': 'books'}})
    assert '</script><script>alert(1)' not in response.json()['dashboard_html']


def test_assistant_command_returns_changed_dashboard_and_browser_state(client, monkeypatch):
    import server
    monkeypatch.setattr(server, 'load_planner', lambda: FakeModel([('set_dashboard_view', {'store_id': 1, 'tab': 'books', 'filters': {'person_name': 'exclude'}})]))
    response = client.post('/api/assistant', headers=headers(), json={'message': 'Exclude books with person names'})
    assert response.status_code == 200
    result = response.json()
    assert result['action']['state']['filters']['person_name'] == 'exclude'
    assert result['view']['preview']['books']['total'] == 1
    assert result['view']['dashboard']['total_revenue_usd'] == 9.99


def test_missing_openai_key_returns_error_without_changing_view(client, monkeypatch):
    import server
    from assistant import ProviderError

    def unavailable():
        raise ProviderError('OPENAI_API_KEY is not configured')

    monkeypatch.setattr(server, 'load_planner', unavailable)
    result = client.post('/api/assistant', headers=headers(), json={'message': 'Apply filters'})
    assert result.status_code == 503
    assert client.get('/api/store/1', headers=headers()).json()['order_count'] == 3


def test_quota_error_returns_429_without_changing_filters(client, monkeypatch):
    import server
    from assistant import ProviderError

    def exhausted():
        raise ProviderError('OpenAI project rate limit exceeded.', status_code=429)

    monkeypatch.setattr(server, 'load_planner', exhausted)
    response = client.post('/api/assistant', headers=headers(), json={'message': 'Select users with s or p emails'})
    assert response.status_code == 429
    assert 'limit' in response.json()['detail']
    assert 'action' not in response.json()
    assert client.get('/api/store/1', headers=headers()).json()['order_count'] == 3


def test_openai_client_is_reused_and_closed_on_server_shutdown(repository, monkeypatch):
    import server
    from assistant import OpenAIPlanner
    from types import SimpleNamespace

    created = []

    class Connection:
        closed = False

        def __init__(self):
            self.responses = self

        def create(self, **kwargs):
            assert not self.closed
            return SimpleNamespace(status='completed', output=[SimpleNamespace(type='function_call', name='query_data', arguments='{"metric":"users"}')])

        def close(self):
            self.closed = True

    def create(cls, api_key, model):
        connection = Connection()
        created.append(connection)
        return OpenAIPlanner(connection, model)

    monkeypatch.setenv('ALLOWED_STORE_IDS', '1')
    monkeypatch.setattr(server, 'load_repository', lambda: repository)
    monkeypatch.setattr(OpenAIPlanner, 'from_settings', classmethod(create))
    with TestClient(server.app, base_url='http://localhost') as session:
        for _ in range(2):
            response = session.post('/api/assistant', headers=headers(), json={'message': 'How many users?'})
            assert response.status_code == 200
            assert response.json()['results'][0]['data']['rows'][0]['value'] == 3
        assert len(created) == 1
        assert not created[0].closed
    assert created[0].closed


def pagination_users(repository):
    import pandas as pd
    repository.stores[1].users = pd.DataFrame({
        'user_id': range(1, 66), 'canonical_user_id': range(1, 66),
        'email': [f'user{i}@example.com' for i in range(1, 66)],
    })


def test_preview_defaults_to_ten_and_uses_page_size_for_offsets(client, repository):
    pagination_users(repository)
    first = client.post('/api/view', headers=headers(), json={'state': {'store_id': 1}}).json()['preview']
    assert first['page_size'] == 10
    assert [row['user_id'] for row in first['users']['rows']] == list(range(1, 11))
    second = client.post('/api/view', headers=headers(), json={'state': {'store_id': 1}, 'page': 2}).json()['preview']
    assert [row['user_id'] for row in second['users']['rows']] == list(range(11, 21))
    last = client.post('/api/view', headers=headers(), json={'state': {'store_id': 1}, 'page': 4, 'page_size': 20}).json()['preview']
    assert [row['user_id'] for row in last['users']['rows']] == [61, 62, 63, 64, 65]


@pytest.mark.parametrize('page_size', [10, 20, 30, 40, 50])
def test_preview_supports_each_rows_per_page_option(client, repository, page_size):
    pagination_users(repository)
    response = client.post('/api/view', headers=headers(), json={'state': {'store_id': 1}, 'page_size': page_size})
    assert response.status_code == 200
    assert len(response.json()['preview']['users']['rows']) == page_size
    assert response.json()['preview']['page_size'] == page_size


@pytest.mark.parametrize('page_size', [1, 15, 100])
def test_unsupported_rows_per_page_is_rejected(client, page_size):
    response = client.post('/api/view', headers=headers(), json={'state': {'store_id': 1}, 'page_size': page_size})
    assert response.status_code == 422


def test_assistant_view_change_keeps_selected_page_size(client, repository, monkeypatch):
    import server
    pagination_users(repository)
    monkeypatch.setattr(server, 'load_planner', lambda: FakeModel([('set_dashboard_view', {'store_id': 1, 'tab': 'users', 'filters': {}})]))
    response = client.post('/api/assistant', headers=headers(), json={'message': 'Show all users', 'page_size': 30})
    assert response.status_code == 200
    result = response.json()
    assert result['view']['preview']['page_size'] == 30
    assert len(result['view']['preview']['users']['rows']) == 30
    assert len(result['results'][0]['data']['users']['rows']) == 30
