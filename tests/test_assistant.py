import json

import pytest

from schemas import AskRequest


class FakeModel:
    def __init__(self, calls):
        self.calls = calls
        self.requests = []

    def generate(self, message, state, allowed):
        self.requests.append({'message': message, 'state': state.model_dump(mode='json'), 'allowed': sorted(allowed)})
        return self.calls


def test_model_returns_query_and_server_produces_count(repository):
    from assistant import Assistant
    model = FakeModel([('query_data', {'metric': 'users'})])
    result = Assistant(repository, model).ask(AskRequest(message='Сколько пользователей?'), {1})
    assert result['results'][0]['data']['rows'][0]['value'] == 3
    assert '3' in result['answer']
    assert result['action'] is None
    assert 'Sam@example.com' not in json.dumps(model.requests)


def test_filter_command_changes_state_and_preview(repository):
    from assistant import Assistant
    model = FakeModel([('set_dashboard_view', {'store_id': 1, 'tab': 'users', 'filters': {'email_prefixes': ['s', 'p']}})])
    result = Assistant(repository, model).ask(AskRequest(message='select users with emails starting s or p'), {1})
    assert result['action']['state']['tab'] == 'users'
    assert result['action']['state']['filters']['email_prefixes'] == ['s', 'p']
    assert result['results'][0]['data']['users']['unique'] == 2


@pytest.mark.parametrize('calls', [
    [('run_sql', {'sql': 'SELECT * FROM users'})],
    [('query_data', {'metric': 'users', 'python': 'anything'})],
    [('query_data', {'metric': 'users', 'store_ids': [2]})],
    [('set_dashboard_view', {'store_id': 2})],
    [('query_data', {'metric': 'users'})] * 5,
    [],
    [('set_dashboard_view', {'store_id': 1}), ('set_dashboard_view', {'store_id': 1})],
])
def test_invalid_calls_never_execute_partially(repository, calls):
    from assistant import Assistant
    with pytest.raises((ValueError, PermissionError)):
        Assistant(repository, FakeModel(calls)).ask(AskRequest(message='test'), {1})


def test_schema_does_not_offer_code_or_raw_row_tools():
    from assistant import tool_declarations
    declarations = tool_declarations()
    assert {x['name'] for x in declarations} == {'query_data', 'best_buyers', 'set_dashboard_view', 'clarify'}
    assert all(x['strict'] for x in declarations)


def test_mixed_action_and_query_is_rejected_as_ambiguous(repository):
    from assistant import Assistant
    calls = [('set_dashboard_view', {'store_id': 1}), ('query_data', {'metric': 'users'})]
    with pytest.raises(ValueError):
        Assistant(repository, FakeModel(calls)).ask(AskRequest(message='do both'), {1})


def test_incomplete_action_cannot_clear_filters_or_change_store(repository):
    from assistant import Assistant
    request = AskRequest.model_validate({'message': 'switch to users', 'state': {'store_id': 2, 'filters': {'email_prefixes': ['p']}}})
    with pytest.raises(ValueError):
        Assistant(repository, FakeModel([('set_dashboard_view', {'tab': 'users'})])).ask(request, {1, 2})


def test_redacted_state_cannot_become_an_applied_filter():
    from assistant import OpenAIPlanner
    from schemas import ViewState
    state = ViewState.model_validate({'filters': {'email_prefixes': ['person@example.com']}})
    with pytest.raises(ValueError, match='contact'):
        OpenAIPlanner(None).generate('Switch to books, keep filters', state, {1})


@pytest.mark.parametrize('call,prefix', [
    (('query_data', {'metric': 'users'}), 'Unique users'),
    (('clarify', {'reason': 'ambiguous_filter'}), 'Should I keep'),
    (('set_dashboard_view', {'store_id': 1, 'tab': 'users', 'filters': {}}), 'Filters applied.'),
])
def test_assistant_answers_in_english_even_for_russian_requests(repository, call, prefix):
    from assistant import Assistant
    response = Assistant(repository, FakeModel([call])).ask(AskRequest(message='Покажи пользователей'), {1})
    assert response['answer'].startswith(prefix)
