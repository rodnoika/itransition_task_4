import json

import pytest
from google.genai import types

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
    assert {x.name for x in declarations} == {'query_data', 'best_buyers', 'set_dashboard_view', 'clarify'}
    types.GenerateContentConfig(tools=[types.Tool(function_declarations=declarations)])


def test_provider_ignores_invented_prose_and_uses_real_sdk_types():
    from assistant import GeminiPlanner
    from schemas import ViewState

    class Models:
        def generate_content(self, **kwargs):
            assert kwargs['model'] == 'gemini-3.1-pro-preview'
            assert kwargs['config'].automatic_function_calling.disable is True
            assert str(kwargs['config'].tool_config.function_calling_config.mode).endswith('ANY')
            assert 'Sam@example.com' not in str(kwargs)
            return types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(role='model', parts=[
                types.Part(text='There are 999999 users'),
                types.Part(function_call=types.FunctionCall(name='query_data', args={'metric': 'users'}), thought_signature=b'signature'),
            ]))])

    class Client:
        models = Models()

    assert GeminiPlanner(Client()).generate('How many users?', ViewState(), {1}) == [('query_data', {'metric': 'users'})]


def test_mixed_action_and_query_is_rejected_as_ambiguous(repository):
    from assistant import Assistant
    calls = [('set_dashboard_view', {'store_id': 1}), ('query_data', {'metric': 'users'})]
    with pytest.raises(ValueError):
        Assistant(repository, FakeModel(calls)).ask(AskRequest(message='do both'), {1})


def test_sdk_transport_receives_no_contacts_and_keeps_strict_function_schema():
    import httpx
    from google import genai
    from assistant import GeminiPlanner
    from schemas import ViewState

    captured = []

    def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={'candidates': [{'content': {'role': 'model', 'parts': [{'functionCall': {'name': 'query_data', 'args': {'metric': 'users'}}}]}, 'finishReason': 'STOP'}]})

    transport = httpx.Client(transport=httpx.MockTransport(handler))
    with genai.Client(api_key='test-key-not-a-real-secret', http_options=types.HttpOptions(httpx_client=transport)) as client:
        calls = GeminiPlanner(client).generate('Email john@example.com phone +1 (234) 567-8901', ViewState(), {1})
    assert calls == [('query_data', {'metric': 'users'})]
    sent = json.dumps(captured)
    assert 'john@example.com' not in sent
    assert '567-8901' not in sent
    assert captured[0]['toolConfig']['functionCallingConfig']['mode'] == 'ANY'
    declaration = captured[0]['tools'][0]['functionDeclarations'][0]
    schema = declaration.get('parametersJsonSchema', declaration.get('parameters_json_schema'))
    assert schema['additionalProperties'] is False


def test_incomplete_action_cannot_clear_filters_or_change_store(repository):
    from assistant import Assistant
    request = AskRequest.model_validate({'message': 'switch to users', 'state': {'store_id': 2, 'filters': {'email_prefixes': ['p']}}})
    with pytest.raises(ValueError):
        Assistant(repository, FakeModel([('set_dashboard_view', {'tab': 'users'})])).ask(request, {1, 2})


def test_redacted_state_cannot_become_an_applied_filter():
    from assistant import GeminiPlanner
    from schemas import ViewState
    state = ViewState.model_validate({'filters': {'email_prefixes': ['person@example.com']}})
    with pytest.raises(ValueError, match='contact'):
        GeminiPlanner(None).generate('Switch to books, keep filters', state, {1})


def test_provider_quota_error_is_clear_and_does_not_expose_google_details():
    import httpx
    from google import genai
    from assistant import GeminiPlanner, ProviderError
    from schemas import ViewState

    def exhausted(request):
        return httpx.Response(429, json={'error': {'code': 429, 'status': 'RESOURCE_EXHAUSTED', 'message': 'Internal project diagnostic that must not be shown'}})

    transport = httpx.Client(transport=httpx.MockTransport(exhausted))
    with genai.Client(api_key='test-only', http_options=types.HttpOptions(httpx_client=transport, retry_options=types.HttpRetryOptions(attempts=1))) as client:
        with pytest.raises(ProviderError) as raised:
            GeminiPlanner(client).generate('Select users with s or p emails', ViewState(), {1})
    assert raised.value.status_code == 429
    assert 'limit' in str(raised.value).casefold()
    assert 'Internal project' not in str(raised.value)


@pytest.mark.parametrize('model,level', [('gemini-3.1-flash-lite', 'MINIMAL'), ('gemini-3.1-pro-preview', 'LOW')])
def test_router_uses_supported_fast_thinking_level(model, level):
    from assistant import GeminiPlanner
    from schemas import ViewState

    class Models:
        def generate_content(self, **kwargs):
            assert kwargs['config'].thinking_config.thinking_level.value == level
            return types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(role='model', parts=[types.Part(function_call=types.FunctionCall(name='query_data', args={'metric': 'users'}))]))])

    class Client:
        models = Models()

    assert GeminiPlanner(Client(), model).generate('How many users?', ViewState(), {1}) == [('query_data', {'metric': 'users'})]


def test_busy_provider_returns_clear_temporary_error():
    import httpx
    from google import genai
    from assistant import GeminiPlanner, ProviderError
    from schemas import ViewState

    def unavailable(request):
        return httpx.Response(503, json={'error': {'code': 503, 'status': 'UNAVAILABLE', 'message': 'Internal busy-service diagnostic'}})

    transport = httpx.Client(transport=httpx.MockTransport(unavailable))
    with genai.Client(api_key='test-only', http_options=types.HttpOptions(httpx_client=transport, retry_options=types.HttpRetryOptions(attempts=1))) as client:
        with pytest.raises(ProviderError) as raised:
            GeminiPlanner(client).generate('How many users?', ViewState(), {1})
    assert raised.value.status_code == 503
    assert 'temporarily' in str(raised.value).casefold()
    assert 'Internal busy' not in str(raised.value)


@pytest.mark.parametrize('call,prefix', [
    (('query_data', {'metric': 'users'}), 'Unique users'),
    (('clarify', {'reason': 'ambiguous_filter'}), 'Should I keep'),
    (('set_dashboard_view', {'store_id': 1, 'tab': 'users', 'filters': {}}), 'Filters applied.'),
])
def test_assistant_answers_in_english_even_for_russian_requests(repository, call, prefix):
    from assistant import Assistant
    response = Assistant(repository, FakeModel([call])).ask(AskRequest(message='Покажи пользователей'), {1})
    assert response['answer'].startswith(prefix)
