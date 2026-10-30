import json

import httpx
import pytest
from openai import OpenAI

from schemas import AskRequest, ViewState


def response_body(output=None, status='completed'):
    return {
        'id': 'resp_test', 'object': 'response', 'created_at': 0,
        'model': 'gpt-5.6-luna', 'status': status,
        'output': output if output is not None else [
            {'type': 'function_call', 'id': 'fc_test', 'call_id': 'call_test',
             'name': 'query_data', 'arguments': '{"metric":"users"}', 'status': 'completed'}
        ],
        'parallel_tool_calls': True, 'tool_choice': 'required', 'tools': [],
    }


def make_client(handler):
    return OpenAI(api_key='test-only-not-a-real-secret', max_retries=0,
                  http_client=httpx.Client(transport=httpx.MockTransport(handler)))


def assert_strict_schema(schema):
    if isinstance(schema, dict):
        if schema.get('type') == 'object':
            assert schema.get('additionalProperties') is False
            assert set(schema.get('required', [])) == set(schema.get('properties', {}))
        assert 'default' not in schema
        for value in schema.values():
            assert_strict_schema(value)
    elif isinstance(schema, list):
        for value in schema:
            assert_strict_schema(value)


def test_openai_transport_uses_responses_strict_tools_and_redacts_contacts(repository):
    from assistant import Assistant, OpenAIPlanner
    captured = []

    def handler(request):
        assert request.url.path == '/v1/responses'
        captured.append(json.loads(request.content))
        output = [
            {'type': 'message', 'id': 'msg_test', 'role': 'assistant', 'status': 'completed',
             'content': [{'type': 'output_text', 'text': 'There are 999999 users', 'annotations': []}]},
            response_body()['output'][0],
        ]
        return httpx.Response(200, json=response_body(output))

    with make_client(handler) as client:
        result = Assistant(repository, OpenAIPlanner(client)).ask(
            AskRequest(message='Count users; email john@example.com phone +1 (234) 567-8901'), {1})
    assert result['results'][0]['data']['rows'][0]['value'] == 3
    assert '999999' not in result['answer']
    sent = captured[0]
    assert sent['model'] == 'gpt-5.6-luna'
    assert sent['tool_choice'] == 'required'
    assert sent['store'] is False
    assert sent['reasoning'] == {'effort': 'none'}
    assert {tool['name'] for tool in sent['tools']} == {'query_data','best_buyers','set_dashboard_view','clarify'}
    for tool in sent['tools']:
        assert tool['type'] == 'function' and tool['strict'] is True
        assert_strict_schema(tool['parameters'])
    wire = json.dumps(sent)
    assert 'john@example.com' not in wire and '567-8901' not in wire and 'Sam@example.com' not in wire


@pytest.mark.parametrize('status,code,expected', [
    (429, 'rate_limit_exceeded', 'rate limit'),
    (429, 'insufficient_quota', 'billing'),
    (401, 'invalid_api_key', 'OPENAI_API_KEY'),
    (403, 'permission_denied', 'permissions'),
    (404, 'model_not_found', 'OPENAI_MODEL'),
    (503, 'server_error', 'temporarily'),
])
def test_openai_errors_are_clear_and_never_echo_provider_details(status, code, expected):
    from assistant import OpenAIPlanner, ProviderError

    def handler(request):
        return httpx.Response(status, json={'error': {'type': code, 'code': code,
            'message': 'Private upstream diagnostic test-only-not-a-real-secret'}})

    with make_client(handler) as client, pytest.raises(ProviderError) as raised:
        OpenAIPlanner(client).generate('How many users?', ViewState(), {1})
    assert expected.casefold() in str(raised.value).casefold()
    assert raised.value.status_code == (429 if status == 429 else 503)
    assert 'Private upstream' not in str(raised.value)
    assert 'test-only-not-a-real-secret' not in str(raised.value)


@pytest.mark.parametrize('arguments', ['{', '[]', 'null'])
def test_malformed_openai_arguments_fail_before_execution(arguments):
    from assistant import OpenAIPlanner, ProviderError
    body = response_body()
    body['output'][0]['arguments'] = arguments
    with make_client(lambda request: httpx.Response(200, json=body)) as client, pytest.raises(ProviderError):
        OpenAIPlanner(client).generate('How many users?', ViewState(), {1})


@pytest.mark.parametrize('body', [response_body(status='incomplete'), response_body(output=[])])
def test_incomplete_or_missing_calls_do_not_become_answers(body):
    from assistant import OpenAIPlanner, ProviderError
    with make_client(lambda request: httpx.Response(200, json=body)) as client, pytest.raises(ProviderError):
        OpenAIPlanner(client).generate('How many users?', ViewState(), {1})


def test_missing_openai_key_is_explicit():
    from assistant import OpenAIPlanner, ProviderError
    with pytest.raises(ProviderError, match='OPENAI_API_KEY'):
        OpenAIPlanner.from_settings('', 'gpt-5.6-luna')


def test_configured_model_is_preserved_without_model_specific_reasoning():
    from assistant import OpenAIPlanner
    captured = []
    def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json=response_body())
    with make_client(handler) as client:
        assert OpenAIPlanner(client,'configured-model').generate('How many users?',ViewState(),{1}) == [('query_data',{'metric':'users'})]
    assert captured[0]['model'] == 'configured-model'
    assert 'reasoning' not in captured[0]
