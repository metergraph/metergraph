from decimal import Decimal

import pytest

from metergraph_core import price_call_from_record

AT = '2026-11-19T00:00:00Z'


def record(**changes):
    return {'provider': 'anthropic', 'model': 'claude-sonnet-4-6',
            'ts': AT, 'endpoint': 'messages', 'status': 'error', 'error': True, **changes}


@pytest.mark.parametrize('status', [400, 401, 403, 404, 413, 429, 500, 529])
def test_anthropic_http_rejection_without_usage_has_known_zero_cost(status):
    result = price_call_from_record(record(http_status_code=status))
    assert result.status == 'priced'
    assert result.cost_usd == Decimal(0)
    assert 'provider_failure_not_billable' in result.reasons


def test_unknown_model_does_not_make_confirmed_free_rejection_unpriced():
    result = price_call_from_record(record(model='example-unknown-model', http_status_code=429))
    assert result.status == 'priced'
    assert result.cost_usd == Decimal(0)


@pytest.mark.parametrize('changes', [
    {'endpoint': 'batch.anthropic.results', 'http_status_code': 401},
    {'endpoint': 'messages.batches.results', 'http_status_code': 401},
    {'endpoint': None, 'http_status_code': 401},
    {'batch': True, 'http_status_code': 401},
    {'stream': True},
    {'http_status_code': 200, 'stream': True},
    {'error_type': 'AnthropicError'},
    {'error_code': 'overloaded_error'},
    {'http_status_code': 429, 'input_tokens': 100},
    {'http_status_code': 429, 'input_tokens': False},
    {'http_status_code': 429, 'web_search_calls': 1},
    {'provider': 'openai', 'model': 'gpt-5.4-mini', 'http_status_code': 429},
])
def test_client_errors_or_ambiguous_evidence_do_not_become_zero(changes):
    result = price_call_from_record(record(**changes))
    assert result.cost_usd is None
    assert 'provider_failure_not_billable' not in result.reasons


def test_partial_stream_counters_are_an_estimate_not_a_final_charge():
    result = price_call_from_record(record(stream=True, usage_complete=False,
        input_tokens=1000, output_tokens=10, cache_read_tokens=0, cache_write_tokens=0))
    assert result.status == 'partial'
    assert result.cost_usd is None
    assert result.estimated_total_usd is not None
    assert result.estimated_total_usd > 0
    assert 'incomplete_usage' in result.missing_dimensions


def test_older_failed_stream_with_counters_also_stays_incomplete():
    result = price_call_from_record(record(stream=True, input_tokens=1000, output_tokens=0))
    assert result.status == 'partial'
    assert result.cost_usd is None


def test_successful_zero_token_call_remains_a_known_zero():
    result = price_call_from_record(record(error=False, status='end_turn',
        input_tokens=0, output_tokens=0, usage_complete=True))
    assert result.status == 'priced'
    assert result.cost_usd == Decimal(0)


def test_qualified_reported_charge_is_preserved_even_when_catalog_classifies_rejection_free():
    from metergraph_core.billing import normalize_gateway_evidence, resolve_billing

    row = record(http_status_code=429, gateway='portkey', reported_cost_usd='.01',
                 reported_cost_source='portkey.cost')
    priced = price_call_from_record(row)
    decision = resolve_billing(priced, normalize_gateway_evidence(row), at=AT)
    assert decision.cost_usd == Decimal('.01')
    assert decision.cost_provenance == 'gateway_reported'


@pytest.mark.parametrize('status', [400, 401, 403, 404, 413, 422, 429])
@pytest.mark.parametrize('provider,model,endpoint', [
    ('google', 'projects/example/locations/global/publishers/google/models/gemini-4-argon', 'models.generate_content'),
    ('vertex-ai', 'gemini-4-argon', 'models.generate_content.stream'),
])
def test_vertex_http_rejection_without_usage_has_known_zero_cost(status, provider, model, endpoint):
    result = price_call_from_record(record(provider=provider, model=model, endpoint=endpoint,
        http_status_code=status, stream=True, usage_complete=False))
    assert result.status == 'priced'
    assert result.cost_usd == Decimal(0)
    assert 'provider_failure_not_billable' in result.reasons


@pytest.mark.parametrize('changes', [
    {'http_status_code': None}, {'http_status_code': '400'}, {'http_status_code': 200},
    {'http_status_code': 408}, {'http_status_code': 500}, {'http_status_code': 504},
    {'server_tool_use': 'invalid'}, {'server_tool_use': {'web_search_requests': 1}},
    {'url_fetch_calls': 1},
    {'batch': True}, {'endpoint': 'batches.results'}, {'endpoint': None},
    {'input_tokens': 1}, {'output_tokens': 1}, {'cache_read_tokens': 1},
    {'grounding_queries': 1}, {'grounded_prompt': 1}, {'web_search_calls': 1},
    {'input_tokens': False}, {'grounding_queries': -1},
    {'provider': 'google', 'model': 'gemini-4-argon'},
    {'provider': 'openai', 'model': 'gpt-6-luna'},
    {'status': 'success', 'error': False},
])
def test_vertex_ambiguous_or_positive_usage_does_not_become_free(changes):
    row = record(provider='google',
        model='projects/example/locations/global/publishers/google/models/gemini-4-argon',
        endpoint='models.generate_content', http_status_code=400)
    row.update(changes)
    result = price_call_from_record(row)
    assert 'provider_failure_not_billable' not in result.reasons


def test_vertex_rejection_with_prior_function_call_has_known_zero_cost():
    result = price_call_from_record(record(provider='vertex-ai', model='gemini-4-argon',
        endpoint='models.generate_content', http_status_code=400,
        tool_calls=[{'id': 'previous-call', 'name': 'lookup_weather',
                     'arguments': {'city': 'Example City'}, 'status': 'completed'}]))
    assert result.status == 'priced'
    assert result.cost_usd == Decimal(0)
    assert 'provider_failure_not_billable' in result.reasons
