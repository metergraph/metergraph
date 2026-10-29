from decimal import Decimal

import pytest

from metergraph_core import (
    ModelUsage, RetrievalUsage, load_catalog, price_call, price_call_from_response,
    cost_from_response,
)

AT = '2026-10-20T00:00:00Z'


def response(search=2):
    return {'type': 'message', 'model': 'claude-sonnet-5-5', 'usage': {
        'input_tokens': 1000, 'output_tokens': 100,
        'server_tool_use': {'web_search_requests': search}}}


def test_native_response_includes_search_without_changing_token_api():
    catalog = load_catalog()
    result = price_call_from_response('anthropic', response(), catalog=catalog, at=AT)
    assert result.status == 'priced'
    assert result.cost_usd == Decimal('.023')
    assert result.known_subtotal_usd == result.cost_usd
    assert result.token_cost.cost_usd == Decimal('.003')
    assert result.retrieval[0].cost_usd == Decimal('.02')
    assert result.retrieval[0].price_id
    assert result.catalog_hash == catalog.content_hash
    assert cost_from_response('anthropic', response(), catalog=catalog, at=AT).cost_usd == Decimal('.003')


@pytest.mark.parametrize('count', [True, 1.5, -1, '2', None])
def test_invalid_count_is_not_a_priced_zero(count):
    result = price_call_from_response('anthropic', response(count), at=AT)
    assert result.status == 'partial'
    assert result.cost_usd is None
    assert result.known_subtotal_usd == Decimal('.003')
    assert result.retrieval[0].cost_usd is None
    assert 'retrieval.web_search' in result.missing_dimensions


def test_no_search_and_explicit_zero_are_complete():
    plain = response()
    del plain['usage']['server_tool_use']
    for data in (plain, response(0)):
        result = price_call_from_response('anthropic', data, at=AT)
        assert result.status == 'priced'
        assert result.cost_usd == Decimal('.003')


def test_malformed_server_usage_is_unresolved():
    data = response()
    data['usage']['server_tool_use'] = []
    assert price_call_from_response('anthropic', data, at=AT).status == 'partial'


def test_claude_on_vertex_does_not_inherit_google_search_rate():
    result = price_call_from_response('anthropic', response(), channel='google-vertex-ai', at=AT)
    assert result.status != 'priced'
    assert result.cost_usd is None
    assert result.retrieval[0].cost_usd is None
    assert 'unsupported_retrieval_scope' in result.retrieval[0].reasons


def test_unknown_model_does_not_present_search_as_complete_call():
    data = response()
    data['model'] = 'not-a-model'
    result = price_call_from_response('anthropic', data, at=AT)
    assert result.status == 'unpriced'
    assert result.cost_usd is None
    assert result.known_subtotal_usd == Decimal('.02')


def test_normalized_calls_share_pricing_and_free_fetch():
    result = price_call(model='claude-sonnet-5-5', provider='anthropic', at=AT,
        usage=ModelUsage(input_tokens=1000, output_tokens=100),
        retrieval=(RetrievalUsage('web_search', 2), RetrievalUsage('web_fetch', 1)))
    assert result.cost_usd == Decimal('.023')
    assert result.retrieval[1].basis == 'exact'
    assert result.retrieval[1].cost_usd == 0


def test_partial_tokens_are_estimates_not_known_subtotal():
    data = response()
    del data['usage']['output_tokens']
    result = price_call_from_response('anthropic', data, at=AT)
    assert result.status == 'partial'
    assert result.cost_usd is None
    assert result.token_basis == 'fallback_estimate'
    assert result.known_subtotal_usd == Decimal('.02')
    assert result.estimated_total_usd == Decimal('.022')


def test_unknown_operation_is_unresolved():
    result = price_call(model='claude-sonnet-5-5', provider='anthropic', at=AT,
        usage=ModelUsage(input_tokens=0, output_tokens=0),
        retrieval=(RetrievalUsage('unknown', 1),))
    assert result.status == 'partial'
    assert result.cost_usd is None


def test_unknown_retrieval_does_not_create_an_estimated_total():
    data = response('malformed')
    del data['usage']['output_tokens']
    result = price_call_from_response('anthropic', data, at=AT)
    assert result.estimated_total_usd is None
    assert result.retrieval[0].units is None


def test_tool_evidence_without_usage_count_stays_unknown():
    data = response()
    del data['usage']['server_tool_use']
    data['content'] = [{'type': 'server_tool_use', 'name': 'web_search'}]
    assert price_call_from_response('anthropic', data, at=AT).status == 'partial'


def test_historical_search_rate_and_fixed_request_fee_are_applied_once():
    from copy import deepcopy
    from metergraph_core import load_catalog_document
    doc = deepcopy(load_catalog().document)
    for model in doc['models']:
        if model['canonical_id'] == 'anthropic/claude-sonnet-5.5':
            for rate in model['prices']:
                if rate['channel'] == 'anthropic-api':
                    rate['request_fee_per_1k_usd'] = '1'
    for fee in doc['retrieval']:
        if fee['channel'] == 'anthropic-api' and fee['operation'] == 'web_search':
            fee['per_1k_usd'] = '8'
    catalog = load_catalog_document(doc)
    result = price_call_from_response('anthropic', response(), catalog=catalog, at=AT)
    assert result.cost_usd == Decimal('.020')  # .003 tokens + .001 request + .016 search
    assert result.token_cost.components.request_usd == Decimal('.001')


def test_duplicate_normalized_operations_are_not_charged_twice():
    with pytest.raises(ValueError, match='aggregated once'):
        price_call(model='claude-sonnet-5-5', provider='anthropic', at=AT,
            usage=ModelUsage(input_tokens=0, output_tokens=0),
            retrieval=(RetrievalUsage('web_search', 1), RetrievalUsage('web_search', 1)))


def test_complete_call_can_be_reconciled_without_adding_search_to_gateway_total():
    from metergraph_core import resolve_billing, normalize_gateway_evidence
    result = price_call_from_response('anthropic', response(), at=AT)
    decision = resolve_billing(result, normalize_gateway_evidence({}), at=AT)
    assert decision.cost_usd == Decimal('.023')
    assert decision.catalog_cost_usd == Decimal('.023')


def test_composition_rejects_token_price_from_another_snapshot():
    from copy import deepcopy
    from metergraph_core import compose_call_cost, load_catalog_document
    original = load_catalog()
    token = cost_from_response('anthropic', response(), catalog=original, at=AT)
    document = deepcopy(original.document)
    for model in document['models']:
        if model['canonical_id'] == token.canonical_model:
            for row in model['prices']:
                if row['channel'] == 'anthropic-api':
                    row['input_per_mtok'] = '999'
    changed = load_catalog_document(document)
    with pytest.raises(ValueError, match='catalog snapshot'):
        compose_call_cost(token, catalog=changed, retrieval=(RetrievalUsage('web_search', 2),))


def test_observed_gemini_grounding_is_unresolved_not_token_only_priced():
    data = {'modelVersion': 'gemini-3.8-flash',
        'usageMetadata': {'promptTokenCount': 1000, 'candidatesTokenCount': 100},
        'candidates': [{'groundingMetadata': {'webSearchQueries': ['example query']}}]}
    result = price_call_from_response('google', data, at=AT)
    assert result.status == 'partial'
    assert result.cost_usd is None
    assert result.retrieval[0].operation == 'google_search_grounding'
