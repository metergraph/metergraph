from decimal import Decimal

import pytest

from metergraph_core import price_call_from_record, price_call_from_response


def record(provider='openai', model='gpt-6-sol', **extra):
    return dict(provider=provider, model=model, ts='2026-11-22T12:00:00Z',
                input_tokens=1000, output_tokens=100, **extra)


@pytest.mark.parametrize('tier', ['priority', 'fast'])
def test_openai_priority_token_cache_and_search_rates(tier):
    result = price_call_from_record(record(service_tier=tier, cache_read_tokens=200, web_search_calls=1))
    assert result.status == 'priced'
    assert result.token_cost.cost_usd == Decimal('.00528')
    assert result.retrieval_cost_usd == Decimal('.01')
    assert result.cost_usd == Decimal('.01528')


def test_google_priority_and_standard_fallback():
    row = record('vertex-ai', 'gemini-4-hydrogen', service_tier='priority', grounding_queries=2)
    priority = price_call_from_record(row)
    assert priority.status == 'priced'
    assert priority.token_cost.cost_usd == Decimal('.00405')
    assert priority.cost_usd == Decimal('.03205')
    standard = price_call_from_record({**row, 'requested_service_tier': 'priority', 'service_tier': 'standard'})
    assert standard.status == 'priced'
    assert standard.cost_usd == Decimal('.03025')


def test_google_native_priority_response():
    response = dict(modelVersion='gemini-4-hydrogen', usageMetadata=dict(
        promptTokenCount=1000, candidatesTokenCount=100, trafficType='ON_DEMAND_PRIORITY'))
    result = price_call_from_response('vertex-ai', response, at='2026-11-22', format='gemini.generate_content')
    assert result.status == 'priced'
    assert result.cost_usd == Decimal('.00405')


def test_priority_long_context():
    row = record(service_tier='priority')
    result = price_call_from_record({**row, 'input_tokens': 300000})
    assert result.status == 'priced'
    assert result.cost_usd == Decimal('2.403')
    google = price_call_from_record({**row, 'provider': 'vertex-ai', 'model': 'gemini-3.1-pro-preview', 'input_tokens': 300000})
    assert google.status == 'priced'
    assert google.cost_usd == Decimal('2.16324')


@pytest.mark.parametrize('changes', [
    {'provider': 'anthropic', 'model': 'claude-sonnet-4-6'},
    {'provider': 'openrouter', 'model': 'openai/gpt-6-sol'},
    {'model': 'gpt-5.4-nano'},
    {'ts': '2026-11-21'},
    {'model': 'gpt-5.4', 'input_tokens': 300000},
])
def test_unverified_priority_rates_stay_partial(changes):
    result = price_call_from_record({**record(service_tier='priority'), **changes})
    assert result.status == 'partial'


@pytest.mark.parametrize('count,total', [(0, '.00225'), (1, '.01225')])
def test_search_api_known_count_resolves_catalog_fee_warning(count, total):
    result = price_call_from_record(record(model='gpt-5-search-api', web_search_calls=count))
    assert result.status == 'priced'
    assert result.cost_usd == Decimal(total)
    assert result.token_cost.reasons == ()


def test_search_api_missing_count_remains_partial():
    for extra in ({}, {'web_search_calls': None}):
        result = price_call_from_record(record(model='gpt-5-search-api', **extra))
        assert result.status == 'partial'
        assert result.cost_usd is None


@pytest.mark.parametrize('tier', ['priority', 'fast'])
def test_openai_native_served_priority(tier):
    result = price_call_from_response('openai', dict(model='gpt-6-sol', service_tier=tier,
        usage=dict(input_tokens=1000, output_tokens=100)), at='2026-11-22', format='openai.responses')
    assert result.status == 'priced'
    assert result.cost_usd == Decimal('.006')


def test_known_search_does_not_hide_missing_tokens_or_unrelated_gateway_fees():
    result = price_call_from_record({**record(model='gpt-5-search-api', web_search_calls=1), 'output_tokens': None})
    assert result.status == 'partial'
    gateway = price_call_from_record(record('openrouter', 'perplexity/sonar', web_search_calls=1))
    assert gateway.status == 'partial'


def test_priority_google_promo_boundary_and_cache():
    row = record('vertex-ai', 'gemini-4-argon', service_tier='priority', cache_read_tokens=200)
    assert price_call_from_record(row).cost_usd == Decimal('.004716')
    assert price_call_from_record({**row, 'ts': '2026-12-01'}).cost_usd == Decimal('.009432')


def test_priority_batch_conflict_stays_partial():
    assert price_call_from_record(record(service_tier='priority', batch=True)).status == 'partial'


@pytest.mark.parametrize('change', [
    {'input_per_mtok': -1}, {'output_per_mtok': 'invalid'},
    {'cache_read_per_mtok': float('nan')}, {'effective_from': 'invalid'},
    {'max_input_tokens': True},
])
def test_invalid_priority_catalog_rates_are_rejected(change):
    from metergraph_core import CatalogError, load_catalog
    from metergraph_core.loader import parse_catalog
    document = load_catalog().document
    row = next(p for m in document['models'] if m['canonical_id'] == 'openai/gpt-6-sol'
               for p in m['prices'] if p['channel'] == 'openai-api')
    row['rules']['priority'].update(change)
    with pytest.raises(CatalogError):
        parse_catalog(document)


def test_openai_priority_cache_write_rates():
    result = price_call_from_record(record(service_tier='priority', cache_write_5m_tokens=50, cache_write_1h_tokens=25))
    assert result.status == 'priced'
    assert result.token_cost.components.cache_write_usd == Decimal('.0005')
    # The 75 write tokens use the write rates instead of ordinary input.
    assert result.cost_usd == Decimal('.0062')


def test_priority_reprices_a_supplied_standard_token_result():
    from metergraph_core import ModelUsage, load_catalog
    catalog = load_catalog()
    standard = catalog.calculate_cost(model='gpt-6-sol', provider='openai',
        at='2026-11-22', usage=ModelUsage(input_tokens=1000, output_tokens=100))
    result = price_call_from_record(record(service_tier='priority'), catalog=catalog, token_cost=standard)
    assert result.status == 'priced'
    assert result.cost_usd == Decimal('.006')


def test_known_search_fee_does_not_promote_incomplete_tokens():
    result = price_call_from_record(record(model='gpt-5-search-api', web_search_calls=1, usage_complete=False))
    assert result.status == 'partial'
    assert result.token_cost.status == 'partial'
    assert result.token_basis != 'exact'
    assert result.known_subtotal_usd == Decimal('.01')
