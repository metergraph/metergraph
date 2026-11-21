from decimal import Decimal

import pytest

from metergraph_core import price_call_from_record


@pytest.mark.parametrize('name,tokens', [
    ('gemini-4-argon', '.0035'),
    ('gemini-4-hydrogen', '.002625'),
])
@pytest.mark.parametrize('resource', [False, True])
@pytest.mark.parametrize('tier,factor', [('standard', 1), ('flex', Decimal('.5'))])
def test_vertex_gemini4_prices_native_records_and_keeps_search_full_price(name, tokens, resource, tier, factor):
    row = {
        'provider': 'google' if resource else 'vertex-ai',
        'model': f'projects/example-project/locations/global/publishers/google/models/{name}' if resource else name,
        'ts': '2026-11-20T12:00:00Z',
        'input_tokens': 1000,
        # Normalized output already includes the 50 thinking tokens.
        'output_tokens': 150,
        'reasoning_tokens': 50,
        'grounding_queries': 2,
        'service_tier': tier,
    }
    result = price_call_from_record(row)
    assert result.status == 'priced'
    assert result.token_cost.price.pricing_channel == 'google-vertex-ai'
    assert result.token_cost.cost_usd == Decimal(tokens) * factor
    assert result.retrieval_cost_usd == Decimal('.028')
    assert result.cost_usd == Decimal(tokens) * factor + Decimal('.028')


def test_vertex_argon_introductory_rate_ends_december_first():
    row = dict(provider='vertex-ai', model='gemini-4-argon', input_tokens=1000, output_tokens=100)
    assert price_call_from_record({**row, 'ts': '2026-11-30T23:59:59Z'}).cost_usd == Decimal('.003')
    assert price_call_from_record({**row, 'ts': '2026-12-01T00:00:00Z'}).cost_usd == Decimal('.006')


def test_requested_flex_standard_fallback_keeps_full_vertex_token_price():
    result = price_call_from_record(dict(provider='vertex-ai', model='gemini-4-hydrogen',
        ts='2026-11-20T12:00:00Z', input_tokens=1000, output_tokens=100,
        requested_service_tier='flex', service_tier='standard'))
    assert result.status == 'priced'
    assert result.cost_usd == Decimal('.00225')
