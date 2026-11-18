from decimal import Decimal

import pytest
from metergraph_core import price_call_from_record


AT = '2026-11-18T12:00:00Z'


def record(**extra):
    return {'provider': 'perplexity', 'model': 'perplexity/sonar', 'retrieval_api': 'agent',
            'search_type': 'web', 'web_search_calls': 3, 'input_tokens': 1000,
            'output_tokens': 100, 'ts': AT, **extra}


@pytest.mark.parametrize('kind, fee', [('web', '.0075'), ('fast', '.003')])
def test_sdk_agent_record_selects_channel_and_operation(kind, fee):
    result = price_call_from_record(record(search_type=kind))
    assert result.token_cost.price.pricing_channel == 'perplexity-agent-api'
    assert result.token_cost.cost_usd == Decimal('.0005')
    assert result.retrieval_cost_usd == Decimal(fee)
    assert result.cost_usd == Decimal(fee) + Decimal('.0005')
    assert result.search_units == 3 and result.status == 'priced'


@pytest.mark.parametrize('extra', [{'search_type': None}, {'search_type': 'unknown'},
    {'web_search_calls': None}, {'web_search_calls': True}, {'web_search_calls': -1}, {'web_search_calls': 1.5}])
def test_agent_unknown_usage_never_returns_tokens_as_a_complete_total(extra):
    result = price_call_from_record(record(**extra))
    assert result.status == 'partial' and result.cost_usd is None
    assert result.known_subtotal_usd == Decimal('.0005')
    assert result.retrieval_cost_usd is None


def test_zero_agent_searches_need_no_search_type():
    result = price_call_from_record(record(web_search_calls=0, search_type=None))
    assert result.cost_usd == Decimal('.0005') and result.search_units == 0


def test_agent_metadata_does_not_reprice_legacy_sonar():
    result = price_call_from_record(record(retrieval_api='sonar'))
    assert result.token_cost.price.pricing_channel == 'perplexity-api'
    assert result.status != 'priced' and result.cost_usd is None


@pytest.mark.parametrize('provider, model, usage, fee', [
    ('openai', 'gpt-5.4-mini', {'web_search_calls': 2}, '.02'),
    ('anthropic', 'claude-sonnet-4-6', {'server_tool_use': {'web_search_requests': 2}}, '.02'),
    ('google', 'gemini-3.8-flash', {'grounding_queries': 2, 'grounded_prompt': 1}, '.028'),
    ('vertex-ai', 'gemini-3.8-flash', {'grounding_queries': 2, 'grounded_prompt': 1}, '.028'),
])
def test_normalized_provider_records_share_the_call_pricer(provider, model, usage, fee):
    result = price_call_from_record({'provider': provider, 'model': model, 'input_tokens': 1000, 'output_tokens': 100, 'ts': AT, **usage})
    assert result.status == 'priced'
    assert result.retrieval_cost_usd == Decimal(fee) and result.search_units == 2
    assert result.cost_usd == result.token_cost.cost_usd + Decimal(fee)


def test_old_google_grounding_has_prompt_units_not_query_prices():
    result = price_call_from_record({'provider': 'google', 'model': 'gemini-2.5-flash', 'input_tokens': 1000, 'output_tokens': 100, 'ts': AT, 'grounding_queries': 2, 'grounded_prompt': 1})
    assert result.retrieval[0].operation == 'google_search_grounded_prompt'
    assert result.search_units == 1
    assert result.cost_usd is None and result.status == 'partial'
    assert result.retrieval_cost_usd is None


def test_record_keeps_unsupported_processing_tiers_explicitly_partial():
    result = price_call_from_record(record(service_tier='priority'))
    assert result.cost_usd is None and result.status == 'partial'
    assert 'service_tier' in result.missing_dimensions


def test_record_can_compose_an_existing_token_result_without_changing_totals():
    from metergraph_core import load_catalog, ModelUsage
    loaded = load_catalog()
    row = {'provider': 'openai', 'model': 'gpt-5.4-mini', 'ts': AT, 'input_tokens': 1000, 'output_tokens': 100, 'web_search_calls': 2}
    tokens = loaded.calculate_cost(provider='openai', model=row['model'], at=AT, usage=ModelUsage(input_tokens=1000, output_tokens=100))
    standalone = price_call_from_record(row, catalog=loaded)
    composed = price_call_from_record(row, catalog=loaded, token_cost=tokens)
    assert standalone == composed
