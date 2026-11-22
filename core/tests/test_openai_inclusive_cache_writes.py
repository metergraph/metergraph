from copy import deepcopy
from decimal import Decimal

import pytest

from metergraph_core import load_catalog, load_catalog_document, price_call_from_record


AT = '2026-11-22T12:00:00Z'


def row(tier, writes=100):
    return dict(provider='openai', model='gpt-6-luna', ts=AT,
                input_tokens=1000, output_tokens=100, cache_read_tokens=200,
                cache_write_tokens=writes, service_tier=tier, batch=tier == 'batch')


@pytest.mark.parametrize('tier,expected', [
    ('standard', '.0001345'), ('flex', '.00006725'),
    ('batch', '.0000745'), ('priority', '.000269'),
])
def test_native_openai_writes_are_part_of_input_total(tier, expected):
    # Ordinary input = 1000 - 200 cache reads - 100 cache writes.
    # https://developers.openai.com/api/docs/guides/prompt-caching
    result = price_call_from_record(row(tier))
    assert result.status == 'priced'
    assert result.cost_usd == Decimal(expected)


@pytest.mark.parametrize('tier,expected', [
    ('standard', '.000132'), ('flex', '.000066'),
    ('batch', '.000067'), ('priority', '.000264'),
])
def test_no_write_usage_keeps_existing_totals(tier, expected):
    assert price_call_from_record(row(tier, writes=0)).cost_usd == Decimal(expected)


def catalog_with_rules(rules):
    document = deepcopy(load_catalog().document)
    model = next(m for m in document['models'] if m['canonical_id'] == 'openai/gpt-6-luna')
    for price in model['prices']:
        if price['channel'] == 'openai-api':
            price['rules'] = {**price.get('rules', {}), **rules}
    return load_catalog_document(document)


def test_explicit_exclusive_input_convention_is_respected():
    catalog = catalog_with_rules({'input_includes_cache_write': False})
    result = price_call_from_record(row('standard'), catalog=catalog)
    assert result.cost_usd == Decimal('.0001445')


def test_inclusive_writes_are_not_added_again_to_context_threshold():
    catalog = catalog_with_rules({'long_context': {
        'threshold': 1050, 'threshold_counts_cached_tokens': True,
        'input_multiplier': 2, 'output_multiplier': 1,
    }})
    assert price_call_from_record(row('standard'), catalog=catalog).cost_usd == Decimal('.0001345')


@pytest.mark.parametrize('publisher,channel,inclusive', [
    ('openai', 'openai-api', True),
    ('openai', 'openrouter-api', False),
    ('openai', 'aws-bedrock', False),
    ('anthropic', 'anthropic-api', False),
])
def test_write_default_is_limited_to_native_openai(publisher, channel, inclusive):
    from metergraph_core import ModelUsage
    catalog = load_catalog_document({
        'version': 'synthetic-inclusive-writes', 'currency': 'USD',
        'pricing_verified_at': '2026-11-22', 'models': [{
            'canonical_id': 'example/cache-model', 'publisher': publisher,
            'aliases': [{'provider': publisher, 'alias': 'cache-model', 'channel': channel}],
            'prices': [{
                'channel': channel, 'region': 'global', 'effective_from': '2026-01-01',
                'input_per_mtok': 1, 'output_per_mtok': 0,
                'cache_read_per_mtok': '.1', 'cache_write_5m_per_mtok': '1.25',
                'rules': {'input_includes_cache_read': True},
                'source_url': 'https://example.com/pricing',
            }],
        }],
    })
    result = catalog.calculate_cost(model='cache-model', provider=publisher, channel=channel,
        at=AT, usage=ModelUsage(input_tokens=1000, output_tokens=0,
                               cache_read_tokens=200, cache_write_tokens=100))
    assert result.status == 'priced'
    assert result.cost_usd == Decimal('.000845' if inclusive else '.000945')


def test_cache_reads_and_writes_cannot_exceed_inclusive_input():
    result = price_call_from_record({**row('standard'), 'cache_read_tokens': 950})
    assert result.status == 'partial'
    assert result.cost_usd is None
    assert 'cache_write_exceeds_input' in result.token_cost.reasons
    assert result.token_cost.components.input_usd == Decimal(0)


def test_explicit_exclusive_writes_still_count_toward_context_threshold():
    catalog = catalog_with_rules({
        'input_includes_cache_write': False,
        'long_context': {'threshold': 1050, 'threshold_counts_cached_tokens': True,
                         'input_multiplier': 2, 'output_multiplier': 1},
    })
    assert price_call_from_record(row('standard'), catalog=catalog).cost_usd == Decimal('.000239')


@pytest.mark.parametrize('tier', ['standard', 'flex', 'batch', 'priority'])
def test_native_response_and_normalized_record_share_inclusive_write_convention(tier):
    from metergraph_core import price_call_from_response
    native = price_call_from_response('openai', {
        'object': 'response', 'model': 'gpt-6-luna', 'service_tier': tier,
        'status': 'completed', 'output': [], 'usage': {
            'input_tokens': 1000, 'output_tokens': 100,
            'input_tokens_details': {'cached_tokens': 200, 'cache_write_tokens': 100},
        },
    }, at=AT)
    normalized = price_call_from_record(row(tier))
    assert native.status == normalized.status == 'priced'
    assert native.cost_usd == normalized.cost_usd
