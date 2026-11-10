from decimal import Decimal

import pytest

from metergraph_core import ModelUsage, load_catalog


CATALOG = load_catalog()
ROWS = [
    pytest.param(model['canonical_id'], price, id=model['canonical_id'] + ':' + price['effective_from'])
    for model in CATALOG.document['models']
    for price in model.get('prices', [])
    if price.get('channel') == 'anthropic-api'
    and price.get('batch_input_per_mtok') is not None
    and price.get('batch_output_per_mtok') is not None
]


@pytest.mark.parametrize('model,price', ROWS)
def test_each_anthropic_batch_row_discounts_its_own_cache_rates(model, price):
    # Historical cache discounts changed across model versions and dates.
    # Each effective-dated row supplies its own base; do not assume 0.1x input.
    for suffix in ('read', 'write_5m', 'write_1h'):
        standard = price.get('cache_' + suffix + '_per_mtok')
        if standard is not None:
            batch = price.get('batch_cache_' + suffix + '_per_mtok')
            assert batch is not None, 'published batch cache rate is missing'
            assert Decimal(str(batch)) == Decimal(str(standard)) / 2
    usage = ModelUsage(input_tokens=0, output_tokens=0, batch=True,
                       cache_read_tokens=1_000_000,
                       cache_write_5m_tokens=1_000_000,
                       cache_write_1h_tokens=1_000_000)
    result = CATALOG.calculate_cost(model=model, usage=usage, at=price['effective_from'],
                                    channel='anthropic-api', region=price.get('region', 'global'))
    assert result.status == 'priced'
    expected = sum((Decimal(str(price.get('cache_' + suffix + '_per_mtok') or 0))
                    for suffix in ('read', 'write_5m', 'write_1h')), Decimal(0)) / 2
    assert result.cost_usd == expected
