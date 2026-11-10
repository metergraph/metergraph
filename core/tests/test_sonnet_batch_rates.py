from decimal import Decimal

from metergraph_core import load_catalog


def test_sonnet_direct_batch_discounts_each_cache_bucket():
    catalog = load_catalog()
    usage = dict(provider='anthropic', model='claude-sonnet-4-6',
                 at='2026-11-01T12:00:00+00:00', input_tokens=0, output_tokens=0,
                 cache_read_tokens=1_000_000, cache_write_5m_tokens=1_000_000,
                 cache_write_1h_tokens=1_000_000)
    standard = catalog.snapshot.cost(**usage)
    batch = catalog.snapshot.cost(**usage, batch=True)
    assert standard.cost_usd == Decimal('10.05')
    assert batch.components.cache_read_usd == Decimal('0.15')
    assert batch.components.cache_write_usd == Decimal('4.875')
    assert batch.cost_usd == Decimal('5.025')
