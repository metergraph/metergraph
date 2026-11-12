from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal

from metergraph_server import usage


def test_usage_reliability_counts_preserve_usage_and_known_denominators(monkeypatch):
    # 8 calls: 3 available, 2 unavailable, 1 excluded, 2 unknown.
    row = ('example-model', 'example-provider', 8, Decimal('2.5'), 100, 50, 0, 0,
           250, 400, Decimal('.5'), 0, 0, 3, 2, 1, 2, 1, 2, 3, 2)

    class Connection:
        def execute(self, sql, params):
            assert "availability is null" in sql
            assert "refused is false" in sql
            return self

        def fetchall(self):
            return [row]

    class Pool:
        @contextmanager
        def connection(self):
            yield Connection()

    monkeypatch.setattr(usage.db, 'pool', lambda: Pool())
    items = usage.usage(group_by='model', from_='2026-01-01', to='2026-01-02',
                        environment=None, include_untagged=None, route=None, model=None)['items']
    item = items[0]
    assert item['calls'] == 8
    assert item['cost_usd'] == 2.5
    assert item['input_tokens'] == 100
    assert item['availability_denominator'] == 5
    assert item['availability_rate'] == .6
    assert item['refusal_denominator'] == 3
    assert item['refusal_rate'] == 1 / 3
    assert item['availability_coverage'] == item['refusal_coverage'] == 'partial'
    assert sum(item[key] for key in ('available_calls','unavailable_calls','availability_excluded_calls','availability_unknown_calls')) == 8
    assert sum(item[key] for key in ('refused_calls','not_refused_calls','refusal_not_assessable_calls','refusal_unknown_calls')) == 8


def test_legacy_usage_reliability_rates_are_unknown(monkeypatch):
    row = ('example-model', 2, Decimal('1'), 10, 5, 0, 0, 100, 100, Decimal('0'),
           0, 0, 0, 0, 0, 2, 0, 0, 0, 2)

    class Connection:
        def execute(self, sql, params):
            return self

        def fetchall(self):
            return [row]

    class Pool:
        @contextmanager
        def connection(self):
            yield Connection()

    monkeypatch.setattr(usage.db, 'pool', lambda: Pool())
    item = usage.usage(group_by='provider', from_='2026-01-01', to='2026-01-02',
                       environment=None, include_untagged=None, route=None, model=None)['items'][0]
    assert item['availability_rate'] is None
    assert item['refusal_rate'] is None
    assert item['availability_denominator'] == item['refusal_denominator'] == 0
    assert item['availability_coverage'] == item['refusal_coverage'] == 'partial'
    assert item['calls'] == 2
