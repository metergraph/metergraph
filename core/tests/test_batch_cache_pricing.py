"""Batch cache reads billed at an explicit batch cache rate.

A row may publish ``batch_cache_read_per_mtok``. A batch call with cached
tokens bills its cache reads at that rate; a standard call never does. A row
without it keeps billing batch cache reads at the standard cache-read rate, so
existing rows keep their totals and status.

The synthetic catalog pins each case against deterministic numbers so the
assertions never depend on the shipped price list.
"""

import textwrap
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal

import pytest
import yaml

from metergraph_core import (
    CatalogError,
    CatalogSnapshot,
    ModelUsage,
    cost_from_response,
    load_catalog_document,
)
from metergraph_core.catalog import Alias, Price

_DOCUMENT = yaml.safe_load(textwrap.dedent(
    """
    version: "test-batch-cache-1"
    currency: USD
    pricing_verified_at: "2026-08-01"
    models:
      - canonical_id: anthropic/cache-batch-x
        publisher: anthropic
        aliases:
          - {provider: anthropic, alias: cache-batch-x, channel: anthropic-api}
        prices:
          - channel: anthropic-api
            region: global
            effective_from: "2026-01-01"
            effective_to: "2026-12-01"
            input_per_mtok: 2.0
            output_per_mtok: 8.0
            cache_read_per_mtok: 0.10
            batch_input_per_mtok: 1.0
            batch_output_per_mtok: 4.0
            batch_cache_read_per_mtok: 0.05
            source_url: https://example.com/pricing
          - channel: anthropic-api
            region: global
            effective_from: "2026-12-01"
            input_per_mtok: 2.0
            output_per_mtok: 8.0
            cache_read_per_mtok: 0.20
            batch_input_per_mtok: 1.0
            batch_output_per_mtok: 4.0
            batch_cache_read_per_mtok: 0.10
            source_url: https://example.com/pricing
      - canonical_id: anthropic/legacy-cache-x
        publisher: anthropic
        aliases:
          - {provider: anthropic, alias: legacy-cache-x, channel: anthropic-api}
        prices:
          - channel: anthropic-api
            region: global
            effective_from: "2026-01-01"
            input_per_mtok: 2.0
            output_per_mtok: 8.0
            cache_read_per_mtok: 0.10
            batch_input_per_mtok: 1.0
            batch_output_per_mtok: 4.0
            source_url: https://example.com/pricing
      - canonical_id: anthropic/batch-only-cache-x
        publisher: anthropic
        aliases:
          - {provider: anthropic, alias: batch-only-cache-x, channel: anthropic-api}
        prices:
          - channel: anthropic-api
            region: global
            effective_from: "2026-01-01"
            input_per_mtok: 2.0
            output_per_mtok: 8.0
            batch_input_per_mtok: 1.0
            batch_output_per_mtok: 4.0
            batch_cache_read_per_mtok: 0.05
            source_url: https://example.com/pricing
      - canonical_id: openai/cache-batch-o
        publisher: openai
        aliases:
          - {provider: openai, alias: cache-batch-o, channel: openai-api}
        prices:
          - channel: openai-api
            region: global
            effective_from: "2026-01-01"
            input_per_mtok: 2.0
            output_per_mtok: 8.0
            cache_read_per_mtok: 0.10
            batch_input_per_mtok: 1.0
            batch_output_per_mtok: 4.0
            batch_cache_read_per_mtok: 0.05
            source_url: https://example.com/pricing
      - canonical_id: anthropic/regional-cache-x
        publisher: anthropic
        aliases:
          - {provider: anthropic, alias: regional-cache-x, channel: anthropic-api}
        prices:
          - channel: anthropic-api
            region: global
            effective_from: "2026-01-01"
            input_per_mtok: 2.0
            output_per_mtok: 8.0
            cache_read_per_mtok: 0.10
            batch_input_per_mtok: 1.0
            batch_output_per_mtok: 4.0
            batch_cache_read_per_mtok: 0.05
            source_url: https://example.com/pricing
          - channel: anthropic-api
            region: us-east-1
            effective_from: "2026-01-01"
            input_per_mtok: 2.0
            output_per_mtok: 8.0
            cache_read_per_mtok: 0.10
            batch_input_per_mtok: 1.0
            batch_output_per_mtok: 4.0
            batch_cache_read_per_mtok: 0.06
            source_url: https://example.com/pricing
      - canonical_id: anthropic/regional-same-x
        publisher: anthropic
        aliases:
          - {provider: anthropic, alias: regional-same-x, channel: anthropic-api}
        prices:
          - channel: anthropic-api
            region: global
            effective_from: "2026-01-01"
            input_per_mtok: 2.0
            output_per_mtok: 8.0
            cache_read_per_mtok: 0.10
            batch_input_per_mtok: 1.0
            batch_output_per_mtok: 4.0
            batch_cache_read_per_mtok: 0.05
            source_url: https://example.com/pricing
          - channel: anthropic-api
            region: us-east-1
            effective_from: "2026-01-01"
            input_per_mtok: 2.0
            output_per_mtok: 8.0
            cache_read_per_mtok: 0.10
            batch_input_per_mtok: 1.0
            batch_output_per_mtok: 4.0
            batch_cache_read_per_mtok: 0.05
            source_url: https://example.com/pricing
    """
))
_CATALOG = load_catalog_document(_DOCUMENT)

_JULY = datetime(2026, 7, 1, tzinfo=timezone.utc)
_DECEMBER = datetime(2026, 12, 1, tzinfo=timezone.utc)
# 1,000 input and output tokens, and one million cached tokens.
_USAGE = dict(input_tokens=1_000, output_tokens=1_000, cache_read_tokens=1_000_000)


def _price(model, *, batch, at=_JULY, **usage):
    return _CATALOG.price(
        model=model, channel="anthropic-api", at=at, batch=batch, **(usage or _USAGE)
    )


def test_cached_batch_call_uses_the_batch_cache_rate():
    result = _price("cache-batch-x", batch=True)

    assert result.status == "priced"
    assert result.reasons == ()
    # 1,000 in at $1, 1,000 out at $4, 1M cached at $0.05 per 1M.
    assert result.components.cache_read == Decimal("0.05")
    assert result.cost == Decimal("0.055")


def test_cached_standard_call_keeps_the_standard_cache_rate():
    result = _price("cache-batch-x", batch=False)

    assert result.status == "priced"
    # 1,000 in at $2, 1,000 out at $8, 1M cached at $0.10 per 1M.
    assert result.components.cache_read == Decimal("0.10")
    assert result.cost == Decimal("0.11")


@pytest.mark.parametrize(
    "at,batch_cache,standard_cache",
    [(_JULY, "0.05", "0.10"), (_DECEMBER, "0.10", "0.20")],
)
def test_batch_cache_rate_follows_its_effective_dated_row(at, batch_cache, standard_cache):
    batch = _price("cache-batch-x", batch=True, at=at)
    standard = _price("cache-batch-x", batch=False, at=at)

    assert batch.components.cache_read == Decimal(batch_cache)
    assert standard.components.cache_read == Decimal(standard_cache)
    assert batch.status == standard.status == "priced"


def test_row_without_a_batch_cache_rate_keeps_the_standard_rate_on_batch_calls():
    result = _price("legacy-cache-x", batch=True)

    assert result.status == "priced"
    assert result.reasons == ()
    assert result.price.batch_cache_read_per_mtok is None
    # Unchanged from before the field existed: 1M cached at the $0.10 rate.
    assert result.components.cache_read == Decimal("0.10")
    assert result.cost == Decimal("0.105")


def _direct_price(**fields):
    return Price(
        id="anthropic/direct:anthropic-api:global:2026-01-01",
        model_id="anthropic/direct",
        pricing_channel="anthropic-api",
        region="global",
        input_per_mtok=Decimal("2"),
        output_per_mtok=Decimal("8"),
        cache_read_per_mtok=Decimal("0.10"),
        cache_write_5m_per_mtok=None,
        cache_write_1h_per_mtok=None,
        batch_input_per_mtok=Decimal("1"),
        batch_output_per_mtok=Decimal("4"),
        rules={},
        effective_from=datetime(2026, 1, 1, tzinfo=timezone.utc),
        effective_to=None,
        source_url="https://example.com/pricing",
        **fields,
    )


_INVALID_DIRECT_RATES = [
    Decimal("-1"),
    Decimal("-0.0001"),
    Decimal("NaN"),
    Decimal("sNaN"),
    Decimal("Infinity"),
    Decimal("-Infinity"),
    -1,
    0.05,
    "0.05",
]


@pytest.mark.parametrize("value", _INVALID_DIRECT_RATES)
def test_direct_constructor_rejects_an_unusable_batch_cache_rate(value):
    with pytest.raises(ValueError, match="batch_cache_read_per_mtok"):
        _direct_price(batch_cache_read_per_mtok=value)


@pytest.mark.parametrize("value", _INVALID_DIRECT_RATES)
def test_replacing_a_price_rejects_an_unusable_batch_cache_rate(value):
    with pytest.raises(ValueError, match="batch_cache_read_per_mtok"):
        replace(_direct_price(), batch_cache_read_per_mtok=value)


def test_direct_constructor_accepts_a_zero_batch_cache_rate():
    price = _direct_price(batch_cache_read_per_mtok=Decimal("0"))
    alias = Alias("anthropic/direct", "anthropic/direct", "anthropic-api", {})
    snapshot = CatalogSnapshot({("anthropic", "direct"): alias}, [price], region="global")

    result = snapshot.cost(provider="anthropic", model="direct", at=_JULY, batch=True, **_USAGE)

    assert price.batch_cache_read_per_mtok == Decimal("0")
    assert replace(price, batch_cache_read_per_mtok=None).batch_cache_read_per_mtok is None
    assert result.status == "priced"
    assert result.components.cache_read == Decimal("0")
    # 1,000 in at $1 and 1,000 out at $4; the cached tokens bill nothing.
    assert result.cost == Decimal("0.005")


def test_legacy_constructor_without_the_field_keeps_the_standard_rate():
    price = Price(
        "anthropic/legacy:anthropic-api:global:2026-01-01",
        "anthropic/legacy",
        "anthropic-api",
        "global",
        Decimal("2"),
        Decimal("8"),
        Decimal("0.10"),
        None,
        None,
        Decimal("1"),
        Decimal("4"),
        {},
        datetime(2026, 1, 1, tzinfo=timezone.utc),
        None,
        "https://example.com/pricing",
    )
    alias = Alias("anthropic/legacy", "anthropic/legacy", "anthropic-api", {})
    snapshot = CatalogSnapshot({("anthropic", "legacy"): alias}, [price], region="global")

    result = snapshot.cost(provider="anthropic", model="legacy", at=_JULY, batch=True, **_USAGE)

    assert price.batch_cache_read_per_mtok is None
    assert result.status == "priced"
    assert result.cost_usd == Decimal("0.105")


@pytest.mark.parametrize("cached", [0, None])
@pytest.mark.parametrize("model", ["cache-batch-x", "legacy-cache-x", "batch-only-cache-x"])
@pytest.mark.parametrize("batch,cost", [(True, "0.005"), (False, "0.01")])
def test_no_cached_tokens_needs_no_cache_rate(model, cached, batch, cost):
    result = _price(
        model, batch=batch, input_tokens=1_000, output_tokens=1_000, cache_read_tokens=cached
    )

    assert result.status == "priced"
    assert result.reasons == ()
    assert result.components.cache_read == Decimal("0")
    assert result.cost == Decimal(cost)


def test_batch_only_cache_rate_fully_prices_a_cached_batch_call():
    result = _price("batch-only-cache-x", batch=True)

    assert result.price.cache_read_per_mtok is None
    assert result.status == "priced"
    assert result.reasons == ()
    assert result.cost == Decimal("0.055")


def test_batch_only_cache_rate_leaves_a_cached_standard_call_partial():
    result = _price("batch-only-cache-x", batch=False)

    assert result.status == "partial"
    assert result.reasons == ("cache_read_rate_unavailable",)
    # The known subtotal: 1,000 in at $2 and 1,000 out at $8.
    assert result.components.cache_read == Decimal("0")
    assert result.cost == Decimal("0.01")


def _anthropic_response(**usage):
    return {
        "type": "message",
        "model": "cache-batch-x",
        "content": [{"type": "text", "text": "synthetic"}],
        "usage": {
            "input_tokens": 1_000,
            "output_tokens": 1_000,
            "cache_read_input_tokens": 1_000_000,
            "cache_creation_input_tokens": 0,
            **usage,
        },
    }


def _openai_response():
    return {
        "object": "chat.completion",
        "model": "cache-batch-o",
        "choices": [{"message": {"role": "assistant", "content": "synthetic"}}],
        "usage": {
            # OpenAI counts cached tokens inside prompt_tokens.
            "prompt_tokens": 1_001_000,
            "completion_tokens": 1_000,
            "prompt_tokens_details": {"cached_tokens": 1_000_000},
        },
    }


@pytest.mark.parametrize("batch", [True, False])
def test_every_entry_point_agrees_on_batch_cache_cost(batch):
    usage = ModelUsage(**_USAGE, batch=batch)
    tier = "batch" if batch else "standard"
    results = [
        _CATALOG.price(model="cache-batch-x", channel="anthropic-api", at=_JULY, batch=batch, **_USAGE),
        _CATALOG.snapshot.cost(provider="anthropic", model="cache-batch-x", at=_JULY, batch=batch, **_USAGE),
        _CATALOG.calculate_cost(model="cache-batch-x", provider="anthropic", at=_JULY, usage=usage),
        cost_from_response(
            "anthropic", _anthropic_response(service_tier=tier), catalog=_CATALOG, at=_JULY
        ),
        cost_from_response(
            "openai", _openai_response(), catalog=_CATALOG, at=_JULY, service_tier=tier
        ),
    ]

    expected_cache = Decimal("0.05") if batch else Decimal("0.10")
    expected_cost = Decimal("0.055") if batch else Decimal("0.11")
    for result in results:
        assert result.status == "priced", result
        assert result.components.cache_read == expected_cache, result
        assert result.cost == expected_cost, result


def test_regions_that_differ_only_in_batch_cache_rate_are_not_interchangeable():
    found = _CATALOG.lookup_price(model="regional-cache-x", channel="anthropic-api", at=_JULY)
    cost = _CATALOG.calculate_cost(
        model="regional-cache-x",
        channel="anthropic-api",
        at=_JULY,
        usage=ModelUsage(**_USAGE, batch=True),
    )
    stated = _CATALOG.calculate_cost(
        model="regional-cache-x",
        channel="anthropic-api",
        at=_JULY,
        region="us-east-1",
        usage=ModelUsage(**_USAGE, batch=True),
    )

    assert found.status == "partial"
    assert found.missing_dimensions == ("region",)
    assert cost.status == "partial"
    assert cost.missing_dimensions == ("region",)
    assert stated.status == "priced"
    assert stated.components.cache_read == Decimal("0.06")


def test_regions_with_equal_batch_cache_rates_stay_interchangeable():
    found = _CATALOG.lookup_price(model="regional-same-x", channel="anthropic-api", at=_JULY)

    assert found.status == "priced"
    assert found.missing_dimensions == ()


def _with_batch_cache_rate(value):
    document = yaml.safe_load(yaml.safe_dump(_DOCUMENT))
    document["models"][1]["prices"][0]["batch_cache_read_per_mtok"] = value
    return document


@pytest.mark.parametrize("value", [-0.05, "-0.05", "not-a-rate", True, float("nan"), float("inf")])
def test_loader_rejects_an_unusable_batch_cache_rate(value):
    with pytest.raises(CatalogError, match="invalid batch_cache_read_per_mtok"):
        load_catalog_document(_with_batch_cache_rate(value))


@pytest.mark.parametrize("value", [0, "0", "0.025"])
def test_loader_accepts_a_non_negative_batch_cache_rate(value):
    catalog = load_catalog_document(_with_batch_cache_rate(value))
    result = catalog.price(
        model="legacy-cache-x", channel="anthropic-api", at=_JULY, batch=True, **_USAGE
    )

    assert result.price.batch_cache_read_per_mtok == Decimal(str(value))
    assert result.status == "priced"
    assert result.components.cache_read == Decimal(str(value))
