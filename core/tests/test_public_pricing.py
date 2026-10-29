"""Public channel-aware deployment pricing and identity helpers.

metergraph-core lets a caller price an observed deployment from the identity it
already has -- a model id, the pricing channel it was served on, the execution
time, and the token usage -- without rebuilding provider/alias indexes from the
catalog document. Channel selection is exact: a model priced on a channel the
catalog does not carry for it stays explicitly unpriced rather than being
repriced off another channel. Two thin identity helpers round out the surface:
canonical-id resolution (never guessing an ambiguous alias) and the direct
channel a recorded source provider bills on.

The synthetic catalog below pins the contract against deterministic numbers so
the assertions never depend on the shipped price list.
"""

import textwrap
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import pytest

import metergraph_core
from metergraph_core import (
    CatalogSnapshot,
    ModelUsage,
    PriceResult,
    direct_channel_for_provider,
    load_catalog,
    normalize_provider,
    parse_catalog,
)


_SYNTHETIC_CATALOG = textwrap.dedent(
    """
    version: "test-public-1"
    currency: USD
    pricing_verified_at: "2026-08-01"
    models:
      - canonical_id: openai/gpt-x
        aliases:
          - {provider: openai, alias: gpt-x, channel: openai-api}
          - {provider: openai, alias: openai/gpt-x, channel: vercel-ai-gateway}
          - {provider: openai, alias: shared-name, channel: openai-api}
        prices:
          - channel: openai-api
            region: global
            effective_from: "2026-01-01"
            effective_to: "2026-06-01"
            input_per_mtok: 2.0
            output_per_mtok: 4.0
            source_url: https://example.test/openai
          - channel: openai-api
            region: global
            effective_from: "2026-06-01"
            input_per_mtok: 1.0
            output_per_mtok: 2.0
            source_url: https://example.test/openai-new
          - channel: vercel-ai-gateway
            region: global
            effective_from: "2026-01-01"
            input_per_mtok: 3.0
            output_per_mtok: 6.0
            source_url: https://example.test/gateway
      - canonical_id: google/gem-x
        aliases:
          - {provider: google, alias: gem-x, channel: google-api}
        prices:
          - channel: google-api
            region: global
            effective_from: "2026-01-01"
            input_per_mtok: 0.5
            output_per_mtok: 1.5
            source_url: https://example.test/google
      - canonical_id: fireworks/fw-x
        aliases:
          - {provider: fireworks, alias: fw-x, channel: fireworks-api}
          - {provider: fireworks, alias: shared-name, channel: fireworks-api}
        prices:
          - channel: fireworks-api
            region: global
            effective_from: "2026-01-01"
            input_per_mtok: 0.9
            output_per_mtok: 0.9
            source_url: https://example.test/fireworks
      - canonical_id: google/gem-old
        aliases:
          - {provider: google, alias: gem-old, channel: google-api}
        prices:
          - channel: google-api
            region: global
            effective_from: "2026-01-01"
            effective_to: "2026-04-01"
            input_per_mtok: 0.8
            output_per_mtok: 2.4
            source_url: https://example.test/google
      - canonical_id: google/gem-gap
        aliases:
          - {provider: google, alias: gem-gap, channel: google-api}
        prices:
          - channel: google-api
            region: global
            effective_from: "2026-01-01"
            effective_to: "2026-04-01"
            input_per_mtok: 0.8
            output_per_mtok: 2.4
            source_url: https://example.test/google
          - channel: google-api
            region: global
            effective_from: "2026-09-01"
            input_per_mtok: 1.6
            output_per_mtok: 4.8
            source_url: https://example.test/google
      - canonical_id: google/gem-reg
        aliases:
          - {provider: google, alias: gem-reg, channel: google-api}
        prices:
          - channel: google-api
            region: us-east-1
            effective_from: "2026-01-01"
            effective_to: "2026-04-01"
            input_per_mtok: 0.9
            output_per_mtok: 2.7
            source_url: https://example.test/google
          - channel: google-api
            region: global
            effective_from: "2026-01-01"
            effective_to: "2026-08-01"
            input_per_mtok: 0.8
            output_per_mtok: 2.4
            source_url: https://example.test/google
      - canonical_id: anthropic/claude-x
        aliases:
          - {provider: bedrock, alias: claude-x-bedrock, channel: aws-bedrock}
        prices:
          - channel: aws-bedrock
            region: global
            effective_from: "2026-01-01"
            input_per_mtok: 3.0
            output_per_mtok: 15.0
            source_url: https://example.test/bedrock
    """
)

_AT = datetime(2026, 3, 1, tzinfo=timezone.utc)


def _catalog(tmp_path):
    path = tmp_path / "prices.yaml"
    path.write_text(_SYNTHETIC_CATALOG)
    return load_catalog(path)


def test_price_direct_openai_channel(tmp_path):
    catalog = _catalog(tmp_path)
    priced = catalog.price(
        model="gpt-x", channel="openai-api", at=_AT,
        input_tokens=1000, output_tokens=500,
    )
    assert priced.status == "priced"
    assert priced.canonical_model == "openai/gpt-x"
    assert priced.price_id == "openai/gpt-x:openai-api:global:2026-01-01"
    # (1000 * 2.0 + 500 * 4.0) / 1e6, computed by core as Decimal.
    assert priced.cost_usd == Decimal("0.00400000")


def test_price_direct_google_channel(tmp_path):
    catalog = _catalog(tmp_path)
    priced = catalog.price(
        model="gem-x", channel="google-api", at=_AT,
        input_tokens=1000, output_tokens=1000,
    )
    assert priced.status == "priced"
    assert priced.canonical_model == "google/gem-x"
    # (1000 * 0.5 + 1000 * 1.5) / 1e6
    assert priced.cost_usd == Decimal("0.00200000")


def test_price_direct_fireworks_channel(tmp_path):
    catalog = _catalog(tmp_path)
    priced = catalog.price(
        model="fw-x", channel="fireworks-api", at=_AT,
        input_tokens=1000, output_tokens=1000,
    )
    assert priced.status == "priced"
    assert priced.canonical_model == "fireworks/fw-x"
    # (1000 * 0.9 + 1000 * 0.9) / 1e6
    assert priced.cost_usd == Decimal("0.00180000")


def test_price_channel_is_exact_never_a_fallback(tmp_path):
    catalog = _catalog(tmp_path)
    # gpt-x is priced on openai-api and (as openai/gpt-x) on the gateway, but
    # never on anthropic-api: it stays unpriced, not repriced off another
    # channel.
    priced = catalog.price(
        model="gpt-x", channel="anthropic-api", at=_AT,
        input_tokens=1000, output_tokens=500,
    )
    assert priced.status == "unpriced"
    assert priced.cost_usd is None
    assert priced.price_id is None
    assert priced.reasons


def test_price_unknown_model_is_explicitly_unpriced_never_zero(tmp_path):
    catalog = _catalog(tmp_path)
    priced = catalog.price(
        model="not-a-model", channel="openai-api", at=_AT,
        input_tokens=1000, output_tokens=500,
    )
    assert priced.status == "unpriced"
    assert priced.cost_usd is None
    assert priced.reasons


def test_price_selects_the_effective_dated_window(tmp_path):
    catalog = _catalog(tmp_path)
    later = datetime(2026, 7, 1, tzinfo=timezone.utc)
    priced = catalog.price(
        model="gpt-x", channel="openai-api", at=later,
        input_tokens=1000, output_tokens=500,
    )
    assert priced.status == "priced"
    # The open second window (1.0/2.0) wins over the closed first (2.0/4.0).
    assert priced.price_id == "openai/gpt-x:openai-api:global:2026-06-01"
    assert priced.cost_usd == Decimal("0.00200000")


def test_price_accepts_bare_date_and_iso_string_effective_times(tmp_path):
    catalog = _catalog(tmp_path)
    from_date = catalog.price(
        model="gpt-x", channel="openai-api", at=date(2026, 3, 1),
        input_tokens=1000, output_tokens=500,
    )
    from_iso = catalog.price(
        model="gpt-x", channel="openai-api", at="2026-03-01T00:00:00",
        input_tokens=1000, output_tokens=500,
    )
    assert from_date.price_id == "openai/gpt-x:openai-api:global:2026-01-01"
    assert from_iso.price_id == from_date.price_id
    assert from_iso.cost_usd == from_date.cost_usd == Decimal("0.00400000")


def test_price_missing_token_count_is_partial_not_zero(tmp_path):
    catalog = _catalog(tmp_path)
    priced = catalog.price(
        model="gpt-x", channel="openai-api", at=_AT,
        input_tokens=None, output_tokens=500,
    )
    assert priced.status == "partial"
    assert "missing_input_tokens" in priced.reasons
    # Output still priced by core; input contributes nothing knowable.
    assert priced.cost_usd == Decimal("0.00200000")


# Direct lookup and usage cost: `lookup_price` and `calculate_cost` resolve the
# same way `price` does and share its arithmetic, but report the resolved
# catalog id, the applied price, its currency, and which inputs were missing.

_REGIONAL_CATALOG = textwrap.dedent(
    """
    version: "test-public-regional-1"
    currency: USD
    pricing_verified_at: "2026-08-01"
    models:
      - canonical_id: acme/varies-x
        aliases:
          - {provider: openai, alias: varies-x, channel: openai-api}
        prices:
          - channel: openai-api
            region: global
            effective_from: "2026-01-01"
            input_per_mtok: 1.0
            output_per_mtok: 2.0
            source_url: https://example.test/varies-global
          - channel: openai-api
            region: us-east-1
            effective_from: "2026-01-01"
            input_per_mtok: 1.1
            output_per_mtok: 2.2
            source_url: https://example.test/varies-us-east-1
      - canonical_id: acme/uniform-x
        aliases:
          - {provider: openai, alias: uniform-x, channel: openai-api}
        prices:
          - channel: openai-api
            region: global
            effective_from: "2026-01-01"
            input_per_mtok: 1.0
            output_per_mtok: 2.0
            source_url: https://example.test/uniform-global
          - channel: openai-api
            region: us-east-1
            effective_from: "2026-01-01"
            input_per_mtok: 1.0
            output_per_mtok: 2.0
            source_url: https://example.test/uniform-us-east-1
      - canonical_id: acme/pinned-x
        aliases:
          - provider: openai
            alias: pinned-x
            channel: openai-api
            price_region: us-east-1
        prices:
          - channel: openai-api
            region: us-east-1
            effective_from: "2026-01-01"
            input_per_mtok: 4.0
            output_per_mtok: 8.0
            source_url: https://example.test/pinned-us-east-1
    """
)


def _regional_catalog(tmp_path, **kwargs):
    path = tmp_path / "regional-prices.yaml"
    path.write_text(_REGIONAL_CATALOG)
    return load_catalog(path, **kwargs)


def test_lookup_price_returns_the_resolved_catalog_id_and_applied_price(tmp_path):
    catalog = _catalog(tmp_path)
    found = catalog.lookup_price(model="gpt-x", channel="openai-api", at=_AT)
    assert isinstance(found, PriceResult)
    assert found.status == "priced"
    # The stable catalog id, never the alias the caller sent.
    assert found.model_id == "openai/gpt-x"
    assert found.price.id == "openai/gpt-x:openai-api:global:2026-01-01"
    assert found.price.input_per_mtok == Decimal("2.0")
    assert found.effective_at == _AT
    assert found.missing_dimensions == ()


def test_lookup_price_keeps_effective_dates_and_bedrock_fallback(tmp_path):
    catalog = _catalog(tmp_path)
    later = catalog.lookup_price(
        model="gpt-x", channel="openai-api", at=date(2026, 7, 1)
    )
    assert later.price.id == "openai/gpt-x:openai-api:global:2026-06-01"
    assert later.effective_at == datetime(2026, 7, 1, tzinfo=timezone.utc)
    versioned = catalog.lookup_price(
        model="claude-x-bedrock-v1:0", channel="aws-bedrock", at=_AT
    )
    assert versioned.status == "priced"
    assert versioned.model_id == "anthropic/claude-x"


def test_lookup_price_carries_the_price_currency(tmp_path):
    found = _catalog(tmp_path).lookup_price(
        model="gpt-x", channel="openai-api", at=_AT
    )
    assert found.price.currency == "USD"


def test_lookup_price_derives_the_direct_channel_from_provider(tmp_path):
    catalog = _catalog(tmp_path)
    direct = catalog.lookup_price(model="openai/gpt-x", provider="openai", at=_AT)
    assert direct.status == "priced"
    assert direct.price.pricing_channel == "openai-api"
    # Provider spellings fold through the same alias map as everywhere else.
    google = catalog.lookup_price(model="gem-x", provider="gemini", at=_AT)
    assert google.price.pricing_channel == "google-api"


def test_lookup_price_explicit_channel_beats_the_provider_channel(tmp_path):
    found = _catalog(tmp_path).lookup_price(
        model="openai/gpt-x", provider="openai", channel="vercel-ai-gateway", at=_AT
    )
    assert found.price.pricing_channel == "vercel-ai-gateway"
    assert found.price.input_per_mtok == Decimal("3.0")


def test_lookup_price_without_a_channel_is_unpriced_and_says_so(tmp_path):
    catalog = _catalog(tmp_path)
    for found in (
        catalog.lookup_price(model="gpt-x", at=_AT),
        catalog.lookup_price(model="gpt-x", provider="no-such-provider", at=_AT),
    ):
        assert found.status == "unpriced"
        assert found.price is None
        assert found.missing_dimensions == ("channel",)


def test_lookup_price_unknown_model_is_unpriced(tmp_path):
    found = _catalog(tmp_path).lookup_price(
        model="not-a-model", channel="openai-api", at=_AT
    )
    assert found.status == "unpriced"
    assert found.model_id is None
    assert found.price is None
    assert found.missing_dimensions == ()


def test_lookup_price_explicit_region_resolves_that_region(tmp_path):
    catalog = _regional_catalog(tmp_path)
    found = catalog.lookup_price(
        model="varies-x", channel="openai-api", at=_AT, region="us-east-1"
    )
    assert found.status == "priced"
    assert found.price.region == "us-east-1"
    assert found.missing_dimensions == ()
    pinned = catalog.lookup_price(
        model="pinned-x", channel="openai-api", at=_AT, region="us-east-1"
    )
    assert pinned.status == "priced"
    assert pinned.model_id == "acme/pinned-x"
    # A region the model is not sold in is simply unpriced: nothing is missing.
    unserved = catalog.lookup_price(
        model="pinned-x", channel="openai-api", at=_AT, region="eu-west-1"
    )
    assert unserved.status == "unpriced"
    assert unserved.missing_dimensions == ()


def test_lookup_price_query_region_overrides_the_load_default(tmp_path):
    catalog = _regional_catalog(tmp_path, region="us-east-1")
    # Omitted, the load-time region still applies as it always has.
    default = catalog.lookup_price(model="varies-x", channel="openai-api", at=_AT)
    assert default.status == "priced"
    assert default.price.region == "us-east-1"
    stated = catalog.lookup_price(
        model="varies-x", channel="openai-api", at=_AT, region="global"
    )
    assert stated.status == "priced"
    assert stated.price.region == "global"


def test_lookup_price_omitted_region_with_differing_regional_prices_is_partial(
    tmp_path,
):
    found = _regional_catalog(tmp_path).lookup_price(
        model="varies-x", channel="openai-api", at=_AT
    )
    assert found.status == "partial"
    assert found.model_id == "acme/varies-x"
    assert found.price.region == "global"
    assert found.missing_dimensions == ("region",)


def test_lookup_price_omitted_region_with_equal_regional_prices_is_priced(tmp_path):
    found = _regional_catalog(tmp_path).lookup_price(
        model="uniform-x", channel="openai-api", at=_AT
    )
    assert found.status == "priced"
    assert found.price.region == "global"
    assert found.missing_dimensions == ()


def test_lookup_price_omitted_region_without_a_default_is_unpriced(tmp_path):
    found = _regional_catalog(tmp_path).lookup_price(
        model="pinned-x", channel="openai-api", at=_AT
    )
    assert found.status == "unpriced"
    assert found.price is None
    assert found.missing_dimensions == ("region",)


def test_model_usage_defaults_to_unknown_counts_and_no_batch():
    usage = ModelUsage()
    assert (
        usage.input_tokens,
        usage.output_tokens,
        usage.cache_read_tokens,
        usage.cache_write_tokens,
        usage.cache_write_5m_tokens,
        usage.cache_write_1h_tokens,
        usage.batch,
    ) == (None, None, None, None, None, None, False)


def test_calculate_cost_matches_the_existing_price_entry_point(tmp_path):
    catalog = _catalog(tmp_path)
    result = catalog.calculate_cost(
        model="gpt-x",
        channel="openai-api",
        at=_AT,
        usage=ModelUsage(input_tokens=1000, output_tokens=500),
    )
    existing = catalog.price(
        model="gpt-x", channel="openai-api", at=_AT,
        input_tokens=1000, output_tokens=500,
    )
    assert result.status == existing.status == "priced"
    assert result.model_id == result.canonical_model == "openai/gpt-x"
    assert result.cost == result.cost_usd == existing.cost_usd == Decimal("0.00400000")
    assert result.currency == "USD"
    assert result.price == catalog.lookup_price(
        model="gpt-x", channel="openai-api", at=_AT
    ).price
    assert result.price_id == result.price.id
    assert result.effective_at == _AT
    assert result.missing_dimensions == ()
    components = result.components
    assert (
        components.input,
        components.output,
        components.cache_read,
        components.cache_write,
        components.request,
    ) == (
        components.input_usd,
        components.output_usd,
        components.cache_read_usd,
        components.cache_write_usd,
        components.request_usd,
    )
    assert components == existing.components


def test_calculate_cost_reports_the_price_currency(tmp_path):
    catalog = _catalog(tmp_path)
    _, aliases, prices = parse_catalog(catalog.document)
    euro = replace(
        catalog,
        snapshot=CatalogSnapshot(
            aliases,
            [replace(price, currency="EUR") for price in prices],
            region="global",
        ),
    )
    result = euro.calculate_cost(
        model="gpt-x",
        channel="openai-api",
        at=_AT,
        usage=ModelUsage(input_tokens=1000, output_tokens=500),
    )
    assert result.currency == "EUR"
    assert result.price.currency == "EUR"
    assert result.cost == Decimal("0.00400000")


def test_calculate_cost_carries_a_missing_region_into_the_cost(tmp_path):
    catalog = _regional_catalog(tmp_path)
    usage = ModelUsage(input_tokens=1000, output_tokens=500)
    global_default = catalog.calculate_cost(
        model="varies-x", channel="openai-api", at=_AT, usage=usage
    )
    assert global_default.status == "partial"
    assert global_default.missing_dimensions == ("region",)
    assert global_default.price.region == "global"
    # (1000 * 1.0 + 500 * 2.0) / 1e6 at the global rate.
    assert global_default.cost == Decimal("0.00200000")
    no_default = catalog.calculate_cost(
        model="pinned-x", channel="openai-api", at=_AT, usage=usage
    )
    assert no_default.status == "unpriced"
    assert no_default.cost is None
    assert no_default.missing_dimensions == ("region",)


def test_calculate_cost_keeps_the_known_subtotal_when_usage_is_incomplete(tmp_path):
    catalog = _catalog(tmp_path)
    output_only = catalog.calculate_cost(
        model="gpt-x", channel="openai-api", at=_AT,
        usage=ModelUsage(output_tokens=500),
    )
    assert output_only.status == "partial"
    # 500 * 4.0 / 1e6: the output subtotal survives the missing input.
    assert output_only.cost == Decimal("0.00200000")
    assert output_only.missing_dimensions == ("input_tokens",)
    nothing = catalog.calculate_cost(
        model="gpt-x", channel="openai-api", at=_AT, usage=ModelUsage()
    )
    assert nothing.status == "partial"
    assert nothing.cost == Decimal("0E-8")
    assert nothing.missing_dimensions == ("input_tokens", "output_tokens")
    regional = _regional_catalog(tmp_path).calculate_cost(
        model="varies-x", channel="openai-api", at=_AT,
        usage=ModelUsage(input_tokens=1000),
    )
    assert regional.missing_dimensions == ("region", "output_tokens")


def test_calculate_cost_unknown_model_is_unpriced(tmp_path):
    result = _catalog(tmp_path).calculate_cost(
        model="not-a-model", channel="openai-api", at=_AT,
        usage=ModelUsage(input_tokens=1000, output_tokens=500),
    )
    assert result.status == "unpriced"
    assert result.model_id is None
    assert result.cost is None
    assert result.price is None
    assert result.reasons


def test_pricing_interface_keeps_the_existing_entry_points(tmp_path):
    catalog = _catalog(tmp_path)
    via_provider = catalog.snapshot.cost(
        provider="openai", model="gpt-x", at=_AT,
        input_tokens=1000, output_tokens=500,
    )
    assert via_provider.canonical_model == "openai/gpt-x"
    assert via_provider.cost_usd == Decimal("0.00400000")
    assert via_provider.components.input_usd == Decimal("0.00200000")
    assert via_provider.components.output_usd == Decimal("0.00200000")
    resolved = catalog.snapshot.resolve_price(
        model="gpt-x", channel="openai-api", at=_AT
    )
    assert resolved.canonical_model == "openai/gpt-x"
    assert resolved.price.id == "openai/gpt-x:openai-api:global:2026-01-01"
    regional = _regional_catalog(tmp_path, region="us-east-1").price(
        model="varies-x", channel="openai-api", at=_AT,
        input_tokens=1000, output_tokens=500,
    )
    assert regional.status == "priced"
    assert regional.price_id == "acme/varies-x:openai-api:us-east-1:2026-01-01"


def test_pricing_interface_types_are_exported():
    assert {"ModelUsage", "PriceResult"} <= set(metergraph_core.__all__)


def test_canonical_model_id_resolves_provider_qualified_aliases(tmp_path):
    catalog = _catalog(tmp_path)
    assert catalog.canonical_model_id("openai", "gpt-x") == "openai/gpt-x"
    assert catalog.canonical_model_id("openai", "openai/gpt-x") == "openai/gpt-x"
    assert catalog.canonical_model_id("google", "gem-x") == "google/gem-x"
    assert catalog.canonical_model_id("fireworks", "fw-x") == "fireworks/fw-x"
    # An unknown model under a known provider stays raw.
    assert catalog.canonical_model_id("openai", "nope") == "nope"


def test_canonical_model_id_uses_provider_to_disambiguate_shared_names(tmp_path):
    catalog = _catalog(tmp_path)
    # "shared-name" is declared under both openai/gpt-x and fireworks/fw-x; the
    # captured provider selects the right canonical -- no guessing needed.
    assert catalog.canonical_model_id("openai", "shared-name") == "openai/gpt-x"
    assert catalog.canonical_model_id("fireworks", "shared-name") == "fireworks/fw-x"
    # An unknown provider cannot be disambiguated -> id returned unchanged.
    assert catalog.canonical_model_id("mystery", "shared-name") == "shared-name"
    # A non-string provider likewise leaves the id unchanged, never guessed.
    assert catalog.canonical_model_id(None, "shared-name") == "shared-name"


def test_canonical_model_id_normalizes_provider_aliases(tmp_path):
    catalog = _catalog(tmp_path)
    # google-genai / gemini normalize to google before the (provider, name) lookup.
    assert catalog.canonical_model_id("google-genai", "gem-x") == "google/gem-x"
    assert catalog.canonical_model_id("gemini", "gem-x") == "google/gem-x"
    # aws / amazon-bedrock normalize to bedrock.
    assert catalog.canonical_model_id("aws", "claude-x-bedrock") == "anthropic/claude-x"
    assert (
        catalog.canonical_model_id("amazon-bedrock", "claude-x-bedrock")
        == "anthropic/claude-x"
    )


def test_direct_channel_for_provider_normalizes_provider_aliases():
    # metergraph-core's provider-alias map applies before the channel lookup, so
    # names core pricing already accepts are not unknown to the direct helper.
    assert direct_channel_for_provider("aws") == "aws-bedrock"
    assert direct_channel_for_provider("amazon-bedrock") == "aws-bedrock"
    assert direct_channel_for_provider("aws-bedrock") == "aws-bedrock"
    assert direct_channel_for_provider("google-genai") == "google-api"
    assert direct_channel_for_provider("gemini") == "google-api"


def test_direct_channel_for_provider_maps_known_and_rejects_unknown():
    assert direct_channel_for_provider("openai") == "openai-api"
    assert direct_channel_for_provider("google") == "google-api"
    assert direct_channel_for_provider("fireworks") == "fireworks-api"
    assert direct_channel_for_provider("vertex-ai") == "google-vertex-ai"
    assert direct_channel_for_provider("bedrock") == "aws-bedrock"
    assert direct_channel_for_provider("anthropic") == "anthropic-api"
    assert direct_channel_for_provider("vercel") == "vercel-ai-gateway"
    assert direct_channel_for_provider("openrouter") == "openrouter-api"
    # A provider with no direct channel defined -- caller keeps it unpriced.
    assert direct_channel_for_provider("mystery") is None
    assert direct_channel_for_provider(None) is None
    assert direct_channel_for_provider(123) is None


def test_catalog_infers_a_unique_direct_channel_from_model_identity():
    catalog = load_catalog()

    assert catalog.infer_direct_channel("claude-opus-5") == "anthropic-api"
    assert catalog.infer_direct_channel("openai/gpt-5.6-luna") == "openai-api"
    assert catalog.infer_direct_channel("google/gemini-3.6-flash") == "google-api"


def test_catalog_does_not_infer_ambiguous_or_unknown_model_identity(tmp_path):
    catalog = _catalog(tmp_path)

    assert catalog.infer_direct_channel("shared-name") is None
    assert catalog.infer_direct_channel("mystery-model") is None


def test_installed_catalog_prices_openai_direct_end_to_end():
    # No path -> the metergraph-core bundled catalog. Structural assertions only
    # so shipped price numbers stay free to change.
    catalog = load_catalog()
    priced = catalog.price(
        model="gpt-5.6", channel="openai-api",
        at=datetime(2026, 8, 1, tzinfo=timezone.utc),
        input_tokens=1000, output_tokens=500,
    )
    assert priced.status in ("priced", "partial")
    assert priced.canonical_model == "openai/gpt-5.6-sol"
    assert priced.price_id
    assert priced.cost_usd is not None and priced.cost_usd > 0


def test_installed_catalog_prices_gemini_31_pro_on_direct_google_api():
    catalog = load_catalog()
    priced = catalog.price(
        model="gemini-3.1-pro-preview",
        channel="google-api",
        at=datetime(2026, 9, 1, tzinfo=timezone.utc),
        input_tokens=1000,
        output_tokens=500,
    )

    assert priced.status == "priced"
    assert priced.canonical_model == "google/gemini-3.1-pro-preview"
    assert priced.price_id == (
        "google/gemini-3.1-pro-preview:google-api:global:2026-02-19"
    )
    assert priced.cost_usd == Decimal("0.00800000")


def test_installed_catalog_prices_retired_gemini_3_pro_at_its_redirect_rate():
    # Google shut Gemini 3 Pro Preview down on 2026-03-09 and pointed the
    # `gemini-3-pro-preview` name at gemini-3.1-pro-preview, so traffic sent
    # under the old name after that date is real 3.1 Pro spend.
    catalog = load_catalog()
    before = catalog.price(
        model="gemini-3-pro-preview",
        channel="google-api",
        at=datetime(2026, 3, 8, tzinfo=timezone.utc),
        input_tokens=1000,
        output_tokens=500,
    )
    after = catalog.price(
        model="gemini-3-pro-preview",
        channel="google-api",
        at=datetime(2026, 9, 1, tzinfo=timezone.utc),
        input_tokens=1000,
        output_tokens=500,
    )

    assert before.price_id == "google/gemini-3-pro:google-api:global:2025-11-18"
    assert after.status == "priced"
    assert after.canonical_model == "google/gemini-3-pro"
    assert after.price_id == "google/gemini-3-pro:google-api:global:2026-03-09"
    assert after.cost_usd == Decimal("0.00800000")



def test_installed_catalog_prices_routed_gemini_35_flash_at_36_flash_rates():
    # Google routes gemini-3.5-flash to gemini-3.6-flash from 2026-10-08.
    catalog = load_catalog()
    for at in (datetime(2026, 10, 21, tzinfo=timezone.utc), datetime(2027, 2, 1, tzinfo=timezone.utc)):
        routed, target = (
            catalog.price(
                model=model, channel="google-api", at=at,
                input_tokens=1_000_000, output_tokens=1_000_000,
            )
            for model in ("gemini-3.5-flash", "gemini-3.6-flash")
        )
        assert routed.status == "priced"
        assert routed.canonical_model == "google/gemini-3.5-flash"
        assert routed.cost_usd == target.cost_usd


_AT_NANO = datetime(2026, 3, 17, tzinfo=timezone.utc)


@pytest.mark.parametrize("channel", ["openai-api", "vercel-ai-gateway"])
def test_installed_catalog_prices_gpt_5_4_nano_on_both_channels(channel):
    catalog = load_catalog()

    resolved = catalog.price(
        model="openai/gpt-5.4-nano", channel=channel, at=_AT_NANO,
        input_tokens=1_000_000, output_tokens=1_000_000,
    )
    assert resolved.status == "priced"
    assert resolved.canonical_model == "openai/gpt-5.4-nano"
    assert resolved.price_id == f"openai/gpt-5.4-nano:{channel}:global:2026-03-17"
    assert resolved.cost_usd == Decimal("1.45000000")

    output_only = catalog.price(
        model="openai/gpt-5.4-nano", channel=channel, at=_AT_NANO,
        input_tokens=0, output_tokens=1_000_000,
    )
    assert output_only.status == "priced"
    assert output_only.cost_usd == Decimal("1.25000000")

    cached = catalog.price(
        model="openai/gpt-5.4-nano", channel=channel, at=_AT_NANO,
        input_tokens=1_000_000, output_tokens=0, cache_read_tokens=1_000_000,
    )
    assert cached.status == "priced"
    assert cached.cost_usd == Decimal("0.02000000")


_AT_GEMINI_25_FLASH = datetime(2026, 8, 20, tzinfo=timezone.utc)


def test_installed_catalog_prices_gemini_2_5_flash_image_on_google_api():
    catalog = load_catalog()

    resolved = catalog.price(
        model="gemini-2.5-flash-image", channel="google-api",
        at=_AT_GEMINI_25_FLASH, input_tokens=1_000_000, output_tokens=1_000_000,
    )

    assert resolved.status == "priced"
    assert resolved.canonical_model == "google/gemini-2.5-flash-image"
    assert (
        resolved.price_id
        == "google/gemini-2.5-flash-image:google-api:global:2025-10-02"
    )
    assert resolved.cost_usd == Decimal("30.30000000")


def test_installed_catalog_ends_gemini_2_5_flash_image_at_its_shutdown():
    """Google shut the model down on 2026-10-02 and states its output price per
    image, so the token row stops there and later calls say the price ended."""
    catalog = load_catalog()
    usage = dict(
        model="gemini-2.5-flash-image", channel="google-api",
        input_tokens=1_000_000, output_tokens=1_000_000,
    )

    last_day = catalog.price(at=datetime(2026, 10, 1, tzinfo=timezone.utc), **usage)
    after = catalog.price(at=datetime(2026, 10, 2, tzinfo=timezone.utc), **usage)

    assert last_day.status == "priced"
    assert last_day.reasons == ()
    assert after.status == "priced"
    assert after.reasons == ("price_ended",)
    assert after.cost_usd == last_day.cost_usd


@pytest.mark.parametrize(
    "channel,cache_read_cost",
    # Google reports cached tokens inside promptTokenCount on both channels, so
    # cache reads come out of billable input and are billed once. Both channels
    # publish the same 0.03 cache-read rate.
    [("google-api", "0.03000000"), ("vercel-ai-gateway", "0.03000000")],
)
def test_installed_catalog_prices_gemini_2_5_flash_on_both_channels(channel, cache_read_cost):
    """The gateway resolves the provider-qualified id and prices its own window.

    The two channels agree on every published rate, but the gateway still needs
    its own priced window rather than a fallback to the direct channel.
    """
    catalog = load_catalog()

    resolved = catalog.price(
        model="google/gemini-2.5-flash", channel=channel, at=_AT_GEMINI_25_FLASH,
        input_tokens=1_000_000, output_tokens=1_000_000,
    )
    assert resolved.status == "priced"
    assert resolved.canonical_model == "google/gemini-2.5-flash"
    assert resolved.price_id == f"google/gemini-2.5-flash:{channel}:global:2025-06-17"
    assert resolved.cost_usd == Decimal("2.80000000")

    output_only = catalog.price(
        model="google/gemini-2.5-flash", channel=channel, at=_AT_GEMINI_25_FLASH,
        input_tokens=0, output_tokens=1_000_000,
    )
    assert output_only.status == "priced"
    assert output_only.cost_usd == Decimal("2.50000000")

    cached = catalog.price(
        model="google/gemini-2.5-flash", channel=channel, at=_AT_GEMINI_25_FLASH,
        input_tokens=1_000_000, output_tokens=0, cache_read_tokens=1_000_000,
    )
    assert cached.status == "priced"
    assert cached.cost_usd == Decimal(cache_read_cost)


_AT_OPUS_48 = datetime(2026, 8, 20, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "model,channel",
    # The provider-qualified id is the gateway's naming, the dashed id is
    # Anthropic's own -- matching every other Claude model in the catalog.
    [("claude-opus-4-8", "anthropic-api"),
     ("anthropic/claude-opus-4.8", "vercel-ai-gateway")],
)
def test_installed_catalog_prices_claude_opus_4_8_on_both_channels(model, channel):
    """The gateway carries the same rates as Anthropic direct, but needs its
    own window: without one, gateway-routed Opus 4.8 calls resolve unpriced."""
    catalog = load_catalog()

    resolved = catalog.price(
        model=model, channel=channel, at=_AT_OPUS_48,
        input_tokens=1_000_000, output_tokens=1_000_000,
    )
    assert resolved.status == "priced"
    assert resolved.canonical_model == "anthropic/claude-opus-4.8"
    assert resolved.price_id == f"anthropic/claude-opus-4.8:{channel}:global:2026-05-28"
    assert resolved.cost_usd == Decimal("30.00000000")

    output_only = catalog.price(
        model=model, channel=channel, at=_AT_OPUS_48,
        input_tokens=0, output_tokens=1_000_000,
    )
    assert output_only.status == "priced"
    assert output_only.cost_usd == Decimal("25.00000000")


_AT_OPUS_47 = datetime(2026, 9, 6, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "model,channel",
    # Every spelling production actually captured. The dashed id is Anthropic's
    # own, the dotted provider-qualified id is the gateway's, and the
    # region-prefixed ids come from Bedrock inference profiles. An alias is keyed
    # on (provider, alias) without the channel, so one spelling cannot appear on
    # two channels -- which is why the dotted id lives only on the gateway.
    [("claude-opus-4-7", "anthropic-api"),
     ("anthropic/claude-opus-4-7", "anthropic-api"),
     ("global.anthropic.claude-opus-4-7", "anthropic-api"),
     ("us.anthropic.claude-opus-4-7", "anthropic-api"),
     ("anthropic/claude-opus-4.7", "vercel-ai-gateway")],
)
def test_installed_catalog_prices_claude_opus_4_7_on_every_captured_alias(model, channel):
    """Opus 4.7 was absent from the catalog entirely, so an analysis whose
    reference baseline used it could not launch at all: the pricing preflight
    blocks a run rather than report a cost it cannot compute. Observed in
    production 2026-09-06, where a scheduled run failed with
    `pricing_preflight_failed` naming `anthropic/claude-opus-4-7`."""
    catalog = load_catalog()

    resolved = catalog.price(
        model=model, channel=channel, at=_AT_OPUS_47,
        input_tokens=1_000_000, output_tokens=1_000_000,
    )
    assert resolved.status == "priced"
    assert resolved.canonical_model == "anthropic/claude-opus-4.7"
    assert resolved.cost_usd == Decimal("30.00000000")

    output_only = catalog.price(
        model=model, channel=channel, at=_AT_OPUS_47,
        input_tokens=0, output_tokens=1_000_000,
    )
    assert output_only.status == "priced"
    assert output_only.cost_usd == Decimal("25.00000000")


_AT_HAIKU_3 = datetime(2026, 8, 20, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "model,channel",
    # The provider-qualified id is the gateway's naming, the dashed id is
    # Anthropic's own -- matching every other Claude model in the catalog.
    [("claude-3-haiku", "anthropic-api"),
     ("anthropic/claude-3-haiku", "vercel-ai-gateway")],
)
def test_installed_catalog_prices_claude_3_haiku_on_both_channels(model, channel):
    """Claude 3 Haiku carried gateway pricing only, so a deployment routing
    candidates directly to Anthropic could not reach the oldest Haiku at all:
    an unpriced candidate is dropped rather than compared."""
    catalog = load_catalog()

    resolved = catalog.price(
        model=model, channel=channel, at=_AT_HAIKU_3,
        input_tokens=1_000_000, output_tokens=1_000_000,
    )
    assert resolved.status == "priced"
    assert resolved.canonical_model == "anthropic/claude-3-haiku"
    assert resolved.cost_usd == Decimal("1.50000000")

    output_only = catalog.price(
        model=model, channel=channel, at=_AT_HAIKU_3,
        input_tokens=0, output_tokens=1_000_000,
    )
    assert output_only.status == "priced"
    assert output_only.cost_usd == Decimal("1.25000000")


_AT_SONNET_46 = datetime(2026, 9, 1, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "model,channel",
    # Sonnet 4.6 carried only the direct Anthropic price. A pipeline that
    # evaluates candidates over a shared gateway channel could not reference
    # this model at all: an unpriced reference blocks the whole run rather
    # than reporting a cost it cannot compute. The dashed id stays
    # Anthropic's own; the provider-qualified id now also resolves the
    # canonical id on both channels, matching every other Claude model.
    [("claude-sonnet-4-6", "anthropic-api"),
     ("anthropic/claude-sonnet-4.6", "anthropic-api"),
     ("anthropic/claude-sonnet-4.6", "vercel-ai-gateway")],
)
def test_installed_catalog_prices_claude_sonnet_4_6_on_both_channels(model, channel):
    """anthropic/claude-sonnet-4.6 had no vercel-ai-gateway price at all, so a
    reference-generation budget reservation against it failed closed on every
    attempt regardless of provider health -- observed against real production
    traffic (96 captured calls) that a gateway-only evaluation pipeline could
    never analyse. The existing direct-channel alias for the dotted id must
    keep resolving unchanged: it now does so through the canonical id, which
    every alias of a model registers for its own channel."""
    catalog = load_catalog()

    resolved = catalog.price(
        model=model, channel=channel, at=_AT_SONNET_46,
        input_tokens=1_000_000, output_tokens=1_000_000,
    )
    assert resolved.status == "priced"
    assert resolved.canonical_model == "anthropic/claude-sonnet-4.6"
    assert resolved.cost_usd == Decimal("18.00000000")

    output_only = catalog.price(
        model=model, channel=channel, at=_AT_SONNET_46,
        input_tokens=0, output_tokens=1_000_000,
    )
    assert output_only.status == "priced"
    assert output_only.cost_usd == Decimal("15.00000000")


_AT_GROK_43 = datetime(2026, 9, 1, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "model,channel",
    [("grok-4.3", "xai-api"), ("xai/grok-4.3", "aws-bedrock")],
)
def test_installed_catalog_prices_grok_4_3_on_both_channels(model, channel):
    """xai/grok-4.3 carried only the direct xAI price, so a reference or
    candidate routed to Bedrock (available there since 2026-06-15, served on
    the bedrock-mantle endpoint at Standard-tier rates matching xAI's own
    price) had no way to be priced on that channel.

    Token counts stay under xai-api's 200k long-context threshold: both
    channels bill the same Standard-tier rate below it, so one pair of
    expected costs covers both parametrized channels. Bedrock pricing is
    region-scoped (In-Region only, and GovCloud differs), so the catalog is
    loaded for us-west-2 -- exactly how a real caller resolves it (see
    execution_profile.bedrock_region)."""
    catalog = load_catalog(region="us-west-2")

    resolved = catalog.price(
        model=model, channel=channel, at=_AT_GROK_43,
        input_tokens=100_000, output_tokens=100_000,
    )
    assert resolved.status == "priced"
    assert resolved.canonical_model == "xai/grok-4.3"
    assert resolved.cost_usd == Decimal("0.37500000")

    output_only = catalog.price(
        model=model, channel=channel, at=_AT_GROK_43,
        input_tokens=0, output_tokens=100_000,
    )
    assert output_only.status == "priced"
    assert output_only.cost_usd == Decimal("0.25000000")


_AT_KIMI_25 = datetime(2026, 9, 1, tzinfo=timezone.utc)


def test_installed_catalog_prices_kimi_k2_5_on_the_gateway():
    """K2.5 was priced on aws-bedrock only, so gateway-routed calls resolved
    unpriced and the analysis pricing guard refused to generate against it."""
    catalog = load_catalog()

    resolved = catalog.price(
        model="moonshotai/kimi-k2.5", channel="vercel-ai-gateway", at=_AT_KIMI_25,
        input_tokens=1_000_000, output_tokens=1_000_000,
    )
    assert resolved.status == "priced"
    assert resolved.canonical_model == "moonshotai/kimi-k2.5"
    assert resolved.price_id == "moonshotai/kimi-k2.5:vercel-ai-gateway:global:2026-08-27"
    assert resolved.cost_usd == Decimal("3.60000000")

    output_only = catalog.price(
        model="moonshotai/kimi-k2.5", channel="vercel-ai-gateway", at=_AT_KIMI_25,
        input_tokens=0, output_tokens=1_000_000,
    )
    assert output_only.status == "priced"
    assert output_only.cost_usd == Decimal("3.00000000")


_AT_GROK_41 = datetime(2026, 9, 1, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "alias,canonical",
    # Vercel serves Grok under both namespaces and answers 200 on either, so
    # both must price -- otherwise a caller sending the id the gateway accepts
    # gets an unpriced call.
    [("spacexai/grok-4.1-fast-reasoning", "xai/grok-4.1-fast-reasoning"),
     ("xai/grok-4.1-fast-reasoning", "xai/grok-4.1-fast-reasoning"),
     ("spacexai/grok-4.1-fast-non-reasoning", "xai/grok-4.1-fast-non-reasoning"),
     ("xai/grok-4.1-fast-non-reasoning", "xai/grok-4.1-fast-non-reasoning"),
     ("spacexai/grok-4.6", "xai/grok-4.6"),
     ("xai/grok-4.6", "xai/grok-4.6")],
)
def test_installed_catalog_prices_grok_under_both_namespaces(alias, canonical):
    catalog = load_catalog()

    resolved = catalog.price(
        model=alias, channel="vercel-ai-gateway", at=_AT_GROK_41,
        input_tokens=1_000_000, output_tokens=1_000_000,
    )
    assert resolved.status == "priced"
    assert resolved.canonical_model == canonical


def test_installed_catalog_prices_grok_4_7_at_both_context_tiers():
    catalog = load_catalog()
    released = datetime(2026, 9, 21, tzinfo=timezone.utc)

    before = catalog.snapshot.cost(
        provider="xai",
        model="grok-4.7",
        at=released - timedelta(seconds=1),
        input_tokens=100_000,
        output_tokens=100_000,
    )
    short = catalog.snapshot.cost(
        provider="xai",
        model="grok-4.7",
        at=released,
        input_tokens=100_000,
        output_tokens=100_000,
    )
    long = catalog.snapshot.cost(
        provider="xai",
        model="grok-4.7",
        at=released,
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )

    assert before.status == "unpriced"
    assert short.status == "priced"
    assert short.canonical_model == "xai/grok-4.7"
    assert short.cost_usd == Decimal("0.80000000")
    assert long.status == "priced"
    assert long.cost_usd == Decimal("16.00000000")


def test_installed_catalog_prices_grok_4_7_cached_input():
    catalog = load_catalog()

    result = catalog.snapshot.cost(
        provider="xai",
        model="grok-4.7-latest",
        at=datetime(2026, 10, 2, tzinfo=timezone.utc),
        input_tokens=100_000,
        output_tokens=0,
        cache_read_tokens=100_000,
    )

    assert result.status == "priced"
    assert result.canonical_model == "xai/grok-4.7"
    assert result.cost_usd == Decimal("0.05000000")


def test_installed_catalog_prices_grok_4_7_on_the_us_regional_endpoint():
    catalog = load_catalog(region="us")

    result = catalog.snapshot.cost(
        provider="xai",
        model="grok-4.7",
        at=datetime(2026, 10, 2, tzinfo=timezone.utc),
        input_tokens=100_000,
        output_tokens=100_000,
    )
    cached = catalog.snapshot.cost(
        provider="xai",
        model="grok-4.7",
        at=datetime(2026, 10, 2, tzinfo=timezone.utc),
        input_tokens=100_000,
        output_tokens=0,
        cache_read_tokens=100_000,
    )

    assert result.status == "priced"
    assert result.cost_usd == Decimal("0.88000000")
    assert cached.status == "priced"
    assert cached.cost_usd == Decimal("0.05500000")


def test_installed_catalog_prices_grok_4_6_on_the_us_regional_endpoint():
    """xAI's US endpoint prices grok-4.6 at a 10% premium."""
    catalog = load_catalog(region="us")
    at = datetime(2026, 8, 12, tzinfo=timezone.utc)

    result = catalog.snapshot.cost(
        provider="xai", model="grok-4.6", at=at,
        input_tokens=100_000, output_tokens=100_000,
    )
    cached = catalog.snapshot.cost(
        provider="xai",
        model="grok-4.6",
        at=at,
        input_tokens=100_000,
        output_tokens=0,
        cache_read_tokens=100_000,
    )

    assert result.status == "priced"
    assert result.cost_usd == Decimal("0.88000000")
    assert cached.status == "priced"
    assert cached.cost_usd == Decimal("0.05500000")


@pytest.mark.parametrize(
    "model,released_at",
    [
        ("grok-4.5", "2026-07-08T00:00:00+00:00"),
        ("grok-4.6", "2026-08-12T00:00:00+00:00"),
    ],
)
def test_installed_catalog_starts_recent_grok_prices_on_release_day(
    model, released_at,
):
    catalog = load_catalog()
    release = datetime.fromisoformat(released_at)

    before = catalog.snapshot.cost(
        provider="xai",
        model=model,
        at=release - timedelta(seconds=1),
        input_tokens=100_000,
        output_tokens=100_000,
    )
    released = catalog.snapshot.cost(
        provider="xai",
        model=model,
        at=release,
        input_tokens=100_000,
        output_tokens=100_000,
    )

    assert before.status == "unpriced"
    assert released.status == "priced"


_AT_BEDROCK_REFRESH = datetime(2026, 10, 2, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "model,region",
    [("us.openai.gpt-6.1-sol", "global"), ("openai.gpt-6.1-sol", "us-east-1")],
)
def test_installed_catalog_prices_gpt_6_1_sol_on_bedrock_at_both_context_tiers(
    model, region,
):
    """Price both Bedrock profiles across the long-context boundary."""
    catalog = load_catalog(region=region)

    def cost(**usage):
        return catalog.snapshot.cost(
            provider="bedrock", model=model, at=_AT_BEDROCK_REFRESH, **usage
        )

    short = cost(
        input_tokens=100_000, output_tokens=10_000,
        cache_read_tokens=50_000, cache_write_tokens=20_000,
    )
    at_threshold = cost(input_tokens=272_000, output_tokens=0)
    over_threshold = cost(input_tokens=272_001, output_tokens=0)
    long = cost(
        input_tokens=300_000, output_tokens=10_000,
        cache_read_tokens=100_000, cache_write_tokens=100_000,
    )

    assert short.status == "priced"
    assert short.canonical_model == "openai/gpt-6.1-sol"
    assert short.cost_usd == Decimal("0.39050000")
    assert at_threshold.cost_usd == Decimal("0.59840000")
    assert over_threshold.cost_usd == Decimal("1.19680440")
    assert long.status == "priced"
    assert long.cost_usd == Decimal("2.05700000")


@pytest.mark.parametrize(
    "model,expected",
    [("us.xai.grok-4.7", "9.35000000"), ("global.xai.grok-4.7", "8.50000000")],
)
def test_installed_catalog_prices_grok_4_7_on_bedrock_profiles(model, expected):
    catalog = load_catalog()

    result = catalog.snapshot.cost(
        provider="bedrock", model=model, at=_AT_BEDROCK_REFRESH,
        input_tokens=1_000_000, output_tokens=1_000_000,
        cache_read_tokens=1_000_000,
    )

    assert result.status == "priced"
    assert result.canonical_model == "xai/grok-4.7"
    assert result.cost_usd == Decimal(expected)


@pytest.mark.parametrize(
    "model,expected",
    [
        ("us.openai.gpt-6-sol", "1.61700000"),
        ("global.openai.gpt-6-sol", "1.47000000"),
        ("us.openai.gpt-6-luna", "0.08085000"),
        ("global.openai.gpt-6-luna", "0.07350000"),
        ("us.openai.gpt-6-astra", "8.08500000"),
        ("global.openai.gpt-6-astra", "7.35000000"),
    ],
)
def test_installed_catalog_prices_gpt_6_family_on_bedrock_profiles(model, expected):
    catalog = load_catalog()

    result = catalog.snapshot.cost(
        provider="bedrock", model=model, at=_AT_BEDROCK_REFRESH,
        input_tokens=100_000, output_tokens=100_000,
        cache_read_tokens=100_000, cache_write_tokens=100_000,
    )

    assert result.status == "priced"
    assert result.cost_usd == Decimal(expected)


_BEDROCK_SNAPSHOT_REGIONS = ("global", "us-east-1", "us-west-2", "eu-west-1")


def _bedrock_refresh_cost(model, region):
    return load_catalog(region=region).snapshot.cost(
        provider="bedrock", model=model, at=_AT_BEDROCK_REFRESH,
        input_tokens=100_000, output_tokens=100_000,
    )


@pytest.mark.parametrize(
    "model,served_regions",
    [
        ("openai.gpt-6.1-sol", ("us-east-1",)),
        ("openai.gpt-6-sol", ("us-east-1",)),
        ("openai.gpt-6-luna", ("us-east-1",)),
        ("openai.gpt-6-astra", ("us-east-1", "us-west-2")),
    ],
)
@pytest.mark.parametrize("region", _BEDROCK_SNAPSHOT_REGIONS)
def test_installed_catalog_prices_bare_bedrock_gpt_6_ids_only_where_served(
    model, served_regions, region,
):
    """Bare IDs do not borrow another region's price."""
    result = _bedrock_refresh_cost(model, region)

    if region in served_regions:
        assert result.status == "priced"
        assert result.price_id.split(":")[1:3] == ["aws-bedrock", region]
    else:
        assert result.status == "unpriced"
        assert result.price_id is None
        assert result.reasons == ("no_effective_alias",)


@pytest.mark.parametrize(
    "model,channel",
    [
        ("global.openai.gpt-6-sol", "aws-bedrock"),
        ("global.openai.gpt-6-luna", "aws-bedrock"),
        ("global.openai.gpt-6-astra", "aws-bedrock"),
        ("us.openai.gpt-6-sol", "aws-bedrock-geo"),
        ("us.openai.gpt-6-luna", "aws-bedrock-geo"),
        ("us.openai.gpt-6-astra", "aws-bedrock-geo"),
        ("us.openai.gpt-6.1-sol", "aws-bedrock-geo"),
    ],
)
@pytest.mark.parametrize("region", _BEDROCK_SNAPSHOT_REGIONS)
def test_installed_catalog_prices_bedrock_gpt_6_profiles_in_every_snapshot(
    model, channel, region,
):
    """A profile id names its own price scope, whatever region is loaded."""
    result = _bedrock_refresh_cost(model, region)

    assert result.status == "priced"
    assert result.price_id.split(":")[1:3] == [channel, "global"]


def test_installed_catalog_prices_gpt_6_sol_bedrock_long_context():
    catalog = load_catalog()

    result = catalog.snapshot.cost(
        provider="bedrock", model="us.openai.gpt-6-sol", at=_AT_BEDROCK_REFRESH,
        input_tokens=300_000, output_tokens=100_000,
    )

    assert result.status == "priced"
    assert result.cost_usd == Decimal("2.97000000")


def test_installed_catalog_prices_grok_4_6_on_the_bedrock_global_profile():
    catalog = load_catalog()

    result = catalog.snapshot.cost(
        provider="bedrock", model="global.xai.grok-4.6", at=_AT_BEDROCK_REFRESH,
        input_tokens=1_000_000, output_tokens=1_000_000,
        cache_read_tokens=1_000_000,
    )

    assert result.status == "priced"
    assert result.canonical_model == "xai/grok-4.6"
    assert result.cost_usd == Decimal("8.50000000")


def test_installed_catalog_prices_captured_sonnet_4_5_identity_on_anthropic_api():
    catalog = load_catalog()

    assert (
        catalog.canonical_model_id("anthropic", "anthropic/claude-sonnet-4.5")
        == "anthropic/claude-sonnet-4.5"
    )

    priced = catalog.price(
        model="anthropic/claude-sonnet-4.5",
        channel="anthropic-api",
        at=datetime(2025, 10, 1, tzinfo=timezone.utc),
        input_tokens=1000,
        output_tokens=500,
        cache_read_tokens=200,
        cache_write_tokens=100,
    )
    assert priced.status == "priced"
    assert priced.canonical_model == "anthropic/claude-sonnet-4.5"
    assert (
        priced.price_id
        == "anthropic/claude-sonnet-4.5:anthropic-api:global:2025-09-29"
    )
    assert priced.cost_usd == Decimal("0.01093500")


_AT_GEMINI_3_FLASH = datetime(2026, 9, 6, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "model",
    ["gemini-3-flash-preview", "models/gemini-3-flash-preview"],
)
def test_installed_catalog_prices_gemini_3_flash_preview_on_the_direct_api(model):
    """Gemini 3 Flash Preview was absent from the catalog while carrying the
    second-largest share of unpriced production calls. A preview id is still a
    billed id."""
    catalog = load_catalog()

    resolved = catalog.price(
        model=model, channel="google-api", at=_AT_GEMINI_3_FLASH,
        input_tokens=1_000_000, output_tokens=1_000_000,
    )
    assert resolved.status == "priced"
    assert resolved.canonical_model == "google/gemini-3-flash-preview"
    assert resolved.cost_usd == Decimal("3.50000000")

    output_only = catalog.price(
        model=model, channel="google-api", at=_AT_GEMINI_3_FLASH,
        input_tokens=0, output_tokens=1_000_000,
    )
    assert output_only.status == "priced"
    assert output_only.cost_usd == Decimal("3.00000000")


@pytest.mark.parametrize(
    "model",
    # The retired `-preview` id resolves to the same entry. Google shut that id
    # down without publishing a preview-era rate, so giving it a separate price
    # would invent one.
    ["gemini-3.1-flash-lite",
     "models/gemini-3.1-flash-lite",
     "gemini-3.1-flash-lite-preview"],
)
def test_installed_catalog_prices_gemini_31_flash_lite_on_every_captured_alias(model):
    """Flash Lite carried the largest share of unpriced production calls."""
    catalog = load_catalog()

    resolved = catalog.price(
        model=model, channel="google-api", at=_AT_GEMINI_3_FLASH,
        input_tokens=1_000_000, output_tokens=1_000_000,
    )
    assert resolved.status == "priced"
    assert resolved.canonical_model == "google/gemini-3.1-flash-lite"
    assert resolved.cost_usd == Decimal("1.75000000")

    output_only = catalog.price(
        model=model, channel="google-api", at=_AT_GEMINI_3_FLASH,
        input_tokens=0, output_tokens=1_000_000,
    )
    assert output_only.status == "priced"
    assert output_only.cost_usd == Decimal("1.50000000")


@pytest.mark.parametrize(
    "model,channel",
    [("gemini-3.5-flash", "google-vertex-ai"),
     ("gemini-3.5-flash", "google-api"),
     ("models/gemini-3.5-flash", "google-api")],
)
def test_installed_catalog_prices_gemini_35_flash_on_both_channels(model, channel):
    """3.5 Flash was priced on Vertex only, so calls captured against the direct
    API resolved unpriced. Google lists the same rate on both."""
    catalog = load_catalog()

    resolved = catalog.price(
        model=model, channel=channel, at=_AT_GEMINI_3_FLASH,
        input_tokens=1_000_000, output_tokens=1_000_000,
    )
    assert resolved.status == "priced"
    assert resolved.canonical_model == "google/gemini-3.5-flash"
    assert resolved.cost_usd == Decimal("10.50000000")


_AT_GEMINI_35_FLASH_LITE = datetime(2026, 9, 6, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "model",
    ["gemini-3.5-flash-lite", "models/gemini-3.5-flash-lite"],
)
def test_installed_catalog_prices_gemini_35_flash_lite_on_the_direct_api(model):
    """Google direct deducts cache reads from billable input: promptTokenCount
    already counts them."""
    catalog = load_catalog()

    resolved = catalog.price(
        model=model, channel="google-api", at=_AT_GEMINI_35_FLASH_LITE,
        input_tokens=1_000_000, output_tokens=1_000_000,
    )
    assert resolved.status == "priced"
    assert resolved.canonical_model == "google/gemini-3.5-flash-lite"
    assert resolved.price_id == (
        "google/gemini-3.5-flash-lite:google-api:global:2026-07-21"
    )
    assert resolved.cost_usd == Decimal("2.80000000")

    cached = catalog.price(
        model=model, channel="google-api", at=_AT_GEMINI_35_FLASH_LITE,
        input_tokens=1_000_000, output_tokens=0, cache_read_tokens=1_000_000,
    )
    assert cached.status == "priced"
    assert cached.cost_usd == Decimal("0.03000000")


def test_installed_catalog_does_not_price_gemini_35_flash_lite_off_google_api():
    """Only the direct API rate is published; other channels stay unpriced."""
    catalog = load_catalog()

    resolved = catalog.price(
        model="gemini-3.5-flash-lite", channel="google-vertex-ai",
        at=_AT_GEMINI_35_FLASH_LITE,
        input_tokens=1_000_000, output_tokens=1_000_000,
    )

    assert resolved.status == "unpriced"
    assert resolved.cost_usd is None


_AT_GLM_53 = datetime(2026, 9, 6, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "model,channel,cost",
    # One canonical id per model, one price per channel. The two channels list
    # materially different rates, so a single blended price would be wrong on
    # both.
    [("zai/glm-5.3", "vercel-ai-gateway", "2.90000000"),
     ("accounts/fireworks/models/glm-5p3", "fireworks-api", "5.80000000"),
     ("fireworks:accounts/fireworks/models/glm-5p3", "fireworks-api", "5.80000000")],
)
def test_installed_catalog_prices_glm_53_on_both_channels(model, channel, cost):
    catalog = load_catalog()

    resolved = catalog.price(
        model=model, channel=channel, at=_AT_GLM_53,
        input_tokens=1_000_000, output_tokens=1_000_000,
    )
    assert resolved.status == "priced"
    assert resolved.canonical_model == "zai/glm-5.3"
    assert resolved.cost_usd == Decimal(cost)


@pytest.mark.parametrize(
    "model,channel,cost",
    [("zai/glm-5.3-flash", "vercel-ai-gateway", "0.30875000"),
     ("accounts/fireworks/models/glm-5p3-flash", "fireworks-api", "0.65000000"),
     ("fireworks:accounts/fireworks/models/glm-5p3-flash", "fireworks-api", "0.65000000")],
)
def test_installed_catalog_prices_glm_53_flash_on_both_channels(model, channel, cost):
    """The gateway lists this at well under half the Fireworks rate. Recorded as
    listed rather than reconciled, so the spread is visible instead of averaged
    away."""
    catalog = load_catalog()

    resolved = catalog.price(
        model=model, channel=channel, at=_AT_GLM_53,
        input_tokens=1_000_000, output_tokens=1_000_000,
    )
    assert resolved.status == "priced"
    assert resolved.canonical_model == "zai/glm-5.3-flash"
    assert resolved.cost_usd == Decimal(cost)


_AT_CATALOG_REFRESH = datetime(2026, 9, 11, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "provider,model,canonical,cost",
    [
        ("google", "gemini-flash-latest", "google/gemini-3.8-flash", "4.50000000"),
        ("unknown", "models/gemini-flash-latest", "google/gemini-3.8-flash", "4.50000000"),
        ("litellm", "google/gemini-flash-latest", "google/gemini-3.8-flash", "4.50000000"),
        ("google", "google/models/gemini-flash-latest", "google/gemini-3.8-flash", "4.50000000"),
        ("google", "gemini-flash-lite-latest", "google/gemini-3.5-flash-lite", "2.80000000"),
        ("unknown", "models/gemini-flash-lite-latest", "google/gemini-3.5-flash-lite", "2.80000000"),
        ("litellm", "google/gemini-flash-lite-latest", "google/gemini-3.5-flash-lite", "2.80000000"),
        ("google", "google/models/gemini-flash-lite-latest", "google/gemini-3.5-flash-lite", "2.80000000"),
    ],
)
def test_installed_catalog_prices_google_latest_aliases(provider, model, canonical, cost):
    catalog = load_catalog()

    result = catalog.snapshot.cost(
        provider=provider,
        model=model,
        at=_AT_CATALOG_REFRESH,
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )

    assert result.status == "priced"
    assert result.canonical_model == canonical
    assert result.cost_usd == Decimal(cost)


def test_installed_catalog_ends_claude_35_haiku_without_contradictory_evidence():
    catalog = load_catalog()

    before = catalog.snapshot.cost(
        provider="anthropic",
        model="claude-3-5-haiku-latest",
        at=datetime(2026, 2, 18, 23, 59, tzinfo=timezone.utc),
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )
    alias_after_retirement = catalog.snapshot.cost(
        provider="anthropic",
        model="claude-3-5-haiku-latest",
        at=datetime(2026, 2, 19, tzinfo=timezone.utc),
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )
    exact_after_retirement = catalog.snapshot.cost(
        provider="anthropic",
        model="claude-3-5-haiku-20241022",
        at=datetime(2026, 2, 19, tzinfo=timezone.utc),
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )

    assert before.status == "priced"
    assert before.cost_usd == Decimal("4.80000000")
    # The retired -latest alias names nothing after its window.
    assert alias_after_retirement.status == "unpriced"
    assert alias_after_retirement.canonical_model is None
    # The exact id keeps the last known price, and says it ended.
    assert exact_after_retirement.status == "priced"
    assert exact_after_retirement.reasons == ("price_ended",)
    assert exact_after_retirement.canonical_model == "anthropic/claude-haiku-3.5"
    assert exact_after_retirement.cost_usd == before.cost_usd


def test_installed_catalog_prices_claude_35_haiku_until_gateway_shutdown():
    catalog = load_catalog()

    before = catalog.snapshot.cost(
        provider="anthropic",
        model="anthropic/claude-haiku-3.5",
        at=datetime(2026, 7, 11, 23, 59, tzinfo=timezone.utc),
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )
    after = catalog.snapshot.cost(
        provider="anthropic",
        model="anthropic/claude-haiku-3.5",
        at=datetime(2026, 7, 12, tzinfo=timezone.utc),
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )

    assert before.status == "priced"
    assert before.cost_usd == Decimal("4.80000000")
    assert after.status == "unpriced"


def test_installed_catalog_keeps_claude_opus_5_fast_on_the_gateway_only():
    """Anthropic's pricing and lifecycle pages name no claude-opus-5-fast, so
    the first-party row ends while the gateway listing stays open."""
    snapshot = load_catalog().snapshot
    at = datetime(2026, 10, 22, tzinfo=timezone.utc)

    direct = snapshot.resolve_price(
        model="claude-opus-5-fast", channel="anthropic-api", at=at
    )
    gateway = snapshot.resolve_price(
        model="anthropic/claude-opus-5-fast", channel="vercel-ai-gateway", at=at
    )

    assert direct is not None and direct.ended is True
    assert direct.price.effective_to == at
    assert gateway is not None and gateway.ended is False


@pytest.mark.parametrize("model", ["gemini-3.6-flash", "gemini-3.7-flash"])
@pytest.mark.parametrize(
    "batch,before_cost,after_cost",
    [
        (False, "4.50000000", "9.00000000"),
        (True, "2.25000000", "4.50000000"),
    ],
)
def test_installed_catalog_ends_gemini_flash_introductory_rates_on_2027(
    model, batch, before_cost, after_cost,
):
    catalog = load_catalog()

    before = catalog.snapshot.cost(
        provider="google",
        model=model,
        at=datetime(2026, 12, 31, 23, 59, tzinfo=timezone.utc),
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        batch=batch,
    )
    after = catalog.snapshot.cost(
        provider="google",
        model=model,
        at=datetime(2027, 1, 1, tzinfo=timezone.utc),
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        batch=batch,
    )

    assert before.status == after.status == "priced"
    assert before.cost_usd == Decimal(before_cost)
    assert after.cost_usd == Decimal(after_cost)


@pytest.mark.parametrize(
    "provider,model,canonical,cost",
    [
        ("deepseek", "deepseek/deepseek-v4.1-flash", "deepseek/deepseek-v4.1-flash", "1.50000000"),
        ("meta", "meta/muse-spark-1.3", "meta/muse-spark-1.3", "5.50000000"),
        ("zai", "zai/glm-5.3-fast", "zai/glm-5.3-fast", "8.70000000"),
        ("zai", "zai/glm-5.3", "zai/glm-5.3", "5.80000000"),
        ("zai", "zai/glm-5.3-flash", "zai/glm-5.3-flash", "0.65000000"),
    ],
)
def test_installed_catalog_prices_current_vercel_gateway_models(
    provider, model, canonical, cost
):
    catalog = load_catalog()

    result = catalog.snapshot.cost(
        provider=provider,
        model=model,
        at=_AT_CATALOG_REFRESH,
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )

    assert result.status == "priced"
    assert result.canonical_model == canonical
    assert result.cost_usd == Decimal(cost)


def test_installed_catalog_prices_direct_meta_muse_spark_1_3():
    catalog = load_catalog()

    result = catalog.snapshot.cost(
        provider="meta",
        model="muse-spark-1.3",
        at=datetime(2026, 9, 11, tzinfo=timezone.utc),
        input_tokens=600_000,
        output_tokens=100_000,
        cache_read_tokens=400_000,
    )

    # Meta reports cached tokens inside input_tokens, so only 0.2M input is
    # billed at the input rate: 0.2M * $1.25 + 0.4M * $0.15 + 0.1M * $4.25.
    assert result.status == "priced"
    assert result.canonical_model == "meta/muse-spark-1.3"
    assert result.price.pricing_channel == "meta-api"
    assert result.components.input_usd == Decimal("0.25000000")
    assert result.components.cache_read_usd == Decimal("0.06000000")
    assert result.components.output_usd == Decimal("0.42500000")
    assert result.cost_usd == Decimal("0.73500000")


_AT_KIMI_26 = datetime(2026, 9, 6, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "model,channel",
    [("moonshotai/kimi-k2.6", "vercel-ai-gateway"),
     ("accounts/fireworks/models/kimi-k2p6", "fireworks-api"),
     ("fireworks:accounts/fireworks/models/kimi-k2p6", "fireworks-api")],
)
def test_installed_catalog_prices_kimi_k2_6_on_both_channels(model, channel):
    """Both channels list the same rate; each still needs its own window, since
    channel selection is exact and never falls back."""
    catalog = load_catalog()

    resolved = catalog.price(
        model=model, channel=channel, at=_AT_KIMI_26,
        input_tokens=1_000_000, output_tokens=1_000_000,
    )
    assert resolved.status == "priced"
    assert resolved.canonical_model == "moonshotai/kimi-k2.6"
    assert resolved.cost_usd == Decimal("4.95000000")


_AT_GEMMA = datetime(2026, 9, 6, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "model,canonical",
    [("gemma-4-26b-a4b-it", "google/gemma-4-26b-a4b-it"),
     ("models/gemma-4-26b-a4b-it", "google/gemma-4-26b-a4b-it"),
     ("gemma-4-31b-it", "google/gemma-4-31b-it"),
     ("models/gemma-4-31b-it", "google/gemma-4-31b-it")],
)
def test_installed_catalog_prices_gemma_free_on_the_direct_api(model, canonical):
    """Google serves Gemma free on the direct API. A priced zero and a missing
    price are different answers: the first says the call cost nothing, the
    second says the cost is unknown and blocks an analysis that needs it. Both
    Gemma variants were captured on the direct API and resolved unpriced,
    because the catalog carried them on the gateway only."""
    catalog = load_catalog()

    resolved = catalog.price(
        model=model, channel="google-api", at=_AT_GEMMA,
        input_tokens=1_000_000, output_tokens=1_000_000,
    )
    assert resolved.status == "priced"
    assert resolved.canonical_model == canonical
    assert resolved.cost_usd == Decimal("0E-8")


@pytest.mark.parametrize(
    "model,channel,cost",
    # The gateway resells Gemma at a real rate, so free on one channel must not
    # leak into the other. Channel selection is exact.
    [("google/gemma-4-26b-a4b-it", "vercel-ai-gateway", "0.75000000"),
     ("google/gemma-4-31b-it", "vercel-ai-gateway", "0.54000000")],
)
def test_gemma_stays_paid_on_the_gateway(model, channel, cost):
    catalog = load_catalog()

    resolved = catalog.price(
        model=model, channel=channel, at=_AT_GEMMA,
        input_tokens=1_000_000, output_tokens=1_000_000,
    )
    assert resolved.status == "priced"
    assert resolved.cost_usd == Decimal(cost)


@pytest.mark.parametrize(
    "model",
    ["gemini-omni-flash-preview", "models/gemini-omni-flash-preview"],
)
def test_installed_catalog_prices_gemini_omni_flash_preview(model):
    """Records Google's text output rate. Google prices this model's output by
    modality ($9.00 text, $17.50 video) and a price row carries one output
    rate, so video-heavy traffic is under-costed until the schema can express
    per-modality output."""
    catalog = load_catalog()

    resolved = catalog.price(
        model=model, channel="google-api", at=_AT_GEMMA,
        input_tokens=1_000_000, output_tokens=1_000_000,
    )
    assert resolved.status == "priced"
    assert resolved.canonical_model == "google/gemini-omni-flash-preview"
    assert resolved.cost_usd == Decimal("10.50000000")


@pytest.mark.parametrize(
    "model", ["gemini-omni-1.1-flash", "models/gemini-omni-1.1-flash"]
)
def test_installed_catalog_prices_gemini_omni_flash_on_the_ga_id(model):
    """The preview endpoint shuts down on 2026-10-22, and the GA id is priced
    from the day it went generally available, at its video output rate."""
    catalog = load_catalog()

    resolved = catalog.price(
        model=model, channel="google-api",
        at=datetime(2026, 10, 22, tzinfo=timezone.utc),
        input_tokens=1_000_000, output_tokens=1_000_000,
    )
    before_ga = catalog.price(
        model=model, channel="google-api",
        at=datetime(2026, 8, 26, tzinfo=timezone.utc),
        input_tokens=1_000_000, output_tokens=1_000_000,
    )

    assert resolved.status == "priced"
    assert resolved.canonical_model == "google/gemini-omni-1.1-flash"
    assert resolved.cost_usd == Decimal("19.00000000")
    assert before_ga.status == "unpriced"


@pytest.mark.parametrize(
    "provider,model,canonical,cost",
    [
        # A provider's own short name, which its SDK is what actually sends.
        ("openai", "gpt-5", "openai/gpt-5", "11.25000000"),
        # 1M input crosses gpt-5.5's 272K long-context threshold.
        ("openai", "gpt-5.5", "openai/gpt-5.5", "55.00000000"),
        ("openai", "o4-mini", "openai/o4-mini", "5.50000000"),
        ("mistral", "mistral-small-4", "mistral/mistral-small-4", "0.75000000"),
        ("zai", "glm-5.1", "zai/glm-5.1", "5.80000000"),
        ("minimax", "minimax-m3", "minimax/minimax-m3", "3.00000000"),
        # Spellings that fold through the provider-alias map before lookup.
        ("xai", "grok-4.6", "xai/grok-4.6", "16.00000000"),
        ("x-ai", "grok-4.6", "xai/grok-4.6", "16.00000000"),
        ("moonshot", "kimi-k3", "moonshotai/kimi-k3", "18.00000000"),
        ("moonshotai", "kimi-k3", "moonshotai/kimi-k3", "18.00000000"),
        # Nova is sold only through Bedrock, so its bare name prices there
        # rather than on a first-party Amazon channel that does not exist.
        ("amazon", "nova-pro", "amazon/nova-pro", "4.00000000"),
        # A proxy that forwards without naming the provider still has to price,
        # on the publisher's own channel rather than a gateway list price.
        ("unknown", "gpt-5.5", "openai/gpt-5.5", "55.00000000"),
        ("litellm", "claude-opus-4.8", "anthropic/claude-opus-4.8", "30.00000000"),
    ],
)
def test_installed_catalog_prices_the_spellings_real_traffic_sends(
    provider, model, canonical, cost
):
    """Traffic names a model however its caller spelled it. A spelling core does
    not carry is not a cosmetic gap: that call stops being priced at all."""
    catalog = load_catalog()

    result = catalog.snapshot.cost(
        provider=provider,
        model=model,
        at=datetime(2026, 9, 18, tzinfo=timezone.utc),
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )

    assert result.status == "priced"
    assert result.canonical_model == canonical
    assert result.cost_usd == Decimal(cost)


def testnormalize_provider_is_public_so_one_map_serves_every_repo():
    """The dashboard and the pipeline have to fold provider spellings the same
    way, or the same call prices differently depending on which one saw it."""
    assert normalize_provider("XAI") == "x-ai"
    assert normalize_provider(" moonshot ") == "moonshotai"
    assert normalize_provider("aws-bedrock") == "bedrock"
    assert normalize_provider("openai") == "openai"


_DEEPSEEK_PEAK_CASES = [
    # DeepSeek's peak window is 01:00-04:00 and 06:00-10:00 UTC on weekdays.
    # 2026-09-16 is a Wednesday, 2026-09-19 a Saturday.
    ("2026-09-16T00:59:00+00:00", False),
    ("2026-09-16T01:00:00+00:00", True),
    ("2026-09-16T03:59:00+00:00", True),
    ("2026-09-16T04:00:00+00:00", False),
    ("2026-09-16T05:59:00+00:00", False),
    ("2026-09-16T06:00:00+00:00", True),
    ("2026-09-16T09:59:00+00:00", True),
    ("2026-09-16T10:00:00+00:00", False),
    ("2026-09-16T23:00:00+00:00", False),
    # A weekend hour inside the weekday window is still off-peak.
    ("2026-09-19T02:00:00+00:00", False),
    ("2026-09-20T07:00:00+00:00", False),
]


@pytest.mark.parametrize("moment,is_peak", _DEEPSEEK_PEAK_CASES)
def test_installed_catalog_bills_deepseek_by_the_hour_of_the_call(moment, is_peak):
    """A call's cost depends on the hour it was made, not only on the model."""
    catalog = load_catalog()

    result = catalog.snapshot.cost(
        provider="deepseek",
        model="deepseek-v4-pro",
        at=datetime.fromisoformat(moment),
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )

    assert result.status == "priced"
    # Published peak rates: 1.32 input, 3.96 output. Off-peak is exactly half.
    assert result.cost_usd == (Decimal("5.28") if is_peak else Decimal("2.64"))


def test_installed_catalog_keeps_deepseeks_pre_september_rates_for_older_calls():
    """A reprice must not rewrite what a call already cost. The older row
    predates the off-peak split, so no discount applies to it."""
    catalog = load_catalog()

    result = catalog.snapshot.cost(
        provider="deepseek",
        model="deepseek-v4-pro",
        at=datetime(2026, 9, 1, 2, tzinfo=timezone.utc),
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )

    assert result.status == "priced"
    assert result.cost_usd == Decimal("1.30500000")


@pytest.mark.parametrize(
    "moment,canonical,status",
    [
        ("2026-01-20T23:59:00+00:00", None, "unpriced"),
        ("2026-01-21T00:00:00+00:00", "google/gemini-3-flash-preview", "priced"),
        ("2026-05-19T00:00:00+00:00", "google/gemini-3.5-flash", "priced"),
        ("2026-07-21T00:00:00+00:00", "google/gemini-3.6-flash", "priced"),
        ("2026-08-13T00:00:00+00:00", "google/gemini-3.7-flash", "priced"),
        ("2026-09-02T00:00:00+00:00", "google/gemini-3.8-flash", "priced"),
    ],
)
def test_installed_catalog_resolves_geminis_moving_flash_alias(
    moment, canonical, status,
):
    catalog = load_catalog()

    result = catalog.snapshot.cost(
        provider="google",
        model="gemini-flash-latest",
        at=datetime.fromisoformat(moment),
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )

    assert result.status == status
    assert result.canonical_model == canonical


@pytest.mark.parametrize(
    "moment,canonical,status",
    [
        ("2026-03-02T23:59:00+00:00", None, "unpriced"),
        ("2026-03-03T00:00:00+00:00", "google/gemini-3.1-flash-lite", "unpriced"),
        ("2026-05-07T00:00:00+00:00", "google/gemini-3.1-flash-lite", "priced"),
        ("2026-07-21T00:00:00+00:00", "google/gemini-3.5-flash-lite", "priced"),
    ],
)
def test_installed_catalog_resolves_geminis_moving_flash_lite_alias(
    moment, canonical, status,
):
    catalog = load_catalog()

    result = catalog.snapshot.cost(
        provider="google",
        model="gemini-flash-lite-latest",
        at=datetime.fromisoformat(moment),
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )

    assert result.status == status
    assert result.canonical_model == canonical


@pytest.mark.parametrize(
    "model,released_at,canonical,cost",
    [
        (
            "mistral-large-latest",
            "2025-12-02T00:00:00+00:00",
            "mistral/mistral-large-3",
            "2.00000000",
        ),
        (
            "ministral-14b-latest",
            "2025-12-02T00:00:00+00:00",
            "mistral/ministral-14b",
            "0.40000000",
        ),
        (
            "mistral-small-latest",
            "2026-03-16T00:00:00+00:00",
            "mistral/mistral-small-4",
            "0.75000000",
        ),
        (
            "mistral-medium-latest",
            "2026-04-28T00:00:00+00:00",
            "mistral/mistral-medium-3.5",
            "9.00000000",
        ),
    ],
)
def test_installed_catalog_starts_mistrals_moving_aliases_at_their_release(
    model, released_at, canonical, cost,
):
    catalog = load_catalog()
    release = datetime.fromisoformat(released_at)

    before = catalog.snapshot.cost(
        provider="mistral",
        model=model,
        at=release - timedelta(seconds=1),
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )
    current = catalog.snapshot.cost(
        provider="mistral",
        model=model,
        at=release,
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )

    assert before.status == "unpriced"
    assert before.canonical_model is None
    assert current.status == "priced"
    assert current.canonical_model == canonical
    assert current.cost_usd == Decimal(cost)


@pytest.mark.parametrize(
    "model,cache_cost",
    [
        ("mistral-large-latest", "0.05000000"),
        ("ministral-14b-latest", "0.02000000"),
        ("mistral-small-latest", "0.01500000"),
        ("mistral-medium-latest", "0.15000000"),
    ],
)
def test_installed_catalog_prices_mistrals_current_prompt_cache(model, cache_cost):
    catalog = load_catalog()

    result = catalog.snapshot.cost(
        provider="mistral",
        model=model,
        at=datetime(2026, 10, 2, tzinfo=timezone.utc),
        input_tokens=0,
        output_tokens=0,
        cache_read_tokens=1_000_000,
    )

    assert result.status == "priced"
    assert result.cost_usd == Decimal(cache_cost)


@pytest.mark.parametrize("alias", ["deepseek-chat", "deepseek-reasoner"])
@pytest.mark.parametrize(
    "moment,canonical,cost,status",
    [
        ("2025-11-30T23:59:00+00:00", None, None, "unpriced"),
        ("2025-12-01T00:00:00+00:00", "deepseek/v3.2", None, "unpriced"),
        ("2026-04-24T00:00:00+00:00", "deepseek/v4-flash", "0.42000000", "priced"),
        ("2026-07-24T15:59:00+00:00", "deepseek/v4-flash", "0.42000000", "priced"),
        ("2026-07-24T16:00:00+00:00", None, None, "unpriced"),
    ],
)
def test_installed_catalog_resolves_deepseeks_moving_legacy_aliases(
    alias, moment, canonical, cost, status,
):
    catalog = load_catalog()

    result = catalog.snapshot.cost(
        provider="deepseek",
        model=alias,
        at=datetime.fromisoformat(moment),
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )

    assert result.status == status
    assert result.canonical_model == canonical
    assert result.cost_usd == (Decimal(cost) if cost is not None else None)


@pytest.mark.parametrize(
    "alias",
    ["deepseek-v4-flash", "deepseek-v4-flash-vision-exp"],
)
def test_installed_catalog_resolves_retired_flash_names_to_v41(alias):
    catalog = load_catalog()

    result = catalog.snapshot.cost(
        provider="deepseek",
        model=alias,
        at=datetime(2026, 9, 16, 12, tzinfo=timezone.utc),
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )

    assert result.status == "priced"
    assert result.canonical_model == "deepseek/deepseek-v4.1-flash"
    assert result.cost_usd == Decimal("0.75")


@pytest.mark.parametrize(
    "provider,model",
    [
        ("deepseek", "deepseek/deepseek-chat"),
        ("litellm", "deepseek-chat"),
        ("litellm", "deepseek/deepseek-chat"),
        ("unknown", "deepseek/deepseek-chat"),
    ],
)
def test_installed_catalog_stops_retired_deepseek_chat_spellings(provider, model):
    catalog = load_catalog()

    result = catalog.snapshot.cost(
        provider=provider,
        model=model,
        at=datetime(2026, 9, 16, 12, tzinfo=timezone.utc),
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )

    assert result.status == "unpriced"
    assert result.canonical_model is None
    assert result.cost_usd is None


def test_installed_catalog_prices_deepseeks_current_flash_name_and_its_legacy_one():
    """A retired name the provider still accepts has to keep costing what the
    provider bills for it, even though it resolves to a different model."""
    catalog = load_catalog()
    at = datetime(2026, 9, 16, 12, tzinfo=timezone.utc)

    current = catalog.snapshot.cost(
        provider="deepseek", model="deepseek-flash", at=at,
        input_tokens=1_000_000, output_tokens=1_000_000,
    )
    legacy = catalog.snapshot.cost(
        provider="deepseek", model="deepseek-v4-flash", at=at,
        input_tokens=1_000_000, output_tokens=1_000_000,
    )

    # Off-peak: half of the published 0.30 input and 1.20 output.
    assert current.cost_usd == legacy.cost_usd == Decimal("0.75")
    assert current.canonical_model == "deepseek/deepseek-v4.1-flash"
    assert legacy.canonical_model == "deepseek/deepseek-v4.1-flash"


@pytest.mark.parametrize(
    "moment,cost",
    [
        # DeepSeek's reprice took effect at 04:00 UTC, not at midnight. A date
        # alone would hand these four hours to the new rate.
        ("2026-09-09T23:59:00+00:00", "1.30500000"),
        ("2026-09-10T00:00:00+00:00", "1.30500000"),
        ("2026-09-10T03:59:00+00:00", "1.30500000"),
        ("2026-09-10T04:00:00+00:00", "2.64000000"),
        ("2026-09-10T05:00:00+00:00", "2.64000000"),
    ],
)
def test_installed_catalog_moves_to_deepseeks_new_rate_at_the_stated_hour(moment, cost):
    """A price row states when it takes effect, not on which day. A reprice
    announced for a particular hour has to hold the old rate until then, or every
    call in between is billed at a rate that was not yet in force."""
    catalog = load_catalog()

    result = catalog.snapshot.cost(
        provider="deepseek",
        model="deepseek-v4-pro",
        at=datetime.fromisoformat(moment),
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )

    assert result.status == "priced"
    assert result.cost_usd == Decimal(cost)


def test_installed_catalog_leaves_no_gap_or_overlap_at_the_reprice_hour():
    """The two rows meet exactly: the hour the old one ends is the hour the new
    one begins, so no call falls between them or is claimed by both."""
    catalog = load_catalog()
    rows = [
        price
        for model in catalog.document["models"]
        if model["canonical_id"] == "deepseek/v4-pro"
        for price in model["prices"]
        if price["channel"] == "deepseek-api"
    ]

    ended = {str(row.get("effective_to")) for row in rows if row.get("effective_to")}
    began = {str(row["effective_from"]) for row in rows}

    assert "2026-09-10T04:00:00+00:00" in ended & began


_GATEWAY = "vercel-ai-gateway"
_GATEWAY_REFRESH = datetime(2026, 10, 2, tzinfo=timezone.utc)


def _gateway_price(model, at):
    return load_catalog().snapshot.resolve_price(model=model, channel=_GATEWAY, at=at)


@pytest.mark.parametrize(
    "model,field,before,after",
    [
        ("openai/gpt-5.6-sol", "input_per_mtok", "2.00", "4.00"),
        ("openai/gpt-5.6-sol", "output_per_mtok", "10.00", "20.00"),
        ("openai/gpt-5.6-sol", "cache_read_per_mtok", "0.20", "0.40"),
        ("openai/gpt-5.6-sol", "cache_write_5m_per_mtok", "2.50", "5.00"),
        ("deepseek/deepseek-v4.1-flash", "cache_read_per_mtok", "0.03", "0.007"),
        ("moonshotai/kimi-k2.6", "cache_read_per_mtok", None, "0.16"),
        ("anthropic/claude-3-haiku", "cache_write_5m_per_mtok", None, "0.30"),
    ],
)
def test_gateway_rate_changes_keep_the_prior_window(model, field, before, after):
    prior = _gateway_price(model, _GATEWAY_REFRESH - timedelta(seconds=1))
    current = _gateway_price(model, _GATEWAY_REFRESH)

    assert prior is not None and current is not None
    assert prior.price.id != current.price.id
    assert getattr(prior.price, field) == (Decimal(before) if before else None)
    assert getattr(current.price, field) == Decimal(after)


@pytest.mark.parametrize(
    "model,flat_cost,tiered_cost",
    [
        ("openai/gpt-5.4", "0.75000000", "1.50000000"),
        ("openai/gpt-5.5", "1.50000000", "3.00000000"),
        ("anthropic/claude-sonnet-4.5", "0.90000000", "1.80000000"),
        ("anthropic/claude-sonnet-4", "0.90000000", "1.80000000"),
        ("google/gemini-2.5-pro", "0.37500000", "0.75000000"),
        ("spacexai/grok-4.6", "0.60000000", "1.20000000"),
        ("alibaba/qwen3.7-plus", "0.12000000", "0.36000000"),
    ],
)
def test_gateway_long_context_tier_starts_at_the_refresh(model, flat_cost, tiered_cost):
    catalog = load_catalog()
    usage = dict(model=model, channel=_GATEWAY, input_tokens=300_000, output_tokens=0)

    before = catalog.price(at=_GATEWAY_REFRESH - timedelta(seconds=1), **usage)
    after = catalog.price(at=_GATEWAY_REFRESH, **usage)
    below_threshold = catalog.price(
        model=model, channel=_GATEWAY, at=_GATEWAY_REFRESH,
        input_tokens=100_000, output_tokens=0,
    )

    assert before.status == after.status == "priced"
    assert before.cost_usd == Decimal(flat_cost)
    assert after.cost_usd == Decimal(tiered_cost)
    assert below_threshold.cost_usd == Decimal(flat_cost) / 3


@pytest.mark.parametrize(
    "model,effective_from,short_cost,tiered_cost",
    [
        ("openai/gpt-5.4-pro", "2026-03-05", "3.00000000", "18.00000000"),
        ("openai/gpt-5.5", "2026-04-24", "0.50000000", "3.00000000"),
        ("openai/gpt-5.5-pro", "2026-04-24", "3.00000000", "18.00000000"),
    ],
)
def test_direct_openai_rows_charge_the_documented_long_context_surcharge(
    model, effective_from, short_cost, tiered_cost
):
    """OpenAI prices a >272K prompt at 2x input and 1.5x output on these models
    from the start of their direct price."""
    catalog = load_catalog()
    at = datetime.fromisoformat(effective_from).replace(tzinfo=timezone.utc)

    long_prompt = catalog.price(
        model=model, channel="openai-api", at=at,
        input_tokens=300_000, output_tokens=0,
    )
    short_prompt = catalog.price(
        model=model, channel="openai-api", at=at,
        input_tokens=100_000, output_tokens=0,
    )

    assert long_prompt.status == short_prompt.status == "priced"
    assert long_prompt.cost_usd == Decimal(tiered_cost)
    assert short_prompt.cost_usd == Decimal(short_cost)


def test_gpt_5_5_pro_long_context_surcharge_skips_batch():
    """OpenAI lists no long-context batch rate for gpt-5.5-pro, so a long
    batch prompt keeps the flat batch rates."""
    catalog = load_catalog()
    usage = dict(
        model="openai/gpt-5.5-pro", channel="openai-api",
        at=datetime(2026, 4, 24, tzinfo=timezone.utc),
        input_tokens=300_000, output_tokens=100_000,
    )

    standard = catalog.price(**usage)
    batch = catalog.price(batch=True, **usage)

    assert standard.status == batch.status == "priced"
    # 300K x $60 + 100K x $270.
    assert standard.cost_usd == Decimal("45.00000000")
    # 300K x $15 + 100K x $90.
    assert batch.cost_usd == Decimal("13.50000000")


@pytest.mark.parametrize(
    "moment,cost",
    [
        # Thursday, before the peak rule is recorded: one flat rate.
        ("2026-10-01T02:00:00+00:00", "2.64000000"),
        # Monday inside and outside the peak windows, then a weekend.
        ("2026-10-05T02:00:00+00:00", "5.28000000"),
        ("2026-10-05T05:00:00+00:00", "2.64000000"),
        ("2026-10-05T09:59:00+00:00", "5.28000000"),
        ("2026-10-05T10:00:00+00:00", "2.64000000"),
        ("2026-10-03T02:00:00+00:00", "2.64000000"),
    ],
)
def test_gateway_deepseek_v4_pro_doubles_in_peak_hours(moment, cost):
    result = load_catalog().price(
        model="deepseek/deepseek-v4-pro", channel=_GATEWAY,
        at=datetime.fromisoformat(moment),
        input_tokens=1_000_000, output_tokens=1_000_000,
    )

    assert result.status == "priced"
    assert result.canonical_model == "deepseek/v4-pro"
    assert result.cost_usd == Decimal(cost)


@pytest.mark.parametrize(
    "model,varies",
    [
        ("openai/gpt-5.6-sol", True),
        ("openai/gpt-5.5", True),
        ("openai/gpt-5.2", True),
        ("deepseek/deepseek-v4-pro", True),
        ("deepseek/deepseek-v4.1-flash", True),
        ("moonshotai/kimi-k2.6", True),
        ("moonshotai/kimi-k3-fast", True),
        ("openai/gpt-6.1-sol", False),
        ("spacexai/grok-4.7", False),
        ("anthropic/claude-opus-5.5-fast", False),
    ],
)
def test_gateway_rows_record_whether_the_rate_varies_by_provider(model, varies):
    resolved = _gateway_price(model, _GATEWAY_REFRESH)

    assert resolved is not None
    assert bool(resolved.rules.get("varies_by_provider")) is varies


@pytest.mark.parametrize(
    "provider,model,canonical,input_rate,output_rate",
    [
        ("openai", "openai/gpt-6.1-sol", "openai/gpt-6.1-sol", "2.00", "10.00"),
        ("anthropic", "anthropic/claude-opus-5.5-fast", "anthropic/claude-opus-5.5-fast", "8.00", "40.00"),
        ("alibaba", "alibaba/qwen3.8-max-prime", "alibaba/qwen3.8-max-prime", "4.00", "12.00"),
        ("moonshotai", "moonshotai/kimi-k3-fast", "moonshotai/kimi-k3-fast", "4.50", "22.50"),
        ("vercel", "openai/gpt-5.4-pro", "openai/gpt-5.4-pro", "30.00", "180.00"),
        ("vercel", "openai/gpt-5.3-codex", "openai/gpt-5.3-codex", "1.75", "14.00"),
        ("vercel", "openai/gpt-5.2", "openai/gpt-5.2", "1.75", "14.00"),
        ("vercel", "openai/gpt-5", "openai/gpt-5", "1.25", "10.00"),
        ("vercel", "openai/gpt-4.1-mini", "openai/gpt-4.1-mini", "0.40", "1.60"),
        ("vercel", "openai/gpt-4.1-nano", "openai/gpt-4.1-nano", "0.10", "0.40"),
        ("vercel", "openai/gpt-4o", "openai/gpt-4o", "2.50", "10.00"),
        ("vercel", "openai/gpt-4o-mini", "openai/gpt-4o-mini", "0.15", "0.60"),
        ("vercel", "anthropic/claude-opus-4", "anthropic/claude-opus-4", "15.00", "75.00"),
        ("vercel", "mistral/mistral-medium-3.5", "mistral/mistral-medium-3.5", "1.50", "7.50"),
        ("vercel", "alibaba/qwen3.7-max", "alibaba/qwen3.7-max", "2.50", "7.50"),
        ("vercel", "amazon/nova-pro", "amazon/nova-pro", "0.80", "3.20"),
        ("vercel", "amazon/nova-lite", "amazon/nova-lite", "0.06", "0.24"),
        ("vercel", "amazon/nova-micro", "amazon/nova-micro", "0.035", "0.14"),
    ],
)
def test_new_gateway_routes_start_at_the_first_verified_listing(
    provider, model, canonical, input_rate, output_rate
):
    """New routes start at their first verified Gateway listing."""
    snapshot = load_catalog().snapshot
    usage = dict(provider=provider, model=model, input_tokens=1_000_000, output_tokens=1_000_000)

    assert _gateway_price(model, _GATEWAY_REFRESH - timedelta(seconds=1)) is None
    before = snapshot.cost(at=_GATEWAY_REFRESH - timedelta(seconds=1), **usage)
    assert (before.status, before.reasons) == ("unpriced", ("no_effective_alias",))

    resolved = _gateway_price(model, _GATEWAY_REFRESH)
    assert resolved is not None
    assert resolved.canonical_model == canonical
    assert resolved.price.id == f"{canonical}:{_GATEWAY}:global:2026-10-02"
    assert resolved.price.input_per_mtok == Decimal(input_rate)
    assert resolved.price.output_per_mtok == Decimal(output_rate)
    after = snapshot.cost(at=_GATEWAY_REFRESH, **usage)
    assert after.status == "priced"
    assert after.price_id == resolved.price.id


@pytest.mark.parametrize(
    "model,canonical,cache_read",
    [("spacexai/grok-4.7", "xai/grok-4.7", "0.50"), ("spacexai/grok-4.5", "xai/grok-4.5", "0.30")],
)
def test_new_gateway_grok_routes_start_at_the_first_verified_listing(
    model, canonical, cache_read
):
    assert _gateway_price(model, _GATEWAY_REFRESH - timedelta(seconds=1)) is None
    resolved = _gateway_price(model, _GATEWAY_REFRESH)

    assert resolved is not None
    assert resolved.canonical_model == canonical
    assert resolved.price.id == f"{canonical}:{_GATEWAY}:global:2026-10-02"
    assert resolved.price.input_per_mtok == Decimal("2.00")
    assert resolved.price.output_per_mtok == Decimal("6.00")
    assert resolved.price.cache_read_per_mtok == Decimal(cache_read)


def test_gateway_gpt_6_1_sol_prices_cache_and_long_context():
    catalog = load_catalog()

    cached = catalog.price(
        model="openai/gpt-6.1-sol", channel=_GATEWAY, at=_GATEWAY_REFRESH,
        input_tokens=1_000_000, output_tokens=0,
        cache_read_tokens=400_000, cache_write_tokens=100_000,
    )
    short = catalog.price(
        model="openai/gpt-6.1-sol", channel=_GATEWAY, at=_GATEWAY_REFRESH,
        input_tokens=100_000, output_tokens=100_000,
    )

    # Long context: 2x on input, cache read, and cache write.
    assert cached.status == "priced"
    assert cached.cost_usd == Decimal("2.58000000")
    assert short.cost_usd == Decimal("1.20000000")


def test_gateway_alias_does_not_reprice_the_direct_channel():
    """Gateway aliases do not change direct-channel prices."""
    catalog = load_catalog()
    usage = dict(
        model="openai/gpt-5.2", at=_GATEWAY_REFRESH,
        input_tokens=1_000_000, output_tokens=0, cache_read_tokens=400_000,
    )

    gateway = catalog.snapshot.cost(provider="vercel", **usage)
    direct = catalog.snapshot.cost(provider="openai", **usage)

    assert gateway.price_id == "openai/gpt-5.2:vercel-ai-gateway:global:2026-10-02"
    assert direct.price_id == "openai/gpt-5.2:openai-api:global:2025-12-11"
    assert gateway.cost_usd == direct.cost_usd == Decimal("1.12000000")


_AT_FIREWORKS_REFRESH = datetime(2026, 10, 2, tzinfo=timezone.utc)

# Current generative serverless offerings and their channel-local rates.
_FIREWORKS_SERVERLESS = [
    ("ember-1", "fireworks/ember-1", "2026-09-22", "3.00", "0.30", "15.00"),
    ("deepseek-v4p1-flash", "deepseek/deepseek-v4.1-flash", "2026-09-10", "0.30", "0.006", "1.20"),
    ("glm-5p3-flash", "zai/glm-5.3-flash", "2026-08-26", "0.15", "0.03", "0.50"),
    ("glm-5p3", "zai/glm-5.3", "2026-08-28", "1.40", "0.26", "4.40"),
    ("kimi-k3", "moonshotai/kimi-k3", "2026-07-19", "3.00", "0.30", "15.00"),
    ("qwen3p8-max", "alibaba/qwen3.8-max", "2026-08-05", "2.00", "0.25", "6.00"),
    ("glm-5p2", "fireworks/glm-5p2", "2026-08-24", "1.40", "0.14", "4.40"),
    ("minimax-m3", "minimax/minimax-m3", "2026-06-11", "0.30", "0.06", "1.20"),
    ("gpt-oss-120b", "openai/gpt-oss-120b", "2025-08-04", "0.15", "0.015", "0.60"),
    ("nemotron-3-ultra-nvfp4", "nvidia/nemotron-3-ultra-nvfp4", "2026-06-02", "0.60", "0.12", "2.40"),
    ("nemotron-lightning-3p5-30b-a3b", "nvidia/nemotron-lightning-3.5-30b-a3b", "2026-08-07", "0.05", "0.01", "0.20"),
    ("inkling", "thinkingmachines/inkling", "2026-07-14", "1.00", "0.17", "4.05"),
]
_FIREWORKS_PATH = "accounts/fireworks/models/"

_fireworks_offerings = pytest.mark.parametrize(
    "slug,canonical,start,input_rate,cached_rate,output_rate",
    _FIREWORKS_SERVERLESS,
    ids=[row[0] for row in _FIREWORKS_SERVERLESS],
)
# Accepted Fireworks model-path forms.
_fireworks_alias_forms = pytest.mark.parametrize(
    "prefix", ["", "fireworks:"], ids=["exact-path", "prefixed"]
)


@_fireworks_offerings
@_fireworks_alias_forms
def test_installed_catalog_prices_fireworks_serverless_at_one_million_tokens(
    prefix, slug, canonical, start, input_rate, cached_rate, output_rate
):
    catalog = load_catalog()
    model = prefix + _FIREWORKS_PATH + slug

    resolved = catalog.price(
        model=model, channel="fireworks-api", at=_AT_FIREWORKS_REFRESH,
        input_tokens=1_000_000, output_tokens=1_000_000,
    )
    assert resolved.status == "priced"
    assert resolved.canonical_model == canonical
    assert resolved.price_id == f"{canonical}:fireworks-api:global:{start}"
    assert resolved.cost_usd == Decimal(input_rate) + Decimal(output_rate)

    observed = catalog.snapshot.cost(
        provider="fireworks", model=model, at=_AT_FIREWORKS_REFRESH,
        input_tokens=1_000_000, output_tokens=1_000_000,
    )
    assert observed.status == "priced"
    assert observed.price_id == resolved.price_id
    assert observed.cost_usd == resolved.cost_usd


@_fireworks_offerings
@_fireworks_alias_forms
def test_installed_catalog_bills_fireworks_cache_reads_out_of_input(
    prefix, slug, canonical, start, input_rate, cached_rate, output_rate
):
    """Cached prompt tokens are not billed again as input."""
    catalog = load_catalog()

    cached = catalog.price(
        model=prefix + _FIREWORKS_PATH + slug, channel="fireworks-api",
        at=_AT_FIREWORKS_REFRESH,
        input_tokens=1_000_000, output_tokens=0, cache_read_tokens=1_000_000,
    )
    assert cached.status == "priced"
    assert cached.cost_usd == Decimal(cached_rate)


@_fireworks_offerings
@_fireworks_alias_forms
def test_installed_catalog_starts_fireworks_serverless_at_its_availability_boundary(
    prefix, slug, canonical, start, input_rate, cached_rate, output_rate
):
    catalog = load_catalog()
    model = prefix + _FIREWORKS_PATH + slug
    boundary = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)

    before = catalog.price(
        model=model, channel="fireworks-api", at=boundary - timedelta(minutes=1),
        input_tokens=1_000_000, output_tokens=1_000_000,
    )
    at_start = catalog.price(
        model=model, channel="fireworks-api", at=boundary,
        input_tokens=1_000_000, output_tokens=1_000_000,
    )

    assert before.status == "unpriced"
    assert before.cost_usd is None
    assert at_start.status == "priced"
    assert at_start.cost_usd == Decimal(input_rate) + Decimal(output_rate)


@_fireworks_offerings
def test_installed_catalog_sources_fireworks_serverless_from_its_model_page(
    slug, canonical, start, input_rate, cached_rate, output_rate
):
    catalog = load_catalog()

    resolved = catalog.snapshot.resolve_price(
        model=_FIREWORKS_PATH + slug, channel="fireworks-api",
        at=_AT_FIREWORKS_REFRESH,
    )
    assert resolved is not None
    assert resolved.price.effective_to is None
    assert resolved.price.cache_read_per_mtok == Decimal(cached_rate)
    assert resolved.rules["input_includes_cache_read"] is True
    assert resolved.price.source_url.startswith("https://fireworks.ai/models/")
    assert resolved.price.source_url.endswith(f"/{slug}")


@pytest.mark.parametrize(
    "slug,other_model,other_channel,other_cost",
    # Creator models retain channel-local pricing.
    [("deepseek-v4p1-flash", "deepseek/deepseek-v4.1-flash", "vercel-ai-gateway", "1.50"),
     ("kimi-k3", "kimi-k3", "moonshot-api", "18.00"),
     ("kimi-k3", "moonshotai/kimi-k3", "vercel-ai-gateway", "18.00"),
     ("qwen3p8-max", "qwen3.8-max", "alibaba-api", "8.00"),
     # Direct-provider tiering does not cross channels.
     ("minimax-m3", "minimax-m3", "minimax-api", "3.00"),
     ("inkling", "thinkingmachines/inkling", "vercel-ai-gateway", "5.05"),
     ("glm-5p3", "zai/glm-5.3", "vercel-ai-gateway", "5.80")],
)
def test_installed_catalog_keeps_fireworks_paths_on_the_fireworks_channel(
    slug, other_model, other_channel, other_cost
):
    catalog = load_catalog()
    tokens = {"input_tokens": 1_000_000, "output_tokens": 1_000_000}

    off_channel = catalog.price(
        model=_FIREWORKS_PATH + slug, channel=other_channel,
        at=_AT_FIREWORKS_REFRESH, **tokens,
    )
    assert off_channel.status == "unpriced"
    assert off_channel.cost_usd is None

    other = catalog.price(
        model=other_model, channel=other_channel, at=_AT_FIREWORKS_REFRESH, **tokens,
    )
    assert other.status == "priced"
    assert ":fireworks-api:" not in other.price_id
    assert other.cost_usd == Decimal(other_cost)

    fireworks = catalog.price(
        model=_FIREWORKS_PATH + slug, channel="fireworks-api",
        at=_AT_FIREWORKS_REFRESH, **tokens,
    )
    assert fireworks.canonical_model == other.canonical_model
    assert ":fireworks-api:" in fireworks.price_id


def test_installed_catalog_prices_deepseek_v41_flash_on_fireworks_at_the_serverless_block_rate():
    """Displayed serverless rates govern when page prose conflicts."""
    catalog = load_catalog()
    rows = [
        price
        for model in catalog.document["models"]
        if model["canonical_id"] == "deepseek/deepseek-v4.1-flash"
        for price in model["prices"]
        if price["channel"] == "fireworks-api"
    ]

    assert len(rows) == 1
    assert rows[0].get("effective_to") is None
    assert Decimal(str(rows[0]["input_per_mtok"])) == Decimal("0.30")
    assert Decimal(str(rows[0]["cache_read_per_mtok"])) == Decimal("0.006")
    assert Decimal(str(rows[0]["output_per_mtok"])) == Decimal("1.20")

    weekend = catalog.price(
        model="accounts/fireworks/models/deepseek-v4p1-flash",
        channel="fireworks-api", at=datetime(2026, 9, 26, 12, tzinfo=timezone.utc),
        input_tokens=1_000_000, output_tokens=1_000_000,
    )
    assert weekend.status == "priced"
    assert weekend.cost_usd == Decimal("1.50")


def test_installed_catalog_keeps_glm_52_open_on_fireworks_past_its_declared_deprecation():
    """Verified availability outranks a declared retirement date."""
    catalog = load_catalog()

    for moment in (datetime(2026, 9, 25, tzinfo=timezone.utc), _AT_FIREWORKS_REFRESH):
        resolved = catalog.price(
            model="accounts/fireworks/models/glm-5p2", channel="fireworks-api",
            at=moment, input_tokens=1_000_000, output_tokens=0,
            cache_read_tokens=1_000_000,
        )
        assert resolved.status == "priced"
        assert resolved.cost_usd == Decimal("0.14")


@pytest.mark.parametrize("slug", ["qwen3-reranker-8b", "qwen3-embedding-8b"])
@_fireworks_alias_forms
def test_installed_catalog_leaves_fireworks_retrieval_only_models_unpriced(prefix, slug):
    """Retrieval-only offerings remain outside generation pricing."""
    catalog = load_catalog()

    resolved = catalog.price(
        model=prefix + _FIREWORKS_PATH + slug, channel="fireworks-api",
        at=_AT_FIREWORKS_REFRESH, input_tokens=1_000_000, output_tokens=0,
    )
    assert resolved.status == "unpriced"
    assert resolved.cost_usd is None


def test_installed_catalog_gives_each_fireworks_path_one_canonical_model():
    """Each Fireworks path belongs to one canonical model."""
    catalog = load_catalog()
    owners: dict[str, set[str]] = {}
    spellings: set[str] = set()
    for model in catalog.document["models"]:
        for alias in model["aliases"]:
            if alias["channel"] != "fireworks-api":
                continue
            assert alias["provider"] == "fireworks"
            assert alias["rules"]["input_includes_cache_read"] is True
            spellings.add(alias["alias"])
            owners.setdefault(
                alias["alias"].removeprefix("fireworks:"), set()
            ).add(model["canonical_id"])

    expected = {
        _FIREWORKS_PATH + slug: {canonical}
        for slug, canonical, *_ in _FIREWORKS_SERVERLESS
    }
    assert {path: owners.get(path) for path in expected} == expected
    assert all(len(canonicals) == 1 for canonicals in owners.values())
    for path in owners:
        assert {path, f"fireworks:{path}"} <= spellings


@pytest.mark.parametrize(
    ("at", "input_rate", "output_rate"),
    [
        (datetime(2026, 7, 29, tzinfo=timezone.utc), "2.5", "15.0"),
        (datetime(2026, 7, 30, tzinfo=timezone.utc), "2.0", "12.0"),
    ],
)
def test_installed_catalog_prices_gpt_5_6_terra_direct_by_date(at, input_rate, output_rate):
    """OpenAI cut gpt-5.6-terra to $2 in and $12 out on 2026-07-30; calls
    before that day keep the launch rate."""
    catalog = load_catalog()
    resolved = catalog.price(
        model="gpt-5.6-terra", channel="openai-api", at=at,
        input_tokens=100_000, output_tokens=100_000,
    )
    assert resolved.status == "priced"
    assert resolved.cost_usd == (Decimal(input_rate) + Decimal(output_rate)) / 10


_GPT_6_1_TERRA_RELEASE = datetime(2026, 10, 26, tzinfo=timezone.utc)


def test_installed_catalog_prices_gpt_6_1_terra_from_its_release_day():
    catalog = load_catalog()

    def cost(at, **usage):
        return catalog.snapshot.cost(
            provider="openai", model="gpt-6.1-terra", at=at, **usage
        )

    before = cost(
        _GPT_6_1_TERRA_RELEASE - timedelta(seconds=1),
        input_tokens=1_000, output_tokens=1_000,
    )
    standard = cost(
        _GPT_6_1_TERRA_RELEASE, input_tokens=100_000, output_tokens=100_000
    )
    at_threshold = cost(_GPT_6_1_TERRA_RELEASE, input_tokens=272_000, output_tokens=0)
    over_threshold = cost(_GPT_6_1_TERRA_RELEASE, input_tokens=272_001, output_tokens=0)
    resolved = catalog.snapshot.resolve_price(
        model="gpt-6.1-terra", channel="openai-api", at=_GPT_6_1_TERRA_RELEASE
    )

    assert before.status == "unpriced"
    assert standard.status == "priced"
    assert standard.canonical_model == "openai/gpt-6.1-terra"
    assert standard.cost_usd == Decimal("0.60000000")
    assert at_threshold.cost_usd == Decimal("0.27200000")
    assert over_threshold.cost_usd == Decimal("0.54400200")
    assert resolved.price.cache_read_per_mtok == Decimal("0.05")
    assert resolved.price.cache_write_5m_per_mtok == Decimal("1.25")
    assert resolved.price.batch_input_per_mtok == Decimal("0.50")
    assert resolved.price.batch_output_per_mtok == Decimal("2.50")


def test_installed_catalog_prices_gpt_6_1_terra_batch_cache_components():
    catalog = load_catalog()
    at = _GPT_6_1_TERRA_RELEASE
    read_usage = dict(input_tokens=100_000, output_tokens=0, cache_read_tokens=100_000)
    write_usage = dict(input_tokens=100_000, output_tokens=0, cache_write_5m_tokens=100_000)

    standard_read = catalog.price(model="gpt-6.1-terra", channel="openai-api", at=at, **read_usage)
    batch_read = catalog.price(model="gpt-6.1-terra", channel="openai-api", at=at, batch=True, **read_usage)
    standard_write = catalog.price(model="gpt-6.1-terra", channel="openai-api", at=at, **write_usage)
    batch_write = catalog.price(model="gpt-6.1-terra", channel="openai-api", at=at, batch=True, **write_usage)

    assert standard_read.status == batch_read.status == "priced"
    assert standard_read.cost_usd == Decimal("0.00500000")
    assert batch_read.cost_usd == Decimal("0.00250000")
    assert standard_write.status == batch_write.status == "priced"
    assert standard_write.cost_usd == Decimal("0.12500000")
    assert batch_write.cost_usd == Decimal("0.06250000")


@pytest.mark.parametrize(
    ("model", "input_rate", "output_rate", "cache_rate", "write_rate"),
    [
        ("kimi-k3.1-pro", "2.00", "10.00", "0.20", "2.00"),
        ("kimi-k3.1-pro-highspeed", "4.00", "20.00", "0.20", "4.00"),
    ],
)
def test_installed_catalog_prices_kimi_k3_1_direct_models(
    model, input_rate, output_rate, cache_rate, write_rate
):
    catalog = load_catalog()
    at = datetime(2026, 10, 24, tzinfo=timezone.utc)
    usage = dict(input_tokens=100_000, output_tokens=100_000)
    before = catalog.price(model=model, channel="moonshot-api", at=at - timedelta(seconds=1), **usage)
    standard = catalog.price(model=model, channel="moonshot-api", at=at, **usage)
    cached = catalog.price(
        model=model, channel="moonshot-api", at=at,
        input_tokens=100_000, output_tokens=0,
        cache_read_tokens=50_000, cache_write_5m_tokens=50_000,
    )

    assert before.status == "unpriced"
    assert standard.status == "priced"
    assert standard.cost_usd == (Decimal(input_rate) + Decimal(output_rate)) / 10
    assert cached.status == "priced"
    assert cached.components.input == Decimal("0")
    assert cached.components.cache_read == Decimal(cache_rate) / 20
    assert cached.components.cache_write == Decimal(write_rate) / 20

    # Moonshot publishes a 1h tier but does not list a verified K3.1 rate.
    one_hour = catalog.price(
        model=model, channel="moonshot-api", at=at,
        input_tokens=100_000, output_tokens=0, cache_write_1h_tokens=50_000,
    )
    assert one_hour.status == "partial"
    assert "cache_write_1h_rate_unavailable" in one_hour.reasons


@pytest.mark.parametrize("channel", ["vercel-ai-gateway", "openrouter-api"])
@pytest.mark.parametrize(
    ("at", "cost"),
    [
        (datetime(2026, 10, 20, 12, tzinfo=timezone.utc), "0.42"),
        (datetime(2026, 10, 21, tzinfo=timezone.utc), "0.357"),
    ],
)
def test_installed_catalog_moves_gateway_mimo_v2_5_to_the_lower_rate_on_october_21(channel, at, cost):
    resolved = load_catalog().price(
        model="xiaomi/mimo-v2.5", channel=channel, at=at,
        input_tokens=1_000_000, output_tokens=1_000_000,
    )
    assert resolved.status == "priced"
    assert resolved.cost_usd == Decimal(cost)


def test_installed_catalog_keeps_gemini_4_argon_batch_cache_partial():
    resolved = load_catalog().price(
        model="gemini-4-argon", channel="google-api",
        at=datetime(2026, 10, 23, tzinfo=timezone.utc),
        input_tokens=1_000_000, output_tokens=0,
        cache_read_tokens=1_000_000, batch=True,
    )
    assert resolved.status == "partial"
    assert "batch_rate_unavailable" in resolved.reasons


@pytest.mark.parametrize(
    ("at", "cost"),
    [
        (datetime(2026, 10, 25, tzinfo=timezone.utc), "6.00000000"),
        (datetime(2026, 12, 1, tzinfo=timezone.utc), "12.00000000"),
    ],
)
def test_installed_catalog_prices_gemini_4_argon_batch_tokens(at, cost):
    resolved = load_catalog().price(
        model="gemini-4-argon", channel="google-api", at=at,
        input_tokens=1_000_000, output_tokens=1_000_000, batch=True,
    )
    assert resolved.status == "priced"
    assert resolved.cost_usd == Decimal(cost)


@pytest.mark.parametrize(
    ("at", "batch_cache_cost", "standard_cache_cost"),
    [
        (datetime(2026, 10, 25, tzinfo=timezone.utc), "0.05", "0.10"),
        (datetime(2026, 12, 1, tzinfo=timezone.utc), "0.10", "0.20"),
    ],
)
def test_installed_catalog_prices_gemini_4_argon_batch_cache_reads(
    at, batch_cache_cost, standard_cache_cost
):
    usage = dict(input_tokens=1_000_000, output_tokens=0, cache_read_tokens=1_000_000)
    batch = load_catalog().price(
        model="gemini-4-argon", channel="google-api", at=at,
        batch=True, **usage,
    )
    standard = load_catalog().price(
        model="gemini-4-argon", channel="google-api", at=at,
        batch=False, **usage,
    )
    assert batch.status == standard.status == "priced"
    assert batch.components.cache_read == Decimal(batch_cache_cost)
    assert standard.components.cache_read == Decimal(standard_cache_cost)
    assert batch.cost_usd == Decimal(batch_cache_cost)
    assert standard.cost_usd == Decimal(standard_cache_cost)


@pytest.mark.parametrize(
    ("at", "cost", "reasons"),
    [
        (datetime(2026, 10, 22, tzinfo=timezone.utc), "2.00000000", ("uncaptured_fees",)),
        (
            datetime(2026, 10, 23, tzinfo=timezone.utc),
            "2.00000000",
            ("price_ended", "uncaptured_fees"),
        ),
    ],
)
def test_installed_catalog_marks_direct_sonar_token_rates_as_last_known(at, cost, reasons):
    resolved = load_catalog().price(
        model="perplexity/sonar", channel="perplexity-api", at=at,
        input_tokens=1_000_000, output_tokens=1_000_000,
    )
    assert resolved.status == "partial"
    assert resolved.reasons == reasons
    assert resolved.cost_usd == Decimal(cost)


def test_installed_catalog_does_not_transfer_agent_api_sonar_cache_rate():
    resolved = load_catalog().snapshot.resolve_price(
        model="perplexity/sonar", channel="perplexity-api",
        at=datetime(2026, 10, 23, tzinfo=timezone.utc),
    )
    assert resolved is not None
    assert resolved.price.cache_read_per_mtok is None


def test_installed_catalog_charges_gateway_sonar_no_per_request_search_fee():
    resolved = load_catalog().price(
        model="perplexity/sonar", channel="vercel-ai-gateway",
        at=datetime(2026, 10, 24, tzinfo=timezone.utc),
        input_tokens=1_000, output_tokens=500,
    )
    assert resolved.status == "partial"
    assert resolved.reasons == ("uncaptured_fees",)
    assert resolved.cost_usd == Decimal("0.00150000")
    assert resolved.components.request_usd == 0


# --- A price nothing replaced -------------------------------------------------

_AFTER_END = datetime(2026, 5, 1, tzinfo=timezone.utc)


def test_a_call_after_an_unreplaced_price_ended_is_priced_at_that_price(tmp_path):
    catalog = _catalog(tmp_path)
    result = catalog.calculate_cost(
        model="gem-old", provider="google", at=_AFTER_END,
        usage=ModelUsage(input_tokens=1_000_000, output_tokens=1_000_000),
    )
    assert result.status == "priced"
    assert result.reasons == ("price_ended",)
    assert result.cost == Decimal("3.20000000")
    assert result.price.id == "google/gem-old:google-api:global:2026-01-01"
    assert result.price.effective_to == datetime(2026, 4, 1, tzinfo=timezone.utc)
    assert result.missing_dimensions == ()


def test_lookup_reports_an_ended_price(tmp_path):
    catalog = _catalog(tmp_path)
    found = catalog.lookup_price(model="gem-old", provider="google", at=_AFTER_END)
    assert found.status == "priced"
    assert found.reasons == ("price_ended",)
    assert found.model_id == "google/gem-old"
    within = catalog.lookup_price(model="gem-old", provider="google", at=_AT)
    assert within.reasons == ()


def test_every_entry_point_extends_an_unreplaced_price(tmp_path):
    catalog = _catalog(tmp_path)
    by_provider = catalog.snapshot.cost(
        provider="google", model="gem-old", at=_AFTER_END,
        input_tokens=1000, output_tokens=1000,
    )
    by_channel = catalog.price(
        model="gem-old", channel="google-api", at=_AFTER_END,
        input_tokens=1000, output_tokens=1000,
    )
    resolved = catalog.snapshot.resolve_price(model="gem-old", channel="google-api", at=_AFTER_END)
    assert by_provider.status == by_channel.status == "priced"
    assert by_provider.reasons == by_channel.reasons == ("price_ended",)
    assert by_provider.cost_usd == by_channel.cost_usd
    assert resolved.ended is True


def test_a_reason_beside_an_ended_price_still_makes_it_partial(tmp_path):
    catalog = _catalog(tmp_path)
    result = catalog.calculate_cost(
        model="gem-old", provider="google", at=_AFTER_END,
        usage=ModelUsage(input_tokens=1000),
    )
    assert result.status == "partial"
    assert result.reasons == ("price_ended", "missing_output_tokens")


def test_a_superseded_price_never_extends(tmp_path):
    catalog = _catalog(tmp_path)
    found = catalog.lookup_price(model="gpt-x", provider="openai", at=datetime(2026, 7, 1, tzinfo=timezone.utc))
    assert found.price.id == "openai/gpt-x:openai-api:global:2026-06-01"
    assert found.reasons == ()


def test_a_gap_before_a_scheduled_price_is_not_bridged(tmp_path):
    catalog = _catalog(tmp_path)
    gap = catalog.lookup_price(model="gem-gap", provider="google", at=_AFTER_END)
    assert gap.status == "unpriced"
    assert gap.price is None
    later = catalog.lookup_price(model="gem-gap", provider="google", at=datetime(2026, 10, 1, tzinfo=timezone.utc))
    assert later.status == "priced" and later.reasons == ()
    by_provider = catalog.snapshot.cost(
        provider="google", model="gem-gap", at=_AFTER_END, input_tokens=1, output_tokens=1,
    )
    assert by_provider.reasons == ("no_effective_price",)


def test_a_call_before_the_first_price_is_still_unpriced(tmp_path):
    catalog = _catalog(tmp_path)
    found = catalog.lookup_price(model="gem-old", provider="google", at=datetime(2025, 12, 1, tzinfo=timezone.utc))
    assert found.status == "unpriced"


def test_the_last_known_price_is_the_one_that_applied_most_recently(tmp_path):
    catalog = _catalog(tmp_path)
    found = catalog.lookup_price(
        model="gem-reg", provider="google", region="us-east-1",
        at=datetime(2026, 10, 1, tzinfo=timezone.utc),
    )
    assert found.reasons == ("price_ended",)
    assert found.price.region == "global"
    assert found.price.input_per_mtok == Decimal("0.8")


@pytest.mark.parametrize("provider", ["moonshotai", "unknown", "litellm"])
@pytest.mark.parametrize("fresh_tokens, expected", [(0, "0.165"), (20_000, "0.225")])
def test_kimi_k3_cache_tokens_are_not_charged_as_fresh_input(provider, fresh_tokens, expected):
    result = load_catalog().calculate_cost(
        model="kimi-k3", provider=provider, channel="moonshot-api",
        at=datetime(2026, 10, 29, tzinfo=timezone.utc),
        usage=ModelUsage(
            input_tokens=100_000 + fresh_tokens, output_tokens=0,
            cache_read_tokens=50_000, cache_write_5m_tokens=50_000,
        ),
    )
    assert result.status == "priced"
    assert result.cost_usd == Decimal(expected)
    assert result.components.input == Decimal(fresh_tokens) * Decimal("0.000003")


def test_mistral_large_4_preserves_batch_discount_at_corrected_rates():
    result = load_catalog().price(
        model="mistral-large-4", channel="mistral-api",
        at=datetime(2026, 10, 29, tzinfo=timezone.utc),
        input_tokens=1_000_000, output_tokens=1_000_000, batch=True,
    )
    assert result.status == "priced"
    assert result.cost_usd == Decimal("1.40")


def test_kimi_k3_preserves_history_and_leaves_one_hour_writes_partial():
    catalog = load_catalog()
    before = catalog.price(
        model="kimi-k3", channel="moonshot-api",
        at=datetime(2026, 10, 28, tzinfo=timezone.utc),
        input_tokens=100_000, output_tokens=0,
        cache_read_tokens=50_000, cache_write_5m_tokens=50_000,
    )
    assert before.status == "partial"
    # Preserve the previously published cache-read double charge in this interval.
    assert before.cost_usd == Decimal("0.315")
    assert "cache_write_5m_rate_unavailable" in before.reasons
    one_hour = catalog.price(
        model="kimi-k3", channel="moonshot-api",
        at=datetime(2026, 10, 29, tzinfo=timezone.utc),
        input_tokens=100_000, output_tokens=0, cache_write_1h_tokens=50_000,
    )
    assert one_hour.status == "partial"
    assert "cache_write_1h_rate_unavailable" in one_hour.reasons


@pytest.mark.parametrize("day, expected", [(28, "10.50"), (29, "19.00")])
def test_gemini_video_reconciles_applied_history_then_corrects_output_rate(day, expected):
    result = load_catalog().price(
        model="gemini-omni-1.1-flash", channel="google-api",
        at=datetime(2026, 10, day, tzinfo=timezone.utc),
        input_tokens=1_000_000, output_tokens=1_000_000,
    )
    assert result.status == "priced"
    assert result.cost_usd == Decimal(expected)
