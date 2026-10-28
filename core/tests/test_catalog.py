from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from metergraph_core import (
    CatalogError,
    CatalogSnapshot,
    counts_cache_read_in_input,
    direct_channel_for_provider,
    load_catalog,
    normalize_provider,
    parse_catalog,
)

LOADED = load_catalog()
VERSION = LOADED.version
DOC = LOADED.document
SNAPSHOT = LOADED.snapshot


def _at(day: str) -> datetime:
    return datetime.fromisoformat(day).replace(tzinfo=timezone.utc)


def test_prices_yaml_parses():
    assert VERSION
    assert DOC["models"]
    assert LOADED.currency == "USD"
    assert LOADED.pricing_verified_at.isoformat() == "2026-10-27"


def test_openrouter_cache_reads_are_not_charged_as_fresh_input():
    result = SNAPSHOT.cost(
        provider="openrouter",
        model="openai/gpt-4o-mini",
        at=_at("2026-10-20"),
        input_tokens=1_000,
        output_tokens=0,
        cache_read_tokens=1_000,
    )

    assert result.status == "priced"
    assert result.cost_usd == Decimal("0.00007500")


def test_current_catalog_prices_more_than_one_thousand_provider_model_pairs():
    at = _at("2026-10-20")
    pairs = {
        (normalize_provider(alias["provider"]), alias["alias"])
        for model in DOC["models"]
        for alias in model.get("aliases") or []
        if SNAPSHOT.cost(
            provider=alias["provider"],
            model=alias["alias"],
            at=at,
            input_tokens=1,
            output_tokens=1,
        ).status
        == "priced"
    }

    # Includes claude-3-5-haiku-20241022 at its last known price.
    assert len(pairs) == 1119
    assert sum(provider == "vercel" for provider, _ in pairs) == 187
    assert sum(provider == "openrouter" for provider, _ in pairs) == 212


def test_resolve_price_by_deployment_identity_and_channel():
    resolved = SNAPSHOT.resolve_price(
        model="  OPENAI/GPT-5.6-LUNA  ",
        channel=" VERCEL-AI-GATEWAY ",
        at=_at("2026-08-01"),
    )

    assert resolved is not None
    assert resolved.canonical_model == "openai/gpt-5.6-luna"
    assert resolved.price.id == (
        "openai/gpt-5.6-luna:vercel-ai-gateway:global:2026-07-30"
    )
    assert resolved.price.input_per_mtok == Decimal("0.20")
    assert resolved.price.source_url == "https://ai-gateway.vercel.sh/v1/models"
    assert resolved.rules["input_includes_cache_read"] is True
    with pytest.raises(TypeError):
        resolved.rules["input_includes_cache_read"] = False
    with pytest.raises(TypeError):
        resolved.price.rules["long_context"] = {}


def test_resolve_price_does_not_guess_another_channel():
    # Direct-only pricing must not cross channels.
    direct = SNAPSHOT.resolve_price(
        model="openai/gpt-5.6-cyber", channel="openai-api", at=_at("2026-10-02")
    )
    assert direct is not None
    assert SNAPSHOT.resolve_price(
        model="openai/gpt-5.6-cyber",
        channel="vercel-ai-gateway",
        at=_at("2026-10-02"),
    ) is None
    # A gateway route does not reach back before its own window either.
    assert SNAPSHOT.resolve_price(
        model="openai/gpt-4o",
        channel="vercel-ai-gateway",
        at=_at("2026-08-01"),
    ) is None


def test_gemini_direct_price_changes_on_its_effective_date():
    before = SNAPSHOT.resolve_price(
        model="gemini-3.6-flash", channel="google-api", at=_at("2026-08-19")
    )
    after = SNAPSHOT.resolve_price(
        model="gemini-3.6-flash", channel="google-api", at=_at("2026-08-20")
    )

    assert before is not None
    assert after is not None
    assert before.price.input_per_mtok == Decimal("1.50")
    assert after.price.input_per_mtok == Decimal("0.75")


def test_cache_write_tiers_use_their_respective_rates():
    result = SNAPSHOT.cost(
        provider="anthropic",
        model="claude-haiku-4-5",
        at=_at("2026-09-11"),
        input_tokens=0,
        output_tokens=0,
        cache_write_5m_tokens=1_000_000,
        cache_write_1h_tokens=1_000_000,
    )

    assert result.cost_usd == Decimal("3.25000000")


@pytest.mark.parametrize(
    "provider,model,expected_channel,expected_cost",
    [
        ("openai", "gpt-5.2", "openai-api", Decimal("15.75000000")),
        ("anthropic", "claude-opus-4-6", "anthropic-api", Decimal("30.00000000")),
        ("vertex-ai", "claude-opus-4-6", "google-vertex-ai", Decimal("33.00000000")),
    ],
)
def test_representative_models_resolve_on_their_observed_billing_channels(
    provider, model, expected_channel, expected_cost
):
    result = SNAPSHOT.cost(
        provider=provider,
        model=model,
        at=_at("2026-09-11"),
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )

    assert result.status == "priced"
    assert f":{expected_channel}:" in result.price_id
    assert result.cost_usd == expected_cost


@pytest.mark.parametrize(
    "model,channel,input_rate,output_rate",
    [
        ("gpt-4o-mini", "openai-api", "0.15", "0.60"),
        ("gpt-5.4-mini", "openai-api", "0.75", "4.50"),
        ("claude-opus-4-8", "anthropic-api", "5.00", "25.00"),
        ("claude-haiku-4-5", "anthropic-api", "1.00", "5.00"),
        ("anthropic/claude-opus-5", "vercel-ai-gateway", "5.00", "25.00"),
        ("anthropic/claude-sonnet-5", "vercel-ai-gateway", "2.00", "10.00"),
        ("anthropic/claude-3-haiku", "vercel-ai-gateway", "0.25", "1.25"),
        ("openai/gpt-5.6-sol", "vercel-ai-gateway", "5.00", "30.00"),
        ("openai/gpt-5.6-luna", "vercel-ai-gateway", "0.20", "1.20"),
        ("gpt-5-mini", "openai-api", "0.25", "2.00"),
        ("openai/gpt-5-mini", "vercel-ai-gateway", "0.25", "2.00"),
        ("openai/gpt-5.6-terra", "vercel-ai-gateway", "2.00", "12.00"),
        ("google/gemini-3.1-pro-preview", "vercel-ai-gateway", "2.00", "12.00"),
        ("gemini-3.6-flash", "google-api", "0.75", "3.75"),
        ("google/gemini-3.6-flash", "vercel-ai-gateway", "1.50", "7.50"),
        ("google/gemini-3.7-flash", "vercel-ai-gateway", "0.75", "3.75"),
        ("google/gemma-4-31b-it", "vercel-ai-gateway", "0.14", "0.40"),
        ("meta/muse-glimmer-30b", "vercel-ai-gateway", "0.35", "1.50"),
        ("thinkingmachines/inkling", "vercel-ai-gateway", "1.00", "4.05"),
        ("zai/glm-5.2", "vercel-ai-gateway", "0.80", "2.55"),
        ("moonshotai/kimi-k3", "vercel-ai-gateway", "2.90", "14.00"),
        ("minimax/minimax-m3", "vercel-ai-gateway", "0.30", "1.20"),
        ("nvidia/nemotron-3-super-120b-a12b", "vercel-ai-gateway", "0.15", "0.65"),
        ("deepseek/deepseek-v4-pro", "vercel-ai-gateway", "0.435", "0.87"),
        ("deepseek/deepseek-v4-flash", "vercel-ai-gateway", "0.14", "0.28"),
        ("fireworks:accounts/fireworks/models/glm-5p2", "fireworks-api", "1.40", "4.40"),
        ("fireworks:accounts/fireworks/models/qwen3p7-plus", "fireworks-api", "0.40", "1.60"),
        ("xai/grok-4.1-fast-reasoning", "vercel-ai-gateway", "0.20", "0.50"),
        # Vercel's current Grok creator namespace (spacexai/*).
        ("spacexai/grok-4.1-fast-non-reasoning", "vercel-ai-gateway", "0.20", "0.50"),
        ("spacexai/grok-4.6", "vercel-ai-gateway", "2.00", "6.00"),
    ],
)
def test_pipeline_catalog_compatibility(model, channel, input_rate, output_rate):
    resolved = SNAPSHOT.resolve_price(
        model=model,
        channel=channel,
        at=_at("2026-08-25"),
    )

    assert resolved is not None
    assert resolved.price.input_per_mtok == Decimal(input_rate)
    assert resolved.price.output_per_mtok == Decimal(output_rate)


@pytest.mark.parametrize(
    "model,channel,input_rate,output_rate,cache_read",
    [
        ("claude-haiku-5-5", "anthropic-api", "0.10", "0.50", "0.01"),
        ("anthropic/claude-haiku-5.5", "vercel-ai-gateway", "0.10", "0.50", "0.01"),
        ("gpt-6.1-sol", "openai-api", "2.00", "10.00", "0.10"),
        ("gemini-4-argon", "google-api", "2.00", "10.00", "0.10"),
        ("google/gemini-4-argon", "vercel-ai-gateway", "2.00", "10.00", "0.10"),
        ("mistral-large-4", "mistral-api", "0.68", "2.09", "0.07"),
        ("mistral/mistral-large-4", "vercel-ai-gateway", "0.68", "2.09", "0.07"),
        ("muse-wonder-1.0", "meta-api", "4.00", "20.00", "0.20"),
        ("meta/muse-wonder-1.0", "vercel-ai-gateway", "4.00", "20.00", "0.20"),
    ],
)
def test_october_catalog_refresh_prices_current_models(
    model, channel, input_rate, output_rate, cache_read
):
    resolved = SNAPSHOT.resolve_price(model=model, channel=channel, at=_at("2026-10-13"))

    assert resolved is not None
    assert resolved.price.input_per_mtok == Decimal(input_rate)
    assert resolved.price.output_per_mtok == Decimal(output_rate)
    assert resolved.price.cache_read_per_mtok == (
        None if cache_read is None else Decimal(cache_read)
    )


def test_mistral_large_4_launch_sale_and_standard_price_boundaries():
    launch = SNAPSHOT.resolve_price(
        model="mistral-large-4", channel="mistral-api", at=_at("2026-10-06")
    )
    middle = SNAPSHOT.resolve_price(
        model="mistral-large-4", channel="mistral-api", at=_at("2026-10-14")
    )
    last_sale_day = SNAPSHOT.resolve_price(
        model="mistral-large-4", channel="mistral-api", at=_at("2026-10-19")
    )
    restored = SNAPSHOT.resolve_price(
        model="mistral-large-4", channel="mistral-api", at=_at("2026-10-20")
    )

    assert launch is not None and middle is not None
    assert last_sale_day is not None and restored is not None
    assert launch.price.input_per_mtok == Decimal("0.68")
    assert launch.price.output_per_mtok == Decimal("2.09")
    assert launch.price.cache_read_per_mtok == Decimal("0.07")
    assert launch.price.batch_input_per_mtok == Decimal("0.34")
    assert launch.price.batch_output_per_mtok == Decimal("1.045")
    assert middle.price.input_per_mtok == Decimal("0.68")
    assert last_sale_day.price.input_per_mtok == Decimal("0.68")
    assert restored.price.input_per_mtok == Decimal("1.36")
    assert restored.price.output_per_mtok == Decimal("4.18")
    assert restored.price.cache_read_per_mtok == Decimal("0.14")
    assert restored.price.batch_input_per_mtok == Decimal("0.68")
    assert restored.price.batch_output_per_mtok == Decimal("2.09")

    gateway_sale = SNAPSHOT.resolve_price(
        model="mistral/mistral-large-4",
        channel="vercel-ai-gateway",
        at=_at("2026-10-19"),
    )
    gateway_restored = SNAPSHOT.resolve_price(
        model="mistral/mistral-large-4",
        channel="vercel-ai-gateway",
        at=_at("2026-10-20"),
    )
    assert gateway_sale is not None and gateway_restored is not None
    assert gateway_sale.price.input_per_mtok == Decimal("0.68")
    assert gateway_restored.price.input_per_mtok == Decimal("1.36")
    assert gateway_restored.price.output_per_mtok == Decimal("4.18")
    assert gateway_restored.price.cache_read_per_mtok == Decimal("0.14")


def test_mythos_5_1_is_priced_for_direct_anthropic_access():
    resolved = SNAPSHOT.resolve_price(
        model="claude-mythos-5-1",
        channel="anthropic-api",
        at=_at("2026-10-15"),
    )

    assert resolved is not None
    assert resolved.canonical_model == "anthropic/claude-mythos-5.1"
    assert resolved.price.input_per_mtok == Decimal("10")
    assert resolved.price.output_per_mtok == Decimal("50")
    assert resolved.price.cache_read_per_mtok == Decimal("0.25")
    assert resolved.price.cache_write_5m_per_mtok == Decimal("12.5")
    assert resolved.price.cache_write_1h_per_mtok == Decimal("20")
    assert resolved.price.batch_input_per_mtok == Decimal("5")
    assert resolved.price.batch_output_per_mtok == Decimal("25")


@pytest.mark.parametrize("model", ["gpt-5.2-pro", "gpt-5.2-pro-2025-12-11"])
def test_gpt_5_2_pro_is_priced_for_direct_openai_access(model):
    resolved = SNAPSHOT.resolve_price(
        model=model,
        channel="openai-api",
        at=_at("2026-10-15"),
    )

    assert resolved is not None
    assert resolved.canonical_model == "openai/gpt-5.2-pro"
    assert resolved.price.input_per_mtok == Decimal("21")
    assert resolved.price.output_per_mtok == Decimal("168")
    assert resolved.price.cache_read_per_mtok is None
    assert resolved.price.batch_input_per_mtok is None
    assert resolved.price.batch_output_per_mtok is None


@pytest.mark.parametrize(
    (
        "model",
        "input_rate",
        "output_rate",
        "cache_read_rate",
        "varies_by_provider",
    ),
    [
        ("meta/muse-spark-open", "1.25", "4.25", "0.15", False),
        ("nvidia/nemotron-3.5-super-vl", "0.20", "0.50", "0.04", True),
        ("liquid/lfm2.5-24b-a2b", "0.05", "0.70", "0.025", False),
        ("reflection/beam-free", "0", "0", None, False),
        ("nvidia/nemotron-3.5-lightning-free", "0", "0", None, False),
    ],
)
def test_october_16_gateway_catalog_additions(
    model, input_rate, output_rate, cache_read_rate, varies_by_provider
):
    resolved = SNAPSHOT.resolve_price(
        model=model,
        channel="vercel-ai-gateway",
        at=_at("2026-10-16"),
    )

    assert resolved is not None
    assert resolved.price.input_per_mtok == Decimal(input_rate)
    assert resolved.price.output_per_mtok == Decimal(output_rate)
    assert resolved.price.cache_read_per_mtok == (
        None if cache_read_rate is None else Decimal(cache_read_rate)
    )
    assert bool(resolved.rules.get("varies_by_provider")) is varies_by_provider


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        ("vercel", "typesafe-ai/jev-1.13.0"),
        ("unknown", "typesafe/jev-1.13.0"),
        ("unknown", "jev-1.13.0"),
    ],
)
def test_jev_113_gateway_aliases_price_input_only_requests(provider, model):
    result = SNAPSHOT.cost(
        provider=provider,
        model=model,
        at=_at("2026-09-15"),
        input_tokens=100_000,
        output_tokens=1_000,
    )

    assert result.status == "priced"
    assert result.canonical_model == "typesafe-ai/jev-1.13.0"
    assert result.cost_usd == Decimal("0.00400000")
    assert result.missing_dimensions == ()


def test_deepseek_v4_1_flash_gateway_reprice_preserves_history_and_peak_windows():
    before = SNAPSHOT.resolve_price(
        model="deepseek/deepseek-v4.1-flash",
        channel="vercel-ai-gateway",
        at=datetime(2026, 10, 18, 2, tzinfo=timezone.utc),
    )
    peak = LOADED.price(
        model="deepseek/deepseek-v4.1-flash",
        channel="vercel-ai-gateway",
        at=datetime(2026, 10, 19, 2, tzinfo=timezone.utc),
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )
    off_peak = LOADED.price(
        model="deepseek/deepseek-v4.1-flash",
        channel="vercel-ai-gateway",
        at=datetime(2026, 10, 19, 5, tzinfo=timezone.utc),
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )

    assert before is not None
    assert before.price.input_per_mtok == Decimal("0.30")
    assert before.price.output_per_mtok == Decimal("1.20")
    assert before.price.cache_read_per_mtok == Decimal("0.007")
    assert peak.status == off_peak.status == "priced"
    assert peak.cost_usd == Decimal("0.75000000")
    assert off_peak.cost_usd == Decimal("0.37500000")


@pytest.mark.parametrize(
    (
        "model",
        "channel",
        "at",
        "input_rate",
        "output_rate",
        "cache_read_rate",
    ),
    [
        (
            "deepseek/deepseek-v4.1-flash-1017",
            "vercel-ai-gateway",
            "2026-10-17",
            "0.15",
            "0.60",
            "0.003",
        ),
        (
            "deepseek/deepseek-v4.1-flash-1017-fast",
            "vercel-ai-gateway",
            "2026-10-19",
            "0.60",
            "2.40",
            "0.014",
        ),
        (
            "elorian/glimpse",
            "vercel-ai-gateway",
            "2026-10-19",
            "0.60",
            "3.00",
            "0.06",
        ),
        (
            "inception/mercury-3",
            "vercel-ai-gateway",
            "2026-10-19",
            "0.10",
            "0.35",
            "0.01",
        ),
        (
            "minimax/minimax-m3.1-flash",
            "minimax-api",
            "2026-10-19",
            "0.15",
            "0.60",
            "0.015",
        ),
    ],
)
def test_october_19_catalog_additions(
    model, channel, at, input_rate, output_rate, cache_read_rate
):
    resolved = SNAPSHOT.resolve_price(model=model, channel=channel, at=_at(at))

    assert resolved is not None
    assert resolved.price.input_per_mtok == Decimal(input_rate)
    assert resolved.price.output_per_mtok == Decimal(output_rate)
    assert resolved.price.cache_read_per_mtok == Decimal(cache_read_rate)


def test_minimax_m3_1_flash_gateway_promotion_ends_after_october_22():
    free = SNAPSHOT.resolve_price(
        model="minimax/minimax-m3.1-flash",
        channel="vercel-ai-gateway",
        at=_at("2026-10-22"),
    )
    standard = SNAPSHOT.resolve_price(
        model="minimax/minimax-m3.1-flash",
        channel="vercel-ai-gateway",
        at=_at("2026-10-23"),
    )

    assert free is not None and standard is not None
    assert free.price.input_per_mtok == Decimal("0")
    assert free.price.output_per_mtok == Decimal("0")
    assert standard.price.input_per_mtok == Decimal("0.15")
    assert standard.price.output_per_mtok == Decimal("0.60")
    assert standard.price.cache_read_per_mtok == Decimal("0.015")


def test_minimax_m3_1_flash_promotion_prices_cache_reads_at_zero():
    result = LOADED.price(
        model="minimax/minimax-m3.1-flash",
        channel="vercel-ai-gateway",
        at=_at("2026-10-22"),
        input_tokens=1_000_000,
        output_tokens=0,
        cache_read_tokens=1_000_000,
    )

    assert result.status == "priced"
    assert result.cost_usd == Decimal("0")


def test_deepseek_v4_1_flash_1017_fast_has_fixed_gateway_pricing():
    costs = {
        LOADED.price(
            model="deepseek/deepseek-v4.1-flash-1017-fast",
            channel="vercel-ai-gateway",
            at=datetime(2026, 10, 19, hour, tzinfo=timezone.utc),
            input_tokens=1_000_000,
            output_tokens=1_000_000,
        ).cost_usd
        for hour in (2, 5)
    }

    assert costs == {Decimal("3.00000000")}


def test_beam_is_recognized_but_unpriced_on_the_mixed_provider_gateway():
    assert (
        LOADED.canonical_model_id("reflection", "reflection/beam")
        == "reflection/beam"
    )
    assert SNAPSHOT.resolve_price(
        model="reflection/beam",
        channel="vercel-ai-gateway",
        at=_at("2026-10-16"),
    ) is None
    result = SNAPSHOT.cost(
        provider="reflection",
        model="reflection/beam",
        at=_at("2026-10-16"),
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )
    assert result.status == "unpriced"
    assert result.canonical_model == "reflection/beam"
    assert result.cost_usd is None
    assert "no_effective_price" in result.reasons


def test_sonnet_5_5_cache_read_cut_preserves_history():
    before = SNAPSHOT.resolve_price(
        model="claude-sonnet-5-5", channel="anthropic-api", at=_at("2026-10-06")
    )
    after = SNAPSHOT.resolve_price(
        model="claude-sonnet-5-5", channel="anthropic-api", at=_at("2026-10-07")
    )

    assert before is not None and after is not None
    assert before.price.cache_read_per_mtok == Decimal("0.20")
    assert after.price.cache_read_per_mtok == Decimal("0.10")


def test_gemini_4_argon_introductory_price_has_a_future_end_date():
    introductory = SNAPSHOT.resolve_price(
        model="gemini-4-argon", channel="google-api", at=_at("2026-11-30")
    )
    standard = SNAPSHOT.resolve_price(
        model="gemini-4-argon", channel="google-api", at=_at("2026-12-01")
    )

    assert introductory is not None and standard is not None
    assert introductory.price.input_per_mtok == Decimal("2.00")
    assert introductory.price.output_per_mtok == Decimal("10.00")
    assert standard.price.input_per_mtok == Decimal("4.00")
    assert standard.price.output_per_mtok == Decimal("20.00")


def test_haiku_5_5_records_its_long_context_tier():
    resolved = SNAPSHOT.resolve_price(
        model="claude-haiku-5-5", channel="anthropic-api", at=_at("2026-10-13")
    )

    assert resolved is not None
    assert resolved.rules["long_context"] == {
        "threshold": 100000,
        "input_multiplier": 5,
        "output_multiplier": 5,
        "threshold_counts_cached_tokens": True,
    }


def test_haiku_5_5_direct_long_context_counts_cached_prompt_tokens():
    result = SNAPSHOT.cost(
        provider="anthropic",
        model="claude-haiku-5-5",
        at=_at("2026-10-13"),
        input_tokens=20_000,
        output_tokens=0,
        cache_read_tokens=130_000,
    )

    assert result.status == "priced"
    assert result.cost_usd == Decimal("0.01650000")


def test_haiku_5_5_direct_long_context_counts_cache_writes():
    result = SNAPSHOT.cost(
        provider="anthropic",
        model="claude-haiku-5-5",
        at=_at("2026-10-13"),
        input_tokens=20_000,
        output_tokens=0,
        cache_write_5m_tokens=90_000,
    )

    assert result.status == "priced"
    assert result.cost_usd == Decimal("0.06625000")


def test_haiku_5_5_gateway_long_context_does_not_double_count_cached_tokens():
    result = SNAPSHOT.cost(
        provider="anthropic",
        model="anthropic/claude-haiku-5.5",
        at=_at("2026-10-13"),
        input_tokens=90_000,
        output_tokens=0,
        cache_read_tokens=20_000,
    )

    assert result.status == "priced"
    assert result.cost_usd == Decimal("0.00720000")


def test_muse_wonder_direct_cached_input_is_not_billed_twice():
    result = SNAPSHOT.cost(
        provider="meta",
        model="muse-wonder-1.0",
        at=_at("2026-10-13"),
        input_tokens=1_000_000,
        output_tokens=0,
        cache_read_tokens=1_000_000,
    )

    assert result.status == "priced"
    assert result.cost_usd == Decimal("0.20000000")


def test_current_spacexai_grok_identities_resolve_at_vercel_prices():
    at = _at("2026-08-25")

    non_reasoning = SNAPSHOT.resolve_price(
        model="spacexai/grok-4.1-fast-non-reasoning",
        channel="vercel-ai-gateway",
        at=at,
    )
    assert non_reasoning is not None
    assert non_reasoning.canonical_model == "xai/grok-4.1-fast-non-reasoning"
    assert non_reasoning.price.input_per_mtok == Decimal("0.20")
    assert non_reasoning.price.output_per_mtok == Decimal("0.50")
    assert non_reasoning.price.cache_read_per_mtok == Decimal("0.05")
    assert non_reasoning.rules["input_includes_cache_read"] is True

    grok46 = SNAPSHOT.resolve_price(
        model="spacexai/grok-4.6", channel="vercel-ai-gateway", at=at
    )
    assert grok46 is not None
    assert grok46.canonical_model == "xai/grok-4.6"
    assert grok46.price.id == "xai/grok-4.6:vercel-ai-gateway:global:2026-08-01"
    assert grok46.price.input_per_mtok == Decimal("2.00")
    assert grok46.price.output_per_mtok == Decimal("6.00")
    assert grok46.price.cache_read_per_mtok == Decimal("0.50")
    assert grok46.price.source_url.endswith("/grok-4.6/providers")

    # The historical xai/* reasoning identity is unchanged and stays distinct.
    reasoning = SNAPSHOT.resolve_price(
        model="xai/grok-4.1-fast-reasoning", channel="vercel-ai-gateway", at=at
    )
    assert reasoning is not None
    assert reasoning.canonical_model == "xai/grok-4.1-fast-reasoning"
    assert reasoning.price.input_per_mtok == Decimal("0.20")

    # Temporal accuracy: grok-4.6 has no price before its 2026-08-01 launch.
    assert SNAPSHOT.resolve_price(
        model="spacexai/grok-4.6",
        channel="vercel-ai-gateway",
        at=_at("2026-07-31"),
    ) is None


@pytest.mark.parametrize(
    "field,value",
    [("currency", "EUR"), ("currency", None), ("pricing_verified_at", "soon"),
     ("pricing_verified_at", None)],
)
def test_catalog_metadata_is_required_and_validated(field, value):
    doc = {"version": "test", "currency": "USD", "pricing_verified_at": "2026-08-24", "models": []}
    if value is None:
        doc.pop(field)
    else:
        doc[field] = value

    with pytest.raises(CatalogError):
        parse_catalog(doc)


def test_price_source_is_required():
    doc = {
        "version": "test",
        "currency": "USD",
        "pricing_verified_at": "2026-08-24",
        "models": [
            {
                "canonical_id": "example/model",
                "aliases": [
                    {"provider": "example", "alias": "model", "channel": "api"}
                ],
                "prices": [
                    {
                        "channel": "api",
                        "effective_from": "2026-08-24",
                        "input_per_mtok": 1,
                        "output_per_mtok": 2,
                    }
                ],
            }
        ],
    }

    with pytest.raises(CatalogError, match="source_url"):
        parse_catalog(doc)


@pytest.mark.parametrize(
    "publisher, channel",
    [
        ("openai", "openai-api"),
        ("google", "google-api"),
        ("google", "google-vertex-ai"),
        ("deepseek", "deepseek-api"),
        ("xai", "xai-api"),
    ],
)
def test_cache_reads_come_out_of_input_where_the_publisher_counts_them(
    publisher, channel
):
    """A catalog author has no way to know which providers report cached tokens
    inside the input total, so the publisher and channel answer together when
    the row does not."""
    assert counts_cache_read_in_input(publisher, channel, {}) is True


@pytest.mark.parametrize(
    "publisher, channel, rules",
    [
        ("openai", "openai-api", {"input_includes_cache_read": False}),
        ("anthropic", "anthropic-api", {}),
        ("anthropic", "aws-bedrock", {}),
        ("openai", "vercel-ai-gateway", {}),
        ("google", None, {}),
        (None, "google-api", {}),
    ],
)
def test_a_stated_rule_and_an_unlisted_pair_keep_input_billable(
    publisher, channel, rules
):
    assert counts_cache_read_in_input(publisher, channel, rules) is False


def test_a_gateway_row_can_claim_the_rule_for_the_provider_behind_it():
    assert counts_cache_read_in_input(
        "openai", "vercel-ai-gateway", {"input_includes_cache_read": True}
    ) is True


def test_vertex_serves_two_publishers_and_only_google_counts_cache_in_input():
    """google-vertex-ai is the one mixed-publisher channel in the catalog.
    Claude on Vertex keeps Anthropic's usage shape, where input_tokens already
    excludes cache reads, so deducting there overcharges in the opposite
    direction from the defect this default removes."""
    assert counts_cache_read_in_input("google", "google-vertex-ai", {}) is True
    assert counts_cache_read_in_input("anthropic", "google-vertex-ai", {}) is False


def test_claude_on_vertex_bills_cache_reads_on_top_of_input():
    """The bundled catalog's only Anthropic row on a Google channel. At 1M
    input including 1M cache reads it is 5.50 + 0.55, not 0.55."""
    result = SNAPSHOT.cost(
        provider="vertex-ai",
        model="claude-opus-4-6",
        at=_at("2026-09-18"),
        input_tokens=1_000_000,
        output_tokens=0,
        cache_read_tokens=1_000_000,
    )
    assert result.status == "priced"
    assert result.cost_usd == Decimal("6.05")


@pytest.mark.parametrize("unusable", ["not-a-number", -1, True, [1]])
def test_an_unusable_cache_read_count_is_reported_not_swallowed(unusable):
    """A count the caller supplied that cannot be read is not the same as one it
    omitted. Reading it as absent bills the cached tokens at the full input rate
    and says nothing, so a consumer trusting a fully priced result cannot tell a
    silent over-bill from a price."""
    result = SNAPSHOT.cost(
        provider="openai",
        model="gpt-5.6-luna",
        at=_at("2026-07-15"),
        input_tokens=100_000,
        output_tokens=0,
        cache_read_tokens=unusable,
    )
    assert result.status == "partial"
    assert "unusable_cache_read_tokens" in result.reasons


def test_an_omitted_cache_read_count_stays_silent():
    """Omitting a field is legitimate and must keep the behaviour it had."""
    result = SNAPSHOT.cost(
        provider="openai",
        model="gpt-5.6-luna",
        at=_at("2026-07-15"),
        input_tokens=100_000,
        output_tokens=0,
    )
    assert result.status == "priced"
    assert not [r for r in result.reasons if r.startswith("unusable_")]


def test_an_unusable_cache_write_split_does_not_fall_back_to_the_aggregate():
    """An absent five-minute split falls back to `cache_write_tokens`. A stated
    split that cannot be read is not an invitation to substitute a different
    field, which would bill a number the caller never gave."""
    result = SNAPSHOT.cost(
        provider="openai",
        model="gpt-5.6-luna",
        at=_at("2026-07-15"),
        input_tokens=100_000,
        output_tokens=0,
        cache_write_5m_tokens="oops",
        cache_write_tokens=1_000_000,
    )
    assert result.status == "partial"
    assert "unusable_cache_write_5m_tokens" in result.reasons
    assert "unusable_cache_write_tokens" not in result.reasons


def test_an_unusable_count_bills_as_nothing_cached_rather_than_guessing():
    """The arithmetic is unchanged; only the reporting is new. A guess would be
    worse than a flagged zero."""
    flagged = SNAPSHOT.cost(
        provider="openai", model="gpt-5.6-luna", at=_at("2026-07-15"),
        input_tokens=100_000, output_tokens=0, cache_read_tokens="oops",
    )
    absent = SNAPSHOT.cost(
        provider="openai", model="gpt-5.6-luna", at=_at("2026-07-15"),
        input_tokens=100_000, output_tokens=0,
    )
    assert flagged.cost_usd == absent.cost_usd


def test_openai_cache_read_included_in_input():
    result = SNAPSHOT.cost(
        provider="openai",
        model="gpt-5.6-luna",
        at=_at("2026-07-15"),
        input_tokens=100_000,
        output_tokens=0,
        cache_read_tokens=50_000,
    )
    assert result.status == "priced"
    assert result.canonical_model == "openai/gpt-5.6-luna"
    assert result.cost_usd == Decimal("0.05") + Decimal("0.005")


def test_gateway_luna_price_drop_does_not_reprice_history():
    result = SNAPSHOT.cost(
        provider="openai",
        model="openai/gpt-5.6-luna",
        at=_at("2026-07-15"),
        input_tokens=100_000,
        output_tokens=100_000,
    )
    assert result.status == "priced"
    assert result.cost_usd == Decimal("0.70")


def test_partial_price_reports_uncaptured_fees():
    result = SNAPSHOT.cost(
        provider="perplexity-ai",
        model="sonar",
        at=_at("2026-08-10"),
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )
    assert result.canonical_model == "perplexity/sonar"
    assert result.price_id == "perplexity/sonar:perplexity-api:global:2025-04-18"
    assert result.cost_usd == Decimal("2.00000000")
    assert result.status == "partial"
    assert result.reasons == ("uncaptured_fees",)


@pytest.mark.parametrize(
    ("provider", "model", "expected_canonical", "expected_channel", "expected_cost"),
    [
        (
            "openai",
            "gpt-5.6-luna",
            "openai/gpt-5.6-luna",
            "openai-api",
            Decimal("0.14"),
        ),
        (
            "openai",
            "openai/gpt-5.6-luna",
            "openai/gpt-5.6-luna",
            "vercel-ai-gateway",
            Decimal("0.14"),
        ),
        (
            "anthropic",
            "claude-haiku-4-5",
            "anthropic/claude-haiku-4.5",
            "anthropic-api",
            Decimal("0.60"),
        ),
        (
            "anthropic",
            "anthropic/claude-haiku-4.5",
            "anthropic/claude-haiku-4.5",
            "vercel-ai-gateway",
            Decimal("0.60"),
        ),
        (
            "anthropic",
            "claude-opus-5",
            "anthropic/claude-opus-5",
            "anthropic-api",
            Decimal("3.00"),
        ),
        (
            "anthropic",
            "anthropic/claude-opus-5",
            "anthropic/claude-opus-5",
            "vercel-ai-gateway",
            Decimal("3.00"),
        ),
        (
            "openai",
            "gpt-5.4-mini",
            "openai/gpt-5.4-mini",
            "openai-api",
            Decimal("0.525"),
        ),
        (
            "openai",
            "openai/gpt-5.4-mini",
            "openai/gpt-5.4-mini",
            "vercel-ai-gateway",
            Decimal("0.525"),
        ),
        (
            "google",
            "google/gemma-4-26b-a4b-it",
            "google/gemma-4-26b-a4b-it",
            "vercel-ai-gateway",
            Decimal("0.075"),
        ),
        (
            "nvidia",
            "nvidia/nemotron-3-super-120b-a12b",
            "nvidia/nemotron-3-super-120b-a12b",
            "vercel-ai-gateway",
            Decimal("0.080"),
        ),
        (
            "meta",
            "meta/muse-spark-1.1",
            "meta/muse-spark-1.1",
            "vercel-ai-gateway",
            Decimal("0.550"),
        ),
    ],
)
def test_design_partner_models_are_priced(
    provider, model, expected_canonical, expected_channel, expected_cost
):
    result = SNAPSHOT.cost(
        provider=provider,
        model=model,
        at=_at("2026-08-17"),
        input_tokens=100_000,
        output_tokens=100_000,
    )
    assert result.status == "priced"
    assert result.canonical_model == expected_canonical
    assert f":{expected_channel}:" in result.price_id
    assert result.cost_usd == expected_cost


@pytest.mark.parametrize(
    (
        "provider",
        "model",
        "expected_canonical",
        "expected_channel",
        "expected_cost",
        "expected_status",
        "expected_reasons",
    ),
    [
        (
            "perplexity-ai",
            "sonar",
            "perplexity/sonar",
            "perplexity-api",
            Decimal("2.00"),
            "partial",
            ("uncaptured_fees",),
        ),
        (
            "openai",
            "gpt-5.4-nano",
            "openai/gpt-5.4-nano",
            "openai-api",
            Decimal("1.45"),
            "priced",
            (),
        ),
        (
            "vertex-ai",
            "gemini-3.5-flash",
            "google/gemini-3.5-flash",
            "google-vertex-ai",
            Decimal("10.50"),
            "priced",
            (),
        ),
        (
            "openai",
            "gpt-5.4",
            "openai/gpt-5.4",
            "openai-api",
            Decimal("27.50"),
            "priced",
            (),
        ),
        (
            "deepseek",
            "deepseek-v4-flash",
            "deepseek/v4-flash",
            "deepseek-api",
            Decimal("0.42"),
            "priced",
            (),
        ),
        (
            "anthropic",
            "claude-opus-4-8",
            "anthropic/claude-opus-4.8",
            "anthropic-api",
            Decimal("30.00"),
            "priced",
            (),
        ),
        (
            "x-ai",
            "grok-4.3",
            "xai/grok-4.3",
            "xai-api",
            Decimal("7.50"),
            "priced",
            (),
        ),
        (
            "vertex-ai",
            "gemini-3.1-pro-preview",
            "google/gemini-3.1-pro-preview",
            "google-vertex-ai",
            Decimal("22.00"),
            "priced",
            (),
        ),
        (
            "deepseek",
            "deepseek-v4-pro",
            "deepseek/v4-pro",
            "deepseek-api",
            Decimal("1.305"),
            "priced",
            (),
        ),
    ],
)
def test_observed_design_partner_models_are_priced(
    provider,
    model,
    expected_canonical,
    expected_channel,
    expected_cost,
    expected_status,
    expected_reasons,
):
    result = SNAPSHOT.cost(
        provider=provider,
        model=model,
        at=_at("2026-08-10"),
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )
    assert result.status == expected_status
    assert result.canonical_model == expected_canonical
    assert f":{expected_channel}:" in result.price_id
    assert result.cost_usd == expected_cost
    assert result.reasons == expected_reasons


def test_gateway_input_excludes_cache_reads_and_writes():
    result = SNAPSHOT.cost(
        provider="openai",
        model="openai/gpt-5.6-luna",
        at=_at("2026-08-17"),
        input_tokens=200_000,
        output_tokens=0,
        cache_read_tokens=40_000,
        cache_write_tokens=60_000,
    )
    assert result.status == "priced"
    assert result.cost_usd == Decimal("0.03580000")


def test_gpt_4o_mini_is_priced():
    result = SNAPSHOT.cost(
        provider="openai",
        model="gpt-4o-mini",
        at=_at("2026-07-22"),
        input_tokens=100_000,
        output_tokens=50_000,
        cache_read_tokens=20_000,
    )
    assert result.status == "priced"
    assert result.canonical_model == "openai/gpt-4o-mini"
    # billable input excludes the cached tokens (input_includes_cache_read)
    expected = (
        Decimal(80_000) * Decimal("0.15") / Decimal(1_000_000)
        + Decimal(50_000) * Decimal("0.60") / Decimal(1_000_000)
        + Decimal(20_000) * Decimal("0.075") / Decimal(1_000_000)
    )
    assert result.cost_usd == expected.quantize(Decimal("0.00000001"))


@pytest.mark.parametrize(
    ("model", "input_rate", "output_rate"),
    [
        ("gpt-4o", Decimal("2.50"), Decimal("10.00")),
        ("gpt-4.1", Decimal("2.00"), Decimal("8.00")),
        ("gpt-4.1-mini", Decimal("0.40"), Decimal("1.60")),
        ("gpt-4.1-nano", Decimal("0.10"), Decimal("0.40")),
    ],
)
def test_legacy_openai_models_are_priced(model, input_rate, output_rate):
    result = SNAPSHOT.cost(
        provider="openai",
        model=model,
        at=_at("2026-07-22"),
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )
    assert result.status == "priced"
    assert result.canonical_model == f"openai/{model}"
    assert result.cost_usd == input_rate + output_rate


@pytest.mark.parametrize(
    ("provider", "model", "region"),
    [
        ("anthropic", "claude-sonnet-5", "global"),
        ("bedrock", "us.anthropic.claude-sonnet-5", "us-west-2"),
    ],
)
@pytest.mark.parametrize(
    "at",
    [
        "2026-08-31T23:59:59",
        "2026-09-01T00:00:00",
        "2026-09-02T00:00:00",
    ],
)
def test_sonnet_5_cancelled_increase_never_takes_effect(provider, model, region, at):
    snapshot = load_catalog(region=region).snapshot
    result = snapshot.cost(
        provider=provider,
        model=model,
        at=_at(at),
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )
    assert result.status == "priced"
    assert result.cost_usd == Decimal("12")


def test_gemini_provider_synonym_and_long_context():
    result = SNAPSHOT.cost(
        provider="gemini",
        model="gemini-2.5-pro",
        at=_at("2026-07-15"),
        input_tokens=300_000,
        output_tokens=1_000,
    )
    assert result.status == "priced"
    expected = (
        Decimal(300_000) * Decimal("1.25") * 2 / Decimal(1_000_000)
        + Decimal(1_000) * Decimal("10") * Decimal("1.5") / Decimal(1_000_000)
    )
    assert result.cost_usd == expected.quantize(Decimal("0.00000001"))


def test_unknown_model_is_unpriced():
    result = SNAPSHOT.cost(
        provider="openai",
        model="gpt-99",
        at=_at("2026-07-15"),
        input_tokens=10,
        output_tokens=10,
    )
    assert result.status == "unpriced"
    assert result.cost_usd is None


def test_batch_rates():
    result = SNAPSHOT.cost(
        provider="anthropic",
        model="claude-haiku-4-5",
        at=_at("2026-07-15"),
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        batch=True,
    )
    assert result.cost_usd == Decimal("0.50") + Decimal("2.50")


def test_overlapping_windows_rejected():
    doc = {
        "version": "test",
        "currency": "USD",
        "pricing_verified_at": "2026-08-24",
        "models": [
            {
                "canonical_id": "x/y",
                "aliases": [{"provider": "x", "alias": "y", "channel": "c"}],
                "prices": [
                    {
                        "channel": "c",
                        "effective_from": "2026-01-01",
                        "input_per_mtok": 1,
                        "source_url": "https://example.test/prices",
                    },
                    {
                        "channel": "c",
                        "effective_from": "2026-02-01",
                        "input_per_mtok": 2,
                        "source_url": "https://example.test/prices",
                    },
                ],
            }
        ],
    }
    try:
        parse_catalog(doc)
    except CatalogError:
        pass
    else:
        raise AssertionError("expected CatalogError for overlapping open windows")


def test_duplicate_alias_rejected():
    doc = {
        "version": "test",
        "currency": "USD",
        "pricing_verified_at": "2026-08-24",
        "models": [
            {
                "canonical_id": "x/y",
                "aliases": [
                    {"provider": "x", "alias": "y", "channel": "c"},
                    {"provider": "x", "alias": "y", "channel": "d"},
                ],
                "prices": [],
            }
        ],
    }
    with pytest.raises(CatalogError):
        parse_catalog(doc)


def _effective_alias_catalog(*aliases):
    models = []
    for canonical, alias in aliases:
        models.append({
            "canonical_id": canonical,
            "publisher": "example",
            "aliases": [{
                "provider": "example",
                "alias": "rolling-model",
                "channel": "example-api",
                "source_url": "https://example.test/model-history",
                **alias,
            }],
            "prices": [{
                "channel": "example-api",
                "region": "global",
                "effective_from": "2026-01-01",
                "input_per_mtok": 1,
                "output_per_mtok": 1,
                "source_url": "https://example.test/prices",
            }],
        })
    _, parsed_aliases, prices = parse_catalog({
        "version": "test",
        "currency": "USD",
        "pricing_verified_at": "2026-09-01",
        "models": models,
    })
    return CatalogSnapshot(parsed_aliases, prices, region="global")


def test_effective_dated_alias_resolves_by_call_timestamp():
    snapshot = _effective_alias_catalog(
        ("example/model-v1", {
            "effective_from": "2026-01-01",
            "effective_to": "2026-06-01T12:00:00+00:00",
        }),
        ("example/model-v2", {
            "effective_from": "2026-06-01T12:00:00+00:00",
        }),
    )

    before = snapshot.cost(
        provider="example", model="rolling-model",
        at=datetime(2026, 6, 1, 11, 59, tzinfo=timezone.utc),
        input_tokens=1_000_000, output_tokens=0,
    )
    after = snapshot.cost(
        provider="example", model="rolling-model",
        at=datetime(2026, 6, 1, 12, tzinfo=timezone.utc),
        input_tokens=1_000_000, output_tokens=0,
    )

    assert before.canonical_model == "example/model-v1"
    assert after.canonical_model == "example/model-v2"


def test_effective_dated_alias_gap_is_explicitly_unpriced():
    snapshot = _effective_alias_catalog(
        ("example/model-v1", {
            "effective_from": "2026-01-01",
            "effective_to": "2026-02-01",
        }),
        ("example/model-v2", {
            "effective_from": "2026-03-01",
        }),
    )

    result = snapshot.cost(
        provider="example", model="rolling-model",
        at=datetime(2026, 2, 15, tzinfo=timezone.utc),
        input_tokens=1, output_tokens=1,
    )

    assert result.status == "unpriced"
    assert result.reasons == ("no_effective_alias",)


def test_overlapping_effective_alias_windows_are_rejected():
    with pytest.raises(CatalogError, match="overlapping alias windows"):
        _effective_alias_catalog(
            ("example/model-v1", {
                "effective_from": "2026-01-01",
                "effective_to": "2026-07-01",
            }),
            ("example/model-v2", {
                "effective_from": "2026-06-01",
            }),
        )


def test_effective_dated_alias_requires_a_provider_source():
    with pytest.raises(CatalogError, match="effective alias needs source_url"):
        _effective_alias_catalog(
            ("example/model-v1", {
                "effective_from": "2026-01-01",
                "source_url": "",
            }),
        )


def test_overlapping_deployment_aliases_remain_rejected():
    document = {
        "version": "test",
        "currency": "USD",
        "pricing_verified_at": "2026-09-01",
        "models": [
            {
                "canonical_id": canonical,
                "publisher": provider,
                "aliases": [{
                    "provider": provider,
                    "alias": "shared-model",
                    "channel": "shared-api",
                    "effective_from": "2026-01-01",
                    "source_url": "https://example.test/model-history",
                }],
                "prices": [],
            }
            for provider, canonical in (
                ("provider-a", "provider-a/model"),
                ("provider-b", "provider-b/model"),
            )
        ],
    }
    _, aliases, prices = parse_catalog(document)

    with pytest.raises(ValueError, match="ambiguous deployment alias"):
        CatalogSnapshot(aliases, prices, region="global")


def test_undated_alias_keeps_existing_timeless_behavior():
    snapshot = _effective_alias_catalog(("example/model-v1", {}))

    result = snapshot.cost(
        provider="example", model="rolling-model",
        at=datetime(2000, 1, 1, tzinfo=timezone.utc),
        input_tokens=1_000_000, output_tokens=0,
    )

    assert result.status == "unpriced"
    assert result.canonical_model == "example/model-v1"
    assert result.reasons == ("no_effective_price",)


def test_malformed_catalog_rejected():
    with pytest.raises(CatalogError):
        parse_catalog({"models": []})
    with pytest.raises(CatalogError):
        parse_catalog({"version": "test"})


def _gateway(model: str, day: str):
    return SNAPSHOT.resolve_price(
        model=model, channel="vercel-ai-gateway", at=_at(day)
    )


# Closing a window (not editing it) leaves resolution at <= 2026-09-01 unchanged.
@pytest.mark.parametrize(
    ("model", "prior_input", "input_rate", "output_rate", "cache_read"),
    [
        ("openai/gpt-5.6-sol", "5.00", "2.00", "10.00", "0.20"),
        ("deepseek/deepseek-v4-flash", "0.14", "0.13", "0.26", "0.028"),
        ("deepseek/deepseek-v4-pro", "0.435", "0.66", "1.98", "0.022"),
        ("moonshotai/kimi-k3", "2.90", "3.00", "15.00", "0.30"),
    ],
)
def test_vercel_gateway_corrections_preserve_prior_window(
    model, prior_input, input_rate, output_rate, cache_read
):
    prior = _gateway(model, "2026-09-01")
    corrected = _gateway(model, "2026-09-02")
    assert prior is not None and corrected is not None
    assert prior.price.input_per_mtok == Decimal(prior_input)
    assert corrected.price.input_per_mtok == Decimal(input_rate)
    assert corrected.price.output_per_mtok == Decimal(output_rate)
    assert corrected.price.cache_read_per_mtok == Decimal(cache_read)


def test_gpt56_sol_direct_and_gateway_channels_price_independently():
    direct = SNAPSHOT.resolve_price(
        model="gpt-5.6-sol", channel="openai-api", at=_at("2026-09-02")
    )
    gateway = _gateway("openai/gpt-5.6-sol", "2026-09-02")
    assert direct is not None and gateway is not None
    assert direct.price.input_per_mtok == Decimal("4.00")
    assert gateway.price.input_per_mtok == Decimal("2.00")
    assert gateway.price.cache_write_5m_per_mtok == Decimal("2.50")
    # Gateway's top-level rate is provider-dependent; the direct rate is exact.
    assert gateway.price.rules.get("varies_by_provider") is True
    assert "varies_by_provider" not in direct.price.rules


def test_gpt56_sol_direct_correction_effective_2026_08_21_keeps_long_context():
    before = SNAPSHOT.resolve_price(
        model="gpt-5.6-sol", channel="openai-api", at=_at("2026-08-20")
    )
    after = SNAPSHOT.resolve_price(
        model="gpt-5.6-sol", channel="openai-api", at=_at("2026-08-21")
    )
    assert before is not None and after is not None
    assert before.price.input_per_mtok == Decimal("5.00")
    assert after.price.input_per_mtok == Decimal("4.00")
    assert after.price.cache_read_per_mtok == Decimal("0.40")
    assert after.price.cache_write_5m_per_mtok == Decimal("5.00")
    long_context = SNAPSHOT.cost(
        provider="openai", model="gpt-5.6-sol", at=_at("2026-08-21"),
        input_tokens=300_000, output_tokens=1_000,
    )
    expected = (
        Decimal(300_000) * Decimal("4.00") * 2 / Decimal(1_000_000)
        + Decimal(1_000) * Decimal("20.00") * Decimal("1.5") / Decimal(1_000_000)
    )
    assert long_context.cost_usd == expected.quantize(Decimal("0.00000001"))


def test_gemini36_flash_gateway_promo_reverts_on_2027_boundary():
    prior = _gateway("google/gemini-3.6-flash", "2026-09-01")
    promo = _gateway("google/gemini-3.6-flash", "2026-12-31")
    standard = _gateway("google/gemini-3.6-flash", "2027-01-01")
    assert prior is not None and promo is not None and standard is not None
    assert prior.price.input_per_mtok == Decimal("1.50")
    assert promo.price.input_per_mtok == Decimal("0.75")
    assert promo.price.output_per_mtok == Decimal("3.75")
    assert promo.price.cache_read_per_mtok == Decimal("0.075")
    # effective_to is exclusive: the standard rate resumes on 2027-01-01.
    assert standard.price.input_per_mtok == Decimal("1.50")
    assert standard.price.cache_read_per_mtok == Decimal("0.15")


# The gateway resells every provider, so an alias on it says nothing about who
# actually billed the call. Only a non-gateway channel identifies a provider's
# own direct billing relationship.
_GATEWAY_CHANNEL = "vercel-ai-gateway"
_NON_BILLING_PROVIDERS = {"unknown", "litellm"}


def _declared_direct_channels():
    """Each provider mapped to the non-gateway channels this catalog prices it on."""
    declared = {}
    for model in DOC["models"]:
        for alias in model.get("aliases") or []:
            provider, channel = alias.get("provider"), alias.get("channel")
            if (
                provider
                and provider not in _NON_BILLING_PROVIDERS
                and channel
                and channel != _GATEWAY_CHANNEL
            ):
                declared.setdefault(provider, set()).add(channel)
    return declared


def test_every_directly_priced_provider_resolves_its_own_channel():
    """A provider this catalog prices directly must resolve that same channel.

    Callers treat a missing direct channel as "explicitly unpriced", so a
    provider present in the data but absent from the channel map makes its
    traffic unpriceable even though the price sits in this very file. That gap
    blocked every analysis of DeepSeek, Perplexity and xAI traffic; this keeps
    the map and the data from drifting apart again.
    """
    mismatched = {
        provider: (sorted(channels), direct_channel_for_provider(provider))
        for provider, channels in _declared_direct_channels().items()
        if direct_channel_for_provider(provider) not in channels
    }
    assert mismatched == {}, (
        "providers priced on a direct channel the map does not resolve to it: "
        f"{mismatched}"
    )


def test_gateway_only_providers_have_no_direct_channel():
    """The converse. A provider reachable only through the gateway has no direct
    billing relationship, and inventing one would price its traffic at a
    reseller's list rate rather than what the customer was actually charged."""
    gateway_only = {
        alias.get("provider")
        for model in DOC["models"]
        for alias in model.get("aliases") or []
        if alias.get("channel") == _GATEWAY_CHANNEL and alias.get("provider")
    } - set(_declared_direct_channels())
    resolved = {
        provider
        for provider in gateway_only
        if direct_channel_for_provider(provider) is not None
    }
    # Each of these is a provider spelling the catalog also prices directly, not
    # an invented direct channel: "xai" is the gateway spelling of "x-ai", and
    # "vercel" names the gateway itself, which is a real billing relationship.
    # "moonshotai" has left this set: it now carries its own spellings on
    # moonshot-api, so it is no longer reachable only through the gateway.
    assert resolved == {"xai", "vercel"}


@pytest.mark.parametrize(
    ("model", "price_id"),
    [
        # Version suffix on the bare, geographic, and global ids.
        ("anthropic.claude-opus-5-v1:0", "anthropic/claude-opus-5:aws-bedrock:global:2026-07-24"),
        ("anthropic.claude-opus-5-v2", "anthropic/claude-opus-5:aws-bedrock:global:2026-07-24"),
        ("us.anthropic.claude-opus-5-v1:0", "anthropic/claude-opus-5:aws-bedrock-geo:global:2026-07-24"),
        ("global.anthropic.claude-opus-5-v1:0", "anthropic/claude-opus-5:aws-bedrock:global:2026-07-24"),
        ("anthropic.claude-sonnet-4-6-v1:0", "anthropic/claude-sonnet-4.6:aws-bedrock:global:2026-02-17"),
        ("eu.anthropic.claude-sonnet-4-6-v1:0", "anthropic/claude-sonnet-4.6:aws-bedrock-geo:global:2026-02-17"),
    ],
)
@pytest.mark.parametrize("provider", ["bedrock", "aws", "amazon-bedrock"])
def test_bedrock_version_suffix_falls_back_to_catalog_id(provider, model, price_id):
    result = SNAPSHOT.cost(
        provider=provider,
        model=model,
        at=_at("2026-09-20"),
        input_tokens=1_000,
        output_tokens=1_000,
    )
    assert result.status == "priced"
    assert result.price_id == price_id


@pytest.mark.parametrize(
    "model",
    [
        "anthropic.claude-sonnet-5-v1:0",
        "us.anthropic.claude-sonnet-5-v1:0",
        "global.anthropic.claude-sonnet-5-v1:0",
        "global.anthropic.claude-sonnet-5",
    ],
)
def test_bedrock_sonnet_5_versioned_ids_price_in_its_region(model):
    snapshot = load_catalog(region="us-west-2").snapshot
    result = snapshot.cost(
        provider="bedrock",
        model=model,
        at=_at("2026-09-20"),
        input_tokens=1_000_000,
        output_tokens=0,
    )
    assert result.status == "priced"
    assert result.canonical_model == "anthropic/claude-sonnet-5"
    assert result.price_id == "anthropic/claude-sonnet-5:aws-bedrock:us-west-2:2026-06-30"


def test_bedrock_global_prefix_falls_back_on_the_in_region_channel():
    resolved = SNAPSHOT.resolve_price(
        model="global.anthropic.claude-sonnet-4-6-v1:0",
        channel="aws-bedrock",
        at=_at("2026-09-20"),
    )
    assert resolved is not None
    assert resolved.canonical_model == "anthropic/claude-sonnet-4.6"
    assert resolved.price.pricing_channel == "aws-bedrock"


def _bedrock_catalog(*entries):
    models = []
    for canonical, alias, channel in entries:
        models.append({
            "canonical_id": canonical,
            "publisher": "acme",
            "aliases": [{"provider": "bedrock", "alias": alias, "channel": channel}],
            "prices": [{
                "channel": channel,
                "region": "global",
                "effective_from": "2026-01-01",
                "input_per_mtok": 1,
                "output_per_mtok": 1,
                "source_url": "https://example.test",
            }],
        })
    _, aliases, prices = parse_catalog({
        "version": "test",
        "currency": "USD",
        "pricing_verified_at": "2026-09-01",
        "models": models,
    })
    return CatalogSnapshot(aliases, prices, region="global")


def test_bedrock_exact_alias_wins_over_normalization():
    snapshot = _bedrock_catalog(
        ("acme/model", "acme.model", "aws-bedrock"),
        ("acme/model-v2", "acme.model-v2:0", "aws-bedrock"),
        ("acme/global-model", "global.acme.model", "aws-bedrock"),
    )
    at = _at("2026-09-20")

    def canonical(model):
        return snapshot.cost(
            provider="bedrock", model=model, at=at, input_tokens=1, output_tokens=1
        ).canonical_model

    assert canonical("acme.model-v2:0") == "acme/model-v2"
    assert canonical("acme.model-v1:0") == "acme/model"
    assert canonical("global.acme.model-v1:0") == "acme/global-model"
    assert snapshot.resolve_price(
        model="acme.model-v2:0", channel="aws-bedrock", at=at
    ).canonical_model == "acme/model-v2"
    assert snapshot.resolve_price(
        model="global.acme.model-v1:0", channel="aws-bedrock", at=at
    ).canonical_model == "acme/global-model"
    # A real catalog id that already carries a suffix still matches exactly.
    assert SNAPSHOT.cost(
        provider="bedrock", model="deepseek.v3-v1:0", at=at,
        input_tokens=1, output_tokens=1,
    ).canonical_model == "deepseek/v3.1"


def test_bedrock_global_prefix_never_reaches_a_geo_price():
    snapshot = _bedrock_catalog(("acme/model", "acme.model", "aws-bedrock-geo"))
    assert snapshot.cost(
        provider="bedrock",
        model="global.acme.model-v1:0",
        at=_at("2026-09-20"),
        input_tokens=1,
        output_tokens=1,
    ).status == "unpriced"
    assert snapshot.resolve_price(
        model="global.acme.model", channel="aws-bedrock-geo", at=_at("2026-09-20")
    ) is None
    # The version suffix alone still resolves on the geo channel.
    assert snapshot.resolve_price(
        model="acme.model-v1:0", channel="aws-bedrock-geo", at=_at("2026-09-20")
    ) is not None


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        # Normalization is Bedrock-only.
        ("anthropic", "claude-sonnet-5-v1:0"),
        ("anthropic", "global.anthropic.claude-haiku-4-5-20251001-v1:0"),
        ("litellm", "anthropic.claude-sonnet-4-6-v1:0"),
        ("openai", "gpt-4o-v1:0"),
        # Geographic prefixes are never stripped to the bare in-region id.
        ("bedrock", "apac.anthropic.claude-opus-5-v1:0"),
        # Only a trailing `-vN` or `-vN:M` is a version suffix.
        ("bedrock", "anthropic.claude-opus-5-v1:0:200k"),
        ("bedrock", "anthropic.claude-opus-5-v1.0"),
        ("bedrock", "anthropic.claude-opus-5v1:0"),
        ("bedrock", "global."),
    ],
)
def test_bedrock_normalization_does_not_widen_matching(provider, model):
    result = SNAPSHOT.cost(
        provider=provider,
        model=model,
        at=_at("2026-09-20"),
        input_tokens=1_000,
        output_tokens=1_000,
    )
    assert result.status == "unpriced"
    assert result.reasons == ("unknown_model",)


@pytest.mark.parametrize(
    "channel", ["anthropic-api", "vercel-ai-gateway", "google-vertex-ai"]
)
def test_version_suffix_is_not_stripped_off_bedrock_channels(channel):
    assert SNAPSHOT.resolve_price(
        model="claude-opus-5-v1:0", channel=channel, at=_at("2026-09-20")
    ) is None
    assert SNAPSHOT.resolve_price(
        model="anthropic/claude-opus-5-v1:0", channel=channel, at=_at("2026-09-20")
    ) is None


@pytest.mark.parametrize(
    ("region", "model", "launched", "price_id"),
    [
        # Profile types use separate pricing channels.
        ("global", "us.openai.gpt-6.1-sol", "2026-09-29", "openai/gpt-6.1-sol:aws-bedrock-geo:global:2026-09-29"),
        ("global", "us.openai.gpt-6-sol", "2026-09-22", "openai/gpt-6-sol:aws-bedrock-geo:global:2026-09-22"),
        ("global", "global.openai.gpt-6-sol", "2026-09-22", "openai/gpt-6-sol:aws-bedrock:global:2026-09-22"),
        ("global", "us.openai.gpt-6-luna", "2026-09-22", "openai/gpt-6-luna:aws-bedrock-geo:global:2026-09-22"),
        ("global", "global.openai.gpt-6-luna", "2026-09-22", "openai/gpt-6-luna:aws-bedrock:global:2026-09-22"),
        ("global", "us.openai.gpt-6-astra", "2026-09-08", "openai/gpt-6-astra:aws-bedrock-geo:global:2026-09-08"),
        ("global", "global.openai.gpt-6-astra", "2026-09-08", "openai/gpt-6-astra:aws-bedrock:global:2026-09-08"),
        ("global", "us.xai.grok-4.7", "2026-09-28", "xai/grok-4.7:aws-bedrock-geo:global:2026-09-28"),
        ("global", "global.xai.grok-4.7", "2026-09-28", "xai/grok-4.7:aws-bedrock:global:2026-09-28"),
        ("global", "us.xai.grok-4.6", "2026-08-18", "xai/grok-4.6:aws-bedrock-geo:global:2026-08-18"),
        ("global", "global.xai.grok-4.6", "2026-08-18", "xai/grok-4.6:aws-bedrock:global:2026-08-18"),
        # Bare IDs use Mantle's regional price.
        ("us-east-1", "openai.gpt-6.1-sol", "2026-09-29", "openai/gpt-6.1-sol:aws-bedrock:us-east-1:2026-09-29"),
        ("us-east-1", "openai.gpt-6-sol", "2026-09-22", "openai/gpt-6-sol:aws-bedrock:us-east-1:2026-09-22"),
        ("us-east-1", "openai.gpt-6-luna", "2026-09-22", "openai/gpt-6-luna:aws-bedrock:us-east-1:2026-09-22"),
        ("us-east-1", "openai.gpt-6-astra", "2026-09-08", "openai/gpt-6-astra:aws-bedrock:us-east-1:2026-09-08"),
        ("us-west-2", "openai.gpt-6-astra", "2026-09-08", "openai/gpt-6-astra:aws-bedrock:us-west-2:2026-09-08"),
        ("us-east-1", "us.openai.gpt-6.1-sol", "2026-09-29", "openai/gpt-6.1-sol:aws-bedrock-geo:global:2026-09-29"),
    ],
)
def test_bedrock_openai_and_xai_ids_start_on_their_bedrock_launch_day(
    region, model, launched, price_id
):
    snapshot = load_catalog(region=region).snapshot
    launch = _at(launched)
    before = snapshot.cost(
        provider="bedrock", model=model, at=launch - timedelta(seconds=1),
        input_tokens=1_000, output_tokens=1_000,
    )
    released = snapshot.cost(
        provider="bedrock", model=model, at=launch,
        input_tokens=1_000, output_tokens=1_000,
    )
    assert before.status == "unpriced"
    assert before.reasons == ("no_effective_price",)
    assert released.status == "priced"
    assert released.price_id == price_id


_BEDROCK_SNAPSHOT_REGIONS = ["global", "us-east-1", "us-west-2", "eu-west-1"]


@pytest.mark.parametrize("region", _BEDROCK_SNAPSHOT_REGIONS)
@pytest.mark.parametrize(
    ("model", "input_rate", "output_rate", "cache_read", "cache_write"),
    [
        ("openai.gpt-6.1-sol", "2.20", "11.00", "0.11", "2.75"),
        ("openai.gpt-6-sol", "2.20", "11.00", "0.22", "2.75"),
        ("openai.gpt-6-luna", "0.11", "0.55", "0.011", "0.1375"),
    ],
)
def test_bedrock_bare_gpt_6_ids_bill_their_mantle_row_only_in_us_east_1(
    region, model, input_rate, output_rate, cache_read, cache_write
):
    """Single-region Mantle IDs are unpriced outside the region serving them."""
    snapshot = load_catalog(region=region).snapshot
    at = _at("2026-10-02")
    assert snapshot.resolve_price(
        model=model, channel="aws-bedrock-geo", at=at
    ) is None
    resolved = snapshot.resolve_price(model=model, channel="aws-bedrock", at=at)
    if region != "us-east-1":
        assert resolved is None
        result = snapshot.cost(
            provider="bedrock", model=model, at=at,
            input_tokens=1_000, output_tokens=1_000,
        )
        assert result.status == "unpriced"
        assert result.reasons == ("no_effective_alias",)
        return
    assert resolved is not None
    assert resolved.price.region == "us-east-1"
    assert resolved.price.input_per_mtok == Decimal(input_rate)
    assert resolved.price.output_per_mtok == Decimal(output_rate)
    assert resolved.price.cache_read_per_mtok == Decimal(cache_read)
    assert resolved.price.cache_write_5m_per_mtok == Decimal(cache_write)


@pytest.mark.parametrize(
    ("region", "price_id"),
    [
        ("us-east-1", "openai/gpt-6-astra:aws-bedrock:us-east-1:2026-09-08"),
        ("us-west-2", "openai/gpt-6-astra:aws-bedrock:us-west-2:2026-09-08"),
        # A multi-region bare ID requires a matching catalog region.
        ("global", None),
        ("eu-west-1", None),
    ],
)
def test_bedrock_bare_gpt_6_astra_prices_only_in_its_mantle_regions(region, price_id):
    result = load_catalog(region=region).snapshot.cost(
        provider="bedrock", model="openai.gpt-6-astra", at=_at("2026-10-02"),
        input_tokens=100_000, output_tokens=0,
    )
    if price_id is None:
        assert result.status == "unpriced"
        assert result.reasons == ("no_effective_alias",)
    else:
        assert result.price_id == price_id
        assert result.cost_usd == Decimal("1.10000000")


@pytest.mark.parametrize("region", _BEDROCK_SNAPSHOT_REGIONS)
@pytest.mark.parametrize(
    ("model", "price_id", "cost"),
    [
        ("global.openai.gpt-6-sol", "openai/gpt-6-sol:aws-bedrock:global:2026-09-22", "0.20000000"),
        ("global.openai.gpt-6-luna", "openai/gpt-6-luna:aws-bedrock:global:2026-09-22", "0.01000000"),
        ("global.openai.gpt-6-astra", "openai/gpt-6-astra:aws-bedrock:global:2026-09-08", "1.00000000"),
        ("global.openai.gpt-6-astra-v1:0", "openai/gpt-6-astra:aws-bedrock:global:2026-09-08", "1.00000000"),
        ("us.openai.gpt-6-sol", "openai/gpt-6-sol:aws-bedrock-geo:global:2026-09-22", "0.22000000"),
        ("us.openai.gpt-6.1-sol", "openai/gpt-6.1-sol:aws-bedrock-geo:global:2026-09-29", "0.22000000"),
    ],
)
def test_bedrock_gpt_6_profiles_keep_their_own_rate_in_every_snapshot(
    region, model, price_id, cost
):
    """Global profiles ignore a catalog's regional Mantle row."""
    result = load_catalog(region=region).snapshot.cost(
        provider="bedrock", model=model, at=_at("2026-10-02"),
        input_tokens=100_000, output_tokens=0,
    )
    assert result.status == "priced"
    assert result.price_id == price_id
    assert result.cost_usd == Decimal(cost)


@pytest.mark.parametrize("region", _BEDROCK_SNAPSHOT_REGIONS)
@pytest.mark.parametrize(
    "model", ["global.openai.gpt-6.1-sol", "global.openai.gpt-6.1-sol-v1:0"]
)
def test_bedrock_gpt_6_1_sol_global_profile_is_unpriced_in_every_snapshot(
    region, model
):
    snapshot = load_catalog(region=region).snapshot
    at = _at("2026-10-02")
    declared = [
        alias["alias"]
        for entry in DOC["models"]
        for alias in entry.get("aliases") or []
    ]
    assert "global.openai.gpt-6.1-sol" not in declared
    result = snapshot.cost(
        provider="bedrock", model=model, at=at,
        input_tokens=1_000, output_tokens=1_000,
    )
    assert result.status == "unpriced"
    assert result.reasons == ("no_effective_alias",)
    for channel in ("aws-bedrock", "aws-bedrock-geo"):
        assert snapshot.resolve_price(model=model, channel=channel, at=at) is None


def test_bedrock_gpt_6_1_sol_channels_resolve_to_their_own_rows():
    at = _at("2026-10-02")
    snapshot = load_catalog(region="us-east-1").snapshot
    geo = snapshot.resolve_price(
        model="openai/gpt-6.1-sol", channel="aws-bedrock-geo", at=at
    )
    in_region = snapshot.resolve_price(
        model="openai/gpt-6.1-sol", channel="aws-bedrock", at=at
    )
    assert geo is not None and in_region is not None
    assert geo.price.id == "openai/gpt-6.1-sol:aws-bedrock-geo:global:2026-09-29"
    assert in_region.price.id == "openai/gpt-6.1-sol:aws-bedrock:us-east-1:2026-09-29"


@pytest.mark.parametrize(
    ("channel", "input_rate", "output_rate", "cache_read", "long_context"),
    [
        ("aws-bedrock-geo", "2.2", "6.6", "0.55", False),
        ("aws-bedrock", "2.0", "6.0", "0.5", False),
        ("xai-api", "2.0", "6.0", "0.5", True),
    ],
)
def test_grok_4_7_channels_price_independently(
    channel, input_rate, output_rate, cache_read, long_context
):
    resolved = SNAPSHOT.resolve_price(
        model="xai/grok-4.7", channel=channel, at=_at("2026-10-02")
    )
    assert resolved is not None
    assert resolved.price.pricing_channel == channel
    assert resolved.price.input_per_mtok == Decimal(input_rate)
    assert resolved.price.output_per_mtok == Decimal(output_rate)
    assert resolved.price.cache_read_per_mtok == Decimal(cache_read)
    # Direct-provider rules must not leak onto Bedrock.
    assert ("long_context" in resolved.rules) is long_context


def test_bedrock_grok_4_7_has_no_in_region_identity():
    result = SNAPSHOT.cost(
        provider="bedrock", model="xai.grok-4.7", at=_at("2026-10-02"),
        input_tokens=1_000, output_tokens=1_000,
    )
    assert result.status == "unpriced"
    assert result.reasons == ("unknown_model",)


@pytest.mark.parametrize(
    ("model", "geo_input", "global_input"),
    [
        ("openai/gpt-6-sol", "2.20", "2.00"),
        ("openai/gpt-6-luna", "0.11", "0.10"),
        ("openai/gpt-6-astra", "11.00", "10.00"),
        ("xai/grok-4.6", "2.2", "2.0"),
    ],
)
def test_bedrock_geo_premium_stays_off_the_global_profile(
    model, geo_input, global_input
):
    at = _at("2026-10-02")
    geo = SNAPSHOT.resolve_price(model=model, channel="aws-bedrock-geo", at=at)
    worldwide = SNAPSHOT.resolve_price(model=model, channel="aws-bedrock", at=at)
    assert geo is not None and worldwide is not None
    assert geo.price.input_per_mtok == Decimal(geo_input)
    assert worldwide.price.input_per_mtok == Decimal(global_input)


def test_new_bedrock_rows_leave_gpt_5_6_sol_on_its_regional_price():
    snapshot = load_catalog(region="us-west-2").snapshot
    at = _at("2026-10-02")
    existing = snapshot.cost(
        provider="bedrock", model="us.openai.gpt-5.6-sol", at=at,
        input_tokens=1_000, output_tokens=1_000,
    )
    added = snapshot.cost(
        provider="bedrock", model="us.openai.gpt-6.1-sol", at=at,
        input_tokens=1_000, output_tokens=1_000,
    )
    assert existing.price_id == "openai/gpt-5.6-sol:aws-bedrock:us-west-2:2026-08-17"
    assert added.price_id == "openai/gpt-6.1-sol:aws-bedrock-geo:global:2026-09-29"


_PINNED_RATES = {"global": 1, "us-east-1": 2, "us-west-2": 3}


def _pinned_catalog(aliases, *, region, price_regions=("global", "us-east-1", "us-west-2")):
    """Build a synthetic catalog with distinct rates by region."""
    _, parsed_aliases, prices = parse_catalog({
        "version": "test",
        "currency": "USD",
        "pricing_verified_at": "2026-09-01",
        "models": [{
            "canonical_id": "test/pinned-region-model",
            "publisher": "test",
            "aliases": [
                {"provider": "bedrock", "channel": "aws-bedrock", **alias}
                for alias in aliases
            ],
            "prices": [
                {
                    "channel": "aws-bedrock",
                    "region": price_region,
                    "effective_from": "2026-01-01",
                    "input_per_mtok": _PINNED_RATES[price_region],
                    "output_per_mtok": 1,
                    "source_url": "https://example.com/pinned-region-prices",
                }
                for price_region in price_regions
            ],
        }],
    })
    return CatalogSnapshot(parsed_aliases, prices, region=region)


def _pinned_cost(snapshot, model):
    return snapshot.cost(
        provider="bedrock", model=model, at=_at("2026-09-20"),
        input_tokens=1_000_000, output_tokens=0,
    )


def _assert_no_effective_alias(snapshot, model):
    result = _pinned_cost(snapshot, model)
    assert result.status == "unpriced"
    assert result.canonical_model is None
    assert result.reasons == ("no_effective_alias",)
    assert snapshot.resolve_price(
        model=model, channel="aws-bedrock", at=_at("2026-09-20")
    ) is None


def test_alias_pinned_to_one_region_resolves_in_that_snapshot_region():
    snapshot = _pinned_catalog(
        [{"alias": "test.pinned-region-model", "price_region": "US-East-1"}],
        region="us-east-1",
    )
    result = _pinned_cost(snapshot, "test.pinned-region-model")
    assert result.status == "priced"
    assert result.price_id == "test/pinned-region-model:aws-bedrock:us-east-1:2026-01-01"
    assert result.cost_usd == Decimal("2.00000000")
    resolved = snapshot.resolve_price(
        model="test.pinned-region-model", channel="aws-bedrock", at=_at("2026-09-20")
    )
    assert resolved is not None and resolved.price.region == "us-east-1"


@pytest.mark.parametrize("region", ["global", "us-west-2", "eu-west-1"])
def test_alias_pinned_to_one_region_does_not_resolve_in_other_snapshots(region):
    """Regional pins do not import prices into other snapshots."""
    snapshot = _pinned_catalog(
        [{"alias": "test.pinned-region-model", "price_region": "us-east-1"}], region=region
    )
    _assert_no_effective_alias(snapshot, "test.pinned-region-model")


def test_alias_price_region_without_that_price_row_is_unpriced():
    snapshot = _pinned_catalog(
        [{"alias": "test.pinned-region-model", "price_region": "us-east-1"}],
        region="us-east-1",
        price_regions=("global", "us-west-2"),
    )
    result = _pinned_cost(snapshot, "test.pinned-region-model")
    assert result.status == "unpriced"
    assert result.canonical_model == "test/pinned-region-model"
    assert result.reasons == ("no_effective_price",)
    assert snapshot.resolve_price(
        model="test.pinned-region-model", channel="aws-bedrock", at=_at("2026-09-20")
    ) is None


@pytest.mark.parametrize("region", ["global", "us-east-1", "us-west-2", "eu-west-1"])
def test_alias_pinned_to_global_resolves_the_global_price_in_every_snapshot(region):
    snapshot = _pinned_catalog(
        [{"alias": "global.test.pinned-region-model", "price_region": "global"}], region=region
    )
    result = _pinned_cost(snapshot, "global.test.pinned-region-model")
    assert result.status == "priced"
    assert result.price_id == "test/pinned-region-model:aws-bedrock:global:2026-01-01"
    assert result.cost_usd == Decimal("1.00000000")


@pytest.mark.parametrize(
    ("region", "expected"),
    [("global", "1.00000000"), ("us-east-1", "2.00000000"),
     ("us-west-2", "3.00000000"), ("eu-west-1", "1.00000000")],
)
def test_alias_without_price_region_keeps_snapshot_then_global_fallback(
    region, expected
):
    snapshot = _pinned_catalog([{"alias": "test.pinned-region-model"}], region=region)
    assert _pinned_cost(snapshot, "test.pinned-region-model").cost_usd == Decimal(expected)


@pytest.mark.parametrize(
    ("region", "expected"),
    [("us-east-1", "2.00000000"), ("us-west-2", "3.00000000"),
     ("global", None), ("eu-west-1", None)],
)
def test_alias_with_several_price_regions_resolves_only_in_a_listed_snapshot_region(
    region, expected
):
    snapshot = _pinned_catalog(
        [{"alias": "test.pinned-region-model", "price_region": ["us-east-1", "US-WEST-2"]}],
        region=region,
    )
    if expected is None:
        _assert_no_effective_alias(snapshot, "test.pinned-region-model")
    else:
        assert _pinned_cost(snapshot, "test.pinned-region-model").cost_usd == Decimal(expected)


@pytest.mark.parametrize(
    ("region", "expected"),
    [("us-east-1", "us-east-1"), ("us-west-2", "us-west-2"),
     ("global", None), ("eu-west-1", None)],
)
def test_candidates_pinned_to_different_regions_never_resolve_an_unmatched_snapshot(
    region, expected
):
    """Unmatched regional candidates do not resolve by list order."""
    snapshot = _pinned_catalog(
        [
            {"alias": "test.pinned-region-model-east", "price_region": "us-east-1"},
            {"alias": "test.pinned-region-model-west", "price_region": "us-west-2"},
        ],
        region=region,
    )
    resolved = snapshot.resolve_price(
        model="test/pinned-region-model", channel="aws-bedrock", at=_at("2026-09-20")
    )
    if expected is None:
        assert resolved is None
    else:
        assert resolved is not None and resolved.price.region == expected


@pytest.mark.parametrize("price_region", ["", "  ", [], ["us-east-1", ""], 7, ["us-east-1", 7]])
def test_malformed_alias_price_region_is_rejected(price_region):
    with pytest.raises(CatalogError, match="price_region"):
        _pinned_catalog(
            [{"alias": "test.pinned-region-model", "price_region": price_region}], region="global"
        )


@pytest.mark.parametrize("region", ["global", "us-east-1", "us-west-2"])
@pytest.mark.parametrize("model", ["global.test.pinned-region-model", "global.test.pinned-region-model-v1:0"])
def test_bedrock_global_prefix_never_falls_back_onto_a_region_pinned_alias(
    region, model
):
    snapshot = _pinned_catalog(
        [{"alias": "test.pinned-region-model", "price_region": "us-east-1"}], region=region
    )
    _assert_no_effective_alias(snapshot, model)
    if region == "us-east-1":
        assert _pinned_cost(
            snapshot, "test.pinned-region-model-v1:0"
        ).cost_usd == Decimal("2.00000000")
    else:
        _assert_no_effective_alias(snapshot, "test.pinned-region-model-v1:0")


def test_bedrock_global_prefix_still_falls_back_onto_a_global_pinned_alias():
    snapshot = _pinned_catalog(
        [{"alias": "test.pinned-region-model", "price_region": "global"}], region="us-east-1"
    )
    assert _pinned_cost(snapshot, "global.test.pinned-region-model-v1:0").cost_usd == Decimal(
        "1.00000000"
    )


@pytest.mark.parametrize(
    ("region", "expected"),
    [("global", "global"), ("us-east-1", "us-east-1"), ("us-west-2", "global")],
)
def test_canonical_id_with_differently_pinned_aliases_follows_the_snapshot_region(
    region, expected
):
    snapshot = _pinned_catalog(
        [
            {"alias": "test.pinned-region-model", "price_region": "us-east-1"},
            {"alias": "global.test.pinned-region-model", "price_region": "global"},
        ],
        region=region,
    )
    resolved = snapshot.resolve_price(
        model="test/pinned-region-model", channel="aws-bedrock", at=_at("2026-09-20")
    )
    assert resolved is not None
    assert resolved.price.region == expected


def test_every_pinned_alias_names_a_price_region_the_catalog_carries():
    missing = []
    for model in DOC["models"]:
        carried = {
            (price["channel"], str(price.get("region") or "global").lower())
            for price in model.get("prices") or []
        }
        for alias in model.get("aliases") or []:
            pinned = alias.get("price_region")
            if pinned is None:
                continue
            for price_region in [pinned] if isinstance(pinned, str) else pinned:
                if (alias["channel"], price_region.lower()) not in carried:
                    missing.append((alias["alias"], price_region))
    assert missing == []


# --- Cost components -----------------------------------------------------------

from metergraph_core import CostComponents, load_catalog as _load_for_components


def test_components_sum_to_the_cost_and_follow_the_rules():
    catalog = _load_for_components(region="us-west-2")
    # Anthropic: cache reads and writes are separate buckets at their own rates.
    result = catalog.snapshot.cost(
        provider="anthropic",
        model="claude-sonnet-4-6",
        at="2026-09-18T12:00:00+00:00",
        input_tokens=1_000_000,
        output_tokens=200_000,
        cache_read_tokens=400_000,
        cache_write_5m_tokens=100_000,
    )
    assert result.status == "priced"
    parts = result.components
    assert isinstance(parts, CostComponents)
    assert parts.input_usd + parts.output_usd + parts.cache_read_usd + parts.cache_write_usd == result.cost_usd
    assert parts.cache_read_usd > 0 and parts.cache_write_usd > 0
    assert result.publisher == "anthropic"


def test_components_split_a_cache_inclusive_input_the_way_the_total_does():
    catalog = _load_for_components()
    # OpenAI counts cached tokens inside input, so the input part is billed
    # on the uncached remainder and the cache part on the cached tokens.
    result = catalog.snapshot.cost(
        provider="openai",
        model="gpt-5.4-mini",
        at="2026-09-18T12:00:00+00:00",
        input_tokens=1_000_000,
        output_tokens=0,
        cache_read_tokens=1_000_000,
    )
    assert result.status == "priced"
    assert result.components.input_usd == 0
    assert result.components.cache_read_usd == result.cost_usd


def test_an_unpriced_result_carries_no_components():
    result = _load_for_components().snapshot.cost(
        provider="openai", model="no-such-model", at="2026-09-18", input_tokens=1, output_tokens=1
    )
    assert result.components is None and result.publisher is None


def test_the_closed_gateway_sonar_row_still_charges_its_per_call_fee():
    catalog = _load_for_components()
    # 1,000 input and 500 output tokens, plus the row's per-call fee:
    # $0.00025 + $0.00125 + $0.0025.
    result = catalog.price(
        model="perplexity/sonar", channel="vercel-ai-gateway",
        at="2026-10-11T12:00:00+00:00", input_tokens=1_000, output_tokens=500,
    )
    assert result.status == "priced"
    assert result.cost_usd == Decimal("0.00400000")
    parts = result.components
    assert parts.request_usd == Decimal("0.00250000")
    assert parts.input_usd + parts.output_usd + parts.cache_read_usd + parts.cache_write_usd + parts.request_usd == result.cost_usd


def test_the_direct_sonar_row_still_reports_its_request_fee_as_uncaptured():
    catalog = _load_for_components()
    result = catalog.price(
        model="perplexity/sonar", channel="perplexity-api",
        at="2026-10-11T12:00:00+00:00", input_tokens=1_000, output_tokens=500,
    )
    assert result.status == "partial"
    assert "uncaptured_fees" in result.reasons
    assert result.components.request_usd == 0


def test_a_row_without_a_request_fee_charges_none():
    catalog = _load_for_components()
    result = catalog.snapshot.cost(
        provider="anthropic", model="claude-sonnet-4-6",
        at="2026-09-18T12:00:00+00:00", input_tokens=1_000_000, output_tokens=0,
    )
    assert result.components.request_usd == 0
    assert result.cost_usd == result.components.input_usd


@pytest.mark.parametrize(
    'day,input_tokens,expected',
    [
        ('2026-10-26', 300000, '3.82500000'),
        ('2026-10-27', 272000, '3.47500000'),
        ('2026-10-27', 272001, '6.91252500'),
    ],
)
def test_gpt56_cyber_long_context_correction_preserves_prior_prices(day, input_tokens, expected):
    result = SNAPSHOT.cost(
        provider='openai', model='gpt-5.6-cyber', at=_at(day),
        input_tokens=input_tokens, output_tokens=1000,
    )
    assert result.status == 'priced'
    assert result.cost_usd == Decimal(expected)


def test_gpt56_cyber_long_context_counts_separate_cache_writes():
    result = SNAPSHOT.cost(
        provider='openai', model='gpt-5.6-cyber', at=_at('2026-10-27'),
        input_tokens=270000, cache_write_tokens=3000, output_tokens=1000,
    )
    assert result.status == 'priced'
    assert result.cost_usd == Decimal('6.95625000')
