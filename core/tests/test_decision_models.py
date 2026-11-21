"""Decision models: a catalog class, priced on input tokens only.

A decision model answers typed questions (choose one option, yes or no, a
score) instead of writing text. Its calls are billed per input token and its
output is free, so a call that reports no output usage is still fully priced.
The rule that makes this work is general and keyed on the rate, not the
class: a usage component whose effective rate is exactly zero is not needed
for a complete figure. These tests also pin that generative pricing is
unchanged.
"""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from metergraph_core import (
    DECISION,
    GENERATIVE,
    CatalogError,
    ModelUsage,
    classify_recorded_call,
    load_catalog,
    load_catalog_document,
    price_call_from_record,
)
from metergraph_core.models import (
    ModelRegistryError,
    parse_model_registry,
    validate_model_registry,
)

CATALOG = load_catalog()
SNAPSHOT = CATALOG.snapshot
AFTER = datetime(2026, 11, 22, tzinfo=timezone.utc)
BEFORE = datetime(2026, 11, 20, tzinfo=timezone.utc)
JEV_RATE = Decimal("0.042")


def _per_million(tokens: int, rate: Decimal) -> Decimal:
    return (Decimal(tokens) * rate / Decimal(1_000_000)).quantize(Decimal("0.00000001"))


def _document(**entry):
    return {
        "version": "test",
        "currency": "USD",
        "pricing_verified_at": "2026-11-21",
        "models": [
            {
                "canonical_id": "example/model-a",
                "publisher": "example",
                "aliases": [{"provider": "example", "alias": "model-a", "channel": "example-api"}],
                "prices": [
                    {
                        "channel": "example-api",
                        "effective_from": "2026-01-01",
                        "input_per_mtok": 1,
                        "output_per_mtok": 2,
                        "source_url": "https://example.test/prices",
                    }
                ],
                **entry,
            }
        ],
    }


# ── the class field ─────────────────────────────────────────────────────


def test_an_entry_without_a_class_is_generative():
    catalog = load_catalog_document(_document())
    assert catalog.model_class("example/model-a") == GENERATIVE
    result = catalog.snapshot.cost(
        provider="example", model="model-a", at=AFTER, input_tokens=1, output_tokens=1
    )
    assert result.model_class == GENERATIVE


def test_an_entry_may_declare_the_decision_class():
    catalog = load_catalog_document(_document(model_class="decision"))
    assert catalog.model_class("example/model-a") == DECISION
    result = catalog.snapshot.cost(
        provider="example", model="model-a", at=AFTER, input_tokens=1, output_tokens=1
    )
    assert result.model_class == DECISION


@pytest.mark.parametrize("value", ["embedding", "Decision", "", None, 1])
def test_a_class_outside_the_closed_set_is_refused(value):
    with pytest.raises(CatalogError, match="model_class"):
        load_catalog_document(_document(model_class=value))


def test_an_unresolved_call_has_no_class():
    result = SNAPSHOT.cost(
        provider="example", model="not-a-model", at=AFTER, input_tokens=1, output_tokens=1
    )
    assert result.status == "unpriced"
    assert result.model_class is None
    assert CATALOG.model_class("example/not-a-model") is None


def test_the_class_reaches_every_token_entry_point():
    channel = CATALOG.lookup_price(
        model="typesafe-ai/jev-1.14.0", at=AFTER, channel="typesafe-api"
    )
    assert channel.status == "priced"
    planned = CATALOG.price(
        model="jev-1.14.0", channel="typesafe-api", at=AFTER, input_tokens=10, output_tokens=None
    )
    assert planned.model_class == DECISION
    computed = CATALOG.calculate_cost(
        model="jev-1.14.0", usage=ModelUsage(input_tokens=10), at=AFTER, provider="typesafe"
    )
    assert computed.model_class == DECISION
    llm = CATALOG.calculate_cost(
        model="gpt-4o-mini", usage=ModelUsage(10, 10), at=AFTER, provider="openai"
    )
    assert llm.model_class == GENERATIVE


# ── the zero-rate rule ──────────────────────────────────────────────────


def test_missing_output_is_not_needed_when_output_is_free():
    catalog = load_catalog_document(
        _document(
            prices=[
                {
                    "channel": "example-api",
                    "effective_from": "2026-01-01",
                    "input_per_mtok": 1,
                    "output_per_mtok": 0,
                    "source_url": "https://example.test/prices",
                }
            ]
        )
    )
    result = catalog.snapshot.cost(
        provider="example", model="model-a", at=AFTER, input_tokens=1000, output_tokens=None
    )
    assert result.status == "priced"
    assert result.reasons == ()
    assert result.missing_dimensions == ()
    assert result.cost_usd == Decimal("0.00100000")


def test_missing_output_still_makes_a_priced_output_partial():
    catalog = load_catalog_document(_document())
    result = catalog.snapshot.cost(
        provider="example", model="model-a", at=AFTER, input_tokens=1000, output_tokens=None
    )
    assert result.status == "partial"
    assert "missing_output_tokens" in result.reasons
    assert result.missing_dimensions == ("output_tokens",)


def test_missing_output_with_no_output_rate_at_all_stays_partial():
    # An absent rate is unknown, not free.
    catalog = load_catalog_document(
        _document(
            prices=[
                {
                    "channel": "example-api",
                    "effective_from": "2026-01-01",
                    "input_per_mtok": 1,
                    "source_url": "https://example.test/prices",
                }
            ]
        )
    )
    result = catalog.snapshot.cost(
        provider="example", model="model-a", at=AFTER, input_tokens=1000, output_tokens=None
    )
    assert result.status == "partial"
    assert "missing_output_tokens" in result.reasons


def test_missing_input_is_not_needed_when_input_is_free():
    catalog = load_catalog_document(
        _document(
            prices=[
                {
                    "channel": "example-api",
                    "effective_from": "2026-01-01",
                    "input_per_mtok": 0,
                    "output_per_mtok": 2,
                    "source_url": "https://example.test/prices",
                }
            ]
        )
    )
    result = catalog.snapshot.cost(
        provider="example", model="model-a", at=AFTER, input_tokens=None, output_tokens=500
    )
    assert result.status == "priced"
    assert result.cost_usd == Decimal("0.00100000")


def test_missing_input_on_a_free_input_with_a_long_context_tier_stays_partial():
    # The tier reads the input count, so it cannot be skipped.
    catalog = load_catalog_document(
        _document(
            prices=[
                {
                    "channel": "example-api",
                    "effective_from": "2026-01-01",
                    "input_per_mtok": 0,
                    "output_per_mtok": 2,
                    "rules": {"long_context": {"threshold": 100, "output_multiplier": 2}},
                    "source_url": "https://example.test/prices",
                }
            ]
        )
    )
    result = catalog.snapshot.cost(
        provider="example", model="model-a", at=AFTER, input_tokens=None, output_tokens=500
    )
    assert result.status == "partial"
    assert "missing_input_tokens" in result.reasons


def test_the_rule_does_not_branch_on_class():
    # A decision entry with a priced output still needs its output count.
    catalog = load_catalog_document(_document(model_class="decision"))
    result = catalog.snapshot.cost(
        provider="example", model="model-a", at=AFTER, input_tokens=1000, output_tokens=None
    )
    assert result.status == "partial"


# ── Jev ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "provider,model,channel",
    [
        ("vercel", "typesafe-ai/jev", "vercel-ai-gateway"),
        ("gateway", "typesafe-ai/jev", "vercel-ai-gateway"),
        ("typesafe-ai", "typesafe-ai/jev", "vercel-ai-gateway"),
        ("typesafe", "jev-latest", "typesafe-api"),
        ("typesafe", "jev-preview", "typesafe-api"),
        ("typesafe-ai", "jev", "typesafe-api"),
        ("unknown", "typesafe/jev", "vercel-ai-gateway"),
    ],
)
def test_version_neutral_ids_resolve_to_the_family_entry(provider, model, channel):
    result = SNAPSHOT.cost(
        provider=provider, model=model, at=AFTER, input_tokens=1000, output_tokens=None
    )
    assert result.canonical_model == "typesafe-ai/jev"
    assert result.model_class == DECISION
    assert result.price.pricing_channel == channel
    assert result.status == "priced"
    assert result.cost_usd == _per_million(1000, JEV_RATE)


@pytest.mark.parametrize("version", ["1.13.0", "1.14.0"])
@pytest.mark.parametrize(
    "provider,spelling,channel",
    [
        ("vercel", "typesafe-ai/jev-{v}", "vercel-ai-gateway"),
        ("gateway", "typesafe-ai/jev-{v}", "vercel-ai-gateway"),
        ("typesafe-ai", "typesafe-ai/jev-{v}", "vercel-ai-gateway"),
        ("typesafe", "jev-{v}", "typesafe-api"),
        ("typesafe", "typesafe/jev-{v}", "typesafe-api"),
        ("typesafe-ai", "jev-{v}", "typesafe-api"),
        ("unknown", "jev-{v}", "vercel-ai-gateway"),
        ("Unknown", "typesafe/jev-{v}", "vercel-ai-gateway"),
    ],
)
def test_versioned_ids_resolve_on_every_provider(version, provider, spelling, channel):
    result = SNAPSHOT.cost(
        provider=provider,
        model=spelling.format(v=version),
        at=AFTER,
        input_tokens=2500,
        output_tokens=None,
    )
    assert result.canonical_model == f"typesafe-ai/jev-{version}"
    assert result.model_class == DECISION
    assert result.price.pricing_channel == channel
    assert result.status == "priced"
    assert result.cost_usd == _per_million(2500, JEV_RATE)
    assert CATALOG.model_family(result.canonical_model).id == "typesafe-ai/jev"


@pytest.mark.parametrize("provider", [None, "", "  "])
def test_a_missing_provider_resolves_through_the_unknown_aliases(provider):
    result = SNAPSHOT.cost(
        provider=provider, model="jev-1.13.0", at=AFTER, input_tokens=1000, output_tokens=None
    )
    assert result.canonical_model == "typesafe-ai/jev-1.13.0"
    assert result.status == "priced"


def test_a_missing_provider_matches_only_unknown_aliases():
    # This id has an alias only for provider vercel.
    result = SNAPSHOT.cost(
        provider=None, model="anthropic/claude-sonnet-4.6-batch", at=AFTER, input_tokens=1, output_tokens=1
    )
    assert result.status == "unpriced"


def test_gateway_and_direct_calls_get_the_same_family_class_and_cost():
    gateway = price_call_from_record(
        {"provider": "vercel", "model": "typesafe-ai/jev", "input_tokens": 1234}, at=AFTER
    )
    direct = price_call_from_record(
        {"provider": "typesafe", "model": "jev-1.14.0", "input_tokens": 1234}, at=AFTER
    )
    for result in (gateway, direct):
        assert result.status == "priced"
        assert result.token_cost.model_class == DECISION
        assert CATALOG.model_family(result.token_cost.canonical_model).id == "typesafe-ai/jev"
    # 1234 input tokens at $0.042 per million.
    assert gateway.cost_usd == direct.cost_usd == Decimal("0.00005183")


def test_a_jev_call_without_input_tokens_is_still_partial():
    result = SNAPSHOT.cost(
        provider="typesafe", model="jev-1.14.0", at=AFTER, input_tokens=None, output_tokens=None
    )
    assert result.status == "partial"
    assert "missing_input_tokens" in result.reasons


def test_reported_output_tokens_are_kept_at_no_cost():
    result = SNAPSHOT.cost(
        provider="typesafe", model="jev-1.14.0", at=AFTER, input_tokens=1000, output_tokens=9
    )
    assert result.status == "priced"
    assert result.components.output == Decimal("0")
    assert result.cost_usd == _per_million(1000, JEV_RATE)


def test_the_gateway_rate_change_is_a_new_row_not_an_edit():
    before = SNAPSHOT.cost(
        provider="vercel", model="typesafe-ai/jev-1.13.0", at=BEFORE,
        input_tokens=1_000_000, output_tokens=0,
    )
    after = SNAPSHOT.cost(
        provider="vercel", model="typesafe-ai/jev-1.13.0", at=AFTER,
        input_tokens=1_000_000, output_tokens=0,
    )
    assert before.cost_usd == Decimal("0.04")
    assert after.cost_usd == Decimal("0.042")


def test_the_direct_channel_starts_with_its_published_rate():
    before = SNAPSHOT.cost(
        provider="typesafe", model="jev-1.13.0", at=BEFORE, input_tokens=1, output_tokens=0
    )
    assert before.status == "unpriced"


def test_an_unknown_decision_model_stays_unpriced():
    result = SNAPSHOT.cost(
        provider="typesafe", model="jev-9.9.9", at=AFTER, input_tokens=1000, output_tokens=None
    )
    assert result.status == "unpriced"
    assert result.cost_usd is None


# ── generative pricing is unchanged ─────────────────────────────────────


@pytest.mark.parametrize(
    "provider,model,input_rate,output_rate",
    [
        ("openai", "gpt-4o-mini", "0.15", "0.6"),
        ("anthropic", "claude-sonnet-4-6", "3", "15"),
        ("google", "gemini-2.5-flash", "0.3", "2.5"),
        ("openai", "gpt-5.4", "2.5", "15"),
    ],
)
def test_existing_llm_prices_are_unchanged(provider, model, input_rate, output_rate):
    result = SNAPSHOT.cost(
        provider=provider, model=model, at=AFTER, input_tokens=12_000, output_tokens=3_000
    )
    expected = _per_million(12_000, Decimal(input_rate)) + _per_million(3_000, Decimal(output_rate))
    assert result.status == "priced"
    assert result.model_class == GENERATIVE
    assert result.cost_usd == expected

    missing = SNAPSHOT.cost(
        provider=provider, model=model, at=AFTER, input_tokens=12_000, output_tokens=None
    )
    assert missing.status == "partial"
    assert missing.missing_dimensions == ("output_tokens",)
    assert missing.cost_usd == _per_million(12_000, Decimal(input_rate))


# ── reliability ─────────────────────────────────────────────────────────


def test_a_successful_decision_call_without_a_finish_reason_is_available():
    outcome = classify_recorded_call({"provider": "typesafe", "status": "success", "model_class": "decision"})
    assert outcome.availability == "available"
    assert outcome.refused is None


def test_a_successful_generative_call_without_a_finish_reason_stays_unknown():
    for row in (
        {"provider": "openai", "status": "success", "model_class": "generative"},
        {"provider": "openai", "status": "success"},
    ):
        assert classify_recorded_call(row).availability is None


@pytest.mark.parametrize(
    "row",
    [
        {"status": "error", "model_class": "decision"},
        {"status": "success", "error": True, "model_class": "decision"},
        {"status": "success", "error_type": "PermissionDeniedError", "model_class": "decision"},
    ],
)
def test_a_failed_decision_call_is_not_counted_available(row):
    assert classify_recorded_call(row).availability is None


def test_a_rate_limited_decision_call_stays_unavailable():
    outcome = classify_recorded_call(
        {"status": "success", "model_class": "decision", "http_status_code": 429}
    )
    assert outcome.availability == "unavailable"


# ── the model registry ──────────────────────────────────────────────────


def _registry(**model):
    return {
        "version": "2026-11-22",
        "models": [
            {
                "canonical_id": "typesafe-ai/jev-1.14.0",
                "display_name": "Jev 1.14",
                "publisher": "typesafe-ai",
                **model,
            }
        ],
        "execution_profiles": [],
        "offer_groups": [],
    }


def test_the_registry_reads_the_class_and_defaults_to_generative():
    assert parse_model_registry(_registry()).model("typesafe-ai/jev-1.14.0").model_class == GENERATIVE
    declared = parse_model_registry(_registry(model_class="decision"))
    assert declared.model("typesafe-ai/jev-1.14.0").model_class == DECISION


def test_the_registry_refuses_an_unknown_class():
    with pytest.raises(ModelRegistryError, match="model_class"):
        parse_model_registry(_registry(model_class="classifier"))


def test_the_registry_class_must_agree_with_the_catalog():
    validate_model_registry(parse_model_registry(_registry(model_class="decision")), CATALOG)
    with pytest.raises(ModelRegistryError, match="differs from the pricing catalog"):
        validate_model_registry(parse_model_registry(_registry()), CATALOG)
