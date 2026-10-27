from datetime import datetime, timezone
from decimal import Decimal

import pytest

from metergraph_core import CatalogError, load_catalog, parse_retrieval


def _retrieval_entry(**overrides):
    entry = {
        "channel": "anthropic-api",
        "operation": "web_search",
        "region": "global",
        "unit": "completed_search",
        "per_1k_usd": "10.00",
        "effective_from": "2026-08-26",
        "source_url": "https://example.test/anthropic-search",
    }
    entry.update(overrides)
    return {"retrieval": [entry]}


def test_parse_retrieval_rejects_missing_unit():
    entry = _retrieval_entry()
    del entry["retrieval"][0]["unit"]
    with pytest.raises(CatalogError, match="needs a unit"):
        parse_retrieval(entry)


def test_parse_retrieval_rejects_blank_unit():
    with pytest.raises(CatalogError, match="needs a unit"):
        parse_retrieval(_retrieval_entry(unit="   "))


def test_parse_retrieval_accepts_a_well_formed_entry():
    [price] = parse_retrieval(_retrieval_entry())
    assert price.unit == "completed_search"
    assert price.id == "anthropic-api:web_search:global:2026-08-26"


def test_bundled_catalog_has_identity_and_prices_a_call():
    loaded = load_catalog()
    assert loaded.version == "2026-10-27"
    assert len(loaded.content_hash) == 64
    result = loaded.snapshot.cost(
        provider="openai",
        model="gpt-5.4-mini",
        at=datetime(2026, 8, 17, tzinfo=timezone.utc),
        input_tokens=100_000,
        output_tokens=100_000,
    )
    assert result.canonical_model == "openai/gpt-5.4-mini"
    assert result.price_id == "openai/gpt-5.4-mini:openai-api:global:2026-03-17"
    assert result.cost_usd == Decimal("0.52500000")
    assert result.status == "priced"
    assert result.reasons == ()


@pytest.mark.parametrize(
    ("model", "input_rate", "output_rate", "cache_read_rate"),
    [
        ("openai/gpt-6-luna", "0.10", "0.50", "0.01"),
        ("openai/gpt-6-sol", "2.00", "10.00", "0.20"),
        ("openai/gpt-6-astra", "10.00", "50.00", "1.00"),
    ],
)
def test_bundled_catalog_prices_gpt6_gateway_candidates(
    model, input_rate, output_rate, cache_read_rate,
):
    loaded = load_catalog()
    result = loaded.price(
        model=model,
        channel="vercel-ai-gateway",
        at=datetime(2026, 9, 23, 12, tzinfo=timezone.utc),
        input_tokens=1_000,
        output_tokens=1_000,
        cache_read_tokens=1_000,
    )
    assert result.status == "priced"
    assert result.canonical_model == model
    assert result.cost_usd == (
        Decimal(output_rate) + Decimal(cache_read_rate)
    ) / 1_000


@pytest.mark.parametrize(
    ("model", "input_rate", "output_rate", "cache_read_rate", "batch_output_rate"),
    [
        ("openai/gpt-6-luna", "0.10", "0.50", "0.01", "0.25"),
        ("openai/gpt-6-sol", "2.00", "10.00", "0.20", "5.00"),
        ("openai/gpt-6-astra", "10.00", "50.00", "1.00", "25.00"),
    ],
)
def test_bundled_catalog_prices_gpt6_on_the_direct_openai_api(
    model, input_rate, output_rate, cache_read_rate, batch_output_rate,
):
    # OpenAI reports cached tokens inside input, so 2k input with 1k cached
    # bills 1k at the input rate and 1k at the cache-read rate.
    loaded = load_catalog()
    at = datetime(2026, 9, 28, 12, tzinfo=timezone.utc)
    result = loaded.price(
        model=model,
        channel="openai-api",
        at=at,
        input_tokens=2_000,
        output_tokens=1_000,
        cache_read_tokens=1_000,
    )
    assert result.status == "priced"
    assert result.canonical_model == model
    assert result.price_id.startswith(f"{model}:openai-api:global:")
    assert result.cost_usd == (
        Decimal(input_rate) + Decimal(output_rate) + Decimal(cache_read_rate)
    ) / 1_000

    batch = loaded.price(
        model=model,
        channel="openai-api",
        at=at,
        input_tokens=0,
        output_tokens=1_000,
        batch=True,
    )
    assert batch.cost_usd == Decimal(batch_output_rate) / 1_000


@pytest.mark.parametrize(
    ("provider", "model", "canonical"),
    [
        ("openai", "gpt-6-luna", "openai/gpt-6-luna"),
        ("litellm", "gpt-6-sol", "openai/gpt-6-sol"),
        ("unknown", "gpt-6-astra", "openai/gpt-6-astra"),
    ],
)
def test_bare_gpt6_names_resolve_to_the_direct_openai_api(provider, model, canonical):
    loaded = load_catalog()
    result = loaded.snapshot.cost(
        provider=provider,
        model=model,
        at=datetime(2026, 9, 28, 12, tzinfo=timezone.utc),
        input_tokens=1_000_000,
        output_tokens=0,
    )
    assert result.status == "priced"
    assert result.canonical_model == canonical
    assert ":openai-api:" in result.price_id


def test_gpt6_luna_direct_long_context_doubles_input_and_lifts_output():
    loaded = load_catalog()
    result = loaded.price(
        model="openai/gpt-6-luna",
        channel="openai-api",
        at=datetime(2026, 9, 28, 12, tzinfo=timezone.utc),
        input_tokens=300_000,
        output_tokens=100_000,
    )
    assert result.status == "priced"
    # 0.3M x $0.20 + 0.1M x $0.75, the published >272K rates.
    assert result.cost_usd == Decimal("0.06") + Decimal("0.075")


# (input, output, cache read, 5m write, 1h write, batch output or None)
_ANTHROPIC_ROWS_2026_09_28 = [
    ("anthropic/claude-opus-5.5", "anthropic-api",
     ("4.00", "20.00", "0.20", "5.00", "8.00", "10.00")),
    ("anthropic/claude-opus-5.5", "aws-bedrock",
     ("4.00", "20.00", "0.20", "5.00", "8.00", None)),
    ("anthropic/claude-opus-5.5", "aws-bedrock-geo",
     ("4.40", "22.00", "0.22", "5.50", "8.80", None)),
    ("anthropic/claude-haiku-4.5", "aws-bedrock",
     ("1.00", "5.00", "0.10", "1.25", "2.00", "2.50")),
    ("anthropic/claude-haiku-4.5", "aws-bedrock-geo",
     ("1.10", "5.50", "0.11", "1.375", "2.20", "2.75")),
    ("anthropic/claude-fable-5.1", "aws-bedrock-geo",
     ("11.00", "55.00", "0.275", "13.75", "22.00", None)),
]


@pytest.mark.parametrize(("model", "channel", "rates"), _ANTHROPIC_ROWS_2026_09_28)
def test_bundled_catalog_prices_anthropic_rows_added_2026_09_28(model, channel, rates):
    # Anthropic and Bedrock report input exclusive of cache reads and writes,
    # so each 1k-token bucket bills at its own rate.
    input_rate, output_rate, read_rate, write_5m_rate, write_1h_rate, batch_out = rates
    loaded = load_catalog()
    at = datetime(2026, 9, 28, 12, tzinfo=timezone.utc)
    result = loaded.price(
        model=model,
        channel=channel,
        at=at,
        input_tokens=1_000,
        output_tokens=1_000,
        cache_read_tokens=1_000,
        cache_write_5m_tokens=1_000,
        cache_write_1h_tokens=1_000,
    )
    assert result.status == "priced"
    assert result.canonical_model == model
    assert result.price_id.startswith(f"{model}:{channel}:global:")
    assert result.cost_usd == sum(
        Decimal(rate)
        for rate in (input_rate, output_rate, read_rate, write_5m_rate, write_1h_rate)
    ) / 1_000

    batch = loaded.price(
        model=model,
        channel=channel,
        at=at,
        input_tokens=0,
        output_tokens=1_000,
        batch=True,
    )
    if batch_out is None:
        # No published batch rate: flagged, not presented as a batch price.
        assert batch.status == "partial"
        assert batch.reasons == ("batch_rate_unavailable",)
    else:
        assert batch.cost_usd == Decimal(batch_out) / 1_000


def test_opus_5_5_gateway_row_prices_the_published_rates():
    # The gateway reports cached tokens inside input.
    loaded = load_catalog()
    result = loaded.price(
        model="anthropic/claude-opus-5.5",
        channel="vercel-ai-gateway",
        at=datetime(2026, 9, 28, 12, tzinfo=timezone.utc),
        input_tokens=2_000,
        output_tokens=1_000,
        cache_read_tokens=1_000,
    )
    assert result.status == "priced"
    assert result.cost_usd == (
        Decimal("4.00") + Decimal("20.00") + Decimal("0.20")
    ) / 1_000


@pytest.mark.parametrize(
    ("provider", "model", "canonical", "channel"),
    [
        ("anthropic", "claude-opus-5-5", "anthropic/claude-opus-5.5", "anthropic-api"),
        ("litellm", "claude-opus-5-5", "anthropic/claude-opus-5.5", "anthropic-api"),
        ("unknown", "claude-opus-5-5", "anthropic/claude-opus-5.5", "anthropic-api"),
        (
            "bedrock", "global.anthropic.claude-opus-5-5",
            "anthropic/claude-opus-5.5", "aws-bedrock",
        ),
        (
            "bedrock", "us.anthropic.claude-opus-5-5",
            "anthropic/claude-opus-5.5", "aws-bedrock-geo",
        ),
        (
            "bedrock", "global.anthropic.claude-haiku-4-5-20251001-v1:0",
            "anthropic/claude-haiku-4.5", "aws-bedrock",
        ),
        (
            "bedrock", "anthropic.claude-haiku-4-5",
            "anthropic/claude-haiku-4.5", "aws-bedrock",
        ),
        (
            "bedrock", "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            "anthropic/claude-haiku-4.5", "aws-bedrock-geo",
        ),
        (
            "bedrock", "us.anthropic.claude-fable-5-1",
            "anthropic/claude-fable-5.1", "aws-bedrock-geo",
        ),
    ],
)
def test_anthropic_ids_from_traffic_resolve_to_their_channel(
    provider, model, canonical, channel,
):
    loaded = load_catalog()
    result = loaded.snapshot.cost(
        provider=provider,
        model=model,
        at=datetime(2026, 9, 28, 12, tzinfo=timezone.utc),
        input_tokens=1_000_000,
        output_tokens=0,
    )
    assert result.status == "priced"
    assert result.canonical_model == canonical
    assert f":{channel}:" in result.price_id


def test_fable_5_1_has_no_global_bedrock_price():
    # AWS publishes Fable 5.1 only for geo/in-region inference.
    loaded = load_catalog()
    result = loaded.snapshot.cost(
        provider="bedrock",
        model="global.anthropic.claude-fable-5-1",
        at=datetime(2026, 9, 28, 12, tzinfo=timezone.utc),
        input_tokens=1_000_000,
        output_tokens=0,
    )
    assert result.status == "unpriced"


@pytest.mark.parametrize(
    ("model", "input_rate", "output_rate", "cache_read_rate"),
    [
        ("mistral/ministral-14b", "0.20", "0.20", "0.02"),
        ("google/gemini-3.8-flash", "0.75", "3.75", "0.075"),
        ("google/gemini-3.5-flash-lite", "0.30", "2.50", "0.03"),
        ("anthropic/claude-fable-5.1", "10.00", "50.00", "0.25"),
    ],
)
def test_bundled_catalog_prices_gateway_rows_added_2026_09_24(
    model, input_rate, output_rate, cache_read_rate,
):
    # The gateway reports cached tokens inside input, so 1k input of which 1k
    # was cached bills only the cache-read rate for input.
    loaded = load_catalog()
    result = loaded.price(
        model=model,
        channel="vercel-ai-gateway",
        at=datetime(2026, 9, 24, 12, tzinfo=timezone.utc),
        input_tokens=1_000,
        output_tokens=1_000,
        cache_read_tokens=1_000,
    )
    assert result.status == "priced"
    assert result.canonical_model == model
    assert result.cost_usd == (
        Decimal(output_rate) + Decimal(cache_read_rate)
    ) / 1_000


@pytest.mark.parametrize(
    ("model", "channel", "canonical", "input_rate", "output_rate"),
    [
        ("ministral-14b-2512", "mistral-api", "mistral/ministral-14b", "0.20", "0.20"),
        ("ministral-14b-latest", "mistral-api", "mistral/ministral-14b", "0.20", "0.20"),
        (
            "mistral.ministral-3-14b-instruct", "aws-bedrock",
            "mistral/ministral-14b", "0.20", "0.20",
        ),
        (
            "claude-fable-5-1", "anthropic-api",
            "anthropic/claude-fable-5.1", "10.00", "50.00",
        ),
    ],
)
def test_bundled_catalog_prices_direct_rows_added_2026_09_24(
    model, channel, canonical, input_rate, output_rate,
):
    loaded = load_catalog(region="us-west-2")
    result = loaded.price(
        model=model,
        channel=channel,
        at=datetime(2026, 9, 24, 12, tzinfo=timezone.utc),
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )
    assert result.status == "priced"
    assert result.canonical_model == canonical
    assert result.cost_usd == Decimal(input_rate) + Decimal(output_rate)


@pytest.mark.parametrize(
    ("model", "canonical", "input_rate", "output_rate"),
    [
        ("us.openai.gpt-5.6-sol", "openai/gpt-5.6-sol", "5.50", "33.00"),
        ("us.openai.gpt-5.6-terra", "openai/gpt-5.6-terra", "2.20", "13.20"),
        ("us.openai.gpt-5.6-luna", "openai/gpt-5.6-luna", "0.22", "1.32"),
        ("us.anthropic.claude-sonnet-5", "anthropic/claude-sonnet-5", "2.00", "10.00"),
        ("google.gemma-3-27b-it", "google/gemma-3-27b-it", "0.23", "0.38"),
        ("moonshotai.kimi-k2.5", "moonshotai/kimi-k2.5", "0.60", "3.00"),
        ("deepseek.v3.2", "deepseek/v3.2", "0.62", "1.85"),
        ("deepseek.v3-v1:0", "deepseek/v3.1", "0.58", "1.68"),
    ],
)
def test_bundled_catalog_prices_aws_bedrock_analysis_models(
    model, canonical, input_rate, output_rate,
):
    loaded = load_catalog(region="us-west-2")
    result = loaded.price(
        model=model,
        channel="aws-bedrock",
        at=datetime(2026, 8, 27, 12, tzinfo=timezone.utc),
        input_tokens=100_000,
        output_tokens=1_000_000,
    )
    assert result.status == "priced"
    assert result.canonical_model == canonical
    assert result.cost_usd == Decimal(input_rate) / 10 + Decimal(output_rate)
    assert loaded.canonical_model_id("bedrock", model) == canonical


def test_sonnet_5_5_prices_the_published_direct_rates():
    loaded = load_catalog()
    at = datetime(2026, 9, 29, 12, tzinfo=timezone.utc)
    result = loaded.price(
        model="claude-sonnet-5-5",
        channel="anthropic-api",
        at=at,
        input_tokens=1_000,
        output_tokens=1_000,
        cache_read_tokens=1_000,
        cache_write_5m_tokens=1_000,
        cache_write_1h_tokens=1_000,
    )
    assert result.status == "priced"
    assert result.canonical_model == "anthropic/claude-sonnet-5.5"
    assert result.price_id == "anthropic/claude-sonnet-5.5:anthropic-api:global:2026-09-28"
    assert result.cost_usd == sum(
        Decimal(rate) for rate in ("2.00", "10.00", "0.20", "2.50", "4.00")
    ) / 1_000

    batch = loaded.price(
        model="claude-sonnet-5-5", channel="anthropic-api", at=at,
        input_tokens=1_000, output_tokens=1_000, batch=True,
    )
    assert batch.cost_usd == (Decimal("1.00") + Decimal("5.00")) / 1_000


def test_sonnet_5_5_gateway_row_prices_the_published_rates():
    # The gateway reports cached tokens inside input.
    loaded = load_catalog()
    result = loaded.price(
        model="anthropic/claude-sonnet-5.5",
        channel="vercel-ai-gateway",
        at=datetime(2026, 9, 29, 12, tzinfo=timezone.utc),
        input_tokens=2_000,
        output_tokens=1_000,
        cache_read_tokens=1_000,
    )
    assert result.status == "priced"
    assert result.cost_usd == (
        Decimal("2.00") + Decimal("10.00") + Decimal("0.20")
    ) / 1_000


def test_sonnet_5_5_is_not_priced_before_its_release():
    loaded = load_catalog()
    result = loaded.price(
        model="claude-sonnet-5-5",
        channel="anthropic-api",
        at=datetime(2026, 9, 27, 12, tzinfo=timezone.utc),
        input_tokens=1_000,
        output_tokens=1_000,
    )
    assert result.status != "priced"


_VERTEX_GEMINI_FLASH = ["gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash"]


@pytest.mark.parametrize("model", _VERTEX_GEMINI_FLASH)
def test_vertex_prices_gemini_flash_at_the_introductory_rate(model):
    loaded = load_catalog()
    result = loaded.snapshot.cost(
        provider="vertex-ai",
        model=model,
        at=datetime(2026, 9, 29, 12, tzinfo=timezone.utc),
        input_tokens=2_000,
        output_tokens=1_000,
        cache_read_tokens=1_000,
    )
    assert result.status == "priced"
    assert result.canonical_model == f"google/{model}"
    assert result.price_id.startswith(f"google/{model}:google-vertex-ai:global:")
    assert result.cost_usd == (
        Decimal("0.75") + Decimal("3.75") + Decimal("0.075")
    ) / 1_000


@pytest.mark.parametrize("model", _VERTEX_GEMINI_FLASH)
def test_vertex_gemini_flash_doubles_in_2027(model):
    loaded = load_catalog()
    result = loaded.price(
        model=model,
        channel="google-vertex-ai",
        at=datetime(2027, 1, 1, tzinfo=timezone.utc),
        input_tokens=1_000,
        output_tokens=1_000,
    )
    assert result.status == "priced"
    assert result.price_id == f"google/{model}:google-vertex-ai:global:2027-01-01"
    assert result.cost_usd == (Decimal("1.50") + Decimal("7.50")) / 1_000


@pytest.mark.parametrize("model", _VERTEX_GEMINI_FLASH)
def test_a_vertex_row_leaves_the_direct_channel_unambiguous(model):
    # Search replay derives a workload's provider from its model's one direct
    # channel; the Vertex alias must not make that ambiguous.
    assert load_catalog().infer_direct_channel(model) == "google-api"


# --- disputes -----------------------------------------------------------------

from metergraph_core import parse_disputes


def _dispute_entry(**overrides):
    entry = {
        "gateway": "portkey",
        "publisher": "deepseek",
        "since": "2026-09-10",
        "note": "Stale table.",
        "source_url": "https://example.test/deepseek",
    }
    entry.update(overrides)
    return {"disputes": [entry]}


def test_parse_disputes_reads_an_entry_and_folds_the_names():
    (dispute,) = parse_disputes(_dispute_entry(gateway=" Portkey ", publisher="DeepSeek"))
    assert (dispute.gateway, dispute.publisher) == ("portkey", "deepseek")
    assert dispute.since.date().isoformat() == "2026-09-10"
    assert dispute.note == "Stale table."


def test_parse_disputes_is_empty_without_the_section():
    assert parse_disputes({"models": []}) == []


@pytest.mark.parametrize(
    "overrides",
    [{"gateway": ""}, {"publisher": None}, {"note": "  "}, {"source_url": ""}, {"since": "soon"}],
)
def test_parse_disputes_rejects_an_incomplete_entry(overrides):
    with pytest.raises(CatalogError):
        parse_disputes(_dispute_entry(**overrides))


def test_parse_disputes_rejects_a_non_list():
    with pytest.raises(CatalogError):
        parse_disputes({"disputes": {"gateway": "portkey"}})
