"""The public pricing source registry: its schema, its seeded entries, and
their fixtures."""

import re
from datetime import date
from pathlib import Path

import pytest
import yaml

from metergraph_core import load_catalog, load_catalog_document
from metergraph_core.sources import (
    SOURCE_REGISTRY_SCHEMA_VERSION,
    SourceRegistryError,
    load_source_registry,
    parse_source_registry,
    validate_source_registry,
)


def _entry(**overrides):
    entry = {
        "id": "example-api-pricing",
        "provider": "openai",
        "channel": "openai-api",
        "catalog_channel": True,
        "url": "https://example.com/pricing",
        "parser": "openai",
        "dimensions": ["input_per_mtok", "output_per_mtok", "cache_read_per_mtok"],
        "fingerprint": "sha256:" + "a" * 64,
        "low_risk": {"eligible": True, "dimensions": ["input_per_mtok", "output_per_mtok"]},
    }
    entry.update(overrides)
    return entry


def _registry(*entries, **overrides):
    document = {
        "schema_version": SOURCE_REGISTRY_SCHEMA_VERSION,
        "sources": list(entries) if entries else [_entry()],
    }
    document.update(overrides)
    return document


def _not_low_risk(**overrides):
    return _entry(
        low_risk={"eligible": False, "dimensions": []},
        **overrides,
    )


def test_parse_reads_an_entry():
    registry = parse_source_registry(_registry())
    source = registry.get("example-api-pricing")
    assert source.provider == "openai"
    assert source.channel == "openai-api"
    assert source.url == "https://example.com/pricing"
    assert source.parser == "openai"
    assert source.dimensions == ("input_per_mtok", "output_per_mtok", "cache_read_per_mtok")
    assert source.low_risk.eligible is True
    assert source.low_risk.dimensions == ("input_per_mtok", "output_per_mtok")


def test_get_rejects_an_unknown_source():
    with pytest.raises(SourceRegistryError, match="unknown source 'missing'"):
        parse_source_registry(_registry()).get("missing")


def test_low_risk_lists_only_eligible_sources():
    registry = parse_source_registry(
        _registry(_entry(), _not_low_risk(id="other-pricing", url="https://example.com/other"))
    )
    assert [source.id for source in registry.low_risk()] == ["example-api-pricing"]


def test_allowed_urls_is_exactly_the_registered_urls():
    registry = parse_source_registry(
        _registry(
            _entry(),
            _not_low_risk(id="other-pricing", url="https://example.com/other"),
        )
    )
    assert registry.allowed_urls() == frozenset(
        {"https://example.com/pricing", "https://example.com/other"}
    )


@pytest.mark.parametrize(
    ("document", "message"),
    [
        (["not", "a", "mapping"], "source registry must be a mapping"),
        (_registry(schema_version=2), "unsupported source registry schema_version 2"),
        (_registry(sources=[]), "source registry needs a non-empty sources list"),
        (_registry(_entry(), _entry()), "duplicate source id 'example-api-pricing'"),
        (_registry(_entry(id="Bad Id")), "invalid source id 'Bad Id'"),
        (_registry(_entry(provider="")), "example-api-pricing: needs provider"),
        (_registry(_entry(channel=None)), "example-api-pricing: needs channel"),
        (_registry(_entry(catalog_channel="yes")), "catalog_channel must be true or false"),
        (_registry(_entry(url="http://example.com/pricing")), "url must be an https URL"),
        (_registry(_entry(url="https://")), "url must be an https URL"),
        (_registry(_entry(parser="scraper")), "unknown parser 'scraper'"),
        (_registry(_entry(dimensions=["input_per_mtok", "vibes"])), "unknown dimension 'vibes'"),
        (_registry(_entry(dimensions=["input_per_mtok", "input_per_mtok"])), "duplicate dimension"),
        (_registry(_entry(fingerprint="abc")), "fingerprint must be null or sha256:"),
    ],
)
def test_parse_rejects_malformed_registries(document, message):
    with pytest.raises(SourceRegistryError, match=message):
        parse_source_registry(document)


def test_a_missing_fingerprint_is_allowed():
    assert parse_source_registry(_registry(_entry(fingerprint=None))).get(
        "example-api-pricing"
    ).fingerprint is None


def test_low_risk_requires_a_catalog_channel():
    with pytest.raises(SourceRegistryError, match="low-risk source needs a catalog channel"):
        parse_source_registry(_registry(_entry(catalog_channel=False)))


def test_low_risk_requires_a_parser_that_emits_price_rows():
    with pytest.raises(SourceRegistryError, match="parser 'html-tables' emits no price rows"):
        parse_source_registry(_registry(_entry(parser="html-tables")))


def test_low_risk_dimensions_must_be_dimensions_the_source_represents():
    low_risk = {"eligible": True, "dimensions": ["input_per_mtok", "batch_input_per_mtok"]}
    with pytest.raises(SourceRegistryError, match="low-risk dimension 'batch_input_per_mtok' is not a source dimension"):
        parse_source_registry(_registry(_entry(low_risk=low_risk)))


def test_low_risk_needs_at_least_one_dimension():
    low_risk = {"eligible": True, "dimensions": []}
    with pytest.raises(SourceRegistryError, match="low-risk source needs dimensions"):
        parse_source_registry(_registry(_entry(low_risk=low_risk)))


def test_a_source_that_is_not_low_risk_allows_no_dimensions():
    low_risk = {"eligible": False, "dimensions": ["input_per_mtok"]}
    with pytest.raises(SourceRegistryError, match="non-low-risk source allows no low-risk dimensions"):
        parse_source_registry(_registry(_entry(low_risk=low_risk)))


@pytest.mark.parametrize("field", ["max_relative_change", "bound"])
def test_low_risk_rejects_fields_beyond_eligibility_and_dimensions(field):
    low_risk = {"eligible": True, "dimensions": ["input_per_mtok"], field: "0.5"}
    with pytest.raises(SourceRegistryError, match=f"example-api-pricing: low_risk has unknown field '{field}'"):
        parse_source_registry(_registry(_entry(low_risk=low_risk)))


def test_low_risk_eligibility_must_be_a_boolean():
    with pytest.raises(SourceRegistryError, match="low_risk.eligible must be true or false"):
        parse_source_registry(_registry(_entry(low_risk={"eligible": "yes"})))


def test_a_gpu_or_license_source_records_its_non_token_dimensions():
    entry = _not_low_risk(
        parser="gpu-license-prose",
        dimensions=["gpu_hour_usd", "license_per_gpu_year_usd"],
    )
    source = parse_source_registry(_registry(entry)).get("example-api-pricing")
    assert source.dimensions == ("gpu_hour_usd", "license_per_gpu_year_usd")


def test_one_provider_may_register_one_url_for_several_channels():
    bedrock = _not_low_risk(
        id="aws-bedrock-pricing",
        provider="aws",
        channel="aws-bedrock",
        parser="html-tables",
        url="https://example.com/bedrock",
    )
    geo = dict(bedrock, id="aws-bedrock-geo-pricing", channel="aws-bedrock-geo")
    registry = parse_source_registry(_registry(bedrock, geo))
    assert registry.allowed_urls() == frozenset({"https://example.com/bedrock"})


def test_two_providers_may_not_share_a_url():
    other = _not_low_risk(id="other-pricing", provider="anthropic", channel="anthropic-api")
    with pytest.raises(SourceRegistryError, match="url https://example.com/pricing is registered by providers anthropic and openai"):
        parse_source_registry(_registry(_entry(), other))


@pytest.mark.parametrize("field", ["evidence", "store_raw", "excerpt_chars", "source_shape", "clarity"])
def test_a_source_entry_rejects_unknown_fields(field):
    with pytest.raises(SourceRegistryError, match=f"example-api-pricing: unknown field '{field}'"):
        parse_source_registry(_registry(_entry(**{field: {}})))


def _catalog():
    return load_catalog_document(
        {
            "version": "test",
            "currency": "USD",
            "pricing_verified_at": "2026-10-01",
            "models": [
                {
                    "canonical_id": "example/model-a",
                    "aliases": [{"provider": "openai", "alias": "model-a", "channel": "openai-api"}],
                    "prices": [
                        {
                            "channel": "openai-api",
                            "effective_from": "2026-01-01",
                            "input_per_mtok": 1,
                            "output_per_mtok": 2,
                            "source_url": "https://example.com/pricing",
                        }
                    ],
                }
            ],
        }
    )


def test_validate_accepts_catalog_and_registry_only_channels():
    azure = _not_low_risk(
        id="azure-foundry-pricing",
        provider="azure",
        channel="azure-foundry",
        catalog_channel=False,
        parser="html-tables",
        url="https://example.com/azure",
    )
    validate_source_registry(parse_source_registry(_registry(_entry(), azure)), _catalog())


def test_validate_rejects_a_catalog_channel_the_catalog_lacks():
    entry = _not_low_risk(channel="fireworks-api", parser="html-tables")
    with pytest.raises(SourceRegistryError, match="example-api-pricing: catalog channel 'fireworks-api' has no catalog rows"):
        validate_source_registry(parse_source_registry(_registry(entry)), _catalog())


def test_validate_rejects_a_registry_only_channel_the_catalog_prices():
    entry = _not_low_risk(catalog_channel=False)
    with pytest.raises(SourceRegistryError, match="example-api-pricing: registry-only channel 'openai-api' is a catalog channel"):
        validate_source_registry(parse_source_registry(_registry(entry)), _catalog())


# --- The bundled registry ----------------------------------------------------

FIXTURES = Path(__file__).parent / "fixtures" / "sources"
LOW_RISK_IDS = {
    "anthropic-api-pricing",
    "google-api-pricing",
    "xai-api-pricing",
}
# Demoted to reviewed discovery while keeping the official parser, so its
# rows remain discovery evidence; its synthetic fixture still covers that parser.
DEMOTED_OFFICIAL_PARSER_IDS = {"openai-api-pricing"}


def test_bundled_registry_validates_against_the_bundled_catalog():
    validate_source_registry(load_source_registry(), load_catalog())


def test_bundled_registry_covers_every_seeded_provider():
    assert {source.provider for source in load_source_registry().sources} == {
        "openai",
        "anthropic",
        "google",
        "deepseek",
        "mistral",
        "xai",
        "perplexity",
        "fireworks",
        "zai",
        "aws",
        "azure",
        "nvidia",
        "vercel",
        "openrouter",
        "moonshotai",
    }


def test_kimi_pricing_source_is_registered_for_reviewed_discovery():
    source = load_source_registry().get("moonshot-api-pricing")
    assert source.provider == "moonshotai"
    assert source.channel == "moonshot-api"
    assert source.url == "https://platform.kimi.ai/"
    assert source.low_risk.eligible is False


def test_gateway_model_feeds_are_registered_for_reviewed_discovery():
    registry = load_source_registry()
    expected = {
        "vercel-ai-gateway-models": (
            "vercel",
            "vercel-ai-gateway",
            "https://ai-gateway.vercel.sh/v1/models",
            "vercel-gateway-json",
        ),
        "openrouter-models": (
            "openrouter",
            "openrouter-api",
            "https://openrouter.ai/api/v1/models",
            "openrouter-json",
        ),
    }
    for source_id, identity in expected.items():
        source = registry.get(source_id)
        assert (source.provider, source.channel, source.url, source.parser) == identity
        assert source.catalog_channel is True
        assert source.low_risk.eligible is False


def test_only_the_three_reviewed_official_parsers_start_low_risk():
    low_risk = load_source_registry().low_risk()
    assert {source.id for source in low_risk} == LOW_RISK_IDS
    assert {source.parser for source in low_risk} == {"anthropic", "google", "xai"}


def test_openai_is_reviewed_discovery_with_its_official_parser():
    source = load_source_registry().get("openai-api-pricing")
    assert source.parser == "openai"
    assert source.low_risk.eligible is False
    assert source.low_risk.dimensions == ()
    assert source.fingerprint is None
    assert source.catalog_channel is True


def test_every_low_risk_source_has_a_recorded_fingerprint():
    for source in load_source_registry().low_risk():
        assert source.fingerprint is not None, source.id


def test_sources_whose_live_parse_is_unreviewed_start_without_low_risk_status():
    registry = load_source_registry()
    for source_id in ("deepseek-api-pricing", "mistral-api-pricing", "perplexity-api-pricing"):
        source = registry.get(source_id)
        assert source.low_risk.eligible is False
        assert source.fingerprint is None
        assert source.catalog_channel is True


def test_deepseek_registers_its_trailing_slash_pricing_url():
    # The slashless URL redirects here, and the exact redirect allowlist
    # rejects that redirect.
    source = load_source_registry().get("deepseek-api-pricing")
    assert source.url == "https://api-docs.deepseek.com/quick_start/pricing/"
    url, channel, _ = _load_check_prices().PROVIDER_PAGES["deepseek"]
    assert (url, channel) == (source.url, source.channel)


def test_low_risk_sources_read_the_pages_the_price_checker_reads():
    check_prices = _load_check_prices()
    registry = load_source_registry()
    for source in registry.low_risk():
        url, channel, _ = check_prices.PROVIDER_PAGES[source.parser]
        assert (source.url, source.channel) == (url, channel)


def test_aws_registers_both_bedrock_channels_on_one_page_and_not_low_risk():
    bedrock = [s for s in load_source_registry().sources if s.provider == "aws"]
    assert sorted(s.channel for s in bedrock) == ["aws-bedrock", "aws-bedrock-geo"]
    assert len({s.url for s in bedrock}) == 1
    assert not any(s.low_risk.eligible for s in bedrock)
    assert all(s.catalog_channel for s in bedrock)


def test_azure_and_nvidia_use_registry_only_channels():
    registry = load_source_registry()
    for provider in ("azure", "nvidia"):
        [source] = [s for s in registry.sources if s.provider == provider]
        assert source.catalog_channel is False
        assert source.low_risk.eligible is False


def test_google_registers_vertex_ai_as_reviewed_discovery_only():
    source = load_source_registry().get("google-vertex-ai-pricing")
    assert (source.provider, source.channel, source.catalog_channel, source.url) == (
        "google",
        "google-vertex-ai",
        True,
        # The current destination of Google's Vertex AI generative pricing
        # page; registering it keeps the allowlisted fetch free of redirects.
        "https://cloud.google.com/gemini-enterprise-agent-platform/generative-ai/pricing",
    )
    assert source.low_risk.eligible is False
    assert source.fingerprint is None
    assert source.parser == "html-tables"


def test_azure_registers_the_foundry_models_pricing_page():
    [azure] = [s for s in load_source_registry().sources if s.provider == "azure"]
    assert azure.url == "https://azure.microsoft.com/en-us/pricing/details/ai-foundry-models/aoai/"


def test_fireworks_and_zai_start_without_low_risk_status():
    registry = load_source_registry()
    for provider in ("fireworks", "zai"):
        [source] = [s for s in registry.sources if s.provider == provider]
        assert source.low_risk.eligible is False
        assert source.catalog_channel is True


def test_nvidia_records_gpu_or_license_pricing_and_no_token_dimensions():
    [nvidia] = [s for s in load_source_registry().sources if s.provider == "nvidia"]
    assert nvidia.parser == "gpu-license-prose"
    assert set(nvidia.dimensions) == {"gpu_hour_usd", "license_per_gpu_year_usd"}


# Generic synthetic pages for the discovery parsers, standing for no
# provider's layout: a token-price table beside a non-token table, and GPU
# or license pricing stated in prose.
TOKEN_TABLE_PAGE = (
    "<table><tr><th>Model</th><th>Input (per 1M tokens)</th><th>Cached input (per 1M tokens)</th>"
    "<th>Output (per 1M tokens)</th></tr>"
    "<tr><td>Example Model Large</td><td>$2.00</td><td>$0.20</td><td>$8.00</td></tr>"
    "<tr><td>Example Model Small</td><td>$0.40</td><td>Free</td><td>$1.60</td></tr></table>"
    "<table><tr><th>Accelerator</th><th>Price per hour</th></tr>"
    "<tr><td>Example Accelerator</td><td>$2.90</td></tr></table>"
)
GPU_LICENSE_PAGE = (
    "<p>Licenses start at $1000 per GPU per year, or about $0.50 per GPU per hour in the cloud.</p>"
    "<p>Developers can prototype at no cost.</p>"
)
# The generic synthetic page each discovery parser is exercised on.
GENERIC_PAGES = {"html-tables": TOKEN_TABLE_PAGE, "gpu-license-prose": GPU_LICENSE_PAGE}


def test_only_low_risk_and_demoted_official_sources_have_source_specific_fixtures():
    fixtures = {path.name.split(".")[0] for path in FIXTURES.iterdir()}
    low_risk = {source.id for source in load_source_registry().low_risk()}
    assert fixtures == low_risk | DEMOTED_OFFICIAL_PARSER_IDS


def test_every_other_non_low_risk_source_uses_a_generic_discovery_parser():
    structured = {"vercel-gateway-json", "openrouter-json"}
    for source in load_source_registry().sources:
        if not source.low_risk.eligible and source.id not in DEMOTED_OFFICIAL_PARSER_IDS:
            assert source.parser in GENERIC_PAGES or source.parser in structured, source.id


def test_every_generic_discovery_source_yields_normalized_discovery_evidence():
    from metergraph_core import discovery_evidence

    catalog = load_catalog()
    for source in load_source_registry().sources:
        if (
            source.low_risk.eligible
            or source.id in DEMOTED_OFFICIAL_PARSER_IDS
            or source.parser not in GENERIC_PAGES
        ):
            continue
        page = GENERIC_PAGES[source.parser]
        evidence = discovery_evidence(source, page, catalog.document, observed_on=date(2026, 10, 20))["evidence"]
        assert evidence["redacted"] is True, source.id
        if evidence["kind"] == "tables":
            assert evidence["tables"][0]["rows"][0] == {"label": "example-model-large", "amounts": ["2.00", "0.20", "8.00"]}
        else:
            assert evidence["candidates"] == {"license_per_gpu_year_usd": "1000", "gpu_hour_usd": "0.50"}


_CREDENTIAL = re.compile(r"token|secret|password|passwd|api[_-]?key|authorization|credential|bearer", re.I)


def _keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _keys(item)


def test_registry_carries_no_credential_fields_or_url_parameters():
    from metergraph_core.sources import DEFAULT_REGISTRY_PATH

    raw = yaml.safe_load(DEFAULT_REGISTRY_PATH.read_text())
    assert not [key for key in _keys(raw) if _CREDENTIAL.search(str(key))]
    for source in load_source_registry().sources:
        assert "?" not in source.url and "@" not in source.url, source.url


def _load_check_prices():
    import importlib.util
    import sys

    script = Path(__file__).resolve().parents[2] / "scripts" / "check_prices.py"
    if "check_prices" in sys.modules:
        return sys.modules["check_prices"]
    spec = importlib.util.spec_from_file_location("check_prices", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_prices"] = module
    spec.loader.exec_module(module)
    return module
