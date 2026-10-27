"""Deterministic extraction from supplied source content: rates, structural
fingerprints, and discovery-only evidence. Nothing here fetches."""

import json
from decimal import Decimal
from pathlib import Path

import pytest

from metergraph_core.extractors import ExtractedRate, extract, structural_fingerprint

FIXTURES = Path(__file__).parent / "fixtures" / "sources"
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
D = Decimal


def _fixture(source_id):
    return (FIXTURES / f"{source_id}.html").read_text()


def _generic(name):
    return {"token-table": TOKEN_TABLE_PAGE, "gpu-license-prose": GPU_LICENSE_PAGE}[name]


def _rates(extraction):
    return {(r.model_key, r.region): dict(r.rates) for r in extraction.rates}


EXPECTED = {
    ("openai", "openai-api-pricing"): {
        ("gpt-legacy", "global"): {"input_per_mtok": D("2.5"), "cache_read_per_mtok": D("1.25"), "output_per_mtok": D("10")},
        ("gpt-x", "global"): {"input_per_mtok": D("2"), "cache_read_per_mtok": D("0.2"), "cache_write_5m_per_mtok": D("2.5"), "output_per_mtok": D("8")},
        ("gpt-x-mini", "global"): {"input_per_mtok": D("0.4"), "cache_read_per_mtok": D("0.04"), "output_per_mtok": D("1.6")},
    },
    ("anthropic", "anthropic-api-pricing"): {
        ("claude-example-5", "global"): {"input_per_mtok": D("5"), "output_per_mtok": D("25"), "cache_write_5m_per_mtok": D("6.25"), "cache_read_per_mtok": D("0.50")},
        ("claude-example-5-5", "global"): {"input_per_mtok": D("4"), "output_per_mtok": D("20"), "cache_write_5m_per_mtok": D("5"), "cache_read_per_mtok": D("0.40")},
    },
    ("google", "google-api-pricing"): {
        ("gemini-x", "global"): {"input_per_mtok": D("0.25"), "output_per_mtok": D("1.50"), "cache_read_per_mtok": D("0.025")},
        ("gemini-x-image", "global"): {"input_per_mtok": D("0.30")},
        ("gemini-x-lite", "global"): {"input_per_mtok": D("0.10"), "output_per_mtok": D("0.40"), "cache_read_per_mtok": D("0.01")},
        ("gemini-x-pro-preview", "global"): {"input_per_mtok": D("2.00"), "output_per_mtok": D("12.00"), "cache_read_per_mtok": D("0.20")},
        ("gemini-x-pro-preview-customtools", "global"): {"input_per_mtok": D("2.00"), "output_per_mtok": D("12.00"), "cache_read_per_mtok": D("0.20")},
    },
    ("xai", "xai-api-pricing"): {
        ("grok-x", "global"): {"input_per_mtok": D("2"), "cache_read_per_mtok": D("0.5"), "output_per_mtok": D("6")},
        ("grok-x", "us"): {"input_per_mtok": D("2.2"), "cache_read_per_mtok": D("0.55"), "output_per_mtok": D("6.6")},
    },
}


@pytest.mark.parametrize(("parser", "source_id"), list(EXPECTED))
def test_each_official_fixture_extracts_its_catalog_rates(parser, source_id):
    extraction = extract(parser, _fixture(source_id))
    assert extraction.parser == parser
    assert extraction.applicable is True
    assert extraction.ambiguities == ()
    assert _rates(extraction) == EXPECTED[(parser, source_id)]


@pytest.mark.parametrize(("parser", "source_id"), list(EXPECTED))
def test_extraction_is_deterministic_and_sorted(parser, source_id):
    first = extract(parser, _fixture(source_id))
    second = extract(parser, _fixture(source_id))
    assert first == second
    keys = [(r.model_key, r.region) for r in first.rates]
    assert keys == sorted(keys)


def test_extracted_rates_are_immutable():
    [rate, *_] = extract("openai", _fixture("openai-api-pricing")).rates
    assert isinstance(rate, ExtractedRate)
    with pytest.raises(TypeError):
        rate.rates["input_per_mtok"] = D("0")


def test_html_price_prose_captures_model_card_prices_for_review():
    page = (
        "<section><h2>K3.1 Pro</h2><p>Flagship model</p>"
        "<div>Input $2.00 / MTok Output $10.00 / MTok "
        "Cache Write $2.00 / MTok Cache Hit $0.20 / MTok</div></section>"
    )
    extraction = extract("html-price-prose", page)
    assert extraction.applicable is False
    assert extraction.rates == ()
    assert extraction.evidence["kind"] == "price_mentions"
    assert any("K3.1 Pro" in item and "$2.00" in item for item in extraction.evidence["mentions"])
    assert any("$0.20" in item for item in extraction.evidence["mentions"])
    assert structural_fingerprint("html-price-prose", page) == structural_fingerprint(
        "html-price-prose", page.replace("$2.00", "$3.00")
    )


def test_vercel_gateway_json_extracts_paid_language_rates_and_rules():
    content = json.dumps(
        {
            "object": "list",
            "data": [
                {
                    "id": "openai/gpt-example",
                    "type": "language",
                    "owned_by": "openai",
                    "pricing": {
                        "input": "0.000002",
                        "output": "0.000008",
                        "input_cache_read": "0.0000002",
                        "input_cache_write": "0.0000025",
                        "varies_by_provider": True,
                    },
                },
                {
                    "id": "openai/image-example",
                    "type": "image",
                    "owned_by": "openai",
                    "pricing": {"input": "0.000001", "output": "0.000004"},
                },
                {
                    "id": "reflection/free-example",
                    "type": "language",
                    "owned_by": "reflection",
                    "pricing": {"input": "0", "output": "0"},
                },
                {
                    "id": "scaledown/input-only",
                    "type": "language",
                    "owned_by": "scaledown",
                    "pricing": {"input": "0.00000005", "output": "0"},
                },
                {
                    "id": "openai/tiered-example",
                    "type": "language",
                    "owned_by": "openai",
                    "pricing": {
                        "input": "0.000001",
                        "output": "0.000004",
                        "input_tiers": [{"min": 0, "cost": "0.000001"}],
                    },
                },
                {
                    "id": "openai/search-example",
                    "type": "language",
                    "owned_by": "openai",
                    "pricing": {
                        "input": "0.000001",
                        "output": "0.000004",
                        "web_search": "0.01",
                    },
                },
                {
                    "id": "openai/unknown-pricing-example",
                    "type": "language",
                    "owned_by": "openai",
                    "pricing": {
                        "input": "0.000001",
                        "output": "0.000004",
                        "future_tier": [{"min": 1}],
                    },
                },
            ],
        }
    )

    result = extract("vercel-gateway-json", content)

    assert _rates(result) == {
        ("openai/gpt-example", "global"): {
            "input_per_mtok": D("2"),
            "output_per_mtok": D("8"),
            "cache_read_per_mtok": D("0.2"),
            "cache_write_5m_per_mtok": D("2.5"),
        },
        ("scaledown/input-only", "global"): {
            "input_per_mtok": D("0.05"),
            "output_per_mtok": D("0"),
        },
    }
    [rate] = [row for row in result.rates if row.model_key == "openai/gpt-example"]
    assert dict(rate.meta) == {
        "publisher": "openai",
        "alias_rules": {
            "input_includes_cache_read": True,
            "input_includes_cache_write": True,
        },
        "price_rules": {"varies_by_provider": True},
    }
    assert [(item.model_key, item.reason) for item in result.exclusions] == [
        ("openai/image-example", "not_language"),
        ("openai/search-example", "unsupported_charge"),
        ("openai/tiered-example", "conditional_price"),
        ("openai/unknown-pricing-example", "unknown_pricing_field"),
        ("reflection/free-example", "non_positive_token_price"),
    ]


def test_openrouter_json_keeps_only_static_complete_token_prices():
    content = json.dumps(
        {
            "data": [
                {
                    "id": "example/model-a",
                    "architecture": {"output_modalities": ["text"]},
                    "pricing": {
                        "prompt": "0.00000125",
                        "completion": "0.0000045",
                        "input_cache_read": "0.000000125",
                        "input_cache_write_1h": "0.00000025",
                        "request": "0.002",
                    },
                },
                {
                    "id": "example/model-free:free",
                    "architecture": {"output_modalities": ["text"]},
                    "pricing": {"prompt": "0", "completion": "0"},
                },
                {
                    "id": "~example/model-latest",
                    "architecture": {"output_modalities": ["text"]},
                    "pricing": {"prompt": "0.000001", "completion": "0.000004"},
                },
                {
                    "id": "scaledown/input-only",
                    "architecture": {"output_modalities": ["text"]},
                    "pricing": {"prompt": "0.00000004", "completion": "0"},
                },
                {
                    "id": "example/model-dynamic",
                    "architecture": {"output_modalities": ["text"]},
                    "pricing": {
                        "prompt": "0.000001",
                        "completion": "0.000004",
                        "overrides": [{"utc_days": ["monday"]}],
                    },
                },
                {
                    "id": "example/model-image",
                    "architecture": {"output_modalities": ["image"]},
                    "pricing": {"prompt": "0.000001", "completion": "0.000004"},
                },
                {
                    "id": "example/model-extra-charge",
                    "architecture": {"output_modalities": ["text"]},
                    "pricing": {
                        "prompt": "0.000001",
                        "completion": "0.000004",
                        "image": "0.001",
                    },
                },
                {
                    "id": "example/model-audio-output",
                    "architecture": {"output_modalities": ["text"]},
                    "pricing": {
                        "prompt": "0.000001",
                        "completion": "0.000004",
                        "audio_output": "0.001",
                    },
                },
                {
                    "id": "example/model-unknown-pricing",
                    "architecture": {"output_modalities": ["text"]},
                    "pricing": {
                        "prompt": "0.000001",
                        "completion": "0.000004",
                        "future_surcharge": "0.001",
                    },
                },
            ]
        }
    )

    result = extract("openrouter-json", content)

    assert _rates(result) == {
        ("example/model-a", "global"): {
            "input_per_mtok": D("1.25"),
            "output_per_mtok": D("4.5"),
            "cache_read_per_mtok": D("0.125"),
            "cache_write_1h_per_mtok": D("0.25"),
            "request_fee_per_1k_usd": D("2"),
        },
        ("scaledown/input-only", "global"): {
            "input_per_mtok": D("0.04"),
            "output_per_mtok": D("0"),
        },
    }
    [rate] = [row for row in result.rates if row.model_key == "example/model-a"]
    assert dict(rate.meta) == {
        "publisher": "example",
        "alias_rules": {
            "input_includes_cache_read": True,
            "input_includes_cache_write": True,
        },
        "price_rules": {"varies_by_provider": True},
    }
    assert [(item.model_key, item.reason) for item in result.exclusions] == [
        ("example/model-audio-output", "unsupported_charge"),
        ("example/model-dynamic", "conditional_price"),
        ("example/model-extra-charge", "unsupported_charge"),
        ("example/model-free:free", "free_route"),
        ("example/model-image", "non_text_output"),
        ("example/model-unknown-pricing", "unknown_pricing_field"),
        ("~example/model-latest", "floating_route"),
    ]


@pytest.mark.parametrize("parser", ["vercel-gateway-json", "openrouter-json"])
def test_gateway_json_fingerprint_tracks_shape_not_values_or_models(parser):
    base = json.dumps(
        {
            "data": [
                {
                    "id": "example/model-a",
                    "type": "language",
                    "owned_by": "example",
                    "architecture": {"output_modalities": ["text"]},
                    "pricing": {"input": "0.1", "output": "0.2", "prompt": "0.1", "completion": "0.2"},
                }
            ]
        }
    )
    changed_values = base.replace("model-a", "model-b").replace("0.1", "0.3")
    changed_shape = base.replace('"pricing":', '"new_field": true, "pricing":')

    assert structural_fingerprint(parser, changed_values) == structural_fingerprint(parser, base)
    assert structural_fingerprint(parser, changed_shape) != structural_fingerprint(parser, base)


@pytest.mark.parametrize("parser", ["vercel-gateway-json", "openrouter-json"])
def test_invalid_gateway_json_is_reported_without_rates(parser):
    result = extract(parser, "not json")

    assert result.applicable is False
    assert result.rates == ()
    assert result.evidence == {"kind": "gateway_models", "valid_json": False}


# Each low-risk parser's fixture with a price changed, with a model added, and
# with a parser-relevant structural change. The other official parsers keep
# their focused inline cases in test_check_prices.py.
VARIANTS = {
    "openai": (
        "openai-api-pricing",
        lambda c: c.replace("[0,2],[0,0.2],[0,2.5],[0,8]", "[0,3],[0,0.3],[0,3.5],[0,9]"),
        lambda c: c.replace(
            "[1,[[0,&quot;gpt-x-mini&quot;]",
            "[1,[[0,&quot;gpt-z&quot;],[0,1],[0,0.1],[0,4]]][1,[[0,&quot;gpt-x-mini&quot;]",
        ),
        lambda c: c.replace("&quot;batch&quot;", "&quot;flex&quot;"),
    ),
    "anthropic": (
        "anthropic-api-pricing",
        lambda c: c.replace("$4 / MTok", "$4.50 / MTok"),
        lambda c: c.replace(
            "</table>",
            "<tr><td>Claude Example 6New</td><td>$6 / MTok</td><td>$30 / MTok</td><td>$7.5 / MTok</td><td>$12 / MTok</td><td>$0.60 / MTok</td></tr></table>",
        ),
        lambda c: c.replace("5m writes", "5-minute writes"),
    ),
    "google": (
        "google-api-pricing",
        lambda c: c.replace("$1.50", "$1.75"),
        lambda c: c.replace(
            "</body>",
            "<h2>Gemini Y</h2><p>gemini-y</p><a>Try it in Google AI Studio</a><table><tr><td>Input price</td><td>$1</td></tr><tr><td>Output price</td><td>$2</td></tr></table></body>",
        ),
        lambda c: c.replace("Context caching price", "Caching price"),
    ),
    "xai": (
        "xai-api-pricing",
        lambda c: c.replace("&quot;20000&quot;", "&quot;25000&quot;"),
        lambda c: c.replace(
            "</script>",
            "{&quot;name&quot;:&quot;grok-y&quot;,&quot;version&quot;:&quot;1.0&quot;,&quot;promptTextTokenPrice&quot;:&quot;1000&quot;,&quot;cachedPromptTokenPrice&quot;:&quot;100&quot;,&quot;completionTextTokenPrice&quot;:&quot;3000&quot;,&quot;cluster&quot;:&quot;us-east-1&quot;}</script>",
        ),
        lambda c: c.replace("cachedPromptTokenPrice", "cachedTokenPrice"),
    ),
}


@pytest.mark.parametrize("parser", list(VARIANTS))
def test_fingerprint_ignores_a_price_change(parser):
    source_id, change_price, _, _ = VARIANTS[parser]
    base = _fixture(source_id)
    changed = change_price(base)
    assert changed != base
    assert _rates(extract(parser, changed)) != _rates(extract(parser, base))
    assert structural_fingerprint(parser, changed) == structural_fingerprint(parser, base)


@pytest.mark.parametrize("parser", [p for p, v in VARIANTS.items() if v[2] is not None])
def test_fingerprint_ignores_a_newly_listed_model(parser):
    source_id, _, add_model, _ = VARIANTS[parser]
    base = _fixture(source_id)
    added = add_model(base)
    assert len(extract(parser, added).rates) > len(extract(parser, base).rates)
    assert structural_fingerprint(parser, added) == structural_fingerprint(parser, base)


@pytest.mark.parametrize("parser", list(VARIANTS))
def test_fingerprint_changes_with_parser_relevant_structure(parser):
    source_id, _, _, restructure = VARIANTS[parser]
    base = _fixture(source_id)
    assert structural_fingerprint(parser, restructure(base)) != structural_fingerprint(parser, base)


def _google_block(heading):
    return (
        f"<h2>Gemini Z</h2>{heading}<a>Try it in Google AI Studio</a>"
        "<table><tr><td>Input price</td><td>$1</td></tr><tr><td>Output price</td><td>$2</td></tr></table>"
    )


def test_google_reads_every_id_joined_by_and():
    page = _google_block("<p>gemini-z</p><p>and</p><p>gemini-z-001</p><p>and</p><p>gemini-z-latest</p>")
    rates = {"input_per_mtok": D("1"), "output_per_mtok": D("2")}
    assert _rates(extract("google", page)) == {
        ("gemini-z", "global"): rates,
        ("gemini-z-001", "global"): rates,
        ("gemini-z-latest", "global"): rates,
    }


def test_google_does_not_read_an_adjacent_lowercase_cell_without_and_as_an_id():
    page = _google_block("<p>preview</p><p>gemini-z</p>")
    assert _rates(extract("google", page)) == {
        ("gemini-z", "global"): {"input_per_mtok": D("1"), "output_per_mtok": D("2")},
    }


def _swap(text, first, second):
    return text.replace(first, "\0").replace(second, first).replace("\0", second)


def test_openai_fingerprint_changes_when_same_width_columns_are_reordered():
    # The embedded cells fill columns by position, so swapping two column
    # labels changes what each cell means without changing any row width.
    base = _fixture("openai-api-pricing")
    reordered = _swap(base, "<th>Cached input</th>", "<th>Cache writes</th>")
    assert reordered != base
    assert structural_fingerprint("openai", reordered) != structural_fingerprint("openai", base)


def test_openai_fingerprint_changes_when_the_column_header_disappears():
    base = _fixture("openai-api-pricing")
    without = base.replace("<th>Short context</th>", "<th>Context</th>")
    assert structural_fingerprint("openai", without) != structural_fingerprint("openai", base)


def test_openai_fingerprint_ignores_labels_outside_the_text_table_columns():
    base = _fixture("openai-api-pricing")
    other_table = base.replace(
        "<script>", "<table><tr><th>Model</th><th>Price per minute</th></tr></table><script>"
    )
    assert structural_fingerprint("openai", other_table) == structural_fingerprint("openai", base)


def test_extraction_carries_the_structural_fingerprint():
    content = _fixture("anthropic-api-pricing")
    fingerprint = structural_fingerprint("anthropic", content)
    assert extract("anthropic", content).fingerprint == fingerprint
    assert fingerprint.startswith("sha256:") and len(fingerprint) == 71


def test_fingerprints_differ_between_parsers_for_the_same_content():
    content = _generic("token-table")
    assert structural_fingerprint("html-tables", content) != structural_fingerprint("gpu-license-prose", content)


def test_two_candidate_values_for_one_model_are_ambiguous():
    content = _fixture("openai-api-pricing").replace(
        "[1,[[0,&quot;gpt-x-mini&quot;]",
        "[1,[[0,&quot;gpt-x&quot;],[0,2.2],[0,0.2],[0,2.5],[0,8.8]]][1,[[0,&quot;gpt-x-mini&quot;]",
    )
    extraction = extract("openai", content)
    assert extraction.ambiguities == ("gpt-x",)
    # The first candidate is still what the price checker reads.
    assert _rates(extraction)[("gpt-x", "global")]["input_per_mtok"] == D("2")


def test_a_repeated_model_with_equal_values_is_not_ambiguous():
    content = _fixture("anthropic-api-pricing").replace(
        "</table>",
        "<tr><td>Claude Example 5For agentic work</td><td>$5 / MTok</td><td>$25 / MTok</td><td>$6.25 / MTok</td><td>$10 / MTok</td><td>$0.50 / MTok</td></tr></table>",
    )
    assert extract("anthropic", content).ambiguities == ()


# Discovery-only parsers are generic: they read any page's tables or GPU and
# license prose, and are tested on generic synthetic pages, never on a mock of
# one provider's site.


def test_the_table_parser_emits_evidence_and_no_rates():
    extraction = extract("html-tables", _generic("token-table"))
    assert extraction.applicable is False
    assert extraction.rates == ()
    assert extraction.evidence == {
        "kind": "tables",
        "tables": [
            {
                "headers": ["Model", "Input (per 1M tokens)", "Cached input (per 1M tokens)", "Output (per 1M tokens)"],
                "rows": [
                    ["Example Model Large", "$2.00", "$0.20", "$8.00"],
                    ["Example Model Small", "$0.40", "Free", "$1.60"],
                ],
            },
            {"headers": ["Accelerator", "Price per hour"], "rows": [["Example Accelerator", "$2.90"]]},
        ],
    }


def test_the_table_parser_reads_a_page_without_tables_as_empty_evidence():
    extraction = extract("html-tables", "<p>No tables here.</p>")
    assert extraction.evidence == {"kind": "tables", "tables": []}


def test_the_table_fingerprint_ignores_prices_but_not_headers():
    base = _generic("token-table")
    assert structural_fingerprint("html-tables", base.replace("$2.00", "$2.50")) == structural_fingerprint("html-tables", base)
    assert structural_fingerprint("html-tables", base.replace("Cached input (per", "Cache hit (per")) != structural_fingerprint("html-tables", base)


def test_the_license_parser_records_license_pricing_and_never_token_rates():
    extraction = extract("gpu-license-prose", _generic("gpu-license-prose"))
    assert extraction.applicable is False
    assert extraction.rates == ()
    assert extraction.evidence["kind"] == "gpu_or_license_statements"
    assert extraction.evidence["candidates"] == {
        "license_per_gpu_year_usd": "1000",
        "gpu_hour_usd": "0.50",
    }
    assert extraction.evidence["statements"] == [
        "Licenses start at $1000 per GPU per year, or about $0.50 per GPU per hour in the cloud."
    ]


def test_the_license_parser_ignores_token_wording():
    content = _generic("gpu-license-prose") + "<p>Hosted trials cost $0.10 per 1M tokens.</p>"
    extraction = extract("gpu-license-prose", content)
    assert extraction.rates == ()
    assert set(extraction.evidence["candidates"]) == {"license_per_gpu_year_usd", "gpu_hour_usd"}


def _xai_object(name, **fields):
    body = "".join(f",&quot;{key}&quot;:&quot;{value}&quot;" for key, value in fields.items())
    return "{&quot;name&quot;:&quot;" + name + "&quot;,&quot;version&quot;:&quot;1.0&quot;" + body + "}"


def test_xai_reads_each_model_only_from_its_own_object():
    # grok-a omits its cluster and cached price; grok-b, right after, has both.
    page = (
        "<script>"
        + _xai_object("grok-a", promptTextTokenPrice="10000", completionTextTokenPrice="30000")
        + _xai_object(
            "grok-b",
            promptTextTokenPrice="20000",
            cachedPromptTokenPrice="5000",
            completionTextTokenPrice="60000",
            cluster="us-central-1",
        )
        + "</script>"
    )
    assert _rates(extract("xai", page)) == {
        ("grok-a", "global"): {"input_per_mtok": D("1"), "output_per_mtok": D("3")},
        ("grok-b", "us"): {"input_per_mtok": D("2"), "cache_read_per_mtok": D("0.5"), "output_per_mtok": D("6")},
    }


def test_xai_skips_a_model_whose_own_object_lacks_a_required_price():
    # grok-a has no output price of its own; grok-b's must not be borrowed.
    page = (
        "<script>"
        + _xai_object("grok-a", promptTextTokenPrice="10000", cluster="us-east-1")
        + _xai_object("grok-b", promptTextTokenPrice="20000", completionTextTokenPrice="60000", cluster="us-east-1")
        + "</script>"
    )
    assert _rates(extract("xai", page)) == {
        ("grok-b", "global"): {"input_per_mtok": D("2"), "output_per_mtok": D("6")},
    }


def test_xai_reads_prices_nested_inside_a_model_object():
    page = (
        "<script>{&quot;name&quot;:&quot;grok-c&quot;,&quot;version&quot;:&quot;1.0&quot;,"
        "&quot;limits&quot;:{&quot;rpm&quot;:&quot;60&quot;},"
        "&quot;promptTextTokenPrice&quot;:&quot;10000&quot;,&quot;completionTextTokenPrice&quot;:&quot;30000&quot;}</script>"
    )
    assert _rates(extract("xai", page)) == {
        ("grok-c", "global"): {"input_per_mtok": D("1"), "output_per_mtok": D("3")},
    }


def test_xai_finds_the_enclosing_object_when_a_nested_one_precedes_the_name():
    page = (
        "<script>{&quot;limits&quot;:{&quot;rpm&quot;:&quot;60&quot;},&quot;name&quot;:&quot;grok-d&quot;,&quot;version&quot;:&quot;1.0&quot;,"
        "&quot;promptTextTokenPrice&quot;:&quot;10000&quot;,&quot;completionTextTokenPrice&quot;:&quot;30000&quot;}</script>"
    )
    assert _rates(extract("xai", page)) == {
        ("grok-d", "global"): {"input_per_mtok": D("1"), "output_per_mtok": D("3")},
    }


def test_unknown_parser_is_rejected():
    with pytest.raises(ValueError, match="unknown parser 'scraper'"):
        extract("scraper", "<html></html>")
    with pytest.raises(ValueError, match="unknown parser 'scraper'"):
        structural_fingerprint("scraper", "<html></html>")


@pytest.mark.parametrize("content", [b"<html></html>", None, 3])
def test_extractors_accept_only_text(content):
    with pytest.raises(TypeError, match="content must be text"):
        extract("openai", content)


def test_an_unrecognised_page_extracts_nothing_and_changes_the_fingerprint():
    page = "<html><body><p>Pricing moved.</p></body></html>"
    extraction = extract("openai", page)
    assert extraction.rates == ()
    assert extraction.fingerprint != structural_fingerprint("openai", _fixture("openai-api-pricing"))
