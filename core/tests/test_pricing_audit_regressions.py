from decimal import Decimal

import pytest

from metergraph_core import load_catalog, price_call_from_response

AT = "2026-10-29T12:00:00Z"


@pytest.mark.parametrize("input_tokens,cached,output_tokens,expected", [
    (1_000_000, 1_000_000, 0, "0.25000000"),
    (100_000, 80_000, 1_000, "0.04400000"),
])
def test_o3_batch_uses_the_published_cached_input_rate(input_tokens, cached, output_tokens, expected):
    response = {"object": "chat.completion", "model": "o3", "usage": {
        "prompt_tokens": input_tokens, "completion_tokens": output_tokens,
        "prompt_tokens_details": {"cached_tokens": cached},
    }}
    result = price_call_from_response("openai", response, at=AT, service_tier="batch")
    assert result.status == "priced"
    assert result.cost_usd == Decimal(expected)
    assert result.token_cost.price.batch_cache_read_per_mtok == Decimal("0.25")


@pytest.mark.parametrize("field", ["webSearchQueries", "groundingChunks"])
@pytest.mark.parametrize("sdk_object", [False, True])
def test_gemini_sdk_grounding_has_the_same_completeness_as_rest(field, sdk_object):
    snake_field = {"webSearchQueries": "web_search_queries", "groundingChunks": "grounding_chunks"}[field]
    rest = {"modelVersion": "gemini-3.8-flash", "usageMetadata": {
        "promptTokenCount": 1_000, "candidatesTokenCount": 100,
    }, "candidates": [{"groundingMetadata": {field: ["synthetic"]}}]}
    sdk = {"model_version": "gemini-3.8-flash", "usage_metadata": {
        "prompt_token_count": 1_000, "candidates_token_count": 100,
    }, "candidates": [{"grounding_metadata": {snake_field: ["synthetic"]}}]}

    class Response:
        def model_dump(self):
            return sdk

    expected = price_call_from_response("google", rest, at=AT)
    actual = price_call_from_response("google", Response() if sdk_object else sdk, at=AT)
    assert actual.status == expected.status == "partial"
    assert actual.cost_usd is expected.cost_usd is None
    assert actual.known_subtotal_usd == expected.known_subtotal_usd
    assert actual.retrieval == expected.retrieval


@pytest.mark.parametrize("at,canonical", [
    ("2026-11-01T23:59:59Z", "alibaba/qwen3-coder-30b-a3b-instruct"),
    ("2026-11-02T00:00:00Z", "alibaba/qwen3-coder-30b-a3b"),
    ("2026-11-03T00:00:00Z", "alibaba/qwen3-coder-30b-a3b"),
])
def test_qwen_alias_migration_preserves_identity_and_price(at, canonical):
    catalog = load_catalog()
    native = "qwen/qwen3-coder-30b-a3b-instruct"
    former = "alibaba/qwen3-coder-30b-a3b-instruct"
    assert catalog.canonical_model_id("openrouter", native, at=at) == canonical
    for name in (native, former):
        result = catalog.lookup_price(model=name, channel="openrouter-api", at=at)
        assert result.status == "priced"
        assert result.model_id == canonical
        assert result.price.input_per_mtok == Decimal("0.07")
        assert result.price.output_per_mtok == Decimal("0.28")


def test_qwen_canonicalization_without_a_date_returns_a_registered_identity():
    catalog = load_catalog()
    result = catalog.canonical_model_id("openrouter", "qwen/qwen3-coder-30b-a3b-instruct")
    assert result in {model["canonical_id"] for model in catalog.document["models"]}
