"""Why an unpriced lookup or cost is unpriced, when the cause is a value the
caller sent.

An unknown channel, a provider with no channel of its own, an unknown model,
a model the channel does not sell, and a name or price outside its effective
window all used to look the same: ``unpriced`` with nothing else said. Each
now carries one warning with a stable code and a message that names the
value and, when one is close, the value the caller probably meant. The result
stays ``unpriced``; nothing raises.
"""

import textwrap
from datetime import datetime, timezone

import pytest

import metergraph_core
from metergraph_core import ModelUsage, cost_from_response, load_catalog


_CATALOG = textwrap.dedent(
    """
    version: "test-warnings-1"
    currency: USD
    pricing_verified_at: "2026-08-01"
    models:
      - canonical_id: openai/gpt-x
        aliases:
          - {provider: openai, alias: gpt-x, channel: openai-api}
          - {provider: anthropic, alias: gpt-x, channel: anthropic-api}
        prices:
          - channel: openai-api
            region: global
            effective_from: "2026-01-01"
            input_per_mtok: 1.0
            output_per_mtok: 2.0
            source_url: https://example.test/openai
      - canonical_id: anthropic/claude-x
        aliases:
          - {provider: anthropic, alias: claude-x, channel: anthropic-api}
        prices:
          - channel: anthropic-api
            region: global
            effective_from: "2026-01-01"
            input_per_mtok: 3.0
            output_per_mtok: 15.0
            source_url: https://example.test/anthropic
      - canonical_id: google/gem-x
        aliases:
          - {provider: google, alias: gem-x, channel: google-api}
          - provider: google
            alias: gem-latest
            channel: google-api
            effective_from: "2026-01-01"
            effective_to: "2026-04-01"
            source_url: https://example.test/google
          - provider: google
            alias: gem-next
            channel: google-api
            effective_from: "2026-09-01"
            source_url: https://example.test/google
        prices:
          - channel: google-api
            region: global
            effective_from: "2026-01-01"
            effective_to: "2026-04-01"
            input_per_mtok: 0.5
            output_per_mtok: 1.5
            source_url: https://example.test/google
          - channel: google-api
            region: global
            effective_from: "2026-09-01"
            input_per_mtok: 0.4
            output_per_mtok: 1.2
            source_url: https://example.test/google
    """
)

_AT = datetime(2026, 3, 1, tzinfo=timezone.utc)
_GAP = datetime(2026, 6, 1, tzinfo=timezone.utc)


@pytest.fixture
def catalog(tmp_path):
    path = tmp_path / "prices.yaml"
    path.write_text(_CATALOG)
    return load_catalog(path)


def _warning(result):
    assert result.status == "unpriced"
    [warning] = result.warnings
    return warning


@pytest.mark.parametrize(("channel", "hint"), [
    ("openai", "openai-api"),
    ("gemini-api", "google-api"),
    ("openai-apii", "openai-api"),
    ("nowhere", None),
])
def test_an_unknown_channel_names_the_nearest_one(catalog, channel, hint):
    warning = _warning(catalog.lookup_price(model="gpt-x", channel=channel, at=_AT))
    assert warning.code == metergraph_core.catalog.UNKNOWN_CHANNEL
    assert warning.message.startswith(f'channel "{channel}" is not a channel the catalog prices.')
    assert (f'Did you mean "{hint}"?' in warning.message) if hint else ("Did you mean" not in warning.message)


def test_an_alias_alone_does_not_make_a_channel_priced(catalog):
    # gpt-x has an alias on anthropic-api but no price there.
    warning = _warning(catalog.lookup_price(model="gpt-x", channel="anthropic-api", at=_AT))
    assert warning.code == "model_not_on_channel"
    assert warning.message == 'model "gpt-x" has no price on channel "anthropic-api". It is priced on openai-api.'


def test_a_provider_without_a_channel_of_its_own_is_explained(catalog):
    typo = _warning(catalog.lookup_price(model="gpt-x", provider="opneai", at=_AT))
    assert typo.code == "unknown_provider"
    assert 'Did you mean "openai"?' in typo.message
    proxy = _warning(catalog.lookup_price(model="gpt-x", provider="litellm", at=_AT))
    assert proxy.code == "unknown_provider"
    assert "name the provider that served it or the channel that billed it" in proxy.message


def test_no_provider_and_no_channel_says_only_that_the_channel_is_missing(catalog):
    found = catalog.lookup_price(model="gpt-x", at=_AT)
    assert found.missing_dimensions == ("channel",)
    assert found.warnings == ()


def test_an_unknown_model_names_a_close_one_only(catalog):
    close = _warning(catalog.lookup_price(model="gptx", provider="openai", at=_AT))
    assert (close.code, close.message) == ("unknown_model", 'model "gptx" is not in the catalog. Did you mean "gpt-x"?')
    far = _warning(catalog.lookup_price(model="nothing-like-it", provider="openai", at=_AT))
    assert far.message == 'model "nothing-like-it" is not in the catalog.'


def test_a_model_the_channel_does_not_sell_lists_the_channels_that_do(catalog):
    warning = _warning(catalog.lookup_price(model="claude-x", provider="openai", at=_AT))
    assert warning.code == "model_not_on_channel"
    assert warning.message.endswith("It is priced on anthropic-api.")


def test_a_retired_or_future_name_says_when(catalog):
    retired = _warning(catalog.lookup_price(model="gem-latest", provider="google", at=_GAP))
    assert retired.code == "model_not_in_effect"
    assert "stopped resolving" in retired.message and "2026-04-01" in retired.message
    future = _warning(catalog.lookup_price(model="gem-next", provider="google", at=_AT))
    assert future.code == "model_not_in_effect"
    assert "from 2026-09-01" in future.message


def test_a_gap_between_prices_is_explained(catalog):
    warning = _warning(catalog.lookup_price(model="gem-x", provider="google", at=_GAP))
    assert warning.code == "price_not_in_effect"
    assert 'channel "google-api"' in warning.message


def test_priced_results_and_matching_case_carry_no_warnings(catalog):
    for found in (
        catalog.lookup_price(model="gpt-x", provider="openai", at=_AT),
        catalog.lookup_price(model="GPT-X", channel="OpenAI-API", at=_AT),
    ):
        assert found.status == "priced"
        assert found.warnings == ()


def test_calculate_cost_carries_the_same_warnings(catalog):
    result = catalog.calculate_cost(
        model="gpt-x", channel="gemini-api", at=_AT,
        usage=ModelUsage(input_tokens=1000, output_tokens=500),
    )
    assert result.cost is None
    assert result.reasons
    assert _warning(result).code == "unknown_channel"


def test_a_response_is_judged_against_the_provider_its_format_implies(catalog):
    # An Anthropic Messages body is priced on anthropic-api, so the warning
    # judges the model against that channel.
    response = {"type": "message", "model": "claude-x", "usage": {"input_tokens": 10, "output_tokens": 5}}
    assert cost_from_response("anthropic", response, catalog=catalog, at=_AT).status == "priced"
    response["model"] = "gpt-x-mini"
    warning = _warning(cost_from_response("anthropic", response, catalog=catalog, at=_AT))
    assert warning.code == "unknown_model"
    response["model"] = "gem-x"
    warning = _warning(cost_from_response("anthropic", response, catalog=catalog, at=_AT))
    assert warning.message == 'model "gem-x" has no price on channel "anthropic-api". It is priced on google-api.'


def test_pricing_warning_is_exported():
    assert "PricingWarning" in metergraph_core.__all__
