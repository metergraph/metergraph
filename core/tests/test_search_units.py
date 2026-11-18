"""Counting a call's searches and fetches is catalog knowledge: every
consumer that prices a call reads the same fields the same way and bills
them as the same operation on each channel."""

import pytest

from metergraph_core import (
    count_fetch_units,
    count_search_units,
    fetch_operation_for_channel,
    load_catalog,
    search_operation_for_channel,
)
from metergraph_core.retrieval import FETCH_OPERATION_BY_CHANNEL, SEARCH_OPERATION_BY_CHANNEL


@pytest.mark.parametrize(
    "row, expected",
    [
        ({"web_search_calls": 3}, 3),
        ({"grounding_queries": 2, "tool_calls": [{"type": "web_search_call"}]}, 2),
        ({"server_tool_use": {"web_search_requests": 4}}, 4),
        ({"tool_calls": [{"type": "web_search_call"}, {"type": "function", "name": "lookup"}, {"type": "web_search_call"}]}, 2),
        ({"provider": "perplexity", "model": "sonar"}, 1),
        ({"provider": "Perplexity-AI"}, 1),
        # A measured zero: the list is there and holds no search.
        ({"provider": "openai", "tool_calls": [{"type": "function"}]}, 0),
        ({"provider": "openai", "tool_calls": []}, 0),
        ({"server_tool_use": {"web_search_requests": 0}}, 0),
        # No field says: unknown, not zero.
        ({"web_search_calls": "many"}, None),
        ({"web_search_calls": True}, None),
        ({"provider": "openai", "tool_calls": "not a list"}, None),
        ({}, None),
    ],
)
def test_searches_are_read_from_the_first_field_that_carries_them(row, expected):
    assert count_search_units(row) == expected


@pytest.mark.parametrize(
    "row, expected",
    [
        ({"server_tool_use": {"web_search_requests": 1, "web_fetch_requests": 2}}, 2),
        ({"tool_calls": [{"type": "web_fetch_call"}, {"type": "web_search_call"}]}, 1),
        ({"tool_calls": [{"type": "web_search_call"}]}, 0),
        ({"server_tool_use": {"web_search_requests": 1}}, None),
        ({}, None),
    ],
)
def test_fetches_are_read_the_same_way(row, expected):
    assert count_fetch_units(row) == expected


def test_each_fee_charging_channel_names_its_operation():
    assert search_operation_for_channel("openai-api") == "web_search"
    assert search_operation_for_channel("google-vertex-ai") == "google_search_grounding"
    assert search_operation_for_channel("perplexity-api") == "search_request_low"
    assert search_operation_for_channel("deepseek-api") is None
    assert search_operation_for_channel(None) is None
    assert fetch_operation_for_channel("anthropic-api") == "web_fetch"
    assert fetch_operation_for_channel("openai-api") is None


def test_every_named_operation_is_priced_in_the_shipped_catalog():
    catalog = load_catalog()
    for table in (SEARCH_OPERATION_BY_CHANNEL, FETCH_OPERATION_BY_CHANNEL):
        for channel, operation in table.items():
            result = catalog.price_retrieval(channel=channel, operation=operation, units=1, at="2026-10-06")
            assert result.status == "priced", (channel, operation, result)


def test_agent_calls_do_not_inherit_legacy_sonar_one_request_assumption():
    assert count_search_units({"provider": "perplexity", "retrieval_api": "agent"}) is None
    assert count_search_units({"provider": "perplexity", "retrieval_api": "agent", "web_search_calls": 3}) == 3


def test_agent_search_rates_are_separate_from_legacy_sonar_requests():
    from decimal import Decimal
    catalog = load_catalog()
    for operation, expected in [("web_search", "0.0075"), ("web_search_fast", "0.003")]:
        result = catalog.price_retrieval(channel="perplexity-agent-api", operation=operation, units=3, at="2026-11-18")
        assert result.cost_usd == Decimal(expected)
        assert catalog.price_retrieval(channel="perplexity-agent-api", operation=operation, units=3, at="2026-10-01").status == "unpriced"


def test_agent_sonar_tokens_use_its_own_channel():
    from decimal import Decimal
    from metergraph_core import ModelUsage
    result = load_catalog().calculate_cost(model="perplexity/sonar", provider="perplexity-agent", at="2026-11-18", usage=ModelUsage(input_tokens=1000, output_tokens=100))
    assert result.cost_usd == Decimal("0.0005") and result.status == "priced"
    assert result.price.pricing_channel == "perplexity-agent-api"
