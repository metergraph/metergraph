import json
from decimal import Decimal

import pytest

from metergraph_server import prices
from metergraph_server.ingest import COLUMNS, project_row

_, _, SNAPSHOT = prices.load()

SENTINEL = "TOP-SECRET-PROMPT-CONTENT"


def _row(**overrides) -> dict:
    row = {
        "ts": "2026-07-15T12:00:00+00:00",
        "route": "ticket-classifier",
        "func": "app.billing:summarize",
        "module": "app.billing",
        "provider": "openai",
        "model": "gpt-5.6-luna",
        "endpoint": "chat.completions",
        "input_tokens": 1000,
        "output_tokens": 100,
        "cache_read_tokens": 200,
        "reasoning_tokens": 10,
        "latency_ms": 812,
        "status": "stop",
        "session_id": "ticket-123",
        "trace_id": "a" * 32,
        "template_hash": "abc123",
        "tags": {"team": "billing"},
        "environment": "production",
        "sdk": "python",
        "sdk_version": "0.2.0",
        "request_id": "req_1",
        "stream": False,
        "error": False,
    }
    row.update(overrides)
    return row


def test_projection_maps_and_prices():
    values = dict(zip(COLUMNS, project_row(_row(), SNAPSHOT)))
    assert values["func"] == "app.billing:summarize"
    assert values["canonical_model"] == "openai/gpt-5.6-luna"
    assert values["cost_status"] == "priced"
    assert values["cost_usd"] is not None and values["cost_usd"] > 0
    assert values["reasoning_tokens"] == 10
    assert values["trace_id"] == "a" * 32
    assert values["status"] == "stop"
    assert values["status_code"] == "unset"
    assert values["finish_reason"] == "stop"
    assert values["finish_reason_raw"] is None


def test_projection_separates_explicit_status_and_finish_reason():
    values = dict(zip(COLUMNS, project_row(_row(
        status="tool-calls",
        status_code="unset",
        finish_reason="tool-calls",
        finish_reason_raw="tool_use",
        error=False,
    ), SNAPSHOT)))
    assert values["status"] == "tool-calls"
    assert values["status_code"] == "unset"
    assert values["finish_reason"] == "tool-calls"
    assert values["finish_reason_raw"] == "tool_use"


def test_projection_maps_legacy_errors_to_otel_status():
    values = dict(zip(COLUMNS, project_row(_row(
        status="error", error=True, error_type="ProviderError"
    ), SNAPSHOT)))
    assert values["status_code"] == "error"
    assert values["finish_reason"] is None


def test_projection_treats_in_band_provider_failure_as_error():
    values = dict(zip(COLUMNS, project_row(_row(
        status="failed",
        status_code="unset",
        finish_reason="error",
        finish_reason_raw="failed",
        error=False,
    ), SNAPSHOT)))
    assert values["status_code"] == "error"
    assert values["finish_reason"] == "error"
    assert values["error"] is True


def test_projection_drops_redundant_or_orphan_raw_finish_reasons():
    redundant = dict(zip(COLUMNS, project_row(_row(
        finish_reason="stop", finish_reason_raw="stop"
    ), SNAPSHOT)))
    orphan = dict(zip(COLUMNS, project_row(_row(
        status=None, finish_reason="not-valid", finish_reason_raw="provider-secret"
    ), SNAPSHOT)))
    assert redundant["finish_reason_raw"] is None
    assert orphan["finish_reason"] is None
    assert orphan["finish_reason_raw"] is None


def test_content_fields_never_survive():
    row = _row(
        request_json=json.dumps({"messages": [{"content": SENTINEL}]}),
        response_text=SENTINEL,
        content_opted_in=True,
        frames_json=[{"m": "app", "f": "fn", "l": 1}],
        tool_calls=[
            {
                "name": "lookup_account",
                "arguments": SENTINEL,
                "result": SENTINEL,
                "status": "completed",
                "idempotency": "idempotent",
            }
        ],
    )
    values = project_row(row, SNAPSHOT)
    assert SENTINEL not in repr(values)
    named = dict(zip(COLUMNS, values))
    assert json.loads(named["tool_names"]) == ["lookup_account"]


def test_route_falls_back_to_template_hash():
    values = dict(zip(COLUMNS, project_row(_row(route=None), SNAPSHOT)))
    assert values["route"] == "template:abc123"


def test_unknown_model_lands_unpriced():
    values = dict(zip(COLUMNS, project_row(_row(model="mystery-1"), SNAPSHOT)))
    assert values["cost_status"] == "unpriced"
    assert values["cost_usd"] is None


def test_qualified_openrouter_reported_cost_wins_over_catalog():
    values = dict(zip(COLUMNS, project_row(_row(
        gateway="openrouter",
        reported_cost_usd="0.00042",
        reported_cost_source="openrouter.usage.cost",
    ), SNAPSHOT)))

    assert values["cost_usd"] == Decimal("0.00042")
    assert values["cost_status"] == "priced"
    assert values["price_id"] is not None


def test_qualified_openrouter_reported_cost_prices_unknown_model():
    values = dict(zip(COLUMNS, project_row(_row(
        model="unknown/openrouter-model",
        gateway="openrouter",
        reported_cost_usd="0.00042",
        reported_cost_source="openrouter.usage.cost",
    ), SNAPSHOT)))

    assert values["cost_usd"] == Decimal("0.00042")
    assert values["cost_status"] == "priced"
    assert values["price_id"] is None


def test_invalid_openrouter_reported_cost_falls_back_to_catalog():
    expected = dict(zip(COLUMNS, project_row(_row(), SNAPSHOT)))
    values = dict(zip(COLUMNS, project_row(_row(
        gateway="openrouter",
        reported_cost_usd=-1,
        reported_cost_source="openrouter.usage.cost",
    ), SNAPSHOT)))

    assert values["cost_usd"] == expected["cost_usd"]
    assert values["cost_status"] == expected["cost_status"]
    assert values["price_id"] == expected["price_id"]


def test_malformed_values_are_tolerated():
    values = dict(
        zip(
            COLUMNS,
            project_row(
                _row(
                    ts="not-a-date",
                    input_tokens="many",
                    stream="yes",
                    tags="not-a-dict",
                ),
                SNAPSHOT,
            ),
        )
    )
    assert values["ts"] is not None
    assert values["input_tokens"] is None
    assert values["stream"] is None
    assert values["tags"] is None


def test_reliability_projection_is_nullable_and_strict():
    legacy = dict(zip(COLUMNS, project_row(_row(), SNAPSHOT)))
    for key in ('http_status_code', 'availability', 'refused'):
        assert legacy[key] is None
    explicit = dict(zip(COLUMNS, project_row(_row(http_status_code=429, availability='unavailable', refused=None), SNAPSHOT)))
    assert explicit['http_status_code'] == 429
    assert explicit['availability'] == 'unavailable'
    invalid = dict(zip(COLUMNS, project_row(_row(http_status_code=True, availability='broken', refused='false'), SNAPSHOT)))
    assert invalid['http_status_code'] is None
    assert invalid['availability'] is None
    assert invalid['refused'] is None


def test_reliability_projection_resolves_conflicting_dimensions():
    refusal = dict(zip(COLUMNS, project_row(_row(availability='unavailable', refused=True), SNAPSHOT)))
    assert refusal['availability'] is None
    assert refusal['refused'] is None
    assessed = dict(zip(COLUMNS, project_row(_row(refused=True), SNAPSHOT)))
    assert assessed['availability'] == 'available'
    assert assessed['refused'] is True
    unavailable = dict(zip(COLUMNS, project_row(_row(availability='unavailable', refused=False), SNAPSHOT)))
    assert unavailable['refused'] is None


@pytest.mark.parametrize('code,availability', [(429,'unavailable'), (503,'unavailable'), (401,'excluded'), (200,'available')])
def test_reliability_projection_derives_http_outcome(code, availability):
    values = dict(zip(COLUMNS, project_row(_row(http_status_code=code), SNAPSHOT)))
    assert values['availability'] == availability
    assert values['refused'] is None


def test_reliability_projection_http_failure_overrides_non_refusal():
    values = dict(zip(COLUMNS, project_row(_row(http_status_code=429, availability='available', refused=False), SNAPSHOT)))
    assert values['availability'] == 'unavailable'
    assert values['refused'] is None


def test_reliability_projection_explicit_refusal_precedes_http_error():
    values = dict(zip(COLUMNS, project_row(_row(http_status_code=403, refused=True), SNAPSHOT)))
    assert values['http_status_code'] == 403
    assert values['availability'] == 'available'
    assert values['refused'] is True
    conflict = dict(zip(COLUMNS, project_row(_row(http_status_code=429, availability='unavailable', refused=True), SNAPSHOT)))
    assert conflict['http_status_code'] == 429
    assert conflict['availability'] is None
    assert conflict['refused'] is None
