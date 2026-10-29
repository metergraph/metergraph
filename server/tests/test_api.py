"""API integration tests. Require Postgres: set MG_TEST_DATABASE_URL."""

import gzip
import json
import os

import pytest

TEST_DSN = os.environ.get("MG_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    not TEST_DSN, reason="MG_TEST_DATABASE_URL not set"
)

SENTINEL = "TOP-SECRET-PROMPT-CONTENT"
TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture()
def client():
    os.environ["DATABASE_URL"] = TEST_DSN
    os.environ["MG_TOKENS"] = TOKEN
    import psycopg

    with psycopg.connect(TEST_DSN) as con:
        con.execute("drop table if exists calls, schema_migrations")

    from fastapi.testclient import TestClient

    from metergraph_server import db
    from metergraph_server.main import create_app

    db.close()
    with TestClient(create_app()) as test_client:
        yield test_client
    db.close()


def _rows():
    return [
        {
            "ts": "2026-07-15T10:00:00+00:00",
            "func": "app.billing:summarize",
            "route": "summarizer",
            "provider": "openai",
            "model": "gpt-5.6-luna",
            "input_tokens": 100_000,
            "output_tokens": 0,
            "latency_ms": 500,
            "error": False,
            "request_json": SENTINEL,
            "response_text": SENTINEL,
            "content_opted_in": True,
        },
        {
            "ts": "2026-07-15T11:00:00+00:00",
            "func": "app.support:classify",
            "provider": "google",
            "model": "gemini-2.5-flash",
            "input_tokens": 10_000,
            "output_tokens": 1_000,
            "latency_ms": 300,
            "error": True,
            "error_type": "APIError",
            "sdk": "js",
        },
        {"event_type": "outcome", "event_id": "e1", "route": "summarizer"},
    ]


def test_ingest_requires_token(client):
    assert client.post("/v1/ingest", json={"rows": [{}]}).status_code == 401
    bad = client.post(
        "/v1/ingest",
        json={"rows": [{}]},
        headers={"Authorization": "Bearer wrong"},
    )
    assert bad.status_code == 401


def test_sdk_0_4_session_exchange_and_ingest_roundtrip(client):
    session = client.post(
        "/v1/ingest/sessions",
        json={
            "protocol_version": 2,
            "repository": "github.com/acme/widgets",
            "sdk_version": "0.4.0",
        },
        headers=AUTH,
    )
    assert session.status_code == 201
    session_token = session.json()["session_token"]

    response = client.post(
        "/v1/ingest",
        json={"schema_version": 1, "rows": _rows()[:1]},
        headers={"Authorization": f"Bearer {session_token}"},
    )

    assert response.status_code == 202
    assert response.json() == {"accepted": 1, "ignored": 0}


def test_ingest_and_usage_roundtrip(client):
    rows = _rows()
    rows[0].update({
        "session_id": "session-e2e",
        "trace_id": "b" * 32,
        "template_hash": "template-e2e",
        "tool_calls": [{"name": "lookup_account", "arguments": {"id": "acct-1"}}],
        "environment": "qualification",
        "sdk": "typescript",
        "sdk_version": "0.6.0",
        "request_id": "request-e2e",
    })
    response = client.post(
        "/v1/ingest",
        json={"schema_version": 1, "rows": rows},
        headers=AUTH,
    )
    assert response.status_code == 202
    assert response.json() == {"accepted": 2, "ignored": 1}

    usage = client.get(
        "/v1/usage",
        params={"group_by": "func", "from": "2026-07-15T00:00:00+00:00", "to": "2026-07-16T00:00:00+00:00"},
        headers=AUTH,
    ).json()["items"]
    by_func = {item["key"]: item for item in usage}
    assert by_func["app.billing:summarize"]["cost_usd"] == pytest.approx(0.1)
    assert by_func["app.support:classify"]["error_rate"] == pytest.approx(1.0)

    series = client.get(
        "/v1/usage/timeseries",
        params={"group_by": "model", "bucket": "day", "from": "2026-07-15T00:00:00+00:00", "to": "2026-07-16T00:00:00+00:00"},
        headers=AUTH,
    ).json()
    assert "2026-07-15" in series["buckets"]
    assert any(entry["key"] == "gpt-5.6-luna" for entry in series["series"])

    calls = client.get("/v1/calls", params={"func": "app.support:classify"}, headers=AUTH).json()["items"]
    assert len(calls) == 1
    assert calls[0]["error_type"] == "APIError"
    assert calls[0]["status_code"] == "error"
    assert calls[0]["finish_reason"] is None
    assert calls[0]["canonical_model"] == "google/gemini-2.5-flash"
    assert calls[0]["catalog_cost_usd"] == calls[0]["cost_usd"]
    assert calls[0]["catalog_price_id"]
    assert calls[0]["catalog_reasons"] == []
    assert calls[0]["reported_cost_usd"] is None
    assert calls[0]["sdk"] == "typescript"
    assert by_func["app.support:classify"]["reported_calls"] == 0

    qualified = client.get(
        "/v1/calls", params={"route": "summarizer"}, headers=AUTH
    ).json()["items"][0]
    assert qualified["tool_names"] == ["lookup_account"]
    assert qualified["template_hash"] == "template-e2e"
    assert qualified["environment"] == "qualification"
    assert qualified["sdk"] == "typescript"
    assert qualified["sdk_version"] == "0.6.0"
    assert qualified["request_id"] == "request-e2e"
    assert qualified["trace_id"] == "b" * 32
    assert qualified["session_id"] == "session-e2e"


def test_environment_options_and_multi_selection_include_untagged(client):
    rows = []
    for index, environment in enumerate(("demo", "prod", None)):
        row = dict(_rows()[0])
        row["func"] = f"environment.choice:{index}"
        if environment is not None:
            row["environment"] = environment
        rows.append(row)
    assert client.post("/v1/ingest", json={"rows": rows}, headers=AUTH).status_code == 202

    window = {
        "from": "2026-07-15T00:00:00+00:00",
        "to": "2026-07-16T00:00:00+00:00",
    }
    options = client.get("/v1/environments", params=window, headers=AUTH).json()["items"]
    assert options == [
        {"value": "demo", "calls": 1},
        {"value": "prod", "calls": 1},
        {"value": None, "calls": 1},
    ]

    selected = client.get(
        "/v1/usage",
        params=[
            ("group_by", "func"),
            ("from", window["from"]),
            ("to", window["to"]),
            ("environment", "prod"),
            ("include_untagged", "true"),
        ],
        headers=AUTH,
    ).json()["items"]
    assert {item["key"] for item in selected} == {
        "environment.choice:1",
        "environment.choice:2",
    }

    legacy_exact = client.get(
        "/v1/usage",
        params={**window, "group_by": "func", "environment": "demo"},
        headers=AUTH,
    ).json()["items"]
    assert [item["key"] for item in legacy_exact] == ["environment.choice:0"]

    none_selected = client.get(
        "/v1/usage",
        params={**window, "include_untagged": "false"},
        headers=AUTH,
    ).json()["items"]
    assert none_selected == []


def test_content_never_reaches_database(client):
    client.post("/v1/ingest", json={"rows": _rows()}, headers=AUTH)
    import psycopg

    with psycopg.connect(TEST_DSN) as con:
        rows = con.execute("select to_jsonb(calls.*)::text from calls").fetchall()
    dump = json.dumps([row[0] for row in rows])
    assert SENTINEL not in dump


def test_gzip_ingest(client):
    body = gzip.compress(json.dumps({"rows": _rows()[:1]}).encode())
    response = client.post(
        "/v1/ingest",
        content=body,
        headers={**AUTH, "Content-Encoding": "gzip", "Content-Type": "application/json"},
    )
    assert response.status_code == 202


def test_config_etag(client):
    first = client.get("/v1/config")
    assert first.status_code == 200
    etag = first.headers["ETag"]
    second = client.get("/v1/config", headers={"If-None-Match": etag})
    assert second.status_code == 304


def test_healthz_reports_catalog_identity(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert response.json()["catalog_version"] == "2026-10-28"
    assert len(response.json()["catalog_hash"]) == 64
