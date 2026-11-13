"""Content-blind ingest: allowlist projection plus inline cost enrichment.

Prompt/completion content (`request_json`, `response_text`, tool-call
arguments) is discarded before anything touches the database.
"""

import json
import os
import zlib
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation

from fastapi import APIRouter, Depends, HTTPException, Request
from metergraph_core import normalize_gateway_evidence, resolve_billing

from . import db
from .auth import authenticated_app_token, require_ingest_token
from .catalog import CatalogSnapshot
from .sessions import issue_session_token

router = APIRouter()

COLUMNS = (
    "ts",
    "route",
    "func",
    "module",
    "provider",
    "model",
    "canonical_model",
    "endpoint",
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "reasoning_tokens",
    "cost_usd",
    "price_id",
    "cost_status",
    "latency_ms",
    "ttft_ms",
    "status",
    "status_code",
    "finish_reason",
    "finish_reason_raw",
    "error",
    "error_type",
    "stream",
    "batch",
    "session_id",
    "trace_id",
    "template_hash",
    "unit_name",
    "unit_count",
    "tool_names",
    "tags",
    "environment",
    "sdk",
    "sdk_version",
    "request_id",
    "http_status_code",
    "availability",
    "refused",
)
_INSERT = (
    f"insert into calls ({', '.join(COLUMNS)})"
    f" values ({', '.join(['%s'] * len(COLUMNS))})"
)
_MAX_TAGS = 32


def _max_body_bytes() -> int:
    return int(os.environ.get("MG_MAX_BODY_BYTES", 8 * 1024 * 1024))


def _max_rows() -> int:
    return int(os.environ.get("MG_MAX_ROWS", 5000))


def _num(value, cast):
    if value is None or isinstance(value, bool):
        return None
    try:
        return cast(value)
    except (ValueError, TypeError, OverflowError, InvalidOperation):
        return None


def _text(value) -> str | None:
    return str(value)[:512] if value is not None else None


def _bool(value) -> bool | None:
    return value if isinstance(value, bool) else None


def _http_status(value) -> int | None:
    return value if type(value) is int and 100 <= value <= 599 else None


def _availability(value) -> str | None:
    return value if isinstance(value, str) and value in {"available", "unavailable", "excluded"} else None


def _timestamp(value) -> datetime:
    ts = value
    if isinstance(ts, (int, float)) and not isinstance(ts, bool):
        try:
            ts = datetime.fromtimestamp(ts, tz=timezone.utc)
        except (ValueError, OSError, OverflowError):
            ts = None
    elif isinstance(ts, str):
        try:
            ts = datetime.fromisoformat(ts)
        except ValueError:
            ts = None
    else:
        ts = None
    if isinstance(ts, datetime) and ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts or datetime.now(timezone.utc)


def _resolved_route(row: dict) -> str | None:
    explicit = str(row.get("route") or "").strip()
    if explicit:
        return explicit[:512]
    fingerprint = str(row.get("template_hash") or "").strip()
    if fingerprint:
        return f"template:{fingerprint}"[:512]
    return None


def _tool_names(value) -> str | None:
    if not isinstance(value, list):
        return None
    names = [
        str(item.get("name"))[:128]
        for item in value
        if isinstance(item, dict) and item.get("name")
    ]
    return json.dumps(names) if names else None


def _tags(value) -> str | None:
    if not isinstance(value, dict) or not value:
        return None
    kept = {
        str(key)[:64]: str(item)[:256]
        for key, item in list(value.items())[:_MAX_TAGS]
        if isinstance(item, (str, int, float, bool)) or item is None
    }
    return json.dumps(kept) if kept else None


def _normalized_finish_reason(value) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = "-".join(value.strip().lower().replace("_", "-").split())
    aliases = {
        "end-turn": "stop",
        "stop-sequence": "stop",
        "completed": "stop",
        "succeeded": "stop",
        "max-tokens": "length",
        "max-output-tokens": "length",
        "safety": "content-filter",
        "blocked": "content-filter",
        "tool-use": "tool-calls",
        "function-call": "tool-calls",
        "failed": "error",
        "unknown": "other",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in {
        "stop", "length", "content-filter", "tool-calls", "error", "other"
    } else None


def _status_fields(row: dict) -> tuple[str, str | None, str | None]:
    legacy_status = _text(row.get("status"))
    status_code = row.get("status_code")
    if status_code not in {"unset", "ok", "error"}:
        status_code = "error" if row.get("error") is True or legacy_status == "error" else "unset"
    finish_reason = _normalized_finish_reason(row.get("finish_reason"))
    if finish_reason is None and legacy_status not in {"error", "success", "abandoned"}:
        finish_reason = _normalized_finish_reason(legacy_status)
    if finish_reason == "error":
        status_code = "error"
    finish_reason_raw = _text(row.get("finish_reason_raw"))
    if not finish_reason or not finish_reason_raw or finish_reason_raw == finish_reason:
        finish_reason_raw = None
    return status_code, finish_reason, finish_reason_raw


def project_row(row: dict, catalog: CatalogSnapshot) -> tuple:
    """Map one wire row onto the calls columns; everything else is dropped."""
    ts = _timestamp(row.get("ts"))
    enrichment = catalog.cost(
        provider=row.get("provider"),
        model=row.get("model"),
        at=ts,
        input_tokens=row.get("input_tokens"),
        output_tokens=row.get("output_tokens"),
        cache_read_tokens=row.get("cache_read_tokens"),
        cache_write_tokens=row.get("cache_write_tokens"),
        batch=row.get("batch") is True,
    )
    billing = resolve_billing(enrichment, normalize_gateway_evidence(row))
    status_code, finish_reason, finish_reason_raw = _status_fields(row)
    http_status_code = _http_status(row.get("http_status_code"))
    availability = _availability(row.get("availability"))
    refused = _bool(row.get("refused"))
    if refused is True:
        if availability in {"unavailable", "excluded"}:
            availability, refused = None, None
        else:
            availability = "available"
    elif http_status_code == 429 or http_status_code is not None and http_status_code >= 500:
        availability = "unavailable"
    elif http_status_code is not None and 400 <= http_status_code < 500:
        availability = "excluded"
    elif http_status_code is not None and 200 <= http_status_code < 300 and availability is None:
        availability = "available"
    if availability in {"unavailable", "excluded"}:
        refused = None
    return (
        ts,
        _resolved_route(row),
        _text(row.get("func")),
        _text(row.get("module")),
        _text(row.get("provider")),
        _text(row.get("model")),
        enrichment.canonical_model,
        _text(row.get("endpoint")),
        _num(row.get("input_tokens"), int),
        _num(row.get("output_tokens"), int),
        _num(row.get("cache_read_tokens"), int),
        _num(row.get("cache_write_tokens"), int),
        _num(row.get("reasoning_tokens"), int),
        billing.cost_usd,
        billing.catalog_price_id,
        billing.cost_status,
        _num(row.get("latency_ms"), int),
        _num(row.get("ttft_ms"), int),
        _text(row.get("status")),
        status_code,
        finish_reason,
        finish_reason_raw,
        status_code == "error",
        _text(row.get("error_type")),
        _bool(row.get("stream")),
        _bool(row.get("batch")),
        _text(row.get("session_id")),
        _text(row.get("trace_id")),
        _text(row.get("template_hash")),
        _text(row.get("unit_name")),
        _num(row.get("unit_count"), Decimal),
        _tool_names(row.get("tool_calls")),
        _tags(row.get("tags")),
        _text(row.get("environment")),
        _text(row.get("sdk")),
        _text(row.get("sdk_version")),
        _text(row.get("request_id")),
        http_status_code,
        availability,
        refused,
    )


def _gunzip_bounded(compressed: bytes, limit: int) -> bytes:
    """Decompress gzip data, aborting once output exceeds limit.

    A single `gzip.decompress()` call materializes the whole output before
    any size check runs, so a small, highly-compressible payload (a
    "decompression bomb") can force an arbitrarily large allocation. Capping
    each `decompress()` call with `max_length` bounds any single allocation
    to ~limit bytes regardless of the true decompressed size; looping on
    `unconsumed_tail` (rather than re-slicing the input ourselves) is
    required because a capped call can leave input unprocessed, and
    resubmitting the wrong slice would silently corrupt the output.
    """
    decompressor = zlib.decompressobj(wbits=zlib.MAX_WBITS | 16)
    chunks = []
    total = 0
    pending = compressed
    while pending:
        piece = decompressor.decompress(pending, limit - total + 1)
        total += len(piece)
        if total > limit:
            raise HTTPException(413, "decompressed body too large")
        chunks.append(piece)
        pending = decompressor.unconsumed_tail
    tail = decompressor.flush()
    total += len(tail)
    if total > limit:
        raise HTTPException(413, "decompressed body too large")
    chunks.append(tail)
    return b"".join(chunks)


def _decode_body(body: bytes, encoding: str | None, limit: int) -> dict:
    if len(body) > limit:
        raise HTTPException(413, "request body too large")
    if (encoding or "").strip().lower() == "gzip":
        try:
            body = _gunzip_bounded(body, limit)
        except (OSError, EOFError, zlib.error) as exc:
            raise HTTPException(400, "invalid gzip body") from exc
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError) as exc:
        raise HTTPException(400, "invalid JSON body") from exc
    if not isinstance(payload, dict):
        raise HTTPException(400, "body must be a JSON object")
    return payload


@router.post("/v1/ingest/sessions", status_code=201)
async def create_ingest_session(
    request: Request,
    app_token: str = Depends(authenticated_app_token),
):
    payload = _decode_body(
        await request.body(),
        request.headers.get("content-encoding"),
        _max_body_bytes(),
    )
    repository = payload.get("repository")
    sdk_version = payload.get("sdk_version")
    if payload.get("protocol_version") != 2:
        raise HTTPException(400, "unsupported SDK ingestion request")
    if not isinstance(repository, str) or not repository.strip():
        raise HTTPException(400, "repository must be a non-empty string")
    if len(repository.strip()) > 512:
        raise HTTPException(400, "repository must be <= 512 characters")
    if not isinstance(sdk_version, str) or not sdk_version.strip():
        raise HTTPException(400, "sdk_version must be a non-empty string")
    if len(sdk_version.strip()) > 64:
        raise HTTPException(400, "sdk_version must be <= 64 characters")

    session_token, expires_at = issue_session_token(
        app_token, repository.strip()
    )
    return {
        "session_token": session_token,
        "expires_at": expires_at.isoformat().replace("+00:00", "Z"),
    }


@router.post(
    "/v1/ingest", status_code=202, dependencies=[Depends(require_ingest_token)]
)
async def ingest(request: Request):
    payload = _decode_body(
        await request.body(),
        request.headers.get("content-encoding"),
        _max_body_bytes(),
    )
    version = payload.get("schema_version", 1)
    if version != 1:
        raise HTTPException(400, "unsupported schema_version")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise HTTPException(400, "rows must be a non-empty list")
    if len(rows) > _max_rows():
        raise HTTPException(400, f"rows must be <= {_max_rows()}")
    if not all(isinstance(row, dict) for row in rows):
        raise HTTPException(400, "each row must be an object")

    catalog = request.app.state.catalog
    calls = [
        project_row(row, catalog)
        for row in rows
        if row.get("event_type") != "outcome"
    ]
    if calls:
        with db.pool().connection() as con:
            with con.cursor() as cur:
                cur.executemany(_INSERT, calls)
    return {"accepted": len(calls), "ignored": len(rows) - len(calls)}
