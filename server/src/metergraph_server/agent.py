"""Content-free Agent Access for the Metergraph OSS server."""

from __future__ import annotations

import base64
import binascii
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, Query, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from . import agent_contract, agent_mcp, db
from .auth import require_agent_token


MAX_AGENT_DAYS = 90
MAX_AGENT_ROWS = 200
_UNSUPPORTED = frozenset(
    {
        "ingestion_health",
        "incidents",
        "reports",
        "report_detail",
        "report_evidence",
        "trace_content",
        "trace_replay",
    }
)

router = APIRouter(dependencies=[Depends(require_agent_token)], redirect_slashes=False)


def _workspace_slug() -> str:
    return os.environ.get("MG_WORKSPACE_SLUG", "self-hosted")


def _workspace_name() -> str:
    return os.environ.get("MG_WORKSPACE_NAME", "Metergraph OSS")


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _error_document(code: str, message: str) -> dict[str, Any]:
    return agent_contract.error_document(code, message)


def _bounded_json(document: Any) -> tuple[dict[str, Any] | None, bytes]:
    encoded = jsonable_encoder(document)
    wire = json.dumps(encoded, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    if len(wire) > agent_mcp.MAX_RESPONSE_BYTES:
        return None, b""
    return encoded, wire


def _rest_response(document: dict[str, Any], status_code: int = 200) -> JSONResponse:
    encoded, _ = _bounded_json(document)
    if encoded is None:
        document = _error_document(
            "response_too_large", "Metergraph API response exceeded 5 MiB"
        )
        status_code = 413
    return JSONResponse(encoded if encoded is not None else document, status_code=status_code)


def _provenance() -> dict[str, str]:
    return agent_contract.provenance(_workspace_slug())


def _envelope(**fields: Any) -> dict[str, Any]:
    return {
        "schema_version": agent_contract.CONTRACT_VERSION,
        "provenance": _provenance(),
        **fields,
    }


def _unsupported(capability: str) -> dict[str, Any]:
    return _error_document(
        "unsupported_capability",
        f"Capability {capability} is not available in the OSS server",
    )


def _capability_available(name: str) -> bool:
    return name not in _UNSUPPORTED


def _workspace_context() -> dict[str, Any]:
    with db.pool().connection() as con:
        row = con.execute("select min(ts) from calls").fetchone()
    retention: int | None = None
    configured = os.environ.get("MG_RETENTION_DAYS")
    if configured is not None:
        try:
            retention = int(configured)
        except ValueError:
            retention = None
    slug = _workspace_slug()
    return _envelope(
        workspace={
            "id": slug,
            "slug": slug,
            "name": _workspace_name(),
            "created_at": _iso(row[0]) if row and row[0] is not None else None,
        },
        retention={"metadata_days": retention},
        content={"captured": False, "included": False},
        access={"scopes": ["agent:read"]},
    )


def _capability_context() -> dict[str, Any]:
    contracts = agent_contract.TOOL_CONTRACT
    agent: dict[str, Any] = {}
    for name, contract in contracts.items():
        capability = contract["capability"]
        agent[capability] = {
            "available": _capability_available(capability),
            "content": capability == "trace_content",
            "mutates": False,
            "privacy_class": contract["privacy_class"],
            "required_scope": contract["required_scope"],
            "schema": contract["schema"],
            "external_calls": contract["external_calls"],
        }
    return _envelope(
        deployment_profile="oss",
        deployment_capabilities=["telemetry", "agent_api"],
        agent=agent,
        privacy_classes={
            privacy_class: {
                "description": agent_contract.PRIVACY_DESCRIPTIONS[privacy_class],
                "fields": sorted(agent_contract.SAFE_FIELDS[privacy_class]),
            }
            for privacy_class in sorted(agent_contract.PRIVACY_CLASSES)
        },
        contracts={
            "agent_access": agent_contract.CONTRACT_VERSION,
            "trace_debug": agent_contract.TRACE_DEBUG_VERSION,
        },
        bounds={
            "max_days": MAX_AGENT_DAYS,
            "max_rows": MAX_AGENT_ROWS,
            "max_response_bytes": agent_mcp.MAX_RESPONSE_BYTES,
            "content_included_by_default": False,
        },
    )


def _routes() -> dict[str, Any]:
    with db.pool().connection() as con:
        rows = con.execute(
            """
            select coalesce(route, '(unrouted)'), count(*), max(ts)
              from calls
             group by 1
             order by 1
            """
        ).fetchall()
    return _envelope(
        routes=[
            {
                "route": row[0],
                "description": None,
                "constraints": {},
                "evaluation_contract": None,
                "evaluation_contract_version": None,
                "evaluation_contract_hash": None,
                "updated_at": row[2].isoformat(),
                "calls": int(row[1]),
                "replay_eligible_calls": 0,
            }
            for row in rows
        ]
    )


def _bounds(days: int) -> tuple[datetime, datetime]:
    until = datetime.now(timezone.utc)
    return until - timedelta(days=days), until


def _usage(days: int = 7, limit: int = 100) -> dict[str, Any]:
    since, until = _bounds(days)
    with db.pool().connection() as con:
        rows = con.execute(
            """
            select date_trunc('day', ts at time zone 'UTC')::date::text,
                   coalesce(route, '(unrouted)'), count(*),
                   coalesce(sum(cost_usd), 0),
                   coalesce(sum(input_tokens), 0),
                   coalesce(sum(output_tokens), 0),
                   round(avg(latency_ms)),
                   percentile_cont(0.95) within group (order by latency_ms),
                   count(*) filter (where error is true or status = 'error'),
                   count(*) filter (where cost_status = 'partial')
              from calls
             where ts >= %s and ts < %s
             group by 1, 2
             order by 1 desc, 3 desc, 2
             limit %s
            """,
            (since, until, limit),
        ).fetchall()
    truncated = len(rows) == limit
    return _envelope(
        window=agent_contract.window(days),
        evidence=agent_contract.evidence(["calls"], len(rows), not truncated),
        warnings=(
            [agent_contract.warning("truncated", "The usage result reached its row limit.")]
            if truncated
            else []
        ),
        days=days,
        content_included=False,
        truncated=truncated,
        items=[
            {
                "date": row[0],
                "route": row[1],
                "calls": int(row[2]),
                "cost_usd": float(row[3]),
                "input_tokens": int(row[4]),
                "output_tokens": int(row[5]),
                "avg_latency_ms": int(row[6]) if row[6] is not None else None,
                "p95_latency_ms": round(row[7]) if row[7] is not None else None,
                "error_calls": int(row[8] or 0),
                "partial_cost_calls": int(row[9] or 0),
            }
            for row in rows
        ],
    )


def _decode_cursor(cursor: str) -> tuple[datetime, str]:
    try:
        padded = cursor.encode("ascii") + b"=" * (-len(cursor) % 4)
        value = json.loads(base64.urlsafe_b64decode(padded))
        timestamp = datetime.fromisoformat(value["last_span_at"])
        trace_id = value["trace_id"]
    except (ValueError, KeyError, TypeError, UnicodeError, binascii.Error) as exc:
        raise ValueError("cursor must be a valid trace cursor") from exc
    if timestamp.tzinfo is None or not isinstance(trace_id, str) or not trace_id:
        raise ValueError("cursor must be a valid trace cursor")
    return timestamp, trace_id


def _encode_cursor(timestamp: datetime, trace_id: str) -> str:
    value = json.dumps(
        {"last_span_at": timestamp.isoformat(), "trace_id": trace_id},
        separators=(",", ":"),
    ).encode("utf-8")
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _traces(
    *,
    days: int = 7,
    limit: int = 50,
    route: str | None = None,
    status: str | None = None,
    workload: str | None = None,
    cursor: str | None = None,
) -> dict[str, Any]:
    since, until = _bounds(days)
    where = ["ts >= %s", "ts < %s", "trace_id is not null"]
    params: list[Any] = [since, until]
    if route:
        where.append("route = %s")
        params.append(route)
    if status:
        if status == "error":
            where.append("(error is true or status = 'error')")
        elif status == "success":
            where.append("coalesce(error, false) is false and coalesce(status, '') <> 'error'")
        else:
            where.append("status = %s")
            params.append(status)
    if workload:
        where.append("route = %s")
        params.append(workload)
    decoded = _decode_cursor(cursor) if cursor else None
    cursor_where = ""
    if decoded:
        cursor_where = " where (max_ts, trace_id) < (%s, %s)"
        params.extend(decoded)
    params.append(limit)
    with db.pool().connection() as con:
        rows = con.execute(
            """
            with grouped as (
                select trace_id,
                       coalesce(max(func), 'LLM trace') as trace_name,
                       min(ts) as started_at,
                       max(ts) as max_ts,
                       count(*) as span_count,
                       coalesce(sum(input_tokens), 0) as input_tokens,
                       coalesce(sum(output_tokens), 0) as output_tokens,
                       coalesce(sum(cache_read_tokens), 0) as cache_read_tokens,
                       coalesce(sum(cache_write_tokens), 0) as cache_write_tokens,
                       coalesce(sum(cost_usd), 0) as cost_usd,
                       count(*) filter (where cost_status = 'partial') as partial_cost_calls,
                       bool_or(coalesce(error, false) or status = 'error') as has_error,
                       array_remove(array_agg(distinct route), null) as routes,
                       array_remove(array_agg(distinct provider), null) as providers,
                       array_remove(array_agg(distinct model), null) as models
                  from calls
                 where """
            + " and ".join(where)
            + """
                 group by trace_id
            )
            select trace_id, trace_name, started_at, max_ts, span_count,
                   input_tokens, output_tokens, cache_read_tokens,
                   cache_write_tokens, cost_usd, partial_cost_calls, has_error, routes,
                   providers, models
              from grouped
            """
            + cursor_where
            + """
             order by max_ts desc, trace_id desc
             limit %s
            """,
            tuple(params),
        ).fetchall()
    truncated = len(rows) == limit
    next_cursor = _encode_cursor(rows[-1][3], rows[-1][0]) if truncated else None
    return _envelope(
        window=agent_contract.window(days),
        page=agent_contract.page(limit, truncated, next_cursor),
        evidence=agent_contract.evidence(
            ["calls (rows with trace_id)"], len(rows), not truncated
        ),
        warnings=(
            [agent_contract.warning("truncated", "The trace result reached its row limit.")]
            if truncated
            else []
        ),
        content_included=False,
        truncated=truncated,
        next_cursor=next_cursor,
        traces=[
            {
                "id": row[0],
                "trace_id": row[0],
                "trace_name": row[1],
                "started_at": row[2].isoformat(),
                "last_span_at": row[3].isoformat(),
                "span_count": int(row[4]),
                "input_tokens": int(row[5]),
                "output_tokens": int(row[6]),
                "cache_read_tokens": int(row[7]),
                "cache_write_tokens": int(row[8]),
                "cost_usd": float(row[9]) if row[9] is not None else None,
                "partial_cost_calls": int(row[10]),
                "status": "error" if row[11] else "success",
                "routes": row[12] or [],
                "providers": row[13] or [],
                "models": row[14] or [],
            }
            for row in rows
        ],
    )


def _dispatch(name: str, arguments: Any) -> dict[str, Any]:
    arguments = agent_mcp._validate_arguments(name, arguments)
    contract = agent_contract.TOOL_CONTRACT[name]
    capability = contract["capability"]
    if not _capability_available(capability):
        return _unsupported(capability)
    if name == "metergraph_get_workspace_context":
        return _workspace_context()
    if name == "metergraph_get_capabilities":
        return _capability_context()
    if name == "metergraph_list_routes":
        return _routes()
    if name == "metergraph_get_usage":
        return _usage(arguments.get("days", 7), arguments.get("limit", 100))
    if name == "metergraph_query_traces":
        return _traces(
            days=arguments.get("days", 7),
            limit=arguments.get("limit", 50),
            route=arguments.get("route"),
            status=arguments.get("status"),
            workload=arguments.get("workload"),
            cursor=arguments.get("cursor"),
        )
    raise KeyError(name)


def _mcp_response(request_id: Any, *, result: Any = None, error: dict[str, Any] | None = None):
    response: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id}
    response["error" if error is not None else "result"] = error if error is not None else result
    return response


def _mcp_tool_error(request_id: Any, document: dict[str, Any]) -> dict[str, Any]:
    _, wire = _bounded_json(document)
    text = wire.decode("utf-8") if wire else json.dumps(document, sort_keys=True)
    return _mcp_response(
        request_id,
        result={
            "content": [{"type": "text", "text": text}],
            "structuredContent": document,
            "isError": True,
        },
    )


def _handle_mcp(message: Any) -> dict[str, Any] | None:
    if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
        return _mcp_response(None, error={"code": -32600, "message": "Invalid Request"})
    if "id" not in message:
        return None
    request_id = message.get("id")
    method = message.get("method")
    if not isinstance(method, str):
        return _mcp_response(request_id, error={"code": -32600, "message": "Invalid Request"})
    if method == "initialize":
        params = message.get("params") or {}
        if not isinstance(params, dict):
            return _mcp_response(request_id, error={"code": -32602, "message": "Invalid params"})
        requested = params.get("protocolVersion")
        selected = requested if requested in agent_mcp.SUPPORTED_PROTOCOLS else agent_mcp.PROTOCOL_VERSION
        return _mcp_response(
            request_id,
            result={
                "protocolVersion": selected,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": agent_mcp.SERVER_NAME, "version": agent_mcp.SERVER_VERSION},
                "instructions": "Metergraph OSS Agent Access is tenant-scoped and content-blind.",
            },
        )
    if method == "ping":
        return _mcp_response(request_id, result={})
    if method == "tools/list":
        return _mcp_response(request_id, result={"tools": agent_mcp.TOOLS})
    if method != "tools/call":
        return _mcp_response(request_id, error={"code": -32601, "message": "Method not found"})
    params = message.get("params")
    if not isinstance(params, dict) or not isinstance(params.get("name"), str):
        return _mcp_response(
            request_id, error={"code": -32602, "message": "Invalid tool call parameters"}
        )
    try:
        document = _dispatch(params["name"], params.get("arguments"))
    except KeyError:
        return _mcp_response(request_id, error={"code": -32602, "message": "Unknown tool"})
    except ValueError as exc:
        return _mcp_response(request_id, error={"code": -32602, "message": str(exc)})
    if document.get("error"):
        return _mcp_tool_error(request_id, document)
    encoded, wire = _bounded_json(document)
    if encoded is None:
        return _mcp_tool_error(
            request_id,
            _error_document("response_too_large", "Metergraph API response exceeded 5 MiB"),
        )
    return _mcp_response(
        request_id,
        result={
            "content": [{"type": "text", "text": wire.decode("utf-8")}],
            "structuredContent": encoded,
            "isError": False,
        },
    )


@router.get("/v1/agent/workspace")
def agent_workspace():
    return _rest_response(_workspace_context())


@router.get("/v1/agent/capabilities")
def agent_capabilities():
    return _rest_response(_capability_context())


@router.get("/v1/agent/routes")
def agent_routes():
    return _rest_response(_routes())


@router.get("/v1/agent/usage")
def agent_usage(
    days: int = Query(7, ge=1, le=MAX_AGENT_DAYS),
    limit: int = Query(100, ge=1, le=MAX_AGENT_ROWS),
):
    return _rest_response(_usage(days, limit))


@router.get("/v1/agent/traces")
def agent_traces(
    days: int = Query(7, ge=1, le=MAX_AGENT_DAYS),
    limit: int = Query(50, ge=1, le=MAX_AGENT_ROWS),
    route: str | None = Query(None, min_length=1, max_length=512),
    status: str | None = Query(None, min_length=1, max_length=128),
    workload: str | None = Query(None, min_length=1, max_length=200),
    cursor: str | None = Query(None, min_length=1, max_length=512),
):
    try:
        document = _traces(
            days=days, limit=limit, route=route, status=status, workload=workload, cursor=cursor
        )
    except ValueError as exc:
        return _rest_response(_error_document("invalid_argument", str(exc)), 422)
    return _rest_response(document)


@router.get("/v1/agent/ingestion-health")
def agent_ingestion_health(days: int = Query(7, ge=1, le=MAX_AGENT_DAYS)):
    return _rest_response(_unsupported("ingestion_health"), 501)


@router.get("/v1/agent/incidents")
def agent_incidents(limit: int = Query(50, ge=1, le=MAX_AGENT_ROWS)):
    return _rest_response(_unsupported("incidents"), 501)


@router.get("/v1/agent/reports")
def agent_reports(limit: int = Query(50, ge=1, le=MAX_AGENT_ROWS)):
    return _rest_response(_unsupported("reports"), 501)


@router.get("/v1/agent/reports/{analysis_id}")
def agent_report(analysis_id: str):
    return _rest_response(_unsupported("report_detail"), 501)


@router.get("/v1/agent/reports/{analysis_id}/workloads/{workload_id}/evidence")
def agent_report_evidence(analysis_id: str, workload_id: str, limit: int = Query(50, ge=1, le=50)):
    return _rest_response(_unsupported("report_evidence"), 501)


@router.get("/v1/agent/debug/traces/{trace_id}")
def agent_trace(trace_id: str):
    return _rest_response(_unsupported("trace_content"), 501)


@router.post("/v1/agent/debug/replay")
def agent_replay(payload: dict[str, Any]):
    return _rest_response(_unsupported("trace_replay"), 501)


@router.post("/v1/agent/mcp")
async def agent_mcp_transport(request: Request):
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > agent_mcp.MAX_MESSAGE_BYTES:
                return _rest_response(
                    _mcp_response(None, error={"code": -32700, "message": "Parse error"}), 413
                )
        except ValueError:
            pass
    raw = await request.body()
    if len(raw) > agent_mcp.MAX_MESSAGE_BYTES:
        return _rest_response(
            _mcp_response(None, error={"code": -32700, "message": "Parse error"}), 413
        )
    try:
        message = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return _rest_response(
            _mcp_response(None, error={"code": -32700, "message": "Parse error"}), 400
        )
    response = _handle_mcp(message)
    if response is None:
        return Response(status_code=202)
    return _rest_response(response)
