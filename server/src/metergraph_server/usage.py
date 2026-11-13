"""Aggregation and drill-down queries over the calls table."""

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from . import db
from .auth import require_token

router = APIRouter(dependencies=[Depends(require_token)])

_GROUPS = {
    "func": "coalesce(func, '(unattributed)')",
    "module": "coalesce(module, '(unattributed)')",
    "route": "coalesce(route, '(unrouted)')",
    "model": "coalesce(model, '(unknown)')",
    "provider": "coalesce(provider, '(unknown)')",
    "day": "to_char(date_trunc('day', ts at time zone 'UTC'), 'YYYY-MM-DD')",
    "hour": "to_char(date_trunc('hour', ts at time zone 'UTC'), 'YYYY-MM-DD\"T\"HH24:00')",
}
_BUCKETS = {"hour": timedelta(hours=1), "day": timedelta(days=1)}


def _window(
    from_: str | None, to: str | None, default_days: int = 7
) -> tuple[datetime, datetime]:
    now = datetime.now(timezone.utc)
    try:
        start = datetime.fromisoformat(from_) if from_ else now - timedelta(days=default_days)
        end = datetime.fromisoformat(to) if to else now
    except ValueError as exc:
        raise HTTPException(400, "from/to must be ISO timestamps") from exc
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    if end <= start:
        raise HTTPException(400, "to must be after from")
    return start, end


def _filters(
    environments: list[str] | None,
    include_untagged: bool | None,
    route: str | None,
    model: str | None,
):
    clauses, params = [], []
    if environments is not None or include_untagged is not None:
        environments = environments or []
        if environments and include_untagged:
            clauses.append("(environment = any(%s) or environment is null)")
            params.append(environments)
        elif environments:
            clauses.append("environment = any(%s)")
            params.append(environments)
        elif include_untagged:
            clauses.append("environment is null")
        else:
            clauses.append("false")
    if route:
        clauses.append("route = %s")
        params.append(route)
    if model:
        clauses.append("model = %s")
        params.append(model)
    return "".join(f" and {clause}" for clause in clauses), params


@router.get("/v1/usage")
def usage(
    group_by: str = Query("func"),
    from_: str | None = Query(None, alias="from"),
    to: str | None = None,
    environment: list[str] | None = Query(None),
    include_untagged: bool | None = None,
    route: str | None = None,
    model: str | None = None,
):
    key = _GROUPS.get(group_by)
    if key is None:
        raise HTTPException(400, f"group_by must be one of {sorted(_GROUPS)}")
    start, end = _window(from_, to)
    where, params = _filters(environment, include_untagged, route, model)
    provider_col = (
        ", coalesce(provider, '(unknown)') as provider" if group_by == "model" else ""
    )
    group = "1, provider" if group_by == "model" else "1"
    sql = f"""
        select {key} as key{provider_col},
               count(*) as calls,
               coalesce(sum(cost_usd), 0) as cost_usd,
               coalesce(sum(input_tokens), 0) as input_tokens,
               coalesce(sum(output_tokens), 0) as output_tokens,
               coalesce(sum(cache_read_tokens), 0) as cache_read_tokens,
               coalesce(sum(reasoning_tokens), 0) as reasoning_tokens,
               round(avg(latency_ms)) as avg_latency_ms,
               percentile_cont(0.95) within group (order by latency_ms) as p95_latency_ms,
               avg(case when error then 1.0 else 0.0 end) as error_rate,
               count(*) filter (where cost_status = 'unpriced') as unpriced_calls,
               0::bigint as reported_calls,
               count(*) filter (where availability = 'available') as available_calls,
               count(*) filter (where availability = 'unavailable') as unavailable_calls,
               count(*) filter (where availability = 'excluded') as availability_excluded_calls,
               count(*) filter (where availability is null) as availability_unknown_calls,
               count(*) filter (where refused is true) as refused_calls,
               count(*) filter (where refused is false) as not_refused_calls,
               count(*) filter (where refused is null and availability in ('unavailable', 'excluded')) as refusal_not_assessable_calls,
               count(*) filter (where refused is null and (availability is null or availability = 'available')) as refusal_unknown_calls
        from calls
        where ts >= %s and ts < %s{where}
        group by {group}
        order by cost_usd desc, calls desc
        limit 500
    """
    with db.pool().connection() as con:
        rows = con.execute(sql, (start, end, *params)).fetchall()
    items = []
    for row in rows:
        offset = 1 if group_by == "model" else 0
        item = {
            "key": row[0],
            "calls": row[1 + offset],
            "cost_usd": float(row[2 + offset]),
            "input_tokens": int(row[3 + offset]),
            "output_tokens": int(row[4 + offset]),
            "cache_read_tokens": int(row[5 + offset]),
            "reasoning_tokens": int(row[6 + offset]),
            "avg_latency_ms": int(row[7 + offset]) if row[7 + offset] is not None else None,
            "p95_latency_ms": round(row[8 + offset]) if row[8 + offset] is not None else None,
            "error_rate": float(row[9 + offset]) if row[9 + offset] is not None else 0.0,
            "unpriced_calls": row[10 + offset],
            "reported_calls": row[11 + offset],
        }
        reliability_names = (
            "available_calls", "unavailable_calls", "availability_excluded_calls",
            "availability_unknown_calls", "refused_calls", "not_refused_calls",
            "refusal_not_assessable_calls", "refusal_unknown_calls",
        )
        item.update({name: int(row[12 + offset + index]) for index, name in enumerate(reliability_names)})
        availability_denominator = item["available_calls"] + item["unavailable_calls"]
        refusal_denominator = item["refused_calls"] + item["not_refused_calls"]
        item["availability_denominator"] = availability_denominator
        item["refusal_denominator"] = refusal_denominator
        item["availability_rate"] = item["available_calls"] / availability_denominator if availability_denominator else None
        item["refusal_rate"] = item["refused_calls"] / refusal_denominator if refusal_denominator else None
        item["availability_coverage"] = "partial" if item["availability_unknown_calls"] else "complete"
        item["refusal_coverage"] = "partial" if item["refusal_unknown_calls"] else "complete"
        if group_by == "model":
            item["provider"] = row[1]
        items.append(item)
    if group_by in ("day", "hour"):
        items.sort(key=lambda item: item["key"])
    return {"items": items}


@router.get("/v1/usage/timeseries")
def timeseries(
    group_by: str = Query("model"),
    bucket: str = Query("day"),
    from_: str | None = Query(None, alias="from"),
    to: str | None = None,
    top: int = Query(8, ge=1, le=25),
    environment: list[str] | None = Query(None),
    include_untagged: bool | None = None,
    func_: str | None = Query(None, alias="func"),
):
    if group_by not in ("func", "route", "model"):
        raise HTTPException(400, "group_by must be func, route, or model")
    step = _BUCKETS.get(bucket)
    if step is None:
        raise HTTPException(400, "bucket must be hour or day")
    start, end = _window(from_, to, default_days=1 if bucket == "hour" else 7)
    where, params = _filters(environment, include_untagged, None, None)
    if func_:
        where += " and func = %s"
        params.append(func_)
    key = _GROUPS[group_by]
    label = "YYYY-MM-DD" if bucket == "day" else 'YYYY-MM-DD"T"HH24:00:00"Z"'
    sql = f"""
        select to_char(date_trunc(%s, ts at time zone 'UTC'), %s) as bucket,
               {key} as key,
               coalesce(sum(cost_usd), 0) as cost_usd
        from calls
        where ts >= %s and ts < %s{where}
        group by 1, 2
    """
    with db.pool().connection() as con:
        rows = con.execute(sql, (bucket, label, start, end, *params)).fetchall()

    fmt = "%Y-%m-%d" if bucket == "day" else "%Y-%m-%dT%H:00:00Z"
    cursor = start.replace(minute=0, second=0, microsecond=0)
    if bucket == "day":
        cursor = cursor.replace(hour=0)
    buckets: list[str] = []
    while cursor < end:
        buckets.append(cursor.strftime(fmt))
        cursor += step
    index = {name: position for position, name in enumerate(buckets)}

    totals: dict[str, float] = {}
    points: dict[str, dict[str, float]] = {}
    for bucket_name, key_name, cost in rows:
        if bucket_name not in index:
            continue
        cost = float(cost)
        totals[key_name] = totals.get(key_name, 0.0) + cost
        points.setdefault(key_name, {})[bucket_name] = cost
    ranked = sorted(totals, key=totals.get, reverse=True)
    series = [
        {
            "key": name,
            "values": [points[name].get(bucket_name, 0.0) for bucket_name in buckets],
        }
        for name in ranked[:top]
    ]
    if len(ranked) > top:
        other = [0.0] * len(buckets)
        for name in ranked[top:]:
            for bucket_name, cost in points[name].items():
                other[index[bucket_name]] += cost
        series.append({"key": "other", "values": other})
    return {"buckets": buckets, "series": series}


@router.get("/v1/environments")
def environments(
    from_: str | None = Query(None, alias="from"),
    to: str | None = None,
):
    start, end = _window(from_, to)
    sql = """
        select environment, count(*)
        from calls
        where ts >= %s and ts < %s
        group by environment
        order by environment nulls last
    """
    with db.pool().connection() as con:
        rows = con.execute(sql, (start, end)).fetchall()
    return {
        "items": [
            {"value": environment, "calls": calls}
            for environment, calls in rows
        ]
    }


@router.get("/v1/calls")
def calls(
    limit: int = Query(50, ge=1, le=500),
    func_: str | None = Query(None, alias="func"),
    route: str | None = None,
    before: str | None = None,
    environment: list[str] | None = Query(None),
    include_untagged: bool | None = None,
):
    where, params = _filters(environment, include_untagged, None, None)
    if func_:
        where += " and func = %s"
        params.append(func_)
    if route:
        where += " and route = %s"
        params.append(route)
    if before:
        try:
            cutoff = datetime.fromisoformat(before)
        except ValueError as exc:
            raise HTTPException(400, "before must be an ISO timestamp") from exc
        where += " and ts < %s"
        params.append(cutoff)
    sql = f"""
        select ts, func, module, route, provider, model, canonical_model,
               input_tokens, output_tokens, cache_read_tokens, cache_write_tokens,
               reasoning_tokens, cost_usd, null::numeric as reported_cost_usd,
               cost_usd as catalog_cost_usd, price_id as catalog_price_id,
               array[]::text[] as catalog_reasons, cost_status, latency_ms, status,
               status_code, finish_reason, finish_reason_raw, error, error_type,
               stream, session_id, trace_id, template_hash,
               tool_names, environment, sdk, sdk_version, request_id,
               http_status_code, availability, refused
        from calls
        where true{where}
        order by ts desc
        limit %s
    """
    with db.pool().connection() as con:
        rows = con.execute(sql, (*params, limit)).fetchall()
    columns = (
        "ts", "func", "module", "route", "provider", "model", "canonical_model",
        "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens",
        "reasoning_tokens", "cost_usd", "reported_cost_usd", "catalog_cost_usd",
        "catalog_price_id", "catalog_reasons", "cost_status", "latency_ms", "status",
        "status_code", "finish_reason", "finish_reason_raw", "error", "error_type",
        "stream", "session_id", "trace_id", "template_hash",
        "tool_names", "environment", "sdk", "sdk_version", "request_id",
        "http_status_code", "availability", "refused",
    )
    items = []
    for row in rows:
        item = dict(zip(columns, row))
        item["ts"] = item["ts"].isoformat()
        for key in ("cost_usd", "reported_cost_usd", "catalog_cost_usd"):
            if item[key] is not None:
                item[key] = float(item[key])
        if item["sdk"] == "js":
            item["sdk"] = "typescript"
        if item["tool_names"] is not None:
            item["tool_names"] = list(item["tool_names"])
        items.append(item)
    return {"items": items}


@router.get("/v1/catalog")
def catalog_doc(request: Request):
    return request.app.state.catalog_doc
