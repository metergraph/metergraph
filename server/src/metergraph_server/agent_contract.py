"""The versioned Agent Access contract shared with the hosted product.

Vendored from metergraph.agent-access/v1. The OSS copy removes private-app
settings and debugging imports while preserving the shared schemas.
"""

from datetime import datetime, timedelta, timezone
from typing import Any

from jsonschema import Draft202012Validator

from . import agent_mcp


CONTRACT_VERSION = "metergraph.agent-access/v1"
TRACE_DEBUG_VERSION = "metergraph.trace-debug/v1"

PRIVACY_CLASSES = frozenset({"metadata", "content", "replay"})
CONTENT_FIELD_NAMES = frozenset(
    {"text", "tool_definitions", "tool_calls", "prompt", "system", "messages", "input", "output"}
)

_METADATA_FIELDS = frozenset(
    {
        "identifiers",
        "timestamps",
        "counts",
        "token totals",
        "cost",
        "latency",
        "status",
        "stable error kinds",
        "route/provider/model names",
        "route descriptions and constraints",
        "incident titles and diagnoses",
        "ingestion counts",
    }
)
SAFE_FIELDS = {
    "metadata": _METADATA_FIELDS,
    "content": _METADATA_FIELDS
    | frozenset(
        {
            "spans[].content.request.text",
            "spans[].content.response.text",
            "spans[].tool_definitions",
            "spans[].tool_calls",
        }
    ),
    "replay": _METADATA_FIELDS
    | frozenset(
        {
            "spans[].content.request.text",
            "spans[].content.response.text",
            "spans[].tool_definitions",
            "spans[].tool_calls",
            "comparison.replay.input",
            "comparison.replay.output",
            "provenance.overrides.prompt",
            "provenance.overrides.system",
        }
    ),
}

PRIVACY_DESCRIPTIONS = {
    "metadata": "Tenant-scoped operational metadata without captured request or response content.",
    "content": "Metadata plus retained request/response text, tool definitions, and tool calls.",
    "replay": "Content access plus a non-persistent call to a configured external provider.",
}

WARNING_CODES = frozenset(
    {
        "truncated",
        "spans_limited",
        "content_omitted",
        "capability_unavailable",
        "unsupported_field",
        "evidence_omitted",
    }
)
ERROR_CODES = frozenset(
    {
        "invalid_argument",
        "not_found",
        "forbidden",
        "replay_not_eligible",
        "response_too_large",
        "unsupported_capability",
        "rate_limited",
    }
)


def provenance(tenant, *, source: str = "live-database") -> dict[str, str]:
    """Return deployment and tenant identity for one response document."""
    workspace_id = getattr(tenant, "id", tenant)
    return {
        "deployment_profile": "oss",
        "workspace_id": str(workspace_id),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
    }


def window(days: int) -> dict[str, Any]:
    until = datetime.now(timezone.utc)
    return {
        "days": days,
        "since": (until - timedelta(days=days)).isoformat(),
        "until": until.isoformat(),
    }


def page(limit: int, truncated: bool, next_cursor: str | None = None) -> dict[str, Any]:
    return {"limit": limit, "truncated": truncated, "next_cursor": next_cursor}


def evidence(sources: list[str], rows: int, complete: bool) -> dict[str, Any]:
    return {"sources": sources, "rows": int(rows), "complete": complete}


def warning(code: str, message: str) -> dict[str, str]:
    if code not in WARNING_CODES:
        raise ValueError(f"unknown warning code: {code}")
    return {"code": code, "message": message}


def error_document(code: str, message: str) -> dict[str, Any]:
    if code not in ERROR_CODES:
        raise ValueError(f"unknown error code: {code}")
    return {"schema_version": CONTRACT_VERSION, "error": {"code": code, "message": message}}


def _object(properties: dict[str, Any], required: list[str] | tuple[str, ...] = ()) -> dict[str, Any]:
    result: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        result["required"] = list(required)
    return result


def _array(items: dict[str, Any]) -> dict[str, Any]:
    return {"type": "array", "items": items}


def _nullable(schema_type: str | list[str]) -> dict[str, Any]:
    return {"type": [schema_type, "null"]} if isinstance(schema_type, str) else {"type": schema_type + ["null"]}


def _nullable_schema(schema: dict[str, Any]) -> dict[str, Any]:
    return {"anyOf": [schema, {"type": "null"}]}


_PROVENANCE = _object(
    {
        "deployment_profile": {"type": "string"},
        "workspace_id": {"type": "string"},
        "generated_at": {"type": "string"},
        "source": {"type": "string"},
    },
    ("deployment_profile", "workspace_id", "generated_at", "source"),
)
_WARNING = _object({"code": {"enum": sorted(WARNING_CODES)}, "message": {"type": "string"}}, ("code", "message"))
_WINDOW = _object(
    {"days": {"type": "integer"}, "since": {"type": "string"}, "until": {"type": "string"}},
    ("days", "since", "until"),
)
_PAGE = _object(
    {"limit": {"type": "integer"}, "truncated": {"type": "boolean"}, "next_cursor": _nullable("string")},
    ("limit", "truncated", "next_cursor"),
)
_EVIDENCE = _object(
    {"sources": _array({"type": "string"}), "rows": {"type": "integer"}, "complete": {"type": "boolean"}},
    ("sources", "rows", "complete"),
)
_ERROR = _object(
    {"code": {"enum": sorted(ERROR_CODES)}, "message": {"type": "string"}},
    ("code", "message"),
)

_ENVELOPE_PROPERTIES = {
    "schema_version": {"const": CONTRACT_VERSION},
    "provenance": _PROVENANCE,
    "warnings": _array(_WARNING),
    "window": _WINDOW,
    "page": _PAGE,
    "evidence": _EVIDENCE,
    "error": _ERROR,
}

_ENVELOPE = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    **_object(_ENVELOPE_PROPERTIES, ("schema_version",)),
    "anyOf": [{"required": ["provenance"]}, {"required": ["error"]}],
}

_WORKSPACE = _object(
    {
        "id": {"type": "string"},
        "slug": {"type": "string"},
        "name": {"type": "string"},
        "created_at": _nullable("string"),
    },
    ("id", "slug", "name", "created_at"),
)
_CAPABILITY = _object(
    {
        "available": {"type": "boolean"},
        "content": {"type": "boolean"},
        "mutates": {"const": False},
        "privacy_class": {"enum": sorted(PRIVACY_CLASSES)},
        "required_scope": {"enum": ["agent:read", "agent:replay"]},
        "schema": {"type": "string"},
        "external_calls": {"type": "boolean"},
    },
    ("available", "content", "mutates", "privacy_class", "required_scope", "schema", "external_calls"),
)
_PRIVACY_CLASS = _object(
    {"description": {"type": "string"}, "fields": _array({"type": "string"})},
    ("description", "fields"),
)

_TRACE_SPAN = _object(
    {
        "span_id": {"type": "string"},
        "parent_span_id": _nullable("string"),
        "timestamp": {"type": "string"},
        "route": _nullable("string"),
        "func": _nullable("string"),
        "module": _nullable("string"),
        "provider": _nullable("string"),
        "model": _nullable("string"),
        "input_tokens": _nullable("integer"),
        "output_tokens": _nullable("integer"),
        "cache_read_tokens": _nullable("integer"),
        "cache_write_tokens": _nullable("integer"),
        "reasoning_tokens": _nullable("integer"),
        "cost_usd": _nullable(["number", "integer"]),
        "latency_ms": _nullable(["number", "integer"]),
        "ttft_ms": _nullable(["number", "integer"]),
        "status": _nullable("string"),
        "error": {"type": "boolean"},
        "error_type": _nullable("string"),
        "stream": _nullable("boolean"),
        "endpoint": _nullable("string"),
        "request_id": _nullable("string"),
        "content": _object(
            {
                "request": _object(
                    {"state": {"type": "string"}, "text": _nullable("string"), "response_limited": {"type": "boolean"}},
                    ("state", "text"),
                ),
                "response": _object(
                    {"state": {"type": "string"}, "text": _nullable("string"), "response_limited": {"type": "boolean"}},
                    ("state", "text"),
                ),
            },
            ("request", "response"),
        ),
        "tool_definitions": _array({}),
        "tool_calls": _array({}),
        "replay_eligible": {"type": "boolean"},
    },
    (
        "span_id",
        "parent_span_id",
        "timestamp",
        "route",
        "func",
        "module",
        "provider",
        "model",
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "reasoning_tokens",
        "cost_usd",
        "latency_ms",
        "ttft_ms",
        "status",
        "error",
        "error_type",
        "stream",
        "endpoint",
        "request_id",
        "content",
        "tool_definitions",
        "tool_calls",
        "replay_eligible",
    ),
)

_REPLAY_PROVENANCE = _object(
    {
        "source_trace_id": {"type": "string"},
        "source_span_id": {"type": "string"},
        "replayed_at": {"type": "string"},
        "overrides": _object(
            {
                "model": {"type": "string"},
                "prompt": {"type": "string"},
                "system": {"type": "string"},
                "temperature": {"type": "number"},
                "top_p": {"type": "number"},
                "max_output_tokens": {"type": "integer"},
            }
        ),
        "writes_production": {"const": False},
    },
    ("source_trace_id", "source_span_id", "replayed_at", "overrides", "writes_production"),
)
_REPLAY_ERROR = _object(
    {"code": {"type": "string"}, "message": {"type": "string"}},
    ("code", "message"),
)
_REPLAY_PART = _object(
    {
        "provider": _nullable("string"),
        "model": _nullable("string"),
        "input": _object(
            {
                "state": {"type": "string"},
                "text": _nullable("string"),
                "system": _nullable("string"),
                "messages": _array({}),
            }
        ),
        "output": _object(
            {"state": {"type": "string"}, "text": _nullable("string")}, ("state", "text")
        ),
        "tool_calls": _array({}),
        "error_type": _nullable("string"),
        "latency_ms": _nullable(["number", "integer"]),
        "cost_usd": _nullable(["number", "integer"]),
        "usage": _object(
            {"input_tokens": {"type": "integer"}, "output_tokens": {"type": "integer"}},
            ("input_tokens", "output_tokens"),
        ),
    },
    ("provider", "model", "input", "output", "tool_calls", "error_type", "latency_ms", "cost_usd"),
)

_REPORT_COVERAGE = _object(
    {
        "evaluated": {"type": "integer"},
        "published": {"type": "integer"},
        "omitted": {"type": "integer"},
    },
    ("evaluated", "published", "omitted"),
)
_REPORT_ARTIFACT = _object(
    {
        "sha256": {"type": "string"},
        "bytes": {"type": "integer"},
        "download": {"type": "string"},
    },
    ("sha256", "bytes"),
)
_REPORT_SUMMARY_PROPERTIES = {
    "analysis_id": {"type": "string"},
    "title": {"type": "string"},
    "schema_version": {"type": "integer", "minimum": 11, "maximum": 15},
    "pipeline_name": _nullable("string"),
    "generated_at": _nullable("string"),
    "imported_at": _nullable("string"),
    "import_source": _nullable("string"),
    "environment": _nullable("string"),
    "outcome": {"type": "string"},
    "workloads": _object(
        {
            "covered": {"type": "integer"},
            "partial": {"type": "integer"},
            "failed": {"type": "integer"},
            "ineligible": _nullable("integer"),
        },
        ("covered", "partial", "failed", "ineligible"),
    ),
    "opportunities": {"type": "integer"},
    "evidence_coverage": _nullable_schema(_REPORT_COVERAGE),
    "artifact": _REPORT_ARTIFACT,
    "warnings": _array(_WARNING),
}
_REPORT_SUMMARY = _object(
    _REPORT_SUMMARY_PROPERTIES,
    tuple(_REPORT_SUMMARY_PROPERTIES),
)
_REPORT_OPPORTUNITY = _object(
    {
        "kind": {"type": "string"},
        "candidate_ids": _array({"type": "string"}),
        "recommendation_status": _nullable("string"),
        "savings_usd": _nullable(["number", "integer"]),
        "quality_vs_reference": _nullable(["number", "integer"]),
        "safe_switch_rate": _nullable(["number", "integer"]),
        "cost_ratio": _nullable(["number", "integer"]),
        "quality_grade_counts": {"type": "object"},
    },
    ("kind", "candidate_ids", "recommendation_status"),
)
_REPORT_READINESS = _object(
    {
        "verdict": _nullable("string"),
        "codes": _array({"type": "string"}),
        "counts": {"type": "object"},
    }
)
_REPORT_WORKLOAD = _object(
    {
        "workload_id": {"type": "string"},
        "workload_name": {"type": "string"},
        "sample_count": _nullable("integer"),
        "state": {"enum": ["covered", "partial", "failed", "ineligible"]},
        "reason": _nullable("string"),
        "reference_readiness": _nullable_schema(_REPORT_READINESS),
        "warnings": _array({}),
        "opportunities": _array(_REPORT_OPPORTUNITY),
        "evidence_coverage": _nullable_schema(_REPORT_COVERAGE),
    },
    ("workload_id", "workload_name", "sample_count", "state", "warnings", "opportunities", "evidence_coverage"),
)
_TEXT_PAGE_VALUE = _object(
    {"text": {"type": "string"}, "truncated": {"type": "boolean"}},
    ("text", "truncated"),
)
_REPORT_EVIDENCE_CASE = _object(
    {
        "case_id": {"type": "string"},
        "question": _TEXT_PAGE_VALUE,
        "reference_model": _nullable("string"),
        "candidate_model": _nullable("string"),
        "reference_response": _TEXT_PAGE_VALUE,
        "candidate_response": _TEXT_PAGE_VALUE,
        "grade": _nullable("string"),
    },
    ("case_id", "question", "reference_model", "candidate_model", "reference_response", "candidate_response", "grade"),
)


SCHEMAS: dict[str, dict[str, Any]] = {
    "agent-access/envelope": _ENVELOPE,
    "agent-access/error": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        **_object({"schema_version": {"const": CONTRACT_VERSION}, "error": _ERROR}, ("schema_version", "error")),
    },
    "agent-access/workspace-context": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        **_object(
            {
                **_ENVELOPE_PROPERTIES,
                "workspace": _WORKSPACE,
                "retention": _object({"metadata_days": _nullable("integer")}, ("metadata_days",)),
                "content": _object({"captured": {"type": "boolean"}, "included": {"type": "boolean"}}, ("captured", "included")),
                "access": _object({"scopes": _array({"type": "string"})}, ("scopes",)),
            },
            ("schema_version", "provenance", "workspace", "retention", "content", "access"),
        ),
    },
    "agent-access/capabilities": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        **_object(
            {
                **_ENVELOPE_PROPERTIES,
                "deployment_profile": {"type": "string"},
                "deployment_capabilities": _array({"type": "string"}),
                "agent": {"type": "object", "additionalProperties": _CAPABILITY},
                "privacy_classes": {"type": "object", "additionalProperties": _PRIVACY_CLASS},
                "contracts": _object(
                    {"agent_access": {"const": CONTRACT_VERSION}, "trace_debug": {"const": TRACE_DEBUG_VERSION}},
                    ("agent_access", "trace_debug"),
                ),
                "bounds": _object(
                    {"max_days": {"type": "integer"}, "max_rows": {"type": "integer"}, "max_response_bytes": {"type": "integer"}, "content_included_by_default": {"type": "boolean"}},
                    ("max_days", "max_rows", "max_response_bytes", "content_included_by_default"),
                ),
            },
            ("schema_version", "provenance", "deployment_profile", "deployment_capabilities", "agent", "privacy_classes", "contracts", "bounds"),
        ),
    },
    "agent-access/routes": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        **_object(
            {
                **_ENVELOPE_PROPERTIES,
                "routes": _array(
                    _object(
                        {
                            "route": {"type": "string"},
                            "description": _nullable("string"),
                            "constraints": {"type": "object"},
                            "evaluation_contract": {},
                            "evaluation_contract_version": _nullable(["integer", "string"]),
                            "evaluation_contract_hash": _nullable("string"),
                            "updated_at": {"type": "string"},
                            "calls": {"type": "integer"},
                            "replay_eligible_calls": {"type": "integer"},
                        },
                        ("route", "description", "constraints", "evaluation_contract", "evaluation_contract_version", "evaluation_contract_hash", "updated_at", "calls", "replay_eligible_calls"),
                    )
                ),
            },
            ("schema_version", "provenance", "routes"),
        ),
    },
    "agent-access/usage": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        **_object(
            {
                **_ENVELOPE_PROPERTIES,
                "days": {"type": "integer"},
                "content_included": {"const": False},
                "truncated": {"type": "boolean"},
                "items": _array(
                    _object(
                        {
                            "date": {"type": "string"},
                            "route": {"type": "string"},
                            "calls": {"type": "integer"},
                            "cost_usd": {"type": "number"},
                            "partial_cost_calls": {"type": "integer"},
                            "input_tokens": {"type": "integer"},
                            "output_tokens": {"type": "integer"},
                            "avg_latency_ms": _nullable("integer"),
                            "p95_latency_ms": _nullable("integer"),
                            "error_calls": {"type": "integer"},
                        },
                        ("date", "route", "calls", "cost_usd", "input_tokens", "output_tokens", "avg_latency_ms", "p95_latency_ms", "error_calls"),
                    )
                ),
            },
            ("schema_version", "provenance", "window", "evidence", "warnings", "days", "content_included", "truncated", "items"),
        ),
    },
    "agent-access/ingestion-health": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        **_object(
            {
                **_ENVELOPE_PROPERTIES,
                "days": {"type": "integer"},
                "content_included": {"const": False},
                "pending_batches": {"type": "integer"},
                "processed_batches": {"type": "integer"},
                "failed_batches": {"type": "integer"},
                "permanent_failures": {"type": "integer"},
                "recoverable_failures": {"type": "integer"},
                "unclassified_failures": {"type": "integer"},
                "last_received_at": _nullable("string"),
                "last_processed_at": _nullable("string"),
            },
            ("schema_version", "provenance", "window", "evidence", "warnings", "days", "content_included", "pending_batches", "processed_batches", "failed_batches", "permanent_failures", "recoverable_failures", "unclassified_failures", "last_received_at", "last_processed_at"),
        ),
    },
    "agent-access/incidents": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        **_object(
            {
                **_ENVELOPE_PROPERTIES,
                "content_included": {"const": False},
                "open": {"type": "integer"},
                "critical": {"type": "integer"},
                "truncated": {"type": "boolean"},
                "incidents": _array(
                    _object(
                        {
                            "id": {"type": "string"},
                            "detector_key": _nullable("string"),
                            "route": _nullable("string"),
                            "environment": _nullable("string"),
                            "severity": _nullable("string"),
                            "status": _nullable("string"),
                            "generation": _nullable("integer"),
                            "title": _nullable("string"),
                            "diagnosis": _nullable(["string", "object"]),
                            "window_start": _nullable("string"),
                            "window_end": _nullable("string"),
                            "first_seen_at": _nullable("string"),
                            "last_seen_at": _nullable("string"),
                            "acknowledged_at": _nullable("string"),
                            "resolved_at": _nullable("string"),
                            "occurrences": {"type": "integer"},
                        },
                        ("id", "detector_key", "route", "environment", "severity", "status", "generation", "title", "diagnosis", "window_start", "window_end", "first_seen_at", "last_seen_at", "acknowledged_at", "resolved_at", "occurrences"),
                    )
                ),
            },
            ("schema_version", "provenance", "page", "evidence", "warnings", "content_included", "open", "critical", "truncated", "incidents"),
        ),
    },
    "agent-access/trace-metadata": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        **_object(
            {
                **_ENVELOPE_PROPERTIES,
                "content_included": {"const": False},
                "truncated": {"type": "boolean"},
                "next_cursor": _nullable("string"),
                "traces": _array(
                    _object(
                        {
                            "id": {"type": "string"},
                            "trace_id": {"type": "string"},
                            "trace_name": {"type": "string"},
                            "started_at": {"type": "string"},
                            "last_span_at": {"type": "string"},
                            "span_count": {"type": "integer"},
                            "input_tokens": {"type": "integer"},
                            "output_tokens": {"type": "integer"},
                            "cache_read_tokens": {"type": "integer"},
                            "cache_write_tokens": {"type": "integer"},
                            "cost_usd": _nullable(["number", "integer"]),
                            "partial_cost_calls": {"type": "integer"},
                            "status": {"type": "string"},
                            "routes": _array({"type": "string"}),
                            "providers": _array({"type": "string"}),
                            "models": _array({"type": "string"}),
                        },
                        ("id", "trace_id", "trace_name", "started_at", "last_span_at", "span_count", "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "cost_usd", "status", "routes", "providers", "models"),
                    )
                ),
            },
            ("schema_version", "provenance", "window", "page", "evidence", "warnings", "content_included", "truncated", "next_cursor", "traces"),
        ),
    },
    "trace-debug/trace": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        **_object(
            {
                "schema_version": {"const": TRACE_DEBUG_VERSION},
                "trace_id": {"type": "string"},
                "trace_name": {"type": "string"},
                "started_at": {"type": "string"},
                "last_span_at": {"type": "string"},
                "span_count": {"type": "integer"},
                "status": {"enum": ["success", "error"]},
                "error_context": _object({"kinds": _array({"type": "string"})}, ("kinds",)),
                "root_cause": _object(
                    {
                        "classifications": _array(
                            _object(
                                {
                                    "analysis_run_id": {"type": "string"},
                                    "classification": _object(
                                        {
                                            "status": _nullable("string"),
                                            "workload_id": _nullable("string"),
                                            "confidence": _nullable(["number", "integer"]),
                                            "method": _nullable("string"),
                                        },
                                        ("status", "workload_id", "confidence", "method"),
                                    ),
                                    "analysis_run": _object(
                                        {
                                            "status": _nullable("string"),
                                            "analysis_id": _nullable("string"),
                                            "report_id": _nullable("string"),
                                            "error_code": _nullable("string"),
                                            "progress": _nullable("object"),
                                            "created_at": _nullable("string"),
                                            "finished_at": _nullable("string"),
                                        },
                                        ("status", "analysis_id", "report_id", "error_code", "progress", "created_at", "finished_at"),
                                    ),
                                },
                                ("analysis_run_id", "classification", "analysis_run"),
                            )
                        )
                    },
                    ("classifications",),
                ),
                "replay": _object(
                    {"eligible": {"type": "boolean"}, "allowlisted_overrides": _array({"type": "string"}), "writes_production": {"const": False}},
                    ("eligible", "allowlisted_overrides", "writes_production"),
                ),
                "spans": _array(_TRACE_SPAN),
                "spans_limited": {"const": True},
            },
            ("schema_version", "trace_id", "trace_name", "started_at", "last_span_at", "span_count", "status", "error_context", "root_cause", "replay", "spans"),
        ),
    },
    "trace-debug/workload": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        **_object(
            {
                "schema_version": {"const": TRACE_DEBUG_VERSION},
                "workload_id": {"type": "string"},
                "match": {"const": "route_or_analysis_workload"},
                "traces": _array(
                    _object(
                        {
                            "trace_id": {"type": "string"},
                            "trace_name": {"type": "string"},
                            "started_at": {"type": "string"},
                            "last_span_at": {"type": "string"},
                            "span_count": {"type": "integer"},
                            "status": {"enum": ["success", "error"]},
                        },
                        ("trace_id", "trace_name", "started_at", "last_span_at", "span_count", "status"),
                    )
                ),
            },
            ("schema_version", "workload_id", "match", "traces"),
        ),
    },
    "trace-debug/replay": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        **_object(
            {
                "schema_version": {"const": TRACE_DEBUG_VERSION},
                "status": {"enum": ["succeeded", "provider_error", "timeout"]},
                "provenance": _REPLAY_PROVENANCE,
                "comparison": _object(
                    {"original": _REPLAY_PART, "replay": _REPLAY_PART}, ("original", "replay")
                ),
                "error": _REPLAY_ERROR,
            },
            ("schema_version", "status", "provenance"),
        ),
        "allOf": [
            {"if": {"properties": {"status": {"const": "succeeded"}}}, "then": {"required": ["comparison"]}},
            {
                "if": {"properties": {"status": {"enum": ["provider_error", "timeout"]}}},
                "then": {"required": ["error"]},
            },
        ],
    },
    "agent-access/reports": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        **_object(
            {
                **_ENVELOPE_PROPERTIES,
                "content_included": {"const": False},
                "reports": _array(_REPORT_SUMMARY),
            },
            ("schema_version", "provenance", "warnings", "page", "content_included", "reports"),
        ),
    },
    "agent-access/report": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        **_object(
            {
                **_ENVELOPE_PROPERTIES,
                **_REPORT_SUMMARY_PROPERTIES,
                "content_included": {"const": False},
                "settings": _nullable("object"),
                "spend_projection": _nullable("object"),
                "analysis_generation_cost": _nullable("object"),
                "workloads": _array(_REPORT_WORKLOAD),
            },
            ("schema_version", "provenance", "warnings", "content_included", *tuple(_REPORT_SUMMARY_PROPERTIES), "settings", "spend_projection", "analysis_generation_cost", "workloads"),
        ),
    },
    "agent-access/report-evidence": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        **_object(
            {
                **_ENVELOPE_PROPERTIES,
                "content_included": {"const": True},
                "workload_id": {"type": "string"},
                "evidence": _array(_REPORT_EVIDENCE_CASE),
                "evidence_coverage": _nullable_schema(_REPORT_COVERAGE),
                "next_cursor": _nullable("string"),
            },
            ("schema_version", "provenance", "warnings", "page", "content_included", "workload_id", "evidence", "evidence_coverage", "next_cursor"),
        ),
    },
}


TOOL_CONTRACT = {
    "metergraph_get_workspace_context": {
        "capability": "workspace_context",
        "privacy_class": "metadata",
        "required_scope": "agent:read",
        "schema": "agent-access/workspace-context",
        "mutates": False,
        "external_calls": False,
    },
    "metergraph_get_capabilities": {
        "capability": "capability_discovery",
        "privacy_class": "metadata",
        "required_scope": "agent:read",
        "schema": "agent-access/capabilities",
        "mutates": False,
        "external_calls": False,
    },
    "metergraph_list_routes": {
        "capability": "routes",
        "privacy_class": "metadata",
        "required_scope": "agent:read",
        "schema": "agent-access/routes",
        "mutates": False,
        "external_calls": False,
    },
    "metergraph_get_usage": {
        "capability": "usage",
        "privacy_class": "metadata",
        "required_scope": "agent:read",
        "schema": "agent-access/usage",
        "mutates": False,
        "external_calls": False,
    },
    "metergraph_get_ingestion_health": {
        "capability": "ingestion_health",
        "privacy_class": "metadata",
        "required_scope": "agent:read",
        "schema": "agent-access/ingestion-health",
        "mutates": False,
        "external_calls": False,
    },
    "metergraph_list_incidents": {
        "capability": "incidents",
        "privacy_class": "metadata",
        "required_scope": "agent:read",
        "schema": "agent-access/incidents",
        "mutates": False,
        "external_calls": False,
    },
    "metergraph_query_traces": {
        "capability": "trace_metadata",
        "privacy_class": "metadata",
        "required_scope": "agent:read",
        "schema": "agent-access/trace-metadata",
        "mutates": False,
        "external_calls": False,
    },
    "metergraph_list_reports": {
        "capability": "reports",
        "privacy_class": "metadata",
        "required_scope": "agent:read",
        "schema": "agent-access/reports",
        "mutates": False,
        "external_calls": False,
    },
    "metergraph_get_report": {
        "capability": "report_detail",
        "privacy_class": "metadata",
        "required_scope": "agent:read",
        "schema": "agent-access/report",
        "mutates": False,
        "external_calls": False,
    },
    "metergraph_get_report_evidence": {
        "capability": "report_evidence",
        "privacy_class": "content",
        "required_scope": "agent:read",
        "schema": "agent-access/report-evidence",
        "mutates": False,
        "external_calls": False,
    },
    "metergraph_get_trace": {
        "capability": "trace_content",
        "privacy_class": "content",
        "required_scope": "agent:read",
        "schema": "trace-debug/trace",
        "mutates": False,
        "external_calls": False,
    },
    "metergraph_replay_trace": {
        "capability": "trace_replay",
        "privacy_class": "replay",
        "required_scope": "agent:replay",
        "schema": "trace-debug/replay",
        "mutates": False,
        "external_calls": True,
    },
}


def validate(schema_id: str, document: Any) -> None:
    """Validate a document against one of the versioned contract schemas."""
    Draft202012Validator(SCHEMAS[schema_id]).validate(document)


if set(TOOL_CONTRACT) != {tool["name"] for tool in agent_mcp.TOOLS}:
    raise RuntimeError("TOOL_CONTRACT must cover exactly the Agent Access tools")
