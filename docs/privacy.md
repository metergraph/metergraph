# Privacy model

Metergraph's open-source server is **content-blind by construction**. This page
states exactly what that means.

## What the server stores

For every captured LLM call, one metadata row:

- `ts`, `latency_ms`, `ttft_ms` (streaming)
- `func`, `module`: which of *your* functions made the call (name only)
- `route`, `trace_id`, `session_id`, `tags`, `environment`, `unit_name`/`unit_count`
- `provider`, `model`, `endpoint`, `request_id`
- `input_tokens`, `output_tokens`, `cache_read_tokens`, `cache_write_tokens`, `reasoning_tokens`
- `status`, `status_code`, `finish_reason`, `finish_reason_raw`,
  `error`/`error_type`, `stream`/`batch` flags
- `http_status_code`, `availability`, `refused`
- `template_hash`: a structural fingerprint of the request *shape* (roles, ordering, placeholders). Literal text is scrubbed before hashing and is not recoverable from it.
- `tool_names`: tool **names** only
- `sdk`, `sdk_version`

No prompt text. No completion text. No tool arguments or results.

## What the server refuses to store

Whatever the SDK sends, the self-hosted server projects every row through a
strict column allowlist at ingest (`server/src/metergraph_server/ingest.py`).
`request_json`, `response_text`, tool-call arguments and results, and any
unknown field are dropped before the database is touched. There is no column
for them to land in. This is covered by a sentinel test
(`server/tests/test_api.py::test_content_never_reaches_database`).

## What the SDK sends

Current SDKs capture scrubbed request and normalized response content by
default, because paid Metergraph uses replayable content to evaluate quality
and recommend changes. Provider credentials and transport headers are always
removed.

- Against a self-hosted server, that content is discarded at ingest, as above.
- To keep content inside your process as well, set `METERGRAPH_CAPTURE_TEXT=0`,
  or pass `capture_text=False` to `metergraph.init()`, `route()`, or `trace()`.

See the [SDK documentation](https://github.com/metergraph/metergraphsdk) for
redaction hooks and size limits.

## Fail-safe defaults

- No `METERGRAPH_APP_TOKEN`: capture is disabled entirely, and the SDK never
  sends anything silently.
- `METERGRAPH_DISABLED=1`: hard off.
- The SDK is fail-open: transport errors never break or slow your LLM calls.
  Rows are dropped, not retried into your request path.
