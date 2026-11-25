"""Native Google model identity conventions."""

import re
from collections.abc import Mapping
from typing import Any

_VERTEX_MODEL = re.compile(
    r'projects/[^/]+/locations/[^/]+/publishers/google/models/([^/]+)'
)


def model_identity(model: Any, channel: str | None) -> tuple[Any, str | None]:
    """A Vertex resource name identifies both the model and its native channel."""
    match = _VERTEX_MODEL.fullmatch(model) if isinstance(model, str) else None
    return (match.group(1), 'google-vertex-ai') if match else (model, channel)


def rejected_without_usage(row: Mapping[str, Any], channel: str | None) -> bool:
    # Vertex request rejections are free; timeout completions can still be charged.
    # Do not transfer this policy to Developer API, batch downloads or client timeouts.
    # https://cloud.google.com/model-garden/pricing
    if channel != 'google-vertex-ai' or row.get('batch') is True:
        return False
    if row.get('endpoint') not in {'models.generate_content', 'models.generate_content.stream'}:
        return False
    if not (row.get('error') is True or row.get('status') in {'error', 'failed'}):
        return False
    status = row.get('http_status_code')
    if type(status) is not int or status not in {400, 401, 403, 404, 413, 422, 429}:
        return False
    if row.get('tool_calls') not in (None, []):
        return False
    counters = [row.get(key) for key in (
        'input_tokens', 'output_tokens', 'cache_read_tokens', 'cache_write_tokens',
        'cache_write_5m_tokens', 'cache_write_1h_tokens', 'reasoning_tokens',
        'web_search_calls', 'grounding_queries', 'grounded_prompt', 'search_queries',
        'search_units', 'url_fetch_calls',
    )]
    native = row.get('server_tool_use')
    if native is not None:
        if not isinstance(native, Mapping):
            return False
        counters.extend(native.values())
    return all(value is None or (type(value) is int and value == 0) for value in counters)
