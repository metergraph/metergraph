"""Anthropic failure billing evidence, independent of model token rates."""
from collections.abc import Mapping
from typing import Any


# Provider HTTP rejections are free; client disconnects/timeouts can still be
# charged. Never infer a free call from an exception name or absent counters.
# https://support.claude.com/en/articles/8977456-how-do-i-pay-for-my-claude-api-usage

def rejected_without_usage(row: Mapping[str, Any]) -> bool:
    # A rejected result download says nothing about the already-submitted job.
    if row.get('batch') is True or row.get('endpoint') not in {'messages', 'messages.create', 'messages.stream'}:
        return False
    status = row.get('http_status_code')
    if not (row.get('error') is True or row.get('status') in {'error', 'failed'}):
        return False
    if type(status) is not int or not 400 <= status <= 599:
        return False
    counters = [row.get(key) for key in (
        'input_tokens', 'output_tokens', 'cache_read_tokens', 'cache_write_tokens',
        'cache_write_5m_tokens', 'cache_write_1h_tokens', 'reasoning_tokens',
        'web_search_calls', 'grounding_queries', 'search_queries', 'search_units',
    )]
    native = row.get('server_tool_use')
    if native is not None:
        if not isinstance(native, Mapping):
            return False
        counters.extend(native.values())
    return all(value is None or (type(value) is int and value == 0) for value in counters)
