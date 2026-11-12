# SPDX-License-Identifier: Apache-2.0
# Copyright Metergraph contributors. Canonical source: metergraph-core reliability.py.
"""Conservative classification of structured provider reliability evidence.

This module intentionally depends only on the standard library so capture
clients can vendor the same classifier without a runtime Core dependency.
"""
from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any, Literal


@dataclass(frozen=True)
class ModelReliability:
    http_status_code: int | None = None
    availability: Literal["available", "unavailable", "excluded"] | None = None
    refused: bool | None = None


def _get(value, key, default=None):
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _seq(value):
    return value if isinstance(value, (list, tuple)) else ()


def _reason(value):
    return value.lower().replace('-', '_') if isinstance(value, str) else None


def _classify_native(*, provider: str | None = None, response: Any = None,
                               error: Any = None, http_status_code: int | None = None,
                               finish_reason: str | None = None) -> ModelReliability:
    """Classify explicit native response facts; missing evidence stays unknown."""
    status = http_status_code
    if status is None:
        status = (_get(error, 'status_code') or _get(error, 'status') or _get(error, 'statusCode')
                  or _get(_get(error, 'response'), 'status_code'))
    if status is None:
        status = _get(response, 'status_code')
    if isinstance(status, bool) or not isinstance(status, int) or not 100 <= status <= 599:
        status = None
    reasons = [_reason(finish_reason), _reason(_get(response, 'stop_reason')),
               _reason(_get(response, 'finish_reason'))]
    known = False
    complete = []
    refused = False
    for choice in _seq(_get(response, 'choices')):
        reason = _reason(_get(choice, 'finish_reason'))
        reasons.append(reason)
        message = _get(choice, 'message')
        complete.append(message is not None and reason in {'stop', 'length', 'tool_calls', 'function_call', 'content_filter'}
                        and (_get(message, 'content') is not None or _get(message, 'tool_calls') is not None
                             or (isinstance(message, Mapping) and 'refusal' in message)
                             or (not isinstance(message, Mapping) and hasattr(message, 'refusal'))))
        if reason in {'stop', 'length', 'tool_calls', 'function_call', 'content_filter'}:
            known = True
        if _get(message, 'refusal'):
            refused = True
    stop = _reason(_get(response, 'stop_reason'))
    if stop in {'end_turn', 'stop_sequence', 'max_tokens', 'tool_use', 'refusal'} and _get(response, 'content') is not None:
        known = True
        complete.append(True)
    output = _get(response, 'output')
    if isinstance(output, (list, tuple)):
        for item in output:
            if _get(item, 'type') == 'refusal':
                refused = True
            for part in _seq(_get(item, 'content')):
                if _get(part, 'type') == 'refusal':
                    refused = True
        if output and _get(response, 'status') in {'completed', 'incomplete'}:
            inspected = []
            for item in output:
                kind = _get(item, 'type')
                content = _get(item, 'content')
                inspected.append((kind == 'message' and isinstance(content, (list, tuple))
                                  and bool(content) and all(_get(part, 'type') in {'output_text', 'refusal'} for part in content))
                                 or kind in {'function_call', 'reasoning'})
            complete.extend(inspected)
            known = any(inspected)
    reasons.append(_reason(_get(_get(response, 'incomplete_details'), 'reason')))
    feedback = _get(response, 'promptFeedback', _get(response, 'prompt_feedback'))
    block = _get(feedback, 'blockReason', _get(feedback, 'block_reason'))
    if _reason(block) in {'safety', 'blocklist', 'prohibited_content', 'spii', 'image_safety'}:
        refused = True
    for candidate in _seq(_get(response, 'candidates')):
        reason = _reason(_get(candidate, 'finishReason', _get(candidate, 'finish_reason')))
        reasons.append(reason)
        if reason in {'stop', 'max_tokens', 'safety', 'recitation', 'blocklist', 'prohibited_content', 'spii'}:
            known = True
            content = _get(candidate, 'content')
            complete.append(isinstance(_get(content, 'parts'), (list, tuple)) and bool(_get(content, 'parts')))
        else:
            complete.append(False)
    if any(reason in {'refusal', 'content_filter', 'safety', 'recitation', 'blocklist', 'prohibited_content', 'spii'} for reason in reasons):
        refused = True
    payload = _get(error, 'error', _get(error, 'body', error))
    payload = _get(payload, 'error', payload)
    code = _get(payload, 'code')
    if isinstance(provider, str) and provider.lower() == 'meta':
        if (status == 400 and code == 'content_policy_violation') or (
            status == 403 and code == 'user_blocked' and _get(payload, 'type') == 'permission_error'
            and 'repeated policy violations' in str(_get(payload, 'message', '')).lower()):
            refused = True
    if refused:
        return ModelReliability(status, 'available', True)
    if status == 429 or status is not None and status >= 500:
        return ModelReliability(status, 'unavailable', None)
    if status is not None and 400 <= status < 500:
        return ModelReliability(status, 'excluded', None)
    error_type = _get(error, 'name') or _get(error, 'type') or (type(error).__name__ if isinstance(error, BaseException) else None)
    if error_type in {'APITimeoutError', 'TimeoutError', 'ReadTimeout', 'ConnectTimeout'}:
        return ModelReliability(status, 'unavailable', None)
    if known or any(reason in {'stop', 'length', 'tool_calls', 'tool_use', 'max_tokens', 'end_turn'} for reason in reasons):
        return ModelReliability(status, 'available', False if complete and all(complete) else None)
    if error is not None:
        return ModelReliability(status, None, None)
    if status is not None and 200 <= status < 300:
        return ModelReliability(status, 'available', None)
    return ModelReliability(status, None, None)


def classify_model_reliability(*, provider: str | None = None, response: Any = None,
                               error: Any = None, http_status_code: int | None = None,
                               finish_reason: str | None = None,
                               stream_chunks: Any = None,
                               stream_complete: bool = False) -> ModelReliability:
    """Classify native facts, retaining explicit refusal evidence across chunks.

    Stream arguments are transient inspection inputs, never stored record fields.
    Chunk-only normal streams conservatively retain unknown refusal coverage.
    """
    result = _classify_native(provider=provider, response=response, error=error,
                              http_status_code=http_status_code, finish_reason=finish_reason)
    stream_available = False
    for chunk in _seq(stream_chunks):
        native = _classify_native(provider=provider, response=chunk)
        nested = _get(chunk, 'response')
        nested_result = _classify_native(provider=provider, response=nested)
        delta = _get(chunk, 'delta')
        chunk_reason = _reason(_get(delta, 'stop_reason'))
        explicit = (native.refused is True or nested_result.refused is True
                    or _get(chunk, 'type') in {'response.refusal.delta', 'response.refusal.done'}
                    or chunk_reason in {'refusal', 'content_filter', 'safety'}
                    or _get(_get(chunk, 'content_block'), 'type') == 'refusal'
                    or _get(_get(chunk, 'part'), 'type') == 'refusal')
        for choice in _seq(_get(chunk, 'choices')):
            if _get(_get(choice, 'delta'), 'refusal'):
                explicit = True
        if explicit:
            return ModelReliability(result.http_status_code, 'available', True)
        stream_available = stream_available or native.availability == 'available' or nested_result.availability == 'available'
        if chunk_reason in {'end_turn', 'stop_sequence', 'max_tokens', 'tool_use'}:
            stream_available = True
    if result.refused is True:
        return result
    if stream_available and result.availability is None:
        return ModelReliability(result.http_status_code, 'available', None)
    if stream_chunks is not None and not stream_complete and result.refused is False:
        return ModelReliability(result.http_status_code, result.availability, None)
    return result
