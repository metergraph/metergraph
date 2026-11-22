"""Cost a provider response object as the caller received it.

Each reader copies only the fields a price depends on: model, usage counts,
service tier and creation time. Prompt, output and every other field are never
read, so nothing downstream sees content the response carried.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Callable, Mapping

from .catalog import CostResult, ModelUsage, _coerce_datetime, _supports_flex
from .loader import LoadedCatalog, load_catalog

_STANDARD_TIERS = frozenset({"default", "standard", "standard_only", "auto", "on_demand"})
_BATCH_TIER = "batch"


@dataclass(frozen=True, slots=True)
class _Facts:
    model: str | None
    usage: Mapping[str, int | None] = field(default_factory=dict)
    # The input total already counts cached tokens, so it is restated in the
    # convention of the price row that bills it.
    input_includes_cache: bool = False
    service_tier: str | None = None
    at: datetime | None = None


def _object(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _text(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _count(value: Any) -> int | None:
    """A token count, or None for anything that is not a whole number."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _sum(*values: Any) -> int | None:
    counts = [_count(value) for value in values]
    if counts[0] is None:
        return None
    return sum(count or 0 for count in counts)


def _first(mapping: Mapping[str, Any], *keys: str) -> Any:
    """The first present key: SDK objects dump snake_case, REST JSON is
    camelCase."""
    for key in keys:
        if mapping.get(key) is not None:
            return mapping[key]
    return None


def _epoch(value: Any) -> datetime | None:
    seconds = _count(value)
    if seconds is None:
        return None
    try:
        return datetime.fromtimestamp(seconds, timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


def _iso(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    # Python 3.10 parses at most six fractional digits; Vertex sends nine.
    head, dot, rest = text.partition(".")
    if dot:
        digits = len(rest) - len(rest.lstrip("0123456789"))
        text = f"{head}.{rest[:min(digits, 6)]}{rest[digits:]}"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _anthropic_messages(response: Mapping[str, Any]) -> _Facts:
    usage = _object(response.get("usage"))
    creation = _object(usage.get("cache_creation"))
    return _Facts(
        model=_text(response.get("model")),
        usage={
            "input_tokens": _count(usage.get("input_tokens")),
            "output_tokens": _count(usage.get("output_tokens")),
            "cache_read_tokens": _count(usage.get("cache_read_input_tokens")),
            "cache_write_tokens": _count(usage.get("cache_creation_input_tokens")),
            "cache_write_5m_tokens": _count(creation.get("ephemeral_5m_input_tokens")),
            "cache_write_1h_tokens": _count(creation.get("ephemeral_1h_input_tokens")),
        },
        service_tier=_text(usage.get("service_tier")),
    )


def _openai_responses(response: Mapping[str, Any]) -> _Facts:
    usage = _object(response.get("usage"))
    details = _object(usage.get("input_tokens_details"))
    return _Facts(
        model=_text(response.get("model")),
        usage={
            "input_tokens": _count(usage.get("input_tokens")),
            "output_tokens": _count(usage.get("output_tokens")),
            "cache_read_tokens": _count(details.get("cached_tokens")),
            # Gateways such as OpenRouter report cache writes beside reads.
            "cache_write_tokens": _count(details.get("cache_write_tokens")),
        },
        input_includes_cache=True,
        service_tier=_text(response.get("service_tier")),
        at=_epoch(response.get("created_at")),
    )


def _openai_chat_completions(response: Mapping[str, Any]) -> _Facts:
    usage = _object(response.get("usage"))
    details = _object(usage.get("prompt_tokens_details"))
    return _Facts(
        model=_text(response.get("model")),
        usage={
            "input_tokens": _count(usage.get("prompt_tokens")),
            "output_tokens": _count(usage.get("completion_tokens")),
            "cache_read_tokens": _count(details.get("cached_tokens")),
            "cache_write_tokens": _count(details.get("cache_write_tokens")),
        },
        # Gateways relaying another publisher's model keep this shape, with
        # cached tokens inside the prompt total.
        input_includes_cache=True,
        service_tier=_text(response.get("service_tier")),
        at=_epoch(response.get("created")),
    )


def _gemini(response: Mapping[str, Any]) -> _Facts:
    usage = _object(_first(response, "usage_metadata", "usageMetadata"))
    traffic = _text(_first(usage, "traffic_type", "trafficType"))
    tier = {"ON_DEMAND_FLEX": "flex", "ON_DEMAND": "standard", "ON_DEMAND_PRIORITY": "priority"}.get(traffic, traffic)
    return _Facts(
        model=_text(_first(response, "model_version", "modelVersion")),
        usage={
            # Tool-use prompt tokens bill as input; thinking tokens bill as
            # output but are reported outside the candidates count.
            "input_tokens": _sum(
                _first(usage, "prompt_token_count", "promptTokenCount"),
                _first(usage, "tool_use_prompt_token_count", "toolUsePromptTokenCount"),
            ),
            "output_tokens": _sum(
                _first(usage, "candidates_token_count", "candidatesTokenCount"),
                _first(usage, "thoughts_token_count", "thoughtsTokenCount"),
            ),
            "cache_read_tokens": _count(
                _first(usage, "cached_content_token_count", "cachedContentTokenCount")
            ),
        },
        input_includes_cache=True,
        service_tier=tier,
        at=_iso(_first(response, "create_time", "createTime")),
    )


def _bedrock_converse(response: Mapping[str, Any]) -> _Facts:
    usage = _object(response.get("usage"))
    return _Facts(
        # Converse responses do not name the model; the caller supplies it.
        model=None,
        usage={
            "input_tokens": _count(usage.get("inputTokens")),
            "output_tokens": _count(usage.get("outputTokens")),
            "cache_read_tokens": _count(usage.get("cacheReadInputTokens")),
            "cache_write_tokens": _count(usage.get("cacheWriteInputTokens")),
        },
        service_tier=_text(_object(response.get("serviceTier")).get("type")),
    )


_READERS: dict[str, Callable[[Mapping[str, Any]], _Facts]] = {
    "openai.chat_completions": _openai_chat_completions,
    "openai.responses": _openai_responses,
    "anthropic.messages": _anthropic_messages,
    "gemini.generate_content": _gemini,
    "bedrock.converse": _bedrock_converse,
}

# The response formats :func:`cost_from_response` reads.
RESPONSE_FORMATS = tuple(_READERS)


def _detect(response: Mapping[str, Any]) -> str | None:
    """The format of this response, told apart by the fields each API
    always returns."""
    if _first(response, "usage_metadata", "usageMetadata") is not None:
        return "gemini.generate_content"
    usage = _object(response.get("usage"))
    if "prompt_tokens" in usage:
        return "openai.chat_completions"
    if "inputTokens" in usage:
        return "bedrock.converse"
    if response.get("type") == "message" or "cache_read_input_tokens" in usage:
        return "anthropic.messages"
    if response.get("object") == "response" or "input_tokens_details" in usage:
        return "openai.responses"
    return None


def response_format(response: Any) -> str | None:
    """The :data:`RESPONSE_FORMATS` entry ``response`` is in, or None when its
    shape is not recognized."""
    return _detect(_as_mapping(response))


def _as_mapping(response: Any) -> Mapping[str, Any]:
    if isinstance(response, Mapping):
        return response
    dump = getattr(response, "model_dump", None)
    if callable(dump):
        # OpenAI, Anthropic and Google GenAI SDK objects are pydantic models.
        return dump()
    to_dict = getattr(response, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    raise TypeError(
        "response must be a mapping or a provider SDK response object, "
        f"not {type(response).__name__}"
    )


@lru_cache(maxsize=1)
def _bundled_catalog() -> LoadedCatalog:
    return load_catalog(region="global")


def _native_input_tokens(
    usage: Mapping[str, int | None], probe: Callable[..., CostResult]
) -> int | None:
    """An input total that includes cache reads and writes, restated in the
    convention the priced row uses.

    The row records whether its input total already counts cached tokens
    (OpenAI's includes reads, Anthropic's includes neither). A response whose
    total includes them, priced on a row that does not deduct them, would
    otherwise bill every cached token twice. ``probe`` costs a usage on the
    same model and channel.
    """
    total = usage.get("input_tokens")
    cache_read = usage.get("cache_read_tokens") or 0
    cache_write = usage.get("cache_write_tokens") or 0
    if total is None or cache_read < 0 or cache_write < 0 or not (cache_read or cache_write):
        return total
    # Cached tokens exceeding the total cannot have been included in it.
    uncached = (
        total - cache_read - cache_write
        if cache_read + cache_write <= total
        else total
    )
    # A zero-input probe reports which cached counts the row deducts from input.
    reasons = probe(
        input_tokens=0, output_tokens=0, cache_read_tokens=1, cache_write_tokens=1
    ).reasons
    native = uncached
    if "cache_read_exceeds_input" in reasons:
        native += cache_read
    if "cache_write_exceeds_input" in reasons:
        native += cache_write
    return native


def cost_from_response(
    provider: str,
    response: Any,
    *,
    catalog: LoadedCatalog | None = None,
    model: str | None = None,
    at: Any = None,
    channel: str | None = None,
    region: str | None = None,
    service_tier: str | None = None,
    format: str | None = None,
) -> CostResult:
    """The cost of one provider response, from the usage it reports.

    ``response`` is the object the provider SDK returned, or its JSON as a
    mapping: OpenAI Chat Completions or Responses, Anthropic Messages, Google
    Gemini ``generateContent``, or Bedrock Converse. ``provider`` names who
    served the call and selects its direct pricing channel unless ``channel``
    is given. ``model`` overrides the response's model id and is required for
    Bedrock Converse, whose responses do not name one. ``at`` defaults to the
    response's creation time when it reports one, else now. ``catalog``
    defaults to the catalog bundled with this package. ``format``, one of
    :data:`RESPONSE_FORMATS`, names the response's format; without it the
    format is detected from the response's fields.

    A streamed call is costed from its final response or final usage chunk.
    Fees outside token usage, such as web search, are not included.
    ``service_tier`` overrides the tier the response reports, for responses
    that do not state it, such as an OpenAI batch output line. A tier other
    than standard, batch, or supported OpenAI/Google Flex turns the result
    ``partial`` with ``service_tier`` missing.
    """
    data = _as_mapping(response)
    if format is not None and format not in _READERS:
        raise ValueError(f"format must be one of {', '.join(RESPONSE_FORMATS)}")
    detected = format or _detect(data)
    if detected is None:
        raise ValueError("response carries no usage this function can read")
    facts = _READERS[detected](data)
    when = at if at is not None else facts.at or datetime.now(timezone.utc)
    model_id = _text(model) or facts.model
    tier = (_text(service_tier) or facts.service_tier or "").lower() or None
    if model_id is None:
        return CostResult(
            None,
            None,
            None,
            "unpriced",
            ("missing_model",),
            effective_at=_coerce_datetime(when),
            missing_dimensions=("model",),
        )
    loaded = catalog if catalog is not None else _bundled_catalog()
    batch = tier == _BATCH_TIER

    def cost(**usage: Any) -> CostResult:
        return loaded.calculate_cost(
            model=model_id,
            usage=ModelUsage(**usage, batch=batch, service_tier=tier),
            at=when,
            channel=channel,
            provider=provider,
            region=region,
        )

    usage = dict(facts.usage)
    if facts.input_includes_cache:
        usage["input_tokens"] = _native_input_tokens(usage, cost)
    result = cost(**usage)
    priority_priced = tier in ("priority", "fast") and "unsupported_service_tier" not in result.reasons
    if tier is None or tier in _STANDARD_TIERS or batch or (tier == "flex" and _supports_flex(result.price)) or priority_priced or result.status == "unpriced":
        return result
    return replace(
        result,
        status="partial",
        missing_dimensions=tuple(dict.fromkeys((*result.missing_dimensions, "service_tier"))),
    )
