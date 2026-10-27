"""Effective-dated model catalog and deterministic token-cost enrichment."""

import difflib
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from types import MappingProxyType
from typing import Any, Mapping

_MILLION = Decimal("1000000")
_COST_QUANTUM = Decimal("0.00000001")
_PROVIDER_ALIASES = {
    "amazon-bedrock": "bedrock",
    "aws": "bedrock",
    "aws-bedrock": "bedrock",
    "gemini": "google",
    "google-genai": "google",
    "moonshot": "moonshotai",
    "perplexity": "perplexity-ai",
    "xai": "x-ai",
}

# The direct catalog channel each captured source provider was actually billed
# on. A captured/reference execution was served by the customer's own provider,
# so its price must come from that provider's direct channel -- never a gateway
# list price and never a guess. A source provider absent from this map yields no
# channel, and the caller must keep the execution explicitly unpriced.
_DIRECT_CHANNEL_BY_PROVIDER = {
    "anthropic": "anthropic-api",
    "openai": "openai-api",
    "google": "google-api",
    "vercel": "vercel-ai-gateway",
    "openrouter": "openrouter-api",
    "vertex-ai": "google-vertex-ai",
    "fireworks": "fireworks-api",
    "bedrock": "aws-bedrock",
    "deepseek": "deepseek-api",
    "perplexity-ai": "perplexity-api",
    "x-ai": "xai-api",
    "alibaba": "alibaba-api",
    # Amazon publishes no first-party inference API: its own models are sold
    # through Bedrock, so a trace naming "amazon" was billed on Bedrock.
    "amazon": "aws-bedrock",
    "meta": "meta-api",
    "minimax": "minimax-api",
    "mistral": "mistral-api",
    "moonshotai": "moonshot-api",
    "zai": "zai-api",
}


# How a call reports cached tokens is set by whoever serves the model, so the
# default is keyed on (publisher, channel) rather than the channel alone.
# OpenAI's `prompt_tokens`, Google's `promptTokenCount`, DeepSeek's and xAI's
# `prompt_tokens` all include the cached tokens they also report separately, so
# a row on these pairs must deduct them before applying the input rate. Without
# it a cached token is billed at the input rate and again at the cache-read
# rate -- several times the real cost on a cache-heavy workload, and enough to
# reorder a customer's spend ranking.
#
# A channel alone cannot decide this: google-vertex-ai serves Anthropic's
# models beside Google's, and Claude on Vertex keeps Anthropic's usage shape,
# where input_tokens already excludes cache reads. Deducting there overcharges
# in the opposite direction -- the same defect this default exists to remove.
# Anthropic and Bedrock are absent by design; a gateway channel depends on the
# provider behind it and states its own rules per row.
_INPUT_INCLUDES_CACHE_READ_PAIRS = frozenset({
    ("openai", "openai-api"),
    ("google", "google-api"),
    ("google", "google-vertex-ai"),
    ("deepseek", "deepseek-api"),
    ("xai", "xai-api"),
})


# Bedrock ids carry a model-version suffix (`anthropic.claude-sonnet-5-v1:0`)
# and may name the global inference profile (`global.`). Both are fallbacks
# tried only after an exact match fails, and only on Bedrock. Geographic
# profile prefixes (`us.`, `eu.`, ...) are never stripped: they bill on
# aws-bedrock-geo, and the bare id would price them at the lower rate.
_BEDROCK_CHANNEL = "aws-bedrock"
_BEDROCK_CHANNELS = frozenset({_BEDROCK_CHANNEL, "aws-bedrock-geo"})
_BEDROCK_VERSION_SUFFIX = re.compile(r"-v\d+(?::\d+)?$")
_BEDROCK_GLOBAL_PREFIX = "global."
_GLOBAL_REGION = "global"
# Reason for a result priced at a price that ended and was not replaced.
PRICE_ENDED = "price_ended"

# Warning codes: why a result is unpriced, when the cause is a value the
# caller sent. The codes are stable; the messages may change.
UNKNOWN_CHANNEL = "unknown_channel"
UNKNOWN_PROVIDER = "unknown_provider"
UNKNOWN_MODEL = "unknown_model"
MODEL_NOT_ON_CHANNEL = "model_not_on_channel"
MODEL_NOT_IN_EFFECT = "model_not_in_effect"
MODEL_NOT_IN_REGION = "model_not_in_region"
PRICE_NOT_IN_EFFECT = "price_not_in_effect"


def _bedrock_fallback_ids(model_key: str) -> tuple[tuple[str, bool], ...]:
    """Candidate ids for an unmatched Bedrock model, most specific first.

    Each candidate is paired with whether it dropped the ``global.`` prefix;
    such a match is only valid on the in-region ``aws-bedrock`` channel.
    """
    unversioned = _BEDROCK_VERSION_SUFFIX.sub("", model_key)
    candidates: list[tuple[str, bool]] = []
    if unversioned != model_key:
        candidates.append((unversioned, False))
    if model_key.startswith(_BEDROCK_GLOBAL_PREFIX):
        for name in dict.fromkeys((model_key, unversioned)):
            bare = name[len(_BEDROCK_GLOBAL_PREFIX):]
            if bare:
                candidates.append((bare, True))
    return tuple(candidates)


def counts_cache_read_in_input(
    publisher: Any, channel: Any, rules: Mapping[str, Any]
) -> bool:
    """Whether cache reads have to come out of billable input for this row.

    A row states the answer when it differs from its publisher's behaviour on
    that channel -- a gateway serving one of these providers, or a provider
    that changes how it counts. Otherwise the (publisher, channel) pair
    decides, because a catalog author has no way to know which providers report
    cached tokens inside the input total.

    An unknown publisher defaults to no deduction: over-billing a cached token
    twice is the failure this exists to prevent, so a row we cannot place
    keeps the arithmetic it had before.
    """
    stated = rules.get("input_includes_cache_read")
    if stated is not None:
        return bool(stated)
    if not isinstance(publisher, str) or not isinstance(channel, str):
        return False
    pair = (publisher.strip().lower(), channel.strip().lower())
    return pair in _INPUT_INCLUDES_CACHE_READ_PAIRS


def normalize_provider(provider: str) -> str:
    """Fold a provider spelling through metergraph-core's provider-alias map
    (e.g. ``aws``/``amazon-bedrock`` -> ``bedrock``, ``google-genai`` ->
    ``google``) so identity and channel lookups accept every spelling the
    pricing path already does."""
    key = provider.strip().lower()
    return _PROVIDER_ALIASES.get(key, key)


def direct_channel_for_provider(provider: Any) -> str | None:
    """The direct catalog channel for a recorded source provider, or ``None``
    when the provider has no direct channel defined -- callers must treat
    ``None`` as explicitly unpriced rather than falling back to another
    channel."""
    if not isinstance(provider, str):
        return None
    return _DIRECT_CHANNEL_BY_PROVIDER.get(normalize_provider(provider))


def _iso_text(value: Any) -> str:
    """ISO text with a trailing Z or z written as +00:00, which Python
    3.10's fromisoformat requires."""
    text = str(value).strip()
    return text[:-1] + "+00:00" if text[-1:] in ("Z", "z") else text


def _coerce_datetime(at: Any) -> datetime:
    """Normalize an effective time to an aware UTC datetime. A bare ``date`` is
    anchored at UTC midnight; an ISO string is parsed. Prices resolve at a
    timestamp, so every entry point funnels through this."""
    if isinstance(at, datetime):
        return at if at.tzinfo else at.replace(tzinfo=timezone.utc)
    if isinstance(at, date):
        return datetime(at.year, at.month, at.day, tzinfo=timezone.utc)
    parsed = datetime.fromisoformat(_iso_text(at))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


@dataclass(frozen=True, slots=True)
class Alias:
    model_id: str
    canonical_id: str
    pricing_channel: str
    rules: Mapping[str, Any]
    publisher: str | None = None
    effective_from: datetime | None = None
    effective_to: datetime | None = None
    source_url: str | None = None
    # Limits an identifier to matching regions; global matches every snapshot.
    price_region: tuple[str, ...] = ()


def _crosses_price_region(alias: Alias) -> bool:
    """Whether global-prefix fallback would cross a price-region boundary."""
    return bool(alias.price_region) and alias.price_region != (_GLOBAL_REGION,)


def _effective_windows_overlap(left: Alias, right: Alias) -> bool:
    return (
        left.effective_to is None
        or right.effective_from is None
        or right.effective_from < left.effective_to
    ) and (
        right.effective_to is None
        or left.effective_from is None
        or left.effective_from < right.effective_to
    )


@dataclass(frozen=True, slots=True)
class Price:
    id: str
    model_id: str
    pricing_channel: str
    region: str
    input_per_mtok: Decimal | None
    output_per_mtok: Decimal | None
    cache_read_per_mtok: Decimal | None
    cache_write_5m_per_mtok: Decimal | None
    cache_write_1h_per_mtok: Decimal | None
    batch_input_per_mtok: Decimal | None
    batch_output_per_mtok: Decimal | None
    rules: Mapping[str, Any]
    effective_from: datetime
    effective_to: datetime | None
    source_url: str
    # The catalog's publisher for the model this row prices. A channel does not
    # imply one: google-vertex-ai serves Anthropic's models beside Google's,
    # and the two report cache reads differently.
    publisher: str | None = None
    # A fixed fee charged once per call on top of tokens, in USD per 1,000
    # calls: a gateway billing a model's built-in search per request.
    request_fee_per_1k_usd: Decimal | None = None
    # The currency every rate on this row is stated in. The `_usd` names on
    # rates and results predate it and are kept for existing callers.
    currency: str = "USD"
    # The rate a batch call's cache reads bill at, when the provider publishes
    # one. Absent, batch cache reads keep the standard cache-read rate; it is
    # never inferred from another rate.
    batch_cache_read_per_mtok: Decimal | None = None
    # OpenAI publishes a separate Batch/Flex cache-write rate.
    batch_cache_write_5m_per_mtok: Decimal | None = None

    def __post_init__(self) -> None:
        # The loader checks the rate it reads, but a row built or replaced
        # directly must not bill a negative or undefined cache component.
        for name in ("batch_cache_read_per_mtok", "batch_cache_write_5m_per_mtok"):
            rate = getattr(self, name)
            if rate is not None and not (
                isinstance(rate, Decimal) and rate.is_finite() and rate >= 0
            ):
                raise ValueError(
                    f"{name} must be a finite non-negative Decimal or None, not {rate!r}"
                )


@dataclass(frozen=True, slots=True)
class CostComponents:
    """A priced call's cost by what it paid for. The parts sum to the call's
    ``cost_usd`` exactly: each is quantized and the rounding residue lands in
    the input part, so a consumer that stores them can rebuild the total
    without a second rounding. ``request_usd`` is a per-call fee, zero on
    every row that has none. Each part is in the applied price's currency;
    ``input``, ``output``, ``cache_read``, ``cache_write`` and ``request``
    name the same values without implying USD."""

    input_usd: Decimal
    output_usd: Decimal
    cache_read_usd: Decimal
    cache_write_usd: Decimal
    request_usd: Decimal = Decimal("0")

    @property
    def input(self) -> Decimal:
        return self.input_usd

    @property
    def output(self) -> Decimal:
        return self.output_usd

    @property
    def cache_read(self) -> Decimal:
        return self.cache_read_usd

    @property
    def cache_write(self) -> Decimal:
        return self.cache_write_usd

    @property
    def request(self) -> Decimal:
        return self.request_usd


@dataclass(frozen=True, slots=True)
class CostResult:
    canonical_model: str | None
    price_id: str | None
    cost_usd: Decimal | None
    status: str
    reasons: tuple[str, ...] = ()
    # Present whenever cost_usd is: the split of that cost by driver.
    components: CostComponents | None = None
    # The catalog's publisher for the priced model, for billing rules that
    # apply per publisher rather than per channel.
    publisher: str | None = None
    # The applied price, its currency, and the time it was resolved at.
    price: Price | None = None
    currency: str | None = None
    effective_at: datetime | None = None
    # Inputs the caller did not supply that the figure depends on: "region",
    # "channel", "input_tokens", "output_tokens".
    missing_dimensions: tuple[str, ...] = ()
    # Why an unpriced result is unpriced, when the cause is a value the
    # caller sent. Empty otherwise.
    warnings: tuple["PricingWarning", ...] = ()

    @property
    def model_id(self) -> str | None:
        """The resolved catalog model id; the same value as ``canonical_model``."""
        return self.canonical_model

    @property
    def cost(self) -> Decimal | None:
        """The total in ``currency``; the same value as ``cost_usd``."""
        return self.cost_usd


@dataclass(frozen=True, slots=True)
class ModelUsage:
    """Token usage for one model call. A count left ``None`` is unknown, not
    zero: input and output turn the result partial, and cached counts bill as
    nothing cached, as in :meth:`CatalogSnapshot.cost`."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    cache_write_5m_tokens: int | None = None
    cache_write_1h_tokens: int | None = None
    batch: bool = False


@dataclass(frozen=True, slots=True)
class PriceResult:
    """The price a model call would be billed at.

    ``status`` is ``priced``, ``partial`` (a price applies but depends on a
    dimension listed in ``missing_dimensions``), or ``unpriced``. ``model_id``
    is the resolved catalog id, not the alias the caller sent."""

    status: str
    model_id: str | None
    price: Price | None
    effective_at: datetime
    missing_dimensions: tuple[str, ...] = ()
    # Notes that do not make the result partial, such as ``price_ended``.
    reasons: tuple[str, ...] = ()
    # Why an unpriced result is unpriced, when the cause is a value the
    # caller sent. Empty otherwise.
    warnings: tuple["PricingWarning", ...] = ()


@dataclass(frozen=True, slots=True)
class PricingWarning:
    """Why a result is unpriced, when the cause is a value the caller sent:
    a channel or provider the catalog cannot bill on, a model name it does
    not know, or a model the channel does not sell at that time. ``code`` is
    stable; ``message`` names the value and, when one is close, the value
    the caller probably meant."""

    code: str
    message: str


def _did_you_mean(
    value: str, options: Any, guess: str | None = None, cutoff: float = 0.6
) -> str:
    options = sorted(options)
    close = difflib.get_close_matches(value, options, 1, cutoff)
    match = guess if guess in options else next(iter(close), None)
    return f' Did you mean "{match}"?' if match else ""


@dataclass(frozen=True, slots=True)
class ResolvedPrice:
    """An effective price selected for a planned model deployment."""

    canonical_model: str
    price: Price
    rules: Mapping[str, Any]
    # The last known price: it ended before the resolved time, unreplaced.
    ended: bool = False


def _billed_terms(resolved: ResolvedPrice) -> tuple[Any, ...]:
    """What a call is billed by: two resolutions with equal terms cost the
    same for any usage, whichever region or source row they came from."""
    price = resolved.price
    return (
        resolved.canonical_model,
        price.pricing_channel,
        price.currency,
        price.input_per_mtok,
        price.output_per_mtok,
        price.cache_read_per_mtok,
        price.cache_write_5m_per_mtok,
        price.cache_write_1h_per_mtok,
        price.batch_input_per_mtok,
        price.batch_output_per_mtok,
        price.batch_cache_read_per_mtok,
        price.batch_cache_write_5m_per_mtok,
        price.request_fee_per_1k_usd,
        resolved.rules,
    )


def _decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None
    return result if result.is_finite() else None


def _tokens(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return result if result >= 0 else None


def _off_peak_multiplier(rules: Mapping[str, Any], at: datetime) -> Decimal:
    """The discount a provider applies outside its published peak hours.

    A row states the peak rate, which is the provider's list price, and the
    window it applies in. A call outside that window is scaled by
    ``multiplier``; a row with no window is time-independent.
    """
    discount = rules.get("off_peak_discount") or {}
    windows = discount.get("peak_hours_utc") or ()
    multiplier = _decimal(discount.get("multiplier"))
    if not windows or multiplier is None:
        return Decimal("1")
    moment = at.astimezone(timezone.utc)
    if discount.get("peak_weekdays_only") and moment.weekday() >= 5:
        return multiplier
    for window in windows:
        try:
            start, end = window
        except (TypeError, ValueError):
            continue
        if _tokens(start) is None or _tokens(end) is None:
            continue
        if int(start) <= moment.hour < int(end):
            return Decimal("1")
    return multiplier

# A count that was never given and a count that was given but cannot be read are
# different facts with opposite billing consequences for cached tokens: absent
# means there was no cache to discount, unusable means we do not know. `_tokens`
# answers both with None, which is right for a count a caller may legitimately
# omit and wrong for one it supplied.
_UNUSABLE = object()


def _token_count(value: Any) -> Any:
    """A usable token count, ``None`` when none was given, or ``_UNUSABLE`` when
    one was given that cannot be read as a count."""
    if value is None:
        return None
    if isinstance(value, bool):
        return _UNUSABLE
    try:
        result = int(value)
    except (ValueError, TypeError, OverflowError):
        return _UNUSABLE
    return result if result >= 0 else _UNUSABLE


def _cached_count(value: Any, name: str, reasons: list[str]) -> int | None:
    """The count to bill for one cached-token field, recording why when the
    value supplied cannot be used.

    Returns ``None`` only when nothing was given, so a caller that omits a field
    keeps the behaviour it had. An unusable value bills as nothing cached, which
    is the same arithmetic as before, but it now says so: the reason turns the
    result ``partial``, and a caller that trusts only a fully priced figure stops
    reading a silent over-bill as a price.
    """
    count = _token_count(value)
    if count is _UNUSABLE:
        reasons.append(f"unusable_{name}")
        return 0
    return count


def _price_tokens(
    price: Price,
    rules: Mapping[str, Any],
    *,
    at: datetime,
    input_tokens: Any,
    output_tokens: Any,
    cache_read_tokens: Any,
    cache_write_tokens: Any,
    cache_write_5m_tokens: Any,
    cache_write_1h_tokens: Any,
    batch: bool,
) -> tuple[Decimal, list[str], CostComponents]:
    """Cost a token usage against one already-selected price and its merged
    rules. Every token entry point reaches it through ``_cost_result`` so all
    compute identically. Returns the total, the reasons it is partial, and the
    split by driver."""
    reasons: list[str] = []
    input_count = _tokens(input_tokens)
    output_count = _tokens(output_tokens)
    cache_read_count = _cached_count(cache_read_tokens, "cache_read_tokens", reasons) or 0
    cache_write_5m_count = _cached_count(
        cache_write_5m_tokens, "cache_write_5m_tokens", reasons
    )
    if cache_write_5m_count is None:
        # Only an absent split falls back to the aggregate. A stated split we
        # cannot read is not an invitation to substitute a different field.
        cache_write_5m_count = (
            _cached_count(cache_write_tokens, "cache_write_tokens", reasons) or 0
        )
    cache_write_1h_count = (
        _cached_count(cache_write_1h_tokens, "cache_write_1h_tokens", reasons) or 0
    )
    cache_write_count = cache_write_5m_count + cache_write_1h_count
    if input_count is None:
        reasons.append("missing_input_tokens")
        input_count = 0
    if output_count is None:
        reasons.append("missing_output_tokens")
        output_count = 0

    billable_input = input_count
    deducted_input = 0
    input_includes_cache_read = counts_cache_read_in_input(
        price.publisher, price.pricing_channel, rules
    )
    if input_includes_cache_read:
        if cache_read_count > input_count:
            reasons.append("cache_read_exceeds_input")
            deducted_input = input_count
        else:
            deducted_input += cache_read_count
    if rules.get("input_includes_cache_write"):
        if cache_write_count > input_count - deducted_input:
            reasons.append("cache_write_exceeds_input")
            deducted_input = input_count
        else:
            deducted_input += cache_write_count
    billable_input -= deducted_input

    input_rate = price.input_per_mtok
    output_rate = price.output_per_mtok
    cache_read_rate = price.cache_read_per_mtok
    cache_write_5m_rate = price.cache_write_5m_per_mtok
    if batch:
        if price.batch_input_per_mtok is None or price.batch_output_per_mtok is None:
            reasons.append("batch_rate_unavailable")
        else:
            input_rate = price.batch_input_per_mtok
            output_rate = price.batch_output_per_mtok
        # A row without a batch cache rate keeps billing batch cache reads at
        # the standard rate, as it always has.
        if price.batch_cache_read_per_mtok is not None:
            cache_read_rate = price.batch_cache_read_per_mtok
        if price.batch_cache_write_5m_per_mtok is not None:
            cache_write_5m_rate = price.batch_cache_write_5m_per_mtok

    input_multiplier = Decimal("1")
    output_multiplier = Decimal("1")
    long_context = rules.get("long_context") or {}
    threshold = _tokens(long_context.get("threshold"))
    prompt_count = input_count
    if long_context.get("threshold_counts_cached_tokens"):
        if not input_includes_cache_read:
            prompt_count += cache_read_count
        if not rules.get("input_includes_cache_write"):
            prompt_count += cache_write_count
    # Some providers publish no long-context batch rate, so batch stays flat.
    applies = not batch or long_context.get("applies_to_batch", True)
    if applies and threshold is not None and prompt_count > threshold:
        input_multiplier = _decimal(long_context.get("input_multiplier")) or Decimal("1")
        output_multiplier = _decimal(
            long_context.get("output_multiplier")
        ) or Decimal("1")
    # Applies to every rate on the row, cache included, and composes with a
    # long-context tier rather than replacing it.
    off_peak = _off_peak_multiplier(rules, at)
    input_multiplier *= off_peak
    output_multiplier *= off_peak

    input_usd = output_usd = cache_read_usd = cache_write_usd = Decimal("0")
    if input_rate is None:
        if billable_input:
            reasons.append("input_rate_unavailable")
    else:
        input_usd = Decimal(billable_input) * input_rate * input_multiplier / _MILLION
    if output_rate is None:
        if output_count:
            reasons.append("output_rate_unavailable")
    else:
        output_usd = Decimal(output_count) * output_rate * output_multiplier / _MILLION
    if cache_read_count:
        if cache_read_rate is None:
            reasons.append("cache_read_rate_unavailable")
        else:
            cache_read_usd = (
                Decimal(cache_read_count)
                * cache_read_rate
                * input_multiplier
                / _MILLION
            )
    if cache_write_5m_count:
        if cache_write_5m_rate is None:
            reasons.append("cache_write_5m_rate_unavailable")
        else:
            cache_write_usd += (
                Decimal(cache_write_5m_count)
                * cache_write_5m_rate
                * input_multiplier
                / _MILLION
            )
    if cache_write_1h_count:
        if price.cache_write_1h_per_mtok is None:
            reasons.append("cache_write_1h_rate_unavailable")
        else:
            cache_write_usd += (
                Decimal(cache_write_1h_count)
                * price.cache_write_1h_per_mtok
                * input_multiplier
                / _MILLION
            )
    if rules.get("uncaptured_fees"):
        reasons.append("uncaptured_fees")
    request_usd = (
        Decimal("0")
        if price.request_fee_per_1k_usd is None
        else price.request_fee_per_1k_usd / Decimal(1000)
    )

    # The total is rounded once, as it always was; the parts are rounded
    # each and the residue goes to input, so they sum to the total exactly.
    cost = (input_usd + output_usd + cache_read_usd + cache_write_usd + request_usd).quantize(
        _COST_QUANTUM, rounding=ROUND_HALF_UP
    )
    parts = [
        part.quantize(_COST_QUANTUM, rounding=ROUND_HALF_UP)
        for part in (input_usd, output_usd, cache_read_usd, cache_write_usd, request_usd)
    ]
    parts[0] += cost - sum(parts)
    return cost, reasons, CostComponents(*parts)


def _cost_result(
    resolved: ResolvedPrice,
    *,
    at: datetime,
    usage: ModelUsage,
    missing: tuple[str, ...] = (),
) -> CostResult:
    """Cost ``usage`` against a resolved price. Every token entry point builds
    its result here, so all of them compute and report identically."""
    cost, reasons, components = _price_tokens(
        resolved.price,
        resolved.rules,
        at=at,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=usage.cache_read_tokens,
        cache_write_tokens=usage.cache_write_tokens,
        cache_write_5m_tokens=usage.cache_write_5m_tokens,
        cache_write_1h_tokens=usage.cache_write_1h_tokens,
        batch=usage.batch,
    )
    reasons = [f"missing_{dimension}" for dimension in missing] + reasons
    partial = bool(reasons)
    if resolved.ended:
        reasons = [PRICE_ENDED] + reasons
    missing += tuple(
        dimension
        for dimension in ("input_tokens", "output_tokens")
        if f"missing_{dimension}" in reasons
    )
    return CostResult(
        resolved.canonical_model,
        resolved.price.id,
        cost,
        "partial" if partial else "priced",
        tuple(dict.fromkeys(reasons)),
        components=components,
        publisher=resolved.price.publisher,
        price=resolved.price,
        currency=resolved.price.currency,
        effective_at=at,
        missing_dimensions=missing,
    )


class CatalogSnapshot:
    def __init__(
        self,
        aliases: Mapping[tuple[str, str], Alias | tuple[Alias, ...] | list[Alias]],
        prices: list[Price],
        *,
        region: str,
    ) -> None:
        self._aliases = {
            key: (value,) if isinstance(value, Alias) else tuple(value)
            for key, value in aliases.items()
        }
        self._prices: dict[tuple[str, str], list[Price]] = {}
        deployment_aliases: dict[tuple[str, str], list[Alias]] = {}
        self._region = region.strip().lower()
        for (_, observed_model), candidates in self._aliases.items():
            for alias in candidates:
                channel = alias.pricing_channel.strip().lower()
                for model in (observed_model, alias.canonical_id):
                    key = (model.strip().lower(), channel)
                    deployment_aliases.setdefault(key, []).append(alias)
        self._deployment_aliases = {
            key: tuple(candidates) for key, candidates in deployment_aliases.items()
        }
        for (model, channel), candidates in self._deployment_aliases.items():
            for index, alias in enumerate(candidates):
                for other in candidates[index + 1:]:
                    if not _effective_windows_overlap(alias, other):
                        continue
                    if (
                        alias.canonical_id != other.canonical_id
                        or dict(alias.rules) != dict(other.rules)
                    ):
                        raise ValueError(
                            f"ambiguous deployment alias {model!r} "
                            f"for channel {channel!r}"
                        )
        for price in prices:
            self._prices.setdefault((price.model_id, price.pricing_channel), []).append(
                price
            )
        for candidates in self._prices.values():
            candidates.sort(key=lambda price: price.effective_from, reverse=True)
        # For explaining an unpriced result: the channels some price row
        # bills on, and each model name with the channels that sell it under
        # that name, an alias there and a price there, at any time.
        priced = {(model_id, channel.lower()) for model_id, channel in self._prices}
        self._priced_channels = frozenset(channel for _, channel in priced)
        sold: dict[str, set[str]] = {}
        for (model, channel), candidates in self._deployment_aliases.items():
            found = sold.setdefault(model, set())
            if any((alias.model_id, channel) in priced for alias in candidates):
                found.add(channel)
        self._sold = {model: frozenset(channels) for model, channels in sold.items()}
        # Every named region a price or pin exists for: the regions an omitted
        # query region could have meant.
        self._regions = tuple(sorted(
            {price.region.lower() for price in prices}.union(
                region
                for candidates in self._aliases.values()
                for alias in candidates
                for region in alias.price_region
            ) - {"*", _GLOBAL_REGION}
        ))

    def _alias_for(
        self, candidates: tuple[Alias, ...], at: datetime, region: str | None = None
    ) -> Alias | None:
        when = _coerce_datetime(at)
        effective = [
            alias
            for alias in candidates
            if (alias.effective_from is None or alias.effective_from <= when)
            and (alias.effective_to is None or when < alias.effective_to)
        ]
        # Regional pins constrain availability; global pins match everywhere.
        for pinned in dict.fromkeys((region or self._region, _GLOBAL_REGION)):
            for alias in effective:
                if pinned in alias.price_region:
                    return alias
        for alias in effective:
            if not alias.price_region:
                return alias
        return None

    def _price_regions(self, alias: Alias, region: str) -> tuple[str, ...]:
        if not alias.price_region:
            return tuple(dict.fromkeys((region, "*", _GLOBAL_REGION)))
        for candidate in (region, _GLOBAL_REGION):
            if candidate in alias.price_region:
                return (candidate,)
        return ()

    def _price_for(
        self, alias: Alias, at: datetime, region: str | None = None
    ) -> tuple[Price | None, bool]:
        """The price in effect at ``at``, and whether it is the last known
        price: when no row covers ``at``, the row that applied most recently,
        provided nothing in these scopes starts at or after its end. A
        superseded price, or a gap before a scheduled one, never extends."""
        at = at if at.tzinfo else at.replace(tzinfo=timezone.utc)
        candidates = self._prices.get((alias.model_id, alias.pricing_channel), [])
        scopes = self._price_regions(alias, region or self._region)
        for scope in scopes:
            for price in candidates:
                if price.region.lower() != scope:
                    continue
                if price.effective_from <= at and (
                    price.effective_to is None or at < price.effective_to
                ):
                    return price, False
        in_scopes = [price for price in candidates if price.region.lower() in scopes]
        ended = [
            price for price in in_scopes
            if price.effective_to is not None and price.effective_to <= at
        ]
        if not ended:
            return None, False
        # The latest end wins; on a tie, the earlier scope, as above.
        last = max(
            ended,
            key=lambda price: (price.effective_to, -scopes.index(price.region.lower())),
        )
        if any(price.effective_from >= last.effective_to for price in in_scopes):
            return None, False
        return last, True

    def infer_direct_channel(self, model: Any) -> str | None:
        """Return the sole direct billing channel known for ``model``.

        This supports captures that preserve a model identity but omit the
        source provider. It uses the catalog's canonical model publisher and
        returns ``None`` whenever model identity or direct channel is
        ambiguous.
        """
        if not isinstance(model, str) or not model.strip():
            return None
        model_key = model.strip().lower()
        canonical_ids = {
            alias.canonical_id
            for (known_model, _channel), candidates in self._deployment_aliases.items()
            if known_model == model_key
            for alias in candidates
        }
        if len(canonical_ids) != 1:
            return None
        channels: set[str] = set()
        for canonical_id in canonical_ids:
            for candidates in self._deployment_aliases.values():
                for alias in candidates:
                    if alias.canonical_id != canonical_id:
                        continue
                    channel = direct_channel_for_provider(alias.publisher)
                    if channel is not None and (canonical_id, channel) in self._prices:
                        channels.add(channel)
        return next(iter(channels)) if len(channels) == 1 else None

    def resolve_price(
        self, *, model: Any, channel: Any, at: datetime
    ) -> ResolvedPrice | None:
        """Resolve pricing for an explicitly selected model and channel.

        This is intended for planners and evaluation pipelines that know the
        deployment channel before making a provider call. It never falls back
        to pricing from a different channel.
        """
        return self._resolve(model=model, channel=channel, at=at)

    def _resolve(
        self, *, model: Any, channel: Any, at: Any, region: str | None = None
    ) -> ResolvedPrice | None:
        """``resolve_price`` in ``region``, or the snapshot's region when
        ``None``."""
        model_key = str(model or "").strip().lower()
        channel_key = str(channel or "").strip().lower()
        when = _coerce_datetime(at)
        candidates = self._deployment_aliases.get((model_key, channel_key), ())
        alias = self._alias_for(candidates, when, region)
        if alias is None and channel_key in _BEDROCK_CHANNELS:
            for candidate, dropped_global in _bedrock_fallback_ids(model_key):
                if dropped_global and channel_key != _BEDROCK_CHANNEL:
                    continue
                candidates = self._deployment_aliases.get((candidate, channel_key), ())
                alias = self._alias_for(candidates, when, region)
                if alias is not None and dropped_global and _crosses_price_region(alias):
                    alias = None
                if alias is not None:
                    break
        if alias is None:
            return None
        price, ended = self._price_for(alias, when, region)
        if price is None:
            return None
        return ResolvedPrice(
            canonical_model=alias.canonical_id,
            price=price,
            rules=MappingProxyType({**price.rules, **alias.rules}),
            ended=ended,
        )

    def _lookup(
        self, *, model: Any, channel: Any, provider: Any, at: Any, region: Any
    ) -> tuple[ResolvedPrice | None, tuple[str, ...]]:
        """Resolve a model call for the public lookup and cost entry points,
        with the dimensions the answer depends on but the caller left out.

        An explicit channel wins; otherwise a provider names its direct
        channel. An explicit region resolves that region. An omitted one uses
        the snapshot's region, and when that is the default ``global`` it is
        checked against every named region: a global price that some region
        bills differently is returned with ``region`` missing, and a model only
        some region prices is unpriced with ``region`` missing.
        """
        if channel is None or not str(channel).strip():
            channel = direct_channel_for_provider(provider)
            if channel is None:
                return None, ("channel",)
        stated = str(region or "").strip().lower() or None
        resolved = self._resolve(model=model, channel=channel, at=at, region=stated)
        if stated is not None or self._region != _GLOBAL_REGION:
            return resolved, ()
        regional = [
            found
            for found in (
                self._resolve(model=model, channel=channel, at=at, region=name)
                for name in self._regions
            )
            if found is not None
        ]
        if resolved is None:
            return None, ("region",) if regional else ()
        if any(_billed_terms(found) != _billed_terms(resolved) for found in regional):
            return resolved, ("region",)
        return resolved, ()

    def _explain(
        self, *, model: Any, channel: Any, provider: Any, at: Any, region: Any
    ) -> tuple[PricingWarning, ...]:
        """Why :meth:`_lookup` found no price, when the cause is a value the
        caller sent. Each check follows the step of the lookup it explains:
        the channel, or the provider's direct channel, then the model name on
        that channel, then the alias and price windows at ``at``."""
        channel_key = str(channel or "").strip().lower()
        if channel_key:
            if channel_key not in self._priced_channels:
                # A provider's name, with or without "-api", points at its
                # direct channel.
                guess = direct_channel_for_provider(
                    channel_key
                ) or direct_channel_for_provider(channel_key.removesuffix("-api"))
                return (PricingWarning(
                    UNKNOWN_CHANNEL,
                    f'channel "{str(channel).strip()}" is not a channel the catalog prices.'
                    f"{_did_you_mean(channel_key, self._priced_channels, guess)}",
                ),)
        else:
            if provider is None or not str(provider).strip():
                return ()
            channel_key = direct_channel_for_provider(provider)
            if channel_key is None:
                providers = set(_DIRECT_CHANNEL_BY_PROVIDER) | {
                    spelling for spelling, name in _PROVIDER_ALIASES.items()
                    if name in _DIRECT_CHANNEL_BY_PROVIDER
                }
                return (PricingWarning(
                    UNKNOWN_PROVIDER,
                    f'provider "{str(provider).strip()}" has no billing channel of its own.'
                    f"{_did_you_mean(str(provider).strip().lower(), providers)} If a gateway or "
                    "proxy relayed the call, name the provider that served it or the channel "
                    "that billed it.",
                ),)
        model_key = str(model or "").strip().lower()
        if not model_key:
            return ()
        # Bedrock fallback ids, under the rules _resolve applies: dropping
        # "global." is valid only on aws-bedrock, and never across a price
        # region.
        keys, dropped, forbidden = [model_key], set(), []
        if channel_key in _BEDROCK_CHANNELS:
            for candidate, dropped_global in _bedrock_fallback_ids(model_key):
                if dropped_global and channel_key != _BEDROCK_CHANNEL:
                    forbidden.append(candidate)
                    continue
                keys.append(candidate)
                if dropped_global:
                    dropped.add(candidate)
        name = str(model).strip()
        if not any(key in self._sold for key in keys):
            # A "global." id on a geo channel points at the bare id that
            # channel sells.
            guess = next(
                (key for key in forbidden if channel_key in self._sold.get(key, ())), None
            )
            return (PricingWarning(
                UNKNOWN_MODEL,
                f'model "{name}" is not in the catalog.'
                f"{_did_you_mean(model_key, self._sold, guess, cutoff=0.8)}",
            ),)
        candidates = tuple(
            alias
            for key in keys
            for alias in self._deployment_aliases.get((key, channel_key), ())
            if key not in dropped or not _crosses_price_region(alias)
        )
        sold = sorted({channel for key in keys for channel in self._sold.get(key, ())})
        if not candidates or channel_key not in sold:
            where = (
                f"It is priced on {', '.join(sold)}."
                if sold
                else "No channel prices it under that name."
            )
            return (PricingWarning(
                MODEL_NOT_ON_CHANNEL,
                f'model "{name}" has no price on channel "{channel_key}". {where}',
            ),)
        when = _coerce_datetime(at)
        stated = str(region or "").strip().lower() or None
        alias = self._alias_for(candidates, when, stated)
        if alias is None:
            return (self._unresolved(name, channel_key, candidates, when, stated),)
        price, _ = self._price_for(alias, when, stated)
        if price is None:
            scope = f' in region "{stated}"' if stated else ""
            return (PricingWarning(
                PRICE_NOT_IN_EFFECT,
                f'model "{name}" has no price on channel "{channel_key}"{scope} in effect at '
                f"{when.isoformat()}.",
            ),)
        return ()

    def _unresolved(
        self,
        name: str,
        channel: str,
        candidates: tuple[Alias, ...],
        when: datetime,
        region: str | None,
    ) -> PricingWarning:
        """Why no alias for ``name`` on ``channel`` applies at ``when``: its
        windows close before or open after it, or the aliases in effect are
        pinned to other regions."""
        current = [
            alias for alias in candidates
            if (alias.effective_from is None or alias.effective_from <= when)
            and (alias.effective_to is None or when < alias.effective_to)
        ]
        if current:
            regions = sorted({pin for alias in current for pin in alias.price_region})
            asked = region or self._region
            return PricingWarning(
                MODEL_NOT_IN_REGION,
                f'model "{name}" resolves on channel "{channel}" only in '
                f"{', '.join(regions)}, not {asked}.",
            )
        ended = [alias.effective_to for alias in candidates
                 if alias.effective_to is not None and alias.effective_to <= when]
        starts = [alias.effective_from for alias in candidates
                  if alias.effective_from is not None and when < alias.effective_from]
        if ended and starts:
            when_text = (
                f"between {max(ended).date().isoformat()} and "
                f"{min(starts).date().isoformat()}"
            )
            message = f'model "{name}" does not resolve on channel "{channel}" {when_text}.'
        elif ended:
            message = (
                f'model "{name}" stopped resolving on channel "{channel}" on '
                f"{max(ended).date().isoformat()}."
            )
        else:
            message = (
                f'model "{name}" resolves on channel "{channel}" from '
                f"{min(starts).date().isoformat()}."
            )
        return PricingWarning(MODEL_NOT_IN_EFFECT, message)

    def cost(
        self,
        *,
        provider: Any,
        model: Any,
        at: datetime,
        input_tokens: Any,
        output_tokens: Any,
        cache_read_tokens: Any = None,
        cache_write_tokens: Any = None,
        cache_write_5m_tokens: Any = None,
        cache_write_1h_tokens: Any = None,
        batch: bool = False,
    ) -> CostResult:
        provider_key = str(provider or "").strip().lower()
        provider_key = _PROVIDER_ALIASES.get(provider_key, provider_key)
        model_key = str(model or "").strip().lower()
        when = _coerce_datetime(at)
        candidates = self._aliases.get((provider_key, model_key), ())
        alias_known = bool(candidates)
        alias = self._alias_for(candidates, when)
        if alias is None and provider_key == "bedrock":
            for candidate, dropped_global in _bedrock_fallback_ids(model_key):
                fallback = self._aliases.get((provider_key, candidate), ())
                alias_known = alias_known or bool(fallback)
                found = self._alias_for(fallback, when)
                if found is None or found.pricing_channel not in _BEDROCK_CHANNELS:
                    continue
                if dropped_global and (
                    found.pricing_channel != _BEDROCK_CHANNEL
                    or _crosses_price_region(found)
                ):
                    continue
                alias = found
                break
        if alias is None:
            if alias_known:
                return CostResult(
                    None, None, None, "unpriced", ("no_effective_alias",)
                )
            return CostResult(None, None, None, "unpriced", ("unknown_model",))
        price, ended = self._price_for(alias, when)
        if price is None:
            return CostResult(
                alias.canonical_id,
                None,
                None,
                "unpriced",
                ("no_effective_price",),
            )
        return _cost_result(
            ResolvedPrice(
                alias.canonical_id, price, {**price.rules, **alias.rules}, ended=ended
            ),
            at=when,
            usage=ModelUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read_tokens,
                cache_write_tokens=cache_write_tokens,
                cache_write_5m_tokens=cache_write_5m_tokens,
                cache_write_1h_tokens=cache_write_1h_tokens,
                batch=batch,
            ),
        )

    def price_deployment(
        self,
        *,
        model: Any,
        channel: Any,
        at: Any,
        input_tokens: Any,
        output_tokens: Any,
        cache_read_tokens: Any = None,
        cache_write_tokens: Any = None,
        cache_write_5m_tokens: Any = None,
        cache_write_1h_tokens: Any = None,
        batch: bool = False,
    ) -> CostResult:
        """Price an observed deployment from the identity a caller already has:
        a model id, the pricing channel it was served on, the execution time,
        and its token usage.

        The channel is honored exactly -- a model priced on a channel the
        catalog does not carry for it comes back ``unpriced`` rather than being
        repriced off a different channel. The caller never has to rebuild a
        provider or alias index from the catalog document; resolution and cost
        both live here.
        """
        when = _coerce_datetime(at)
        resolved = self.resolve_price(model=model, channel=channel, at=when)
        if resolved is None:
            return CostResult(None, None, None, "unpriced", ("unknown_deployment",))
        return _cost_result(
            resolved,
            at=when,
            usage=ModelUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read_tokens,
                cache_write_tokens=cache_write_tokens,
                cache_write_5m_tokens=cache_write_5m_tokens,
                cache_write_1h_tokens=cache_write_1h_tokens,
                batch=batch,
            ),
        )
