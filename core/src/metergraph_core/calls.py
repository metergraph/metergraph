"""Compose token and counted retrieval costs from one catalog snapshot."""
from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
import re
from typing import Any, Mapping, Sequence

from .catalog import CostComponents, CostResult, ModelUsage, direct_channel_for_provider, normalize_provider, _supports_flex
from .loader import LoadedCatalog
from .providers import model_identity
from .providers.anthropic import rejected_without_usage
from .responses import _as_mapping, _bundled_catalog, _first, cost_from_response, response_format, _native_input_tokens

_ZERO = Decimal('0')


@dataclass(frozen=True, slots=True)
class RetrievalUsage:
    operation: str
    units: Any


@dataclass(frozen=True, slots=True)
class RetrievalComponent:
    operation: str
    units: Any
    cost_usd: Decimal | None
    price_id: str | None
    basis: str
    reasons: tuple[str, ...] = ()
    source_url: str | None = None


@dataclass(frozen=True, slots=True)
class CallCostResult:
    token_cost: CostResult
    retrieval: tuple[RetrievalComponent, ...]
    cost_usd: Decimal | None
    known_subtotal_usd: Decimal
    estimated_total_usd: Decimal | None
    status: str
    token_basis: str
    missing_dimensions: tuple[str, ...]
    catalog_hash: str


    @property
    def publisher(self) -> str | None:
        return self.token_cost.publisher

    @property
    def price_id(self) -> str | None:
        return self.token_cost.price_id

    @property
    def reasons(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.token_cost.reasons,
            *(f"missing_{d}" for d in self.missing_dimensions))))


    @property
    def retrieval_cost_usd(self) -> Decimal | None:
        if not self.retrieval or any(item.cost_usd is None for item in self.retrieval):
            return None
        return sum((item.cost_usd for item in self.retrieval), _ZERO)

    @property
    def search_units(self) -> int | None:
        searches = [item for item in self.retrieval if item.operation.startswith(
            ('web_search', 'google_search', 'search_request'))]
        if not searches or any(type(item.units) is not int or item.units < 0 for item in searches):
            return None
        return sum(item.units for item in searches)


def _grounding_operation(model: str | None) -> str:
    match = re.search(r'(?:^|/)gemini-(\d+)', model or '')
    if match is None:
        return 'google_search_unknown'
    return 'google_search_grounding' if int(match[1]) >= 3 else 'google_search_grounded_prompt'


def _supported_retrieval(channel, publisher, model, operation):
    if channel == 'perplexity-agent-api':
        return operation in {'web_search', 'web_search_fast'}
    if channel == 'perplexity-api':
        return operation in {'search_request_low', 'search_request_medium', 'search_request_high'}
    if channel == 'openai-api' and publisher == 'openai':
        return operation == 'web_search'
    if channel == 'anthropic-api' and publisher == 'anthropic':
        return operation in {'web_search', 'web_fetch'}
    if channel in {'google-api', 'google-vertex-ai'} and publisher == 'google':
        return operation == _grounding_operation(model)
    return False


def compose_call_cost(
    token_cost: CostResult, *, catalog: LoadedCatalog,
    retrieval: Sequence[RetrievalUsage] = (), provider: str | None = None,
    channel: str | None = None, region: str = 'global',
) -> CallCostResult:
    """Combine retrieval with a token result from the same catalog and time.

    Partial token results remain estimates, excluded from the known subtotal.
    """
    if token_cost.effective_at is None:
        raise ValueError('token_cost must carry its effective_at')
    if token_cost.price is not None:
        selected = catalog.lookup_price(model=token_cost.canonical_model,
            channel=token_cost.price.pricing_channel, region=token_cost.price.region,
            at=token_cost.effective_at)
        if selected.price != token_cost.price:
            raise ValueError('token price does not match the catalog snapshot')
    if len({item.operation for item in retrieval}) != len(retrieval):
        raise ValueError('retrieval operations must be aggregated once per call')
    resolved_channel = token_cost.price.pricing_channel if token_cost.price else (channel or direct_channel_for_provider(provider))
    publisher = token_cost.publisher or normalize_provider(provider or '')
    missing = list(token_cost.missing_dimensions)
    token_basis = ('exact' if token_cost.status == 'priced' else
                   'fallback_estimate' if token_cost.cost_usd is not None else 'unknown')
    # Retrieval rates are USD. Do not add them to a different currency.
    currency_ok = token_cost.currency in (None, 'USD')
    known = token_cost.cost_usd if token_basis == 'exact' and currency_ok else _ZERO
    if not currency_ok:
        missing.append('currency')
        token_basis = 'unknown'
    parts = []
    for usage in retrieval:
        if _supported_retrieval(resolved_channel, publisher, token_cost.canonical_model, usage.operation):
            fee = catalog.price_retrieval(channel=resolved_channel, operation=usage.operation,
                units=usage.units, at=token_cost.effective_at, region=region)
            # A measured zero incurs no fee even when its current rate is unknown.
            zero = type(usage.units) is int and usage.units == 0
            part = RetrievalComponent(usage.operation, usage.units, _ZERO if zero else fee.cost_usd,
                fee.price_id, 'exact' if zero or fee.status == 'priced' else 'unknown',
                () if zero else fee.reasons, fee.source_url)
        else:
            part = RetrievalComponent(usage.operation, usage.units, None, None,
                'unknown', ('unsupported_retrieval_scope',))
        part = RetrievalComponent(part.operation,
            part.units if type(part.units) is int and part.units >= 0 else None,
            part.cost_usd, part.price_id, part.basis, part.reasons, part.source_url)
        parts.append(part)
        if part.cost_usd is None:
            missing.append(f'retrieval.{usage.operation}')
        else:
            known += part.cost_usd
    unresolved = any(p.basis == 'unknown' for p in parts)
    status = ('unpriced' if token_cost.price is None or token_cost.status == 'unpriced' else
              'partial' if token_basis != 'exact' or unresolved or missing else 'priced')
    estimated = None
    if token_basis == 'fallback_estimate' and not unresolved:
        estimated = token_cost.cost_usd + sum((p.cost_usd or _ZERO for p in parts), _ZERO)
    return CallCostResult(token_cost, tuple(parts), known if status == 'priced' else None,
        known, estimated, status, token_basis, tuple(dict.fromkeys(missing)), catalog.content_hash)


def price_call(
    *, model: Any, usage: ModelUsage, at: Any, catalog: LoadedCatalog | None = None,
    provider: str | None = None, channel: str | None = None, region: str | None = None,
    retrieval: Sequence[RetrievalUsage] = (),
) -> CallCostResult:
    """Price normalized token and retrieval usage against one snapshot."""
    loaded = catalog if catalog is not None else _bundled_catalog()
    tokens = loaded.calculate_cost(model=model, usage=usage, at=at, provider=provider,
        channel=channel, region=region)
    return compose_call_cost(tokens, catalog=loaded, retrieval=retrieval, provider=provider,
        channel=channel or direct_channel_for_provider(provider), region=region or 'global')


def response_retrieval_usage(response: Any, *, format: str | None = None) -> tuple[RetrievalUsage, ...]:
    """Read operation metadata without reading prompt or response text.

    Tool evidence without a valid usage counter remains unresolved.
    """
    data = _as_mapping(response)
    detected = format or response_format(data)
    if detected == 'anthropic.messages':
        usage = data.get('usage')
        usage = usage if isinstance(usage, Mapping) else {}
        server = usage.get('server_tool_use')
        if server is not None and not isinstance(server, Mapping):
            return (RetrievalUsage('web_search', None),)
        server = server or {}
        found = [RetrievalUsage(op, server[key]) for key, op in (
            ('web_search_requests', 'web_search'), ('web_fetch_requests', 'web_fetch')) if key in server]
        observed = set()
        content = data.get('content')
        if isinstance(content, list):
            for block in content:
                if isinstance(block, Mapping) and block.get('type') == 'server_tool_use':
                    name = block.get('name')
                    if name in ('web_search', 'web_fetch'):
                        observed.add(name)
        found.extend(RetrievalUsage(op, None) for op in sorted(observed) if op not in {f.operation for f in found})
        return tuple(found)
    if detected == 'gemini.generate_content':
        candidates = data.get('candidates')
        if isinstance(candidates, list):
            for item in candidates:
                metadata = _first(item, 'groundingMetadata', 'grounding_metadata') if isinstance(item, Mapping) else None
                if isinstance(metadata, Mapping) and any(
                    _first(metadata, camel, snake) is not None
                    for camel, snake in (("webSearchQueries", "web_search_queries"),
                                         ("groundingChunks", "grounding_chunks"))
                ):
                    # Grounding evidence does not establish billable query count.
                    return (RetrievalUsage('google_search_grounding', None),)
    if detected == 'openai.responses':
        output = data.get('output')
        if isinstance(output, list):
            count = sum(1 for item in output if isinstance(item, Mapping) and item.get('type') == 'web_search_call')
            if count:
                return (RetrievalUsage('web_search', count),)
    return ()


def price_call_from_response(
    provider: str, response: Any, *, catalog: LoadedCatalog | None = None,
    model: str | None = None, at: Any = None, channel: str | None = None,
    region: str | None = None, service_tier: str | None = None, format: str | None = None,
) -> CallCostResult:
    """Price a provider response, including supported counted retrieval fees."""
    loaded = catalog if catalog is not None else _bundled_catalog()
    tokens = cost_from_response(provider, response, catalog=loaded, model=model, at=at,
        channel=channel, region=region, service_tier=service_tier, format=format)
    return compose_call_cost(tokens, catalog=loaded,
        retrieval=response_retrieval_usage(response, format=format), provider=provider,
        channel=channel or direct_channel_for_provider(provider), region=region or 'global')


def _record_retrieval_usage(row: Mapping[str, Any], model: str | None) -> tuple[RetrievalUsage, ...]:
    provider = normalize_provider(row.get('provider') or '')
    if provider in {'perplexity-ai', 'perplexity-agent'}:
        if row.get('retrieval_api') == 'agent' or provider == 'perplexity-agent':
            units = row.get('web_search_calls')
            operation = {'web': 'web_search', 'fast': 'web_search_fast'}.get(row.get('search_type'))
            if type(units) is int and units == 0:
                operation = 'web_search'
            return (RetrievalUsage(operation or 'web_search_unknown', units),)
        context = row.get('search_context_size', 'low')
        operation = f'search_request_{context}' if context in {'low', 'medium', 'high'} else 'search_request_unknown'
        return (RetrievalUsage(operation, 1),)
    if provider in {'google', 'vertex-ai'} and 'grounding_queries' in row:
        operation = _grounding_operation(model)
        units = row['grounding_queries']
        if operation == 'google_search_grounded_prompt':
            units = row.get('grounded_prompt')
            if units is None and type(row['grounding_queries']) is int and row['grounding_queries'] >= 0:
                units = int(row['grounding_queries'] > 0)
        return (RetrievalUsage(operation, units),)
    found = {}
    if 'server_tool_use' in row:
        found = {item.operation: item for item in response_retrieval_usage(
            {'usage': {'server_tool_use': row['server_tool_use']}}, format='anthropic.messages')}
    if 'web_search_calls' in row and (row['web_search_calls'] is not None or 'web_search' not in found):
        found['web_search'] = RetrievalUsage('web_search', row['web_search_calls'])
    tools = row.get('tool_calls')
    if isinstance(tools, list):
        for kind, operation in (('web_search_call', 'web_search'), ('web_fetch_call', 'web_fetch')):
            units = sum(isinstance(tool, Mapping) and tool.get('type') == kind for tool in tools)
            if units and operation not in found:
                found[operation] = RetrievalUsage(operation, units)
    return tuple(found.values())


def price_call_from_record(
    row: Mapping[str, Any], *, catalog: LoadedCatalog | None = None,
    token_cost: CostResult | None = None, at: Any = None, region: str = 'global',
) -> CallCostResult:
    """Price normalized SDK/relay usage; provider billing policy stays here.

    A receiver may supply its resolved token result from the same catalog.
    Explicit API identity takes precedence when it selects a different channel.
    No prompt or response content is read.
    """
    loaded = catalog if catalog is not None else _bundled_catalog()
    provider = row.get('provider')
    model, channel = model_identity(provider, row.get('model'))
    identity_changed = (model, channel) != (row.get('model'), direct_channel_for_provider(provider))
    when = at if at is not None else token_cost.effective_at if token_cost is not None and token_cost.effective_at is not None else row.get('ts')
    agent = normalize_provider(provider or '') in {'perplexity-ai', 'perplexity-agent'} and (
        row.get('retrieval_api') == 'agent' or provider == 'perplexity-agent')
    if agent:
        channel = 'perplexity-agent-api'
    tier = row.get('service_tier')
    if token_cost is None or token_cost.effective_at is None or agent or identity_changed or tier == 'flex':
        values = {key: row.get(key) for key in ('input_tokens', 'output_tokens', 'cache_read_tokens',
            'cache_write_tokens', 'cache_write_5m_tokens', 'cache_write_1h_tokens')}
        def cost(**usage):
            return loaded.calculate_cost(model=model, usage=ModelUsage(**usage, batch=row.get('batch') is True, service_tier=tier),
                at=when, provider=provider, channel=channel, region=region)
        if row.get('input_tokens_include_cache') is True:
            values['input_tokens'] = _native_input_tokens(values, cost)
        token_cost = cost(**values)
    if normalize_provider(provider or '') == 'anthropic' and rejected_without_usage(row):
        free = replace(token_cost, cost_usd=_ZERO, status='priced',
            reasons=('provider_failure_not_billable',), missing_dimensions=(),
            components=CostComponents(_ZERO, _ZERO, _ZERO, _ZERO))
        return CallCostResult(free, (), _ZERO, _ZERO, None, 'priced', 'exact', (), loaded.content_hash)
    failed_stream = row.get('stream') is True and (
        row.get('error') is True or row.get('status') in {'error', 'failed', 'abandoned'})
    if row.get('usage_complete') is False or failed_stream:
        token_cost = replace(token_cost, status='partial',
            missing_dimensions=tuple(dict.fromkeys((*token_cost.missing_dimensions, 'incomplete_usage'))))
    flex_priced = tier == 'flex' and _supports_flex(token_cost.price)
    if tier is not None and tier not in ('standard', 'default', 'auto', 'batch') and not flex_priced:
        token_cost = replace(token_cost, status='partial',
            missing_dimensions=tuple(dict.fromkeys((*token_cost.missing_dimensions, 'service_tier'))))
    model = token_cost.canonical_model or row.get('model')
    return compose_call_cost(token_cost, catalog=loaded, provider=provider, channel=channel,
        region=region, retrieval=_record_retrieval_usage(row, model))
