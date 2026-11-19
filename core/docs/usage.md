# Pricing and model reference

Start with the [Core README](../README.md) for installation, a first cost
calculation, and service lifecycle examples. This guide covers detailed behavior.

- [Loading catalogs](#loading-catalogs)
- [Catalog lifecycle](#catalog-lifecycle)
- [Pricing calls](#pricing-calls)
- [Model registry](#model-registry)
- [Billing evidence](#billing-evidence)
- [Model reliability](#model-reliability)
- [Package scope](#package-scope)
- [Public API](#public-api)

## Loading catalogs

`load_catalog()` loads the catalog bundled in the installed package. Pass an
explicit `path` for tests or a self-hosted catalog replacement. The returned
`LoadedCatalog` exposes the declared catalog `version`, the SHA-256
`content_hash` of the loaded bytes, the parsed `document`, and the immutable
`snapshot` used for pricing.

A catalog that is already parsed, for example a canonical JSON document read
from storage, loads with `load_catalog_document()`:

```python
from metergraph_core import load_catalog_document

catalog = load_catalog_document(document, region="global")
```

It validates exactly as `load_catalog()` does, because both build through the
same path, and a file and a document with the same content produce the same
catalog. The document is deep-copied, so later changes to the caller's mapping
never reach the loaded catalog. Its `content_hash` is the SHA-256 of the
document's canonical JSON (sorted keys, compact separators, ISO dates), so two
documents with the same content hash the same whatever their key order. That
differs from the raw-byte hash `load_catalog()` reports for a file.

## Catalog lifecycle

Use the [two lifecycle examples](../README.md#serving-real-time-requests) to choose
who owns scheduling. Share one `Catalogs` instance per source/region and treat
its returned values, including their documents, as read-only.

| Operation | Behavior |
|---|---|
| `current()` | Returns the prepared object without copying it. Without an active lifecycle, loads on first access and refreshes on demand after `refresh_seconds`. Concurrent readers retain the previous object during loading. |
| `prepare()` | Loads if needed and enables prepared-only reads. Starts no thread. |
| `refresh()` | Synchronously builds a replacement before publishing it. Raises on failure and preserves the previous value. |
| `start()` | Prepares data and starts one owned refresh thread. |
| `stop()` | Waits for owned refresh work, retains the data, and restores on-demand refresh. |

Repeated `prepare()`, `start()`, and `stop()` calls are safe. After `prepare()` or
`start()`, request reads never trigger loading or refresh. Background and
on-demand refresh failures retain the previous object and log a warning. An
initial load without a fallback must succeed before data can be served.

The default source is the bundled catalog. For a changing source, supply a
`loader` callable, such as `lambda: load_catalog("prices.yaml")`, or a function
that fetches a document and calls `load_catalog_document()`. A loader can build
a complete application pricing view before returning it. An optional `initial`
value provides a prepared fallback during the first source refresh.

Bound the loader's I/O: `stop()` waits for an owned refresh to finish.
Applications remain responsible for their own scheduled work. The explicit
`load_catalog()` and `load_catalog_document()` functions remain uncached loaders
for callers that need direct control of loading.

## Pricing calls

The examples below use a loaded catalog:

```python
from datetime import datetime, timezone
from metergraph_core import load_catalog

catalog = load_catalog(region="global")
```

### Direct lookup and usage cost

`lookup_price` returns the price a call is billed at, and `calculate_cost`
costs a call's usage at that price:

```python
from metergraph_core import ModelUsage

found = catalog.lookup_price(
    model="gpt-5.4-mini",
    provider="openai",
    at=datetime(2026, 8, 17, tzinfo=timezone.utc),
)
print(found.status, found.model_id, found.price.id, found.price.currency)

result = catalog.calculate_cost(
    model="gpt-5.4-mini",
    provider="openai",
    at=datetime(2026, 8, 17, tzinfo=timezone.utc),
    usage=ModelUsage(input_tokens=1000, output_tokens=200, cache_read_tokens=500),
)
print(result.status, result.model_id, result.cost, result.currency)
print(result.components.input, result.components.cache_read)
```

Both resolve exactly as `resolve_price` and `price` do, including
effective-date windows and the Bedrock fallback, and cost through the same
arithmetic. `model_id` is the resolved catalog ID, not the alias passed in, and
`price` is the applied catalog row. An explicit `channel` is authoritative;
without one, `provider` supplies its direct channel, and with neither the
result is `unpriced` with `missing_dimensions=("channel",)`.

Every `Price` states its `currency`, and `CostResult.currency` is the currency
of the applied price. The bundled catalog is USD. `cost` and
`components.input`, `output`, `cache_read`, `cache_write` and `request` are in
that currency; the older `cost_usd` and `*_usd` names return the same values.

`ModelUsage(batch=True)`, or `batch=True` on `price` and `cost`, bills at the
row's batch input and output rates. Its cache reads bill at
`Price.batch_cache_read_per_mtok` when the row has one, and otherwise at the
standard `cache_read_per_mtok`, as before. A standard call always uses the
standard cache-read rate. A row with only a batch cache rate fully prices a
cached batch call, while a cached standard call on it is `partial` with
`cache_read_rate_unavailable`.
Batch cache writes use `Price.batch_cache_write_5m_per_mtok` or
`Price.batch_cache_write_1h_per_mtok` for the matching TTL when stated;
otherwise each retains the row's standard cache-write rate.

A `ModelUsage` count left `None` is unknown, not zero. A missing input or
output count keeps the subtotal that is known, marks the result `partial`, and
lists `input_tokens` or `output_tokens` in `missing_dimensions`.
Google API Omni 1.1 and 1.2 calls with aggregate output tokens are `partial`
with `output_modality_unavailable` because text and video have different rates.

`region` selects a regional price and overrides the `region` passed to
`load_catalog`. When neither names a region:

- if every region bills the model the same, the result is `priced`;
- if a global price applies but some region bills differently, the global
  price is returned as `partial` with `missing_dimensions=("region",)`;
- if only specific regions price the model, the result is `unpriced` with
  `missing_dimensions=("region",)`.

A region the model is not sold in is `unpriced` with nothing missing.

A price closed with `effective_to` and not replaced is the last known price.
A call after it ended resolves to that price, stays `priced`, and carries
`price_ended` in `reasons` (on both `PriceResult` and `CostResult`). The
applied `price.effective_to` says when it ended. A closed price followed by a
later one never extends, so a gap before a scheduled price stays `unpriced`.

An `unpriced` result whose cause is a value you passed, such as an unknown
channel, provider or model, says why in `warnings`. Each `PricingWarning` has
a stable `code` and a `message` that may suggest the value you meant. Match on
`code`; the message text may change.

### Resolving a model and channel

Planning and evaluation systems that know the deployment channel before making
a call can resolve the exact effective price without emulating a provider
response:

```python
price = catalog.snapshot.resolve_price(
    model="openai/gpt-5.6-luna",
    channel="vercel-ai-gateway",
    at=datetime(2026, 8, 17, tzinfo=timezone.utc),
)
if price is not None:
    print(
        price.canonical_model,
        price.price.id,
        price.price.input_per_mtok,
        price.price.source_url,
    )
```

Resolution accepts canonical IDs and channel-scoped aliases, normalizes case
and surrounding whitespace, applies the configured region fallback and
effective-date windows, and returns `None` when no exact model/channel price
exists. It never substitutes a direct-provider price for a gateway price.

On Bedrock only (provider `bedrock` for `cost`, channel `aws-bedrock` or
`aws-bedrock-geo` for `resolve_price`), an id with no exact alias falls back to
the id without its model-version suffix (`anthropic.claude-sonnet-5-v1:0` to
`anthropic.claude-sonnet-5`) and then without a `global.` inference-profile
prefix, the latter only onto an `aws-bedrock` alias. Geographic prefixes such as
`us.` or `eu.` are never stripped, because they bill on `aws-bedrock-geo`.

`LoadedCatalog.currency` is currently always `USD`, and
`LoadedCatalog.pricing_verified_at` records when the bundled catalog was last
checked against its linked provider sources.

### Costing a provider response

`cost_from_response` reads the usage out of a response you already have and
costs it, so you do not extract token counts yourself:

```python
from openai import OpenAI
from metergraph_core import cost_from_response

response = OpenAI().chat.completions.create(model="gpt-5.4-mini", messages=messages)
result = cost_from_response("openai", response)
print(result.status, result.model_id, result.cost, result.currency)
```

It accepts the object a provider SDK returned or the same response as JSON:

| Provider API | Example `provider` |
|---|---|
| OpenAI Chat Completions, and gateways that return its shape | `openai`, `openrouter`, `deepseek` |
| OpenAI Responses | `openai` |
| Anthropic Messages | `anthropic` |
| Google Gemini `generateContent` | `google` |
| Bedrock Converse (pass `model=`, the response does not name one) | `bedrock` |

`provider` names who served the call and selects its direct channel unless
`channel` is given. The response format is detected from its fields; pass
`format=`, one of `RESPONSE_FORMATS`, for a response too sparse to detect.
`response_format(response)` returns the detected format, or `None`. Only the
model, usage counts, service tier and creation time are read; prompt and
output content never are.

Cached tokens are billed once whichever convention the response uses. OpenAI
and Gemini count cached tokens inside the input total, so the total is
restated in the convention of the price row that bills it, including a
gateway row for another publisher's model. Gemini thinking tokens bill as
output.

The price in effect when the response was created applies, from `created`,
`created_at` or `createTime`; `at=` overrides it, and a response with no
creation time is priced now. `catalog=` costs against a catalog you loaded;
the default is the bundled one.

A batch service tier bills at batch rates, with cache reads priced as for
`ModelUsage(batch=True)` above. Pass `service_tier=` when the
response does not state its tier, such as an OpenAI batch output line. Any other tier besides standard,
such as `priority` or `flex`, returns `partial` with `service_tier` in
`missing_dimensions`, since the catalog carries no tier rates. Cost a streamed
call from its final response or final usage chunk. Fees outside token usage,
such as web search, are not included.

### Complete call pricing

`price_call_from_response` combines token costs with supported retrieval fees.
`cost_from_response` remains token-only. Retrieval pricing supports Anthropic
`web_search` and `web_fetch` on `anthropic-api`; other observed scopes are unresolved.

```python
from metergraph_core import price_call_from_response

result = price_call_from_response(
    "anthropic",
    {"type": "message", "model": "claude-sonnet-5-5", "usage": {
        "input_tokens": 1000, "output_tokens": 100,
        "server_tool_use": {"web_search_requests": 2},
    }},
    at="2026-10-20T00:00:00Z",
)
assert str(result.cost_usd) == "0.02300000"
```

`cost_usd` is set only when the call is fully priced. `known_subtotal_usd`
contains exact components; `estimated_total_usd` separates token fallback
estimates when all retrieval fees are known. Missing charges are reported in
`missing_dimensions`. Counts must be nonnegative integers.

For normalized usage, use `price_call` with `ModelUsage` and `RetrievalUsage`.
To compose an existing token result from the same catalog, use `compose_call_cost`.
`resolve_billing` accepts complete-call results for gateway reconciliation.

### Complete costs from captured usage

`price_call_from_record()` accepts normalized SDK or relay usage, without reading
prompt or response content:

```python
from metergraph_core import price_call_from_record

result = price_call_from_record({
    "provider": "perplexity", "model": "perplexity/sonar",
    "retrieval_api": "agent", "search_type": "web", "web_search_calls": 3,
    "input_tokens": 1000, "output_tokens": 100, "ts": "2026-11-18T12:00:00Z",
})
print(result.status, result.cost_usd, result.retrieval_cost_usd)
```

Core selects the API channel, retrieval operation, and billing unit. Supported
record evidence includes OpenAI search counts, Anthropic server-tool usage,
Google grounding queries, and Perplexity Agent standard/fast invocation counts.
Older Gemini grounded prompts do not inherit newer per-query rates. An unknown
rate, count, search type, or processing tier stays partial or unpriced.
`search_units` and `retrieval_cost_usd` expose the resolved components for storage.
`known_subtotal_usd` remains separate from a complete total.

For failed calls, missing usage is not evidence of a free request. Anthropic
Messages HTTP rejections with no observed usage are priced at zero with reason
`provider_failure_not_billable`. This rule excludes batch operations and unknown
endpoints: a failed result download can follow already-billed inference. Failed
streams or `usage_complete: false` remain partial; any observed-usage estimate is
separate from the final charge. Qualified gateway billing evidence retains its
existing precedence.

A receiver that already resolved tokens against the same catalog may pass
`token_cost=`. Explicit API identity can select a different token channel; other
resolved token results are preserved. Qualified gateway totals can still be
reconciled separately with `resolve_billing()`.

## Model registry

```python
from metergraph_core import load_model_registry

models = load_model_registry()
for candidate in models.candidates("gateway"):
    print(candidate.id, candidate.display_name, candidate.pricing_channel)
```

`load_model_registry()` reads the packaged `models.yaml`. Execution profiles
describe every route the pipeline can run, while offer groups describe the
smaller ordered fields a product exposes for configured credentials. Candidate
IDs may intentionally repeat across execution profiles when gateway and
Bedrock use different physical routes; each route therefore also has a unique
internal `key`.

A model can omit `routes` or use an empty list to record identity without
execution support. It remains available through `registry.model(canonical_id)`,
but contributes no executable routes or offer-group candidates.

A model entry may record `released`, the publisher's general-availability
date as an ISO date. `models.release_date(canonical_id, load_catalog())`
returns that date with basis `released`, or, when none is recorded, the
earliest `effective_from` of the model's catalog prices with basis
`first_price`, so a consumer can tell a launch date from a pricing date
without computing the fallback itself.

Use `validate_model_registry(models, load_catalog())` to verify that every
route's provider-facing model ID is declared on its exact pricing channel,
resolves to its canonical model, and has an active price on the registry
version date. The parser also rejects unknown route providers, provider/channel
mismatches on any route, mismatched offer-group credentials/providers, and duplicate physical routes. Pricing
remains exclusively in `prices.yaml`; being priceable does not automatically
make a model an approved analysis candidate.

### Model families

A model entry in `prices.yaml` may name the stable product line it belongs to:

```yaml
  - canonical_id: anthropic/claude-sonnet-5
    publisher: anthropic
    family: {id: anthropic/claude-sonnet, display_name: Claude Sonnet}
```

```python
family = catalog.model_family("anthropic/claude-sonnet-5")
print(family.id, family.display_name)  # anthropic/claude-sonnet Claude Sonnet

catalog.family_for_model("anthropic", "claude-sonnet-4-6")  # same family
```

`model_family(canonical_id)` returns the frozen `ModelFamily` the catalog
declares for a canonical ID, or `None`. `LoadedCatalog.families` maps each
declared canonical ID to its family. `family_for_model(provider, model)` takes
a raw provider model name, such as one recorded before its model was known,
and returns a family only when every model the catalog binds that name to
under that provider or any spelling of it (`google-genai` for `google`, `aws`
for `bedrock`), as an alias at any date or as its canonical ID, declares the
same family. A name that has moved between product lines, or that any
unlabeled model holds, returns `None`. It applies no Bedrock suffix fallback.

A family is display and grouping metadata. It does not say that one model
succeeds, upgrades or can replace another, and it never changes alias
resolution, canonical IDs, routes or prices. Families are curated explicitly
and never inferred from model names, so a model without `family`, including
every model in an older catalog, has no known family.

`prices.yaml` is the only source of family metadata; `models.yaml` does not
repeat it. `parse_catalog`, both loaders and `add_model` deltas reject a
`family` that is not a mapping of exactly `id` and `display_name`; a
`display_name` that is blank, spans lines, or has leading or trailing
whitespace (it is rejected, never trimmed); an `id` that is not lowercase
`<publisher>/<line>` (`[a-z0-9._-]` slugs) under the model's own `publisher`;
a model with a family whose canonical ID is not under that publisher; and one
`id` given two different display names. Adding or correcting a family is a
model metadata change, not a price change: it increments the catalog
`version` but not `pricing_verified_at`.

### Training policy

`models.effective_training_policy(route_key)` says whether the route's
serving provider may train on customer inputs and outputs under its default
terms. It returns an `EffectiveTrainingPolicy` holding the `TrainingPolicy`,
the level it came from (`origin`) and the registry `version` that recorded
it. An unknown route key raises `KeyError`.

```python
effective = models.effective_training_policy("default:openai/gpt-5.6-sol")
print(effective.policy.training_on_customer_data, effective.origin)
```

`training_on_customer_data` is one of:

- `prohibited`: the default applicable terms exclude training on customer
  inputs and outputs.
- `conditional`: training is excluded only when `conditions` hold, such as a
  paid tier or an opt-out.
- `permitted`: the terms allow training.
- `unknown`: no explicit promise was found. A missing promise is `unknown`,
  not `permitted`.

Every value except `unknown` cites an HTTPS `source`. Only `conditional` has
`conditions`. No value verifies an account’s configuration, opt-ins, or
separate agreements. The registry version and Git history record when a
policy changed.

The document records policies at three levels, keyed by the route's serving
`provider` and `pricing_channel`:

```yaml
provider_defaults:
  openai:
    openai-api:
      training_policy:
        training_on_customer_data: prohibited
        source: https://example.com/terms
models:
- canonical_id: example/model
  training_policy_overrides:     # this model on one provider and channel
    vercel:
      vercel-ai-gateway:
        training_policy: {training_on_customer_data: unknown}
  routes:
  - key: default:example/model
    training_policy:             # this route only
      training_on_customer_data: unknown
```

The route's own `training_policy` wins. Otherwise the model's override for
the route's provider and channel applies, then that pair's provider default,
then `unknown` with origin `unspecified`. The first policy found replaces the
others whole, so an explicit `unknown` stops inheritance. The publisher is
never used. A gateway or cloud host such as `vercel` or `bedrock` follows its
own terms, not those of the model's publisher. The parser rejects unknown
policy fields, a provider/channel pair that no route could use, and a model
override for a pair the model has no route on. Documents without policies
still load, and every route resolves to `unknown`.

The bundled registry records a default for every serving provider. The YAML
is the source of truth for current policy values, sources and conditions:

```python
for provider, channels in models.training_policy_defaults.items():
    for channel, policy in channels.items():
        print(provider, channel, policy.training_on_customer_data, policy.source)
```

Adding a model normally requires no policy entry: it inherits the serving
provider's default. Add a model or route override only when its terms differ.
To update a provider policy, edit that one YAML object and increment the
existing registry version; source URLs and values are not duplicated in tests
or a documentation table. Changes still require review against the linked
official terms. There is no network lookup, scheduled refresher, extra
dependency or per-policy date to maintain. Policy lookup is explicit and does
not run during pricing or candidate selection.

## Billing evidence

Servers can pass content-blind, already-extracted gateway fields through the
shared trust boundary and combine them with a catalog result:

```python
from metergraph_core import normalize_gateway_evidence, resolve_billing

evidence = normalize_gateway_evidence({
    "gateway": "openrouter",
    "endpoint": "chat.completions",
    "reported_cost_usd": "0.00482",
    "reported_cost_source": "openrouter.usage.cost",
})
decision = resolve_billing(result, evidence)
print(decision.cost_usd, decision.cost_provenance)
```

The initial qualified contract is OpenRouter Chat Completions. A finite,
non-negative `openrouter.usage.cost` value, including zero, takes precedence
over a catalog estimate. The decision retains both values and never adds the
separately reported upstream inference cost to the OpenRouter account charge.
Unsupported or malformed evidence falls back to the catalog result.

The billing module validates only gateway, endpoint, fixed source names, and
decimal cost values. It does not inspect provider response content or own SDK
capture, HTTP, timestamps, trace context, persistence, or tenant behavior.

## Model reliability

SDK and import adapters collect content-free provider facts; Core owns the rules
that classify them. Hosted and self-hosted receivers use `classify_recorded_call`
to derive stored availability and refusal outcomes from those facts.

`classify_model_reliability` inspects structured native response/error facts and
returns a frozen `ModelReliability` with nullable `http_status_code`,
`availability` (`available`, `unavailable`, `excluded`) and `refused` fields.
Explicit policy refusals are available responses; rate limits and provider
failures are unavailable. Sparse records remain unknown, and response prose is
never used to infer a refusal. HTTP status is separate from telemetry span status.

```python
from metergraph_core import classify_model_reliability

result = classify_model_reliability(provider="openai", http_status_code=429)
assert result.availability == "unavailable"
assert result.refused is None
```

The optional `attributes` argument accepts existing OpenTelemetry and
OpenInference facts: `http.response.status_code` (or legacy `http.status_code`),
`error.type`, `gen_ai.response.finish_reasons`, and `llm.finish_reason`.
The classifier inspects every reported finish reason; `refused=True` means at
least one returned choice/candidate was explicitly refused. Normal finish
reasons alone do not establish `refused=False` when response evidence is absent.

`http_status_code` represents a transport fact. `availability` and `refused` are
Metergraph-derived projections, not new OpenTelemetry status conventions.
Availability describes recorded model calls, which can encompass SDK retries;
it does not measure every physical attempt or provider-wide uptime. Existing
span errors and raw finish reasons keep their separate diagnostic meaning.

Conventions: [OpenTelemetry GenAI](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/client-inference.md),
[HTTP](https://opentelemetry.io/docs/specs/semconv/http/http-spans/), and
[OpenInference](https://github.com/Arize-ai/openinference/blob/main/spec/semantic_conventions.md).

## Package scope

`metergraph-core` is the reusable model registry, price catalog, and
deterministic billing engine for
[Metergraph](https://github.com/metergraph/metergraph), which attributes LLM
cost to the function, agent, and workflow that made each call.

It owns two separate public datasets. `models.yaml` records
canonical model identity, customer-facing names, executable provider routes,
and the ordered candidate fields MeterGraph products offer. `prices.yaml`
records effective-dated rates and billing rules. Core parses and validates both
datasets, resolves provider aliases and routes, calculates deterministic costs,
and selects a qualified gateway-reported charge or catalog estimate as the
effective call cost. A third dataset, `sources.yaml`, registers the official
pricing sources that keep `prices.yaml` current; see
[Source registry](catalog-maintenance.md#source-registry).

It does not own HTTP routes, database access, migrations, authentication,
tenancy, ingest, dashboard code, or any hosted-only concern, and it never reads
server environment variables.

## Public API

```python
from metergraph_core import (
    Alias,
    BillingDecision,
    CatalogDeltaConflict,
    CatalogDeltaError,
    CatalogError,
    CatalogSnapshot,
    Catalogs,
    CostResult,
    GatewayBillingEvidence,
    LoadedCatalog,
    ModelFamily,
    ModelRegistry,
    ModelRoute,
    ModelUsage,
    Price,
    PriceResult,
    PriceRowKey,
    PriceSource,
    RESPONSE_FORMATS,
    RefreshResult,
    ResolvedPrice,
    SourceRegistry,
    apply_catalog_delta,
    cost_from_response,
    discovery_evidence,
    extract,
    load_catalog,
    load_catalog_document,
    load_model_registry,
    load_source_registry,
    normalize_gateway_evidence,
    parse_catalog,
    parse_catalog_delta,
    plan_low_risk_refresh,
    resolve_billing,
    response_format,
    structural_fingerprint,
    validate_model_registry,
    validate_source_registry,
)
```
