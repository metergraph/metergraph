# metergraph-core

Calculate the cost of an LLM call from its model, provider, tokens, region, and
date. `metergraph-core` ships an open, effective-dated model pricing catalog and
a deterministic Python cost engine.

## Install

```bash
python -m pip install metergraph-core
```

## Usage

```python
from datetime import datetime, timezone

from metergraph_core import load_catalog

catalog = load_catalog(region="global")
result = catalog.snapshot.cost(
    provider="openai",
    model="gpt-5.4-mini",
    at=datetime(2026, 8, 17, tzinfo=timezone.utc),
    input_tokens=1000,
    output_tokens=200,
)
print(result.canonical_model, result.price_id, result.cost_usd, result.status)
```

The result names the canonical model and the catalog price row it was billed
at, so every cost traces back to an effective-dated rate.

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
[Source registry](#source-registry).

It does not own HTTP routes, database access, migrations, authentication,
tenancy, ingest, dashboard code, or any hosted-only concern, and it never reads
server environment variables.

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
Batch cache writes use `Price.batch_cache_write_5m_per_mtok` when stated;
otherwise they retain the row's standard cache-write rate.

A `ModelUsage` count left `None` is unknown, not zero. A missing input or
output count keeps the subtotal that is known, marks the result `partial`, and
lists `input_tokens` or `output_tokens` in `missing_dimensions`.

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

Each offer group is reachable through one credential and one route provider:

| Offer group | Credential | Route provider | Pricing channel |
| --- | --- | --- | --- |
| `gateway` | `AI_GATEWAY_API_KEY` | `vercel` | `vercel-ai-gateway` |
| `fireworks` | `FIREWORKS_API_KEY` | `fireworks` | `fireworks-api` |
| `openai-direct` | `OPENAI_API_KEY` | `openai` | `openai-api` |
| `anthropic-direct` | `ANTHROPIC_API_KEY` | `anthropic` | `anthropic-api` |
| `moonshot-direct` | `MOONSHOT_API_KEY` | `moonshotai` | `moonshot-api` |
| `bedrock` | none (AWS credentials) | `bedrock` | `aws-bedrock` |

A route's `provider` is the catalog's normalized provider name, so Moonshot
routes use `moonshotai`, not the `moonshot` input alias. Direct route IDs take
the form `<provider>:<provider model ID>`, for example
`moonshotai:kimi-k3.1-pro` with `model_id` `kimi-k3.1-pro`; Moonshot's API is
served at `https://api.moonshot.ai/v1`.

Adding a route provider is a minor release, even though it only adds data: a
client that executes every default-profile route and checks providers against
a fixed list would reject the new one. `metergraph-core` 0.8.0 adds the
`moonshotai` provider and `moonshot-direct` offer group, and puts Moonshot
routes in the default profile. A client that executes default routes must
support the `moonshotai` provider, or skip providers it does not support,
before it allows `metergraph-core>=0.8`. Clients pinned below 0.8 keep the
0.7 provider set. Clients that only price calls are unaffected.

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

## Pricing deltas

A pricing delta proposes changes to a catalog's price rows. It is a change
artifact to review and apply, never a catalog to serve.

```yaml
schema_version: 2
kind: pricing_delta
provenance:
  observations: [openai-api-pricing@2026-10-20]
  evidence: [{source_id: openai-api-pricing, url: "https://...", fingerprint: "sha256:..."}]
operations:
  - op: replace_price
    key: {canonical_id: openai/gpt-x, channel: openai-api, region: global, effective_from: "2026-01-01"}
    expect: {...the full current row...}
    price: {channel: openai-api, region: global, effective_from: "2026-10-20",
            input_per_mtok: "2.50", output_per_mtok: "10", source_url: "https://..."}
```

Each operation names its row by `(canonical_id, channel, region,
effective_from)` on a model the catalog already has:

- `add_price` appends a row; `expect` is `null` because no such row may exist.
  It may not start on the same UTC date as another row for that model,
  channel and region, even one it does not overlap, because a price id names
  its row by the UTC date of its start.
- `replace_price` closes the open row at the new row's `effective_from` and
  appends the new row; `expect` is the exact current row. The new row must
  start on a later UTC date than the row it replaces, for the same reason.
- `close_price` sets `effective_to` on the open row without a replacement.

A price id ends with the UTC date of its row's start. If two rows for one
model, channel and region would still share an id (possible in a hand-edited
catalog), each that does not start at UTC midnight also names its UTC start
time, such as `2025-01-02T10:00:00Z`.

New rates are decimal strings. `apply_catalog_delta(document, delta)` returns a
new document and never changes either input. It checks every operation's
precondition first; if any fails, `CatalogDeltaConflict.conflicts` names every
failing operation and nothing is applied. History is never rewritten, the
top-level `version` and `pricing_verified_at` are left for the publisher to
update, and the complete result is validated with `load_catalog_document()`.
Because preconditions cover only the rows a delta changes, an approved delta
still applies after unrelated catalog changes. `parse_catalog_delta()` validates
a delta without applying it.

## Source registry

`src/metergraph_core/data/sources.yaml` lists the official pricing sources the
refresh tools read. `load_source_registry()` loads it, and
`validate_source_registry(registry, catalog)` checks its channels against a
catalog. Each entry records:

- the provider and pricing `channel`, and whether the catalog carries that
  channel yet (`catalog_channel`);
- the official pricing `url`, the only address a refresh tool may fetch for it;
- the `parser` that reads it;
- the price dimensions it represents (`dimensions`): catalog price columns, or
  GPU-hour and license units;
- the page's expected structural `fingerprint`;
- `low_risk`: whether automated refreshes may use it (`eligible`), and the
  `dimensions` an eligible source may change.

To inspect the registered source for a provider:

```python
from metergraph_core import load_source_registry

for source in load_source_registry().sources:
    if source.provider == "anthropic":
        print(source.id, source.url, source.parser, source.dimensions, source.low_risk)
```

Seeding a source does not make it low-risk. Low-risk means a proven official
parser whose live structural fingerprint has been reviewed and recorded.
Anthropic, Google (Gemini API) and xAI start low-risk; each may change
existing rates, in its allowed dimensions, on existing rows.

OpenAI is reviewed discovery only. Its official parser still reads the live
rows, but the live pricing data no longer exposes the rendered column header
needed to verify the parser's positional columns, so its rows are evidence for
review, never automated changes.

Every other source is reviewed discovery only. It is read with a generic
parser, `html-tables` for any page's tables or `gpu-license-prose` for GPU or
license pricing stated in prose, which records normalized evidence and never
produces price rows. A generic parser does not mean the page's layout is
supported. DeepSeek, Mistral and Perplexity have official parsers (used by
`scripts/check_prices.py`) whose live pages do not currently parse cleanly to
the catalog; a source is promoted only through a reviewed registry change once
its parser and fingerprint are reviewed. NVIDIA's NIM production pricing is per
GPU or license, so it is recorded with GPU-hour and license dimensions and never
as token pricing.

### Extractors

`extract(parser_id, content)` reads source content the caller supplies and
returns the rates it finds, mapped to catalog price columns, or evidence for a
generic discovery parser. `structural_fingerprint(parser_id, content)` summarizes
only the labels, headers and key paths that parser relies on, so a price change
or a newly listed model keeps the fingerprint and a format change alters it.
Extractors never fetch. The package adds no browser or LLM dependency.

## Pricing maintenance scripts

The repository scripts run the registry against live sources without credentials
or GitHub access.

The fetch-capable scripts never emit, store or redistribute raw page content. Their
reports and evidence files hold only structural fingerprints, normalized
extracted values (row labels as slugs, parsed amounts, numeric candidates),
candidates, operations and short non-verbatim diagnostics. No output may name
an input the run read.

`scripts/refresh_prices.py` takes the registry and a current catalog document.
For each low-risk source it fetches the registered URL, checks the page's
fingerprint, extracts prices, checks invariants, and prints a JSON
report: a pricing delta for the changes that pass, one observation per source
(including sources with no change), and every escalation. New models, new
regions, new dimensions, rows a page no longer lists, ambiguous parses, format
changes, non-positive or missing rates, changes outside the allowed
dimensions, duplicated model entries, changes to a row that already has a
scheduled end, a row more than one selected source proposes (whether or not
they agree), invalid current catalog rates, and backdated or same-day
effective dates escalate. A change takes effect on the UTC observation date. It exits 0 with no change, 2 with a delta or
an escalation, and 1 on error.

`scripts/discover_prices.py` scans sources that are not low-risk, the sources
behind a refresh report's escalations (`--escalations`), and new-source content
from a file or stdin (`--new-source`, never a URL). It prints evidence for each:
candidate and prior values, the reason for review, normalized page facts, and a
proposed delta for supported catalog changes. An optional adapter
(`--llm-adapter module:callable`) is called as
`propose(source, content, current_rows)` and returns delta operations. Its
output is kept as review evidence only after it passes schema validation and
encodes as strict JSON. Adapter failures and rejected items are reported with
stable codes, never with the adapter's own text. Everything discovery emits
requires review.

For structured gateway feeds, discovery also proposes `add_model` and
`add_alias` operations when an exact model identity is safe to extend. Free,
conditional, tiered, regional and unsupported-charge rows remain exclusions.
After review, `scripts/apply_price_delta.py --reviewed` writes additive operations
to a new catalog path while preserving existing comments. It rejects in-place
writes and history-changing operations, then verifies the rendered catalog
against the pure delta result.

Apply overlapping gateway sources sequentially. Run discovery for the next
source against the prior candidate so a shared new model becomes an alias and
price on one canonical entry instead of a conflicting second `add_model`.

The fetch-capable scripts read `<source-id>.*` files from `--content-dir` instead
of fetching. The
tests use one synthetic fixture per low-risk source, plus OpenAI's, in
`tests/fixtures/sources/`, reproducing only the structure its parser reads;
the generic discovery parsers are tested on small inline synthetic pages.

## Provider review and contributions

Model providers, new and existing, can check and improve how MeterGraph prices
their models:

1. **Review your entry.** Find your provider in `sources.yaml` (or run the
   snippet above) and check the registered pricing URL, its parser, and the
   dimensions it represents. Then check your models' current rows in
   `prices.yaml`.
2. **Propose a correction or a new source.** Edit your entry or add a new
   provider with its official pricing URL. A new source starts non-low-risk.
3. **Add the extractor, fixture and tests when proposing low-risk status.** A
   discovery-only source needs no fixture of its own. A source proposed for
   low-risk status needs an extractor in `extractors.py` and a minimal synthetic
   fixture in `tests/fixtures/sources/` that reproduces only the structure the
   parser reads, never a saved copy of the page, with tests.
4. **Open a pull request** with the official-source evidence and an effective
   date for each pricing change. Price changes close the old row's window and
   add a new row; they never edit history.

Provider-submitted changes go through the same validation and review as any
other catalog change. A provider cannot mark its own source low-risk or publish
directly to the live catalog: low-risk status changes only through a reviewed
registry change, and publication is done by MeterGraph's release process.

## Public API

```python
from metergraph_core import (
    Alias,
    BillingDecision,
    CatalogDeltaConflict,
    CatalogDeltaError,
    CatalogError,
    CatalogSnapshot,
    CostResult,
    GatewayBillingEvidence,
    LoadedCatalog,
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

## Catalog maintenance

The two manually maintained public datasets live under
`src/metergraph_core/data/`. Update `models.yaml` when model identity, display,
routing, or managed candidate policy changes. Update `prices.yaml` when rates
or billing rules change. `sources.yaml` changes through reviewed pull requests,
as described in
[Provider review and contributions](#provider-review-and-contributions). Price
corrections close or add effective windows; they never rewrite historical
prices in place. Either data change increments
its own declared version and produces a patch release of `metergraph-core`,
unless it adds a route provider or offer group, which is a minor release (see
[Model registry](#model-registry)).
Software and data versions remain separate because they have different
lifecycles.

The model registry version is an ISO date. When more than one registry ships on
the same date, append a positive revision such as `2026-09-25.2` so consumers
can distinguish the policies while pricing validation still uses that date.
