# metergraph-core

Calculate the cost of an LLM call from its model, provider, tokens, region, and
date. `metergraph-core` ships an open, effective-dated model pricing catalog and
a deterministic Python cost engine.

## Install

```bash
python -m pip install metergraph-core
```

## Usage

Use `calculate_cost()` when you already have token counts:

```python
from metergraph_core import ModelUsage, load_catalog

catalog = load_catalog(region="global")
result = catalog.calculate_cost(
    model="gpt-5.4-mini",
    provider="openai",
    at="2026-08-17T00:00:00Z",
    usage=ModelUsage(input_tokens=1000, output_tokens=200),
)
print(result.status, result.cost, result.currency)
```

The result identifies the applied model and price. Check `status` before using
its cost as a complete total: missing usage or unsupported rates can make a
result `partial` or `unpriced`.

Azure Foundry pricing requires the deployment's underlying model version, SKU,
and Azure resource region. Verified GPT-6 Astra, Sol, and Luna Global Standard
deployments use a versioned model such as `model="gpt-6-luna@2026-09-22"`,
`channel="azure-foundry-global-standard"`, and `region="eastus"` with
`calculate_cost()`. A generic `provider="azure"` call stays unpriced because its
deployment type is unknown. Batch and cache-write usage remain partial where
the catalog lacks a matching rate.

Served OpenAI `priority`/`fast` and Vertex `priority` calls use model-specific
catalog rates where verified. Priority coverage starts on November 22, 2026.
Requested Priority that falls back to standard uses standard pricing. Search
fees are priced separately from the token tier.

## Choose the API for your input

| You have… | Start with… |
|---|---|
| Token counts and a model | `catalog.calculate_cost(..., usage=ModelUsage(...))` |
| A provider SDK response | `price_call_from_response(provider, response)` for token costs and supported retrieval fees |
| A normalized SDK or relay record | `price_call_from_record(record)` |
| Request handlers that need ready-to-use pricing data | `Catalogs`, with one instance shared across handlers |
| Model identities, executable routes, or candidate lists | `load_model_registry()` |

For price lookup, token-only response pricing, missing-data rules, gateway
billing evidence, and model metadata, see the
[pricing and model reference](https://github.com/metergraph/metergraph/blob/main/core/docs/usage.md).

## Serving real-time requests

Prepare catalogs once before accepting traffic. Request handlers then call
`catalogs.current()` and use the same pricing methods shown above. Choose one
of the following lifecycle options.

### Let the catalog own its refresh thread

```python
from metergraph_core import Catalogs

catalogs = Catalogs(region="global", refresh_seconds=300)
catalogs.start()
try:
    print(catalogs.current().version)
finally:
    catalogs.stop()
```

### Use your own scheduler or threads

`prepare()` and `refresh()` are synchronous and create no threads. Prepare
before serving; run refreshes outside request handlers. This example schedules
one refresh on an executor owned by the application:

```python
from concurrent.futures import ThreadPoolExecutor
from metergraph_core import Catalogs

catalogs = Catalogs(region="global")
catalogs.prepare()
try:
    with ThreadPoolExecutor(max_workers=1) as executor:
        refresh = executor.submit(catalogs.refresh)
        print(catalogs.current().version)
        refresh.result()
finally:
    catalogs.stop()
```

Both options enable prepared-only reads. Without either option, `current()`
loads and refreshes on demand. The default source is the bundled catalog;
custom loaders can supply changing data. See the
[lifecycle contract](https://github.com/metergraph/metergraph/blob/main/core/docs/usage.md#catalog-lifecycle)
for refresh failures, custom sources, read-only values, and shutdown behavior.

## Reference and contributions

- [Pricing and model reference](https://github.com/metergraph/metergraph/blob/main/core/docs/usage.md):
  loading, regional and dated prices, provider responses, retrieval fees,
  model families, routes, training policy, billing evidence, and reliability.
- [Catalog maintenance](https://github.com/metergraph/metergraph/blob/main/core/docs/catalog-maintenance.md):
  data files and versions, official sources, reviewed deltas, refresh tooling,
  and provider contributions.
- [Public API exports](https://github.com/metergraph/metergraph/blob/main/core/docs/usage.md#public-api).

Core contains the shared datasets and pricing rules for
[Metergraph](https://github.com/metergraph/metergraph). Core does not own HTTP routes, database access,
authentication, ingestion, or dashboards; these belong to the applications using it.
