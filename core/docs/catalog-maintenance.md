# Maintaining catalog data

This guide is for contributors who review sources, update catalog data, or
maintain refresh tooling. For application usage, start with the
[Core README](../README.md).

- [Catalog maintenance](#catalog-maintenance)
- [Pricing deltas](#pricing-deltas)
- [Source registry](#source-registry)
- [Pricing maintenance scripts](#pricing-maintenance-scripts)
- [Provider review and contributions](#provider-review-and-contributions)

## Catalog maintenance

The two manually maintained public datasets live under
`src/metergraph_core/data/`. Update `models.yaml` when model identity, display,
routing, or managed candidate policy changes. Update `prices.yaml` when rates
or billing rules change, or when a model's product-line `family` does. `sources.yaml` changes through reviewed pull requests,
as described in
[Provider review and contributions](#provider-review-and-contributions). Price
corrections close or add effective windows; they never rewrite historical
prices in place. Either data change increments
its own declared version and produces a patch release of `metergraph-core`.
Software and data versions remain separate because they have different
lifecycles.

The model registry version is an ISO date. When more than one registry ships on
the same date, append a positive revision such as `2026-09-25.2` so consumers
can distinguish the policies while pricing validation still uses that date.

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

Catalog operations change which models exist and what calls they match:

- `add_model` appends one complete model entry.
- `add_alias` appends one alias to an existing model. A spelling another
  model already holds is a conflict only while both aliases are in effect.
- `close_alias` sets `effective_to` on an open alias, named by
  `(canonical_id, provider, alias, channel)`; `expect` is the exact current
  alias. An alias with dates must cite its source, so a close carries
  `source_url` when the alias has none and must not when it has one.

A `close_alias`, `close_price`, `add_alias` and `add_price` together move a
spelling from a duplicate entry to the model it duplicates from a date. Calls
before that date keep the old model and price, calls from it resolve to the
surviving model, and the old entry stays for history with nothing in effect:

```yaml
  - op: close_alias
    key: {canonical_id: example-host/model-a-instruct, provider: openrouter,
          alias: example-host/model-a-instruct, channel: openrouter-api}
    expect: {provider: openrouter, alias: example-host/model-a-instruct, channel: openrouter-api}
    effective_to: "2026-10-20"
    source_url: "https://..."
  - op: close_price      # the duplicate's open openrouter-api row
  - op: add_alias
    key: {canonical_id: example/model-a, provider: openrouter,
          alias: example-host/model-a-instruct, channel: openrouter-api}
    expect: null
    alias: {provider: openrouter, alias: example-host/model-a-instruct,
            channel: openrouter-api, effective_from: "2026-10-20", source_url: "https://..."}
  - op: add_price        # the same rate on example/model-a from 2026-10-20
```

Gateway discovery matches a spelling only through aliases in effect on the
observation date, so it follows a moved spelling to its new model.

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
Anthropic and xAI start low-risk; each may change existing rates, in its
allowed dimensions, on existing rows.

OpenAI is reviewed discovery only. Its official parser still reads the live
rows, but the live pricing data no longer exposes the rendered column header
needed to verify the parser's positional columns, so its rows are evidence for
review, never automated changes.

Google (Gemini API) is reviewed discovery only. The registry points at the
pricing page's current home, so the allowlisted fetch needs no redirect, and
its official parser's reading of that page and a fingerprint need review
before it can propose changes again.

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
tests use one synthetic fixture per low-risk source, plus OpenAI's and
Google's, in `tests/fixtures/sources/`, reproducing only the structure its
parser reads;
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
