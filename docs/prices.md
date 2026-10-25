# Price catalog

`core/src/metergraph_core/data/prices.yaml` is a versioned, effective-dated model price catalog. It is the single public catalog, owned by the `metergraph-core` package and shared across MeterGraph systems. The server prices each call **at the call's timestamp**, so historical rows stay correct when prices change.

## Structure

```yaml
version: "2026-08-24"
currency: USD
pricing_verified_at: "2026-08-24"
models:
  - canonical_id: anthropic/claude-sonnet-5
    publisher: anthropic
    aliases:                      # (provider, alias) pairs the SDKs may report
      - provider: anthropic
        alias: claude-sonnet-latest
        channel: anthropic-api
        effective_from: "2026-06-30" # optional; omit both dates for a timeless alias
        effective_to: "2026-09-15"   # exclusive; preserve the row when the alias moves
        source_url: https://example.com/provider/model-history
    prices:
      - channel: anthropic-api
        region: global            # matched against MG_REGION, then '*', then 'global'
        effective_from: "2026-06-30"
        input_per_mtok: 2.00
        output_per_mtok: 10.00
        cache_read_per_mtok: 0.20
        cache_write_5m_per_mtok: 2.50
        batch_input_per_mtok: 1.00
        batch_output_per_mtok: 5.00
        rules: {}
        source_url: https://platform.claude.com/docs/en/about-claude/pricing
```

A row may also set `request_fee_per_1k_usd`: a fixed fee charged once per call on top of
tokens, in USD per 1,000 calls. Use it where a channel bills a model's built-in work per
request, such as a gateway pricing a search model's search per call. It appears in the
call's cost and in its `request_usd` component.

A row may also set `batch_cache_read_per_mtok`: the non-negative rate a batch call's
cache reads bill at, in USD per million tokens. Set it only where the provider
publishes a distinct batch cache rate; never derive it from another rate. A batch call
with cached tokens uses it in place of `cache_read_per_mtok`, and a standard call never
does. It may be set without `cache_read_per_mtok`, in which case a cached batch call can
be fully priced while a cached standard call is `partial`. A row without it keeps billing
batch cache reads at `cache_read_per_mtok`, unchanged; that fallback is not a claim that
the provider bills batch cache reads at the standard rate.

`rules` options:
- `input_includes_cache_read` — whether the provider reports cached tokens inside
  `input_tokens`, in which case cache reads are deducted from billable input. Assumed for
  the channels whose providers do this (`openai-api`, `google-api`, `google-vertex-ai`,
  `deepseek-api`, `xai-api`), so a row states it only to disagree with its channel: `false`
  where a provider stops counting them that way, `true` on a gateway serving one of them.
- `input_includes_cache_write: true` — provider reports cache-write tokens inside `input_tokens` (Vercel AI Gateway); cache writes are deducted before their cache rate is applied.
- `long_context: {threshold, input_multiplier, output_multiplier}` — surcharge above a prompt-size threshold (OpenAI GPT-5.6, Gemini Pro).
- `varies_by_provider: true` — a gateway's top-level rate is indicative; the provider it routes to may bill a different rate (Vercel AI Gateway). Informational: it does not change the computed cost.
- `uncaptured_fees: true` — provider charges fees tokens can't express; rows are marked `partial`.

`currency` is required and currently limited to `USD`.
`pricing_verified_at` is the ISO date when the catalog was last checked against
the linked provider sources. A price's own effective window still controls
historical selection.

Alias windows are independent from price windows. They answer which canonical
model a provider's moving name identified at the call timestamp. Reusing the
same `(provider, alias)` pair is allowed only when its windows do not overlap.
An alias without either effective field remains valid for all timestamps, which
preserves existing fixed aliases. A dated alias requires its own provider
`source_url`.

An alias may set `price_region` to limit where the id is valid. A regional pin
(`price_region: us-east-1`, or a list such as `[us-east-1, us-west-2]`) resolves
only when `MG_REGION` is a listed region, and then uses that region's price row.
In any other deployment the alias does not resolve and the call is `unpriced`
with `no_effective_alias`; a regional pin never imports its region's price into
another deployment. `price_region: global` is valid in every deployment and
always uses the `global` row, because the id itself names the global profile. A
pinned alias never falls back to `*` or to another region; a missing row is
`unpriced` with `no_effective_price`. On Bedrock, a `global.`-prefixed id that
has no alias of its own does not fall back onto a bare id pinned to a non-global
region.

## Cost status

Every stored call gets a `cost_status`:
- `priced` — fully priced from the catalog
- `partial` — priced, but something was missing (e.g. cache rate unavailable); the stored cost is a lower bound
- `unpriced` — unknown model, no alias effective at the call time or valid in
  the loaded region, or no effective price window;
  the dashboard surfaces these so you know to update the catalog

## Updating

1. Never edit a historical price or moving alias entry. Close its window with
   `effective_to` and add a new entry.
2. Treat verified successful calls as stronger availability evidence than a
   provider's declared retirement date, but only for the exact model id and
   serving channel observed. Extend that alias and price through the last
   verified call, then set `effective_to` to the earliest boundary after it
   supported by the observation's timestamp precision. Never leave a window
   open solely because an earlier call succeeded.
3. Include a `source_url` for every price.
4. Run `python scripts/check_prices.py` to compare the rows in effect today
   with the providers' own price pages (OpenAI, Anthropic, Google, DeepSeek,
   xAI, Perplexity, the Vercel AI Gateway list). Where a provider price is
   readable it decides; second opinions (LiteLLM's price map, models.dev;
   Portkey's pricing API with `--sources`) are compared only for rows no
   provider page covers. Each row is graded confirmed, disputed, update or
   unchecked, the models a provider lists that the catalog lacks are named,
   and the script exits 2 on an update signal. The `Price check` workflow
   runs it daily.

   The official sources it reads are registered, with their classification
   and dimensions, in `core/src/metergraph_core/data/sources.yaml`.
   `python scripts/refresh_prices.py` turns low-risk sources into a proposed
   pricing delta plus per-source observations and escalations, and
   `python scripts/discover_prices.py` gathers evidence and proposals for
   everything else. After reviewing an additive proposal,
   `python scripts/apply_price_delta.py --reviewed` can render it to a new
   catalog path without rewriting comments. A change still lands through a
   reviewed pull request. See `core/README.md` for the registry, the delta
   format, and the provider contribution path.
5. Open a PR; CI validates structure, dates, and window overlaps.
6. A catalog change updates the declared catalog `version` and ships as a patch release of `metergraph-core`; the software version and catalog version stay separate because code and price data have different lifecycles.
7. Self-hosters: mount an updated file with `MG_PRICES_PATH=/path/to/prices.yaml` — no rebuild needed.
8. If a change adds, removes, or renames a route used by
   `core/src/metergraph_core/data/models.yaml`, update that registry in the same
   release and run its cross-catalog validation. A price entry alone never adds
   a model to a managed candidate field.
