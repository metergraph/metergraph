# Metergraph

[![CI](https://github.com/metergraph/metergraph/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/metergraph/metergraph/actions/workflows/ci.yml)
[![metergraph-core on PyPI](https://img.shields.io/pypi/v/metergraph-core?label=metergraph-core)](https://pypi.org/project/metergraph-core/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](core/pyproject.toml)
[![License: Apache-2.0](https://img.shields.io/github/license/metergraph/metergraph)](LICENSE)

**Optimize agentic AI for cost and quality.**

Metergraph helps companies reduce LLM cost and improve AI quality. It starts
with what most teams cannot see today: which function, agent, and workflow in
your code is spending money on LLM calls, and how much.

This repository is the open-source foundation. It gives you a current model
and pricing catalog, a content-blind self-hosted server, and a dashboard for
understanding LLM usage and cost. Run it with one `docker compose up`.

![Metergraph self-hosted dashboard with seeded demo traffic, showing LLM cost by function, route, and model](docs/images/dashboard-overview.png)

## Why attribution by function, agent, and workflow

Provider bills tell you what you spent per model. They do not tell you which
part of your product spent it. Metergraph attributes every LLM call to the code
that made it:

- **Function**: the application function that issued the call, such as
  `billing:summarize_invoice`, found automatically in Python or named
  explicitly with `track()`.
- **Agent**: a stable workload name, such as `ticket-classifier`, set with
  `route()` around an agent or product surface.
- **Workflow**: every call that belongs to one user-visible run, grouped with
  `trace()`, so a multi-step agent loop is costed as a whole.

Each call is priced on the server from an effective-dated catalog, so history
reprices correctly when providers change rates. You can see where cost
concentrates before you decide what to change.

## Open source and paid Metergraph

| | Open source (this repository) | Paid Metergraph |
|---|---|---|
| Model and pricing catalog | Current, effective-dated catalog in [`metergraph-core`](core) | Included |
| Cost attribution by function, agent, and workflow | Yes | Yes |
| Storage | Usage metadata only; the server discards prompt and completion content | Content-aware, with redaction and per-route opt-outs |
| Dashboard | Basic self-hosted usage and cost dashboard | Evaluation, experimentation, and optimization workspace |
| Experiments | | Compare models, prompts, search stacks, batching, and prompt caching against cost and quality |
| Implementation help | | MetergraphBot opens reviewable pull requests for changes that won an experiment |
| Deployment | Deploy and operate it yourself, Apache-2.0 | Metergraph Cloud, your VPC, or a paid self-hosted deployment |

Open source answers "where is my LLM spend going?" Paid Metergraph answers
"what should I change, and which version works best?" It runs experiments on
your real workloads, measures each option on both cost and quality, and helps
you ship the winner. [MetergraphBot](https://github.com/apps/metergraph) turns
approved recommendations into pull requests you review; it never merges
changes automatically.

To learn about paid Metergraph, visit [metergraph.dev](https://www.metergraph.dev/).

## Quickstart

### 1. Run the server and dashboard

```bash
git clone https://github.com/metergraph/metergraph && cd metergraph
MG_TOKENS=dev-token docker compose up
```

This starts Postgres and the server (API and dashboard) on port 8787. Open
http://localhost:8787 and enter the same token.

### 2. See it with demo traffic

No provider API key is needed. The demo wraps local OpenAI-, Anthropic-, and
Google-shaped clients with the published SDK and sends calls through the same
path a real application uses:

```bash
python -m pip install 'metergraph>=0.4,<1'
MG_TOKEN=dev-token python scripts/seed_demo.py
```

Demo calls use the `demo` environment, so you can filter them out later.

### 3. Instrument your application

```bash
pip install metergraph
export METERGRAPH_INGEST_URL=http://localhost:8787
export METERGRAPH_APP_TOKEN=dev-token
```

```python
import metergraph
from openai import OpenAI

metergraph.init()
client = metergraph.wrap(OpenAI())

@metergraph.track
def summarize_invoice(invoice):
    return client.chat.completions.create(model="gpt-5.6-luna", messages=[...])

with metergraph.trace("month-end-close"):
    with metergraph.route("invoice-summarizer"):
        summarize_invoice(invoice)
```

Every call is attributed to `yourapp.billing:summarize_invoice`, grouped under
the `invoice-summarizer` route and the `month-end-close` trace, and priced from
the [public catalog](core/src/metergraph_core/data/prices.yaml).

The SDKs for Python and TypeScript, with OpenAI, Anthropic, Gemini, Vercel AI
SDK, and OpenTelemetry support, live in
[metergraph/metergraphsdk](https://github.com/metergraph/metergraphsdk).

## How attribution works

- **Python**: automatic. The SDK walks the stack at call time and attributes
  each LLM call to the nearest function under your app root. Add
  `@metergraph.track` (or `@metergraph.track("billing.summarize")`) for
  explicit, stable names.
- **TypeScript**: use `track(fn)` or `track("billing.summarize", fn)`. This
  stays reliable across bundlers and minifiers, where stack parsing does not.
  Best-effort stack attribution is the fallback.
- `metergraph.route("ticket-classifier")` names an agent or product surface,
  independent of which functions it calls.
- `metergraph.trace("checkout")` groups every call in one workflow run.

The dashboard breaks cost down by function, route, model, and time. The usage
API also groups by module, provider, day, and hour.

## Privacy: what a row contains, and what it never contains

Stored: timestamp, function and module, route, trace and session IDs,
provider, model, input/output/cache-read/cache-write/reasoning token counts,
latency, time to first token, status, stream and batch flags, a content-free
structural template hash, tool-call **names**, tags, environment, and SDK
version.

Never stored by this server: prompts, completions, tool-call arguments, or tool
results. Rows are projected through a column allowlist at ingest, so content
fields cannot reach the database even if an SDK sends them. To keep content
inside your process as well, set `METERGRAPH_CAPTURE_TEXT=0` or pass
`capture_text=False` to `metergraph.init()`. See
[docs/privacy.md](docs/privacy.md).

## Where SDK data goes

With `METERGRAPH_INGEST_URL` set, the SDK sends telemetry to your self-hosted
server. Without it, the SDK sends telemetry to Metergraph Cloud. Without a
`METERGRAPH_APP_TOKEN`, capture is disabled and nothing is sent.

Metergraph Cloud captures scrubbed request and response content by default,
because replayable content is what lets paid Metergraph evaluate quality. The
SDK supports custom redaction and global or per-route opt-outs. This
open-source server discards content regardless of SDK configuration.

## Packages

| Package | Where | What |
|---|---|---|
| `metergraph` (PyPI and npm) | [metergraph/metergraphsdk](https://github.com/metergraph/metergraphsdk) | Zero-dependency capture SDKs for Python and TypeScript |
| `metergraph-core` (PyPI) | [`core`](core) | Model registry, price catalog, and deterministic pricing engine; the canonical `models.yaml` and `prices.yaml` |
| `metergraph-server` | [`server`](server) | FastAPI and Postgres ingest and usage API; prices traffic through `metergraph-core` |
| dashboard | [`dashboard`](dashboard) | React single-page app served by the server |

## Reference

### Self-hosting configuration

Environment variables, the HTTP API, and security notes are in
[docs/self-hosting.md](docs/self-hosting.md).

### Coding agent access (MCP)

The self-hosted server exposes a read-only, metadata-only Agent Access API. It
can provide workspace context, capability discovery, routes, usage, and trace
metadata. Content, prompts, completions, tool arguments, ingestion health,
incidents, imported reports, trace details, and replay are unavailable because
the open-source server is content-blind and does not include paid Metergraph
analysis.

Set one or more read-only agent tokens with `MG_AGENT_TOKENS`. Values are
comma-separated. To rotate a token, add the new token to the list and restart
the server, update clients to use it, then remove the old token and restart
again. `MG_TOKENS` tokens can also read Agent Access, but `MG_AGENT_TOKENS`
tokens cannot ingest telemetry.

For a local coding-agent client, configure the stdio bridge:

```json
{
  "mcpServers": {
    "metergraph": {
      "command": "metergraph-mcp",
      "env": {
        "METERGRAPH_URL": "http://localhost:8787",
        "METERGRAPH_AGENT_TOKEN": "replace-with-agent-token"
      }
    }
  }
}
```

The same transport is available over HTTP:

```bash
curl -sS http://localhost:8787/v1/agent/mcp \
  -H 'Authorization: Bearer replace-with-agent-token' \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
```

See [docs/agent-access.md](docs/agent-access.md) for the shared contract,
tool availability, bounds, and error behavior.

### Updating model prices

Prices live in
[`core/src/metergraph_core/data/prices.yaml`](core/src/metergraph_core/data/prices.yaml),
the single public catalog. It is effective-dated so history reprices
correctly. To update a price, close the old window with `effective_to`, add a
new entry with `effective_from` and a `source_url`, and open a pull request. A
catalog change updates the declared catalog version and produces a patch
release of `metergraph-core`. Self-hosters can mount a newer file with
`MG_PRICES_PATH` without rebuilding. See [docs/prices.md](docs/prices.md).

Model identity, provider routes, and managed candidate fields live separately
in
[`core/src/metergraph_core/data/models.yaml`](core/src/metergraph_core/data/models.yaml).
Keeping these facts separate prevents a priceable model from becoming an
analysis candidate implicitly. Model providers can review and correct how
their models are priced; see
[Provider review and contributions](core/docs/catalog-maintenance.md#provider-review-and-contributions).

## Development

The server, dashboard, public catalog, and reusable pricing core live in this
repository. The SDKs live in
[metergraph/metergraphsdk](https://github.com/metergraph/metergraphsdk).

Install the local core before the server so the server resolves its
`metergraph-core` dependency from source, not a published release:

```bash
# Core and server (needs Postgres). Core must install first.
python -m pip install -e './core[dev]' -e './server[dev]'
MG_TEST_DATABASE_URL=postgresql://localhost:5432/metergraph_test pytest core/tests server/tests

# Dashboard
cd dashboard && npm install && npm run dev
```

See [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

## License

[Apache-2.0](LICENSE)
