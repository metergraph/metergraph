# Self-hosting

## One-liner

```bash
MG_TOKENS=$(openssl rand -hex 16) docker compose up -d
```

Brings up Postgres 16 and the server (API + dashboard) on port 8787. Note the token you generated — the SDK and the dashboard both use it.

## Configuration

| Env var | Default | Meaning |
|---|---|---|
| `DATABASE_URL` | `postgresql://metergraph:metergraph@localhost:5432/metergraph` | Postgres DSN |
| `MG_TOKENS` | *(empty — all requests rejected)* | Comma-separated static bearer tokens |
| `MG_PRICES_PATH` | bundled `prices.yaml` | Mount an updated price catalog without rebuilding |
| `MG_REGION` | `global` | Preferred price region (falls back to `*`, then `global`) |
| `MG_MAX_BODY_BYTES` | `8388608` | Ingest body cap |
| `MG_MAX_ROWS` | `5000` | Rows per ingest request cap |
| `MG_SESSION_TTL_SECONDS` | `300` | SDK 0.4+ ingest-session lifetime in seconds (bounded to `60`–`3600`) |
| `MG_DASHBOARD_DIST` | `dashboard/dist` / `/app/static` in Docker | Static dashboard directory |

## Pointing the SDK at your server

```bash
export METERGRAPH_INGEST_URL=http://your-host:8787
export METERGRAPH_APP_TOKEN=<one of MG_TOKENS>
```

SDK 0.4 and later associate calls with their source repository. The SDK sends
the app token once to create a short-lived ingest session, then authenticates
telemetry batches with that session token. Session tokens are stateless and
expire after five minutes by default. Older SDKs and direct integrations can
continue sending a configured app token to `/v1/ingest`.

To exercise this exact SDK path locally without a provider API key:

```bash
python -m pip install 'metergraph>=0.4,<1'
MG_URL=http://localhost:8787 MG_TOKEN=<one of MG_TOKENS> python scripts/seed_demo.py
```

Seeded calls use the `demo` environment. The dashboard environment selector
lists every value present in the selected time range, plus **Untagged** when
applicable. All are selected initially; deselect `demo` to keep production and
untagged traffic while excluding synthetic rows.

## API

- `POST /v1/ingest/sessions` — SDK 0.4+ app-token exchange. Returns a short-lived session token for repository-aware ingestion.
- `POST /v1/ingest` — SDK wire format (`{"schema_version":1,"rows":[...]}`), gzip-aware, returns 202. Accepts SDK session tokens and, for backward compatibility, configured app tokens. Content fields are stripped; `event_type: "outcome"` rows are accepted and ignored (hosted feature).
- `GET /v1/environments?from=&to=` — environment values and call counts for the selected time range; `null` represents untagged traffic
- `GET /v1/usage?group_by=func|module|route|model|provider|day|hour&from=&to=&environment=&include_untagged=` — aggregates; repeat `environment` to select multiple values
- `GET /v1/usage/timeseries?group_by=func|route|model&bucket=hour|day&top=8&environment=&include_untagged=` — chart series
- `GET /v1/calls?func=&route=&limit=&before=&environment=&include_untagged=` — recent metadata rows
- `GET /v1/catalog` — loaded price catalog
- `GET /v1/config` — empty canary config (hosted feature), stable ETag
- `GET /healthz`

All `/v1/*` endpoints except `/v1/config` and `/healthz` require
`Authorization: Bearer <token>`. Session creation and dashboard/query APIs
require an app token. Only `/v1/ingest` also accepts a short-lived SDK session
token.

## Security notes

- App authentication uses static bearer tokens; run the server inside your network boundary and use distinct tokens per producer so they can be rotated independently. SDK 0.4+ limits routine ingest exposure by exchanging the app token for a short-lived, HMAC-signed session token.
- The dashboard stores its token in browser localStorage; treat dashboard access as trusted-network access.
- Durability is a synchronous Postgres commit before the 202; the SDK is fire-and-forget, so a down server drops rows rather than blocking your app. At self-host scale (SDK batches ≤512 KiB) a single instance comfortably handles typical traffic.

The call record also accepts nullable `http_status_code` (integer 100–599),
`availability` (`available`, `unavailable`, `excluded`) and `refused` (boolean).
Missing fields on older clients stay unknown. These fields do not alter pricing,
span status, error counts or call cardinality. They appear in `/v1/calls`.
`/v1/usage` includes availability/refusal counts, known denominators, rates and
coverage. Rates divide summed known outcomes; unknown observations are exposed
separately, and a zero denominator yields null. Explicit refusals with unknown
availability imply an available response; conflicting explicit dimensions become
unknown. No response content is retained for reliability classification.
