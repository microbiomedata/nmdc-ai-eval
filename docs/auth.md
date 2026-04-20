# Authentication details

## Quick start

1. Copy `.env.example` to `.env` and fill in only the providers you have credentials for.
2. Run:

   ```bash
   just verify-auth
   ```

   Each configured provider is hit with one real "reply with only: OK" call. Output shape:

   ```
   OK    gpt-4o-mini                                  -> OK
   SKIP  CBORG                                        -> no credentials in .env
   FAIL  PNNL AI Incubator                            -> Error code: 403 - ...
   ```

   - `OK` — credentials work.
   - `SKIP` — the provider's env vars aren't set. Fine; you only need providers you intend to use.
   - `FAIL` — the vars are set but the call failed. Read the error; common causes include expired keys, wrong base URL, or IP/VPN restrictions.

The script always exits 0; missing providers are expected. Only genuine failures (with credentials present) are highlighted.

For the three llm-plugin providers (OpenAI / Anthropic / Gemini direct), each `OK`/`FAIL` line also shows the resolved key source — `llm-store` (from `uv run llm keys set <provider>`) or `env` (from `.env` or the shell). **The llm key store takes priority over env vars**, so if you updated `.env` but the old key is still in the store, the store wins. Inspect it with:

```bash
uv run llm keys list                # which providers have store entries
uv run llm keys path                # path to the JSON file
```

## Provider comparison

| Provider | Who | Use for | Auth mechanism | Status |
|---|---|---|---|---|
| **OpenAI direct** | Anyone | Dev, eval | `OPENAI_API_KEY` (or llm key store) | Tested |
| **Anthropic direct** | Anyone | Dev, eval | `ANTHROPIC_API_KEY` (or llm key store) | Tested |
| **Gemini direct (AI Studio)** | Anyone | Dev, eval | `GEMINI_API_KEY` (or llm key store) | Tested |
| **CBORG** (LBNL) | LBL staff | Dev, eval | `CBORG_API_KEY` + `CBORG_BASE_URL` | Tested |
| **Vertex AI** (`nmdc-llm` GCP) | Team (shared SA) | Production suggestor + eval | Service account JSON + project ID | Tested |
| **PNNL AI Incubator** | PNNL staff | Dev, eval | `AI_INCUBATOR_KEY` + `AI_INCUBATOR_BASE_URL` | Tested |

"Tested" means `just verify-auth` makes one real LLM call through that provider when its env vars are set.

## Model-name routing

`llm-matrix` dispatches to a provider based on the model-name prefix. Knowing the routing helps debug "why won't this model run":

| Name pattern | Routes through | Backend |
|---|---|---|
| `gpt-*` | `llm` built-in openai plugin | OpenAI direct |
| `anthropic/*` | `llm-claude-3` plugin | Anthropic direct |
| `gemini/*` | `llm-gemini` plugin | Google **AI Studio** (not Vertex) |
| `*-project` suffix | Pipeline backend | PNNL AI Incubator |
| (Vertex / CBORG) | Not via `llm-matrix` — see notes below | |

**CBORG** fronts OpenAI, Anthropic, and Gemini model families through one OpenAI-compatible endpoint. It is *not* currently wired into `llm-matrix`; `just verify-auth` checks CBORG with a direct `openai` SDK call against `CBORG_BASE_URL`. Evals that want to use CBORG need a dedicated code path (see issue #62).

**Vertex** is Google-only (Gemini natively, Anthropic-on-Vertex via the anthropic SDK). It powers the production suggestor pipeline but is not a drop-in `llm-matrix` target.

## Direct LLM APIs (personal keys)

Models are called via [llm](https://llm.datasette.io/) plugins:

| Plugin | Provides | Install |
|---|---|---|
| (built-in) | OpenAI models (`gpt-4o`, etc.) | Always available |
| [llm-claude-3](https://github.com/simonw/llm-claude-3) | Anthropic models (`anthropic/claude-*`) | Listed in `pyproject.toml` |
| [llm-gemini](https://github.com/simonw/llm-gemini) | Google Gemini models (`gemini/*`) | Listed in `pyproject.toml` |

Set keys via env vars (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`) or the llm key store:

```bash
uv run llm keys set openai       # paste your OpenAI key
uv run llm keys set anthropic
uv run llm keys set gemini
```

The key store takes priority over env vars.

## CBORG (LBNL)

CBORG is LBL's internal OpenAI-compatible proxy. One endpoint, many model families (OpenAI, Anthropic, Gemini, and others). Get a key from the [CBORG portal](https://cborg.lbl.gov/).

```bash
CBORG_API_KEY=sk-...
CBORG_BASE_URL=https://api.cborg.lbl.gov/v1
```

`just verify-auth` tests CBORG with `gpt-4o-mini` by default. Override with `CBORG_TEST_MODEL=...` if that model isn't in your CBORG allowlist.

## Gemini: AI Studio vs Vertex

The `llm-gemini` plugin only supports [Google AI Studio](https://aistudio.google.com/) API keys. It does **not** support Vertex AI authentication.

The suggestor tool uses Vertex AI via Sierra Moxon's `nmdc-llm` service account. Those credentials work with the **pipeline backend** (`--provider gcp`) but not with the **llm backend** or `llm-matrix` suites.

For Gemini with the `llm` backend: generate a free Google AI Studio key at <https://aistudio.google.com/apikey> and run `uv run llm keys set gemini`. The free tier provides 1,500 requests/day — sufficient for eval runs.

## Vertex AI (GCP)

Used by the production suggestor pipeline. Requires a service account key.

```bash
GOOGLE_APPLICATION_CREDENTIALS=/path/to/nmdc-llm-service-account.json
VERTEX_PROJECT_ID=nmdc-llm
```

**Service account (SA)** = a non-human Google Cloud identity. The JSON file is its long-lived credential; whoever holds it can act as that identity. The `nmdc-llm` SA is shared across the team — contact Sierra Moxon for the file. Treat the JSON like a password (gitignored; don't share over Slack).

> **Budget reminder:** The `nmdc-llm` GCP project has a shared $500 total budget. Prefer personal or CBORG keys for iterative dev and model comparisons.

## PNNL AI Incubator

PNNL's internal OpenAI-compatible endpoint. PNNL staff only.

```bash
AI_INCUBATOR_KEY=...
AI_INCUBATOR_BASE_URL=https://...
```

Contact Olivia Hess for the endpoint URL and key. Model names use a `-project` suffix (e.g. `gpt-5-project`, `gpt-4.1-project`) — see `datasets/models.yaml` for the list with pricing notes.

## Troubleshooting

- **`FAIL` with a truncated error** — the script truncates to 200 chars. For full tracebacks, run the script directly: `uv run python scripts/verify_auth.py` and re-raise as needed.
- **I updated `.env` but my eval still uses the old key** — for OpenAI / Anthropic / Gemini direct, the llm key store (`uv run llm keys path`) takes priority over env vars. Check what's set with `uv run llm keys list`; overwrite an entry with `uv run llm keys set <provider>`, or clear the store by editing the JSON file directly.
- **`FAIL` on PNNL with a 403 or timeout** — PNNL's endpoint may enforce IP restrictions. Check whether you need VPN.
- **`FAIL` on Vertex with "creds file not found"** — `GOOGLE_APPLICATION_CREDENTIALS` points at a file that doesn't exist at that path. Check the path is absolute and the file is readable.
- **CBORG returns "model not found"** — your CBORG allowlist may not include the default test model. Set `CBORG_TEST_MODEL` to something in your allowlist.
