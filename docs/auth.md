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
| `vertex/gemini-*` | local `llm_plugin_vertex` | **Vertex AI** (Gemini via generateContent) |
| `vertex/claude-*` | local `llm_plugin_vertex` | **Vertex AI** (Claude via rawPredict) |
| `*-project` suffix | Pipeline backend | PNNL AI Incubator |
| (CBORG) | Not via `llm-matrix` — see notes below | |

**CBORG** fronts OpenAI, Anthropic, and Gemini model families through one OpenAI-compatible endpoint. It is *not* currently wired into `llm-matrix`; `just verify-auth` checks CBORG with a direct `openai` SDK call against `CBORG_BASE_URL`. Evals that want to use CBORG need a dedicated code path (see issue #62).

**Vertex** models are now reachable from `llm-matrix` via the in-repo `llm_plugin_vertex` — use `vertex/gemini-*` or `vertex/claude-*` anywhere a model name is accepted. Auth reuses the existing `GOOGLE_APPLICATION_CREDENTIALS` + `VERTEX_PROJECT_ID` setup below. Region defaults to `us-east5`; override with `CLOUD_ML_REGION` (or `GEMINI_REGION`) if your project uses a different Vertex location.

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

CBORG is LBL's internal OpenAI-compatible proxy. One endpoint, many model families (OpenAI, Anthropic, Gemini, and others). Manage keys at <https://api.cborg.lbl.gov/key/manage> (portal home: <https://cborg.lbl.gov/>; API docs: <https://cborg.lbl.gov/api_docs/>).

### Basic setup

```bash
CBORG_API_KEY=<your key>
CBORG_BASE_URL=https://api.cborg.lbl.gov
```

Note: no `/v1` suffix — the OpenAI SDK appends the path automatically.

`just verify-auth` tests CBORG with `gpt-4o-mini` by default. Override with `CBORG_TEST_MODEL=...` if that model isn't in your CBORG allowlist.

### Using CBORG models in eval suites

`just run-env-triad` (and every other `run-*` suite target) resolves model names through the `llm` library's plugin ecosystem, not through the `.env` that `verify-auth` / `probe-tiers` read. So setting `CBORG_API_KEY` alone **does not** make CBORG models usable in suite evals.

The bridge is `llm`'s `extra-openai-models.yaml` feature. Entries in that file register custom model aliases that hit any OpenAI-compatible endpoint. This repo ships a starter at `~/.config/io.datasette.llm/extra-openai-models.yaml` with seven aliases (`cborg/gpt-4o-mini`, `cborg/claude-sonnet-4-6`, `cborg/gemini-2.5-pro`, etc.). See the file's inline comments for how to add more.

**One-time activation:**

```bash
uv run llm keys set cborg
# paste the same value you have in CBORG_API_KEY
```

Verify with `uv run llm models list | grep cborg` — the aliases should appear. After that, any `cborg/*` model name is valid in `just pilot-env-triad` or other suite runs.

**Discover the full CBORG catalog** (~200 models) with:

```bash
just probe-tiers --list-cborg-models
```

Pick names you want and add new entries to `extra-openai-models.yaml` following the same pattern.

### Known limitation: key duplication

Right now CBORG credentials have to live in **both** `.env` (as `CBORG_API_KEY` — for `verify-auth` and `probe-tiers`, which use the `openai` SDK directly) **and** the `llm` key store (as `cborg` — for suite evals that route through `llm-matrix`). Changing CBORG's key means updating both places.

Tracked as [#71](https://github.com/microbiomedata/nmdc-ai-eval/issues/71); low priority given the friction is a one-time setup per dev.

### Adding more models from any other OpenAI-compatible endpoint

The same `extra-openai-models.yaml` pattern works for PNNL's AI Incubator, any other LiteLLM/OpenAI-proxy service, or a locally-hosted endpoint. Entry shape:

```yaml
- model_id: <alias-you-pick>        # what you'll type in suites
  model_name: <name-on-the-server>  # what the server expects
  api_base: <server base URL>
  api_key_name: <keystore entry>    # run `llm keys set <key_name>` once
```

## Adding more Gemini models via AI Studio

The `llm-gemini` plugin supports the full AI Studio catalog, not just the models listed in `datasets/models.yaml`. Add any `gemini/<name>` form to your command:

```bash
just pilot-env-triad 50 "gpt-4o-mini,gemini/gemini-2.5-pro,gemini/gemini-2.0-flash"
```

`uv run llm models list | grep -i gemini` shows what's installed. AI Studio's free tier allows 1500 requests per model per day — enough for most eval sweeps.

## Gemini: AI Studio vs Vertex

The `llm-gemini` plugin only supports [Google AI Studio](https://aistudio.google.com/) API keys. It does **not** support Vertex AI authentication.

For Vertex-backed Gemini (and Claude) via `llm` / `llm-matrix`, use the in-repo `llm_plugin_vertex` with model names like `vertex/gemini-2.5-flash` or `vertex/claude-haiku-4-5`. See the [Vertex AI section below](#vertex-ai-gcp).

For Gemini with the `llm` backend (AI Studio, not Vertex): generate a free Google AI Studio key at <https://aistudio.google.com/apikey> and run `uv run llm keys set gemini`. The free tier provides 1,500 requests/day — sufficient for eval runs.

## Vertex AI (GCP)

Used by the production suggestor pipeline. Requires a service account key.

```bash
GOOGLE_APPLICATION_CREDENTIALS=/path/to/nmdc-llm-service-account.json
VERTEX_PROJECT_ID=nmdc-llm
```

**Service account (SA)** = a non-human Google Cloud identity. The JSON file is its long-lived credential; whoever holds it can act as that identity. The `nmdc-llm` SA is shared across the team — contact Sierra Moxon for the file. Treat the JSON like a password (gitignored; don't share over Slack).

### Vertex is not Gemini-only — but the suggestor's dispatcher is

The `nmdc-llm` project has **multiple Model Garden publishers enabled**, including Google (Gemini) and Anthropic (Claude). Sierra's 2026-02-06 `vertex-usage` report shows active usage on `claude-opus-4-6`, `claude-sonnet-4-5`, and `claude-haiku-4-5`.

However, Vertex exposes each publisher through a **different API endpoint**:

| Publisher | Endpoint | Python SDK |
|---|---|---|
| Google (Gemini) | `:generateContent` | `google.genai.Client(vertexai=True)` |
| Anthropic (Claude) | `:rawPredict` / `:streamRawPredict` | `from anthropic import AnthropicVertex` |
| Others (e.g. Meta via MaaS) | varies | varies |

The suggestor's `LLMClient(access_provider="gcp")` currently only dispatches via `generateContent` (see [`llm_client.py:229`](https://github.com/microbiomedata/nmdc-metadata-suggestor-ai-tool/blob/main/src/nmdc_metadata_suggestor_ai_tool/llm_client.py#L229)), which means Gemini works but Claude calls return `400 "not supported in the generateContent API"` even though the project has Claude enabled. This is a **dispatcher gap, not an access restriction.**

The in-repo `llm_plugin_vertex` routes each publisher through the correct SDK, so `vertex/gemini-*` and `vertex/claude-*` both work from `llm-matrix` eval suites. Run `just probe-vertex-garden` to see which specific model names the SA can reach on your project.

> **Budget reminder:** The `nmdc-llm` GCP project has a shared $500 total budget. Claude Opus is ~$15/$75 per 1M tokens — use it sparingly. Prefer personal or CBORG keys for iterative dev.

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
