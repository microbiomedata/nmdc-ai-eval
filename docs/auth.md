# Authentication details

## llm plugin providers (personal API keys)

Models are called via [llm](https://llm.datasette.io/) plugins:

| Plugin | Provides | Install status |
|---|---|---|
| (built-in) | OpenAI models (`gpt-4o`, etc.) | Always available |
| [llm-claude-3](https://github.com/simonw/llm-claude-3) | Anthropic models (`anthropic/claude-*`) | Listed in pyproject.toml |
| [llm-gemini](https://github.com/simonw/llm-gemini) | Google Gemini models (`gemini/*`) | Listed in pyproject.toml |

Set keys via:

```bash
uv run llm keys set openai       # paste your OpenAI key
uv run llm keys set anthropic    # paste your Anthropic key
uv run llm keys set gemini       # paste your Google AI Studio key
```

Or set environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`). The key store takes priority.

## Which access provider should I use?

| Provider | Who | Use for | Auth mechanism | Status |
|---|---|---|---|---|
| **Personal API keys** | Anyone | Dev, eval | llm key store | Working for OpenAI, Anthropic, Gemini AI Studio |
| **CBORG** (LBNL) | LBL staff | Dev, eval | CBORG API key + `OPENAI_API_BASE` | Not yet tested. See [suggestor-ai-tool#33](https://github.com/microbiomedata/nmdc-metadata-suggestor-ai-tool/issues/33) |
| **PNNL AI Incubator** | PNNL staff | Dev, eval | PNNL API key + custom `base_url` | Not yet tested. Contact Olivia Hess |
| **Vertex AI** (`nmdc-llm` GCP) | Team | Production/demo only | Service account JSON | Works via pipeline backend |

## Gemini auth: AI Studio vs Vertex AI

The `llm-gemini` plugin only supports [Google AI Studio](https://aistudio.google.com/) API keys. It does **not** support Vertex AI authentication.

The suggestor tool uses Vertex AI via Sierra Moxon's `nmdc-llm` service account. Those credentials work with the **pipeline backend** (`--provider gcp`) but not with the **llm backend** or llm-matrix suites.

**To get Gemini working with the llm backend:** Generate a free Google AI Studio key at https://aistudio.google.com/apikey and run:

```bash
uv run llm keys set gemini
```

The AI Studio free tier provides 1,500 requests/day — sufficient for eval runs.

## Other Google auth options

| Method | Works with `llm-gemini`? | Notes |
|---|---|---|
| Google AI Studio API key | **Yes** | Free tier, 1500 req/day |
| Vertex AI service account (`nmdc-llm`) | No (llm plugin) / Yes (pipeline backend) | $500 shared budget |
| gcloud ADC (`culturebot-476200`) | No | Works with Gemini CLI but not `llm-gemini` |
| CBORG | Untested | Would use the OpenAI plugin, not `llm-gemini` |

> **Note for CBORG and PNNL users:** These endpoints are OpenAI-compatible, so in principle you can point the OpenAI plugin at them by setting `OPENAI_API_BASE`. This has not been tested with llm-matrix yet and may conflict if you also need direct OpenAI access in the same eval run.

## Pipeline backend credentials (GCP or PNNL)

The pipeline backend calls `run_recommendation_pipeline()` from `nmdc-metadata-suggestor-ai-tool`. This is separate from the llm key store. Credentials go in `.env`:

**GCP Vertex AI:**

```bash
GOOGLE_APPLICATION_CREDENTIALS=/path/to/nmdc-llm-service-account.json
VERTEX_PROJECT_ID=nmdc-llm
```

Contact Sierra Moxon for the service account JSON.

**PNNL AI Incubator:**

```bash
AI_INCUBATOR_KEY=your-key
AI_INCUBATOR_BASE_URL=https://...
```

Contact Olivia Hess for the endpoint URL and key.

> **Budget reminder:** The `nmdc-llm` GCP project has a shared $500 total budget. Use personal API keys for iterative dev and model comparisons.
