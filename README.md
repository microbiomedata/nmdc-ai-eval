# nmdc-ai-eval

Framework and data for performing evaluations for AI-powered NMDC tools.

## Prerequisites

These must be installed before you start. Everything else is handled by `uv sync`.

| Tool | Minimum Version | Install |
|------|----------------|---------|
| [uv](https://docs.astral.sh/uv/) | 0.6+ | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| [just](https://just.systems/) | 1.0+ | `cargo install just` or `brew install just` or [other methods](https://just.systems/man/en/packages.html) |
| [git](https://git-scm.com/) | 2.0+ | System package manager |
| Python | 3.11+ | Managed by uv (`uv python install 3.11`) |

## Quickstart

```bash
git clone git@github.com:microbiomedata/nmdc-ai-eval.git
cd nmdc-ai-eval
just setup
```

### MongoDB setup (required for field-guidance pipeline eval)

The pipeline eval path (Option A) fetches full submission documents from a local MongoDB instance. You need the `nmdc_data_dev` database with the `nmdc_submissions` collection.

**Load submissions:**

```bash
cd ~/gitrepos/external-metadata-awareness   # or wherever you have it cloned
make -f Makefiles/nmdc_metadata.Makefile nmdc-submissions-to-mongo-dev
```

**Verify:**

```bash
mongosh nmdc_data_dev --eval "db.nmdc_submissions.countDocuments()"
# Expected: 400+
```

If you don't have `external-metadata-awareness` cloned, ask Mark Miller for a MongoDB dump or a data-dev connection. The llm-matrix path (Option B) does **not** require MongoDB — it uses committed suite YAMLs.

### API keys and access providers

Models are called via native [llm](https://llm.datasette.io/) plugins — one per provider:

| Plugin | Provides | Install status |
|---|---|---|
| (built-in) | OpenAI models (`gpt-4o`, etc.) | Always available |
| [llm-claude-3](https://github.com/simonw/llm-claude-3) | Anthropic models (`anthropic/claude-*`) | Listed in pyproject.toml |
| [llm-gemini](https://github.com/simonw/llm-gemini) | Google Gemini models (`gemini/*`) | Listed in pyproject.toml |

#### Setting up keys

The `llm` key store (`~/.config/io.datasette.llm/keys.json`) is the recommended way to manage API keys:

```bash
uv run llm keys set openai       # paste your OpenAI key
uv run llm keys set anthropic    # paste your Anthropic key
uv run llm keys set gemini       # paste your Google AI Studio key
```

Alternatively, set environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`). The `llm` key store takes priority over env vars.

You only need keys for the models you intend to run. If a key is missing, the eval will fail at runtime with a clear error for that model.

#### Verifying your setup

```bash
uv run llm models list           # shows all registered models
uv run llm keys list             # shows which providers have keys
just test                        # includes a test that every model in models.yaml is registered
```

#### Which access provider should I use?

| Provider | Who | Use for | Auth mechanism | Status in this repo |
|---|---|---|---|---|
| **Personal API keys** | Anyone | Dev, eval | API keys in llm key store | **Working** for OpenAI and Anthropic. Gemini needs a Google AI Studio key (see below). |
| **CBORG** (LBNL) | LBL staff | Dev, eval | CBORG API key + `OPENAI_API_BASE` | **Not yet tested.** See [suggestor tool #33](https://github.com/microbiomedata/nmdc-metadata-suggestor-ai-tool/issues/33). |
| **PNNL AI Incubator** | PNNL staff | Dev, eval | PNNL API key + custom `base_url` | **Not yet tested.** Check with Olivia Hess for endpoint details. |
| **Vertex AI** (`nmdc-llm` GCP) | Team | **Production/demo only** | Service account JSON or gcloud ADC | **Not supported** by `llm-gemini` plugin. See note below. |

#### Gemini auth: AI Studio vs Vertex AI

**This is a known gap.** The `llm-gemini` plugin only supports [Google AI Studio](https://aistudio.google.com/) API keys. It does **not** support Vertex AI authentication (service accounts, `GOOGLE_APPLICATION_CREDENTIALS`, gcloud ADC).

The suggestor tool (`nmdc-metadata-suggestor-ai-tool`) uses Vertex AI via Sierra Moxon's `nmdc-llm` service account. **Those credentials will not work with this eval repo.** If you have a Vertex AI service account but no AI Studio key, you cannot currently run Gemini evals here.

**To get Gemini working in this repo:** Generate a free Google AI Studio key at https://aistudio.google.com/apikey and run:
```bash
uv run llm keys set gemini    # paste the AI Studio key
```

The AI Studio free tier provides 1,500 requests/day — sufficient for eval runs.

**Vertex AI budget reminder:** The `nmdc-llm` GCP project has a shared $500 total budget. Even if Vertex support is added later, it should not be used for iterative eval runs.

#### Other Google auth options you may already have

| Method | Works with `llm-gemini`? | Notes |
|---|---|---|
| Google AI Studio API key | **Yes** | Free tier, 1500 req/day. This is what you need. |
| Vertex AI service account (`nmdc-llm`) | No | Suggestor tool uses this. $500 shared budget. |
| gcloud ADC (`culturebot-476200`) | No | Works with Gemini CLI but not `llm-gemini`. $25/mo LBL allowance. |
| CBORG (routes to Gemini via Google Cloud) | Untested | Would use the OpenAI plugin, not `llm-gemini`. $50/mo LBL credit. |

> **Note for CBORG and PNNL users:** These endpoints are OpenAI-compatible, so in principle you can point the OpenAI plugin at them by setting `OPENAI_API_BASE`. However, this has not been tested with llm-matrix yet and may conflict if you also need direct OpenAI access in the same eval run. File an issue if you need help with this setup.

### Pipeline eval credentials (GCP or PNNL, Option A only)

The pipeline eval calls `run_recommendation_pipeline()` from `nmdc-metadata-suggestor-ai-tool`. This is separate from the `llm` key store and requires one of:

**GCP Vertex AI (team default):**

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/nmdc-llm-service-account.json
export VERTEX_PROJECT_ID=nmdc-llm
# Optional (defaults to us-east5): export CLOUD_ML_REGION=us-east5
```

Contact Sierra Moxon for the `nmdc-llm` service account JSON (snappass).

**PNNL AI Incubator:**

```bash
export AI_INCUBATOR_KEY=your-key
export AI_INCUBATOR_BASE_URL=https://...
```

Contact Olivia Hess for the endpoint URL and key.

> **Budget reminder:** The `nmdc-llm` GCP project has a shared $500 total budget. Use Option B (llm-matrix with personal keys) for iterative dev and model comparisons.

## Eval approaches

### Field Guidance eval (Task 1) — which slots to recommend

The field guidance eval predicts which biosample metadata fields a submitter should fill, scored against hand-curated ground truth (6 submissions from Montana Smith and Bea Meluch).

Three ways to run it, all using the suggestor's production prompt and scoring with precision/recall/F1:

| | Pipeline backend | llm backend | llm-matrix |
|---|---|---|---|
| **What it is** | Suggestor's `LLMClient` | `llm` library adapter | `llm-matrix` suites |
| **Models** | GCP Gemini, PNNL GPT | Any model with an `llm` plugin | Models listed in `models.yaml` |
| **DOI/PDF enrichment** | Yes | Yes | No ([#28](https://github.com/microbiomedata/nmdc-ai-eval/issues/28)) |
| **Credentials** | GCP or PNNL service accounts | Personal API keys | Personal API keys |
| **MongoDB required** | Yes | Yes | Only for regenerating suite YAML |
| **Run** | `just run-field-guidance-pipeline` | `just run-field-guidance-llm gpt-4o` | `just run-field-guidance` |

The **pipeline** and **llm** backends produce directly comparable results — same prompt, same DOI/PDF ingestion, same scoring. The only difference is which LLM API processes the request.

**Flags:**

- `--no-enrichment` — skip DOI waterfall and PDF download (context ablation: how much does publication content matter?)
- `--verify` — re-prompt the model to cite evidence for each recommendation; drops unsupported ones (reduces tautological suggestions like "soil studies typically measure pH")
- `--strict` — count env triad fields in precision scoring (by default they're excluded since the ground truth intentionally omits them)
- `--sweep` — run all available models across all configured backends

**Results** are written to `datasets/field-guidance/pipeline-results/` as timestamped YAML files — one per model per run, never overwritten. Each file includes per-submission predictions, scores, reasons, token counts, and estimated cost.

```bash
# Compare results across models and runs
just compare-pipeline-results              # all results
just compare-pipeline-results --latest     # most recent per model
just compare-pipeline-results --detail     # per-submission breakdown
```

### Value prediction evals (env_broad_scale, sampleData)

These use llm-matrix suites with ontology-aware scoring. No MongoDB needed.

```bash
uv run llm keys set openai       # or anthropic, or gemini (AI Studio key)

# env_broad_scale: 100 cases × 5 models, ontology-scored
just eval-ebs

# sampleData prediction: 9 cases × 5 models (smoke test)
just eval-sampledata
```

Results include `input_tokens`, `output_tokens`, `duration_ms`, and `est_cost_usd` per call (captured from the llm logs DB).

### Cost estimation

Model pricing lives in the `pricing:` section of [`datasets/models.yaml`](datasets/models.yaml). Edit that file to add models or update prices — no code changes needed. If a model isn't in the pricing table, the eval still runs; cost just shows as unavailable.

## Usage

```bash
just --list              # see all available commands
just all                 # fix + check everything (no evals, no API calls)
just verify-auth         # test all configured API credentials
just full-eval           # field guidance: standard models × enrichment × verification
just full-eval --full    # field guidance: all provider tiers
just full-eval --cheap   # field guidance: budget models only
just compare-pipeline-results --latest   # compare field guidance results
just eval-ebs            # env_broad_scale: 100 cases × models in models.yaml
just eval-sampledata     # sampleData: smoke test (9 cases)
```

## QC and automation

All checks are defined once in `.pre-commit-config.yaml`. The justfile and CI both delegate to pre-commit so there is a single source of truth.

### What runs where

| Check | `just all` | git commit | git push | CI (PR) |
|---|---|---|---|---|
| ruff auto-fix + format | yes | — | — | — |
| check-toml, check-yaml | yes | yes | yes | yes |
| end-of-file-fixer | yes | yes | yes | yes |
| trailing-whitespace | yes | yes | yes | yes |
| codespell | yes | yes | yes | yes |
| typos | yes | yes | yes | yes |
| ruff (lint check) | yes | yes | yes | yes |
| ruff-format (verify) | yes | yes | yes | yes |
| uv-lock | yes | yes | yes | yes |
| mypy | yes | yes | yes | yes |
| deptry | yes | yes | yes | yes |
| pytest (excludes `@api`) | yes | yes | yes | yes |
| pip-audit | yes | yes | yes | yes |

`just all` is the only entry point that auto-fixes before checking. All other contexts (commit hook, push hook, CI) run the same 13 checks without fixing — they fail instead.

The git commit hook and git push hook run **identical checks**. Install both with `just setup`.

### Key just targets

| Target | What it does | Costs money? |
|---|---|---|
| `just all` | Fix + run all checks (~22s) | No |
| `just setup` | Install deps + pre-commit hooks | No |
| `just verify-auth` | Test all API credentials (1 cheap call each) | ~$0.001 |
| `just full-eval` | Field guidance: standard models × enrichment × verification | ~$0.50 |
| `just full-eval --full` | Field guidance: all provider tiers | ~$3 |
| `just full-eval --cheap` | Field guidance: budget models only | ~$0.10 |
| `just compare-pipeline-results` | Compare field guidance results (no LLM calls) | No |
| `just eval-ebs` | env_broad_scale: generate + run + ontology score | ~$0.10 |
| `just eval-sampledata` | sampleData: generate + run (smoke test) | ~$0.01 |
| `just generate` | Regenerate llm-matrix suite YAMLs | No |
| `just clean-outputs` | Delete all eval outputs | No |
| `just clean-all` | Delete outputs + suites + caches | No |

### Model configuration

[`datasets/models.yaml`](datasets/models.yaml) is the single config file for models. It has three sections:

- **`models:`** — which models go in llm-matrix suites (`just eval-ebs`, `just eval-sampledata`). Edit and run `just generate`.
- **`tiers:`** — which models run at each cost level in `just full-eval` (cheap/standard/full). Update when providers release new flagship or budget models.
- **`pricing:`** — cost per 1M tokens for cost estimation. Update when prices change.

Model names must match `uv run llm models list`. `just test` verifies every model in `models:` is recognized by an installed llm plugin.

To add a model for field guidance eval only (not llm-matrix suites), add it to the appropriate tier in `tiers:` and optionally to `pricing:`. No code changes needed.

### Test coverage

Run `just coverage` to see current coverage. As of the initial PR:

| File | Coverage | Notes |
|---|---|---|
| `envo_scorer.py` | 96% | Scoring math, oaklib integration, orchestrator, CLI |
| `run_suite.py` | excluded | Requires live LLM API calls — omitted from coverage measurement |

The minimum coverage threshold is **90%** (enforced via `--cov-fail-under` in pre-commit). `run_suite.py` is excluded from measurement because it requires live API calls. All other source code is tested without mocking.

## Datasets

- [`datasets/submission-metadata-prediction/`](datasets/submission-metadata-prediction/README.md) — **sampleData prediction** (smoke test): predict the MIxS environmental package from study name + description. 1 stratum (soil_data), 9 eval cases. Limited by source data diversity — see dataset README.
- [`datasets/ebs-prediction/`](datasets/ebs-prediction/README.md) — **env_broad_scale prediction**: predict the broad-scale environmental context (typically an ENVO biome term) from all non-GOLD metadata. Ontology-aware scoring with hierarchy, enum compliance, and CURIE-label validation. 10 strata, 100 eval cases (10 per stratum at default `--min-pool 10`).
- [`datasets/field-guidance/`](datasets/field-guidance/README.md) — **Metadata Field Guidance** (Task 1): predict which biosample slots a submitter should fill, given submission-level metadata only (study description, DOIs, MIxS extension). Ground truth: 6 hand-curated submissions from Montana Smith and Bea Meluch. Scored with precision/recall/F1 on slot name sets. Supports both eval paths; Option A uses the full suggestor pipeline including DOI/PDF enrichment.

## Access restrictions

Submission portal data is behind authentication. Do not publish to public lakehouses or buckets. This repo should remain internal to the team.
