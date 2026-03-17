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

## Usage

```bash
just --list                  # see all available commands
just all                     # fix + check everything (no evals, no API calls)
just eval-sampledata         # end-to-end sampleData eval
just eval-ebs                # end-to-end env_broad_scale eval (generate + run + score)
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

| Target | What it does |
|---|---|
| `just all` | Fix + run all checks (~22s) |
| `just fix` | Auto-fix lint/format only |
| `just check` | Run all checks without fixing |
| `just test` | pytest only (excludes `@api` tests) |
| `just coverage` | pytest with coverage report |
| `just generate` | Regenerate all suite YAMLs from `models.yaml` + source TSV |
| `just eval-all` | Full eval: generate + run all models + score |

### What has automation but is NOT in the checks

| What | How to run | Why excluded |
|---|---|---|
| Eval suite runs (LLM calls) | `just eval-sampledata`, `just eval-ebs`, `just eval-all` | Requires API keys, costs money |
| ENVO ontology scoring | `just score-ebs` | Requires eval output to exist |
| Suite YAML generation | `just generate` | Changes committed YAMLs, depends on source TSV |
| Cleanup | `just clean-outputs`, `just clean-all` | Destructive, on-demand only |

### Model configuration

Model names are defined once in `datasets/models.yaml` and read by both suite generators. To add or change models, edit that file and run `just generate`. Model names must match what `uv run llm models list` shows — these come from native llm plugins (OpenAI built-in, llm-claude-3 for Anthropic, llm-gemini for Gemini).

**Adding a model is three steps:**

1. Find the exact name: `uv run llm models list | grep <model>`
2. Add the name to `datasets/models.yaml`
3. Run `just generate` to regenerate suite YAMLs

**`just test` verifies that every model in `models.yaml` is recognized by an installed llm plugin.** If you add a model name that doesn't match any plugin, the test suite will fail with a clear message telling you which model is unrecognized and which plugin you may need.

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

## Access restrictions

Submission portal data is behind authentication. Do not publish to public lakehouses or buckets. This repo should remain internal to the team.
