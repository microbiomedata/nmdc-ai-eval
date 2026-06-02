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

### API keys

Set at least one provider's key:

```bash
uv run llm keys set openai       # paste your OpenAI key
uv run llm keys set anthropic    # paste your Anthropic key
uv run llm keys set gemini       # paste your Google AI Studio key
```

You only need keys for the models you intend to run. Verify with:

```bash
just verify-auth     # tests all configured providers (1 cheap call each)
```

For Vertex AI (GCP pipeline backend), PNNL, CBORG, and other auth options, see [docs/auth.md](docs/auth.md).

### MongoDB setup (required for field guidance eval)

The field guidance eval (`just full-eval`) fetches submission documents from a local MongoDB instance.

```bash
cd ~/gitrepos/external-metadata-awareness
make -f Makefiles/nmdc_metadata.Makefile nmdc-submissions-to-mongo-dev
mongosh nmdc_data_dev --eval "db.nmdc_submissions.countDocuments()"  # expected: 400+
```

The value prediction evals (`just eval-ebs`, `just eval-sampledata`) do **not** require MongoDB.

## Eval approaches

### Field Guidance eval (Task 1) — which slots to recommend

Predicts which biosample metadata fields a submitter should fill, scored against hand-curated ground truth (6 submissions from Montana Smith and Bea Meluch).

```bash
just full-eval           # standard tier models × enrichment × verification
just full-eval --full    # all provider tiers (cheap/mid/top per provider)
just full-eval --cheap   # budget models only
```

Each model is tested in up to 4 variants: with/without DOI enrichment × with/without evidence verification. All variants use the suggestor's production prompt including DOI waterfall and PDF ingestion (when enrichment is enabled).

**Flags** (passed via `just full-eval --no-verify` etc.):

- `--no-enrichment` — skip DOI waterfall and PDF download (context ablation)
- `--verify` / `--no-verify` — enable/skip evidence verification step
- `--strict` — count env triad fields in precision scoring
- `--models gpt-4o anthropic/claude-sonnet-4-6` — run specific models only
- `--no-clean` — keep previous results (default: clean first)

**Results** are written to `datasets/field-guidance/pipeline-results/` as timestamped YAMLs + `summary.tsv`.

```bash
just compare-pipeline-results --latest   # comparison table
just compare-pipeline-results --detail   # per-submission breakdown
```

### Value prediction evals (env_broad_scale, sampleData)

These use llm-matrix suites with the models listed in `datasets/models.yaml`. No MongoDB needed.

```bash
just eval-ebs            # env_broad_scale: 100 cases × 5 models, ontology-scored
just eval-sampledata     # sampleData: 9 cases × 5 models (smoke test)
```

### Cost estimation

Model pricing lives in the `pricing:` section of [`datasets/models.yaml`](datasets/models.yaml). Edit that file to add models or update prices — no code changes needed. If a model isn't in the pricing table, the eval still runs; cost just shows as unavailable.

## Usage

```bash
just --list              # see all available commands
just all                 # fix + check everything (no evals, no API calls)
just verify-auth         # test all configured API credentials
just full-eval           # field guidance: standard models × enrichment × verification
just full-eval --full    # field guidance: all provider tiers
just compare-pipeline-results --latest   # compare field guidance results
just eval-ebs            # env_broad_scale value prediction
just eval-sampledata     # sampleData smoke test
```

## Key just targets

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

## Model configuration

[`datasets/models.yaml`](datasets/models.yaml) is the single config file for models. It has three sections:

- **`models:`** — which models go in llm-matrix suites (`just eval-ebs`, `just eval-sampledata`). Edit and run `just generate`.
- **`tiers:`** — which models run at each cost level in `just full-eval` (cheap/standard/full). Update when providers release new flagship or budget models.
- **`pricing:`** — cost per 1M tokens for cost estimation. Update when prices change.

Model names must match `uv run llm models list`. `just test` verifies every model in `models:` is recognized by an installed llm plugin.

## QC and automation

All PR-gating checks are defined once in `.pre-commit-config.yaml`. The justfile and CI both delegate to pre-commit so there is a single source of truth. Dependency vulnerability scanning is intentionally not one of these checks; it runs separately via `just audit` (see below).

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

Dependency vulnerability scanning (`pip-audit`) runs separately from the commit/PR gates so an upstream advisory never blocks unrelated work: on demand via `just audit`, and weekly in CI (`.github/workflows/pip-audit.yml`, also runnable on demand from the Actions tab).

The minimum coverage threshold is **90%** (enforced via `--cov-fail-under` in pre-commit). `run_suite.py` is excluded from measurement because it requires live LLM calls. `llm_adapter.py` is tested with mocked LLM calls.

## Datasets

- [`datasets/submission-metadata-prediction/`](datasets/submission-metadata-prediction/README.md) — **sampleData prediction** (smoke test): predict the MIxS environmental package from study name + description. 1 stratum (soil_data), 9 eval cases.
- [`datasets/ebs-prediction/`](datasets/ebs-prediction/README.md) — **env_broad_scale prediction**: predict the broad-scale environmental context (ENVO biome term). Ontology-aware scoring. 10 strata, 100 eval cases.
- [`datasets/field-guidance/`](datasets/field-guidance/README.md) — **Metadata Field Guidance** (Task 1): predict which biosample slots to fill, given submission-level metadata. Ground truth: 6 hand-curated submissions. Run via `just full-eval`.

## Access restrictions

Submission portal data is behind authentication. Do not publish to public lakehouses or buckets. This repo should remain internal to the team.
