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

```bash
cp .env.example .env
# Edit .env: fill in OPENAI_API_KEY and ANTHROPIC_API_KEY
```

The `.env` file is the single source of truth for API keys. It is loaded automatically by `run_suite.py` (via python-dotenv), which makes the keys available to both direct SDK calls and the `llm` package that llm-matrix uses under the hood.

## Usage

```bash
just --list          # see all available commands
just qc              # lint + test
just run             # run the default sampleData eval suite
just generate 10     # regenerate suite with 10 samples per category
just fix             # auto-fix lint/format issues
```

## Datasets

- [`datasets/submission-metadata-prediction/`](datasets/submission-metadata-prediction/README.md) — 5,052 submission–biosample pairs from released NMDC submissions, with an llm-matrix eval suite for predicting MIxS environmental packages.

## Access restrictions

Submission portal data is behind authentication. Do not publish to public lakehouses or buckets. This repo should remain internal to the team.
