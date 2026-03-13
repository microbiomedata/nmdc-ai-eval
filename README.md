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

Two separate key stores are needed:

1. **`.env` file** — used by scripts that call the OpenAI/Anthropic SDKs directly:
   ```bash
   cp .env.example .env
   # Edit .env: fill in OPENAI_API_KEY and ANTHROPIC_API_KEY
   ```

2. **`llm` key store** — the [`llm`](https://llm.datasette.io/) package (used by llm-matrix under the hood) has its own encrypted key store at `~/.llm/keys.json`:
   ```bash
   uv run llm keys set openai       # paste your OpenAI key when prompted
   uv run llm keys set anthropic    # paste your Anthropic key when prompted
   ```

   Verify with `uv run llm keys` (lists names, not values).

**Why both?** `.env` is for direct SDK usage in project code. The `llm` key store is for the `llm` CLI/library that llm-matrix delegates to. They don't share state.

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
