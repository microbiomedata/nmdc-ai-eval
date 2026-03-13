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

llm-matrix uses Simon Willison's [llm](https://llm.datasette.io/) package, which manages API keys in its own store (`~/.config/io.datasette.llm/keys.json`). Set keys with:

```bash
uv run llm keys set openai       # paste your OpenAI key when prompted
uv run llm keys set anthropic    # paste your Anthropic key when prompted
```

The `llm` key store takes priority over environment variables. Setting `OPENAI_API_KEY` in your shell or a `.env` file will **not** override a key already in the store.

## Usage

```bash
just --list          # see all available commands
just qc              # lint + test
just run             # run the default sampleData eval suite
just generate 10     # regenerate suite with 10 samples per category
just fix             # auto-fix lint/format issues
```

## QC: just commands and pre-commit hooks

Every QC check can be run manually with `just` or automatically via pre-commit/pre-push hooks. Install hooks with `just pre-commit-install`.

| Check | `just` command | Hook stage |
|---|---|---|
| ruff lint | `just lint` | pre-commit |
| ruff format | `just lint` | pre-commit |
| mypy | `just typecheck` | pre-commit |
| deptry | `just deptry` | pre-commit |
| pytest | `just test` | pre-commit |
| codespell | — | pre-commit |
| typos | — | pre-commit |
| check-toml/yaml | — | pre-commit |
| trailing whitespace | — | pre-commit |
| uv-lock | `just lock` | pre-commit |
| pip-audit | `just audit` | pre-push |

`just qc` runs lint + typecheck + deptry + audit + test in one shot. The pre-commit hooks cover everything except pip-audit (which runs on push since it hits the network).

### Test coverage

Source code coverage is currently **0%** — and that's expected. The tests validate **data and suite integrity** (TSV schema, value constraints, row counts, suite-vs-data consistency), not Python source code. The only source file (`run_suite.py`) is a thin CLI wrapper around llm-matrix that requires live API calls to exercise.

Coverage will become meaningful when:
- `generate_suite.py` logic (dedup, sampling, YAML generation) is tested via imports rather than just via its output artifacts
- Additional source modules are added (e.g. custom scorers, data loaders)

`pytest-cov` is installed for when that time comes: `uv run pytest --cov=nmdc_ai_eval --cov-report=term-missing`

## Datasets

- [`datasets/submission-metadata-prediction/`](datasets/submission-metadata-prediction/README.md) — 5,052 submission–biosample pairs from released NMDC submissions, with an llm-matrix eval suite for predicting MIxS environmental packages.

## Access restrictions

Submission portal data is behind authentication. Do not publish to public lakehouses or buckets. This repo should remain internal to the team.
