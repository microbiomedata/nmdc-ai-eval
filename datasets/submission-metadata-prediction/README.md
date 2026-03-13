# Submission Metadata Prediction

Eval dataset for the NMDC metadata suggestor AI tool. Each row pairs **submission-level text** (input) with **biosample slot values** (target) from the same submission.

## Schema

### Input columns (what the submitter provides)

| Column | Description |
|---|---|
| `study_name` | Submission study name |
| `description` | Submission description |
| `notes` | Submitter notes |

### Target columns (what the model should predict)

| Column | Description |
|---|---|
| `sampleData` | Sample data template (e.g. `soil_data`, `water_data`) |
| `env_broad_scale` | ENVO biome |
| `env_local_scale` | ENVO local environment |
| `env_medium` | ENVO environmental material |
| `geo_loc_name` | Geographic location |
| `depth` | Sample depth |
| `ecosystem` | GOLD ecosystem |
| `ecosystem_type` | GOLD ecosystem type |
| `ecosystem_subtype` | GOLD ecosystem subtype |
| `ecosystem_category` | GOLD ecosystem category |
| `specific_ecosystem` | GOLD specific ecosystem |
| `analysis_type` | Analysis type |

## Files

- **`eval_input_target_pairs.tsv`** — Materialized join of `nmdc_submissions` × `flattened_submission_biosamples`, filtered to `status = 'Released'`.
- **`generate_suite.py`** — Samples N rows per `sampleData` value and generates an [llm-matrix](https://github.com/monarch-initiative/llm-matrix) suite YAML. Run `python generate_suite.py --help` for options.
- **`sampledata-suite.yaml`** — Generated llm-matrix suite (25 cases, 5 per category).

## Setup

```bash
git clone git@github.com:microbiomedata/nmdc-ai-eval.git
cd nmdc-ai-eval
uv sync
```

### API keys

Two separate key stores are needed:

1. **`.env` file** — used by `run_suite.py` and any scripts that call the OpenAI/Anthropic SDKs directly:
   ```bash
   cp .env.example .env
   # Edit .env and fill in OPENAI_API_KEY and ANTHROPIC_API_KEY
   ```

2. **`llm` key store** — Simon Willison's [`llm`](https://llm.datasette.io/) package (which llm-matrix uses under the hood) keeps its own encrypted key store at `~/.llm/keys.json`. You must register keys there too:
   ```bash
   uv run llm keys set openai       # paste your OpenAI key when prompted
   uv run llm keys set anthropic    # paste your Anthropic key when prompted
   ```

   Verify with:
   ```bash
   uv run llm keys       # lists registered key names (not values)
   uv run llm models     # lists available models from registered providers
   ```

**Why both?** The `.env` file is for direct SDK usage in project code. The `llm` key store is for the `llm` CLI/library that llm-matrix delegates to for model calls. They don't share state.

## Running the eval

```bash
# Source .env (needed for direct SDK calls)
set -a && source .env && set +a

# Run the suite
uv run python -m nmdc_ai_eval.run_suite datasets/submission-metadata-prediction/sampledata-suite.yaml

# Regenerate suite with more samples
uv run python datasets/submission-metadata-prediction/generate_suite.py --per-category 10
```

## Stats

- 5,052 rows (Released submission–biosample pairs)
- Derived from 717 total submissions (450 non-test)

## Regeneration

The TSV is generated from the NMDC submission portal via [external-metadata-awareness](https://github.com/microbiomedata/external-metadata-awareness):

```bash
cd external-metadata-awareness

# 1. Fetch submissions into local MongoDB (requires NMDC_DATA_SUBMISSION_REFRESH_TOKEN)
make -f Makefiles/nmdc_metadata.Makefile nmdc-submissions-to-mongo

# 2. Export MongoDB → DuckDB → TSV
make -f Makefiles/nmdc_metadata.Makefile eval-input-target-tsv
```

The intermediate DuckDB (`local/nmdc_submissions.duckdb`, ~37MB) contains four tables and is not committed here due to size. See `sql/eval_input_target_pairs.sql` for the join query.

## Access restrictions

Submission portal data is behind authentication (`data.microbiomedata.org/api/metadata_submission`). Do not publish to public lakehouses or buckets. This repo should remain internal to the team.
