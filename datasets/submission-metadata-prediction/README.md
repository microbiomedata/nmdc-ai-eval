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

See the [top-level README](../../README.md) for prerequisites and API key setup.

## Running the eval

```bash
just run    # or: uv run python -m nmdc_ai_eval.run_suite datasets/submission-metadata-prediction/sampledata-suite.yaml

# Regenerate suite with more samples
just generate 10
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
