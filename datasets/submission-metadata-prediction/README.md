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
- **`generate_suite.py`** — Samples N rows per `sampleData` value and generates per-provider [llm-matrix](https://github.com/monarch-initiative/llm-matrix) suite YAMLs. Run `python generate_suite.py --help` for options.
- **`sampledata-suite-openai.yaml`** — Generated llm-matrix suite for OpenAI models (gpt-4o-mini, gpt-4o).
- **`sampledata-suite-anthropic.yaml`** — Generated llm-matrix suite for Anthropic models (claude-3-5-sonnet-latest, claude-3-5-haiku-latest).

## Setup

See the [top-level README](../../README.md) for prerequisites and API key setup.

## Running the eval

```bash
just run-openai      # run OpenAI suite only
just run-anthropic   # run Anthropic suite only
just run-all         # run both

# Regenerate suites with more samples
just generate 10
```

## Duplicate rows

The TSV is a materialized join of submissions × biosamples. When biosamples within a study share the same slot values (which is common — e.g. all soil samples from one submission have the same `sampleData`, `env_broad_scale`, `geo_loc_name`, etc.), the resulting rows are identical. Currently 5,052 rows collapse to 322 distinct rows.

**It is the caller's responsibility to deduplicate before sampling or evaluation.** `generate_suite.py` does this automatically, but if you consume the TSV directly, deduplicate first:

```python
import pandas as pd

df = pd.read_csv("eval_input_target_pairs.tsv", sep="\t").drop_duplicates()
```

## Stats

- 5,052 rows (Released submission–biosample pairs), 322 distinct
- Derived from 717 total submissions (450 non-test)

### Category coverage after deduplication

Not all `sampleData` categories have enough unique rows to fill the requested `--per-category` count. The generator warns when this happens. Current distribution:

| Category | Unique rows |
|---|---|
| soil_data | 244 |
| water_data | 64 |
| misc_envs_data | 11 |
| plant_associated_data | 2 |
| metagenome_sequencing_non_interleaved_data | 1 |
| host_associated_data | 0 |
| sediment_data | 0 |
| air_data | 0 |

Categories with 0 rows are valid `sampleData` values but have no Released submissions in the current dataset. To improve coverage, see issue [#312](https://github.com/microbiomedata/external-metadata-awareness/issues/312) (dropping the Released status filter).

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
