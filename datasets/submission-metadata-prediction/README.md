# sampleData Prediction Eval (Smoke Test)

Predict the MIxS environmental package (`sampleData`) from submission-level text.

> **Smoke test only.** The source data has only 17 distinct prompt-level cases (see [Deduplication](#deduplication)). At default settings, this eval runs **9 soil_data cases** — enough to validate the pipeline and catch regressions, but not enough for statistically meaningful per-category comparisons. For a more robust eval, see [env_broad_scale (EBS) prediction](../ebs-prediction/).

`sampleData` is the NMDC submission portal's name for the **MIxS environmental package** — a template that determines which metadata slots are required for a biosample. For example, `soil_data` triggers soil-specific slots like `soil_horizon`, while `water_data` triggers `water_body_type`. The 8 valid values correspond to MIxS environmental packages supported by the NMDC submission portal.

Each row in the eval TSV pairs **submission-level text** (input) with **biosample slot values** (target) from the same submission.

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
| `sampleData` | MIxS environmental package / template (e.g. `soil_data`, `water_data`) |
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
- **`sampledata-suite.yaml`** — Generated llm-matrix suite with all models from `datasets/models.yaml`.

## Setup

See the [top-level README](../../README.md) for prerequisites and API key setup.

## Running the eval

```bash
just run-sampledata             # run all models
just generate-sampledata        # regenerate suite YAML (defaults: 10 per category, min pool 5)
just generate-sampledata 10 1   # include rare categories (less reliable per-stratum scores)
```

## Deduplication

The TSV has 5,052 rows (materialized join of submissions × biosamples), but most are duplicates at the prompt level. The generator deduplicates on the columns that actually appear in the prompt (`study_name`, `description`, `notes`) plus the prediction target (`sampleData`). This yields **17 distinct cases** — the true number of unique inputs the model sees.

The much larger raw row count (322 when deduplicating on all 15 TSV columns) is misleading because columns like `env_broad_scale`, `geo_loc_name`, `depth`, and the GOLD ecosystem fields are not part of the sampleData prediction prompt. Rows that differ only in those biosample-level columns produce identical prompts.

**If you consume the TSV directly**, deduplicate on prompt-relevant columns:

```python
import pandas as pd

df = pd.read_csv("eval_input_target_pairs.tsv", sep="\t")
df = df.drop_duplicates(subset=["study_name", "description", "notes", "sampleData"])
```

## Stats

- 5,052 rows (Released submission–biosample pairs), **17 distinct at prompt level**
- Derived from 717 total submissions (450 non-test)
- 15 distinct prompts (2 studies map to 2 different sampleData values each)

### Category coverage after deduplication

By default, the generator excludes categories with fewer than `--min-pool` (default 5) unique rows. Current distribution:

| Category | Unique rows | Included |
|---|---|---|
| soil_data | 9 | **yes** |
| water_data | 3 | no |
| misc_envs_data | 2 | no |
| plant_associated_data | 2 | no |
| metagenome_sequencing_non_interleaved_data | 1 | no |
| host_associated_data | 0 | no |
| sediment_data | 0 | no |
| air_data | 0 | no |

Categories with 0 rows are valid `sampleData` values but have no Released submissions in the current dataset. To improve coverage, see issue [#312](https://github.com/microbiomedata/external-metadata-awareness/issues/312) (dropping the Released status filter).

## Artifacts

Running an eval suite (`just run-sampledata-openai`) produces:

| File | Description |
|---|---|
| `sampledata-suite.db` | DuckDB database created by llm-matrix. Caches results — **must be deleted before re-running with a regenerated suite**, otherwise llm-matrix reuses cached responses for matching cases. Use `just clean-outputs` to clear. |
| `sampledata-suite-output/results.tsv` | Tabular results extracted from the .db — one row per (case × model) combination. |

The `.db` files can be explored with DuckDB CLI or any DuckDB client. Use `just clean-outputs` to remove all eval artifacts.

## Data Coverage

At the defaults (`--per-category 10 --min-pool 5`), the generator produces **9 cases in 1 stratum** (soil_data only). The other categories have too few distinct prompt-level cases (water_data: 3, misc_envs_data: 2, plant_associated_data: 2, metagenome_sequencing_non_interleaved_data: 1). Three more valid templates (host_associated_data, sediment_data, air_data) have 0 rows in the current dataset.

**This is a smoke test, not a benchmark.** The source data has only 17 distinct prompt-level cases total — the prompt uses only `study_name` and `description`, and there are only 15 unique studies in the Released submissions. This eval validates the pipeline (prompt formatting, model connectivity, scoring) and catches regressions, but cannot support per-category accuracy comparisons.

To include all available categories (at the cost of 2–3 cases per stratum), pass `--min-pool 1`. To improve coverage, increase the source data pool (see [external-metadata-awareness#312](https://github.com/microbiomedata/external-metadata-awareness/issues/312)).

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
