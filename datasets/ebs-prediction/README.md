# env_broad_scale (EBS) Prediction Eval

Predict `env_broad_scale` from non-GOLD biosample metadata, scored with ontology-aware metrics.

`env_broad_scale` is a MIxS slot representing the broad-scale environmental context of a sample. In NMDC submissions, it is typically populated with an ENVO biome term in `"label [CURIE]"` format (e.g. `"terrestrial biome [ENVO:00000446]"`), but the slot itself is defined by MIxS, not ENVO.

> **See also**: [repo README](../../README.md) for setup; [`envo_scorer.py` module docstring](../../src/nmdc_ai_eval/envo_scorer.py) for the scoring formula and implementation details.

## Why This Eval

### The biology

Environmental samples in NMDC are annotated with an "environmental triad": `env_broad_scale`, `env_local_scale`, and `env_medium` — three slots (defined by MIxS) typically populated with ENVO terms describing where a sample came from at decreasing spatial scales. Getting these right matters because they drive downstream data discovery, cross-study comparison, and ecological analysis. Mis-annotated env_broad_scale means a forest soil sample might not show up when someone searches for forest biome studies.

### The information science

Unlike the sampleData classification task (8 discrete labels), env_broad_scale is a prediction into a structured ontology (~23 distinct values in this dataset, drawn from a hierarchy of ENVO biome terms). This means:
- **Near misses have meaning**: predicting "coniferous forest biome" when the truth is "forest biome" is much better than predicting "oceanic zone biome"
- **Specificity matters**: a more-specific correct descendant shows better understanding than a vague ancestor
- **Schema compliance can be checked independently**: each MIxS template (soil_data, water_data, etc.) defines which ENVO terms are valid — so we can score whether the LLM's prediction respects those constraints

### The software engineering

The eval is designed as a **post-hoc scorer** that enriches llm-matrix output with ontology-aware metrics. This avoids coupling to llm-matrix's limited built-in metrics and lets us iterate on scoring without re-running expensive LLM calls.

## Setup

```bash
# From repo root
uv sync                          # installs oaklib + all deps
uv run llm keys set openai       # if using OpenAI models
uv run llm keys set anthropic    # if using Anthropic models
```

First run of the scorer downloads ~50MB ENVO sqlite (cached in `~/.data/oaklib/`).

## Pipeline

```
generate_suite.py → suite YAML → run_suite.py → results.tsv → score_envo.py → results_envo_scored.tsv
```

### Quick start

```bash
just generate-ebs               # generate suite YAMLs (both providers)
just run-ebs-openai             # run OpenAI eval
just score-ebs openai           # apply ontology scorer
# or end-to-end:
just eval-ebs                   # generate + run all + score all
```

### Step by step

```bash
# 1. Generate suites (default 10 per category, min pool 10, both providers)
uv run python datasets/ebs-prediction/generate_suite.py

# 2. Run eval (produces results.tsv)
uv run python -m nmdc_ai_eval.run_suite datasets/ebs-prediction/ebs-suite-openai.yaml

# 3. Score with ontology metrics (produces results_envo_scored.tsv)
uv run python -m nmdc_ai_eval.score_envo datasets/ebs-prediction/ebs-suite-openai-output/results.tsv
```

## Task Design

**Target**: `env_broad_scale` — a MIxS slot typically populated with an ENVO biome term in `"label [CURIE]"` format

**Inputs** (all non-GOLD metadata):
| Slot | Role |
|---|---|
| study_name, description, notes | Context about the study |
| sampleData | MIxS environmental package (soil_data, water_data, etc.) |
| env_local_scale, env_medium | Other triad members — used as **predictors only** |
| geo_loc_name, depth | Physical location clues |
| analysis_type | Sequencing method |

**Excluded** (GOLD ecosystem path — too correlated, would make the task trivial):
ecosystem, ecosystem_type, ecosystem_subtype, ecosystem_category, specific_ecosystem

## Scoring Formula

Composite score bounded **[0, 1]**, computed as a weighted sum of four dimensions:

```
ontology_score = 0.1 × parse_ok + 0.1 × curie_label_valid + 0.5 × hierarchy_score + 0.3 × enum_score
```

### Dimensions

| Component | Weight | Meaning | Tool |
|---|---|---|---|
| `parse_ok` | 0.1 | Did the LLM output valid `"label [CURIE]"` syntax? | regex |
| `curie_label_valid` | 0.1 | Does the CURIE resolve to the stated label in ENVO? | oaklib `label()` |
| `hierarchy_score` | 0.5 | Ontology proximity to ground truth | oaklib `ancestors()` + BFS |
| `enum_score` | 0.3 | Is prediction in the template's allowed value set? | bundled TSVs from [submission-schema](https://github.com/microbiomedata/submission-schema) |

### Hierarchy scoring (descendants weighted higher)

| Relationship | Formula | Example at 1 hop |
|---|---|---|
| Exact match | 1.0 | 1.0 |
| Descendant, d hops | max(0, 1.0 − 0.10 × d) | 0.90 |
| Ancestor, d hops | max(0, 1.0 − 0.15 × d) | 0.85 |
| Unrelated | 0.0 | 0.0 |

Descendants are weighted higher: predicting a more-specific child term ("coniferous forest biome" for truth "forest biome") shows domain knowledge. Predicting a vaguer ancestor ("biome" for truth "forest biome") is less useful.

### Enum scoring

| Condition | Score |
|---|---|
| In template enum | 1.0 |
| Not in template enum | 0.0 |
| No enum for template | 0.5 (neutral) |

### Score examples

| Scenario | parse | label | hier | enum | **Total** |
|---|---|---|---|---|---|
| Perfect: exact, valid label, in enum | 0.1 | 0.1 | 0.5 | 0.3 | **1.00** |
| Descendant 1 hop, valid, in enum | 0.1 | 0.1 | 0.45 | 0.3 | **0.95** |
| Ancestor 1 hop, valid, in enum | 0.1 | 0.1 | 0.425 | 0.3 | **0.925** |
| Exact but wrong label, no enum file | 0.1 | 0.0 | 0.5 | 0.15 | **0.75** |
| Unrelated but in enum, valid label | 0.1 | 0.1 | 0.0 | 0.3 | **0.50** |
| Parse failure | 0.0 | 0.0 | 0.0 | 0.0 | **0.00** |

## Interpreting Results

The scored TSV (`results_envo_scored.tsv`) adds these columns:

| Column | Type | Meaning |
|---|---|---|
| `parse_success` | bool | LLM output matched `"label [CURIE]"` format |
| `pred_curie`, `pred_label` | str | Parsed prediction |
| `truth_curie`, `truth_label` | str | Parsed ground truth |
| `curie_label_valid` | bool | CURIE's canonical ENVO label matches predicted label |
| `exact_match` | bool | Predicted CURIE = truth CURIE |
| `relationship` | str | exact / descendant / ancestor / unrelated |
| `hop_distance` | int | Shortest path in ENVO subClassOf graph (None if unrelated) |
| `in_template_enum` | bool | Prediction in template's allowed value set (None if no enum) |
| `ontology_score` | float | Composite score [0, 1] |
| `response_time_s` | float | *Stub — not yet populated* |
| `prompt_tokens` | int | *Stub — not yet populated* |
| `completion_tokens` | int | *Stub — not yet populated* |
| `est_cost_usd` | float | *Stub — not yet populated* |

### What to look for

- **High ontology_score, low exact_match**: model understands the domain but picks nearby terms — might indicate ambiguity in the data or overly specific ground truth
- **Many ancestors**: model is hedging with vague terms — may need stronger system prompt
- **Many descendants**: model is more specific than ground truth — arguably good
- **Low enum compliance with high hierarchy_score**: model picks valid ENVO terms that aren't in the submission-schema's allowed set — suggests the enum is too restrictive or the model doesn't know the constraints
- **curie_label_valid=False**: model hallucinated a CURIE-label pair that doesn't exist in ENVO

## Enum Data

Bundled in `enum_data/`, copied from [submission-schema](https://github.com/microbiomedata/submission-schema) `notebooks/environmental_context_value_sets/`:

| Template | Enum file | Count |
|---|---|---|
| soil_data | soil_env_broad_scale.tsv | 52 |
| water_data | water_env_broad_scale.tsv | 56 |
| sediment_data | sediment_env_broad_scale.tsv | 15 |
| plant_associated_data | plant_associated_env_broad_scale.tsv | 72 |

Templates without enums: air_data, host_associated_data, misc_envs_data, metagenome_sequencing_non_interleaved_data → enum_score = 0.5 (neutral).

## Artifacts

Running an eval suite (`just run-ebs-openai`) produces:

| File | Description |
|---|---|
| `ebs-suite-{provider}.db` | DuckDB database created by llm-matrix. Caches results — **must be deleted before re-running with a regenerated suite**, otherwise llm-matrix reuses cached responses for matching cases. Use `just clean-outputs` to clear. |
| `ebs-suite-{provider}-output/results.tsv` | Tabular results extracted from the .db — one row per (case × model) combination. |
| `ebs-suite-{provider}-output/results_envo_scored.tsv` | Enriched results after `just score-ebs {provider}` — adds ontology-aware scoring columns. |

The `.db` files can be explored with DuckDB CLI or any DuckDB client. Use `just clean-outputs` to remove all eval artifacts.

## Data Coverage

The source TSV has 5,052 rows (318 unique with non-empty env_broad_scale after dedup on prompt-relevant columns), spanning 22 distinct values. Distribution is heavily skewed — the top 3 categories account for half the unique rows.

Deduplication uses only the 9 INPUT_COLUMNS (the non-GOLD slots that appear in the prompt) plus `env_broad_scale` (the target). GOLD ecosystem columns are excluded from dedup because they are excluded from prompts.

At the defaults (`--per-category 10 --min-pool 10`), the generator produces **100 cases across 10 strata** (10 each). The 12 excluded categories each have fewer than 10 unique rows — too few for meaningful per-stratum evaluation. The 10 included categories cover 283 of 318 unique rows (89% of the data).

The `--min-pool` threshold exists because with fewer than ~10 observations per stratum, confidence intervals are too wide to distinguish signal from noise. For example, a model scoring 4/5 correct has a 95% CI of [28%, 99%] — that tells you nothing about per-category performance. With 10 samples, the intervals are still wide but directionally useful.

To include rare categories at the cost of statistical reliability, pass `--min-pool 1`. To improve coverage for underrepresented categories, increase the source data pool (see [external-metadata-awareness#312](https://github.com/microbiomedata/external-metadata-awareness/issues/312)).

## Response Time & Cost Tracking

Not yet implemented. llm-matrix does not expose per-request timing or token counts. The scored TSV includes stub columns (`response_time_s`, `prompt_tokens`, `completion_tokens`, `est_cost_usd`) so the output schema is stable for downstream consumers. Implementation options:

1. **Wrapper timing**: time `run_suite.py` calls and join on case ID
2. **Provider API logs**: extract from Anthropic/OpenAI usage dashboards
3. **llm-matrix enhancement**: contribute timing to upstream

## Known Limitations

- Leading underscores in source data: `"__temperate woodland biome [ENVO:01000221]"` — parser strips these
- 4 of 8 templates lack enum files → neutral score for enum dimension
- oaklib's ENVO sqlite may lag behind the latest ENVO release
- LinkML schema validation (checking predictions directly against submission-schema's LinkML enums) is stubbed but not implemented — see `validate_via_linkml()` in `envo_scorer.py`
