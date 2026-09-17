# Supplement-driven env triad eval (Task 2, phyllosphere)

Does giving the suggestor a publication's supplementary tables change the env triad it
suggests, and in which direction? Proposed in
[nmdc-metadata-suggestor-ai-tool#143](https://github.com/microbiomedata/nmdc-metadata-suggestor-ai-tool/issues/143).

## Test case

Study `nmdc:sty-11-e4yb9z58`, "Seasonal activities of the phyllosphere microbiome of perennial
crops", 192 biosamples. Its publication DOI `10.1038/s41467-023-36515-y` serves twelve
supplements through Europe PMC; `41467_2023_36515_MOESM4_ESM.csv` is a per-sample table with the
authors' own triad. Its award DOI yields a funding record with nothing to fetch.

The case itself is defined in the suggestor package, `nmdc_metadata_suggestor_ai_tool.evaluation.phyllosphere`:
the identifiers, a snapshot of the study and biosamples from the NMDC API (2026-09-11), the
submission object the pipeline needs, and the two references. This directory holds only the
runner and its results. That split is deliberate: the suggestor owns what is specific to its test
cases, this repo owns how they are run and scored.

## Two arms, two references

| Arm | Context the model sees |
|---|---|
| `publication` | study name and description, the Crossref abstract for the publication DOI (fetched by the pipeline's own submission path), and the biosample records with their triad removed |
| `publication+supplements` | the same, plus `format_supplement_context(retrieve_supplements(doi))`: an inventory of the kept files and the full text of the csv ones |

| Reference | Source |
|---|---|
| `supplement` | the authors' triad per sample in MOESM4, keyed by `sample_name` = biosample `name`. Underscored CURIEs, capitalised labels and a pipe-joined `env_medium` are normalized by the suggestor's `env_triad_scoring`; a prediction is scored against whichever of several terms it sits closest to in ENVO |
| `nmdc` | the triad stored on each biosample in the snapshot |

## Scoring

Per suggestion, `nmdc_ai_eval.triad_output_scorer` applies the same ENVO relationship, hop
distance and composite formula as `envo_scorer` (see its docstring for weights). The enum term
comes from the suggestor's validation gate: `submission_enum` provenance means the value is in the
extension's curated set. Arm comparison (`changed` / `toward` / `away` per slot) comes from the
suggestor's `compare_outputs`, graded by ENVO proximity rather than exact match.

Scoring is ENVO-only. A PO term such as `leaf [PO:0025034]` has no ENVO label or ancestors, so
against an ENVO reference it scores parse + enum only. Read `env_medium` numbers with that in mind.

## Running

```bash
just eval-supplement-triad                          # gcp, gemini default, chunk size 25, ~18 min
just eval-supplement-triad --limit 8 --chunk-size 8 # smoke run, ~2 min
just eval-supplement-triad --provider gcp --model gemini-2.5-pro
```

Needs the suggestor's Vertex credentials (`GOOGLE_APPLICATION_CREDENTIALS`, see
[docs/auth.md](../../docs/auth.md)). Results go to `pipeline-results/` as one timestamped YAML per
run (both arms, per-sample scores, arm deltas, the model's reasons) plus a `summary.tsv` row and a
`report.md`.

The chunk size defaults to 25, not the pipeline's 50: at 50 the model dropped the `id` field for a
whole chunk in one arm and truncated its JSON in the other, and the run could not be scored.
Coverage (samples answered, suggestions without an id, chunk errors) is reported first for that
reason.
