# Supplement-driven env triad eval (Task 2, phyllosphere)

Does giving the suggestor a publication's supplementary tables change the env triad it
suggests, and in which direction? Proposed in
[nmdc-metadata-suggestor-ai-tool#143](https://github.com/microbiomedata/nmdc-metadata-suggestor-ai-tool/issues/143).

## Test case

Study `nmdc:sty-11-e4yb9z58`, "Seasonal activities of the phyllosphere microbiome of perennial
crops", 192 biosamples. Its publication DOI `10.1038/s41467-023-36515-y` serves twelve
supplements through Europe PMC; `41467_2023_36515_MOESM4_ESM.csv` is a per-sample table with the
authors' own triad. Its award DOI yields a funding record with nothing to fetch.

The case is defined in `phyllosphere.py`: the identifiers, a snapshot of the study and
biosamples from the NMDC API (2026-09-11, in `data/`), the submission object the pipeline needs,
and the two references. The eval depends on nothing from the suggestor beyond its public API on
`main`. Full write-up and results: [docs/eval-phyllosphere-supplements.md](../../docs/eval-phyllosphere-supplements.md).

## Two arms, two references

| Arm | Context the model sees |
|---|---|
| `publication` | study name and description, the Crossref abstract for the publication DOI (fetched by the pipeline's own submission path), and the biosample records with their triad removed |
| `publication+supplements` | the same, plus `format_supplement_context(retrieve_supplements(doi))` (`nmdc_ai_eval.supplement_context`): an inventory of the kept files and the full text of the csv ones |

| Reference | Source |
|---|---|
| `supplement` | the authors' triad per sample in MOESM4, keyed by `sample_name` = biosample `name`. Underscored CURIEs, capitalised labels and a pipe-joined `env_medium` are normalized by `nmdc_ai_eval.env_triad_scoring`; a prediction is scored against whichever of several terms it sits closest to in ENVO |
| `nmdc` | the triad stored on each biosample in the snapshot |

## Scoring

Per suggestion, `nmdc_ai_eval.triad_output_scorer` applies the same ENVO relationship, hop
distance and composite formula as `envo_scorer` (see its docstring for weights). The enum term
comes from the suggestor's validation gate: `submission_enum` provenance means the value is in the
extension's curated set. Arm comparison (`changed` / `toward` / `away` per slot) comes from
`env_triad_scoring.compare_outputs`, graded by ENVO proximity rather than exact match.

Hierarchy proximity is ENVO-only. A PO term such as `leaf [PO:0025034]` shares no ancestors with
an ENVO reference, so it gets zero proximity; it still earns parse, label and enum credit, which is
where its 0.5 comes from. Read `env_medium` numbers with that in mind.

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

## Snapshot provenance

`data/phyllosphere_study.json` and `data/phyllosphere_biosamples.json` were fetched verbatim from
the NMDC runtime API on **2026-09-11**:

```bash
curl -A "nmdc-metadata-suggestor" \
  "https://api.microbiomedata.org/nmdcschema/study_set/nmdc:sty-11-e4yb9z58"
curl -A "nmdc-metadata-suggestor" \
  "https://api.microbiomedata.org/nmdcschema/biosample_set?filter=%7B%22associated_studies%22%3A%22nmdc%3Asty-11-e4yb9z58%22%7D&max_page_size=200"
```

The API returns HTTP 403 to requests with no `User-Agent` header (Python's `urllib` default), so
send one. The biosample records are the API's `resources` list, unmodified;
`provenance_metadata.mod_date` on every record was `2026-09-09`, so the triads reflect a curation
pass from two days before the fetch.

Every one of the 192 biosamples carries the same env triad, and every one of the three values
pairs a label with a CURIE that belongs to a different term in ENVO:

| Slot | Stored value | What the CURIE actually is |
|---|---|---|
| `env_broad_scale` | `agricultural biome [ENVO:01001442]` | `agriculture` |
| `env_local_scale` | `phyllosphere biome [ENVO:01001442]` | `agriculture` |
| `env_medium` | `plant-associated biome [ENVO:01001001]` | `plant-associated environment` |

The `nmdc` reference scores against these as "what is currently in NMDC". No valid suggestion can
exactly match a CURIE that is wrong for its label, so only the ontology-distance part of the score
can move there.

The publication DOI `10.1038/s41467-023-36515-y` (PMC9950430) yields twelve supplementary files
from Europe PMC. `41467_2023_36515_MOESM4_ESM.csv` (Supplementary Data 1) has one row per
biosample, keyed by `sample_name`, which equals the biosample `name`/`samp_name`, and gives the
authors' own triad: `Terrestrial Biome [ENVO_00000446]` / `Area of cropland [ENVO_01000892]` /
`agricultural soil [ENVO_00002259] | plant matter [ENVO_01001121]`. The table is fetched at run
time, not stored here. The award DOI `10.46936/10.25585/60000818` resolves to a funding record
with nothing to fetch.
