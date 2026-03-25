# Metadata Field Guidance Eval (Task 1)

Predict which biosample slots a submitter should fill, given only submission-level metadata.

## Ground truth

[`ground_truth.yaml`](ground_truth.yaml) — 6 hand-curated submissions from Montana Smith and Bea Meluch on [data-dev](https://data-dev.microbiomedata.org). Each lists the biosample slots the AI **should** recommend, with text-based justification from the study abstract.

Source: [microbiomedata/issues#1551](https://github.com/microbiomedata/issues/issues/1551)

## Data source

The full submission documents live in MongoDB:

```
mongodb://localhost:27017/nmdc_data_dev  (collection: nmdc_submissions)
```

Fetched via:

```bash
cd ~/gitrepos/external-metadata-awareness
make -f Makefiles/nmdc_metadata.Makefile nmdc-submissions-to-mongo-dev
```

## Why MongoDB, not a committed data file?

The suggestor pipeline (`run_recommendation_pipeline`) expects the **raw submission document** as input — the same shape returned by the portal API. This includes fields not present in the repo's flat TSV:

| Field | In TSV | In MongoDB | Used by suggestor |
|---|---|---|---|
| `studyForm.studyName` | yes | yes | yes |
| `studyForm.description` | yes | yes | yes |
| `studyForm.notes` | yes | yes | yes |
| `studyForm.dataDois` | no | yes | yes |
| `studyForm.publicationDois` | no | yes | yes |
| `studyForm.GOLDStudyId` | no | yes | yes |
| `multiOmicsForm.JGIStudyId` | no | yes | yes |
| `multiOmicsForm.*Protocols` | no | yes | yes |
| `packageName` / `templates` | partial (`sampleData` key) | yes | yes |
| `sampleData` (biosample values) | partial (select cols) | yes | **no** (Task 1 input is submission-level only) |

## Eval workflow

```
ground_truth.yaml
        │
        ▼
  Load expected_slots per submission_id
        │
        ▼
  Fetch full submission doc from nmdc_data_dev.nmdc_submissions
        │
        ▼
  run_recommendation_pipeline(submission_object=doc, llm_client=client)
        │
        ▼
  LLMOutput.metadata_fields → set of predicted field_names
        │
        ▼
  Score: precision, recall, F1 vs expected_slots
```

## Scoring

- **Precision**: of the slots the model recommends, how many are in the expected set?
- **Recall**: of the expected slots, how many did the model recommend?
- **F1**: harmonic mean

Note: the expected set is intentionally small (non-obvious, study-specific slots). Standard slots like `env_broad_scale`, `geo_loc_name`, `depth` are not listed as "expected" because any reasonable model will recommend them. The eval measures whether the model surfaces the **domain-specific** slots that require reading the abstract.

## Ground truth scope: slot selection only, NOT values

These submissions are ground truth for **which slots should be filled** (Task 1 / Metadata Field Guidance). They are **not** ground truth for the correctness of values in those slots. Montana and Bea filled in biosample values as examples, but the values themselves are not curated for accuracy — some contain placeholder or incorrect values (e.g. `env_broad_scale = "city [ENVO:00000856]"` in submission `d882c556` — the aerobiome/public transit study — where `city` is a valid ENVO term but not a biome). Do not use these submissions as ground truth for value prediction (Task 2 / Metadata Completion). For Task 2 eval, use the production submissions in the `ebs-prediction` and `submission-metadata-prediction` datasets.
