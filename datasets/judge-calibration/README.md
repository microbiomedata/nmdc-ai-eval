# Judge calibration set: synthetic submissions with graded suggester outputs

Ten invented NMDC-style submissions, each with three hand-written suggester outputs (good, moderate, bad) and the verdicts the judge rubric `suggestion-judge-v1` (https://github.com/microbiomedata/nmdc-ai-eval/pull/125) should give. That rubric says to calibrate against about 30 hand-graded outputs before trusting judge scores; this is a first such set, written on 2026-10-16 so that calibration does not wait on collecting and grading 30 real pipeline outputs.

## Layout

```
cases.tsv                       one row per case: folder, package, schema_fit_rank, schema_fit_why, unsupported_in_scope
outputs.tsv                     one row per output: case, grade, counts, why it earns that grade
cases/NN-<package>-<topic>/     NN is the schema_fit rank, 01 easiest to 10 hardest
  input/study.txt               study name and description, as a submitter would write them
  input/samples.tsv             id, samp_name, then one empty column per field in scope
  outputs/{good,moderate,bad}.json   the suggester output to grade, in the suggester's LLMOutput shape
  outputs/{good,moderate,bad}.tsv    a view of the same JSON: id, field_name, value, reason
  expected/{good,moderate,bad}.tsv   the answer key: kind (fail or open), criterion, row, field_name, id, why
```

`outputs/*.json` is what the judge sees, and the JSON is the source; the TSV beside it is a view and the test checks they agree. `expected/*.tsv` is what the judge should say about the output; the judge never sees it. Outputs the suggester itself produces go in the same folder as `outputs/suggester-<provider>-<model>-<timestamp>.json`.

## What is in it

- 10 inputs, 30 outputs, 219 suggestions, 120 expected fails, 26 open questions (counted on 2026-10-16 after two review passes).
- The inputs are in the style of submission-portal studies NMDC has handled: fertilized bioenergy crop soil, warmed peat, a stratified lake, estuarine sediment, greenhouse Populus roots, an insect gut, transit air, post-fire forest soil, contaminated groundwater and activated sludge. All names, dates, places and numbers are made up. Nothing is copied from a real submission or produced by the suggester.
- `env_broad_scale`, `env_local_scale` and `env_medium` are left out on purpose. Those slots already have their own eval sets in this repo; this set exercises the other biosample slots.
- The columns of `input/samples.tsv` after `samp_name` are the fields in scope, the list the judge is told about for completeness. Real traces do not record scope (see the 2026-10-16 judge run notes), so this set supplies it. Completeness is judged per field, not per sample, as in the rubric's own example.
- Every study text defines what its sample names encode (treatment codes, depth intervals, dates), so a per-sample value read off a sample name counts as supported by the input. Depths are given in the schema's unit (meters, numeric only) even where the text says centimeters; the conversion is treated as supported because the field definition the judge sees demands it.
- A `fail` row in `expected/*.tsv` names one criterion and, for accuracy, factuality and relevancy, the one suggestion it applies to: `row` is the 1-based position in `metadata_fields`, and `field_name` and `id` repeat what that row holds so the key reads on its own. Completeness and coherence rows apply to the whole output and leave `row` empty. Any (criterion, suggestion) pair with no row is an expected pass; accuracy, factuality and coherence were audited by hand on 2026-10-16 after two review passes, and the completeness rule below is enforced by the test.
- `unsupported_in_scope` in `cases.tsv` lists the in-scope fields the input gives no per-sample value for (a range across samples, a group mean, a yearly average, a Latin name only). Omitting one is a pass. Suggesting one with a number that is in the input but stated for a group (a range endpoint, a mean, an influent average) is marked open; suggesting a number that appears nowhere in the input (a midpoint) is a fail. For every other in-scope field, an output that omits it has an expected completeness fail, and the test checks this both ways.
- An `open` row marks a suggestion whose verdict depends on a rubric decision not yet made: whether a value or reason may use background knowledge (a taxon id for a Latin name, a common name, natural ventilation from open windows); whether a group mean or range endpoint supports one sample; whether a value that is true but filed under the wrong field also fails accuracy, or only relevancy; and whether a general claim in a reason ("estuarine sediments are anoxic") counts as unsupported. Open rows are not counted as expected fails.
- `schema_fit_rank` in `cases.tsv`, and the folder prefix, order the inputs from easiest (1) to hardest (10) to encode in nmdc-submission-schema, with a one-line reason. The ranking is one person's judgment, not measured. The current ten sit mostly in the middle: there is no trivially easy case (one sample, literal lat_lon, date and depth) and no case whose main variables have no slot at all (for example a stable isotope probing design with labelled substrate, incubation time and density fractions). Those two would be the next additions.

## Sending an input to the suggester

`run_suggester.py` builds a submission-portal-shaped object from `input/study.txt` and `input/samples.tsv`, calls `run_recommendation_pipeline` (the single-pass path the portal uses) and saves the LLMOutput JSON plus a TSV view next to the hand-written outputs. Because that path reads only the study form, the sample sheet is also written into `studyForm.notes` as one line (`Samples: s1 samp_name "..."; s2 ...`), the same line the hand-written outputs were graded against.

```
just judge-calibration-dry-run 04-soil-miscanthus-fertilization        # prints the object, makes no request
just judge-calibration-run 04-soil-miscanthus-fertilization gcp gemini-2.5-flash
```

Credentials load in-process from the repo's `.env` and are never printed.

## Known limits

- The single-pass path reads only the study form and the package name (`get_submission_fields` in the suggester); sample rows are read only by the env triad path. The runner passes the sample sheet through `studyForm.notes` so the model sees the sample names, but its outputs are still study-level suggestions with no sample `id`, while the hand-written outputs give per-sample ids. Per-sample outputs match the suggester's agentic and env triad paths, not this one.
- The suggester fills `value` only when the submission states it literally; an inferred value goes into `reason` and `value` stays empty, by design (its README). The hand-written good outputs compose values such as `USA: Illinois, Urbana`. Before real outputs are scored against the expected keys, the rubric needs a rule for empty values; the 2026-10-16 judge run skipped them.
- String value formats (for example the MIxS compound format of `agrochem_addition`) are not checked.
- One slot carries `label [CURIE]` values because its pattern requires one: host_body_site in the termite case (`hindgut [UBERON:0001046]`, and `foregut [UBERON:0001041]` in the bad output). Both labels were checked in OLS on 2026-10-16. Nothing else exercises the exact-term check.

## How it was checked

`tests/test_judge_calibration.py` checks, with `nmdc-submission-schema` from this repo's lock file and `SchemaView`: every case folder is indexed and ranked; every in-scope column and every suggested `field_name` is a slot of the named package's interface (SoilInterface, WaterInterface, SedimentInterface, PlantAssociatedInterface, AirInterface, HostAssociatedInterface); every value matches the slot as induced on that interface (enum membership, float range and the slot's regular expression, so no value the portal would reject is a suggestion here); every output parses as the suggester's `LLMOutput`; each TSV view matches its JSON; every per-suggestion verdict resolves to exactly one row of that output and output-level verdicts to none; a completeness fail is listed exactly when a supported in-scope field is omitted; the index counts match; and the runner builds a submission object for every case without a request.

## How to use it

Feed each output to the judge the way the 2026-10-16 run over real traces did: the input text (study.txt plus the samples line the runner builds), the scope list with the schema's definitions, and one suggestion per call (or the whole output for completeness and coherence). Compare the judge's verdicts with `expected/*.tsv`. Agreement on the 120 expected fails and on the passes is the calibration number; disagreement on `open` rows is a rubric decision to make, not a judge error.
