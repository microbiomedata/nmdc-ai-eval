# Judge calibration set: synthetic submissions with graded suggester outputs

Ten invented NMDC-style submissions, each with three hand-written suggester outputs (good, moderate, bad) and the verdicts the judge rubric `suggestion-judge-v1` (https://github.com/microbiomedata/nmdc-ai-eval/pull/125) should give. That rubric says to calibrate against about 30 hand-graded outputs before trusting judge scores; this is a first such set, written on 2026-10-16 so that calibration does not wait on collecting and grading 30 real pipeline outputs.

## Layout

```
cases.tsv                       one row per case: folder, package, schema_fit_rank, schema_fit_why
outputs.tsv                     one row per output: case, grade, counts, why it earns that grade
cases/NN-<package>-<topic>/     NN is the schema_fit rank, 01 easiest to 10 hardest
  input/study.txt               study name and description, as a submitter would write them
  input/samples.tsv             id, samp_name, then one empty column per field in scope
  outputs/{good,moderate,bad}.tsv    the suggester output to grade: id, field_name, value, reason
  expected/{good,moderate,bad}.tsv   the answer key: kind (fail or open), criterion, field_name, id, why
```

`outputs/*.tsv` is what the judge sees. `expected/*.tsv` is what the judge should say about it; the judge never sees it.

## What is in it

- 10 inputs, 30 outputs, 226 suggestions, 92 expected fails, 7 open questions (counted by the check below).
- The inputs are in the style of submission-portal studies NMDC has handled: fertilized bioenergy crop soil, warmed peat, a stratified lake, estuarine sediment, greenhouse Populus roots, an insect gut, transit air, post-fire forest soil, contaminated groundwater and activated sludge. All names, dates, places and numbers are made up. Nothing is copied from a real submission or produced by the suggester.
- `env_broad_scale`, `env_local_scale` and `env_medium` are left out on purpose. Those slots already have their own eval sets in this repo; this set exercises the other biosample slots.
- The columns of `input/samples.tsv` after samp_name are the fields in scope, the list the judge is told about for completeness. Real traces do not record scope (see the 2026-10-16 judge run notes), so this set supplies it.
- A `fail` row in `expected/*.tsv` names one (criterion, field_name, id) that should fail. Everything else in that output should pass.
- An `open` row marks a suggestion whose verdict depends on a rubric decision not yet made, mainly whether a value or reason may use background knowledge (a taxon id for a Latin name, a common name), and whether a group mean or range endpoint counts as support for one sample. They are not counted as expected fails.
- `schema_fit_rank` in `cases.tsv`, and the folder prefix, order the inputs from easiest (1) to hardest (10) to encode in nmdc-submission-schema, with a one-line reason. The ranking is one person's judgment, not measured. The current ten sit mostly in the middle: there is no trivially easy case (one sample, literal lat_lon, date and depth) and no case whose main variables have no slot at all (for example a stable isotope probing design with labelled substrate, incubation time and density fractions). Those two would be the next additions.

## How it was checked

On 2026-10-16, with `nmdc-submission-schema` from this repo's lock file and `SchemaView`: every `field_name` in every output and every in-scope column is a slot of the named package's interface (SoilInterface, WaterInterface, SedimentInterface, PlantAssociatedInterface, AirInterface, HostAssociatedInterface); every value for an enum-ranged slot is a permissible value; every float-ranged slot has a numeric value; every expected fail names a field present in that output. Value formats for string slots (for example the MIxS compound format of `agrochem_addition`) were not checked.

## How to use it

Feed each output to the judge exactly as `local/judge-run-2026-10-16/run_judge.py` fed trace outputs: input text, the scope list with schema definitions, and one suggestion (or the whole output for completeness and coherence). Compare the judge's verdicts with `expected/*.tsv`. Agreement on the 92 expected fails and on the passes is the calibration number; disagreement on `open_questions` is a rubric decision to make, not a judge error.
