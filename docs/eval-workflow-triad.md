# Does the suggestor's full workflow improve env triad suggestions?

Three NMDC studies from different sources, scored against the triads stored in NMDC:
the 1000 Soils Research Campaign (`nmdc:sty-11-28tm5d36`, soil, 139 biosamples), GEODES lakes
(`nmdc:sty-11-fkbnah04`, freshwater, 119), and a crop rhizosphere study
(`nmdc:sty-11-prtb4s28`, plant-associated, 89). Modeled on the
[supplement eval](eval-phyllosphere-supplements.md).

## Where the pieces live

| File | Holds |
|---|---|
| `datasets/workflow-triad/studies.py` | the three cases, snapshot loaders, the submission object built from each study record, the fields withheld from the model |
| `datasets/workflow-triad/data/` | NMDC API snapshots of each study and its biosamples (2026-10-14) |
| `datasets/workflow-triad/fetch_snapshots.py` | refreshes the snapshots |
| `datasets/workflow-triad/run_workflow_eval.py` | runs both arms per study through `get_env_triad_recommendation`, probes the DOIs, scores, writes results |
| `src/nmdc_ai_eval/env_triad_scoring.py`, `triad_output_scorer.py` | parsing, coverage, per-suggestion ENVO scoring and arm comparison, shared with the supplement eval |
| `datasets/workflow-triad/pipeline-results/` | the runs |

Design, the reasons for choosing these studies, and the candidates set aside are in the
[README](../datasets/workflow-triad/README.md).

## Design in brief

| Arm | What `get_env_triad_recommendation` gets |
|---|---|
| `samples-only` | the biosample records with the triad, GOLD's ecosystem path and NMDC bookkeeping removed; no submission object |
| `full-workflow` | the same records plus a submission object from the study record: name, description, every associated DOI with its provider, the package(s) |

Both arms pass the same `interface_names`, so the ENVO gate and expansion list are scoped
identically. The arms differ only in study context. The reference is the triad stored in NMDC.

## Result (chunk size 25, 2026-10-15, gemini-2.5-flash via Vertex)

Coverage was complete: 347/347 samples answered in both arms, no suggestions without an id, no
chunk errors. Run time was 36 minutes.

Mean ontology score (exact-match rate in parentheses):

| Slot | samples-only | full-workflow |
|---|---|---|
| env_broad_scale | 0.892 (50%) | 0.884 (35%) |
| env_local_scale | 0.532 (14%) | 0.537 (15%) |
| env_medium | 0.839 (53%) | 0.752 (43%) |

| Study | Slot | samples-only | full-workflow |
|---|---|---|---|
| 1000-soils | env_broad_scale | 0.745 (3%) | 0.742 (3%) |
| 1000-soils | env_local_scale | 0.536 (22%) | 0.557 (25%) |
| 1000-soils | env_medium | 0.934 (34%) | 0.891 (37%) |
| geodes-lakes | env_broad_scale | 1.000 (100%) | 1.000 (100%) |
| geodes-lakes | env_local_scale | 0.576 (15%) | 0.576 (15%) |
| geodes-lakes | env_medium | 0.861 (72%) | 0.735 (47%) |
| crop-rhizosphere | env_broad_scale | 0.978 (56%) | 0.950 (0%) |
| crop-rhizosphere | env_local_scale | 0.466 (0%) | 0.456 (0%) |
| crop-rhizosphere | env_medium | 0.658 (57%) | 0.560 (45%) |

The full report, with per-study relationships, suggested values and arm deltas, is in
`datasets/workflow-triad/pipeline-results/report.md`.

## Reading it

**The study context did not bring the model closer to NMDC's triads.** It was flat on
`env_local_scale`, slightly lower on `env_broad_scale`, and lower on `env_medium` (0.839 to
0.752). Where it changed answers, it mostly moved them toward terms that are more specific or
more descriptive of the study, and that the stored triad does not use:

- **GEODES `env_medium`**: 36 answers changed, 33 of them away from NMDC. `lake water` became
  `oligotrophic water` or `eutrophic water`, terms the lake type supports and that ENVO places
  apart from `lake water`, so the scorer treats them as unrelated.
- **Rhizosphere `env_broad_scale`**: all 50 `terrestrial biome` answers became
  `anthropogenic terrestrial biome`. The study covers crops and growth-chamber plants, so the
  change is defensible, but it loses the exact match.
- **1000 Soils `env_medium`**: the context introduced `surface soil` (30 answers), replacing
  habitat soils such as `forest soil`. The model described where in the core the sample came from
  rather than the soil type. On the 67 samples with submitter-chosen triads, `env_medium` fell
  from 0.902 to 0.798; on the 72 cores taken at NEON sites (stored as plain `soil`) it rose
  slightly.

The context the workflow added was modest. None of the 12 DOIs yields a PDF link, so no paper
reached the model; each DOI gave an abstract or dataset description of about 1–3k characters, and
the JGI award DOIs gave under 30.

**The validation gate overrides the stored gold for rhizosphere samples.** NMDC stores
`rhizosphere [ENVO:00005801]` as `env_medium` for this study's rhizosphere samples. The model
answered `rhizosphere` for both `env_local_scale` and `env_medium`. The gate does not allow one
term in two slots, keeps it for local scale, and replaces the medium with
`environmental material [ENVO:00010483]`, the most generic term available. Every
`environmental material` answer in the run comes from that replacement (provenance
`generalized` / `replaced`): 38 samples in `samples-only`, 49 in `full-workflow`. This is tool
behavior, not model behavior, and it conflicts with how NMDC itself stores these samples. It is
worth raising with the suggestor.

**`env_local_scale` is weak everywhere (about 0.53), and much of that is how it is scored.**
Hierarchy credit goes only to exact, ancestor and descendant matches. Siblings score zero:
`freshwater lake` against a stored `oligotrophic lake` or `humic lake` is "unrelated", although
both sit under `lake`. In the rhizosphere study the stored local scale is a PO term
(`root system [PO:0025025]`) or the weak `land`, so no ENVO answer can earn hierarchy credit at
all. A sibling-aware score, or a check by a curator, would change this slot's numbers more than
any model change would.

**`env_broad_scale` in 1000 Soils scores low (0.745) mostly because of the gold.** On the 72 NEON-site
cores, stored as plain `terrestrial biome`, the model's specific biomes are descendants and score
0.87–0.89. On the 67 submitter-chosen triads it scores about 0.6, with 50 answers unrelated to the
submitter's biome.

## What this does not cover

- **One model.** Only gemini-2.5-flash was run. The runner takes `--model` for others.
- **No PDFs.** The PDF step never fired for these studies, so this says nothing about it.
- **NMDC's triad is the only reference.** Where the model's answer is arguably better than what
  is stored (`anthropogenic terrestrial biome` for growth-chamber plants, `oligotrophic water` for
  an oligotrophic lake), the score counts it against the model.
- **Descriptive sample names.** GEODES names state the lake type, so its `env_local_scale` is an
  upper bound on what the model can infer from records alone.
- **One run per arm.** Run-to-run variation was not measured; small differences (0.532 vs 0.537)
  should not be read as changes.
