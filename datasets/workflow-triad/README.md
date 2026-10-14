# Full-workflow env triad eval (three NMDC studies)

How close does the suggestor's env triad workflow come to the triads curators and submitters
stored in NMDC, and how much does the workflow's study context (study text, DOI lookups, PDFs)
add over the biosample records alone? Modeled on the
[supplement-triad eval](../supplement-triad/README.md), but scoring the whole pipeline across
three sources rather than one study's supplements.

## Test cases

| Key | Study | Source | Biosamples | Distinct stored triads | Portal package(s) |
|---|---|---|---|---|---|
| `1000-soils` | `nmdc:sty-11-28tm5d36`, 1000 Soils Research Campaign | soil | 139 | 29 | `soil` |
| `geodes-lakes` | `nmdc:sty-11-fkbnah04`, GEODES lakes, Wisconsin | freshwater | 119 | 4 | `water` |
| `crop-rhizosphere` | `nmdc:sty-11-prtb4s28`, Arabidopsis, maize, Boechera and miscanthus rhizosphere | plant-associated | 89 | 3 | `plant-associated`, `soil` |

The cases are defined in `studies.py`. Snapshots of each study and its biosamples from the NMDC
API (2026-10-14) are in `data/`. Regenerate them with `fetch_snapshots.py`; doing so changes the
gold standard, so commit the new snapshot and update the date here.

None of the three came through the submission portal. `submission_object_for` builds the portal
shaped object the pipeline takes from the study record: name, description, every associated DOI
under its portal field with the record's `doi_provider`, and the package(s).

### What the model does not see

`withhold` removes, from each biosample record:

- the triad itself;
- GOLD's ecosystem path (`ecosystem` … `specific_ecosystem`). The portal does not collect it, and
  here it names the answer: GEODES' `Epilimnion` subtype, the rhizosphere study's
  `Rhizosphere` / `Endosphere` types;
- identifiers and bookkeeping NMDC adds on ingest (`associated_studies`, `provenance_metadata`,
  GOLD/IMG/INSDC/EMSL ids, …).

Sample names stay. For GEODES they are GOLD-style and descriptive ("Oligotrophic lake water
microbial communities from Sparkling Lake…"), which is a strong cue for `env_local_scale`.
Treat GEODES' local-scale score as an upper bound for that reason.

### Why these three

The goal was varied sources, stored triads that vary within each study, and DOIs for the
workflow to use. A survey of all 92 NMDC studies on 2026-10-14 found few that meet all three.
Candidates considered and set aside:

| Study | Why not |
|---|---|
| `nmdc:sty-11-h1m9nj62` Arctic MOSAiC (seawater, sea ice) | 4 triads, broad and local scale constant for 289/291; GOLD fields map 1:1 onto `env_medium`; 17 sediment-trap samples stored as `sea water`; one id/label mismatch; no DOIs |
| `nmdc:sty-11-34xj1150` NEON soil metagenomes | 6,489 biosamples; `env_medium` is `soil` throughout; mostly `terrestrial biome`; triads look assigned from land-cover class, not curated; no DOIs |
| `nmdc:sty-11-2c3s5473` MicroFlora Danica | the richest agricultural gold (58 triads), but each biosample carries only an id, a date and a country: nothing to infer a triad from without the paper's sample table |
| `nmdc:sty-11-5tgfr349` rivers worldwide | 5 triads, no publication DOI; a reasonable water alternative |
| `nmdc:sty-11-pzmd0x14` NEON benthic | varied `env_medium`, but NEON-style assigned triads and no DOIs |
| `nmdc:sty-11-547rwq94` EMP500 | very varied, but a sampler of everything rather than one source; 178 sterile-water controls |

### Known limits of the gold

- **1000 Soils holds two kinds of triad.** The 72 cores taken at NEON sites carry
  `terrestrial biome | <land-cover area> | soil`. The other 67 carry the submitters' specific
  choices (`subalpine biome`, `forest soil`, …). The report scores the two separately.
- **`env_medium` is constant in GEODES** (`lake water`), so it tests little there.
- **The rhizosphere study's soil samples have `land` as `env_local_scale`**, a weak term.
- GEODES and rhizosphere raw values are bare CURIEs (`ENVO_01000252`);
  `env_triad_scoring.parse_triad_value` reads them.

## Two arms, one reference

| Arm | What the pipeline gets |
|---|---|
| `samples-only` | the withheld biosample records; no submission object, so no study text or DOI context |
| `full-workflow` | the same records plus the submission object: study text, context fetched for every DOI, PDFs when a DOI yields one, all through the ENVO gate |

Both arms pass `interface_names` (the case's packages), so the gate and the ENVO expansion list
are scoped the same way in both; the arms differ only in study context.

Reference: the triad stored on each biosample in the snapshot.

### What the DOI step actually retrieves

Probed 2026-10-14 (the run records this per study as `doi_probes`): every DOI returns an abstract
or dataset description of about 1–3k characters, except the JGI award DOIs (under 30 characters).
None yields a PDF link, so the PDF step does not run for these studies. The rhizosphere study's two
MassIVE DOIs return the same text.

## Scoring

Same as the supplement-triad eval: `nmdc_ai_eval.triad_output_scorer` per suggestion (ENVO
relationship, hop distance, composite score), `env_triad_scoring.coverage` first so dropped ids
are not read as wrong answers, and `compare_outputs` graded by ENVO proximity for the arm delta.
Results are reported per study and pooled over all samples.

Hierarchy proximity is ENVO-only. A PO term (`rhizosphere [PO:…]`-style answers for root samples)
shares no ancestors with an ENVO reference and gets zero proximity, though it still earns parse,
label and enum credit.

## Running

```bash
just eval-workflow-triad                                  # gcp, gemini default, chunk size 25
just eval-workflow-triad --limit 8 --chunk-size 8         # smoke run
just eval-workflow-triad --studies geodes-lakes           # one study
just eval-workflow-triad --provider gcp --model gemini-2.5-pro
```

Needs the suggestor's Vertex credentials (`GOOGLE_APPLICATION_CREDENTIALS`, `VERTEX_PROJECT_ID`;
see [docs/auth.md](../../docs/auth.md)). Results go to `pipeline-results/`: one timestamped YAML
per run (every arm's output with the model's reasons, scores, coverage, DOI probes, deltas), a
`_scores.tsv` with one row per scored suggestion, a `summary.tsv` row and a `report.md`.
