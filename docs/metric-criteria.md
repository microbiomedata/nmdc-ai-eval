# Metric criteria for NMDC metadata suggestion evals

This answers the task recorded in the 2026-09-18 evals notes, "Mark to begin to create the metric criteria we will evaluate against", and the question in the same notes, "are we grading on a 1-4 scale? Yes or no? 1-10?"

It leads with the direction those notes chose: "Start with llm as judge", using "different family of llm" from the one that made the suggestions. The judge grades with a structured rubric, one yes/no question per criterion, over the five criteria the notes name: accuracy, factuality, relevancy, completeness and coherence. Checks against a spec and counts from the trace stay underneath it as cheap guards. Scoring by distance from a curated answer becomes optional until the curated values are shown to be sound.

It also follows the notes' first step, "evaluate the data we have, ask questions and be curious about what we have and where the llm might have gone wrong that can then become a metric." Every question below comes from something in the recorded agent runs rather than from a list of generic LLM metrics.

## The measurement this is built on

Run on 2026-09-22 against the AI Suggester Agent Langfuse project, with no model calls and no curated answers. These are developer and test runs: 76 `local`, 7 `testing`, 1 `default`, none tagged `production`. The tooling is `src/nmdc_ai_eval/trace_report.py`, from https://github.com/microbiomedata/nmdc-ai-eval/pull/116.

| Measure | Value |
|---|---|
| env triad values suggested | 918 |
| parse as `label [CURIE]` | 918 of 918 |
| CURIE resolves in ENVO | 918 of 918 |
| label matches ENVO's own label | 911 of 918 |
| traces carrying a run health block | 25 of 84 |
| of those, hit a permission denial | 12 |
| of those, still reported `terminal_reason: completed` | 12 of 12 |
| worst single run | 205 denials across 214 turns, returned 3 values |

## The two things that went wrong, and what each becomes

**The 7 label mismatches are two failure modes, not one.** They are 6 distinct values; one was suggested twice.

| Suggested | What the CURIE actually is | Times | Kind |
|---|---|---|---|
| mountainous area [ENVO:00000218] | black smoker | 1 | asserts something false |
| tropical savanna biome [ENVO:01000186] | polar desert biome | 1 | asserts something false |
| boreal forest biome [ENVO:01000250] | subpolar coniferous forest biome | 1 | asserts something false |
| savanna soil [ENVO:00005750] | grassland soil | 1 | asserts something false |
| meadow [ENVO:00000108] | meadow ecosystem | 2 | wording |
| biofilm material [ENVO:00002034] | biofilm | 1 | wording |

A single accuracy number scores those the same. They are not the same. The validation gate repairs a wording near-miss, while a CURIE naming a different term poisons retrieval in both directions: a query for the right term misses the sample, and a query for the wrong term returns it.

**12 of 12 runs that were denied their tool calls reported success.** An output score cannot see this, because every value those runs produced was well formed.

## The organizing rule

Group criteria by what the score is checked against. A criterion is well defined when its definition is a computation, and it is separable from its neighbours when it is checked against a different thing. The judge's questions are written the same way: each names what the judge compares the suggestion with.

## The judge, first

### The rubric

One yes/no question per criterion, CheckEval style (https://arxiv.org/abs/2403.18771): a broad quality is broken into questions a judge can answer the same way each time. Each answer carries a short quote from the suggestion or the input as evidence, and "yes" with no evidence counts as "no".

| Criterion | Question the judge answers, per suggested slot value | Checked against |
|---|---|---|
| accuracy | Is this value right for this sample, given everything the input says? | the input |
| factuality | Is every claim in the suggestion's `reason` supported by the input text? | the input |
| relevancy | Does the value answer the slot it was suggested for, rather than a neighbouring slot? | the slot definition |
| completeness | Did the suggestion fill every slot the input gives evidence for? Asked once per sample. | the input |
| coherence | Can this value and the other two env triad values describe one sample? Asked once per triad. | the other suggested values |

The three questions an earlier draft listed as needing a judge fit inside these: evidence support is factuality, slot selection is completeness, and citation placement is relevancy for the slot the citation sits on.

Anchors are written before any judge runs, in the form "yes means ... ; no means ...", with one worked example of each drawn from the recorded runs, and are stored with a version string so every score records the rubric it was produced with.

### Why yes/no, not 1-4 or 1-10

This is the answer to the scale question. A pilot in https://github.com/turbomam/local-llm-evals on 2026-09-28 scored the same fifteen answers with a yes/no checklist and with 3-point anchored scales for relevancy and coherence. Every answer got the top score on both 3-point scales, so they ranked nothing; the yes/no checklist separated the models. A 10-point scale on a judgement with no reference gives more false precision, not less. Results: https://github.com/turbomam/local-llm-evals/tree/main/results/scores/photosynthesis/20260928T160326Z

The same pilot found the judge's count of false statements moved between versions of its prompt on the same text. So the judge is checked against a small set of hand labels, with the agreement reported, before its scores are used to compare anything.

### The judge's model family

Never the generator's family. The agentic path runs on Claude, so the judge is from another family, for example Gemini. https://github.com/turbomam/local-llm-evals already implements this: `config/judges.yaml` lists judges in order of preference and skips any from the generator's family, and the judge prompt is versioned (`judges/explainer-v3.yaml`), with the untrusted answer passed as data the judge is told not to follow. The NMDC rubric can reuse that code with its own prompt.

## Cheap guards: checks against a spec, no judge, yes or no

These run on every suggestion before the judge does, and cost nothing. A value that does not parse, or whose CURIE does not exist, is scored as failed and not sent to the judge. A label mismatch is not fatal: it is recorded as its own count, and the judge sees the value with the ontology's label for that CURIE, the way the validation gate repairs it. That keeps the two kinds of mismatch apart: a wording near-miss can still be judged accurate, while a label attached to the wrong term is judged against the term the CURIE actually names.

| Check | Computation | Evidence it is needed |
|---|---|---|
| parses as `label [CURIE]` | regex | 918 of 918 pass today, so this is a regression guard |
| CURIE exists in the ontology | ontology lookup | 918 of 918 pass today |
| label matches the ontology's label | string compare | 911 of 918, and the 7 are the interesting cases |
| value is in the slot's enum where one exists | schema lookup | the env triad slots use `any_of: [enum, pattern]`, so LinkML alone never enforces membership |
| a value conversion produces the expected output | apply the expression to sample rows | https://github.com/microbiomedata/nmdc-metadata-suggestor-ai-tool/pull/167 emits executable Python and nothing checks it |

There is no middle state for whether a CURIE exists.

## Cheap guards: counts from the trace, no judge

| Count | Why |
|---|---|
| permission denials per run | 12 of 12 denied runs reported completed |
| did an ontology lookup succeed at least once | a run that never reached ENVO answered from its prompt |
| turns, cost, duration | already in `run_health()` |
| validation gate outcome: accepted, repaired, replaced | the gate already writes this and nobody reads it |

Always state the denominator. A run that fails these is reported as a failed run, not scored by the judge.

## Optional: distance from a curated answer

Where a sample has a curated value, the suggestion can also be scored by its ontology distance from it. This stays optional until the curated values are shown to be sound (see Open, below): the reference for the phyllosphere study pairs every label with a CURIE for a different term, and the `ebs-prediction` suite carries the wrong CURIE for the epipelagic biome in 10 of 100 cases (https://github.com/microbiomedata/nmdc-ai-eval/issues/9).

When it is used, split every wrong answer into what it failed to say and what it said that was false.

| Answer, when the correct term is "cold water" | Missed information | False information |
|---|---|---|
| cold water | none | none |
| a subclass of cold water | none | some, if the extra detail is wrong |
| water | some, lost "cold" | none |
| hot water | some | high |

The current `envo_scorer` cannot make the last distinction: `check_relationship` returns `unrelated` for a sibling, so "hot water" and "basalt" score identically against "cold water". Its decay constants also penalize an ancestor more than a descendant, which is backwards under this reading, because an ancestor asserts only true things. I set those constants in e70c2ce on the reasoning that a vaguer answer is worse; this reading replaces that reasoning.

CAFA formalized exactly this split as remaining uncertainty against misinformation, and https://github.com/BioComputingUP/CAFA-evaluator implements it. Report the two components separately rather than only their combination.

## Coherence, the criterion worth the most work

Coherence is on the 2026-09-18 list of five, and Chris Mungall named it independently. I nearly cut it, reading it as prose quality checked against nothing. That was the wrong reading for this domain. Here it has concrete references, and it is the only criterion that catches a failure every other check passes.

**A triad can be three individually valid values that cannot describe one sample.** Every spec check passes on `marine biome` plus `forest floor` plus `soil`.

| Checked against | What it catches |
|---|---|
| the other two triad slots | a local feature that cannot sit inside that biome |
| the GOLD ecosystem path already on the record | `ecosystem_type: Plant-associated` against a marine biome |
| the rest of the sample's metadata | `host_common_name: switchgrass` against an aquatic medium |

The second is the cheapest and needs no judge, where the path is on the record: `ecosystem`, `ecosystem_category`, `ecosystem_type`, `ecosystem_subtype`, `specific_ecosystem`. It is not always there. In the 5,052 rows of `datasets/submission-metadata-prediction/eval_input_target_pairs.tsv`, `ecosystem` is blank in 74 and `ecosystem_category` in 308 (counted 2026-09-29). So this check applies only where the fields it needs are present, and samples without them stay in the denominator, reported as not checkable rather than dropped.

**What I tried, and why it did not work.** I tested whether `env_local_scale` sits under `env_broad_scale` by `rdfs:subClassOf` across the 306 complete triads in the recorded runs: zero nested, 14 inverted, 292 with no path at all. That is an artifact. `env_broad_scale` takes biome terms and `env_local_scale` takes landscape features, which live in different ENVO branches, so subsumption was never the right relation. `tropical moist broadleaf forest biome` with `tropical forest`, and `tundra biome` with `area of tundra`, both read as coherent to a person.

So coherence starts as a judge question, and moves to a computed check once a relation is chosen. Candidates: `part of` through Ubergraph, membership in the same curated interface value set (the four interfaces' `env_medium` sets are completely disjoint), or the GOLD ecosystem path.

**The model already does this reasoning, and it is scored by nothing.** In the supplement eval of https://github.com/microbiomedata/nmdc-metadata-suggestor-ai-tool/pull/171, it rejected the reference value with "'agricultural soil' is incorrect for a leaf sample, and 'plant matter' is too broad". That is a coherence judgement stated in `reason`.

## What we do not measure

Fluency and helpfulness. Both are checked against nothing here and neither is on the 2026-09-18 list.

## Slots orthogonal to the env triad

The other half of the 2026-09-18 assignment. None needs a curated metadata value, so none waits on whether the curated triads can be trusted.

1. **Value conversion correctness.** Running a `ValueConversion.expression` only shows what it does, so this needs expected outputs: a small hand-written set of input and output pairs for each conversion type (a date format, a unit scale factor, a delimiter), independent of any submission. With those, apply the expression and compare. This is the highest-risk suggestion the tool makes, because `type='custom'` emits Python that the executor runs. No judge needed.
2. **Confidence calibration.** The metadata mapper sorts each mapping into `high_confidence`, `needs_review` or `cant_place`. Whether `high_confidence` is right more often than `needs_review` needs an outcome for each mapping: a curator's accept or reject, which the planned thumbs-up signal in Langfuse would record, or the judge's accuracy verdict. Until one exists, report only how mappings are distributed across the three buckets.
3. **Provenance tier correctness.** `tier: submission_enum` asserts a value came from a curated set. Both agentic call sites passed `interface_names=None`, so the label could claim grounding it did not have. Checkable against the curated value sets with no model call.

## Decisions for the squad

1. Do we accept the rubric's five questions and their yes/no scale, with the spec and trace checks as guards run first?
2. Who labels the small hand set the judge is checked against, and how many samples?
3. Do we adopt CAFA's split of missed information against false information for the optional distance score, or keep hand-set weights?
4. Which relation do we use for triad coherence: `part of` through Ubergraph, shared membership in a curated interface value set, or the GOLD ecosystem path?
5. Which of the three orthogonal slots do we do first?

## Open

Whether the curated env triad values are sound enough to use as reference answers. The NMDC reference for the phyllosphere study pairs every label with a CURIE for a different term, so no correct suggestion can exact-match it. That question is testable and is not answered here.
