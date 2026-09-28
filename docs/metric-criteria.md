# Metric criteria for NMDC metadata suggestion evals

This answers the task recorded in the 2026-09-18 evals notes, "Mark to begin to create the metric criteria we will evaluate against", and the question in the same notes, "are we grading on a 1-4 scale? Yes or no? 1-10?"

It follows the first step those notes set: "evaluate the data we have, ask questions and be curious about what we have and where the llm might have gone wrong that can then become a metric." Every criterion below comes from something in the recorded agent runs rather than from a list of generic LLM metrics.

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

**The 7 label mismatches are two failure modes, not one.**

| Suggested | What the CURIE actually is | Kind |
|---|---|---|
| mountainous area [ENVO:00000218] | black smoker | asserts something false |
| tropical savanna biome [ENVO:01000186] | polar desert biome | asserts something false |
| boreal forest biome [ENVO:01000250] | subpolar coniferous forest biome | asserts something false |
| savanna soil [ENVO:00005750] | grassland soil | asserts something false |
| meadow [ENVO:00000108] | meadow ecosystem | wording |
| biofilm material [ENVO:00002034] | biofilm | wording |

A single accuracy number scores those the same. They are not the same. The validation gate repairs a wording near-miss, while a CURIE naming a different term poisons retrieval in both directions: a query for the right term misses the sample, and a query for the wrong term returns it.

**12 of 12 runs that were denied their tool calls reported success.** An output score cannot see this, because every value those runs produced was well formed.

## The organizing rule

Group criteria by what the score is checked against. A criterion is well defined when its definition is a computation, and it is separable from its neighbours when it is checked against a different thing. Criteria with nothing to check against are the ones published metric lists disagree about most.

## The criteria

### Tier 1, checked against a formal spec. No judge. Yes or no.

| Criterion | Computation | Evidence it is needed |
|---|---|---|
| parses as `label [CURIE]` | regex | 918 of 918 pass today, so this is a regression guard |
| CURIE exists in the ontology | ontology lookup | 918 of 918 pass today |
| label matches the ontology's label | string compare | 911 of 918, and the 7 are the interesting cases |
| value is in the slot's enum where one exists | schema lookup | the env triad slots use `any_of: [enum, pattern]`, so LinkML alone never enforces membership |
| a value conversion produces the expected output | apply the expression to sample rows | https://github.com/microbiomedata/nmdc-metadata-suggestor-ai-tool/pull/167 emits executable Python and nothing checks it |

There is no middle state for whether a CURIE exists.

### Tier 2, checked against ontology structure. No judge. 0 to 1.

Split every wrong answer into what it failed to say and what it said that was false.

| Answer, when the correct term is "cold water" | Missed information | False information |
|---|---|---|
| cold water | none | none |
| a subclass of cold water | none | some, if the extra detail is wrong |
| water | some, lost "cold" | none |
| hot water | some | high |

The current `envo_scorer` cannot make the last distinction: `check_relationship` returns `unrelated` for a sibling, so "hot water" and "basalt" score identically against "cold water". Its decay constants also penalize an ancestor more than a descendant, which is backwards under this reading, because an ancestor asserts only true things. I set those constants in e70c2ce on the reasoning that a vaguer answer is worse; this reading replaces that reasoning.

CAFA formalized exactly this split as remaining uncertainty against misinformation, and https://github.com/BioComputingUP/CAFA-evaluator implements it. Report the two components separately rather than only their combination.

### Tier 3, checked against the request. Needs a judge.

| Criterion | Question the judge answers |
|---|---|
| evidence support | does the text cited in `reason` support the value |
| slot selection | were slots filled that the input supports, and left empty where it does not |
| citation placement | is the citation attached to the right slot |

These use a frontier model as judge, from a different model family than the generator, as the 2026-09-18 notes decided. Grade on a 3-point scale with written anchors (supported, partially supported, unsupported), written before any judge runs. A 10-point scale on a judgement with no reference gives false precision and poor agreement between runs.

### Tier 4, checked against the trace. No judge. Counts.

| Criterion | Why |
|---|---|
| permission denials per run | 12 of 12 denied runs reported completed |
| did an ontology lookup succeed at least once | a run that never reached ENVO answered from its prompt |
| turns, cost, duration | already in `run_health()` |
| validation gate outcome: accepted, repaired, replaced | the gate already writes this and nobody reads it |

Always state the denominator.

## Coherence, the criterion worth the most work

Coherence is on the 2026-09-18 list of five, and Chris Mungall named it independently. I nearly cut it, reading it as prose quality checked against nothing. That was the wrong reading for this domain. Here it has concrete references, and it is the only criterion that catches a failure every other tier passes.

**A triad can be three individually valid values that cannot describe one sample.** Every tier 1 and tier 2 check passes on `marine biome` plus `forest floor` plus `soil`.

| Checked against | What it catches |
|---|---|
| the other two triad slots | a local feature that cannot sit inside that biome |
| the GOLD ecosystem path already on the record | `ecosystem_type: Plant-associated` against a marine biome |
| the rest of the sample's metadata | `host_common_name: switchgrass` against an aquatic medium |

The second is the cheapest and needs no judge, because the path is already on every biosample: `ecosystem`, `ecosystem_category`, `ecosystem_type`, `ecosystem_subtype`, `specific_ecosystem`.

**What I tried, and why it did not work.** I tested whether `env_local_scale` sits under `env_broad_scale` by `rdfs:subClassOf` across the 306 complete triads in the recorded runs: zero nested, 14 inverted, 292 with no path at all. That is an artifact. `env_broad_scale` takes biome terms and `env_local_scale` takes landscape features, which live in different ENVO branches, so subsumption was never the right relation. `tropical moist broadleaf forest biome` with `tropical forest`, and `tundra biome` with `area of tundra`, both read as coherent to a person.

So the open work on coherence is choosing the relation. Candidates: `part of` through Ubergraph, membership in the same curated interface value set (the four interfaces' `env_medium` sets are completely disjoint), or the GOLD ecosystem path.

**The model already does this reasoning, and it is scored by nothing.** In the supplement eval of https://github.com/microbiomedata/nmdc-metadata-suggestor-ai-tool/pull/171, it rejected the reference value with "'agricultural soil' is incorrect for a leaf sample, and 'plant matter' is too broad". That is a coherence judgement stated in `reason`.

Grade coherence on the tier 3 scale until the ontology relation is settled, then move whatever becomes computable down into tier 2.

## What we do not measure

Fluency and helpfulness. Both are checked against nothing here and neither is on the 2026-09-18 list.

## Slots orthogonal to the env triad

The other half of the 2026-09-18 assignment. Each needs no judge and no gold standard, so none waits on whether the curated triads can be trusted.

1. **Value conversion correctness.** Apply `ValueConversion.expression` to sample rows and compare with expected output. This is the highest-risk suggestion the tool makes, because `type='custom'` emits Python that the executor runs.
2. **Confidence calibration.** The metadata mapper already emits `high`, `review` and `cant_place`. Measure whether `high` is right more often than `review`, as an accuracy-by-bucket table.
3. **Provenance tier correctness.** `tier: submission_enum` asserts a value came from a curated set. Both agentic call sites passed `interface_names=None`, so the label could claim grounding it did not have. Checkable against the curated value sets with no model call.

## Decisions for the squad

1. Do we accept the four tiers, and that different tiers get different scales?
2. Do we adopt CAFA's split of missed information against false information for tier 2, or keep hand-set weights?
3. Which relation do we use for triad coherence: `part of` through Ubergraph, shared membership in a curated interface value set, or the GOLD ecosystem path?
4. Which of the three orthogonal slots do we do first?

## Open

Whether the curated env triad values are sound enough to teach a judge with. The NMDC reference for the phyllosphere study pairs every label with a CURIE for a different term, so no correct suggestion can exact-match it. That question is testable and is not answered here.
