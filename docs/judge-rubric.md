# Judge rubric for suggester output

`src/nmdc_ai_eval/rubrics/suggestion-judge-v1.yaml` is the set of questions an LLM judge answers about the metadata suggester's output. `src/nmdc_ai_eval/rubric.py` is its schema, and `tests/test_rubric.py` checks that every rubric file loads against it.

This is a proposal for the squad. The five criteria and the rule that the judge comes from a different model family than the generator are from the 2026-09-18 evals meeting. The rest is for the squad to accept or change: the yes/no scale, writing the criteria as a YAML file with a schema, one judge call per question with quoted evidence, and the calibration plan below.

## Scope

The whole pipeline output, not only `env_broad_scale`, `env_local_scale` and `env_medium`. That follows Olivia's comment on https://github.com/microbiomedata/nmdc-ai-eval/issues/61#issuecomment-5891682266: "we want a llm as a judge approach across the whole output and not necessarily just the env triad". The output is the suggester's `metadata_fields` list, where each entry has a `field_name`, a `value` and a `reason`.

## What is in it, and what is not

Five criteria, from the 2026-09-18 evals meeting: accuracy, factuality, relevancy, completeness and coherence. Each is one yes/no question, with a pass and a fail definition and one or two labelled examples. Three are asked once per suggestion and two once per output. Coherence is judged within each sample, using the suggester's optional `id` on sample-level suggestions.

The judge is given the schema's definition of each field in scope, so relevancy is judged against the definition rather than the judge's memory.

Anything code can check is left out. Whether a value parses, whether its CURIE exists and whether its label matches the ontology are lookups (`envo_scorer.py`). Permission denials and ontology lookups during a run are trace counts (`trace_report.py`). A judge would only add noise to those.

## Why a file and not a prose spec

https://github.com/microbiomedata/nmdc-ai-eval/pull/120 tried to settle the criteria in a prose document. Codex reviewed it six times and found 2, 2, 2, 2, 3 and 2 problems, mostly in text that an earlier fix had added. Its open questions moved to https://github.com/microbiomedata/nmdc-ai-eval/issues/123.

Tools that run judges keep the rubric as data: OpenAI Evals' model-graded templates (https://github.com/openai/evals/blob/main/docs/eval-templates.md), DeepEval's DAG metrics (https://deepeval.com/docs/metrics-dag), and CheckEval's yes/no checklists (https://arxiv.org/abs/2403.18771). A file can be versioned, validated and tested, and every score can record the version it came from.

Criteria also change once someone grades real outputs. Shankar et al. (https://arxiv.org/abs/2404.12272) found that "users need criteria to grade outputs, but grading outputs helps users define criteria". So v1 is a starting point for grading, not a finished spec.

## Why yes/no

A pilot in https://github.com/turbomam/local-llm-evals on 2026-09-28 scored the same fifteen answers with a yes/no checklist and with 3-point scales. Every answer got the top score on both 3-point scales, so the scales ranked nothing, while the checklist separated the models: https://github.com/turbomam/local-llm-evals/tree/main/results/scores/photosynthesis/20260928T160326Z

## How it is meant to be run

The rubric does not depend on a harness. It is not wired into `run_suite.py` or llm-matrix, which https://github.com/microbiomedata/nmdc-ai-eval/issues/118 plans to remove. A runner asks each question in its own judge call, because several criteria scored in one call share the judge's position and verbosity biases (https://arxiv.org/abs/2306.05685). It uses a judge from a different model family than the one that made the suggestions, and validates each reply as `JudgeAnswer`. A pass counts only when its evidence is found word for word in the input or output the judge was given.

## Next: calibrate against hand-graded outputs

Before its scores compare anything, the judge is checked against about 30 outputs graded by a person.

1. Pick about 30 recorded pipeline runs from the AI Suggester Agent Langfuse project, spread across MIxS interfaces, and include runs with known problems.
2. Export each run's input, fields in scope and output into one grading file per run.
3. Graders the squad agrees on grade each criterion pass or fail with a one-line reason, before seeing any judge answer.
4. Run the judge on the same outputs and report its agreement with the grades for each criterion. The statistic and its minimum are open in https://github.com/microbiomedata/nmdc-ai-eval/issues/123.
5. Where the graders could not decide, or the judge disagrees for one reason again and again, rewrite the question and bump the version.
