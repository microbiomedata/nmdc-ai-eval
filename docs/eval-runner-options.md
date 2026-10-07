# Eval runner options: what we use from llm-matrix, and what else exists

Written 2026-09-29 as input for https://github.com/microbiomedata/nmdc-ai-eval/issues/118 (Refactor to remove `llm_matrix` dependency), which @JamesTessmer owns, and for the judge rubric in https://github.com/microbiomedata/nmdc-ai-eval/pull/125. It records evidence and options. It is not a plan for #118.

How each claim is known is marked: **read** (read from source, not run), **ran** (run locally on 2026-09-29), or **mock** (run locally with mock models, no real model calls).

## What this repo uses from llm-matrix

All **read** from `main` at 9c77e3c (after https://github.com/microbiomedata/nmdc-ai-eval/pull/112 merged).

- Four names, all in `src/nmdc_ai_eval/run_suite.py`: `LLMRunner` (imported at line 28, used at 393), `load_suite` (29, 328), `results_to_dataframe` (29, 474), and `LLMRunnerConfig` (363, 381). Four test files also import it.
- The committed suites use one prompt template and one of llm-matrix's scoring methods (`simple_question`), over a matrix of models × `temperature: [0.0]`.
- The parts with real logic: the suite YAML schema, a DuckDB result cache (the dataset READMEs already warn that it can serve stale results), and the judge call.
- When the judge's reply can't be parsed, llm-matrix raises and the row is lost (`run_suite.py:405-416`, which counts and logs it).
- For env-triad suites the judge is still called, then its score is replaced by the direct per-field comparison (`run_suite.py:422-426`), so that call is spent without being used.
- In `uv.lock`, `diskcache` 5.6.3, `duckdb` 1.5.5 and `typer` 0.12.5 are required only by llm-matrix. `diskcache` is the subject of https://github.com/microbiomedata/nmdc-ai-eval/issues/27.

## Experiments

1. **ran**: the test suite on `main` passes, 242 tests.
2. **ran**: with `llm_matrix` imports blocked, 202 tests pass and 5 test files fail to load. That includes all the env-triad scorer tests, only because the scorer lives in `run_suite.py` (https://github.com/microbiomedata/nmdc-ai-eval/issues/113).
3. **ran**: a 153-line throwaway stand-in for the four llm-matrix names; all 242 tests pass.
4. **mock**: `run_suite` on 6 EBS cases, real llm-matrix against the stand-in. The scores were the same, and the 28 shared result columns were identical.
5. **mock**: the PR 125 rubric run through the plain `llm` library with a mock judge took about 40 lines, and it rejected an invented quote.

The throwaway code is not in this repo. Ask @turbomam for it.

## Observed with mock models, not yet filed: empty token, time and cost columns

**read** and **mock**. `run_suite.py:36` reads usage from `~/.config/io.datasette.llm/logs.db`. The `llm` Python API does not write that log when called from Python, and on macOS `llm` keeps its files under `~/Library/Application Support/` anyway. In a mock run of 12 calls, with a log database created at the hard-coded path, 0 rows were added, so the columns stayed empty. `src/nmdc_ai_eval/llm_adapter.py:102-109` already reads usage from the response object instead. This has not been checked against real model calls.

## What the PR 125 rubric needs from a runner

**read** from the `judge-rubric` branch.

- **Inputs:** the input, the fields in scope with their definitions, and the output's `metadata_fields`.
- **Calls:** one judge call per criterion.
- **Checks:** each reply validated as `JudgeAnswer`, a pass counted only when its quoted evidence appears in the input or output, ignoring whitespace and case (`rubric.py`), and the rubric version recorded.
- **Judge model:** from a different model family than the generator. The suggester's agentic path runs Claude through the Agent SDK, so a Gemini or GPT judge is needed there.

## Options for #118 (for @JamesTessmer to weigh)

- Keep the suite YAML format, and keep calling models through the `llm` library with this repo's Vertex and PNNL plugins.
- Copy the suite schema in (llm-matrix is MIT licensed, so keep its notice) and write the loop over models and settings in this repo.
- Read token usage from the response rather than the log database.
- Record judge parse failures as rows rather than dropping them.
- Move the env-triad scorer out of `run_suite.py` (#113), and skip the unused judge call for env-triad.
- Dropping llm-matrix also drops `diskcache`, `duckdb` and the old `typer` from the lock. `ipython` is a direct dependency that no file imports (**read**).

## Running whole agents across vendors is a separate question

Comparing agent harnesses (Claude Code, Codex, OpenCode) is a bigger job than #118, and it relates to https://github.com/microbiomedata/nmdc-ai-eval/issues/99. Existing tools to compare before building anything (versions **read** from their package registries on 2026-09-29):

- promptfoo 0.123.1 has claude-agent-sdk, openai-codex-sdk and opencode-sdk providers.
- inspect_swe 0.2.71 has claude_code, codex_cli and opencode agents, and needs a Docker or Kubernetes sandbox.
- The BERIL team has a private workbench that runs the same skills under Claude Code, Codex and OpenCode behind one adapter interface. Ask @dileep-kishore about reusing that layer.

## Related public repos

Found by searching repo names, descriptions and default-branch code in related GitHub orgs on 2026-09-29. Dates are last push.

| last push | repo | what it does |
|---|---|---|
| 2026-09-28 | https://github.com/turbomam/local-llm-evals | sends one task to several local and remote models and scores with a judge from another family; judge scores are not yet sent to Langfuse (https://github.com/turbomam/local-llm-evals/issues/5) |
| 2026-09-28 | https://github.com/monarch-initiative/dismech-evals | weekly model assessments over a corpus, kept in a repo separate from the data and the benchmark |
| 2026-09-20 | https://github.com/contextualizer-ai/cyberian | a CLI over agentapi that drives several coding agents from YAML workflows |
| 2026-09-20 | https://github.com/cmungall/tmux-pilot | manages tmux sessions that run coding agents |
| 2026-08-24 | https://github.com/justaddcoffee/beril-model-comparison | one measured comparison of Claude Code with Opus against another harness with Kimi K3 |
| 2026-05-31 | https://github.com/ai4curation/ai4c-scribe | mines pull requests into datasets for LLM judges |
