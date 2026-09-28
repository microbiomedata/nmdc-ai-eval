# Dependency upgrades

A weekly workflow (`.github/workflows/uv-upgrade.yml`) upgrades locked dependencies and opens a pull request that merges itself when CI passes. It upgrades every package except those listed in [`.github/held-packages.txt`](../.github/held-packages.txt).

## What is held, and why

The held packages are the model clients, agent frameworks and tracing libraries, plus the suggestor itself: `anthropic`, `openai`, `google-genai`, `claude-agent-sdk`, `llm` and its plugins (`llm-anthropic`, `llm-claude-3`, `llm-gemini`, `llm-matrix`), `langfuse`, the `openinference-*` packages, and `nmdc-metadata-suggestor-ai-tool`.

A version change in any of them can change what a model returns or what gets traced. If one moved inside a routine upgrade, eval scores could shift and nobody would know whether the model, the library or the change under test caused it. On 2026-09-28 the unfiltered weekly upgrade would have moved `claude-agent-sdk` (0.2.157 to 0.2.160), `llm` (0.35 to 0.36), `openai` (3.17.0 to 3.19.2) and two `openinference` packages in one automatic merge.

## How a held package moves

Only in its own pull request, with an eval run before and after the bump on the same inputs, and the two results compared in the PR. Upgrade one with `uv lock --upgrade-package <name>`.

## How the exclusion works

uv has no option to leave packages out of `uv lock --upgrade`: it upgrades everything, or upgrades the packages named with `--upgrade-package` ([uv docs: Upgrading locked package versions](https://docs.astral.sh/uv/concepts/projects/sync/#upgrading-locked-package-versions)). So [`scripts/upgrade_except_held.py`](../scripts/upgrade_except_held.py) reads every package from `uv.lock`, drops the held ones, and passes each remaining name as `--upgrade-package`. Run it with `--dry-run` to see what it would hold and upgrade.
