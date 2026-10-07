# Dependency upgrades

A weekly workflow (`.github/workflows/uv-upgrade.yml`) upgrades locked dependencies and opens a pull request that merges itself when CI passes. It upgrades every package except those listed in [`.github/held-packages.txt`](../.github/held-packages.txt).

## What is held, and why

The held packages are the model clients, agent frameworks and tracing libraries, plus the suggestor itself: `anthropic`, `openai`, `google-genai`, `claude-agent-sdk`, `llm` and its plugins (`llm-anthropic`, `llm-claude-3`, `llm-gemini`, `llm-matrix`), `langfuse`, the `openinference-*` packages, and `nmdc-metadata-suggestor-ai-tool`.

A version change in any of them can change what a model returns or what gets traced. If one moved inside a routine upgrade, eval scores could shift and nobody would know whether the model, the library or the change under test caused it. On 2026-09-28 the unfiltered weekly upgrade would have moved `claude-agent-sdk` (0.2.157 to 0.2.160), `llm` (0.35 to 0.36), `openai` (3.17.0 to 3.19.2) and two `openinference` packages in one automatic merge.

## How a held package moves

Only in its own pull request, with an eval run before and after the bump on the same inputs, and the two results compared in the PR. There are two ways to do it:

- One package: `uv lock --upgrade-package <name>`.
- Everything, held packages included: `just upgrade-held` locally, or run the "Weekly dependency upgrade" workflow from the Actions tab with "Also upgrade the held AI packages" ticked. The workflow opens a separate PR on the `chore/uv-upgrade-held` branch, titled as needing an eval before and after, and never turns on auto-merge for it.

`just upgrade` runs the same upgrade as the weekly job.

## When the weekly upgrade fails

The workflow opens an issue titled "Weekly dependency upgrade failed", or comments on it if one is already open. One cause is a held package that moved during resolution (below). Until it moves in its own PR, the weekly upgrade stays blocked.

## What still moves

The weekly PR still merges itself when CI passes. That is intended, because it holds none of the AI packages. The dependencies of the held packages (`httpx`, `pydantic`, `opentelemetry-*` and others) are not held and still move weekly; https://github.com/microbiomedata/nmdc-ai-eval/issues/124 tracks whether to hold them too.

## How the exclusion works

uv has no option to leave packages out of `uv lock --upgrade`: it upgrades everything, or upgrades the packages named with `--upgrade-package` ([uv docs: Upgrading locked package versions](https://docs.astral.sh/uv/concepts/projects/sync/#upgrading-locked-package-versions)). So [`scripts/upgrade_except_held.py`](../scripts/upgrade_except_held.py) reads every package from `uv.lock`, drops the held ones, and passes each remaining name as `--upgrade-package`. Run it with `--dry-run` to see what it would hold and upgrade.

Leaving a package out of `--upgrade-package` does not pin it: uv treats the existing lock as a preference, so an upgraded package that needs a newer held one can still pull it forward (`oaklib` depends on `llm`, for example). The script therefore records each held package's lock entry, version and source together, before resolving, and exits with an error naming any that changed. The source matters because `nmdc-metadata-suggestor-ai-tool` is locked from a git branch: its commit can change while its version stays the same. Held patterns are matched again after resolving, so a newly added package that matches one also counts as a change. The workflow then fails and opens no pull request, and the held package has to move in its own PR.
