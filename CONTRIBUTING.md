# Contributing to nmdc-ai-eval

## Getting started

Follow the [Quickstart in README](README.md#quickstart) to clone the repo, install [prerequisites](README.md#prerequisites), and run `just setup`. Then come back here.

### Branch workflow

- Work on feature branches, not `main`.
- One PR per issue. Reference the issue in your commits (e.g. `Closes #42`).
- Keep PRs focused — if you find an unrelated problem, open a separate issue/PR.

## Pre-commit hooks explained

Pre-commit hooks are automated checks that run every time you `git commit`. If any check fails, the commit is blocked until you fix the issue. This repo uses them to catch lint errors, type problems, test failures, and dependency vulnerabilities before code reaches a PR.

Every `git commit` runs the full suite of checks automatically. You can also run them manually:

```bash
just check    # runs all pre-commit hooks against all files
```

The hooks and what they enforce are documented in the [QC and automation table in README.md](README.md#qc-and-automation). Don't worry about memorizing them — the hook name in the output tells you which tool failed, and the lines below it tell you why.

### Common failures and fixes

**pytest (suite YAML validation)** — Read the assertion or validation error. The most common cause for new contributors: suite YAML files with null required fields. Each suite YAML is loaded and checked against a strict Pydantic schema (`llm_matrix.schema.Suite`) — if a field is missing or the wrong type, you get a `ValidationError`. In YAML, a bare key with no value (e.g. `input:`) parses as `null`. `TestCase.input` must be a non-null string. Even for work-in-progress cases, use a placeholder:

```yaml
# Wrong — parses as null
- input:
  ideal: "some expected output"

# Right
- input: "TODO"
  ideal: "some expected output"
```

**pip-audit** (CVEs, run via `just audit` or the weekly CI audit, not as part of `just check`) — A transitive dependency has a known vulnerability. Update it:

```bash
uv lock --upgrade-package <package-name>
uv sync
```

Then re-run `just audit` to confirm the CVE is resolved.

**ruff / ruff-format** — Usually auto-fixed by `just fix`. If the error persists after that, read the rule code in the output (e.g. `E501`, `I001`) and fix manually.

**mypy** — Add or fix the type annotations it complains about. The error message includes the file, line, and expected type.

**uv-lock** — The lockfile is out of sync with `pyproject.toml`. Run `uv lock` to regenerate it.

### The fix-then-check workflow

Run this before committing to catch and auto-fix issues in one pass:

```bash
just all      # equivalent to: just fix && just check
```

## Adding a new eval task

### 1. Set up the dataset directory

Copy an existing directory (e.g. `datasets/ebs-prediction/`) as a starting point:

```bash
cp -r datasets/ebs-prediction/ datasets/my-new-eval/
```

### 2. Write the suite YAML

The suite YAML must parse into `llm_matrix.schema.Suite` (Pydantic-validated). Key requirements:

- **`cases[].input`** — required, must be a string (not null). Even for WIP, use `"TODO"`.
- **`cases[].ideal`** — required, must be a non-null string. Tests enforce that every case has an ideal answer. Even for WIP, use a placeholder like `"TODO"`.
- **`matrix.hyperparameters`** — must include `model` and `temperature`.
- **File naming** — must match the `*-suite*.yaml` glob to be discovered by tests.

### 3. Write a suite generator (if applicable)

If the suite is derived from source data (TSV, MongoDB export, etc.), write a `generate_suite.py` script in the dataset directory. See `datasets/ebs-prediction/generate_suite.py` for an example.

### 4. Wire it up

- Add `just` targets in the [justfile](justfile) for generating and running the new eval (follow the pattern of existing targets like `generate-ebs` and `run-ebs`).
- Add a `README.md` in the dataset directory explaining the task, the biology behind it, and how scoring works.

### 5. Verify

```bash
just check    # must pass before committing
```

## When to use `--no-verify`

Rarely. The only acceptable case is when the failure is a known upstream issue completely unrelated to your changes — for example, a new CVE in a transitive dependency that hasn't been patched yet.

If you do use `--no-verify`:

- Mention it in your PR description with the specific reason.
- Prefer fixing the issue in a separate PR instead, so the hooks stay green for everyone.
