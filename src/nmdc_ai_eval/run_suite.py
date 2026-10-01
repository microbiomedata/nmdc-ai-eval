"""Run an evaluation suite and write results with inline cost/timing.

Usage:
    uv run python -m nmdc_ai_eval.run_suite datasets/ebs-prediction/ebs-suite.yaml

Models and scoring calls use the llm library. Token counts and wall-clock
timing are collected from each llm response. Cost is estimated using the
pricing table in nmdc_ai_eval.pricing.

For env-triad-style evals where ``case_ideal`` is a JSON string of shape
``{"metadata_fields": [{field_name, value, ...}, ...]}``, the output TSV
also gets per-field columns (``expected_broad``/``got_broad``/
``broad_match`` etc.) so downstream analysis can pivot without re-parsing.
Non-env-triad evals see those columns as ``None`` — harmless.
"""

import itertools
import json
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
import llm
from dotenv import load_dotenv

from nmdc_ai_eval.pricing import estimate_cost
from nmdc_ai_eval.suite import Suite, Template, TestCase

if TYPE_CHECKING:
    import pandas as pd

# Slot names for env-triad extraction. Used by _try_parse_env_triad below.
_ENV_TRIAD_SLOTS = ("env_broad_scale", "env_local_scale", "env_medium")


def _try_parse_env_triad(text: str | None) -> dict[str, str | None]:
    """Extract env-triad field values from a JSON string.

    Accepts raw JSON, JSON wrapped in ``` ```json ``` fences, or a mixed
    response where JSON appears after prose. Returns ``{"broad": ..., "local":
    ..., "medium": ...}`` with ``None`` for any field that can't be parsed.
    Never raises — callers use the None sentinels to decide whether a cell
    is meaningful.
    """
    empty: dict[str, str | None] = {"broad": None, "local": None, "medium": None}
    # Guard against non-string input (pandas passes NaN floats for rows where
    # case_ideal was never populated, e.g. when a scorer parse error drops
    # the result mid-run). `not text` alone doesn't catch NaN.
    if not isinstance(text, str) or not text:
        return empty
    # Greedy match between fences — env-triad JSON has nested {} (one per
    # field in metadata_fields) so a non-greedy inner match would stop at
    # the first inner } and fail to parse. The outer fence anchors the end.
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        payload = fenced.group(1)
    else:
        braces = re.search(r"\{.*\}", text, re.DOTALL)
        if not braces:
            return empty
        payload = braces.group(0)
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return empty
    if not isinstance(data, dict):
        return empty
    fields = data.get("metadata_fields")
    if not isinstance(fields, list):
        return empty
    field_map: dict[str, str | None] = {}
    for item in fields:
        if isinstance(item, dict):
            name = item.get("field_name")
            if isinstance(name, str):
                value = item.get("value")
                field_map[name] = value if isinstance(value, str) else None
    return {
        "broad": field_map.get("env_broad_scale"),
        "local": field_map.get("env_local_scale"),
        "medium": field_map.get("env_medium"),
    }


def _short_label(text: str | None, fallback_chars: int = 80) -> str:
    """Short display label for a JSON ideal/response.

    If the text parses as an env-triad JSON, returns
    ``"broad | local | medium"`` with question marks for missing fields.
    Otherwise returns the first ``fallback_chars`` of the text with
    newlines replaced by spaces, so it fits on one console line.
    """
    # Mirror the guard in _try_parse_env_triad: pandas passes NaN (float)
    # for lost-result rows and str slicing would crash.
    if not isinstance(text, str) or not text:
        return ""
    parsed = _try_parse_env_triad(text)
    if any(parsed.values()):
        return " | ".join(parsed.get(k) or "?" for k in ("broad", "local", "medium"))
    return text[:fallback_chars].replace("\n", " ").replace("\r", " ")


_TRIAD_SCORE_MAP = {0: 0.0, 1: 0.33, 2: 0.67, 3: 1.0}


def _env_triad_score(ideal: str | None, response: str | None) -> float | None:
    """Direct per-field score for env-triad responses.

    Returns one of 0.0 / 0.33 / 0.67 / 1.0 based on how many of the three
    env-triad fields (broad, local, medium) match exactly. Returns ``None``
    only when the ideal doesn't parse as env-triad JSON — the caller then
    falls back to the original metric for non-env-triad suites.

    When the *response* is unparsable (prose, empty, bad JSON), all fields
    compare as non-matching and the score is 0.0, not ``None``.

    Note: the judge call from ``simple_question`` still executes; this
    function merely overrides its result. See module docstring for context.
    """
    ideal_fields = _try_parse_env_triad(ideal)
    if not any(ideal_fields.values()):
        return None  # not an env-triad case — leave scoring to the original metric
    response_fields = _try_parse_env_triad(response)
    matches = sum(
        1
        for k in ("broad", "local", "medium")
        if ideal_fields.get(k) is not None and ideal_fields.get(k) == response_fields.get(k)
    )
    return _TRIAD_SCORE_MAP[matches]


def _preflight(model_names: list[str]) -> list[str]:
    """Check that all models are available via llm plugins.

    Returns a list of human-readable error strings (empty = all OK).
    """
    errors: list[str] = []
    for name in model_names:
        try:
            llm.get_model(name)
        except llm.UnknownModelError:
            errors.append(
                f"Unknown model '{name}'. Run `uv run llm models list` to see available models. "
                f"You may need a plugin: llm-claude-3 (Anthropic), llm-gemini (Gemini)."
            )
    return errors


def _models_with_credentials(model_names: list[str]) -> tuple[list[str], list[tuple[str, str]]]:
    """Split registered models by whether their required llm key is configured.

    Models with no ``needs_key`` are retained because they may authenticate by
    another mechanism, such as Vertex service-account credentials.
    """
    available: list[str] = []
    unavailable: list[tuple[str, str]] = []
    for name in model_names:
        model = llm.get_model(name)
        key_alias = model.needs_key
        if key_alias and not llm.get_key(key_alias=key_alias):
            unavailable.append((name, key_alias))
        else:
            available.append(name)
    return available, unavailable


def _pnnl_models_if_configured() -> list[str]:
    """Return repository-provided PNNL aliases when their llm key is stored."""
    if not llm.get_key(key_alias="pnnl"):
        return []
    from nmdc_ai_eval.llm_plugin_pnnl import _load_pnnl_models

    return [f"pnnl/{name}" for name in _load_pnnl_models()]


def _template_for(case: TestCase, suite: Suite) -> Template | None:
    template_name = case.template or suite.template
    if not template_name:
        return None
    if not suite.templates or template_name not in suite.templates:
        raise ValueError(f"Template '{template_name}' is not defined in suite '{suite.name}'.")
    return suite.templates[template_name]


def _format_case(case: TestCase, template: Template | None) -> tuple[str, str | None]:
    if template is None:
        return case.input, None
    params: dict[str, Any] = {"input": case.input}
    params.update(case.original_input or {})
    prompt = template.prompt.format(**params) if template.prompt else case.input
    system = template.system.format(**params) if template.system else None
    return prompt, system


def _matrix_parameters(suite: Suite) -> list[dict[str, Any]]:
    names = list(suite.matrix.hyperparameters)
    values = list(suite.matrix.hyperparameters.values())
    return [dict(zip(names, combination, strict=True)) for combination in itertools.product(*values)]


def _model_call_config(
    suite: Suite, model_name: str, params: dict[str, Any]
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    model_info = suite.models.get(model_name)
    if model_info and model_info.plugins:
        raise ValueError(
            f"Suite model '{model_name}' declares llm-matrix plugin(s) {model_info.plugins}; "
            "suite plugins are not supported by the llm-based runner."
        )
    configured = {**params, **(model_info.parameters if model_info else {})}
    effective_model = str(configured.pop("model", model_name))
    configured.pop("key", None)
    configured.pop("plugins", None)
    call_params = {key: value for key, value in configured.items() if key != "model" and value is not None}
    effective_params = {"model": effective_model, **call_params}
    return effective_model, call_params, effective_params


def _score_simple_question(scorer: llm.Model, ideal: str | None, output: str) -> tuple[float | None, str | None, Any]:
    """Ask the configured judge for a score; keep unparsable replies for reporting."""
    response = scorer.prompt(
        "The expected answer is: {ideal}. The output to score is: {output}.".format(ideal=ideal or "", output=output),
        system=(
            "Compare the answer given to the expected output. "
            "The response should be a score between 0 and 1. "
            "The answer should be provided first, explanations may follow "
            "A precise correct answer is 1, a wrong answer is 0. "
            "You can use values in between for imprecise answers"
        ),
    )
    message = response.text().strip()
    match = re.match(r"(\d+(?:\.\d+)?)", message)
    score = float(match.group(1)) if match else None
    if score is not None and not 0.0 <= score <= 1.0:
        score = None
    return score, message, response


def _usage(response: Any) -> tuple[int | None, int | None, int | None]:
    input_tokens = getattr(response, "input_tokens", None)
    output_tokens = getattr(response, "output_tokens", None)
    try:
        duration_ms = response.duration_ms()
    except Exception:
        duration_ms = None
    return input_tokens, output_tokens, duration_ms


def _flat_result(
    case: TestCase,
    params: dict[str, Any],
    template: Template | None,
    prompt: str,
    system: str | None,
    response_text: str,
    score: float | None,
    evaluation_message: str | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "hyperparameters": "_".join(f"{key}={value}" for key, value in params.items()),
        "metrics": template.metrics if template else None,
        "score": score,
        "evaluation_message": evaluation_message,
        "case_input": case.input,
        "case_ideal": case.ideal,
        "case_template": case.template,
        "case_tags": case.tags,
        "case_comments": case.comments,
        "response_text": response_text,
        "response_prompt": prompt,
        "response_system": system,
    }
    row.update(case.original_input or {})
    row.update({key: str(value) for key, value in params.items()})
    return row


def _print_summary(df: "pd.DataFrame") -> None:
    """Print a human-readable summary: per-model scores, cost, and misses."""
    # Rows without ideals cannot contribute to score or category summaries.
    lost = df["case_ideal"].isna().sum() if "case_ideal" in df.columns else 0
    if lost:
        click.echo(f"\n  (skipping {lost} row(s) without an ideal answer)")
        df = df[df["case_ideal"].notna()].reset_index(drop=True)
    if df.empty:
        click.echo("\nNo scorable rows — nothing to summarize.")
        return

    click.echo("\n── Model ranking ──")
    model_scores = df.groupby("model")["score"].agg(["mean", "count", "sum"])
    model_scores.columns = ["accuracy", "cases", "correct"]
    model_scores["correct"] = model_scores["correct"].astype(int)
    model_scores = model_scores.sort_values("accuracy", ascending=False)
    for model, row in model_scores.iterrows():
        click.echo(f"  {row['accuracy']:.0%}  {model}  ({row['correct']:.0f}/{row['cases']:.0f} correct)")

    # Majority-class baseline
    if "case_ideal" in df.columns:
        most_common = df["case_ideal"].value_counts()
        baseline = most_common.iloc[0] / len(df)
        click.echo(
            f"\n  Majority-class baseline: {baseline:.0%} (always predict '{_short_label(most_common.index[0])}')"
        )

    # Cost and timing summary (if data is available)
    has_cost = "est_cost_usd" in df.columns and df["est_cost_usd"].notna().any()
    has_tokens = "input_tokens" in df.columns and df["input_tokens"].notna().any()
    has_timing = "duration_ms" in df.columns and df["duration_ms"].notna().any()

    if has_cost or has_tokens or has_timing:
        click.echo("\n── Cost and timing ──")
        for model in model_scores.index:
            subset = df[df["model"] == model]
            parts: list[str] = [f"  {model}:"]
            if has_tokens:
                in_tok = subset["input_tokens"].sum()
                out_tok = subset["output_tokens"].sum()
                if in_tok > 0 or out_tok > 0:
                    parts.append(f"{int(in_tok):,} in / {int(out_tok):,} out tokens")
            if has_cost:
                cost = subset["est_cost_usd"].sum()
                if cost > 0:
                    parts.append(f"~${cost:.4f}")
                elif has_tokens:
                    parts.append("$0 (free tier or zero-cost provider)")
            if has_timing:
                dur = subset["duration_ms"].sum()
                if dur > 0:
                    avg_dur = dur / len(subset)
                    parts.append(f"{dur / 1000:.1f}s total ({avg_dur / 1000:.1f}s avg)")
            click.echo("  ".join(parts))
        if has_cost:
            total_cost = df["est_cost_usd"].sum()
            click.echo(f"\n  Total estimated cost: ~${total_cost:.4f}")

    click.echo("\n── Per-category accuracy (by model) ──")
    if "case_ideal" in df.columns:
        # Group by a short label derived from the ideal, not the full ideal
        # string, so category rows fit on one console line.
        df_with_cat = df.assign(_cat=df["case_ideal"].map(_short_label))
        pivot = df_with_cat.pivot_table(values="score", index="_cat", columns="model", aggfunc="mean")
        pivot["support"] = df_with_cat.groupby("_cat")["score"].count() // len(df["model"].unique())
        for cat, row in pivot.iterrows():
            models_str = "  ".join(f"{row.get(m, float('nan')):.0%}" for m in model_scores.index)
            click.echo(f"  {cat:<70s} {models_str}  (n={row['support']:.0f})")
        click.echo(f"  {'models:':<70s} {'  '.join(str(m) for m in model_scores.index)}")

    # Show misses grouped by expected category (short label)
    misses = df[df["score"] < 1.0]
    if not misses.empty:
        click.echo(f"\n── Misses ({len(misses)}/{len(df)}) ──")
        misses_with_cat = misses.assign(_cat=misses["case_ideal"].map(_short_label))
        for cat in sorted(misses_with_cat["_cat"].unique()):
            cat_misses = misses_with_cat[misses_with_cat["_cat"] == cat]
            click.echo(f"  expected '{cat}' ({len(cat_misses)} misses):")
            for _, row in cat_misses.iterrows():
                response = _short_label(str(row.get("response_text", "")), 200)
                study = str(row.get("study_name", ""))[:40] if "study_name" in row.index else ""
                model_short = str(row["model"]).split("/")[-1]
                click.echo(f"    {model_short:<30s} got '{response}'  [{study}]")


@click.command()
@click.argument("suite_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory (default: <suite>-output/)",
)
@click.option(
    "--scorer-model",
    default=None,
    envvar="LLM_SCORER_MODEL",
    help=(
        "llm model to use for scoring (default: llm's default model). "
        "Override when the default scorer returns prose instead of a leading number — "
        "e.g. --scorer-model gpt-4o-mini. Also reads LLM_SCORER_MODEL env var."
    ),
)
def main(suite_path: Path, output_dir: Path | None = None, scorer_model: str | None = None) -> None:
    """Run an evaluation suite and write results to TSV."""
    import pandas as pd

    load_dotenv()
    suite = Suite.load(suite_path)
    unsupported_plugins = {name: model_info.plugins for name, model_info in suite.models.items() if model_info.plugins}
    if unsupported_plugins:
        for name, plugins in unsupported_plugins.items():
            click.echo(
                f"Error: suite model '{name}' declares unsupported llm-matrix plugin(s): {', '.join(plugins or [])}.",
                err=True,
            )
        sys.exit(1)

    model_names: list[str] = list(suite.matrix.hyperparameters.get("model", []))
    model_names.extend(name for name in _pnnl_models_if_configured() if name not in model_names)
    errors = _preflight(model_names)
    if errors:
        for err in errors:
            click.echo(f"Error: {err}", err=True)
        sys.exit(1)

    model_names, unavailable_models = _models_with_credentials(model_names)
    for name, key_alias in unavailable_models:
        click.echo(f"Skipping {name}: no credential configured for '{key_alias}'.", err=True)
    if not model_names:
        click.echo("Error: no suite models have configured credentials.", err=True)
        click.echo(
            "To use PNNL AI Incubator, set AI_INCUBATOR_BASE_URL in .env and "
            "run `uv run llm keys set pnnl`. See docs/auth.md.",
            err=True,
        )
        sys.exit(1)
    if output_dir is None:
        output_dir = suite_path.parent / (suite_path.stem + "-output")
    output_dir.mkdir(parents=True, exist_ok=True)

    n_cases = len(suite.cases)
    matrix = suite.matrix.hyperparameters
    if "model" not in matrix:
        matrix["model"] = model_names
    else:
        matrix["model"] = [name for name in matrix["model"] if name in model_names]
        matrix["model"].extend(name for name in model_names if name not in matrix["model"])
    parameter_sets = _matrix_parameters(suite)
    n_total = n_cases * len(parameter_sets)
    click.echo(
        f"Running {n_cases} cases × {len(parameter_sets)} model/settings combinations = "
        f"{n_total} results (~{n_total * 3}–{n_total * 5}s)"
    )

    if scorer_model:
        scorer_errors = _preflight([scorer_model])
        scorer_available, scorer_unavailable = (
            _models_with_credentials([scorer_model]) if not scorer_errors else ([], [])
        )
        if scorer_errors or scorer_unavailable or not scorer_available:
            for err in scorer_errors:
                click.echo(f"Error: scorer model: {err}", err=True)
            for _, key_alias in scorer_unavailable:
                click.echo(f"Error: scorer model '{scorer_model}' needs credential '{key_alias}'.", err=True)
            sys.exit(1)
    else:
        scorer_model = model_names[0]

    scoring_models = {
        template_name: template.metrics or [] for template_name, template in (suite.templates or {}).items()
    }
    unsupported_metrics = sorted(
        {metric for metrics in scoring_models.values() for metric in metrics if metric != "simple_question"}
    )
    if unsupported_metrics:
        click.echo(
            f"Error: unsupported suite metric(s): {', '.join(unsupported_metrics)}. "
            "This runner currently supports only 'simple_question'.",
            err=True,
        )
        sys.exit(1)

    scorer = llm.get_model(scorer_model)
    click.echo(f"  (scorer model: {scorer_model})")
    model_cache: dict[str, llm.Model] = {}
    rows: list[dict[str, Any]] = []
    score_parse_failures = 0
    for i, (case, params) in enumerate(((case, params) for params in parameter_sets for case in suite.cases), start=1):
        model_name = str(params["model"])
        template = _template_for(case, suite)
        metrics = template.metrics or [] if template else []
        try:
            effective_model_name, call_params, effective_params = _model_call_config(suite, model_name, params)
            if effective_model_name not in model_cache:
                model_cache[effective_model_name] = llm.get_model(effective_model_name)
            model = model_cache[effective_model_name]
            prompt, system = _format_case(case, template)
            response = model.prompt(prompt, system=system, **call_params)
            response_text = response.text()
            input_tokens, output_tokens, duration_ms = _usage(response)
            judge_input_tokens = judge_output_tokens = judge_duration_ms = None
            evaluation_message = None
            score: float | None = None
            judge_response = None
            if "simple_question" in metrics:
                score, evaluation_message, judge_response = _score_simple_question(scorer, case.ideal, response_text)
                judge_input_tokens, judge_output_tokens, judge_duration_ms = _usage(judge_response)
                if score is None:
                    score_parse_failures += 1
                    click.echo(
                        f"\n  ! {i:>3d}/{n_total} [score=None] unusable judge response: {evaluation_message}",
                        err=True,
                    )
            direct = _env_triad_score(case.ideal, response_text)
            if direct is not None:
                score = direct
        except Exception as exc:  # noqa: BLE001
            click.echo(f"\nError during eval: {exc}", err=True)
            click.echo("Check model names and API keys. Run: uv run llm models list", err=True)
            break

        row = _flat_result(case, effective_params, template, prompt, system, response_text, score, evaluation_message)
        row["input_tokens"] = sum(x for x in (input_tokens, judge_input_tokens) if x is not None) or None
        row["output_tokens"] = sum(x for x in (output_tokens, judge_output_tokens) if x is not None) or None
        row["duration_ms"] = sum(x for x in (duration_ms, judge_duration_ms) if x is not None) or None
        generation_cost = estimate_cost(effective_model_name, input_tokens, output_tokens)
        judge_cost = estimate_cost(scorer_model, judge_input_tokens, judge_output_tokens)
        costs = [cost for cost in (generation_cost, judge_cost) if cost is not None]
        row["est_cost_usd"] = round(sum(costs), 6) if costs else None
        rows.append(row)

        score_str = f"{score:.2f}" if score is not None else "N/A"
        mark = "+" if score is not None and score >= 1.0 else "-"
        model_short = model_name.split("/")[-1][:15]
        study = str((case.original_input or {}).get("study_name", ""))[:30]
        cost_str = f" ${row['est_cost_usd']:.4f}" if row.get("est_cost_usd") else ""
        tok_str = ""
        if row["input_tokens"] is not None:
            tok_str = f" {row['input_tokens']}+{row['output_tokens']}tok"
        click.echo(
            f"  {mark} {i:>3d}/{n_total} [{score_str}] {model_short:<15s} {study:<30s}"
            f"  expected={_short_label(case.ideal)}  got={_short_label(response_text, 200)}{tok_str}{cost_str}"
        )
    if score_parse_failures:
        click.echo(
            f"\n  Note: {score_parse_failures} unusable judge response(s) — those results have "
            f"score=None in the TSV. The full scorer response is in the "
            f"evaluation_message column. To prevent this, set LLM_SCORER_MODEL=gpt-4o-mini "
            f"(or --scorer-model gpt-4o-mini) to pin the scorer to a reliable model.",
            err=True,
        )

    if not rows:
        click.echo("No results generated.", err=True)
        sys.exit(1)

    df = pd.DataFrame(rows)

    # Add per-field env-triad columns for downstream analysis. For evals whose
    # ideal isn't env-triad-shaped, these come out as None — harmless.
    if "case_ideal" in df.columns:
        ideal_fields = df["case_ideal"].apply(_try_parse_env_triad)
        response_fields = df["response_text"].apply(_try_parse_env_triad) if "response_text" in df.columns else None
        for key in ("broad", "local", "medium"):
            df[f"expected_{key}"] = [d.get(key) for d in ideal_fields]
            if response_fields is not None:
                df[f"got_{key}"] = [d.get(key) for d in response_fields]
                # Match is True/False only when both sides parsed to a value.
                # If either side is None (non-env-triad eval, or parse failure)
                # the match column is None, not spuriously True.
                exp = df[f"expected_{key}"]
                got = df[f"got_{key}"]
                df[f"{key}_match"] = [
                    (e == g) if e is not None and g is not None else None for e, g in zip(exp, got, strict=True)
                ]

    tsv_path = output_dir / "results.tsv"
    df.to_csv(tsv_path, sep="\t", index=False)
    click.echo(f"\nResults: {tsv_path} ({len(rows)} rows)")

    _print_summary(df)


if __name__ == "__main__":
    main()
