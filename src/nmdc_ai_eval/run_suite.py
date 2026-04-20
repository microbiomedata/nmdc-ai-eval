"""Run an llm-matrix eval suite and write results with inline cost/timing.

Usage:
    uv run python -m nmdc_ai_eval.run_suite datasets/ebs-prediction/ebs-suite.yaml

Token counts and wall-clock timing are captured from the llm library's logs
database (~/.config/io.datasette.llm/logs.db) after each LLM call. Cost is
estimated using the pricing table in nmdc_ai_eval.pricing. All three appear
as columns in the output TSV alongside accuracy scores.

For env-triad evals, the ``simple_question`` LLM-as-judge score is
overridden by direct per-field comparison via ``_env_triad_score``. The
judge call still executes (llm-matrix evaluates before yielding the
result) but its score is discarded. Eliminating the call entirely
requires a fork of llm-matrix; left as a future improvement. See #72.
"""

import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import click
import llm
from llm_matrix import LLMRunner  # type: ignore[import-untyped]
from llm_matrix.schema import load_suite, results_to_dataframe  # type: ignore[import-untyped]

from nmdc_ai_eval.pricing import estimate_cost

if TYPE_CHECKING:
    import pandas as pd

_LLM_LOGS_DB = Path.home() / ".config" / "io.datasette.llm" / "logs.db"


def _try_parse_env_triad(text: str | None) -> dict[str, str | None]:
    """Extract env-triad field values from a JSON string.

    Accepts raw JSON, JSON wrapped in ```json``` fences, or JSON embedded
    in prose. Returns ``{"broad": ..., "local": ..., "medium": ...}`` with
    ``None`` for any field that can't be parsed. Never raises.
    """
    empty: dict[str, str | None] = {"broad": None, "local": None, "medium": None}
    if not text:
        return empty
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        payload: str | None = fenced.group(1)
    else:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        payload = m.group(0) if m else None
    if not payload:
        return empty
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


def _open_llm_logs_db() -> sqlite3.Connection | None:
    """Open the llm library's logs database, if it exists."""
    if not _LLM_LOGS_DB.exists():
        return None
    try:
        return sqlite3.connect(str(_LLM_LOGS_DB))
    except sqlite3.Error:
        return None


def _get_max_rowid(db: sqlite3.Connection) -> int:
    """Get the current max rowid in the responses table."""
    row = db.execute("SELECT COALESCE(MAX(rowid), 0) FROM responses").fetchone()
    return int(row[0]) if row else 0


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


def _capture_log_entry(db: sqlite3.Connection, after_rowid: int) -> tuple[int, dict[str, int | None]]:
    """Get the most recent log entry after the given rowid.

    Returns (new_last_rowid, {input_tokens, output_tokens, duration_ms}).
    """
    row = db.execute(
        "SELECT rowid, input_tokens, output_tokens, duration_ms "
        "FROM responses WHERE rowid > ? ORDER BY rowid DESC LIMIT 1",
        (after_rowid,),
    ).fetchone()
    if row is not None:
        return int(row[0]), {
            "input_tokens": row[1],
            "output_tokens": row[2],
            "duration_ms": row[3],
        }
    return after_rowid, {"input_tokens": None, "output_tokens": None, "duration_ms": None}


def _print_summary(df: "pd.DataFrame") -> None:
    """Print a human-readable summary: per-model scores, cost, and misses."""
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
        click.echo(f"\n  Majority-class baseline: {baseline:.0%} (always predict '{most_common.index[0]}')")

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
        pivot = df.pivot_table(values="score", index="case_ideal", columns="model", aggfunc="mean")
        pivot["support"] = df.groupby("case_ideal")["score"].count() // len(df["model"].unique())
        for cat, row in pivot.iterrows():
            models_str = "  ".join(f"{row.get(m, float('nan')):.0%}" for m in model_scores.index)
            click.echo(f"  {cat:<50s} {models_str}  (n={row['support']:.0f})")
        click.echo(f"  {'models:':<50s} {'  '.join(str(m) for m in model_scores.index)}")

    # Show misses grouped by expected category
    misses = df[df["score"] < 1.0]
    if not misses.empty:
        click.echo(f"\n── Misses ({len(misses)}/{len(df)}) ──")
        for cat in sorted(misses["case_ideal"].unique()):
            cat_misses = misses[misses["case_ideal"] == cat]
            click.echo(f"  expected '{cat}' ({len(cat_misses)} misses):")
            for _, row in cat_misses.iterrows():
                response = str(row.get("response_text", ""))[:50]
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
def main(suite_path: Path, output_dir: Path | None = None) -> None:
    """Run an llm-matrix eval suite and write results to TSV."""
    import pandas as pd

    suite = load_suite(suite_path)

    model_names: list[str] = suite.matrix.hyperparameters.get("model", [])
    errors = _preflight(model_names)
    if errors:
        for err in errors:
            click.echo(f"Error: {err}", err=True)
        sys.exit(1)

    store_path = suite_path.parent / (suite_path.stem + ".db")
    if output_dir is None:
        output_dir = suite_path.parent / (suite_path.stem + "-output")
    output_dir.mkdir(parents=True, exist_ok=True)

    n_cases = len(suite.cases)
    n_models = len(model_names)
    n_total = n_cases * n_models
    click.echo(f"Running {n_cases} cases × {n_models} models = {n_total} calls (~{n_total * 3}–{n_total * 5}s)")

    # Warn when simple_question is active and no scorer model is pinned.
    # simple_question makes a second LLM call per result using llm's default
    # model (currently gpt-4o per llm_matrix/metrics.py:DEFAULT_EVALUATION_MODEL_NAME).
    # That default is implicit and can change. Env-triad suites bypass this via
    # _env_triad_score(); other suites should pin --scorer-model for reproducibility.
    uses_simple_question = any("simple_question" in (t.metrics or []) for t in (suite.templates or {}).values())
    scorer_model_env = __import__("os").environ.get("LLM_SCORER_MODEL")
    if uses_simple_question and not scorer_model_env:
        click.echo(
            "  Note: suite uses simple_question metric. Scorer model is llm's default "
            "(currently gpt-4o). Pin it with --scorer-model or LLM_SCORER_MODEL env var "
            "for reproducible scoring across runs.",
            err=True,
        )

    # Open llm logs DB for inline token/timing capture
    logs_db = _open_llm_logs_db()
    last_rowid = _get_max_rowid(logs_db) if logs_db else 0
    if logs_db:
        click.echo("  (token/timing capture enabled via llm logs DB)")
    else:
        click.echo("  (llm logs DB not found — token/timing will not be captured)")

    runner = LLMRunner(store_path=store_path)
    results = []
    token_data: list[dict[str, float | None]] = []
    try:
        for i, r in enumerate(runner.run_iter(suite), 1):
            # For env-triad cases, override the LLM-as-judge score with a
            # direct per-field comparison — cheaper, faster, and not subject
            # to scorer flakiness. Falls back to the original score for
            # non-env-triad evals (when _env_triad_score returns None).
            direct = _env_triad_score(r.case.ideal, r.response.text)
            if direct is not None:
                r.score = direct

            results.append(r)

            # Capture tokens/timing from llm logs
            entry: dict[str, float | None] = {
                "input_tokens": None,
                "output_tokens": None,
                "duration_ms": None,
                "est_cost_usd": None,
            }
            if logs_db:
                last_rowid, log_entry = _capture_log_entry(logs_db, last_rowid)
                entry.update(log_entry)
                model_name = str(r.hyperparameters.get("model", ""))
                cost = estimate_cost(model_name, log_entry["input_tokens"], log_entry["output_tokens"])
                if cost is not None:
                    entry["est_cost_usd"] = round(cost, 6)
            token_data.append(entry)

            score_str = f"{r.score:.2f}" if r.score is not None else "N/A"
            mark = "+" if r.score and r.score >= 1.0 else "-"
            model_short = str(r.hyperparameters.get("model", "?")).split("/")[-1][:15]
            study = r.case.original_input.get("study_name", "")[:30] if r.case.original_input else ""
            cost_str = f" ${entry['est_cost_usd']:.4f}" if entry.get("est_cost_usd") else ""
            tok_str = ""
            if entry.get("input_tokens") is not None:
                tok_str = f" {entry['input_tokens']}+{entry['output_tokens']}tok"
            click.echo(
                f"  {mark} {i:>3d}/{n_total} [{score_str}] {model_short:<15s} {study:<30s}"
                f"  expected={r.case.ideal}  got={r.response.text[:50]}{tok_str}{cost_str}"
            )
    except Exception as exc:  # noqa: BLE001
        click.echo(f"\nError during eval: {exc}", err=True)
        click.echo("Check model names and API keys. Run: uv run llm models list", err=True)
        sys.exit(1)

    if not results:
        click.echo("No results generated.", err=True)
        sys.exit(1)

    # Merge accuracy results with token/timing/cost data
    df = results_to_dataframe(results)
    token_df = pd.DataFrame(token_data)
    df = pd.concat([df, token_df], axis=1)

    tsv_path = output_dir / "results.tsv"
    df.to_csv(tsv_path, sep="\t", index=False)
    click.echo(f"\nResults: {tsv_path} ({len(results)} rows)")

    _print_summary(df)

    if logs_db:
        logs_db.close()


if __name__ == "__main__":
    main()
