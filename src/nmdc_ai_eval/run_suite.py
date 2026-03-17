"""Run an llm-matrix eval suite and write results.

Usage:
    uv run python -m nmdc_ai_eval.run_suite datasets/submission-metadata-prediction/sampledata-suite-openai.yaml
    uv run python -m nmdc_ai_eval.run_suite datasets/submission-metadata-prediction/sampledata-suite-anthropic.yaml
"""

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import click
import llm
from llm_matrix import LLMRunner  # type: ignore[import-untyped]
from llm_matrix.schema import load_suite, results_to_dataframe  # type: ignore[import-untyped]

if TYPE_CHECKING:
    import pandas as pd


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


def _print_summary(df: "pd.DataFrame") -> None:
    """Print a human-readable summary: per-model scores, per-category breakdown, and misses."""
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

    runner = LLMRunner(store_path=store_path)
    results = []
    try:
        for i, r in enumerate(runner.run_iter(suite), 1):
            results.append(r)
            score_str = f"{r.score:.2f}" if r.score is not None else "N/A"
            mark = "+" if r.score and r.score >= 1.0 else "-"
            model_short = str(r.hyperparameters.get("model", "?")).split("/")[-1][:15]
            study = r.case.original_input.get("study_name", "")[:30] if r.case.original_input else ""
            click.echo(
                f"  {mark} {i:>3d}/{n_total} [{score_str}] {model_short:<15s} {study:<30s}"
                f"  expected={r.case.ideal}  got={r.response.text[:50]}"
            )
    except Exception as exc:  # noqa: BLE001
        click.echo(f"\nError during eval: {exc}", err=True)
        click.echo("Check model names and API keys. Run: uv run llm models list", err=True)
        sys.exit(1)

    if not results:
        click.echo("No results generated.", err=True)
        sys.exit(1)

    df = results_to_dataframe(results)
    tsv_path = output_dir / "results.tsv"
    df.to_csv(tsv_path, sep="\t", index=False)
    click.echo(f"\nResults: {tsv_path} ({len(results)} rows)")

    _print_summary(df)


if __name__ == "__main__":
    main()
