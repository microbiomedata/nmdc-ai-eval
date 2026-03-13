"""Run an llm-matrix eval suite and write results.

Usage:
    uv run python -m nmdc_ai_eval.run_suite datasets/submission-metadata-prediction/sampledata-suite.yaml
    uv run python -m nmdc_ai_eval.run_suite suite.yaml --output-dir results/
"""

import sys
from pathlib import Path

import click
from dotenv import load_dotenv
from llm_matrix import LLMRunner  # type: ignore[import-untyped]
from llm_matrix.schema import load_suite, results_to_dataframe  # type: ignore[import-untyped]

load_dotenv()  # loads .env into os.environ so llm picks up OPENAI_API_KEY/ANTHROPIC_API_KEY


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
    store_path = suite_path.parent / (suite_path.stem + ".db")
    if output_dir is None:
        output_dir = suite_path.parent / (suite_path.stem + "-output")
    output_dir.mkdir(parents=True, exist_ok=True)

    runner = LLMRunner(store_path=store_path)
    results = []
    for r in runner.run_iter(suite):
        results.append(r)
        score_str = f"{r.score:.2f}" if r.score is not None else "N/A"
        click.echo(f"[{score_str}] {r.case.ideal} <- {r.response.text[:80]}")

    if not results:
        click.echo("No results generated.", err=True)
        sys.exit(1)

    df = results_to_dataframe(results)
    tsv_path = output_dir / "results.tsv"
    df.to_csv(tsv_path, sep="\t", index=False)
    click.echo(f"\nResults: {tsv_path} ({len(results)} rows)")
    click.echo(str(df.describe()))


if __name__ == "__main__":
    main()
