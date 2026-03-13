"""CLI wrapper for ENVO ontology-aware scoring.

Usage:
    uv run python -m nmdc_ai_eval.score_envo results.tsv
    uv run python -m nmdc_ai_eval.score_envo results.tsv --enum-dir datasets/ebs-prediction/enum_data
    uv run python -m nmdc_ai_eval.score_envo results.tsv -o output/
"""

from pathlib import Path

import click

from nmdc_ai_eval.envo_scorer import DEFAULT_ENUM_DIR, score_envo_results


@click.command()
@click.argument("results_tsv", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--enum-dir",
    type=click.Path(exists=True, path_type=Path),
    default=DEFAULT_ENUM_DIR,
    help="Directory containing template enum TSVs",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory (default: same as input)",
)
def main(results_tsv: Path, enum_dir: Path, output_dir: Path | None) -> None:
    """Score env_broad_scale predictions with ontology-aware metrics."""
    df = score_envo_results(results_tsv, enum_dir=enum_dir, output_dir=output_dir)
    click.echo(f"\nScored {len(df)} rows. Output in {output_dir or results_tsv.parent}/results_envo_scored.tsv")


if __name__ == "__main__":
    main()
