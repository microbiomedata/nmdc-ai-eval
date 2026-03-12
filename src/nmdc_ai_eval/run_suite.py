"""Run an llm-matrix eval suite and write results.

Usage:
    uv run python -m nmdc_ai_eval.run_suite datasets/submission-metadata-prediction/sampledata-suite.yaml
"""

import sys
from pathlib import Path

from llm_matrix import LLMRunner
from llm_matrix.schema import load_suite, results_to_dataframe


def main(suite_path: str, output_dir: str | None = None) -> None:
    suite_path = Path(suite_path)
    if not suite_path.exists():
        print(f"Error: {suite_path} not found", file=sys.stderr)
        sys.exit(1)

    suite = load_suite(suite_path)
    store_path = suite_path.parent / (suite_path.stem + ".db")
    if output_dir is None:
        output_dir_path = suite_path.parent / (suite_path.stem + "-output")
    else:
        output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    runner = LLMRunner(store_path=store_path)
    results = []
    for r in runner.run_iter(suite):
        results.append(r)
        score_str = f"{r.score:.2f}" if r.score is not None else "N/A"
        print(f"[{score_str}] {r.case.ideal} <- {r.response.text[:80]}")

    if not results:
        print("No results generated.", file=sys.stderr)
        sys.exit(1)

    df = results_to_dataframe(results)
    tsv_path = output_dir_path / "results.tsv"
    df.to_csv(tsv_path, sep="\t", index=False)
    print(f"\nResults: {tsv_path} ({len(results)} rows)")
    print(df.describe())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <suite.yaml> [output_dir]", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
