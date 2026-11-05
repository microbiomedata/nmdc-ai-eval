#!/usr/bin/env python3
"""Evaluate the submission suggestor with a configurable LLM judge."""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import llm
from dotenv import load_dotenv

from nmdc_ai_eval.llm_judge import (
    SubmissionApiClient,
    evaluate_submission,
    load_rubric,
    load_submission_ids,
    resolve_suggestor_provider,
)

HERE = Path(__file__).parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids-file", type=Path, default=HERE / "submission-ids.txt")
    parser.add_argument("--rubric-file", type=Path, default=HERE / "judge-rubric.yaml")
    parser.add_argument("--output-dir", type=Path, default=HERE / "pipeline-results")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print bounded API response diagnostics to stderr; submission contents may be sensitive.",
    )
    parser.add_argument(
        "--suggestor-provider",
        choices=("auto", "gcp", "pnnl", "cborg"),
        default="auto",
        help="Suggestor provider; auto-detect when exactly one is configured.",
    )
    parser.add_argument(
        "--suggestor-model",
        help="Provider-specific suggestor model; defaults to the suggestor package's provider default.",
    )
    parser.add_argument(
        "--judge-model",
        default="gpt-5.2",
        help="Judge model with sufficient output capacity for the complete individual score set.",
    )
    args = parser.parse_args()

    load_dotenv()
    try:
        suggestor_provider = resolve_suggestor_provider(args.suggestor_provider)
    except ValueError as exc:
        parser.error(str(exc))
    submission_ids = load_submission_ids(str(args.ids_file))
    rubric = load_rubric(str(args.rubric_file))
    refresh_token = os.environ.get("REFRESH_TOKEN")
    if not refresh_token:
        parser.error("Set REFRESH_TOKEN in the environment or .env file.")
    api = SubmissionApiClient(
        base_url=os.environ.get("SERVER_API_URL", "https://data.microbiomedata.org"),
        refresh_token=refresh_token,
        debug=args.debug,
    )
    if args.debug:
        print(
            "Debug mode is enabled; API response content may include sensitive submission metadata.",
            file=sys.stderr,
        )

    suggestor_model_label = args.suggestor_model or "provider default"
    print(
        f"Scoring {len(submission_ids)} submissions "
        f"(suggestor: {suggestor_provider}/{suggestor_model_label}; judge: {args.judge_model})"
    )
    results = []
    for submission_id in submission_ids:
        submission = api.fetch_submission(submission_id)
        # Keep judge instances isolated: each submission gets a new model object
        # and a prompt containing only that submission's context.
        judge_model: Any = llm.get_model(args.judge_model)
        result = evaluate_submission(
            submission_id=submission_id,
            submission=submission,
            suggestor_provider=suggestor_provider,
            suggestor_model=args.suggestor_model,
            judge_model=judge_model,
            judge_model_name=args.judge_model,
            rubric=rubric,
        )
        results.append(result)
        print(f"  {submission_id}: mean judge score {result['mean_score']}/4")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output_path = args.output_dir / f"results-{timestamp}.json"
    output_path.write_text(
        json.dumps(
            {
                "eval_name": "submission-suggestor-judge",
                "timestamp": timestamp,
                "suggestor_provider": suggestor_provider,
                "suggestor_model": args.suggestor_model,
                "judge_model": args.judge_model,
                "rubric": rubric,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
