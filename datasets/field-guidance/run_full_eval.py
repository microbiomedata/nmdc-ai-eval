#!/usr/bin/env python3
"""Run a full evaluation matrix: multiple models × enrichment × verification.

Runs run_pipeline_eval.py across all combinations and then synthesizes
results via compare_pipeline_results.py.

Usage:
    python run_full_eval.py                    # default model set
    python run_full_eval.py --models gpt-4o gpt-5.2
    python run_full_eval.py --cheap            # only low-cost models
    python run_full_eval.py --no-verify        # skip verification variants

Each model is run in up to 4 configurations:
    1. with enrichment, no verification
    2. with enrichment, with verification
    3. without enrichment, no verification
    4. without enrichment, with verification

Results accumulate in pipeline-results/ and a comparison table is printed
at the end.
"""

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
EVAL_SCRIPT = HERE / "run_pipeline_eval.py"
COMPARE_SCRIPT = HERE / "compare_pipeline_results.py"

# Model tiers — edit to change defaults
CHEAP_MODELS: list[tuple[str, str | None]] = [
    # (model_name, backend_args)
    ("gpt-4o-mini", "--backend llm"),
    ("gemini/gemini-2.5-flash", "--backend llm"),
]

STANDARD_MODELS: list[tuple[str, str | None]] = [
    ("gpt-4o-mini", "--backend llm"),
    ("gpt-4o", "--backend llm"),
    ("gemini/gemini-2.5-flash", "--backend llm"),
    ("anthropic/claude-sonnet-4-5", "--backend llm"),
]

FULL_MODELS: list[tuple[str, str | None]] = [
    # OpenAI: low, mid, top
    ("gpt-4o-mini", "--backend llm"),
    ("gpt-4o", "--backend llm"),
    ("gpt-5.2", "--backend llm"),
    # Anthropic: low, top
    ("anthropic/claude-haiku-4-5-20251001", "--backend llm"),
    ("anthropic/claude-sonnet-4-6", "--backend llm"),
    # Google: low, top
    ("gemini/gemini-2.5-flash", "--backend llm"),
    ("gemini/gemini-2.5-pro", "--backend llm"),
]

# GCP pipeline models (added if credentials are available)
GCP_MODELS: list[tuple[str, str | None]] = [
    ("gemini-2.5-flash", "--provider gcp"),
    ("gemini-2.5-pro", "--provider gcp"),
]


def _has_gcp_creds() -> bool:
    return bool(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or os.environ.get("VERTEX_PROJECT_ID"))


def _run_eval(model: str, backend_args: str, flags: list[str]) -> bool:
    """Run a single eval configuration. Returns True on success."""
    cmd = [
        sys.executable,
        str(EVAL_SCRIPT),
        *backend_args.split(),
        "--model",
        model,
        *flags,
    ]
    label = f"{model} [{' '.join(flags) if flags else 'baseline'}]"
    print(f"\n{'─' * 70}")
    print(f"Running: {label}")
    print(f"{'─' * 70}")

    result = subprocess.run(cmd, cwd=str(HERE.parent.parent))  # noqa: S603
    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode})")
        return False
    return True


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run full eval matrix")
    parser.add_argument(
        "--models",
        nargs="+",
        help="Specific model names to run (uses llm backend)",
    )
    parser.add_argument(
        "--cheap",
        action="store_true",
        help="Only run cheap models (gpt-4o-mini, gemini-2.5-flash)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Include expensive models (gpt-5.2)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip verification variants",
    )
    parser.add_argument(
        "--no-enrichment-ablation",
        action="store_true",
        help="Skip no-enrichment variants",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Keep existing results (default: clean pipeline-results/ first)",
    )
    args = parser.parse_args()

    # Clean previous results unless --no-clean
    results_dir = HERE / "pipeline-results"
    if not args.no_clean and results_dir.exists():
        import shutil

        shutil.rmtree(results_dir)
        print("Cleaned pipeline-results/\n")

    # Determine model set
    if args.models:
        models = [(m, "--backend llm") for m in args.models]
    elif args.cheap:
        models = list(CHEAP_MODELS)
    elif args.full:
        models = list(FULL_MODELS)
    else:
        models = list(STANDARD_MODELS)

    # Add GCP pipeline models if credentials are available
    if _has_gcp_creds() and not args.models:
        models.extend(GCP_MODELS)

    # Build configuration matrix
    configs: list[tuple[str, str, list[str]]] = []
    for model, backend_args in models:
        # Baseline: enrichment on, no verification
        configs.append((model, backend_args, []))

        # With verification
        if not args.no_verify:
            configs.append((model, backend_args, ["--verify"]))

        # Without enrichment
        if not args.no_enrichment_ablation:
            configs.append((model, backend_args, ["--no-enrichment"]))

            # Without enrichment + with verification
            if not args.no_verify:
                configs.append((model, backend_args, ["--no-enrichment", "--verify"]))

    print(f"Full eval matrix: {len(configs)} runs across {len(models)} models")
    print(f"Models: {[m for m, _ in models]}")
    print(
        f"Variants: enriched, {'enriched+verified, ' if not args.no_verify else ''}"
        f"no-enrichment{', no-enrichment+verified' if not args.no_verify else ''}"
    )
    print()

    successes = 0
    failures = 0
    for model, backend_args, flags in configs:
        ok = _run_eval(model, backend_args, flags)
        if ok:
            successes += 1
        else:
            failures += 1

    # Synthesize results
    print(f"\n{'═' * 70}")
    print(f"FULL EVAL COMPLETE: {successes} succeeded, {failures} failed")
    print(f"{'═' * 70}\n")

    summary_tsv = HERE / "pipeline-results" / "summary.tsv"
    subprocess.run(  # noqa: S603
        [sys.executable, str(COMPARE_SCRIPT), "--latest", "--save-tsv", str(summary_tsv)],
        cwd=str(HERE.parent.parent),
    )

    print("\nDetailed per-submission breakdown:")
    subprocess.run(  # noqa: S603
        [sys.executable, str(COMPARE_SCRIPT), "--latest", "--detail"],
        cwd=str(HERE.parent.parent),
    )


if __name__ == "__main__":
    main()
