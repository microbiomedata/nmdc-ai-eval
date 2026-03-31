#!/usr/bin/env python3
"""Run the real suggestor pipeline against ground truth and score results.

This is Option A from issue #24 — calls run_recommendation_pipeline() with
the actual production code, testing what users will see in the portal.

Requires: GCP Vertex credentials or PNNL AI Incubator key.
Requires: nmdc_data_dev MongoDB with nmdc_submissions collection loaded.

Usage:
    python run_pipeline_eval.py --provider gcp
    python run_pipeline_eval.py --provider gcp --model gemini-2.5-pro
    python run_pipeline_eval.py --provider pnnl --model gpt-4.1-project
    python run_pipeline_eval.py --sweep              # run all available models

Output: field-guidance-pipeline-results.yaml with per-submission scores,
        including elapsed_seconds, input_tokens, output_tokens, est_cost_usd.

Scoring
-------
By default, env triad fields (env_broad_scale, env_local_scale, env_medium)
are excluded from precision scoring because the ground truth intentionally
omits them — any reasonable model will recommend them, so they are neither
true positives nor false positives. Use --strict to count them.

Token tracking
--------------
The suggestor's LLMClient.generate() discards usage metadata from API responses.
This script monkey-patches the underlying API client's generation method to
capture token counts before they're lost. The patch is applied per-submission
on a fresh LLMClient instance, so counts reflect only that submission's calls.
"""

import argparse
import time
from pathlib import Path
from typing import Any

import yaml
from pymongo import MongoClient

from nmdc_ai_eval.pricing import estimate_cost

HERE = Path(__file__).parent
GROUND_TRUTH = HERE / "ground_truth.yaml"
RESULTS_FILE = HERE / "field-guidance-pipeline-results.yaml"

# Fields excluded from precision scoring by default. The ground truth
# intentionally omits these because any reasonable model will recommend them.
# They are not false positives — they are correct but not informative.
ENV_TRIAD = {"env_broad_scale", "env_local_scale", "env_medium"}


def load_ground_truth() -> list[dict[str, Any]]:
    with open(GROUND_TRUTH) as f:
        data = yaml.safe_load(f)
        return list(data["submissions"])  # type: ignore[index]


def score_slots(
    predicted: set[str],
    expected: set[str],
    exclude_from_precision: set[str] | None = None,
) -> dict[str, Any]:
    """Precision, recall, F1 on slot name sets.

    exclude_from_precision: slots to remove from the predicted set before
    computing precision (but NOT recall). Use this for fields like env triads
    that are correct but not in the ground truth by design.
    """
    # For recall: did we find the expected slots?
    if not predicted and not expected:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not expected:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Precision is computed on the filtered set
    precision_set = predicted - (exclude_from_precision or set())
    excluded_correct = predicted & (exclude_from_precision or set())

    tp = len(precision_set & expected)
    fp = len(precision_set - expected)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positives": sorted(precision_set & expected),
        "false_positives": sorted(precision_set - expected),
        "false_negatives": sorted(expected - predicted),
        "excluded_correct": sorted(excluded_correct),
    }


def instrument_for_tokens(llm_client: Any) -> dict[str, int | None]:
    """Monkey-patch an LLMClient to capture token usage from API responses.

    The suggestor's LLMClient.generate() discards usage metadata returned by
    the underlying API clients. This patches the API client's generation method
    on the given instance to log token counts before they're lost.

    A fresh LLMClient should be created per submission before calling this,
    so the returned usage dict reflects only that submission's API calls.

    Returns a mutable dict that accumulates: {"input_tokens": N, "output_tokens": N}.
    Values remain None if the provider does not return usage metadata.
    """
    usage: dict[str, int | None] = {"input_tokens": None, "output_tokens": None}

    if llm_client.access_provider == "gcp":
        original = llm_client.client.models.generate_content

        def patched_gcp(*args: Any, **kwargs: Any) -> Any:
            response = original(*args, **kwargs)
            meta = getattr(response, "usage_metadata", None)
            if meta is not None:
                in_tok = getattr(meta, "prompt_token_count", None)
                out_tok = getattr(meta, "candidates_token_count", None)
                if in_tok is not None:
                    usage["input_tokens"] = (usage["input_tokens"] or 0) + in_tok
                if out_tok is not None:
                    usage["output_tokens"] = (usage["output_tokens"] or 0) + out_tok
            return response

        llm_client.client.models.generate_content = patched_gcp

    elif llm_client.access_provider == "pnnl":
        original = llm_client.client.responses.create

        def patched_pnnl(*args: Any, **kwargs: Any) -> Any:
            response = original(*args, **kwargs)
            u = getattr(response, "usage", None)
            if u is not None:
                in_tok = getattr(u, "input_tokens", None)
                out_tok = getattr(u, "output_tokens", None)
                if in_tok is not None:
                    usage["input_tokens"] = (usage["input_tokens"] or 0) + in_tok
                if out_tok is not None:
                    usage["output_tokens"] = (usage["output_tokens"] or 0) + out_tok
            return response

        llm_client.client.responses.create = patched_pnnl

    return usage


def run_one_model(
    provider: str,
    model: str | None,
    ground_truth: list[dict[str, Any]],
    collection: Any,
    mongo_uri: str,
    exclude_from_precision: set[str],
) -> dict[str, Any]:
    """Run the pipeline eval for a single provider/model combination."""
    from nmdc_metadata_suggestor_ai_tool.llm_client import LLMClient
    from nmdc_metadata_suggestor_ai_tool.recommendation_pipeline import run_recommendation_pipeline

    probe = LLMClient(access_provider=provider, model=model)
    model_name = probe.model
    print(f"\n{'=' * 70}")
    print(f"Provider: {provider}  Model: {model_name}")
    print(f"{'=' * 70}")

    results: list[dict[str, Any]] = []
    for entry in ground_truth:
        submission_id = entry["submission_id"]
        doc = collection.find_one({"id": submission_id})
        if doc is None:
            print(f"SKIP: {submission_id} not in MongoDB")
            results.append(
                {
                    "submission_id": submission_id,
                    "study_name": entry["study_name"],
                    "model": model_name,
                    "provider": provider,
                    "status": "skipped",
                    "reason": "submission not found in MongoDB",
                }
            )
            continue

        doc.pop("_id", None)
        expected = {s["field_name"] for s in entry["expected_slots"]}
        print(f"=== {entry['study_name'][:70]} ===")
        print(f"  Expected: {sorted(expected)}")

        llm_client = LLMClient(access_provider=provider, model=model)
        usage = instrument_for_tokens(llm_client)

        t0 = time.time()
        try:
            output = run_recommendation_pipeline(
                submission_object=doc,
                llm_client=llm_client,
            )
            elapsed = round(time.time() - t0, 2)

            predicted = {s.field_name for s in output.metadata_fields}
            scores = score_slots(predicted, expected, exclude_from_precision)
            est_cost = estimate_cost(model_name, usage["input_tokens"], usage["output_tokens"])

            print(f"  Predicted ({len(predicted)}): {sorted(predicted)}")
            print(f"  P={scores['precision']}  R={scores['recall']}  F1={scores['f1']}")
            if scores.get("excluded_correct"):
                print(f"  Excluded from precision (correct but expected): {scores['excluded_correct']}")
            tokens_str = (
                f"in={usage['input_tokens']} out={usage['output_tokens']}"
                if usage["input_tokens"] is not None
                else "tokens=unavailable"
            )
            cost_str = f"~${est_cost:.4f}" if est_cost is not None else "cost=unavailable"
            print(f"  Time: {elapsed}s  Tokens: {tokens_str}  Cost: {cost_str}\n")

            results.append(
                {
                    "submission_id": submission_id,
                    "study_name": entry["study_name"],
                    "package": entry["package"],
                    "model": model_name,
                    "provider": provider,
                    "elapsed_seconds": elapsed,
                    "input_tokens": usage["input_tokens"],
                    "output_tokens": usage["output_tokens"],
                    "est_cost_usd": round(est_cost, 6) if est_cost is not None else None,
                    "expected_slots": sorted(expected),
                    "predicted_slots": sorted(predicted),
                    "scores": scores,
                    "all_suggestions": [
                        {"field_name": s.field_name, "reason": s.reason, "value": s.value}
                        for s in output.metadata_fields
                    ],
                }
            )

        except Exception as e:
            elapsed = round(time.time() - t0, 2)
            print(f"  ERROR: {e} ({elapsed}s)\n")
            results.append(
                {
                    "submission_id": submission_id,
                    "study_name": entry["study_name"],
                    "model": model_name,
                    "provider": provider,
                    "status": "error",
                    "error": str(e),
                    "elapsed_seconds": elapsed,
                    "input_tokens": usage["input_tokens"],
                    "output_tokens": usage["output_tokens"],
                }
            )

    # Per-model aggregate
    scored = [r for r in results if "scores" in r]
    if scored:
        avg_p = sum(r["scores"]["precision"] for r in scored) / len(scored)
        avg_r = sum(r["scores"]["recall"] for r in scored) / len(scored)
        avg_f1 = sum(r["scores"]["f1"] for r in scored) / len(scored)
        total_time = sum(r["elapsed_seconds"] for r in scored)
        total_in = sum(r["input_tokens"] or 0 for r in scored)
        total_out = sum(r["output_tokens"] or 0 for r in scored)
        total_cost = sum(r["est_cost_usd"] or 0.0 for r in scored)
        tokens_known = any(r["input_tokens"] is not None for r in scored)

        print(f"--- {model_name} ({provider}) ---")
        print(f"  Accuracy:  P={avg_p:.3f}  R={avg_r:.3f}  F1={avg_f1:.3f}")
        print(f"  Time:      {total_time:.1f}s total  ({total_time / len(scored):.1f}s avg)")
        if tokens_known:
            print(f"  Tokens:    {total_in:,} input  {total_out:,} output")
            print(f"  Cost:      ~${total_cost:.4f} total  (~${total_cost / len(scored):.4f} avg)")

    return {
        "model": model_name,
        "provider": provider,
        "submission_count": len(results),
        "results": results,
    }


def _get_sweep_configs() -> list[tuple[str, str | None]]:
    """Return (provider, model) pairs for all available models.

    Only includes providers whose credentials are configured.
    """
    import os

    configs: list[tuple[str, str | None]] = []

    # GCP models (if credentials are available)
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or os.environ.get("VERTEX_PROJECT_ID"):
        from nmdc_metadata_suggestor_ai_tool.llm_client import GEMINI_MODELS

        for m in GEMINI_MODELS:
            configs.append(("gcp", m))

    # PNNL models (if credentials are available)
    if os.environ.get("AI_INCUBATOR_KEY") and os.environ.get("AI_INCUBATOR_BASE_URL"):
        from nmdc_metadata_suggestor_ai_tool.llm_client import PNNL_GPT_MODELS

        for m in PNNL_GPT_MODELS:
            configs.append(("pnnl", m))

    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run suggestor pipeline eval")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--provider", choices=["gcp", "pnnl"], help="LLM access provider")
    group.add_argument("--sweep", action="store_true", help="Run all models for all configured providers")
    parser.add_argument("--model", default=None, help="Model name (default: provider's default)")
    parser.add_argument("--strict", action="store_true", help="Count env triad fields in precision scoring")
    parser.add_argument(
        "--mongo-uri",
        default="mongodb://localhost:27017/nmdc_data_dev",
        help="MongoDB URI for data-dev submissions",
    )
    args = parser.parse_args()

    ground_truth = load_ground_truth()
    mongo_client = MongoClient(args.mongo_uri)
    db = mongo_client.get_default_database()
    collection = db["nmdc_submissions"]

    exclude = set() if args.strict else ENV_TRIAD
    if not args.strict:
        print(f"Excluding from precision: {sorted(exclude)}")
        print("(use --strict to count these as false positives)\n")

    # Determine which models to run
    if args.sweep:
        configs = _get_sweep_configs()
        if not configs:
            print("ERROR: --sweep found no configured providers.")
            print("Set GOOGLE_APPLICATION_CREDENTIALS + VERTEX_PROJECT_ID for GCP")
            print("or AI_INCUBATOR_KEY + AI_INCUBATOR_BASE_URL for PNNL")
            raise SystemExit(1)
        print(f"Sweep: {len(configs)} model(s) across {len(set(p for p, _ in configs))} provider(s)")
        for p, m in configs:
            print(f"  {p}: {m}")
    else:
        configs = [(args.provider, args.model)]

    all_runs: list[dict[str, Any]] = []
    for provider, model in configs:
        run_data = run_one_model(
            provider=provider,
            model=model,
            ground_truth=ground_truth,
            collection=collection,
            mongo_uri=args.mongo_uri,
            exclude_from_precision=exclude,
        )
        all_runs.append(run_data)

    # Write results
    output_data: dict[str, Any] = {
        "eval_name": "field-guidance-pipeline",
        "scoring": "strict" if args.strict else "env-triad-excluded",
        "runs": all_runs,
    }

    with open(RESULTS_FILE, "w") as f:
        yaml.dump(output_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(f"\nResults written to {RESULTS_FILE}")

    # Cross-model comparison (if multiple models)
    if len(all_runs) > 1:
        print(f"\n{'=' * 70}")
        print("CROSS-MODEL COMPARISON")
        print(f"{'=' * 70}")
        for run in all_runs:
            scored = [r for r in run["results"] if "scores" in r]
            if not scored:
                continue
            avg_p = sum(r["scores"]["precision"] for r in scored) / len(scored)
            avg_r = sum(r["scores"]["recall"] for r in scored) / len(scored)
            avg_f1 = sum(r["scores"]["f1"] for r in scored) / len(scored)
            total_cost = sum(r.get("est_cost_usd") or 0 for r in scored)
            total_time = sum(r["elapsed_seconds"] for r in scored)
            print(
                f"  {run['model']:<25s} ({run['provider']})  "
                f"P={avg_p:.3f}  R={avg_r:.3f}  F1={avg_f1:.3f}  "
                f"${total_cost:.4f}  {total_time:.0f}s"
            )


if __name__ == "__main__":
    main()
