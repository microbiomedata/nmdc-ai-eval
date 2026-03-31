#!/usr/bin/env python3
"""Run the real suggestor pipeline against ground truth and score results.

This is Option A from issue #24 — calls run_recommendation_pipeline() with
the actual production code, testing what users will see in the portal.

Requires: GCP Vertex credentials or PNNL AI Incubator key.
Requires: nmdc_data_dev MongoDB with nmdc_submissions collection loaded.

Usage:
    python run_pipeline_eval.py --provider gcp
    python run_pipeline_eval.py --provider gcp --model gemini-2.5-flash
    python run_pipeline_eval.py --provider pnnl --model gpt-4.1-project

Output: field-guidance-pipeline-results.yaml with per-submission scores,
        including elapsed_seconds, input_tokens, output_tokens, est_cost_usd.

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


def load_ground_truth() -> list[dict[str, Any]]:
    with open(GROUND_TRUTH) as f:
        data = yaml.safe_load(f)
        return list(data["submissions"])  # type: ignore[index]


def score_slots(predicted: set[str], expected: set[str]) -> dict[str, Any]:
    """Precision, recall, F1 on slot name sets."""
    if not predicted and not expected:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not predicted:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if not expected:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp = len(predicted & expected)
    precision = tp / len(predicted)
    recall = tp / len(expected)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positives": sorted(predicted & expected),
        "false_positives": sorted(predicted - expected),
        "false_negatives": sorted(expected - predicted),
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run suggestor pipeline eval")
    parser.add_argument("--provider", required=True, choices=["gcp", "pnnl"], help="LLM access provider")
    parser.add_argument("--model", default=None, help="Model name (default: provider's default)")
    parser.add_argument(
        "--mongo-uri",
        default="mongodb://localhost:27017/nmdc_data_dev",
        help="MongoDB URI for data-dev submissions",
    )
    args = parser.parse_args()

    # Import here so the script fails fast with a clear message
    # if the suggestor isn't installed
    from nmdc_metadata_suggestor_ai_tool.llm_client import LLMClient
    from nmdc_metadata_suggestor_ai_tool.recommendation_pipeline import run_recommendation_pipeline

    ground_truth = load_ground_truth()
    mongo_client = MongoClient(args.mongo_uri)
    db = mongo_client.get_default_database()
    collection = db["nmdc_submissions"]

    # Resolve model name from a throw-away client (no LLM calls)
    probe = LLMClient(access_provider=args.provider, model=args.model)
    model_name = probe.model
    print(f"Provider: {args.provider}  Model: {model_name}")
    print(f"Ground truth: {len(ground_truth)} submissions")
    print(f"MongoDB: {args.mongo_uri}\n")

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
                    "provider": args.provider,
                    "status": "skipped",
                    "reason": "submission not found in MongoDB",
                }
            )
            continue

        doc.pop("_id", None)
        expected = {s["field_name"] for s in entry["expected_slots"]}
        print(f"=== {entry['study_name'][:70]} ===")
        print(f"  Expected: {sorted(expected)}")

        # Fresh client per submission to avoid stale conversation context
        # (reusing a client across submissions causes earlier messages to
        # accumulate in self.messages, which inflates context and costs)
        llm_client = LLMClient(access_provider=args.provider, model=args.model)
        usage = instrument_for_tokens(llm_client)

        t0 = time.time()
        try:
            output = run_recommendation_pipeline(
                submission_object=doc,
                llm_client=llm_client,
            )
            elapsed = round(time.time() - t0, 2)

            predicted = {s.field_name for s in output.metadata_fields}
            scores = score_slots(predicted, expected)
            est_cost = estimate_cost(model_name, usage["input_tokens"], usage["output_tokens"])

            print(f"  Predicted ({len(predicted)}): {sorted(predicted)}")
            print(f"  P={scores['precision']}  R={scores['recall']}  F1={scores['f1']}")
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
                    "provider": args.provider,
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
                    "provider": args.provider,
                    "status": "error",
                    "error": str(e),
                    "elapsed_seconds": elapsed,
                    "input_tokens": usage["input_tokens"],
                    "output_tokens": usage["output_tokens"],
                }
            )

    # Aggregate summary
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

        print(f"=== AGGREGATE ({model_name}, n={len(scored)}) ===")
        print(f"  Accuracy:  P={avg_p:.3f}  R={avg_r:.3f}  F1={avg_f1:.3f}")
        print(f"  Time:      {total_time:.1f}s total  ({total_time / len(scored):.1f}s avg)")
        if tokens_known:
            print(f"  Tokens:    {total_in:,} input  {total_out:,} output")
            print(f"  Cost:      ~${total_cost:.4f} total  (~${total_cost / len(scored):.4f} avg)")
        else:
            print("  Tokens:    unavailable (provider did not return usage metadata)")

    output_data: dict[str, Any] = {
        "eval_name": "field-guidance-pipeline",
        "model": model_name,
        "provider": args.provider,
        "submission_count": len(results),
        "results": results,
    }

    with open(RESULTS_FILE, "w") as f:
        yaml.dump(output_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(f"\nResults written to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
