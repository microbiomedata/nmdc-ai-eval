#!/usr/bin/env python3
"""Run the real suggestor pipeline against ground truth and score results.

This is Option A from issue #24 — calls run_recommendation_pipeline() with
the actual production code, testing what users will see in the portal.

Requires: GCP Vertex credentials or PNNL AI Incubator key.

Usage:
    python run_pipeline_eval.py --provider gcp
    python run_pipeline_eval.py --provider gcp --model gemini-2.5-flash
    python run_pipeline_eval.py --provider pnnl --model gpt-4.1-project

Output: field-guidance-pipeline-results.yaml with per-submission scores.
"""

import argparse
import time
from pathlib import Path

import yaml
from pymongo import MongoClient

HERE = Path(__file__).parent
GROUND_TRUTH = HERE / "ground_truth.yaml"
RESULTS_FILE = HERE / "field-guidance-pipeline-results.yaml"


def load_ground_truth() -> list[dict]:
    with open(GROUND_TRUTH) as f:
        return yaml.safe_load(f)["submissions"]


def score_slots(predicted: set[str], expected: set[str]) -> dict:
    """Precision, recall, F1 on slot name sets."""
    if not predicted and not expected:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not predicted:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if not expected:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp = len(predicted & expected)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positives": sorted(predicted & expected),
        "false_positives": sorted(predicted - expected),
        "false_negatives": sorted(expected - predicted),
    }


def main():
    parser = argparse.ArgumentParser(description="Run suggestor pipeline eval")
    parser.add_argument("--provider", required=True, choices=["gcp", "pnnl"], help="LLM access provider")
    parser.add_argument("--model", default=None, help="Model name (default: provider's default)")
    parser.add_argument(
        "--mongo-uri", default="mongodb://localhost:27017/nmdc_data_dev", help="MongoDB URI for data-dev submissions"
    )
    args = parser.parse_args()

    # Import here so the script fails fast with a clear message
    # if the suggestor isn't installed
    from nmdc_metadata_suggestor_ai_tool.llm_client import LLMClient
    from nmdc_metadata_suggestor_ai_tool.recommendation_pipeline import run_recommendation_pipeline

    ground_truth = load_ground_truth()
    client_mongo = MongoClient(args.mongo_uri)
    db = client_mongo.get_default_database()
    collection = db["nmdc_submissions"]

    llm_client = LLMClient(access_provider=args.provider, model=args.model)
    model_name = llm_client.model

    results = []
    for entry in ground_truth:
        submission_id = entry["submission_id"]
        doc = collection.find_one({"id": submission_id})
        if doc is None:
            print(f"SKIP: {submission_id} not in MongoDB")
            continue

        # Remove MongoDB _id (not JSON-serializable)
        doc.pop("_id", None)

        expected = {s["field_name"] for s in entry["expected_slots"]}
        print(f"\n=== {entry['study_name'][:70]} ===")
        print(f"  Expected: {sorted(expected)}")

        t0 = time.time()
        try:
            output = run_recommendation_pipeline(
                submission_object=doc,
                llm_client=llm_client,
            )
            elapsed = round(time.time() - t0, 2)

            predicted = {s.field_name for s in output.metadata_fields}
            scores = score_slots(predicted, expected)

            print(f"  Predicted ({len(predicted)}): {sorted(predicted)}")
            print(f"  Scores: P={scores['precision']} R={scores['recall']} F1={scores['f1']}")
            print(f"  Time: {elapsed}s")

            results.append(
                {
                    "submission_id": submission_id,
                    "study_name": entry["study_name"],
                    "package": entry["package"],
                    "model": model_name,
                    "provider": args.provider,
                    "elapsed_seconds": elapsed,
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
            print(f"  ERROR: {e} ({elapsed}s)")
            results.append(
                {
                    "submission_id": submission_id,
                    "study_name": entry["study_name"],
                    "model": model_name,
                    "provider": args.provider,
                    "error": str(e),
                    "elapsed_seconds": elapsed,
                }
            )

    # Aggregate scores
    scored = [r for r in results if "scores" in r]
    if scored:
        avg_p = sum(r["scores"]["precision"] for r in scored) / len(scored)
        avg_r = sum(r["scores"]["recall"] for r in scored) / len(scored)
        avg_f1 = sum(r["scores"]["f1"] for r in scored) / len(scored)
        total_time = sum(r["elapsed_seconds"] for r in scored)
        print(f"\n=== AGGREGATE ({model_name}) ===")
        print(f"  Mean P={avg_p:.3f}  R={avg_r:.3f}  F1={avg_f1:.3f}")
        print(f"  Total time: {total_time:.1f}s over {len(scored)} submissions")

    output_data = {
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
