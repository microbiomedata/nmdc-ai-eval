#!/usr/bin/env python3
"""Generate an llm-matrix suite YAML for Metadata Field Guidance evaluation.

Reads ground_truth.yaml and fetches full submission documents from MongoDB
(nmdc_data_dev.nmdc_submissions). Uses the production suggestor's own system
prompt, submission field extraction, and schema context — the only difference
from the real pipeline is the LLM provider (llm-matrix vs Vertex/PNNL).

For testing the real end-to-end pipeline, see run_pipeline_eval.py.

Usage:
    python generate_suite.py                    # defaults
    python generate_suite.py --mongo-uri ...    # custom MongoDB
"""

import argparse
from pathlib import Path

import yaml
from nmdc_metadata_suggestor_ai_tool.recommendation_pipeline import get_submission_fields
from nmdc_metadata_suggestor_ai_tool.schema_context import SchemaContextBuilder
from nmdc_metadata_suggestor_ai_tool.system_prompt import system_prompt as PRODUCTION_SYSTEM_PROMPT
from pymongo import MongoClient

HERE = Path(__file__).parent
GROUND_TRUTH = HERE / "ground_truth.yaml"
MODELS_YAML = HERE.parent / "models.yaml"
OUTPUT_YAML = HERE / "field-guidance-suite.yaml"


def load_ground_truth() -> list[dict]:
    with open(GROUND_TRUTH) as f:
        return yaml.safe_load(f)["submissions"]


def load_models() -> list[str]:
    with open(MODELS_YAML) as f:
        return yaml.safe_load(f)["models"]


def fetch_submission(collection, submission_id: str) -> dict | None:  # type: ignore[type-arg]
    return collection.find_one({"id": submission_id})


def format_prompt(doc: dict) -> str:
    """Build a prompt from the submission document using the production
    suggestor's get_submission_fields() and SchemaContextBuilder.

    This reproduces what run_recommendation_pipeline() sends to the LLM,
    minus DOI abstract retrieval and PDF downloads (which are network-dependent
    and can't be baked into a static suite YAML).
    """
    parsed = get_submission_fields(submission_object=doc)

    # Submission context — same format as recommendation_pipeline.py
    parts = [
        f"Submission description: {parsed.get('description', '')}",
        f"Notes: {parsed.get('notes', '')}",
        f"Study name: {parsed.get('study_name', '')}",
        f"Protocol descriptions: {'; '.join(parsed.get('protocol_descs', []))}",
        f"Protocol names: {'; '.join(parsed.get('protocol_names', []))}",
    ]
    submission_context = "\n".join(parts)

    # Schema context — the real slot definitions, enums, exclusions
    mixis_extensions = parsed.get("mixis_extensions", [])
    schema_context = ""
    if mixis_extensions:
        builder = SchemaContextBuilder()
        schema_context = builder.format_multi_interface_context(mixis_extensions)

    prompt_parts = [submission_context]
    if schema_context:
        prompt_parts.append(f"\n--- NMDC Schema Context ---\n{schema_context}")
    prompt_parts.append("\nUtilize the provided information to inform your metadata field recommendations.")

    return "\n".join(prompt_parts)


def make_ideal(expected_slots: list[dict]) -> str:
    """JSON-formatted ideal answer matching the production output schema."""
    import json

    fields = [
        {"field_name": s["field_name"], "reason": s.get("justification", ""), "value": ""} for s in expected_slots
    ]
    return json.dumps({"metadata_fields": fields})


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate field-guidance llm-matrix suite")
    parser.add_argument(
        "--mongo-uri",
        default="mongodb://localhost:27017/nmdc_data_dev",
        help="MongoDB URI for data-dev submissions",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_YAML,
        help="Output suite YAML path",
    )
    args = parser.parse_args()

    ground_truth = load_ground_truth()
    models = load_models()

    client = MongoClient(args.mongo_uri)
    db = client.get_default_database()
    collection = db["nmdc_submissions"]

    cases = []
    for entry in ground_truth:
        doc = fetch_submission(collection, entry["submission_id"])
        if doc is None:
            print(f"WARNING: submission {entry['submission_id']} not found in MongoDB, skipping")
            continue

        prompt_text = format_prompt(doc)
        ideal = make_ideal(entry["expected_slots"])

        cases.append(
            {
                "input": prompt_text,
                "ideal": ideal,
                "tags": [entry["package"], entry["curator"]],
                "original_input": {
                    "submission_id": entry["submission_id"],
                    "study_name": entry["study_name"],
                    "package": entry["package"],
                    "expected_slot_count": len(entry["expected_slots"]),
                },
            }
        )

    suite = {
        "name": "nmdc-field-guidance",
        "description": (
            "Predict which biosample slots should be filled, given submission-level "
            "metadata. Uses the production suggestor's system prompt and schema context. "
            "Ground truth from Montana Smith and Bea Meluch (microbiomedata/issues#1551)."
        ),
        "template": "predict_slots",
        "templates": {
            "predict_slots": {
                "system": PRODUCTION_SYSTEM_PROMPT,
                "prompt": "{input}",
                "metrics": ["simple_question"],
            }
        },
        "matrix": {
            "hyperparameters": {
                "model": models,
                "temperature": [0.0],
            }
        },
        "cases": cases,
    }

    with open(args.output, "w") as f:
        yaml.dump(suite, f, default_flow_style=False, sort_keys=False, width=120, allow_unicode=True)

    print(f"Generated {len(cases)} cases x {len(models)} models -> {args.output}")


if __name__ == "__main__":
    main()
