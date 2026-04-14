#!/usr/bin/env python3
"""Generate an llm-matrix suite YAML for env triad prediction.

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
from nmdc_api_utilities.study_search import StudySearch
from nmdc_metadata_suggestor_ai_tool.schema_context import SchemaContextBuilder

HERE = Path(__file__).parent
MODELS_YAML = HERE.parent / "models.yaml"
OUTPUT_YAML = HERE / "env-triad-test-suite.yaml"


def load_models() -> list[str]:
    with open(MODELS_YAML) as f:
        return yaml.safe_load(f)["models"]


def get_bioscales_study_example() -> list[dict]:
    """Get all biosamples associated with the Bioscales study."""
    study_id = "nmdc:sty-11-r2h77870"
    # call NMDC API
    search = StudySearch()
    # biosample_search = BiosampleSearch()
    study = search.get_record_by_id(collection_id=study_id)
    # get the linked biosamples
    biosamples = search.get_linked_instances(ids=[study_id], types="nmdc:Biosample", hydrate=True, max_page_size=1999)
    # # get the ids of the biosamples
    # biosample_ids = [biosample["id"] for biosample in biosamples]
    # # get the full biosample records
    # biosamples = biosample_search.get_batch_records(id_list=biosample_ids, search_field="id")

    return {"study": study, "biosamples": biosamples}


def format_prompt(biosample: dict, study: dict) -> str:
    """Build a prompt from the biosample, study, and SchemaContextBuilder."""
    copy = biosample.copy()  # avoid mutating the original
    # remove env triad from biosample
    copy.pop("env_broad_scale", None)
    copy.pop("env_local_scale", None)
    copy.pop("env_medium", None)

    # Schema context
    builder = SchemaContextBuilder()
    mixs_schema = builder.format_env_triad_context(class_names=builder.list_interfaces())

    prompt_parts = [mixs_schema]
    prompt_parts.append(f"\n--- NMDC Schema Context ---\n{mixs_schema}")
    prompt_parts.append(f"\n--- Study Metadata ---\n{study}")
    prompt_parts.append(f"\n--- Biosample Metadata ---\n{copy}")
    prompt_parts.append("Suggest env triad values for the biosample")

    return "\n".join(prompt_parts)


def make_ideal(env_broad_scale, env_local_scale, env_medium) -> str:
    """JSON-formatted ideal answer matching the production output schema."""
    import json

    fields = [
        {"field_name": "env_broad_scale", "reason": "", "value": env_broad_scale},
        {"field_name": "env_local_scale", "reason": "", "value": env_local_scale},
        {"field_name": "env_medium", "reason": "", "value": env_medium},
    ]
    return json.dumps({"metadata_fields": fields})


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate env-triad llm-matrix suite")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_YAML,
        help="Output suite YAML path",
    )
    args = parser.parse_args()

    models = load_models()
    study_example = get_bioscales_study_example()
    cases = []
    for biosample in study_example["biosamples"]:
        cases.append(
            {
                "input": format_prompt(biosample, study_example["study"]),
                "ideal": make_ideal(
                    biosample.get("env_broad_scale").get("raw_value"),
                    biosample.get("env_local_scale").get("raw_value"),
                    biosample.get("env_medium").get("raw_value"),
                ),
                "tags": ["value_prediction"],
            }
        )

    suite = {
        "cases": cases,
    }

    with open(args.output, "w") as f:
        yaml.dump(suite, f, default_flow_style=False, sort_keys=False, width=120, allow_unicode=True)

    print(f"Generated {len(cases)} cases x {len(models)} models -> {args.output}")


if __name__ == "__main__":
    main()
