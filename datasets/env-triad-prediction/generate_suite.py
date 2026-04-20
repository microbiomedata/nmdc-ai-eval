#!/usr/bin/env python3
"""Generate an llm-matrix suite YAML for env triad prediction.

Pulls biosamples from NMDC studies via the public API and builds a suite
where each case asks the model to predict ``env_broad_scale``,
``env_local_scale``, and ``env_medium`` for one biosample. Ideals are
built as ``label [CURIE]`` from the curated ``term`` data on the biosample
(e.g. ``"terrestrial biome [ENVO:00000446]"``).

The MIxS schema context — the per-interface definitions of the three env
triad slots — is included once in the template's ``system`` prompt, not
per case. It is built dynamically via the production suggestor's
``SchemaContextBuilder`` so it stays in sync with upstream MIxS changes.

Biosamples missing any of the three env-triad term values are skipped
(they can't produce a scorable ideal). The skipped count is reported.

Usage::

    uv run python datasets/env-triad-prediction/generate_suite.py
    uv run python datasets/env-triad-prediction/generate_suite.py --study-id nmdc:sty-...
"""

import argparse
import json
from pathlib import Path
from typing import Any

import yaml
from nmdc_api_utilities.study_search import StudySearch
from nmdc_metadata_suggestor_ai_tool.schema_context import SchemaContextBuilder, format_slot

HERE = Path(__file__).parent
MODELS_YAML = HERE.parent / "models.yaml"
OUTPUT_YAML = HERE / "env-triad-suite.yaml"
STUDIES_YAML = HERE / "studies.yaml"

ENV_TRIAD_SLOTS = ["env_broad_scale", "env_local_scale", "env_medium"]

SYSTEM_PROMPT_HEAD = """Use the provided information to suggest values for the
following three metadata fields: env_broad_scale, env_local_scale, and env_medium.
- env_broad_scale: the broad environmental category of the sample (e.g. aquatic, terrestrial, host-associated).
- env_local_scale: the more specific local environment of the sample (e.g. sediment, water column, rhizosphere).
- env_medium: the medium in which the sample was collected (e.g. soil, water, air).

Values must be chosen from the enumerations in the NMDC schema and should be
based on the content of the information provided. The value format is
``label [CURIE]`` (e.g. ``terrestrial biome [ENVO:00000446]``).

Output schema:
```json
{
    "metadata_fields": [
        {"field_name": "env_broad_scale", "reason": "...", "value": ""},
        {"field_name": "env_local_scale", "reason": "...", "value": ""},
        {"field_name": "env_medium", "reason": "...", "value": ""}
    ]
}
```

Schema context to choose from:
"""


def build_env_triad_schema_context() -> str:
    """Build the env-triad-specific MIxS schema context string.

    Replicates ``SchemaContextBuilder.format_env_triad_context`` (present
    in newer suggestor versions) using v1.1-compatible primitives. Once
    the suggestor dep is bumped to a version shipping
    ``format_env_triad_context``, this can collapse to a single call.
    """
    builder = SchemaContextBuilder()
    sections: list[str] = []
    for interface_name in builder.list_interfaces():
        schema = builder.get_interface_schema(interface_name)
        triad_slots = [s for s in schema.slots if s.name in ENV_TRIAD_SLOTS]
        if not triad_slots:
            continue
        lines = [f"# {schema.class_name} - Selected Fields"]
        for slot in triad_slots:
            lines.extend(format_slot(slot))
        sections.append("\n".join(lines))
    return "\n\n---\n\n".join(sections)


def build_system_prompt() -> str:
    return SYSTEM_PROMPT_HEAD + build_env_triad_schema_context()


def load_models() -> list[str]:
    with open(MODELS_YAML) as f:
        return yaml.safe_load(f)["models"]


def load_study_ids(studies_yaml: Path) -> list[str]:
    """Read the curated list of study IDs from studies.yaml."""
    with open(studies_yaml) as f:
        config = yaml.safe_load(f)
    return [entry["id"] for entry in config.get("studies", [])]


def get_study_with_biosamples(study_id: str) -> dict[str, Any]:
    """Fetch one study and its linked biosamples from the NMDC public API."""
    search = StudySearch()
    study = search.get_record_by_id(collection_id=study_id)
    biosamples = search.get_linked_instances(
        ids=[study_id],
        types="nmdc:Biosample",
        hydrate=True,
        max_page_size=1999,
    )
    return {"study": study, "biosamples": biosamples}


def format_prompt(biosample: dict, study: dict) -> str:
    """Build the per-case prompt from biosample + study metadata.

    Schema context is intentionally NOT here — that lives in the template's
    system prompt so it's sent once per run, not once per case.
    """
    clean = biosample.copy()
    # Strip the ground-truth env triad fields so the model can't cheat
    for slot in ENV_TRIAD_SLOTS:
        clean.pop(slot, None)
    return (
        f"--- Study Metadata ---{study}\n--- Biosample Metadata ---{clean}\nSuggest env triad values for the biosample"
    )


def _term_to_label_curie(term_value: dict | None) -> str | None:
    """Format a TermValue dict as ``'label [CURIE]'``.

    Returns ``None`` if the term data is missing or incomplete (biosample
    field absent, ``term`` missing, or either ``name``/``id`` empty).
    Callers should skip cases where any env-triad term is ``None`` rather
    than emit an ideal the scorer can't match.
    """
    if not term_value:
        return None
    term = term_value.get("term")
    if not term:
        return None
    name = term.get("name")
    term_id = term.get("id")
    if not name or not term_id:
        return None
    return f"{name} [{term_id}]"


def make_ideal(biosample: dict) -> str | None:
    """Build the ideal answer as JSON, or ``None`` if any env-triad term
    is missing/malformed on the biosample."""
    values: dict[str, str | None] = {slot: _term_to_label_curie(biosample.get(slot)) for slot in ENV_TRIAD_SLOTS}
    if any(v is None for v in values.values()):
        return None
    fields = [{"field_name": slot, "reason": "", "value": values[slot]} for slot in ENV_TRIAD_SLOTS]
    return json.dumps({"metadata_fields": fields})


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate env-triad llm-matrix suite")
    parser.add_argument("--output", type=Path, default=OUTPUT_YAML)
    parser.add_argument(
        "--study-id",
        type=str,
        action="append",
        default=None,
        help=(
            "NMDC study ID. May be repeated. If omitted, reads the curated "
            f"list from {STUDIES_YAML.relative_to(HERE.parent.parent)}."
        ),
    )
    parser.add_argument(
        "--studies-yaml",
        type=Path,
        default=STUDIES_YAML,
        help="Alternate studies.yaml config path (ignored if --study-id is given).",
    )
    args = parser.parse_args()

    study_ids: list[str] = args.study_id if args.study_id else load_study_ids(args.studies_yaml)
    models = load_models()

    cases: list[dict[str, Any]] = []
    per_study_counts: list[tuple[str, int, int]] = []  # (id, kept, skipped)

    for study_id in study_ids:
        data = get_study_with_biosamples(study_id)
        study = data["study"]
        kept = 0
        skipped = 0
        for biosample in data["biosamples"]:
            ideal = make_ideal(biosample)
            if ideal is None:
                skipped += 1
                continue
            cases.append(
                {
                    "input": format_prompt(biosample, study),
                    "ideal": ideal,
                    "tags": ["value_prediction", study_id],
                }
            )
            kept += 1
        per_study_counts.append((study_id, kept, skipped))

    suite = {
        "name": "Env Triad Prediction Test Suite",
        "template": "env_triad_prediction",
        "templates": {
            "env_triad_prediction": {
                "system": build_system_prompt(),
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

    total_kept = sum(k for _, k, _ in per_study_counts)
    total_skipped = sum(s for _, _, s in per_study_counts)
    print(f"Generated {total_kept} cases ({total_skipped} skipped) x {len(models)} models -> {args.output}")
    for study_id, kept, skipped in per_study_counts:
        print(f"  {study_id}: {kept} cases ({skipped} skipped)")


if __name__ == "__main__":
    main()
