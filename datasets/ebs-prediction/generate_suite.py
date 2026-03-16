#!/usr/bin/env python3
"""Generate llm-matrix suite YAML for env_broad_scale prediction.

Samples N rows per env_broad_scale value and writes a single suite YAML
with all models from models.yaml.

Usage:
    python generate_suite.py                              # defaults: 10 per category, min pool 10
    python generate_suite.py --per-category 10            # 10 per category
    python generate_suite.py --min-pool 5                 # include categories with >=5 unique rows
"""

import argparse
import csv
import random
import textwrap
from collections import defaultdict
from pathlib import Path

import yaml

HERE = Path(__file__).parent
DEFAULT_TSV = HERE.parent / "submission-metadata-prediction" / "eval_input_target_pairs.tsv"
ENUM_DIR = HERE / "enum_data"
MODELS_YAML = Path(__file__).parent.parent / "models.yaml"

# Template → enum file prefix (matches envo_scorer._TEMPLATE_TO_ENUM_PREFIX)
_TEMPLATE_TO_ENUM_PREFIX: dict[str, str] = {
    "soil_data": "soil",
    "water_data": "water",
    "sediment_data": "sediment",
    "plant_associated_data": "plant_associated",
}

SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert in environmental ontologies (ENVO) and the MIxS metadata standard.
    Given metadata about an NMDC biosample, predict the env_broad_scale value.

    Your answer MUST be in the exact format: label [CURIE]
    For example: terrestrial biome [ENVO:00000446]

    Use only ENVO terms. Reply with ONLY the env_broad_scale value, nothing else.""")

PROMPT_TEMPLATE = textwrap.dedent("""\
    Predict the env_broad_scale (broad-scale environmental context) for this biosample.

    Study: {study_name}
    Description: {description}
    Notes: {notes}
    Environmental package: {sampleData}
    env_local_scale: {env_local_scale}
    env_medium: {env_medium}
    Geographic location: {geo_loc_name}
    Depth: {depth}
    Analysis type: {analysis_type}
    {allowed_values_section}""")

# Non-GOLD input columns (excludes ecosystem, ecosystem_type, etc.)
INPUT_COLUMNS = [
    "study_name",
    "description",
    "notes",
    "sampleData",
    "env_local_scale",
    "env_medium",
    "geo_loc_name",
    "depth",
    "analysis_type",
]


def load_models() -> list[str]:
    """Load model list from shared models.yaml."""
    with open(MODELS_YAML) as f:
        return yaml.safe_load(f)["models"]


def load_allowed_values(template: str) -> list[str] | None:
    """Load allowed env_broad_scale values for a template as 'label [CURIE]' strings."""
    prefix = _TEMPLATE_TO_ENUM_PREFIX.get(template)
    if prefix is None:
        return None
    tsv_path = ENUM_DIR / f"{prefix}_env_broad_scale.tsv"
    if not tsv_path.exists():
        return None
    values: list[str] = []
    with open(tsv_path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            values.append(f"{row['label']} [{row['id']}]")
    return sorted(values)


def load_rows(tsv_path: Path) -> list[dict]:
    with open(tsv_path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def sample_by_category(rows: list[dict], n_per_category: int, min_pool: int, seed: int = 42) -> list[dict]:
    # Filter to rows with non-empty env_broad_scale
    rows = [r for r in rows if r["env_broad_scale"].strip()]

    # Deduplicate on prompt-relevant columns + target only.
    # Excludes GOLD ecosystem columns (not used in prompts).
    dedup_columns = [*INPUT_COLUMNS, "env_broad_scale"]
    seen: set[tuple[str, ...]] = set()
    unique_rows: list[dict] = []
    for row in rows:
        key = tuple(row[col] for col in dedup_columns)
        if key not in seen:
            seen.add(key)
            unique_rows.append(row)

    by_cat: dict[str, list[dict]] = defaultdict(list)
    for row in unique_rows:
        by_cat[row["env_broad_scale"]].append(row)

    # Report included vs excluded categories
    included = {cat: pool for cat, pool in by_cat.items() if len(pool) >= min_pool}
    excluded = {cat: pool for cat, pool in by_cat.items() if len(pool) < min_pool}
    if excluded:
        print(f"  Excluded {len(excluded)} categor{'y' if len(excluded) == 1 else 'ies'} with <{min_pool} unique rows:")
        for cat in sorted(excluded):
            print(f"    {len(excluded[cat]):4d}  {cat}")
    print(f"  Including {len(included)} categor{'y' if len(included) == 1 else 'ies'}:")
    for cat in sorted(included):
        print(f"    {len(included[cat]):4d}  {cat} (sampling {min(n_per_category, len(included[cat]))})")

    rng = random.Random(seed)
    sampled = []
    for cat in sorted(included):
        pool = included[cat]
        k = min(n_per_category, len(pool))
        sampled.extend(rng.sample(pool, k))
    return sampled


def _clean_value(val: str) -> str:
    """Strip leading underscores from values (data quirk)."""
    return val.lstrip("_")


def _allowed_values_section(template: str) -> str:
    """Build the allowed values prompt section for a template."""
    values = load_allowed_values(template)
    if values is None:
        return ""
    lines = "\n".join(f"  - {v}" for v in values)
    return f"Allowed env_broad_scale values for {template}:\n{lines}\n\nChoose from the above list."


# Cache allowed values sections per template (same template = same section).
_allowed_cache: dict[str, str] = {}


def make_cases(sampled_rows: list[dict]) -> list[dict]:
    cases = []
    for row in sampled_rows:
        desc = row["description"]
        if len(desc) > 500:
            desc = desc[:497] + "..."

        template = row.get("sampleData", "")
        if template not in _allowed_cache:
            _allowed_cache[template] = _allowed_values_section(template)

        prompt_values = {}
        for col in INPUT_COLUMNS:
            prompt_values[col] = _clean_value(row.get(col, ""))

        prompt_values["description"] = desc
        prompt_values["allowed_values_section"] = _allowed_cache[template]

        ideal = _clean_value(row["env_broad_scale"])

        cases.append(
            {
                "input": PROMPT_TEMPLATE.format(**prompt_values),
                "ideal": ideal,
                "tags": [row["sampleData"], ideal],
                "original_input": {
                    "study_name": row["study_name"],
                    "sampleData": row["sampleData"],
                    "env_broad_scale": row["env_broad_scale"],
                },
            }
        )
    return cases


def make_suite(cases: list[dict], models: list[str]) -> dict:
    return {
        "name": "nmdc-ebs-prediction",
        "description": (
            "Predict env_broad_scale from non-GOLD biosample metadata. "
            "Generated from eval_input_target_pairs.tsv. "
            "Post-hoc scoring with envo_scorer adds ontology-aware metrics."
        ),
        "template": "predict_ebs",
        "templates": {
            "predict_ebs": {
                "system": SYSTEM_PROMPT,
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tsv", type=Path, default=DEFAULT_TSV, help="Input TSV path")
    parser.add_argument(
        "--per-category",
        type=int,
        default=10,
        help="Number of samples per env_broad_scale category (default: 10)",
    )
    parser.add_argument(
        "--min-pool",
        type=int,
        default=10,
        help="Exclude categories with fewer unique rows than this (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for suite YAML (default: same directory as this script)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    out_dir = args.output_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)

    models = load_models()
    rows = load_rows(args.tsv)
    sampled = sample_by_category(rows, args.per_category, min_pool=args.min_pool, seed=args.seed)
    cases = make_cases(sampled)

    suite = make_suite(cases, models)
    output_path = out_dir / "ebs-suite.yaml"
    with open(output_path, "w") as f:
        yaml.dump(suite, f, default_flow_style=False, sort_keys=False, width=120)
    print(f"Wrote {len(cases)} cases × {len(models)} models to {output_path}")


if __name__ == "__main__":
    main()
