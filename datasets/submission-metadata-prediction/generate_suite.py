#!/usr/bin/env python3
"""Generate llm-matrix suite YAML from the eval TSV.

Samples N rows per sampleData value and writes a single suite YAML
with all models from models.yaml.

Usage:
    python generate_suite.py                              # defaults: 10 per category, min pool 5
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
DEFAULT_TSV = HERE / "eval_input_target_pairs.tsv"
MODELS_YAML = Path(__file__).parent.parent / "models.yaml"

SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert in environmental metadata standards (MIxS, NMDC, GOLD).
    Given a study name and description from an NMDC submission, predict the
    MIxS environmental package (sampleData template).

    Valid values: soil_data, water_data, plant_associated_data, misc_envs_data,
    host_associated_data, sediment_data, air_data,
    metagenome_sequencing_non_interleaved_data

    Reply with ONLY the package name, nothing else.""")

PROMPT_TEMPLATE = "Study: {study_name}\nDescription: {description}"


def load_models() -> list[str]:
    """Load model list from shared models.yaml."""
    with open(MODELS_YAML) as f:
        return yaml.safe_load(f)["models"]


def load_rows(tsv_path: Path) -> list[dict]:
    with open(tsv_path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def sample_by_category(rows: list[dict], n_per_category: int, min_pool: int, seed: int = 42) -> list[dict]:
    # Deduplicate on prompt-relevant columns + target only.
    # Excludes GOLD ecosystem columns and env triad (not used in prompts).
    dedup_columns = ["study_name", "description", "notes", "sampleData"]
    seen: set[tuple[str, ...]] = set()
    unique_rows: list[dict] = []
    for row in rows:
        key = tuple(row[col] for col in dedup_columns)
        if key not in seen:
            seen.add(key)
            unique_rows.append(row)

    by_cat: dict[str, list[dict]] = defaultdict(list)
    for row in unique_rows:
        by_cat[row["sampleData"]].append(row)

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


def make_cases(sampled_rows: list[dict]) -> list[dict]:
    cases = []
    for row in sampled_rows:
        desc = row["description"]
        if len(desc) > 500:
            desc = desc[:497] + "..."
        cases.append(
            {
                "input": PROMPT_TEMPLATE.format(study_name=row["study_name"], description=desc),
                "ideal": row["sampleData"],
                "tags": [row["sampleData"]],
                "original_input": {
                    "study_name": row["study_name"],
                    "sampleData": row["sampleData"],
                },
            }
        )
    return cases


def make_suite(cases: list[dict], models: list[str]) -> dict:
    return {
        "name": "nmdc-sampledata-prediction",
        "description": (
            "Predict MIxS environmental package (sampleData) from NMDC "
            "submission study name and description. "
            "Generated from eval_input_target_pairs.tsv."
        ),
        "template": "predict_sample_type",
        "templates": {
            "predict_sample_type": {
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
        help="Number of samples per category (default: 10)",
    )
    parser.add_argument(
        "--min-pool",
        type=int,
        default=5,
        help="Exclude categories with fewer unique rows than this (default: 5)",
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
    output_path = out_dir / "sampledata-suite.yaml"
    with open(output_path, "w") as f:
        yaml.dump(suite, f, default_flow_style=False, sort_keys=False, width=120)
    print(f"Wrote {len(cases)} cases × {len(models)} models to {output_path}")


if __name__ == "__main__":
    main()
