#!/usr/bin/env python3
"""Generate an llm-matrix suite YAML from the eval TSV.

Samples N rows per sampleData value and writes a suite YAML that
llm-runner can execute directly.

Usage:
    python generate_suite.py                         # defaults: 5 per category
    python generate_suite.py --per-category 10       # 10 per category
    python generate_suite.py -o my-suite.yaml        # custom output path
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
DEFAULT_OUTPUT = HERE / "sampledata-suite.yaml"

SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert in environmental metadata standards (MIxS, NMDC, GOLD).
    Given a study name and description from an NMDC submission, predict the
    MIxS environmental package (sampleData template).

    Valid values: soil_data, water_data, plant_associated_data, misc_envs_data,
    host_associated_data, sediment_data, air_data,
    metagenome_sequencing_non_interleaved_data

    Reply with ONLY the package name, nothing else.""")

PROMPT_TEMPLATE = "Study: {study_name}\nDescription: {description}"


def load_rows(tsv_path: Path) -> list[dict]:
    with open(tsv_path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def sample_by_category(
    rows: list[dict], n_per_category: int, seed: int = 42
) -> list[dict]:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_cat[row["sampleData"]].append(row)
    rng = random.Random(seed)
    sampled = []
    for cat in sorted(by_cat):
        pool = by_cat[cat]
        k = min(n_per_category, len(pool))
        sampled.extend(rng.sample(pool, k))
    return sampled


def make_suite(sampled_rows: list[dict]) -> dict:
    cases = []
    for row in sampled_rows:
        desc = row["description"]
        # Truncate very long descriptions to keep the suite readable
        if len(desc) > 500:
            desc = desc[:497] + "..."
        cases.append(
            {
                "input": PROMPT_TEMPLATE.format(
                    study_name=row["study_name"], description=desc
                ),
                "ideal": row["sampleData"],
                "tags": [row["sampleData"]],
                "original_input": {
                    "study_name": row["study_name"],
                    "sampleData": row["sampleData"],
                },
            }
        )
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
                "model": ["gpt-4o-mini", "gpt-4o"],
                "temperature": [0.0],
            }
        },
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tsv", type=Path, default=DEFAULT_TSV, help="Input TSV path"
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=DEFAULT_OUTPUT, help="Output YAML path"
    )
    parser.add_argument(
        "--per-category",
        type=int,
        default=5,
        help="Number of samples per sampleData category",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    rows = load_rows(args.tsv)
    sampled = sample_by_category(rows, args.per_category, seed=args.seed)
    suite = make_suite(sampled)
    with open(args.output, "w") as f:
        yaml.dump(suite, f, default_flow_style=False, sort_keys=False, width=120)
    print(f"Wrote {len(sampled)} cases to {args.output}")


if __name__ == "__main__":
    main()
