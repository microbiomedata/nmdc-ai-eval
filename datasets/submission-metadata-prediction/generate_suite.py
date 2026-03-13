#!/usr/bin/env python3
"""Generate llm-matrix suite YAMLs from the eval TSV.

Samples N rows per sampleData value and writes per-provider suite YAMLs.

Usage:
    python generate_suite.py                              # both providers, 5 per category
    python generate_suite.py --per-category 10            # 10 per category
    python generate_suite.py --provider openai             # OpenAI only
    python generate_suite.py --provider anthropic          # Anthropic only
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

SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert in environmental metadata standards (MIxS, NMDC, GOLD).
    Given a study name and description from an NMDC submission, predict the
    MIxS environmental package (sampleData template).

    Valid values: soil_data, water_data, plant_associated_data, misc_envs_data,
    host_associated_data, sediment_data, air_data,
    metagenome_sequencing_non_interleaved_data

    Reply with ONLY the package name, nothing else.""")

PROMPT_TEMPLATE = "Study: {study_name}\nDescription: {description}"

PROVIDER_MODELS: dict[str, list[str]] = {
    "openai": ["gpt-4o-mini", "gpt-4o"],
    "anthropic": ["anthropic/claude-sonnet-4-5", "anthropic/claude-haiku-4-5-20251001"],
}


def load_rows(tsv_path: Path) -> list[dict]:
    with open(tsv_path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def sample_by_category(rows: list[dict], n_per_category: int, seed: int = 42) -> list[dict]:
    # Deduplicate: biosamples within a study often share identical metadata.
    seen: set[tuple[str, ...]] = set()
    unique_rows: list[dict] = []
    for row in rows:
        key = tuple(row.values())
        if key not in seen:
            seen.add(key)
            unique_rows.append(row)

    by_cat: dict[str, list[dict]] = defaultdict(list)
    for row in unique_rows:
        by_cat[row["sampleData"]].append(row)
    rng = random.Random(seed)
    sampled = []
    for cat in sorted(by_cat):
        pool = by_cat[cat]
        k = min(n_per_category, len(pool))
        if k < n_per_category:
            print(f"  Warning: {cat} has only {len(pool)} unique row(s), requested {n_per_category}")
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


def make_suite(cases: list[dict], models: list[str], provider: str) -> dict:
    return {
        "name": f"nmdc-sampledata-prediction-{provider}",
        "description": (
            f"Predict MIxS environmental package (sampleData) from NMDC "
            f"submission study name and description ({provider} models). "
            f"Generated from eval_input_target_pairs.tsv."
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
        default=5,
        help="Number of samples per sampleData category",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "both"],
        default="both",
        help="Which provider suite(s) to generate (default: both)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    rows = load_rows(args.tsv)
    sampled = sample_by_category(rows, args.per_category, seed=args.seed)
    cases = make_cases(sampled)

    providers = list(PROVIDER_MODELS) if args.provider == "both" else [args.provider]
    for provider in providers:
        suite = make_suite(cases, PROVIDER_MODELS[provider], provider)
        output_path = HERE / f"sampledata-suite-{provider}.yaml"
        with open(output_path, "w") as f:
            yaml.dump(suite, f, default_flow_style=False, sort_keys=False, width=120)
        print(f"Wrote {len(cases)} cases × {len(PROVIDER_MODELS[provider])} models to {output_path}")


if __name__ == "__main__":
    main()
