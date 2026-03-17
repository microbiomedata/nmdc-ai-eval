"""Validate that suite YAML cases are consistent with the source TSV."""

import csv
from pathlib import Path

from llm_matrix.schema import load_suite  # type: ignore[import-untyped]

DATASETS_DIR = Path(__file__).parent.parent / "datasets"
TSV_PATH = DATASETS_DIR / "submission-metadata-prediction" / "eval_input_target_pairs.tsv"


def _load_tsv_values(column: str) -> set[str]:
    with open(TSV_PATH, newline="") as f:
        values = set()
        for row in csv.DictReader(f, delimiter="\t"):
            val = row[column].strip().lstrip("_")
            if val:
                values.add(val)
        return values


def _find_suites(pattern: str = "*-suite*.yaml") -> list[Path]:
    return sorted(DATASETS_DIR.rglob(pattern))


def test_sampledata_suite_ideals_match_tsv() -> None:
    """Every ideal in a sampledata suite must be a sampleData value in the TSV."""
    tsv_values = _load_tsv_values("sampleData")
    for suite_path in _find_suites("sampledata-suite*.yaml"):
        suite = load_suite(suite_path)
        for i, case in enumerate(suite.cases):
            if case.ideal:
                assert case.ideal in tsv_values, (
                    f"{suite_path.name} case {i}: ideal '{case.ideal}' not found in TSV sampleData values"
                )


def test_envo_suite_ideals_match_tsv() -> None:
    """Every ideal in an envo suite must be an env_broad_scale value in the TSV."""
    tsv_values = _load_tsv_values("env_broad_scale")
    for suite_path in _find_suites("ebs-suite*.yaml"):
        suite = load_suite(suite_path)
        for i, case in enumerate(suite.cases):
            if case.ideal:
                assert case.ideal in tsv_values, (
                    f"{suite_path.name} case {i}: ideal '{case.ideal}' not found in TSV env_broad_scale values"
                )


def test_suite_original_inputs_match_tsv() -> None:
    """If a suite case has original_input.study_name, it must appear in the TSV."""
    with open(TSV_PATH, newline="") as f:
        tsv_study_names = {row["study_name"] for row in csv.DictReader(f, delimiter="\t")}
    for suite_path in _find_suites():
        suite = load_suite(suite_path)
        for i, case in enumerate(suite.cases):
            if case.original_input and "study_name" in case.original_input:
                assert case.original_input["study_name"] in tsv_study_names, (
                    f"{suite_path.name} case {i}: study_name '{case.original_input['study_name'][:50]}' not in TSV"
                )
