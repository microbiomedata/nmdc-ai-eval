"""Test EBS (env_broad_scale) suite generation and YAML validity."""

import csv
import subprocess
import sys
from pathlib import Path

import pytest
from llm_matrix.schema import load_suite  # type: ignore[import-untyped]

DATASETS_DIR = Path(__file__).parent.parent / "datasets"
EBS_DIR = DATASETS_DIR / "ebs-prediction"
TSV_PATH = DATASETS_DIR / "submission-metadata-prediction" / "eval_input_target_pairs.tsv"


@pytest.fixture(scope="module")
def generated_suite(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate suite into a temp dir so we don't clobber the real YAML."""
    tmp_dir = tmp_path_factory.mktemp("ebs_suites")
    subprocess.run(
        [
            sys.executable,
            str(EBS_DIR / "generate_suite.py"),
            "--tsv",
            str(TSV_PATH),
            "--per-category",
            "2",
            "--min-pool",
            "2",
            "--output-dir",
            str(tmp_dir),
        ],
        check=True,
    )
    suite_path = tmp_dir / "ebs-suite.yaml"
    assert suite_path.exists(), "ebs-suite.yaml was not generated"
    return suite_path


def test_suite_created(generated_suite: Path) -> None:
    assert generated_suite.name == "ebs-suite.yaml"


def test_suite_parses(generated_suite: Path) -> None:
    suite = load_suite(generated_suite)
    assert suite.name
    assert suite.cases
    assert suite.matrix.hyperparameters


def test_cases_have_envo_ideals(generated_suite: Path) -> None:
    suite = load_suite(generated_suite)
    for i, case in enumerate(suite.cases):
        assert case.ideal, f"Case {i} missing ideal"
        assert "ENVO:" in case.ideal, f"Case {i} ideal '{case.ideal}' not an ENVO term"


def test_original_input_has_required_fields(generated_suite: Path) -> None:
    suite = load_suite(generated_suite)
    for i, case in enumerate(suite.cases):
        assert case.original_input, f"Case {i} missing original_input"
        assert "sampleData" in case.original_input, f"Case {i} missing sampleData"
        assert "env_broad_scale" in case.original_input, f"Case {i} missing env_broad_scale"


def test_ideals_match_tsv_values(generated_suite: Path) -> None:
    with open(TSV_PATH, newline="") as f:
        tsv_values = set()
        for row in csv.DictReader(f, delimiter="\t"):
            val = row["env_broad_scale"].strip().lstrip("_")
            if val:
                tsv_values.add(val)
    suite = load_suite(generated_suite)
    for i, case in enumerate(suite.cases):
        assert case.ideal in tsv_values, (
            f"ebs-suite.yaml case {i}: ideal '{case.ideal}' not in TSV env_broad_scale values"
        )
