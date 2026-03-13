"""Test envo suite generation and YAML validity."""

import csv
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from llm_matrix.schema import load_suite  # type: ignore[import-untyped]

DATASETS_DIR = Path(__file__).parent.parent / "datasets"
ENVO_DIR = DATASETS_DIR / "ebs-prediction"
TSV_PATH = DATASETS_DIR / "submission-metadata-prediction" / "eval_input_target_pairs.tsv"


@pytest.fixture(scope="module")
def generated_suites(tmp_path_factory: pytest.TempPathFactory) -> list[Path]:
    """Generate suites into a temp dir so we don't clobber the real YAMLs."""
    tmp_dir = tmp_path_factory.mktemp("ebs_suites")
    # Copy the generator script and enum_data to temp dir
    shutil.copy(ENVO_DIR / "generate_suite.py", tmp_dir / "generate_suite.py")
    enum_src = ENVO_DIR / "enum_data"
    if enum_src.exists():
        shutil.copytree(enum_src, tmp_dir / "enum_data")
    subprocess.run(
        [
            sys.executable,
            str(tmp_dir / "generate_suite.py"),
            "--tsv",
            str(TSV_PATH),
            "--per-category",
            "2",
            "--min-pool",
            "2",
        ],
        check=True,
        cwd=str(tmp_dir),
    )
    return sorted(tmp_dir.glob("ebs-suite-*.yaml"))


def test_suites_created(generated_suites: list[Path]) -> None:
    assert len(generated_suites) == 2
    names = {p.name for p in generated_suites}
    assert "ebs-suite-openai.yaml" in names
    assert "ebs-suite-anthropic.yaml" in names


def test_suites_parse(generated_suites: list[Path]) -> None:
    for path in generated_suites:
        suite = load_suite(path)
        assert suite.name
        assert suite.cases
        assert suite.matrix.hyperparameters


def test_cases_have_envo_ideals(generated_suites: list[Path]) -> None:
    for path in generated_suites:
        suite = load_suite(path)
        for i, case in enumerate(suite.cases):
            assert case.ideal, f"Case {i} missing ideal"
            assert "ENVO:" in case.ideal, f"Case {i} ideal '{case.ideal}' not an ENVO term"


def test_original_input_has_required_fields(generated_suites: list[Path]) -> None:
    for path in generated_suites:
        suite = load_suite(path)
        for i, case in enumerate(suite.cases):
            assert case.original_input, f"Case {i} missing original_input"
            assert "sampleData" in case.original_input, f"Case {i} missing sampleData"
            assert "env_broad_scale" in case.original_input, f"Case {i} missing env_broad_scale"


def test_ideals_match_tsv_values(generated_suites: list[Path]) -> None:
    with open(TSV_PATH, newline="") as f:
        tsv_values = set()
        for row in csv.DictReader(f, delimiter="\t"):
            val = row["env_broad_scale"].strip().lstrip("_")
            if val:
                tsv_values.add(val)
    for path in generated_suites:
        suite = load_suite(path)
        for i, case in enumerate(suite.cases):
            assert case.ideal in tsv_values, (
                f"{path.name} case {i}: ideal '{case.ideal}' not in TSV env_broad_scale values"
            )
