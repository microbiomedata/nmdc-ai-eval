"""Validate that all llm-matrix suite YAMLs parse correctly."""

from pathlib import Path

import pytest
from llm_matrix.schema import load_suite  # type: ignore[import-untyped]

DATASETS_DIR = Path(__file__).parent.parent / "datasets"


def find_suite_yamls() -> list[Path]:
    return sorted(DATASETS_DIR.rglob("*-suite*.yaml"))


@pytest.fixture(params=find_suite_yamls(), ids=lambda p: p.name)
def suite_path(request: pytest.FixtureRequest) -> Path:
    return request.param  # type: ignore[no-any-return]


def test_suite_loads(suite_path: Path) -> None:
    """Each suite YAML must parse into a valid Suite object."""
    suite = load_suite(suite_path)
    assert suite.name, "Suite must have a name"
    assert suite.cases, "Suite must have at least one test case"
    assert suite.matrix.hyperparameters, "Suite must define hyperparameters"


def test_suite_cases_have_ideals(suite_path: Path) -> None:
    """Every test case should have an ideal answer for scoring."""
    suite = load_suite(suite_path)
    for i, case in enumerate(suite.cases):
        assert case.ideal is not None, f"Case {i} ({case.input[:50]}...) missing ideal answer"


def test_suite_templates_referenced(suite_path: Path) -> None:
    """If a suite references a template, it must be defined."""
    suite = load_suite(suite_path)
    if suite.template and suite.templates:
        assert suite.template in suite.templates, (
            f"Suite references template '{suite.template}' but it's not in templates dict"
        )
