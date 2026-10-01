"""Validate that all evaluation suite YAMLs parse correctly."""

from io import StringIO
from pathlib import Path

import pytest
from pydantic import ValidationError

from nmdc_ai_eval.suite import Suite

DATASETS_DIR = Path(__file__).parent.parent / "datasets"


def find_suite_yamls() -> list[Path]:
    return sorted(DATASETS_DIR.rglob("*-suite*.yaml"))


@pytest.fixture(params=find_suite_yamls(), ids=lambda p: p.name)
def suite_path(request: pytest.FixtureRequest) -> Path:
    return request.param  # type: ignore[no-any-return]


def test_suite_loads(suite_path: Path) -> None:
    """Each suite YAML must parse into a valid Suite object."""
    suite = Suite.load(suite_path)
    assert suite.name, "Suite must have a name"
    assert suite.cases, "Suite must have at least one test case"
    assert suite.matrix.hyperparameters, "Suite must define hyperparameters"


def test_suite_cases_have_ideals(suite_path: Path) -> None:
    """Every test case should have an ideal answer for scoring."""
    suite = Suite.load(suite_path)
    for i, case in enumerate(suite.cases):
        assert case.ideal is not None, f"Case {i} ({case.input[:50]}...) missing ideal answer"


def test_suite_templates_referenced(suite_path: Path) -> None:
    """If a suite references a template, it must be defined."""
    suite = Suite.load(suite_path)
    if suite.template:
        assert suite.templates, f"Suite references template '{suite.template}' but templates dict is missing/empty"
        assert suite.template in suite.templates, (
            f"Suite references template '{suite.template}' but it's not in templates dict"
        )


def test_suite_rejects_null_case_input() -> None:
    suite_yaml = StringIO(
        """name: invalid-suite
matrix:
  hyperparameters:
    model: [gpt-4o-mini]
cases:
  - input:
    ideal: expected
"""
    )

    with pytest.raises(ValidationError, match="input"):
        Suite.load(suite_yaml)


def test_suite_rejects_unknown_fields() -> None:
    suite_yaml = StringIO(
        """name: invalid-suite
unexpected: true
matrix:
  hyperparameters:
    model: [gpt-4o-mini]
cases:
  - input: prompt
    ideal: expected
"""
    )

    with pytest.raises(ValidationError, match="unexpected"):
        Suite.load(suite_yaml)
