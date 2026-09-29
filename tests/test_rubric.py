"""The judge rubrics in rubrics/ must load against their schema and stay versioned."""

import pytest
from pydantic import ValidationError

from nmdc_ai_eval.rubric import RUBRICS_DIR, JudgeAnswer, Rubric, load_rubric

RUBRIC_FILES = sorted(RUBRICS_DIR.glob("*.yaml"))
FIVE = {"accuracy", "factuality", "relevancy", "completeness", "coherence"}


def test_there_is_a_rubric() -> None:
    assert RUBRIC_FILES


@pytest.mark.parametrize("path", RUBRIC_FILES, ids=lambda p: p.name)
def test_rubric_loads_and_its_version_is_its_file_name(path) -> None:
    """Every score records the version, so two files must not share one."""
    assert load_rubric(path).version == path.stem


def test_suggestion_rubric_asks_the_five_criteria() -> None:
    rubric = load_rubric(RUBRICS_DIR / "suggestion-judge-v1.yaml")
    assert {c.id for c in rubric.criteria} == FIVE


def _rubric(**criterion) -> dict:
    base = {
        "id": "accuracy",
        "unit": "suggestion",
        "checked_against": "the input",
        "question": "q",
        "pass": "p",
        "fail": "f",
        "examples": [{"verdict": "pass", "input": "i", "output": {"field_name": "x"}, "why": "w"}],
    }
    return {"version": "v", "instructions": "i", "criteria": [base | criterion]}


def test_a_criterion_without_examples_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Rubric.model_validate(_rubric(examples=[]))


def test_an_unknown_verdict_is_rejected() -> None:
    bad = [{"verdict": "partial", "input": "i", "output": {"field_name": "x"}, "why": "w"}]
    with pytest.raises(ValidationError):
        Rubric.model_validate(_rubric(examples=bad))


def test_an_output_criterion_needs_a_whole_output_example() -> None:
    with pytest.raises(ValidationError, match="output example"):
        Rubric.model_validate(_rubric(unit="output"))


def test_a_misspelled_key_is_rejected() -> None:
    """extra='forbid', so a typo such as 'questoin' cannot silently drop the question."""
    with pytest.raises(ValidationError):
        Rubric.model_validate(_rubric(questoin="q"))


def test_a_pass_without_evidence_does_not_count() -> None:
    assert JudgeAnswer(verdict="pass", evidence="the input says lake").passed
    assert not JudgeAnswer(verdict="pass", evidence="  ").passed
    assert not JudgeAnswer(verdict="fail", evidence="quote").passed
