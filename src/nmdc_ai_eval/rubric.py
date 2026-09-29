"""Schema for judge rubrics in rubrics/, and for the verdict a judge returns.

A rubric is data, not code, so any harness can read it. See docs/judge-rubric.md for why.
"""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

RUBRICS_DIR = Path(__file__).resolve().parents[2] / "rubrics"

Verdict = Literal["pass", "fail"]


class Example(BaseModel):
    """One labelled anchor: what a pass or a fail looks like for this criterion."""

    model_config = ConfigDict(extra="forbid")

    verdict: Verdict
    input: str = Field(min_length=1)
    output: dict[str, Any]
    why: str = Field(min_length=1)


class Criterion(BaseModel):
    """One yes/no question the judge answers, per suggestion or once per output."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str = Field(pattern=r"^[a-z][a-z_]*$")
    unit: Literal["suggestion", "output"]
    checked_against: str = Field(min_length=1)
    question: str = Field(min_length=1)
    pass_: str = Field(alias="pass", min_length=1)
    fail: str = Field(min_length=1)
    examples: list[Example] = Field(min_length=1, max_length=2)

    @model_validator(mode="after")
    def _example_shape_matches_unit(self) -> "Criterion":
        # A per-suggestion example holds one suggestion; a per-output example holds the list.
        for example in self.examples:
            has_list = "metadata_fields" in example.output
            if (self.unit == "output") != has_list:
                raise ValueError(f"{self.id}: a {self.unit} example needs output of that shape")
        return self


class Rubric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str = Field(min_length=1)
    instructions: str = Field(min_length=1)
    criteria: list[Criterion] = Field(min_length=1)

    @field_validator("criteria")
    @classmethod
    def _unique_ids(cls, criteria: list[Criterion]) -> list[Criterion]:
        ids = [c.id for c in criteria]
        if len(ids) != len(set(ids)):
            raise ValueError(f"criterion ids repeat: {ids}")
        return criteria


class JudgeAnswer(BaseModel):
    """What the judge returns for one question. A pass with no evidence counts as a fail."""

    verdict: Verdict
    evidence: str = ""

    @property
    def passed(self) -> bool:
        return self.verdict == "pass" and bool(self.evidence.strip())


def load_rubric(path: Path) -> Rubric:
    return Rubric.model_validate(yaml.safe_load(path.read_text()))
