"""Pydantic models and loading helpers for evaluation suite YAML files."""

from pathlib import Path
from typing import Any, TextIO

import yaml
from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Template(StrictModel):
    system: str | None = None
    prompt: str | None = None
    metrics: list[str] | None = None


class TestCase(StrictModel):
    input: str
    original_input: dict[str, Any] | None = None
    ideal: str | None = None
    template: str | None = None
    tags: list[str] | None = None
    comments: list[str] | None = None


class Matrix(StrictModel):
    hyperparameters: dict[str, list[Any]]
    exclude: "Matrix | None" = None


class ModelInfo(StrictModel):
    plugins: list[str] | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class Suite(StrictModel):
    name: str
    version: str | None = None
    description: str | None = None
    comments: list[str] | None = None
    models: dict[str, ModelInfo] = Field(default_factory=dict)
    matrix: Matrix
    cases: list[TestCase]
    template: str | None = None
    templates: dict[str, Template] | None = None

    @classmethod
    def load(cls, source: str | Path | TextIO) -> "Suite":
        if isinstance(source, (str, Path)):
            with Path(source).open() as suite_file:
                data = yaml.safe_load(suite_file)
        else:
            data = yaml.safe_load(source)
        return cls.model_validate(data)
