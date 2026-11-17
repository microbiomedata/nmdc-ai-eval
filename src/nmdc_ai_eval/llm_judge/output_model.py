"""Pydantic models for LLM judge structured output."""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator


class Scores(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accuracy: int = Field(..., ge=1, le=4, strict=True)
    factuality: int = Field(..., ge=1, le=4, strict=True)
    relevancy: int = Field(..., ge=1, le=4, strict=True)
    completeness: Optional[int] = Field(..., ge=1, le=4, strict=True)
    coherence: int = Field(..., ge=1, le=4, strict=True)


class Suggestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int = Field(..., ge=0, strict=True)
    sample_id: Optional[str] = None
    field_name: str = Field(..., min_length=1)
    suggestion_type: Literal["new_value", "validated_existing", "corrected_existing", "abstained"]
    evidence_check: Literal["verified", "not found", "unverifiable"] = Field(
        ...,
        description="Evidence status; use exactly one of these string values.",
    )
    reasoning: str = Field(..., description="Concise justification covering all five metrics")
    scores: Scores
    critical_flags: list[str] = Field(default_factory=list)
    suggested_correction: Optional[str] = Field(None, description="Better value, 'abstain', or null")

    @model_validator(mode="after")
    def validate_completeness(self) -> "Suggestion":
        if (self.scores.completeness is None) != (self.suggestion_type == "abstained"):
            raise ValueError("Judge completeness must be null only for an abstained suggestion.")
        return self


class MissedSupportedSlot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field_name: str = Field(..., min_length=1)
    sample_id: Optional[str] = None
    evidence: str = Field(..., description="Quote + source")


class JudgeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    suggestions: list[Suggestion]
    missed_supported_slots: list[MissedSupportedSlot] = Field(default_factory=list)
    input_data_issues: list[str] = Field(
        default_factory=list,
        description="Plain-text issue descriptions; do not return objects.",
    )
    summary: str = Field(..., description="2-4 sentences on quality and main failure patterns")

    @model_validator(mode="after")
    def validate_suggestions(self, info: ValidationInfo) -> "JudgeOutput":
        seen_indexes: set[int] = set()
        for suggestion in self.suggestions:
            if suggestion.index in seen_indexes:
                raise ValueError(f"Judge response contains suggestion index {suggestion.index} more than once.")
            seen_indexes.add(suggestion.index)

        context = info.context
        allowed_flags = context.get("allowed_flags") if isinstance(context, dict) else None
        if allowed_flags is not None:
            for suggestion in self.suggestions:
                unexpected = sorted(set(suggestion.critical_flags) - allowed_flags)
                if unexpected:
                    raise ValueError(f"Judge response contains unsupported critical flags: {unexpected}.")
        return self
