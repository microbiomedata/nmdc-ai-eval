"""Pydantic models for LLM judge structured output."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class Scores(BaseModel):
    accuracy: int = Field(..., ge=1, le=4)
    factuality: int = Field(..., ge=1, le=4)
    relevancy: int = Field(..., ge=1, le=4)
    completeness: Optional[int] = Field(None, ge=1, le=4)
    coherence: int = Field(..., ge=1, le=4)


class Suggestion(BaseModel):
    index: int
    sample_id: Optional[str] = None
    field_name: str
    suggestion_type: Literal["new_value", "validated_existing", "corrected_existing", "abstained"]
    evidence_check: str = Field(..., description="verified / not found / unverifiable claims")
    reasoning: str = Field(..., description="Concise justification covering all five metrics")
    scores: Scores
    critical_flags: list[str] = Field(default_factory=list)
    suggested_correction: Optional[str] = Field(None, description="Better value, 'abstain', or null")


class MissedSupportedSlot(BaseModel):
    field_name: str
    sample_id: Optional[str] = None
    evidence: str = Field(..., description="Quote + source")


class JudgeOutput(BaseModel):
    suggestions: list[Suggestion]
    missed_supported_slots: list[MissedSupportedSlot] = Field(default_factory=list)
    input_data_issues: list[str] = Field(default_factory=list)
    summary: str = Field(..., description="2-4 sentences on quality and main failure patterns")
