"""Proof-of-life: verify the suggestor pipeline is importable and introspectable.

These tests do NOT call any LLM — they verify that the eval repo can import
the production suggestor package, inspect its API, and prepare inputs in the
format it expects.
"""

import inspect

import pytest
from nmdc_metadata_suggestor_ai_tool.llm_client import (
    DEFAULT_GEMINI_MODEL,
    GEMINI_MODELS,
    PNNL_GPT_MODELS,
    LLMClient,
)
from nmdc_metadata_suggestor_ai_tool.models.llm_output import (
    LLMOutput,
    MetadataFieldSuggestion,
)
from nmdc_metadata_suggestor_ai_tool.recommendation_pipeline import (
    run_recommendation_pipeline,
)
from nmdc_metadata_suggestor_ai_tool.schema_context import SchemaContextBuilder


def test_pipeline_signature() -> None:
    """Pipeline takes a submission_object dict and returns LLMOutput."""
    sig = inspect.signature(run_recommendation_pipeline)
    params = list(sig.parameters.keys())
    assert params[0] == "submission_object"
    assert params[1] == "llm_client"
    assert sig.return_annotation is LLMOutput


def test_llm_output_model() -> None:
    """LLMOutput has the expected fields for eval scoring."""
    fields = set(LLMOutput.model_fields.keys())
    assert "metadata_fields" in fields
    assert "model" in fields
    assert "access_provider" in fields


def test_metadata_field_suggestion_model() -> None:
    """MetadataFieldSuggestion has field_name, reason, and value."""
    fields = set(MetadataFieldSuggestion.model_fields.keys())
    assert fields == {"field_name", "reason", "value"}

    suggestion = MetadataFieldSuggestion(
        field_name="env_broad_scale",
        reason="The study describes a temperate forest soil",
        value="temperate broadleaf and mixed forest biome [ENVO:01000202]",
    )
    assert suggestion.field_name == "env_broad_scale"
    assert suggestion.value != ""


def test_gemini_models_available() -> None:
    """At least one Gemini model is configured."""
    assert len(GEMINI_MODELS) > 0
    assert DEFAULT_GEMINI_MODEL in GEMINI_MODELS


def test_pnnl_models_available() -> None:
    """PNNL model list is populated."""
    assert len(PNNL_GPT_MODELS) > 0


def test_schema_context_builder_exists() -> None:
    """SchemaContextBuilder can be instantiated (no LLM calls)."""
    builder = SchemaContextBuilder()
    assert hasattr(builder, "format_multi_interface_context")


def test_llm_client_rejects_unknown_provider() -> None:
    """LLMClient raises on unknown access_provider."""
    with pytest.raises(ValueError, match="Unknown access_provider"):
        LLMClient(access_provider="nonexistent")


# -- Submission object format --


MINIMAL_SUBMISSION_OBJECT: dict = {
    "metadata_submission": {
        "studyForm": {
            "studyName": "Test study",
            "description": "A test submission for eval",
            "notes": "No real data",
        },
        "templates": ["soil_data"],
        "packageName": "soil_data",
    }
}


def test_minimal_submission_object_shape() -> None:
    """Document the minimal submission object the pipeline expects.

    We don't call the pipeline (needs LLM credentials), but we verify
    the structure matches what get_submission_fields will parse.
    """
    from nmdc_metadata_suggestor_ai_tool.recommendation_pipeline import get_submission_fields

    parsed = get_submission_fields(submission_object=MINIMAL_SUBMISSION_OBJECT)
    assert "study_name" in parsed
    assert parsed["study_name"] == "Test study"
    assert "description" in parsed
    assert parsed["description"] == "A test submission for eval"
