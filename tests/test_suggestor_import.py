"""Proof-of-life: verify the suggestor pipeline is importable and introspectable.

These tests do NOT call any LLM — they verify that the eval repo can import
the production suggestor package, inspect its API, and prepare inputs in the
format it expects.
"""

import inspect

import pytest
from nmdc_metadata_suggestor_ai_tool.llm_client import (
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
    # Presence check, not exact match: the suggestor model gains fields over
    # time (e.g. the optional `id` added in 1.2.0 for sample-level suggestions),
    # and this proof-of-life test should not break on benign upstream additions.
    assert {"field_name", "reason", "value"} <= fields

    suggestion = MetadataFieldSuggestion(
        field_name="env_broad_scale",
        reason="The study describes a temperate forest soil",
        value="temperate broadleaf and mixed forest biome [ENVO:01000202]",
    )
    assert suggestion.field_name == "env_broad_scale"
    assert suggestion.value != ""


def test_llm_client_accepts_arbitrary_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """LLMClient passes model names through without enum/whitelist validation.

    Uses the ``pnnl`` access_provider rather than ``gcp`` so this test runs
    in CI without Google credentials. ``LLMClient.__init__`` for ``gcp``
    eagerly calls ``google.auth.default()`` which fails in credential-less
    environments; ``pnnl`` just reads env vars and constructs an ``OpenAI``
    client (no network call on init). The point being verified — that the
    model string is stored as-is — is provider-independent.
    """
    # The suggestor module may capture AI_INCUBATOR_KEY and BASE_URL at module
    # import time, so monkeypatch.setenv alone doesn't help once loaded.
    # Patch both the module-level constants and env vars for compatibility
    # with versions that read env vars inside __init__.
    from nmdc_metadata_suggestor_ai_tool import llm_client as llm_client_module

    monkeypatch.setenv("AI_INCUBATOR_KEY", "test-key")
    monkeypatch.setenv("AI_INCUBATOR_BASE_URL", "https://example.invalid")
    monkeypatch.setattr(llm_client_module, "AI_INCUBATOR_KEY", "test-key", raising=False)
    monkeypatch.setattr(llm_client_module, "BASE_URL", "https://example.invalid", raising=False)
    client = LLMClient(access_provider="pnnl", model="any-weird-name")
    assert client.model == "any-weird-name"

    # Verify patched configuration is actually wired into the instantiated
    # low-level client when those attributes are exposed by the implementation.
    inner_client = getattr(client, "client", None)
    if inner_client is not None:
        base_url = getattr(inner_client, "base_url", None)
        if base_url is not None:
            assert str(base_url).rstrip("/") == "https://example.invalid"

        api_key = getattr(inner_client, "api_key", None)
        if api_key is not None:
            assert api_key == "test-key"


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
