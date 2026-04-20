"""Tests for the Vertex AI llm plugin (llm_plugin_vertex.py).

These tests verify model registration and configuration — they do NOT
make real Vertex API calls (those are integration tests that require SA
credentials and are excluded from CI).
"""

import llm
import pytest


def test_vertex_gemini_models_registered() -> None:
    """All expected vertex/gemini-* models appear in llm model list."""
    model_ids = {m.model_id for m in llm.get_models()}
    assert "vertex/gemini-2.5-flash" in model_ids
    assert "vertex/gemini-2.5-pro" in model_ids
    assert "vertex/gemini-2.0-flash" in model_ids


def test_vertex_claude_models_registered() -> None:
    """All expected vertex/claude-* models appear in llm model list."""
    model_ids = {m.model_id for m in llm.get_models()}
    assert "vertex/claude-haiku-4-5" in model_ids
    assert "vertex/claude-sonnet-4-5@20250929" in model_ids


def test_vertex_model_ids_have_vertex_prefix() -> None:
    """No vertex model is accidentally registered without the prefix."""
    from nmdc_ai_eval.llm_plugin_vertex import _CLAUDE_MODELS, _GEMINI_MODELS, VertexClaudeModel, VertexGeminiModel

    for name in _GEMINI_MODELS:
        m = VertexGeminiModel(f"vertex/{name}", name)
        assert m.model_id.startswith("vertex/")
        assert m.vertex_model_name == name

    for name in _CLAUDE_MODELS:
        m = VertexClaudeModel(f"vertex/{name}", name)
        assert m.model_id.startswith("vertex/")
        assert m.vertex_model_name == name


def test_vertex_models_retrievable_via_llm_get_model() -> None:
    """llm.get_model() resolves vertex/ model IDs without error."""
    m = llm.get_model("vertex/gemini-2.5-flash")
    assert m.model_id == "vertex/gemini-2.5-flash"

    m2 = llm.get_model("vertex/claude-haiku-4-5")
    assert m2.model_id == "vertex/claude-haiku-4-5"


def test_vertex_project_id_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """_get_vertex_project() raises ValueError when VERTEX_PROJECT_ID unset."""
    import nmdc_ai_eval.llm_plugin_vertex as plugin
    from nmdc_ai_eval.llm_plugin_vertex import _get_vertex_project

    # _VERTEX_PROJECT_ID is captured at module import time; patch the module-level var.
    monkeypatch.setattr(plugin, "_VERTEX_PROJECT_ID", None)
    with pytest.raises(ValueError, match="VERTEX_PROJECT_ID"):
        _get_vertex_project()


def test_vertex_unknown_model_raises() -> None:
    """llm.get_model() raises UnknownModelError for unregistered vertex names."""
    with pytest.raises(llm.UnknownModelError):
        llm.get_model("vertex/gpt-4o")  # OpenAI is not on Vertex Model Garden


def test_vertex_options_accept_temperature() -> None:
    """Vertex models accept temperature=0.0 without pydantic rejecting it.

    Regression: llm-matrix passes temperature through Options; earlier
    the Options class inherited llm.Options with no temperature field
    and raised `extra_forbidden`, aborting every eval call.
    """
    gemini = llm.get_model("vertex/gemini-2.5-flash")
    claude = llm.get_model("vertex/claude-haiku-4-5")
    # Each model's Options class must parse temperature without error.
    assert gemini.Options(temperature=0.0).temperature == 0.0
    assert claude.Options(temperature=0.7).temperature == 0.7
    assert gemini.Options(max_tokens=100).max_tokens == 100
