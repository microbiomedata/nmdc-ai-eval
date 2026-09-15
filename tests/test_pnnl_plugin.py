"""Tests for the PNNL AI Incubator llm plugin.

These tests verify model registration and configuration — they do NOT make
real PNNL API calls.
"""

from types import SimpleNamespace

import pytest

from nmdc_ai_eval.llm_plugin_pnnl import PNNLChatModel, _load_pnnl_models, register_models


def test_pnnl_models_registered_when_base_url_is_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every configured model is registered with a pnnl/ prefix."""
    monkeypatch.setenv("AI_INCUBATOR_BASE_URL", "https://example.invalid")
    registered: list[PNNLChatModel] = []

    register_models(registered.append)

    assert [model.model_id for model in registered] == [f"pnnl/{name}" for name in _load_pnnl_models()]


def test_pnnl_models_are_not_registered_without_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """The plugin does not register aliases without an endpoint URL."""
    monkeypatch.delenv("AI_INCUBATOR_BASE_URL", raising=False)
    registered: list[PNNLChatModel] = []

    register_models(registered.append)

    assert registered == []


def test_pnnl_model_configuration() -> None:
    """Registered models preserve the bare model name and require the PNNL key."""
    model = PNNLChatModel(
        "pnnl/gpt-5-project",
        model_name="gpt-5-project",
        api_base="https://example.invalid",
        reasoning=True,
    )

    assert model.model_id == "pnnl/gpt-5-project"
    assert model.model_name == "gpt-5-project"
    assert model.needs_key == "pnnl"


def test_pnnl_model_omits_temperature() -> None:
    """PNNL's GPT-5-compatible endpoint does not receive temperature."""
    model = PNNLChatModel("pnnl/gpt-5-project", model_name="gpt-5-project", reasoning=True)
    prompt = SimpleNamespace(options=model.Options(temperature=0.0), schema=None, tools=[])

    assert "temperature" not in model.build_kwargs(prompt, stream=False)
