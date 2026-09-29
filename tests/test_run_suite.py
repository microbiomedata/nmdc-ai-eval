"""Tests for credential-aware suite runner helpers."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from nmdc_ai_eval.llm_plugin_pnnl import PNNLChatModel, _load_pnnl_models
from nmdc_ai_eval.run_suite import _models_with_credentials, _pnnl_models_if_configured


@patch("nmdc_ai_eval.run_suite.llm")
def test_models_with_credentials_filters_models_missing_required_key(mock_llm: MagicMock) -> None:
    openai_model = MagicMock(needs_key="openai")
    vertex_model = MagicMock(needs_key=None)
    mock_llm.get_model.side_effect = [openai_model, vertex_model]
    mock_llm.get_key.return_value = None

    available, unavailable = _models_with_credentials(["gpt-4o-mini", "vertex/gemini-2.5-flash"])

    assert available == ["vertex/gemini-2.5-flash"]
    assert unavailable == [("gpt-4o-mini", "openai")]
    mock_llm.get_key.assert_called_once_with(key_alias="openai")


@patch("nmdc_ai_eval.run_suite.llm")
def test_models_with_credentials_keeps_model_with_configured_key(mock_llm: MagicMock) -> None:
    model = MagicMock(needs_key="gemini")
    mock_llm.get_model.return_value = model
    mock_llm.get_key.return_value = "configured-key"

    available, unavailable = _models_with_credentials(["gemini/gemini-2.5-flash"])

    assert available == ["gemini/gemini-2.5-flash"]
    assert unavailable == []


@patch("nmdc_ai_eval.run_suite.llm")
def test_pnnl_models_are_added_when_llm_key_is_configured(mock_llm: MagicMock) -> None:
    mock_llm.get_key.return_value = "configured-key"

    assert _pnnl_models_if_configured() == [f"pnnl/{name}" for name in _load_pnnl_models()]


@patch("nmdc_ai_eval.run_suite.llm")
def test_pnnl_models_are_not_added_without_llm_key(mock_llm: MagicMock) -> None:
    mock_llm.get_key.return_value = None

    assert _pnnl_models_if_configured() == []


def test_pnnl_model_omits_temperature() -> None:
    model = PNNLChatModel("pnnl/gpt-5-project", model_name="gpt-5-project", reasoning=True)
    prompt = SimpleNamespace(options=model.Options(temperature=0.0), schema=None, tools=[])

    assert "temperature" not in model.build_kwargs(prompt, stream=False)


def test_runner_config_has_empty_model_name_map() -> None:
    from llm_matrix.runner import LLMRunnerConfig

    config = LLMRunnerConfig(evaluation_model_name="pnnl/gpt-5-project", model_name_map={})

    assert config.model_name_map == {}
