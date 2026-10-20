"""Tests for credential-aware suite runner helpers."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from click.testing import CliRunner

from nmdc_ai_eval.llm_plugin_pnnl import PNNLChatModel, _load_pnnl_models
from nmdc_ai_eval.run_suite import _models_with_credentials, _pnnl_models_if_configured, main


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


@pytest.mark.parametrize(
    ("judge_text", "expected_score"),
    [
        ("1 Correct", 1.0),
        ("Score: 1", 1.0),
        ("**Score: 0.5** partially right", 0.5),
        ("Looks correct to me", None),
        ("1.5 Too high", None),
    ],
)
def test_main_runs_models_and_writes_usage_and_score(tmp_path, judge_text: str, expected_score: float | None) -> None:
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        """name: test-suite
template: basic
templates:
  basic:
    system: "Be exact, {study_name}."
    prompt: "Answer: {input}"
    metrics: [simple_question]
models:
  gpt-4o-mini:
    parameters:
      temperature: 0.4
      max_tokens: 64
matrix:
  hyperparameters:
    model: [gpt-4o-mini]
    temperature: [0.0]
    top_p: [0.8, 0.9]
cases:
  - input: "What is 2+2?"
    ideal: "4"
    original_input:
      study_name: "Example study"
      sampleData: soil_data
"""
    )
    generation_response = MagicMock(spec=["text", "input_tokens", "output_tokens", "duration_ms"])
    generation_response.text.return_value = "4"
    generation_response.input_tokens = 10
    generation_response.output_tokens = 2
    generation_response.duration_ms.return_value = 100
    judge_response = MagicMock(spec=["text", "input_tokens", "output_tokens", "duration_ms"])
    judge_response.text.return_value = judge_text
    judge_response.input_tokens = 20
    judge_response.output_tokens = 1
    judge_response.duration_ms.return_value = 50
    generator_model = MagicMock(spec=["prompt"])
    generator_model.prompt.return_value = generation_response
    judge_model = MagicMock(spec=["prompt"])
    judge_model.prompt.return_value = judge_response

    with (
        patch("nmdc_ai_eval.run_suite._preflight", return_value=[]),
        patch("nmdc_ai_eval.run_suite._models_with_credentials", return_value=(["gpt-4o-mini"], [])),
        patch("nmdc_ai_eval.run_suite._pnnl_models_if_configured", return_value=[]),
        patch("nmdc_ai_eval.run_suite.llm.get_model", side_effect=[judge_model, generator_model]),
        patch("nmdc_ai_eval.run_suite.estimate_cost", return_value=0.001),
    ):
        result = CliRunner().invoke(main, [str(suite_path), "--output-dir", str(tmp_path / "out")])

    assert result.exit_code == 0, result.output
    rows = pd.read_csv(tmp_path / "out" / "results.tsv", sep="\t")
    assert rows.loc[0, "model"] == "gpt-4o-mini"
    assert len(rows) == 2
    assert rows["temperature"].tolist() == [0.4, 0.4]
    assert rows["top_p"].tolist() == [0.8, 0.9]
    assert rows.loc[0, "study_name"] == "Example study"
    assert rows.loc[0, "sampleData"] == "soil_data"
    assert str(rows.loc[0, "response_text"]) == "4"
    if expected_score is None:
        assert pd.isna(rows.loc[0, "score"])
    else:
        assert rows.loc[0, "score"] == expected_score
    assert rows.loc[0, "evaluation_message"] == judge_text
    assert rows.loc[0, "input_tokens"] == 30
    assert rows.loc[0, "output_tokens"] == 3
    assert rows.loc[0, "duration_ms"] == 150
    assert rows["est_cost_usd"].tolist() == [0.002, 0.002]
    assert generator_model.prompt.call_args_list[0].args == ("Answer: What is 2+2?",)
    assert generator_model.prompt.call_args_list[0].kwargs == {
        "system": "Be exact, Example study.",
        "temperature": 0.4,
        "max_tokens": 64,
        "top_p": 0.8,
    }
    assert judge_model.prompt.call_args_list[0].args == ("The expected answer is: 4. The output to score is: 4.",)
    assert "score between 0 and 1" in judge_model.prompt.call_args_list[0].kwargs["system"]


def test_main_rejects_suite_plugins_before_model_calls(tmp_path) -> None:
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        """name: test-suite
models:
  gpt-4o-mini:
    plugins: [citeseek]
matrix:
  hyperparameters:
    model: [gpt-4o-mini]
cases:
  - input: "question"
    ideal: "answer"
"""
    )
    with patch("nmdc_ai_eval.run_suite.llm.get_model") as get_model:
        result = CliRunner().invoke(main, [str(suite_path), "--output-dir", str(tmp_path / "out")])

    assert result.exit_code == 1
    assert "unsupported llm-matrix plugin" in result.output
    get_model.assert_not_called()


def test_main_aborts_without_writing_partial_results_on_failure(tmp_path) -> None:
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        """name: test-suite
template: basic
templates:
  basic:
    prompt: "{input}"
matrix:
  hyperparameters:
    model: [gpt-4o-mini]
cases:
  - input: "first"
    ideal: "a"
  - input: "second"
    ideal: "b"
"""
    )
    ok_response = MagicMock(spec=["text", "input_tokens", "output_tokens", "duration_ms"])
    ok_response.text.return_value = "a"
    ok_response.input_tokens = 1
    ok_response.output_tokens = 1
    ok_response.duration_ms.return_value = 1
    model = MagicMock(spec=["prompt"])
    model.prompt.side_effect = [ok_response, RuntimeError("boom")]

    with (
        patch("nmdc_ai_eval.run_suite._preflight", return_value=[]),
        patch("nmdc_ai_eval.run_suite._models_with_credentials", return_value=(["gpt-4o-mini"], [])),
        patch("nmdc_ai_eval.run_suite._pnnl_models_if_configured", return_value=[]),
        patch("nmdc_ai_eval.run_suite.llm.get_model", return_value=model),
    ):
        result = CliRunner().invoke(main, [str(suite_path), "--output-dir", str(tmp_path / "out")])

    assert result.exit_code != 0
    assert "boom" in result.output
    assert not (tmp_path / "out" / "results.tsv").exists()
