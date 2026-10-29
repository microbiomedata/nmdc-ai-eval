"""Tests for the authenticated submission LLM-judge evaluation helpers."""

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests

from nmdc_ai_eval.submission_judge import (
    SubmissionApiClient,
    build_judge_system_prompt,
    evaluate_submission,
    load_rubric,
    load_submission_ids,
    parse_judge_response,
    resolve_suggestor_provider,
)

RUBRIC = {
    "metrics": [
        {
            "name": "relevance",
            "description": "How relevant the suggestions are.",
            "score_anchors": {
                1: "Not relevant.",
                2: "Partly relevant.",
                3: "Mostly relevant.",
                4: "Highly relevant.",
            },
        },
        {
            "name": "completeness",
            "description": "How complete the suggestions are.",
            "score_anchors": {
                1: "Not complete.",
                2: "Partly complete.",
                3: "Mostly complete.",
                4: "Complete.",
            },
        },
    ]
}


def _response(payload: object) -> MagicMock:
    response = MagicMock(status_code=200)
    response.json.return_value = payload
    response.text = json.dumps(payload)
    return response


def test_fetch_submission_exchanges_refresh_token_then_uses_access_token() -> None:
    with patch(
        "nmdc_ai_eval.submission_judge.requests.request",
        side_effect=[
            _response({"access_token": "access-token"}),
            _response({"id": "nmdc:sub-1", "metadata_submission": {}}),
        ],
    ) as mocked_request:
        api = SubmissionApiClient("https://data.example.org/", "refresh-token")
        submission = api.fetch_submission("nmdc:sub-1")

    assert submission == {"id": "nmdc:sub-1", "metadata_submission": {}}
    refresh_request, fetch_request = mocked_request.call_args_list
    assert refresh_request.args == ("POST", "https://data.example.org/auth/refresh")
    assert refresh_request.kwargs["json"] == {"refresh_token": "refresh-token"}
    assert fetch_request.args == (
        "GET",
        "https://data.example.org/api/metadata_submission/nmdc%3Asub-1",
    )
    assert fetch_request.kwargs["headers"]["Authorization"].startswith("Bearer ")
    assert refresh_request.kwargs["timeout"] == 30.0
    assert mocked_request.call_count == 2


def test_fetch_submission_normalizes_direct_api_shape_for_suggestor() -> None:
    direct_submission = {
        "id": "submission-1",
        "study_form": {"studyName": "Test study", "description": "Soil samples"},
        "sample_sets": [
            {"templates": ["soil", "water"]},
            {"templates": ["soil"]},
        ],
    }
    with patch(
        "nmdc_ai_eval.submission_judge.requests.request",
        side_effect=[
            _response({"access_token": "access-token"}),
            _response(direct_submission),
        ],
    ):
        submission = SubmissionApiClient(
            "https://data.example.org",
            "refresh-token",
        ).fetch_submission("submission-1")

    assert submission["id"] == "submission-1"
    assert submission["metadata_submission"] == {
        "studyForm": {"studyName": "Test study", "description": "Soil samples"},
        "multiOmicsForm": {},
        "packageName": ["soil", "water"],
    }


def test_api_client_rejects_empty_refresh_token() -> None:
    with pytest.raises(ValueError, match="refresh token is required"):
        SubmissionApiClient("https://data.example.org", " ")


def test_api_client_rejects_non_https_base_urls() -> None:
    with pytest.raises(ValueError, match="absolute HTTPS"):
        SubmissionApiClient("file:///tmp/submissions", "refresh-token")


def test_api_client_surfaces_http_error_body_and_redacts_refresh_token() -> None:
    api = SubmissionApiClient("https://data.example.org", "refresh-token")
    response = MagicMock(status_code=403, text='{"detail":"Invalid refresh-token"}')
    error = requests.HTTPError("Denied", response=response)
    with (
        patch("nmdc_ai_eval.submission_judge.requests.request", side_effect=error),
        pytest.raises(RuntimeError, match=r'HTTP 403\. Response body: \{"detail":"Invalid \[REDACTED\]"\}'),
    ):
        api.fetch_submission("submission-1")


def test_api_client_truncates_long_http_error_body() -> None:
    response = MagicMock(status_code=403, text="x" * 2000)
    error = requests.HTTPError("Denied", response=response)
    with (
        patch("nmdc_ai_eval.submission_judge.requests.request", side_effect=error),
        pytest.raises(RuntimeError, match=r"\.\.\. \[truncated\]"),
    ):
        SubmissionApiClient("https://data.example.org", "refresh-token").fetch_submission("id")


def test_api_client_rejects_non_object_json_response() -> None:
    with patch("nmdc_ai_eval.submission_judge.requests.request", return_value=_response([])):
        api = SubmissionApiClient("https://data.example.org", "refresh-token")
        with pytest.raises(ValueError, match="not an object"):
            api.fetch_submission("submission-1")


def test_api_client_rejects_refresh_response_without_access_token() -> None:
    with patch("nmdc_ai_eval.submission_judge.requests.request", return_value=_response({})):
        api = SubmissionApiClient("https://data.example.org", "refresh-token")
        with pytest.raises(ValueError, match="did not contain an access token"):
            api.fetch_submission("submission-1")


def test_api_client_rejects_submission_without_metadata_document() -> None:
    with patch(
        "nmdc_ai_eval.submission_judge.requests.request",
        side_effect=[_response({"access_token": "access-token"}), _response({"id": "submission-1"})],
    ):
        api = SubmissionApiClient("https://data.example.org", "refresh-token")
        with pytest.raises(ValueError, match="metadata_submission object"):
            api.fetch_submission("submission-1")


def test_api_client_debug_logs_response_and_redacts_token_fields(capsys: pytest.CaptureFixture[str]) -> None:
    with patch(
        "nmdc_ai_eval.submission_judge.requests.request",
        return_value=_response({"access_token": "secret-access-token", "message": "ready"}),
    ):
        api = SubmissionApiClient("https://data.example.org", "secret-refresh-token", debug=True)
        api.refresh_access_token()

    output = capsys.readouterr().err
    assert "POST /auth/refresh returned HTTP 200" in output
    assert '"access_token": "[REDACTED]"' in output
    assert "secret-access-token" not in output
    assert "secret-refresh-token" not in output
    assert '"message": "ready"' in output


def test_api_client_debug_logs_response_shape_on_invalid_submission(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with patch(
        "nmdc_ai_eval.submission_judge.requests.request",
        side_effect=[
            _response({"access_token": "access-token"}),
            _response({"id": "submission-1", "error": "not found"}),
        ],
    ):
        api = SubmissionApiClient("https://data.example.org", "refresh-token", debug=True)
        with pytest.raises(ValueError, match="metadata_submission object"):
            api.fetch_submission("submission-1")

    output = capsys.readouterr().err
    assert '"error": "not found"' in output
    assert "top-level keys: error, id" in output


def test_api_client_reuses_access_token_until_near_expiry() -> None:
    with patch(
        "nmdc_ai_eval.submission_judge.requests.request",
        side_effect=[
            _response({"access_token": "access-token"}),
            _response({"id": "first", "metadata_submission": {}}),
            _response({"id": "second", "metadata_submission": {}}),
        ],
    ) as mocked_request:
        api = SubmissionApiClient("https://data.example.org", "refresh-token")
        api.fetch_submission("first")
        api.fetch_submission("second")

    assert mocked_request.call_count == 3
    assert mocked_request.call_args_list[1].kwargs["headers"]["Authorization"].startswith("Bearer ")
    assert mocked_request.call_args_list[2].kwargs["headers"]["Authorization"].startswith("Bearer ")


def test_api_client_refreshes_access_token_before_expiry() -> None:
    from nmdc_ai_eval import submission_judge

    with (
        patch(
            "nmdc_ai_eval.submission_judge.requests.request",
            side_effect=[
                _response({"access_token": "first-token"}),
                _response({"id": "first", "metadata_submission": {}}),
                _response({"access_token": "second-token"}),
                _response({"id": "second", "metadata_submission": {}}),
            ],
        ) as mocked_request,
        patch.object(
            submission_judge.time,
            "monotonic",
            side_effect=[100.0, 100.0 + 24 * 60 * 60 - 5 * 60, 200.0],
        ),
    ):
        api = SubmissionApiClient("https://data.example.org", "refresh-token")
        api.fetch_submission("first")
        api.fetch_submission("second")

    assert mocked_request.call_count == 4
    assert mocked_request.call_args_list[2].args == (
        "POST",
        "https://data.example.org/auth/refresh",
    )


def test_load_submission_ids_rejects_empty_and_duplicates(tmp_path) -> None:
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("\n  \n", encoding="utf-8")
    with pytest.raises(ValueError, match="no submission IDs"):
        load_submission_ids(str(ids_file))

    ids_file.write_text("same\nsame\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_submission_ids(str(ids_file))


def test_load_submission_ids_strips_surrounding_whitespace(tmp_path) -> None:
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("# comment\n\n  submission-1  \n\n submission-2\n", encoding="utf-8")
    assert load_submission_ids(str(ids_file)) == ["submission-1", "submission-2"]


def test_checked_in_expert_rubric_is_valid() -> None:
    rubric_path = Path(__file__).parents[1] / "datasets" / "submission-judge" / "judge-rubric.yaml"
    rubric = load_rubric(str(rubric_path))
    assert [metric["name"] for metric in rubric["metrics"]] == [
        "accuracy",
        "factuality",
        "relevancy",
        "completeness",
        "coherence",
    ]
    assert set(rubric["grading_scale"]) == {1, 2, 3, 4}
    assert rubric["metrics"][3]["null_when"] == "The suggestion is an abstention."


def test_load_rubric_rejects_missing_score_anchor(tmp_path) -> None:
    rubric_file = tmp_path / "rubric.yaml"
    rubric_file.write_text(
        """metrics:
  - name: quality
    description: response quality
    score_anchors:
      1: low
      2: medium
      3: high
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="anchors 1, 2, 3, and 4"):
        load_rubric(str(rubric_file))


@pytest.mark.parametrize(
    ("rubric_text", "message"),
    [
        ("metrics: []\n", "non-empty 'metrics' list"),
        ("metrics: [invalid]\n", "metric that is not an object"),
        (
            """metrics:
  - name: quality
    description: " "
    score_anchors: {1: low, 2: fair, 3: good, 4: great}
""",
            "must have a description",
        ),
        (
            """metrics:
  - name: quality
    description: response quality
    score_anchors: {1: " ", 2: fair, 3: good, 4: great}
""",
            "empty score anchor",
        ),
    ],
)
def test_load_rubric_rejects_incomplete_configuration(tmp_path, rubric_text: str, message: str) -> None:
    rubric_file = tmp_path / "rubric.yaml"
    rubric_file.write_text(rubric_text, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_rubric(str(rubric_file))


def test_load_rubric_rejects_duplicate_metric_names(tmp_path) -> None:
    rubric_file = tmp_path / "rubric.yaml"
    rubric_file.write_text(
        """metrics:
  - name: quality
    description: first
    score_anchors: {1: low, 2: fair, 3: good, 4: great}
  - name: quality
    description: second
    score_anchors: {1: low, 2: fair, 3: good, 4: great}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate metric name"):
        load_rubric(str(rubric_file))


def test_parse_judge_response_validates_and_averages_all_metrics() -> None:
    response = """```json
{"scores":[
  {"metric":"completeness","score":3,"rationale":"Most expected details are covered."},
  {"metric":"relevance","score":4,"rationale":"The suggestions use the study context."}
]}
```"""
    parsed = parse_judge_response(response, RUBRIC)
    assert parsed == {
        "scores": [
            {"metric": "relevance", "score": 4, "rationale": "The suggestions use the study context."},
            {"metric": "completeness", "score": 3, "rationale": "Most expected details are covered."},
        ],
        "mean_score": 3.5,
    }


def test_parse_judge_response_excludes_allowed_null_score_from_mean() -> None:
    rubric = load_rubric(str(Path(__file__).parents[1] / "datasets" / "submission-judge" / "judge-rubric.yaml"))
    response = """{"scores":[
      {"metric":"accuracy","score":4,"rationale":"Abstention is correct."},
      {"metric":"factuality","score":4,"rationale":"No unsupported claims."},
      {"metric":"relevancy","score":4,"rationale":"Slot is not applicable."},
      {"metric":"completeness","score":null,"rationale":"Not applicable for abstention."},
      {"metric":"coherence","score":4,"rationale":"Reasoning is consistent."}
    ]}"""
    parsed = parse_judge_response(response, rubric)
    assert parsed["scores"][3]["score"] is None
    assert parsed["mean_score"] == 4.0


def test_parse_judge_response_rejects_null_for_metric_without_null_condition() -> None:
    response = """{"scores":[
      {"metric":"relevance","score":null,"rationale":"Not allowed."},
      {"metric":"completeness","score":3,"rationale":"Mostly complete."}
    ]}"""
    with pytest.raises(ValueError, match="integer from 1 through 4"):
        parse_judge_response(response, RUBRIC)


def test_parse_judge_response_accepts_json_label_on_separate_fence_line() -> None:
    response = """```
json
{"scores":[
  {"metric":"relevance","score":4,"rationale":"Relevant."},
  {"metric":"completeness","score":3,"rationale":"Mostly complete."}
]}
```"""
    assert parse_judge_response(response, RUBRIC)["mean_score"] == 3.5


@pytest.mark.parametrize(
    ("judge_text", "message"),
    [
        ('{"scores":[]}', "omitted required metrics"),
        (
            '{"scores":[{"metric":"relevance","score":5,"rationale":"Invalid range."},'
            '{"metric":"completeness","score":4,"rationale":"Complete."}]}',
            "integer from 1 through 4",
        ),
        (
            '{"scores":[{"metric":"relevance","score":true,"rationale":"Invalid type."},'
            '{"metric":"completeness","score":4,"rationale":"Complete."}]}',
            "integer from 1 through 4",
        ),
    ],
)
def test_parse_judge_response_rejects_invalid_scores(judge_text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_judge_response(judge_text, RUBRIC)


@pytest.mark.parametrize(
    ("judge_text", "message"),
    [
        ("not json", "not valid JSON"),
        ("```json\n{}\n", "incomplete JSON code fence"),
        ("{}", "must contain a 'scores' list"),
        ('{"scores":[null]}', "not an object"),
        (
            '{"scores":[{"metric":"unknown","score":3,"rationale":"Unknown."}]}',
            "unknown metric",
        ),
        (
            '{"scores":[{"metric":"relevance","score":3,"rationale":"First."},'
            '{"metric":"relevance","score":4,"rationale":"Duplicate."},'
            '{"metric":"completeness","score":2,"rationale":"Second."}]}',
            "more than once",
        ),
        (
            '{"scores":[{"metric":"relevance","score":3,"rationale":"First."},'
            '{"metric":"completeness","score":2,"rationale":"  "}]}',
            "non-empty string",
        ),
    ],
)
def test_parse_judge_response_rejects_malformed_responses(judge_text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_judge_response(judge_text, RUBRIC)


def test_judge_prompt_contains_configured_rubric() -> None:
    prompt = build_judge_system_prompt(RUBRIC)
    assert '"name": "relevance"' in prompt
    assert "integers from 1 to 4" in prompt


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        ({"VERTEX_PROJECT_ID": "project"}, "gcp"),
        ({"AI_INCUBATOR_KEY": "key", "AI_INCUBATOR_BASE_URL": "url"}, "pnnl"),
        ({"CBORG_API_KEY": "key", "CBORG_BASE_URL": "url"}, "cborg"),
    ],
)
def test_resolve_suggestor_provider_auto_detects_single_provider(
    environment: dict[str, str],
    expected: str,
) -> None:
    assert resolve_suggestor_provider("auto", environment) == expected


def test_resolve_suggestor_provider_requires_explicit_choice_when_ambiguous() -> None:
    with pytest.raises(ValueError, match="Multiple suggestor providers"):
        resolve_suggestor_provider(
            "auto",
            {
                "VERTEX_PROJECT_ID": "project",
                "AI_INCUBATOR_KEY": "key",
                "AI_INCUBATOR_BASE_URL": "url",
            },
        )


def test_resolve_suggestor_provider_requires_configuration() -> None:
    with pytest.raises(ValueError, match="Could not auto-detect"):
        resolve_suggestor_provider("auto", {})


def test_resolve_suggestor_provider_accepts_explicit_provider() -> None:
    assert resolve_suggestor_provider("cborg", {}) == "cborg"


def test_cborg_client_uses_repository_api_key_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nmdc_ai_eval import submission_judge

    monkeypatch.delenv("CBORG_KEY", raising=False)
    monkeypatch.setenv("CBORG_API_KEY", "test-key")
    client = object()

    def make_client(**_kwargs: Any) -> object:
        assert os.environ["CBORG_KEY"] == "test-key"
        return client

    with patch.object(submission_judge, "LLMClient", side_effect=make_client) as client_type:
        assert submission_judge._create_suggestor_client("cborg", "test-model") is client

    client_type.assert_called_once_with(access_provider="cborg", model="test-model")
    assert "CBORG_KEY" not in os.environ


def test_evaluate_submission_runs_production_pipeline_and_judge() -> None:
    suggestion_data = {"metadata_fields": [{"field_name": "env_medium", "value": "soil"}]}
    pipeline_output = SimpleNamespace(model_dump=lambda mode: suggestion_data)
    judge_response = SimpleNamespace(
        text=lambda: (
            '{"scores":[{"metric":"relevance","score":4,"rationale":"Relevant."},'
            '{"metric":"completeness","score":3,"rationale":"Mostly complete."}]}'
        )
    )
    judge_model = MagicMock()
    judge_model.prompt.return_value = judge_response
    suggestor_client = SimpleNamespace(access_provider="pnnl", model="test-model")
    client_type = MagicMock(return_value=suggestor_client)
    submission = {"metadata_submission": {"studyForm": {"studyName": "Test"}}}
    parsed_submission = {"study_name": "Test"}

    def run_pipeline_with_context(**kwargs: Any) -> Any:
        from nmdc_metadata_suggestor_ai_tool.llm_client import ConversationManager

        manager = ConversationManager(llm_client=kwargs["llm_client"], system_prompt="test")
        manager.add_message("Submission description: Test")
        manager.add_message("Retrieved publication context: Test abstract")
        return pipeline_output

    with (
        patch(
            "nmdc_ai_eval.submission_judge.get_submission_fields",
            return_value=parsed_submission,
        ),
        patch(
            "nmdc_ai_eval.submission_judge.run_recommendation_pipeline",
            side_effect=run_pipeline_with_context,
        ) as pipeline,
        patch("nmdc_ai_eval.submission_judge.LLMClient", client_type),
    ):
        result = evaluate_submission(
            submission_id="submission-1",
            submission=submission,
            suggestor_provider="pnnl",
            suggestor_model="test-model",
            judge_model=judge_model,
            judge_model_name="gpt-4o-mini",
            rubric=RUBRIC,
        )

    client_type.assert_called_once_with(access_provider="pnnl", model="test-model")
    pipeline.assert_called_once_with(submission_object=submission, llm_client=suggestor_client)
    judge_input = json.loads(judge_model.prompt.call_args.args[0])
    assert judge_input == {
        "submission": parsed_submission,
        "suggestor_context_messages": [
            "Submission description: Test",
            "Retrieved publication context: Test abstract",
        ],
        "suggestor_response": suggestion_data,
    }
    assert result["submission_id"] == "submission-1"
    assert result["mean_score"] == 3.5
    assert result["suggestor_provider"] == "pnnl"
    assert result["suggestor_model"] == "test-model"


def test_runner_creates_a_distinct_judge_model_for_each_submission(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner_path = Path(__file__).parents[1] / "datasets" / "submission-judge" / "run_eval.py"
    spec = importlib.util.spec_from_file_location("submission_judge_runner", runner_path)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)

    first_judge = object()
    second_judge = object()
    api = MagicMock()
    api.fetch_submission.side_effect = [
        {"metadata_submission": {"id": "first"}},
        {"metadata_submission": {"id": "second"}},
    ]
    evaluate = MagicMock(
        side_effect=[
            {"submission_id": "first", "mean_score": 4},
            {"submission_id": "second", "mean_score": 3},
        ]
    )
    monkeypatch.setattr(runner, "load_dotenv", lambda: None)
    monkeypatch.setattr(runner, "load_submission_ids", lambda _path: ["first", "second"])
    monkeypatch.setattr(runner, "load_rubric", lambda _path: RUBRIC)
    monkeypatch.setattr(runner, "SubmissionApiClient", lambda **_kwargs: api)
    monkeypatch.setattr(runner, "evaluate_submission", evaluate)
    monkeypatch.setattr(runner.llm, "get_model", MagicMock(side_effect=[first_judge, second_judge]))
    monkeypatch.setenv("REFRESH_TOKEN", "test-refresh-token")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_eval.py",
            "--suggestor-provider",
            "pnnl",
            "--output-dir",
            str(tmp_path),
        ],
    )

    runner.main()

    assert runner.llm.get_model.call_args_list == [
        (("gpt-4o-mini",),),
        (("gpt-4o-mini",),),
    ]
    assert evaluate.call_args_list[0].kwargs["judge_model"] is first_judge
    assert evaluate.call_args_list[1].kwargs["judge_model"] is second_judge
    assert evaluate.call_args_list[0].kwargs["suggestor_provider"] == "pnnl"
    assert evaluate.call_args_list[0].kwargs["suggestor_model"] is None
