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

from nmdc_ai_eval.llm_judge import (
    SubmissionApiClient,
    _submission_sample_records,
    build_judge_json_schema,
    build_judge_system_prompt,
    evaluate_submission,
    load_rubric,
    load_submission_ids,
    parse_judge_response,
    resolve_suggestor_provider,
)
from nmdc_ai_eval.llm_judge.output_model import JudgeOutput


def test_submission_sample_records_match_data_suffix_and_assign_row_ids() -> None:
    submission = {
        "metadata_submission": {
            "packageName": ["soil"],
            "sampleData": {
                "data": {
                    "soil_data": [
                        {"samp_name": "sample-a", "depth": "0-10 cm"},
                        {"id": "sample-b", "samp_name": "sample-b"},
                    ]
                }
            },
        }
    }

    assert _submission_sample_records(submission, "soil") == [
        {"id": "0", "samp_name": "sample-a", "depth": "0-10 cm"},
        {"id": "sample-b", "samp_name": "sample-b"},
    ]


RUBRIC = {
    "metrics": [
        {
            "name": "accuracy",
            "description": "Whether the value is correct and supported.",
            "score_anchors": {
                1: "Wrong.",
                2: "Partly wrong.",
                3: "Mostly correct.",
                4: "Correct.",
            },
        },
        {
            "name": "factuality",
            "description": "Whether claims are factually supported.",
            "score_anchors": {1: "Fabricated.", 2: "Unclear.", 3: "Mostly supported.", 4: "Supported."},
        },
        {
            "name": "relevancy",
            "description": "Whether the slot is relevant.",
            "score_anchors": {1: "Irrelevant.", 2: "Weak.", 3: "Mostly relevant.", 4: "Relevant."},
        },
        {
            "name": "completeness",
            "description": "Whether supported information is captured.",
            "null_when": "The suggestion is an abstention.",
            "score_anchors": {1: "Incomplete.", 2: "Partly complete.", 3: "Mostly complete.", 4: "Complete."},
        },
        {
            "name": "coherence",
            "description": "Whether the reasoning supports the value.",
            "score_anchors": {1: "Incoherent.", 2: "Weak.", 3: "Mostly coherent.", 4: "Coherent."},
        },
    ],
    "critical_flags": [
        "confident_wrong_value",
        "fabricated_evidence",
        "fabricated_ontology",
        "value_reason_mismatch",
        "unflagged_input_conflict",
        "invalid_inference",
    ],
}


def _judge_suggestion(
    index: int = 0,
    sample_id: str | None = None,
    field_name: str = "env_medium",
    suggestion_type: str = "new_value",
    completeness: int | None = 3,
) -> dict[str, Any]:
    return {
        "index": index,
        "sample_id": sample_id,
        "field_name": field_name,
        "suggestion_type": suggestion_type,
        "evidence_check": "verified",
        "reasoning": "The value is supported by the provided sample record.",
        "scores": {
            "accuracy": 4,
            "factuality": 4,
            "relevancy": 4,
            "completeness": completeness,
            "coherence": 4,
        },
        "critical_flags": [],
        "suggested_correction": None,
    }


def _judge_payload(*suggestions: dict[str, Any]) -> dict[str, Any]:
    return {
        "suggestions": list(suggestions) if suggestions else [_judge_suggestion()],
        "missed_supported_slots": [],
        "input_data_issues": [],
        "summary": "The suggestion is well supported. No major failure pattern was identified.",
    }


def _response(payload: object) -> MagicMock:
    response = MagicMock(status_code=200)
    response.json.return_value = payload
    response.text = json.dumps(payload)
    return response


def test_fetch_submission_exchanges_refresh_token_then_uses_access_token() -> None:
    with patch(
        "nmdc_ai_eval.llm_judge.submission_api.requests.request",
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
        "nmdc_ai_eval.llm_judge.submission_api.requests.request",
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


def test_fetch_submission_collects_sample_set_data() -> None:
    direct_submission = {
        "study_form": {"studyName": "Test study"},
        "sample_sets": [{"id": "set/1", "templates": ["soil"]}],
    }
    sample_set = {"sample_data": {"data": {"soil_data": [{"samp_name": "sample-a"}]}}}
    with patch(
        "nmdc_ai_eval.llm_judge.submission_api.requests.request",
        side_effect=[
            _response({"access_token": "access-token"}),
            _response(direct_submission),
            _response(sample_set),
        ],
    ) as mocked_request:
        submission = SubmissionApiClient(
            "https://data.example.org",
            "refresh-token",
        ).fetch_submission("submission-1")

    assert submission["metadata_submission"]["sampleData"] == {
        "soil_data": [{"samp_name": "sample-a"}],
    }
    sample_set_request = mocked_request.call_args_list[2]
    assert sample_set_request.args == (
        "GET",
        "https://data.example.org/api/metadata_submission/sample_set/set%2F1",
    )
    assert sample_set_request.kwargs["headers"]["Authorization"].startswith("Bearer ")


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
        patch("nmdc_ai_eval.llm_judge.submission_api.requests.request", side_effect=error),
        pytest.raises(RuntimeError, match=r'HTTP 403\. Response body: \{"detail":"Invalid \[REDACTED\]"\}'),
    ):
        api.fetch_submission("submission-1")


def test_api_client_truncates_long_http_error_body() -> None:
    response = MagicMock(status_code=403, text="x" * 2000)
    error = requests.HTTPError("Denied", response=response)
    with (
        patch("nmdc_ai_eval.llm_judge.submission_api.requests.request", side_effect=error),
        pytest.raises(RuntimeError, match=r"\.\.\. \[truncated\]"),
    ):
        SubmissionApiClient("https://data.example.org", "refresh-token").fetch_submission("id")


def test_api_client_rejects_refresh_response_without_access_token() -> None:
    with patch("nmdc_ai_eval.llm_judge.submission_api.requests.request", return_value=_response({})):
        api = SubmissionApiClient("https://data.example.org", "refresh-token")
        with pytest.raises(ValueError, match="did not contain an access token"):
            api.fetch_submission("submission-1")


def test_api_client_rejects_submission_without_metadata_document() -> None:
    with patch(
        "nmdc_ai_eval.llm_judge.submission_api.requests.request",
        side_effect=[_response({"access_token": "access-token"}), _response({"id": "submission-1"})],
    ):
        api = SubmissionApiClient("https://data.example.org", "refresh-token")
        with pytest.raises(ValueError, match="metadata_submission object"):
            api.fetch_submission("submission-1")


def test_api_client_debug_logs_response_and_redacts_token_fields(capsys: pytest.CaptureFixture[str]) -> None:
    with patch(
        "nmdc_ai_eval.llm_judge.submission_api.requests.request",
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
        "nmdc_ai_eval.llm_judge.submission_api.requests.request",
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
        "nmdc_ai_eval.llm_judge.submission_api.requests.request",
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
    from nmdc_ai_eval.llm_judge import submission_api

    with (
        patch(
            "nmdc_ai_eval.llm_judge.submission_api.requests.request",
            side_effect=[
                _response({"access_token": "first-token"}),
                _response({"id": "first", "metadata_submission": {}}),
                _response({"access_token": "second-token"}),
                _response({"id": "second", "metadata_submission": {}}),
            ],
        ) as mocked_request,
        patch.object(
            submission_api.time,
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
    rubric_path = Path(__file__).parents[1] / "datasets" / "llm-judge" / "judge-rubric.yaml"
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


def test_load_rubric_rejects_metrics_not_supported_by_output_model(tmp_path) -> None:
    rubric_file = tmp_path / "rubric.yaml"
    rubric_file.write_text(
        """metrics:
  - name: quality
    description: Overall quality.
    score_anchors: {1: poor, 2: fair, 3: good, 4: great}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must match the fixed JudgeOutput model"):
        load_rubric(str(rubric_file))


def test_parse_judge_response_validates_pi_output_and_averages_suggestion_scores() -> None:
    payload = _judge_payload(
        _judge_suggestion(index=0, field_name="env_medium", completeness=3),
        _judge_suggestion(index=1, field_name="env_local_scale", completeness=2),
    )
    response = json.dumps(payload)

    parsed = parse_judge_response(response, RUBRIC)

    assert parsed["suggestions"] == payload["suggestions"]
    assert parsed["summary"] == payload["summary"]
    assert parsed["mean_score"] == 3.7


def test_parse_judge_response_excludes_abstention_completeness_from_mean() -> None:
    payload = _judge_payload(_judge_suggestion(suggestion_type="abstained", completeness=None))

    parsed = parse_judge_response(json.dumps(payload), RUBRIC)

    assert parsed["suggestions"][0]["scores"]["completeness"] is None
    assert parsed["mean_score"] == 4.0


def test_parse_judge_response_rejects_malformed_json() -> None:
    payload = json.dumps(_judge_payload())
    payload = payload[:-1] + ",}"

    with pytest.raises(ValueError, match="not valid JSON"):
        parse_judge_response(payload, RUBRIC)


def test_parse_judge_response_rejects_boolean_evidence_check() -> None:
    suggestion = _judge_suggestion()
    suggestion["evidence_check"] = True

    with pytest.raises(ValueError, match="output model"):
        parse_judge_response(json.dumps(_judge_payload(suggestion)), RUBRIC)


def test_parse_judge_response_rejects_object_input_data_issue() -> None:
    payload = _judge_payload()
    payload["input_data_issues"] = [{"field_name": "lat_lon", "issue": "Invalid coordinates"}]

    with pytest.raises(ValueError, match="output model"):
        parse_judge_response(json.dumps(payload), RUBRIC)


def test_parse_judge_response_rejects_incomplete_or_unexpected_suggestion_indexes() -> None:
    payload = _judge_payload(_judge_suggestion(index=1))
    expected = [{"field_name": "env_medium", "id": None}, {"field_name": "env_local_scale", "id": None}]

    with pytest.raises(ValueError, match="coverage mismatch"):
        parse_judge_response(json.dumps(payload), RUBRIC, expected_suggestions=expected)


def test_parse_judge_response_rejects_duplicate_suggestion_indexes() -> None:
    payload = _judge_payload(_judge_suggestion(), _judge_suggestion())

    with pytest.raises(ValueError, match="more than once"):
        parse_judge_response(json.dumps(payload), RUBRIC)


def test_parse_judge_response_rejects_mismatched_suggestion_identity() -> None:
    payload = _judge_payload(_judge_suggestion(field_name="wrong_field"))

    with pytest.raises(ValueError, match="field_name does not match"):
        parse_judge_response(
            json.dumps(payload),
            RUBRIC,
            expected_suggestions=[{"field_name": "env_medium", "id": None}],
        )


def test_parse_judge_response_rejects_mismatched_sample_id() -> None:
    payload = _judge_payload(_judge_suggestion(sample_id="wrong-sample"))

    with pytest.raises(ValueError, match="sample_id does not match"):
        parse_judge_response(
            json.dumps(payload),
            RUBRIC,
            expected_suggestions=[{"field_name": "env_medium", "id": "sample-1"}],
        )


@pytest.mark.parametrize(
    ("suggestion_type", "completeness"),
    [("new_value", None), ("abstained", 3)],
)
def test_parse_judge_response_enforces_abstention_completeness(
    suggestion_type: str,
    completeness: int | None,
) -> None:
    payload = _judge_payload(_judge_suggestion(suggestion_type=suggestion_type, completeness=completeness))

    with pytest.raises(ValueError, match="completeness"):
        parse_judge_response(json.dumps(payload), RUBRIC)


def test_parse_judge_response_rejects_unknown_critical_flags() -> None:
    suggestion = _judge_suggestion()
    suggestion["critical_flags"] = ["not_a_real_flag"]

    with pytest.raises(ValueError, match="unsupported critical flags"):
        parse_judge_response(json.dumps(_judge_payload(suggestion)), RUBRIC)


def test_parse_judge_response_rejects_out_of_range_scores() -> None:
    suggestion = _judge_suggestion()
    suggestion["scores"]["accuracy"] = 5

    with pytest.raises(ValueError, match="output model"):
        parse_judge_response(json.dumps(_judge_payload(suggestion)), RUBRIC)


@pytest.mark.parametrize(
    "judge_text",
    [
        "not json",
        "```json\n{}\n```",
        "{}",
    ],
)
def test_parse_judge_response_rejects_noncontract_responses(judge_text: str) -> None:
    with pytest.raises(ValueError):
        parse_judge_response(judge_text, RUBRIC)


def test_judge_prompt_uses_pi_output_contract_and_configured_rubric() -> None:
    prompt = build_judge_system_prompt(RUBRIC)
    assert '"name": "accuracy"' in prompt
    assert "exactly one" in prompt
    assert "missed_supported_slots" in prompt
    assert "completeness" in prompt
    assert "Include suggestion_scores as one entry per suggestion" not in prompt
    assert build_judge_json_schema(RUBRIC) == JudgeOutput.model_json_schema()


def test_judge_rejects_rubric_that_does_not_match_fixed_output_model() -> None:
    rubric = {**RUBRIC, "metrics": [metric for metric in RUBRIC["metrics"] if metric["name"] != "accuracy"]}

    with pytest.raises(ValueError, match="must match the fixed JudgeOutput model"):
        build_judge_system_prompt(rubric)


def test_checked_in_rubric_uses_pi_output_contract() -> None:
    rubric = load_rubric(str(Path(__file__).parents[1] / "datasets" / "llm-judge" / "judge-rubric.yaml"))
    output_format = rubric["judge_output"]["format"]

    assert "exactly one" in output_format
    assert "reasoning" in output_format
    assert "scores object" in output_format
    assert "suggestion_scores" in output_format
    assert "Do not use the legacy" in output_format


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
    from nmdc_ai_eval import llm_judge

    monkeypatch.delenv("CBORG_KEY", raising=False)
    monkeypatch.setenv("CBORG_API_KEY", "test-key")
    client = object()

    def make_client(**_kwargs: Any) -> object:
        assert os.environ["CBORG_KEY"] == "test-key"
        return client

    with patch.object(llm_judge, "LLMClient", side_effect=make_client) as client_type:
        assert llm_judge._create_suggestor_client("cborg", "test-model") is client

    client_type.assert_called_once_with(access_provider="cborg", model="test-model")
    assert "CBORG_KEY" not in os.environ


@pytest.mark.parametrize("supports_schema", [False, True])
def test_evaluate_submission_runs_production_pipeline_and_judge(supports_schema: bool) -> None:
    suggestion_data = {"metadata_fields": [{"field_name": "env_medium", "value": "soil"}]}
    pipeline_output = SimpleNamespace(model_dump=lambda mode: suggestion_data)
    judge_response = SimpleNamespace(text=lambda: json.dumps(_judge_payload(_judge_suggestion(completeness=3))))
    judge_model = MagicMock()
    judge_model.supports_schema = supports_schema
    judge_model.prompt.return_value = judge_response
    suggestor_client = SimpleNamespace(access_provider="pnnl", model="test-model")
    client_type = MagicMock(return_value=suggestor_client)
    submission = {"metadata_submission": {"studyForm": {"studyName": "Test"}}}

    with (
        patch(
            "nmdc_ai_eval.llm_judge.run_recommendation_pipeline",
            return_value=pipeline_output,
        ) as pipeline,
        patch("nmdc_ai_eval.llm_judge.LLMClient", client_type),
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
    judge_model.prompt.assert_called_once()
    judge_input = json.loads(judge_model.prompt.call_args_list[0].args[0])
    assert judge_input == {
        "submission": submission,
        "suggestor_response": suggestion_data,
        "judge_task": {
            "suggestion_count": 1,
            "required_suggestion_indexes": [0],
            "suggestions_to_grade": [
                {"index": 0, "field_name": "env_medium", "sample_id": None},
            ],
        },
    }
    assert result["submission_id"] == "submission-1"
    assert result["mean_score"] == 3.8
    assert result["suggestions"][0]["field_name"] == "env_medium"
    assert result["summary"] == "The suggestion is well supported. No major failure pattern was identified."
    assert result["suggestor_provider"] == "pnnl"
    assert result["suggestor_model"] == "test-model"
    judge_model.prompt.assert_called_once()
    assert judge_model.prompt.call_args.kwargs["max_tokens"] == 100_000
    if supports_schema:
        assert judge_model.prompt.call_args.kwargs["schema"] is JudgeOutput
    else:
        assert "schema" not in judge_model.prompt.call_args.kwargs


def test_runner_creates_a_distinct_judge_model_for_each_submission(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner_path = Path(__file__).parents[1] / "datasets" / "llm-judge" / "run_eval.py"
    spec = importlib.util.spec_from_file_location("llm-judge_runner", runner_path)
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
        (("gpt-5.2",),),
        (("gpt-5.2",),),
    ]
    assert evaluate.call_args_list[0].kwargs["judge_model"] is first_judge
    assert evaluate.call_args_list[1].kwargs["judge_model"] is second_judge
    assert evaluate.call_args_list[0].kwargs["suggestor_provider"] == "pnnl"
    assert evaluate.call_args_list[0].kwargs["suggestor_model"] is None
