"""Build judge instructions and validate structured judge responses."""

from __future__ import annotations

import json
from typing import Any

import yaml
from pydantic import ValidationError

from nmdc_ai_eval.llm_judge.output_model import JudgeOutput, Scores

SCORE_VALUES = (1, 2, 3, 4)
JUDGE_METRIC_NAMES = frozenset(Scores.model_fields)


def load_rubric(path: str) -> dict[str, Any]:
    """Load rubric YAML and validate metric descriptions, anchors, and names."""
    with open(path, encoding="utf-8") as file:
        rubric = yaml.safe_load(file)
    if not isinstance(rubric, dict) or not isinstance(rubric.get("metrics"), list) or not rubric["metrics"]:
        raise ValueError(f"{path} must define a non-empty 'metrics' list.")

    seen: set[str] = set()
    for metric in rubric["metrics"]:
        if not isinstance(metric, dict):
            raise ValueError(f"{path} contains a metric that is not an object.")
        name = metric.get("name")
        anchors = metric.get("score_anchors")
        if not isinstance(name, str) or not name.strip() or name in seen:
            raise ValueError(f"{path} contains an empty or duplicate metric name.")
        if not isinstance(metric.get("description"), str) or not metric["description"].strip():
            raise ValueError(f"Metric '{name}' in {path} must have a description.")
        null_when = metric.get("null_when")
        if null_when is not None and (not isinstance(null_when, str) or not null_when.strip()):
            raise ValueError(f"Metric '{name}' in {path} has an invalid null condition.")
        if not isinstance(anchors, dict) or set(anchors) != set(SCORE_VALUES):
            raise ValueError(f"Metric '{name}' in {path} must define score anchors 1, 2, 3, and 4.")
        if any(not isinstance(anchors[score], str) or not anchors[score].strip() for score in SCORE_VALUES):
            raise ValueError(f"Metric '{name}' in {path} has an empty score anchor.")
        seen.add(name)

    _validate_rubric_metrics(rubric)
    return rubric


def _validate_rubric_metrics(rubric: dict[str, Any]) -> None:
    """Require the rubric metric names to match the fixed structured output."""
    rubric_metric_names = {metric["name"] for metric in rubric["metrics"]}
    if rubric_metric_names == JUDGE_METRIC_NAMES:
        return

    missing = sorted(JUDGE_METRIC_NAMES - rubric_metric_names)
    unsupported = sorted(rubric_metric_names - JUDGE_METRIC_NAMES)
    raise ValueError(
        "The judge rubric metrics must match the fixed JudgeOutput model. "
        f"Missing metrics={missing}, unsupported metrics={unsupported}."
    )


def build_judge_system_prompt(rubric: dict[str, Any]) -> str:
    """Serialize the configured rubric as the judge's system prompt."""
    _validate_rubric_metrics(rubric)
    return json.dumps(rubric, ensure_ascii=False, indent=2)


def build_judge_json_schema(rubric: dict[str, Any]) -> dict[str, Any]:
    """Return the response schema after checking rubric/model compatibility."""
    _validate_rubric_metrics(rubric)
    return JudgeOutput.model_json_schema()


def _parse_json(response_text: str) -> Any:
    """Decode a plain JSON response and report concise syntax errors."""
    preview = " ".join(response_text.split())
    if len(preview) > 500:
        preview = preview[:500] + "..."
    try:
        return json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Judge response is not valid JSON at line {exc.lineno} column {exc.colno}: "
            f"{exc.msg}. Response preview: {preview!r}"
        ) from exc


def _validate_judged_suggestion_identity(
    output: JudgeOutput,
    expected_suggestions: list[dict[str, Any]],
) -> None:
    """Ensure every judged item matches the suggestor's index, field, and sample."""
    expected_by_index: dict[int, tuple[str, str | None]] = {}
    for index, expected in enumerate(expected_suggestions):
        field_name = expected.get("field_name")
        sample_id = expected.get("id")
        if not isinstance(field_name, str) or not field_name.strip():
            raise ValueError(f"Suggestor suggestion at index {index} has no valid field_name.")
        if sample_id is not None and not isinstance(sample_id, str):
            raise ValueError(f"Suggestor suggestion at index {index} has an invalid sample id.")
        expected_by_index[index] = (field_name, sample_id)

    actual_indexes = {item.index for item in output.suggestions}
    expected_indexes = set(expected_by_index)
    if actual_indexes != expected_indexes:
        missing = sorted(expected_indexes - actual_indexes)
        unexpected = sorted(actual_indexes - expected_indexes)
        raise ValueError(
            "Judge suggestions coverage mismatch: "
            f"missing indexes={missing[:20]}, unexpected indexes={unexpected[:20]}."
        )

    for item in output.suggestions:
        expected_field, expected_sample_id = expected_by_index[item.index]
        if item.field_name != expected_field:
            raise ValueError(f"Judge suggestion index {item.index} field_name does not match the suggestor response.")
        if item.sample_id != expected_sample_id:
            raise ValueError(f"Judge suggestion index {item.index} sample_id does not match the suggestor response.")


def parse_judge_response(
    response_text: str,
    rubric: dict[str, Any],
    expected_suggestions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate judge output and derive its mean score."""
    try:
        output = JudgeOutput.model_validate(_parse_json(response_text))
    except ValidationError as exc:
        raise ValueError(f"Judge response does not match the output model: {exc}") from exc

    allowed_flags = set(rubric.get("critical_flags", []))
    seen_indexes: set[int] = set()
    for item in output.suggestions:
        if item.index in seen_indexes:
            raise ValueError(f"Judge response contains suggestion index {item.index} more than once.")
        seen_indexes.add(item.index)
        unexpected = sorted(set(item.critical_flags) - allowed_flags)
        if unexpected:
            raise ValueError(f"Judge response contains unsupported critical flags: {unexpected}.")
        if (item.scores.completeness is None) != (item.suggestion_type == "abstained"):
            raise ValueError("Judge completeness must be null only for an abstained suggestion.")

    if expected_suggestions is not None:
        _validate_judged_suggestion_identity(output, expected_suggestions)

    result = output.model_dump(mode="json")
    scores = (score for item in output.suggestions for score in item.scores.model_dump().values())
    numeric_scores = [score for score in scores if score is not None]
    result["mean_score"] = round(sum(numeric_scores) / len(numeric_scores), 2) if numeric_scores else None
    return result
