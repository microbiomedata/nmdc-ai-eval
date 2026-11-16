"""Run the NMDC submission suggestor and score its output with an LLM judge."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

from nmdc_metadata_suggestor_ai_tool.env_triad_recommendation import (
    get_env_triad_recommendation,
)
from nmdc_metadata_suggestor_ai_tool.llm_client import LLMClient
from nmdc_metadata_suggestor_ai_tool.recommendation_pipeline import (
    run_recommendation_pipeline,
)

from nmdc_ai_eval.llm_judge.judge_response import (
    build_judge_json_schema,
    build_judge_system_prompt,
    load_rubric,
    parse_judge_response,
)
from nmdc_ai_eval.llm_judge.output_model import JudgeOutput
from nmdc_ai_eval.llm_judge.submission_api import SubmissionApiClient

JUDGE_MAX_OUTPUT_TOKENS = 100_000

__all__ = [
    "SubmissionApiClient",
    "build_judge_json_schema",
    "build_judge_system_prompt",
    "evaluate_submission",
    "load_rubric",
    "load_submission_ids",
    "parse_judge_response",
    "resolve_suggestor_provider",
]


def load_submission_ids(path: str) -> list[str]:
    """Read submission IDs, ignoring blank lines and comments; reject duplicates."""
    with open(path, encoding="utf-8") as file:
        submission_ids = [line.strip() for line in file if line.strip() and not line.lstrip().startswith("#")]
    if not submission_ids:
        raise ValueError(f"{path} has no submission IDs; add at least one ID before running.")
    if len(set(submission_ids)) != len(submission_ids):
        raise ValueError(f"{path} contains duplicate submission IDs.")
    return submission_ids


def resolve_suggestor_provider(
    provider: str,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Validate a provider name or infer the sole configured provider."""
    if provider not in {"auto", "gcp", "pnnl", "cborg"}:
        raise ValueError("Suggestor provider must be one of: auto, gcp, pnnl, cborg.")
    if provider != "auto":
        return provider

    env = os.environ if environ is None else environ
    configured = [
        name
        for name, enabled in (
            ("gcp", bool(env.get("VERTEX_PROJECT_ID") or env.get("GOOGLE_APPLICATION_CREDENTIALS"))),
            ("pnnl", bool(env.get("AI_INCUBATOR_KEY") and env.get("AI_INCUBATOR_BASE_URL"))),
            ("cborg", bool((env.get("CBORG_KEY") or env.get("CBORG_API_KEY")) and env.get("CBORG_BASE_URL"))),
        )
        if enabled
    ]
    if len(configured) == 1:
        return configured[0]
    if len(configured) > 1:
        choices = ", ".join(configured)
        raise ValueError(
            f"Multiple suggestor providers are configured ({choices}); pass --suggestor-provider to select one."
        )
    raise ValueError(
        "Could not auto-detect a suggestor provider. Configure Vertex "
        "(VERTEX_PROJECT_ID or GOOGLE_APPLICATION_CREDENTIALS), PNNL "
        "(AI_INCUBATOR_KEY and AI_INCUBATOR_BASE_URL), or CBORG "
        "(CBORG_API_KEY and CBORG_BASE_URL), or pass --suggestor-provider."
    )


def _create_suggestor_client(provider: str, model: str | None) -> LLMClient:
    """Create the suggestor client, temporarily adapting the CBORG key name if needed."""
    # If the provider isn't CBORG then we can just plug the params in
    if provider != "cborg":
        return LLMClient(access_provider=provider, model=model)

    key_name = "CBORG_KEY"
    had_key = key_name in os.environ
    original_key = os.environ.get(key_name)
    api_key = os.environ.get("CBORG_API_KEY")
    if original_key or not api_key:
        return LLMClient(access_provider=provider, model=model)

    os.environ[key_name] = api_key
    try:
        return LLMClient(access_provider=provider, model=model)
    finally:
        if had_key:
            os.environ[key_name] = original_key or ""
        else:
            os.environ.pop(key_name, None)


def _submission_packages(submission: dict[str, Any]) -> list[str]:
    """Return unique, non-empty package names from the normalized submission."""
    metadata = submission.get("metadata_submission")
    packages = metadata.get("packageName") if isinstance(metadata, dict) else None
    if not isinstance(packages, list):
        return []
    return sorted({package for package in packages if isinstance(package, str) and package})


def _submission_sample_records(submission: dict[str, Any], package: str | None) -> list[Any]:
    """Return package sample records, filling missing IDs with stable row indexes."""
    metadata = submission.get("metadata_submission")
    sample_data = metadata.get("sampleData") if isinstance(metadata, dict) else None
    if not isinstance(sample_data, dict) or not package:
        return []

    package_data = sample_data.get("data")
    if not isinstance(package_data, dict):
        package_data = sample_data
    records: list[Any] = []
    for candidate in (package, f"{package}_data", package.removesuffix("_data")):
        candidate_records = package_data.get(candidate)
        if isinstance(candidate_records, list):
            records = candidate_records
            break
    return [
        {**record, "id": record.get("id") or str(index)}
        for index, record in enumerate(records)
        if isinstance(record, dict)
    ]


def _run_package_suggestor(
    submission: dict[str, Any],
    suggestor: LLMClient,
    package: str | None,
) -> Any:
    """Run the suggestor for one package and add any environmental-triad fields."""
    kwargs: dict[str, Any] = {"submission_object": submission, "llm_client": suggestor}
    if package:
        kwargs["interface_name"] = package
    suggestion = run_recommendation_pipeline(**kwargs)

    records = _submission_sample_records(submission, package)
    if records and package:
        triad = get_env_triad_recommendation(
            llm_client=suggestor,
            samples=records,
            submission_object=submission,
            interface_names=[package],
        )
        suggestion.metadata_fields.extend(triad.metadata_fields)
    return suggestion


def _run_suggestor(
    submission: dict[str, Any],
    suggestor: LLMClient,
) -> Any:
    """Run the production pipeline for each package in the submission."""
    packages: list[str | None] = []
    packages.extend(_submission_packages(submission))
    if not packages:
        packages.append(None)
    suggestions = [_run_package_suggestor(submission, suggestor, package) for package in packages]

    primary = suggestions[0]
    for additional in suggestions[1:]:
        primary.metadata_fields.extend(additional.metadata_fields)
    return primary


def _build_judge_input(
    submission: dict[str, Any],
    suggestion_data: dict[str, Any],
    suggestions: list[dict[str, Any]],
) -> str:
    """Serialize the submission, suggestor output, and required suggestion identities."""
    task = {
        "suggestion_count": len(suggestions),
        "required_suggestion_indexes": list(range(len(suggestions))),
        "suggestions_to_grade": [
            {
                "index": index,
                "field_name": item.get("field_name"),
                "sample_id": item.get("id"),
            }
            for index, item in enumerate(suggestions)
        ],
    }
    return json.dumps(
        {
            "submission": submission,
            "suggestor_response": suggestion_data,
            "judge_task": task,
        },
        ensure_ascii=False,
        indent=2,
    )


def evaluate_submission(
    submission_id: str,
    submission: dict[str, Any],
    suggestor_provider: str,
    suggestor_model: str | None,
    judge_model: Any,
    judge_model_name: str,
    rubric: dict[str, Any],
) -> dict[str, Any]:
    """Run the suggestor, judge each result, and return a serializable report."""
    suggestor = _create_suggestor_client(suggestor_provider, suggestor_model)
    suggestion = _run_suggestor(submission, suggestor)
    suggestion_data = suggestion.model_dump(mode="json")
    suggestions = suggestion_data.get("metadata_fields")
    if not isinstance(suggestions, list):
        raise ValueError("Suggestor response must contain a metadata_fields list.")

    judge_input = _build_judge_input(submission, suggestion_data, suggestions)
    prompt_kwargs: dict[str, Any] = {
        "system": build_judge_system_prompt(rubric),
        "temperature": 0.0,
        "max_tokens": JUDGE_MAX_OUTPUT_TOKENS,
    }
    if getattr(judge_model, "supports_schema", False) is True:
        prompt_kwargs["schema"] = JudgeOutput
    response = judge_model.prompt(judge_input, **prompt_kwargs)
    judged = parse_judge_response(response.text(), rubric, expected_suggestions=suggestions)

    return {
        "submission_id": submission_id,
        "suggestor_provider": suggestor.access_provider,
        "suggestor_model": suggestor.model,
        "judge_model": judge_model_name,
        "suggestor_response": suggestion_data,
        **judged,
    }
