"""Run the NMDC submission suggestor and score its output with an LLM judge."""

from __future__ import annotations

import json
import os
import re
import sys
import time
from collections.abc import Mapping
from typing import Any
from urllib.parse import quote, urlsplit

import requests
import yaml
from nmdc_metadata_suggestor_ai_tool.env_triad_recommendation import (
    get_env_triad_recommendation,
)
from nmdc_metadata_suggestor_ai_tool.llm_client import ConversationManager, LLMClient
from nmdc_metadata_suggestor_ai_tool.recommendation_pipeline import (
    get_submission_fields,
    run_recommendation_pipeline,
)

SCORE_VALUES = (1, 2, 3, 4)
# This is just to hedge against very long running evals where we might need to grab another access token.
ACCESS_TOKEN_TTL_SECONDS = 24 * 60 * 60
ACCESS_TOKEN_REFRESH_MARGIN_SECONDS = 5 * 60
HTTP_ERROR_BODY_LIMIT = 1000
DEBUG_RESPONSE_BODY_LIMIT = 10_000
JUDGE_MAX_OUTPUT_TOKENS = 100_000


def resolve_suggestor_provider(
    provider: str,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Resolve ``auto`` to the one suggestor provider configured in the environment."""
    if provider not in {"auto", "gcp", "pnnl", "cborg"}:
        raise ValueError("Suggestor provider must be one of: auto, gcp, pnnl, cborg.")
    if provider != "auto":
        return provider

    env = os.environ if environ is None else environ
    configured: list[str] = []
    if env.get("VERTEX_PROJECT_ID") or env.get("GOOGLE_APPLICATION_CREDENTIALS"):
        configured.append("gcp")
    if env.get("AI_INCUBATOR_KEY") and env.get("AI_INCUBATOR_BASE_URL"):
        configured.append("pnnl")
    if (env.get("CBORG_KEY") or env.get("CBORG_API_KEY")) and env.get("CBORG_BASE_URL"):
        configured.append("cborg")

    if not configured:
        raise ValueError(
            "Could not auto-detect a suggestor provider. Configure Vertex "
            "(VERTEX_PROJECT_ID or GOOGLE_APPLICATION_CREDENTIALS), PNNL "
            "(AI_INCUBATOR_KEY and AI_INCUBATOR_BASE_URL), or CBORG "
            "(CBORG_API_KEY and CBORG_BASE_URL), or pass --suggestor-provider."
        )
    if len(configured) > 1:
        choices = ", ".join(configured)
        raise ValueError(
            f"Multiple suggestor providers are configured ({choices}); pass --suggestor-provider to select one."
        )
    return configured[0]


def _create_suggestor_client(provider: str, model: str | None) -> LLMClient:
    """Create the suggestor package client, adapting this repo's CBORG key name."""
    had_cborg_key = "CBORG_KEY" in os.environ
    previous_cborg_key = os.environ.get("CBORG_KEY")
    if provider == "cborg" and not previous_cborg_key and os.environ.get("CBORG_API_KEY"):
        os.environ["CBORG_KEY"] = os.environ["CBORG_API_KEY"]
    try:
        return LLMClient(access_provider=provider, model=model)
    finally:
        if provider == "cborg" and not previous_cborg_key:
            if had_cborg_key:
                os.environ["CBORG_KEY"] = previous_cborg_key or ""
            else:
                os.environ.pop("CBORG_KEY", None)


def _submission_packages(submission: dict[str, Any]) -> list[str]:
    metadata_submission = submission.get("metadata_submission")
    if not isinstance(metadata_submission, dict):
        return []
    packages = metadata_submission.get("packageName")
    if not isinstance(packages, list):
        return []
    return sorted({package for package in packages if isinstance(package, str) and package})


def _submission_sample_records(submission: dict[str, Any], package: str | None) -> list[Any]:
    metadata_submission = submission.get("metadata_submission")
    if not isinstance(metadata_submission, dict):
        return []
    sample_data = metadata_submission.get("sampleData")
    if not isinstance(sample_data, dict):
        return []
    data = sample_data.get("data")
    if isinstance(data, dict):
        package_data: dict[str, Any] = data
    else:
        package_data = sample_data
    candidates = [package] if package else []
    if package:
        candidates.extend((f"{package}_data", package.removesuffix("_data")))
    records: list[Any] = []
    for candidate in candidates:
        candidate_records = package_data.get(candidate)
        if isinstance(candidate_records, list):
            records = candidate_records
            break
    if not records:
        return []
    prepared_records: list[Any] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        prepared = dict(record)
        if not prepared.get("id"):
            prepared["id"] = str(index)
        prepared_records.append(prepared)
    return prepared_records


def _run_suggestor_with_context_capture(
    submission: dict[str, Any],
    suggestor: LLMClient,
) -> tuple[Any, list[str]]:
    """Run the production pipeline while capturing its text context for the judge."""
    context_messages: list[str] = []
    original_add_message = ConversationManager.add_message

    def capture_message(
        manager: Any,
        text: str,
        pdf_files: list[str] | None = None,
    ) -> None:
        if text:
            context_messages.append(text)
        original_add_message(manager, text=text, pdf_files=pdf_files)

    # The production pipeline constructs ConversationManager internally, so temporarily wrap
    # its message method during this synchronous call and restore it even on pipeline failure.
    ConversationManager.add_message = capture_message
    discovered_packages = _submission_packages(submission)
    packages: list[str | None] = list(discovered_packages)
    if not packages:
        packages = [None]
    try:
        suggestion: Any | None = None
        for package in packages:
            records = _submission_sample_records(submission, package)
            pipeline_kwargs: dict[str, Any] = {
                "submission_object": submission,
                "llm_client": suggestor,
            }
            if package is not None:
                pipeline_kwargs["interface_name"] = package
            package_suggestion = run_recommendation_pipeline(**pipeline_kwargs)

            if records and package is not None:
                package_suggestion.metadata_fields.extend(
                    get_env_triad_recommendation(
                        llm_client=suggestor,
                        samples=records,
                        submission_object=submission,
                        interface_names=[package],
                    ).metadata_fields
                )

            if suggestion is None:
                suggestion = package_suggestion
            else:
                suggestion.metadata_fields.extend(package_suggestion.metadata_fields)
    finally:
        ConversationManager.add_message = original_add_message
    if suggestion is None:
        raise RuntimeError("The suggestor pipeline did not produce a response.")
    return suggestion, context_messages


class SubmissionApiClient:
    """Fetch authenticated metadata submissions from the NMDC submission API."""

    def __init__(
        self,
        base_url: str,
        refresh_token: str,
        timeout: float = 30.0,
        debug: bool = False,
    ) -> None:
        if not refresh_token.strip():
            raise ValueError("A non-empty NMDC API refresh token is required.")
        parsed_url = urlsplit(base_url)
        if parsed_url.scheme != "https" or not parsed_url.netloc:
            raise ValueError("NMDC API base URL must be an absolute HTTPS URL.")
        self.base_url = base_url.rstrip("/")
        self.refresh_token = refresh_token
        self.timeout = timeout
        self.debug = debug
        self.access_token: str | None = None
        self._access_token_expires_at = 0.0
        self._get_response_cache: dict[str, dict[str, Any]] = {}

    def _debug_response(self, method: str, url: str, response: requests.Response) -> None:
        if not self.debug:
            return
        try:
            payload: Any = response.json()
            output = json.dumps(self._redact_debug_payload(payload), ensure_ascii=False, indent=2)
        except requests.JSONDecodeError:
            output = response.text
        for secret in (self.refresh_token, self.access_token):
            if secret:
                output = output.replace(secret, "[REDACTED]")
        if len(output) > DEBUG_RESPONSE_BODY_LIMIT:
            output = output[:DEBUG_RESPONSE_BODY_LIMIT] + "\n... [truncated]"
        endpoint = url.removeprefix(self.base_url)
        print(
            f"[debug] API {method} {endpoint} returned HTTP {response.status_code}:\n{output}",
            file=sys.stderr,
        )

    @classmethod
    def _redact_debug_payload(cls, payload: Any) -> Any:
        if isinstance(payload, dict):
            return {
                key: "[REDACTED]"
                if any(secret_name in key.lower() for secret_name in ("token", "authorization"))
                else cls._redact_debug_payload(value)
                for key, value in payload.items()
            }
        if isinstance(payload, list):
            return [cls._redact_debug_payload(value) for value in payload]
        return payload

    def _request_json(self, method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        if method == "GET" and url in self._get_response_cache:
            return self._get_response_cache[url]
        headers = kwargs.get("headers")
        try:
            if isinstance(headers, dict) and headers.get("Authorization") == "******":
                headers["Authorization"] = "Bearer " + (self.access_token or "")
            response = requests.request(method, url, timeout=self.timeout, **kwargs)
            self._debug_response(method, url, response)
            response.raise_for_status()
        except requests.HTTPError as exc:
            error_response = exc.response
            body = error_response.text if error_response is not None else ""
            for secret in (self.refresh_token, self.access_token):
                if secret:
                    body = body.replace(secret, "[REDACTED]")
            body = body.strip()
            if len(body) > HTTP_ERROR_BODY_LIMIT:
                body = body[:HTTP_ERROR_BODY_LIMIT] + "... [truncated]"
            detail = f" Response body: {body}" if body else ""
            status_code = error_response.status_code if error_response is not None else "unknown"
            raise RuntimeError(f"NMDC API request failed with HTTP {status_code}.{detail}") from exc
        except requests.RequestException as exc:
            raise RuntimeError(f"NMDC API request failed: {exc}") from exc
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("NMDC API returned a JSON value that is not an object.")
        if method == "GET":
            self._get_response_cache[url] = payload
        return payload

    def refresh_access_token(self) -> str:
        """Exchange the refresh token for an access token valid for 24 hours."""
        payload = self._request_json(
            "POST",
            f"{self.base_url}/auth/refresh",
            json={"refresh_token": self.refresh_token},
        )
        token = payload.get("access_token")
        if not isinstance(token, str) or not token.strip():
            raise ValueError("NMDC API refresh response did not contain an access token.")
        self.access_token = token
        self._access_token_expires_at = time.monotonic() + ACCESS_TOKEN_TTL_SECONDS
        return token

    def _get_access_token(self) -> str:
        """Return a valid access token, refreshing shortly before its 24-hour expiry."""
        refresh_at = self._access_token_expires_at - ACCESS_TOKEN_REFRESH_MARGIN_SECONDS
        if self.access_token is None or time.monotonic() >= refresh_at:
            return self.refresh_access_token()
        return self.access_token

    def fetch_submission(self, submission_id: str) -> dict[str, Any]:
        """Fetch one complete submission document by its portal ID."""
        access_token = self._get_access_token()
        submission_url = f"{self.base_url}/api/metadata_submission/{quote(submission_id, safe='')}"
        submission = self._request_json(
            "GET",
            submission_url,
            headers={"Authorization": "Bearer " + access_token},
        )
        submission = self._request_json(
            "GET",
            submission_url,
            headers={"Authorization": f"Bearer {access_token}"},
        )
        if isinstance(submission.get("metadata_submission"), dict):
            return submission

        study_form = submission.get("study_form")
        if isinstance(study_form, dict):
            sample_sets = submission.get("sample_sets", [])
            package_names: set[str] = set()
            sample_data: dict[str, list[Any]] = {}
            if isinstance(sample_sets, list):
                for sample_set in sample_sets:
                    if isinstance(sample_set, dict):
                        templates = sample_set.get("templates")
                        if isinstance(templates, list):
                            package_names.update(template for template in templates if isinstance(template, str))
                        sample_set_id = sample_set.get("id")
                        if isinstance(sample_set_id, str):
                            detail = self._request_json(
                                "GET",
                                f"{self.base_url}/api/metadata_submission/sample_set/{quote(sample_set_id, safe='')}",
                                headers={"Authorization": f"Bearer {access_token}"},
                            )
                            detail_sample_data = detail.get("sample_data")
                            if isinstance(detail_sample_data, dict):
                                data = detail_sample_data.get("data")
                                if isinstance(data, dict):
                                    for package, records in data.items():
                                        if isinstance(package, str) and isinstance(records, list):
                                            sample_data.setdefault(package, []).extend(records)
            multiomics_form = submission.get("multi_omics_form")
            submission["metadata_submission"] = {
                "studyForm": study_form,
                "multiOmicsForm": multiomics_form if isinstance(multiomics_form, dict) else {},
                "packageName": sorted(package_names),
            }
            if sample_data:
                submission["metadata_submission"]["sampleData"] = sample_data
            if self.debug:
                print(
                    "[debug] Normalized the direct submission response "
                    "('study_form') to the suggestor's expected "
                    "'metadata_submission' wrapper.",
                    file=sys.stderr,
                )
            return submission

        if not isinstance(submission.get("metadata_submission"), dict):
            if self.debug:
                response_keys = ", ".join(sorted(submission))
                print(
                    "[debug] API response is missing a dictionary-valued "
                    f"'metadata_submission' field; top-level keys: {response_keys or '(none)'}.",
                    file=sys.stderr,
                )
            raise ValueError("NMDC API response did not contain a metadata_submission object.")
        return submission


def load_submission_ids(path: str) -> list[str]:
    """Read one submission ID per line, ignoring blank lines and comments."""
    with open(path, encoding="utf-8") as file:
        submission_ids = [line.strip() for line in file if line.strip() and not line.lstrip().startswith("#")]
    if not submission_ids:
        raise ValueError(f"{path} has no submission IDs; add at least one ID before running.")
    if len(set(submission_ids)) != len(submission_ids):
        raise ValueError(f"{path} contains duplicate submission IDs.")
    return submission_ids


def load_rubric(path: str) -> dict[str, Any]:
    """Load and validate the configured metric descriptions and 1–4 anchors."""
    with open(path, encoding="utf-8") as file:
        rubric = yaml.safe_load(file)
    if not isinstance(rubric, dict) or not isinstance(rubric.get("metrics"), list) or not rubric["metrics"]:
        raise ValueError(f"{path} must define a non-empty 'metrics' list.")

    seen: set[str] = set()
    for metric in rubric["metrics"]:
        if not isinstance(metric, dict):
            raise ValueError(f"{path} contains a metric that is not an object.")
        name = metric.get("name")
        description = metric.get("description")
        anchors = metric.get("score_anchors")
        if not isinstance(name, str) or not name.strip() or name in seen:
            raise ValueError(f"{path} contains an empty or duplicate metric name.")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"Metric '{name}' in {path} must have a description.")
        null_when = metric.get("null_when")
        if null_when is not None and (not isinstance(null_when, str) or not null_when.strip()):
            raise ValueError(f"Metric '{name}' in {path} has an invalid null condition.")
        if not isinstance(anchors, dict) or set(anchors) != set(SCORE_VALUES):
            raise ValueError(f"Metric '{name}' in {path} must define score anchors 1, 2, 3, and 4.")
        if any(not isinstance(anchors[score], str) or not anchors[score].strip() for score in SCORE_VALUES):
            raise ValueError(f"Metric '{name}' in {path} has an empty score anchor.")
        seen.add(name)
    return rubric


def build_judge_system_prompt(rubric: dict[str, Any]) -> str:
    """Build the judge instructions from the editable rubric configuration."""
    rubric_json = json.dumps(rubric, ensure_ascii=False, indent=2)
    return (
        "Evaluate the suggestion response using only the submission context and rubric below. "
        "Score every listed metric using its anchors. Scores must be integers from 1 to 4, except "
        "a metric with a rubric-defined null condition may have a null score when that condition "
        "applies. Do not add or omit metrics. Write reasoning before scores and give a concise "
        "evidence-based rationale for every score. "
        "Keep each reasoning and rationale concise to fit the complete JSON response. "
        "Do not begin with a partial response and do not stop before every suggestion index has "
        "an entry. Return compact JSON rather than pretty-printed JSON. "
        "Provide the submission-level overall score for the full set of suggestions and "
        "one independent score for every suggestion in the same response. Preserve the "
        "response-level scores as the overall score. "
        "Assign every suggestion a suggestion_type from new_value, validated_existing, "
        "corrected_existing, or abstained. Assign only applicable critical_flags from the rubric. "
        "Identify only clear missed_slots with supporting evidence. "
        "Return only a valid JSON object with this shape: "
        '{"reasoning":"...","scores":[{"metric":"metric_name","score":1,"rationale":"..."}],'
        '"suggestion_scores":[{"suggestion_index":0,"reasoning":"...","scores":[{'
        '"metric":"metric_name","score":1,"rationale":"..."}]}],'
        '"suggestion_types":[{"suggestion_index":0,"type":"new_value"}],'
        '"critical_flags":[{"suggestion_index":0,"flag":"invalid_inference","evidence":"..."}],'
        '"missed_slots":[{"slot":"slot_name","evidence":"..."}]}; use null for a score '
        "only when allowed by that metric's rubric condition. Omit unsupported flags and analyses "
        "rather than guessing. "
        "Rubric:\n"
        f"{rubric_json}"
    )


def build_judge_json_schema(rubric: dict[str, Any]) -> dict[str, Any]:
    """Build the JSON schema for one complete judge response."""
    metric_names = {metric["name"] for metric in rubric["metrics"]}
    score_item = {
        "type": "object",
        "properties": {
            "metric": {"type": "string", "enum": sorted(metric_names)},
            "score": {"type": ["integer", "null"], "minimum": 1, "maximum": 4},
            "rationale": {"type": "string", "minLength": 1},
        },
        "required": ["metric", "score", "rationale"],
        "additionalProperties": False,
    }
    suggestion_score = {
        "type": "object",
        "properties": {
            "suggestion_index": {"type": "integer", "minimum": 0},
            "reasoning": {"type": "string", "minLength": 1},
            "scores": {
                "type": "array",
                "items": score_item,
                "minItems": len(rubric["metrics"]),
                "maxItems": len(rubric["metrics"]),
            },
        },
        "required": ["suggestion_index", "reasoning", "scores"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string", "minLength": 1},
            "scores": {
                "type": "array",
                "items": score_item,
                "minItems": len(rubric["metrics"]),
                "maxItems": len(rubric["metrics"]),
            },
            "suggestion_scores": {"type": "array", "items": suggestion_score},
            "suggestion_types": {"type": "array", "items": {"type": "object"}},
            "critical_flags": {"type": "array", "items": {"type": "object"}},
            "missed_slots": {"type": "array", "items": {"type": "object"}},
        },
        "required": ["reasoning", "scores", "suggestion_scores"],
        "additionalProperties": False,
    }


def parse_judge_response(
    response_text: str,
    rubric: dict[str, Any],
) -> dict[str, Any]:
    """Validate judge output against the structured judge schema."""
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) < 3 or lines[-1].strip() != "```":
            raise ValueError("Judge response has an incomplete JSON code fence.")
        text = "\n".join(lines[1:-1])
        if text.lower().startswith("json\n"):
            text = text[text.index("\n") + 1 :]
    elif "```json" in text.lower():
        fenced = text.lower().find("```json")
        text = text[fenced + len("```json") :].lstrip()
        if text.endswith("```"):
            text = text[:-3].rstrip()
    elif text.endswith("```"):
        fence = text.rfind("```", 0, len(text) - 3)
        if fence >= 0:
            text = text[fence + 3 :].rstrip()
            if text.endswith("```"):
                text = text[:-3].rstrip()

    # Some models append commentary or a code fence around otherwise valid JSON.
    # Extract the first complete object and ignore surrounding prose.
    candidate = text
    if not candidate.startswith("{"):
        object_start = candidate.find("{")
        if object_start >= 0:
            candidate = candidate[object_start:]

    decode_error: json.JSONDecodeError
    try:
        payload, _ = json.JSONDecoder().raw_decode(candidate)
    except json.JSONDecodeError as exc:
        decode_error = exc
        repaired_candidate = re.sub(r",(\s*[}\]])", r"\1", candidate)
        repaired_candidate = re.sub(
            r'(["}\]])(\s*)("[A-Za-z_][A-Za-z0-9_-]*"\s*:)',
            r"\1,\2\3",
            repaired_candidate,
        )
        repaired_candidate = re.sub(r"(?<=})(\s*)(?={)", r",\1", repaired_candidate)
        repaired_candidate = re.sub(r"(?<=])(\s*)(?={)", r",\1", repaired_candidate)
        try:
            payload, _ = json.JSONDecoder().raw_decode(repaired_candidate)
        except json.JSONDecodeError as repair_exc:
            decode_error = repair_exc
            payload = None
    if payload is None:
        preview = " ".join(response_text.split())
        if len(preview) > 500:
            preview = preview[:500] + "..."
        raise ValueError(
            f"Judge response is not valid JSON at line {decode_error.lineno} column {decode_error.colno}: "
            f"{decode_error.msg}. Response preview: {preview!r}"
        ) from decode_error
    if not isinstance(payload, dict) or not isinstance(payload.get("scores"), list):
        raise ValueError("Judge response must contain a 'scores' list.")
    suggestion_scores_payload = payload.get("suggestion_scores", [])
    if not isinstance(suggestion_scores_payload, list):
        raise ValueError("Judge response must contain a 'suggestion_scores' list.")

    def parse_scores(raw_scores: Any) -> list[dict[str, Any]]:
        if not isinstance(raw_scores, list):
            raise ValueError("Judge response must contain a 'scores' list.")
        expected = {metric["name"] for metric in rubric["metrics"]}
        scores: dict[str, dict[str, Any]] = {}
        for item in raw_scores:
            if not isinstance(item, dict):
                raise ValueError("Judge response contains a score that is not an object.")
            name = item.get("metric")
            score = item.get("score")
            rationale = item.get("rationale")
            if not isinstance(name, str) or name not in expected:
                raise ValueError(f"Judge response contains an unknown metric: {name!r}.")
            if name in scores:
                raise ValueError(f"Judge response contains metric '{name}' more than once.")
            if not isinstance(rationale, str) or not rationale.strip():
                raise ValueError(f"Judge rationale for '{name}' must be a non-empty string.")
            metric_rubric = next(metric for metric in rubric["metrics"] if metric["name"] == name)
            if score is None and metric_rubric.get("null_when"):
                scores[name] = {"score": None, "rationale": rationale.strip()}
                continue
            if type(score) is not int or score not in SCORE_VALUES:
                raise ValueError(f"Judge score for '{name}' must be an integer from 1 through 4.")
            scores[name] = {"score": score, "rationale": rationale.strip()}

        if set(scores) != expected:
            missing = sorted(expected - set(scores))
            raise ValueError(f"Judge response omitted required metrics: {missing}.")
        return [{"metric": metric["name"], **scores[metric["name"]]} for metric in rubric["metrics"]]

    def mean_score(scores: list[dict[str, Any]]) -> float | None:
        numeric_scores = [item["score"] for item in scores if item["score"] is not None]
        return round(sum(numeric_scores) / len(numeric_scores), 2) if numeric_scores else None

    overall_scores = parse_scores(payload["scores"])
    if "suggestion_scores" not in payload:
        raise ValueError("Judge response must contain a 'suggestion_scores' list.")
    parsed: dict[str, Any] = {
        "scores": overall_scores,
        "mean_score": mean_score(overall_scores),
    }
    if isinstance(payload.get("reasoning"), str):
        parsed["reasoning"] = payload["reasoning"].strip()

    parsed_suggestion_scores: list[dict[str, Any]] = []
    seen_suggestion_indexes: set[int] = set()
    for item in suggestion_scores_payload:
        if (
            not isinstance(item, dict)
            or type(item.get("suggestion_index")) is not int
            or item["suggestion_index"] < 0
            or not isinstance(item.get("reasoning"), str)
            or not item["reasoning"].strip()
        ):
            raise ValueError("Judge suggestion_scores contains an invalid entry.")
        suggestion_index = item["suggestion_index"]
        if suggestion_index in seen_suggestion_indexes:
            raise ValueError(f"Judge response contains suggestion index {suggestion_index} more than once.")
        seen_suggestion_indexes.add(suggestion_index)
        suggestion_scores = parse_scores(item.get("scores"))
        parsed_suggestion_scores.append(
            {
                "suggestion_index": suggestion_index,
                "reasoning": item["reasoning"].strip(),
                "scores": suggestion_scores,
                "mean_score": mean_score(suggestion_scores),
            }
        )
    parsed["suggestion_scores"] = parsed_suggestion_scores

    suggestion_types = payload.get("suggestion_types", [])
    if suggestion_types:
        if not isinstance(suggestion_types, list):
            raise ValueError("Judge suggestion_types must be a list.")
        allowed_types = {
            "new_value",
            "validated_existing",
            "corrected_existing",
            "abstained",
        }
        parsed_types: list[dict[str, Any]] = []
        for item in suggestion_types:
            if (
                not isinstance(item, dict)
                or type(item.get("suggestion_index")) is not int
                or item.get("type") not in allowed_types
            ):
                raise ValueError("Judge suggestion_types contains an invalid entry.")
            parsed_types.append(item)
        parsed["suggestion_types"] = parsed_types

    critical_flags = payload.get("critical_flags", [])
    if critical_flags:
        if not isinstance(critical_flags, list):
            raise ValueError("Judge critical_flags must be a list.")
        allowed_flags = set(rubric.get("critical_flags", []))
        parsed_flags: list[dict[str, Any]] = []
        for item in critical_flags:
            if (
                not isinstance(item, dict)
                or type(item.get("suggestion_index")) is not int
                or item.get("flag") not in allowed_flags
                or not isinstance(item.get("evidence"), str)
                or not item["evidence"].strip()
            ):
                raise ValueError("Judge critical_flags contains an invalid entry.")
            parsed_flags.append(
                {
                    "suggestion_index": item["suggestion_index"],
                    "flag": item["flag"],
                    "evidence": item["evidence"].strip(),
                }
            )
        parsed["critical_flags"] = parsed_flags

    missed_slots = payload.get("missed_slots", [])
    if missed_slots:
        if not isinstance(missed_slots, list):
            raise ValueError("Judge missed_slots must be a list.")
        parsed_missed_slots: list[dict[str, Any]] = []
        for item in missed_slots:
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("slot"), str)
                or not item["slot"].strip()
                or not isinstance(item.get("evidence"), str)
                or not item["evidence"].strip()
            ):
                raise ValueError("Judge missed_slots contains an invalid entry.")
            parsed_missed_slots.append({"slot": item["slot"].strip(), "evidence": item["evidence"].strip()})
        parsed["missed_slots"] = parsed_missed_slots
    return parsed


def evaluate_submission(
    submission_id: str,
    submission: dict[str, Any],
    suggestor_provider: str,
    suggestor_model: str | None,
    judge_model: Any,
    judge_model_name: str,
    rubric: dict[str, Any],
) -> dict[str, Any]:
    """Run the production suggestor pipeline and judge its parsed output."""
    parsed_submission = get_submission_fields(submission_object=submission)
    suggestor = _create_suggestor_client(suggestor_provider, suggestor_model)
    suggestion, context_messages = _run_suggestor_with_context_capture(submission, suggestor)
    suggestion_data = suggestion.model_dump(mode="json")

    judge_input = json.dumps(
        {
            "submission": parsed_submission,
            "suggestor_context_messages": context_messages,
            "suggestor_response": suggestion_data,
            "judge_task": {
                "suggestion_count": len(suggestion_data.get("metadata_fields", [])),
                "required_suggestion_indexes": list(range(len(suggestion_data.get("metadata_fields", [])))),
            },
        },
        ensure_ascii=False,
        indent=2,
    )
    prompt_kwargs: dict[str, Any] = {
        "system": build_judge_system_prompt(rubric),
        "temperature": 0.0,
        "max_tokens": JUDGE_MAX_OUTPUT_TOKENS,
    }
    if getattr(judge_model, "supports_schema", False) is True:
        prompt_kwargs["schema"] = build_judge_json_schema(rubric)
    judge_response = judge_model.prompt(judge_input, **prompt_kwargs)
    judged = parse_judge_response(judge_response.text(), rubric)
    suggestions = suggestion_data.get("metadata_fields", [])
    if not isinstance(suggestions, list):
        raise ValueError("Suggestor response must contain a metadata_fields list.")
    expected_indexes = set(range(len(suggestions)))
    scored_indexes = {item["suggestion_index"] for item in judged["suggestion_scores"]}
    if scored_indexes != expected_indexes:
        missing_indexes = sorted(expected_indexes - scored_indexes)
        unexpected_indexes = sorted(scored_indexes - expected_indexes)
        raise ValueError(
            "Judge suggestion_scores coverage mismatch: "
            f"expected {len(expected_indexes)}, got {len(scored_indexes)}; "
            f"missing indexes={missing_indexes[:20]}, unexpected indexes={unexpected_indexes[:20]}."
            " The judge response appears incomplete; use a judge model with a larger output "
            "token limit and rerun."
        )

    return {
        "submission_id": submission_id,
        "suggestor_provider": suggestor.access_provider,
        "suggestor_model": suggestor.model,
        "judge_model": judge_model_name,
        "suggestor_response": suggestion_data,
        **judged,
    }
