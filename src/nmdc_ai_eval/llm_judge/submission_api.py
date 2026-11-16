"""Client for fetching and normalizing NMDC metadata submissions."""

from __future__ import annotations

import json
import sys
import time
from typing import Any, NoReturn
from urllib.parse import quote, urlsplit

import requests

ACCESS_TOKEN_TTL_SECONDS = 24 * 60 * 60
ACCESS_TOKEN_REFRESH_MARGIN_SECONDS = 5 * 60
HTTP_ERROR_BODY_LIMIT = 1000
DEBUG_RESPONSE_BODY_LIMIT = 10_000


class SubmissionApiClient:
    """Fetch authenticated metadata submissions from the NMDC submission API."""

    def __init__(
        self,
        base_url: str,
        refresh_token: str,
        timeout: float = 30.0,
        debug: bool = False,
    ) -> None:
        """Configure the API client and validate credentials and URL."""
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
        """Print a bounded response diagnostic with token fields and values redacted."""
        if not self.debug:
            return

        try:
            payload: Any = response.json()
            body = json.dumps(self._redact_payload(payload), ensure_ascii=False, indent=2)
        except requests.JSONDecodeError:
            body = response.text

        for secret in (self.refresh_token, self.access_token):
            if secret:
                body = body.replace(secret, "[REDACTED]")
        if len(body) > DEBUG_RESPONSE_BODY_LIMIT:
            body = body[:DEBUG_RESPONSE_BODY_LIMIT] + "\n... [truncated]"

        endpoint = url.removeprefix(self.base_url)
        print(
            f"[debug] API {method} {endpoint} returned HTTP {response.status_code}:\n{body}",
            file=sys.stderr,
        )

    @classmethod
    def _redact_payload(cls, payload: Any) -> Any:
        """Recursively redact token-like fields in a decoded API response."""
        if isinstance(payload, dict):
            return {
                key: "[REDACTED]"
                if any(secret_name in key.lower() for secret_name in ("token", "authorization"))
                else cls._redact_payload(value)
                for key, value in payload.items()
            }
        if isinstance(payload, list):
            return [cls._redact_payload(value) for value in payload]
        return payload

    def _request_json(self, method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        """Send an API request and return an object response, caching GET results."""
        if method == "GET" and url in self._get_response_cache:
            return self._get_response_cache[url]

        headers = kwargs.get("headers")
        # A placeholder lets cached sample-set requests use the current access token.
        if isinstance(headers, dict) and headers.get("Authorization") == "******":
            headers["Authorization"] = f"Bearer {self.access_token or ''}"

        try:
            response = requests.request(method, url, timeout=self.timeout, **kwargs)
            self._debug_response(method, url, response)
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise self._http_error(exc) from exc
        except requests.RequestException as exc:
            raise RuntimeError(f"NMDC API request failed: {exc}") from exc

        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("NMDC API returned a JSON value that is not an object.")
        if method == "GET":
            self._get_response_cache[url] = payload
        return payload

    def _http_error(self, error: requests.HTTPError) -> RuntimeError:
        """Format an HTTP failure with a bounded, credential-redacted response body."""
        response = error.response
        body = response.text if response is not None else ""
        for secret in (self.refresh_token, self.access_token):
            if secret:
                body = body.replace(secret, "[REDACTED]")
        body = body.strip()
        if len(body) > HTTP_ERROR_BODY_LIMIT:
            body = body[:HTTP_ERROR_BODY_LIMIT] + "... [truncated]"

        status = response.status_code if response is not None else "unknown"
        detail = f" Response body: {body}" if body else ""
        return RuntimeError(f"NMDC API request failed with HTTP {status}.{detail}")

    def refresh_access_token(self) -> str:
        """Exchange the refresh token and record when the access token expires."""
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
        """Return a valid access token, refreshing shortly before its expiry."""
        refresh_at = self._access_token_expires_at - ACCESS_TOKEN_REFRESH_MARGIN_SECONDS
        if self.access_token is not None and time.monotonic() < refresh_at:
            return self.access_token
        return self.refresh_access_token()

    def fetch_submission(self, submission_id: str) -> dict[str, Any]:
        """Fetch a submission and normalize direct API responses for the suggestor."""
        access_token = self._get_access_token()
        url = f"{self.base_url}/api/metadata_submission/{quote(submission_id, safe='')}"
        submission = self._request_json(
            "GET",
            url,
            headers={"Authorization": f"Bearer {access_token}"},
        )
        if isinstance(submission.get("metadata_submission"), dict):
            return submission

        study_form = submission.get("study_form")
        if not isinstance(study_form, dict):
            self._raise_invalid_submission(submission)
        return self._normalize_submission(submission, study_form)

    def _normalize_submission(
        self,
        submission: dict[str, Any],
        study_form: dict[str, Any],
    ) -> dict[str, Any]:
        """Wrap direct study/sample-set fields in the suggestor's expected shape."""
        package_names, sample_data = self._fetch_sample_sets(submission.get("sample_sets"))
        multiomics_form = submission.get("multi_omics_form")
        normalized: dict[str, Any] = {
            "studyForm": study_form,
            "multiOmicsForm": multiomics_form if isinstance(multiomics_form, dict) else {},
            "packageName": sorted(package_names),
        }
        if sample_data:
            normalized["sampleData"] = sample_data
        submission["metadata_submission"] = normalized

        if self.debug:
            print(
                "[debug] Normalized the direct submission response "
                "('study_form') to the suggestor's expected "
                "'metadata_submission' wrapper.",
                file=sys.stderr,
            )
        return submission

    def _fetch_sample_sets(
        self,
        sample_sets: Any,
    ) -> tuple[set[str], dict[str, list[Any]]]:
        """Collect package names and merge sample records from sample-set details."""
        package_names: set[str] = set()
        sample_data: dict[str, list[Any]] = {}
        if not isinstance(sample_sets, list):
            return package_names, sample_data

        for sample_set in sample_sets:
            if not isinstance(sample_set, dict):
                continue
            templates = sample_set.get("templates")
            if not isinstance(templates, list):
                templates = []
            package_names.update(template for template in templates if isinstance(template, str))
            sample_set_id = sample_set.get("id")
            if not isinstance(sample_set_id, str):
                continue

            detail = self._request_json(
                "GET",
                f"{self.base_url}/api/metadata_submission/sample_set/{quote(sample_set_id, safe='')}",
                headers={"Authorization": "******"},
            )
            self._merge_sample_data(sample_data, detail.get("sample_data"))
        return package_names, sample_data

    @staticmethod
    def _merge_sample_data(
        sample_data: dict[str, list[Any]],
        sample_data_response: Any,
    ) -> None:
        """Append well-formed package records from one sample-set response."""
        if not isinstance(sample_data_response, dict):
            return
        data = sample_data_response.get("data")
        if not isinstance(data, dict):
            return
        for package, records in data.items():
            if isinstance(package, str) and isinstance(records, list):
                sample_data.setdefault(package, []).extend(records)

    def _raise_invalid_submission(self, submission: dict[str, Any]) -> NoReturn:
        """Raise a useful error when the API response has neither supported shape."""
        if self.debug:
            response_keys = ", ".join(sorted(submission))
            print(
                "[debug] API response is missing a dictionary-valued "
                f"'metadata_submission' field; top-level keys: {response_keys or '(none)'}.",
                file=sys.stderr,
            )
        raise ValueError("NMDC API response did not contain a metadata_submission object.")
