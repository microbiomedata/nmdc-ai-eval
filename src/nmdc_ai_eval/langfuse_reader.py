"""Read traces and agent observations from a Langfuse project.

Standard library only. The suggestor writes traces (microbiomedata/nmdc-metadata-suggestor-ai-tool
PR 139 logs every agent message); this reads them back so an eval can score what production
actually did rather than a prompt this repo wrote.

Credentials come from the environment and are never logged. Set either the bare Langfuse names or
an ``NMDC_``-prefixed set:

    LANGFUSE_PUBLIC_KEY / NMDC_LANGFUSE_PUBLIC_KEY
    LANGFUSE_SECRET_KEY / NMDC_LANGFUSE_SECRET_KEY
    LANGFUSE_BASE_URL   / NMDC_LANGFUSE_BASE_URL

Two endpoint gotchas, both found empirically on 2026-09-18 and both load-bearing:

* ``GET /api/public/scores`` returns nothing; the real scores are on
  ``GET /api/public/v2/scores``.
* The v2 observations endpoint reports ``model: null`` on every row. The legacy
  ``GET /api/public/observations?type=AGENT`` returns the model that actually served the
  request. Trace-level ``metadata.model`` disagrees with it on a third of traces, because the
  gcp path defaults to a Gemini id, so prefer the AGENT observation.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Iterator

DEFAULT_TIMEOUT_SECONDS = 60
_PAGE_LIMIT = 100


class LangfuseCredentialsMissing(RuntimeError):
    """Raised when no usable Langfuse credentials are present in the environment."""


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


@dataclass(frozen=True)
class LangfuseEndpoint:
    """Where to read from, and how to authenticate. Holds secrets; never log it."""

    base_url: str
    public_key: str
    secret_key: str

    def __repr__(self) -> str:  # pragma: no cover - trivial, but keeps keys out of tracebacks
        return f"LangfuseEndpoint(base_url={self.base_url!r}, public_key=<redacted>, secret_key=<redacted>)"

    @property
    def auth_header(self) -> str:
        raw = f"{self.public_key}:{self.secret_key}".encode()
        return "Basic " + base64.b64encode(raw).decode()


def endpoint_from_env(load_dotenv_file: bool = True) -> LangfuseEndpoint:
    """Build an endpoint from the environment, optionally loading a .env first."""
    if load_dotenv_file:
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:  # pragma: no cover - dotenv is a declared dependency
            pass

    public = _first_env("LANGFUSE_PUBLIC_KEY", "NMDC_LANGFUSE_PUBLIC_KEY")
    secret = _first_env("LANGFUSE_SECRET_KEY", "NMDC_LANGFUSE_SECRET_KEY")
    base = _first_env("LANGFUSE_BASE_URL", "NMDC_LANGFUSE_BASE_URL")

    missing = [
        name
        for name, value in (
            ("LANGFUSE_PUBLIC_KEY", public),
            ("LANGFUSE_SECRET_KEY", secret),
            ("LANGFUSE_BASE_URL", base),
        )
        if not value
    ]
    if missing:
        raise LangfuseCredentialsMissing(
            "Missing Langfuse credentials: " + ", ".join(missing) + ". An NMDC_-prefixed name is accepted for each."
        )

    # mypy: the `missing` check above guarantees all three are non-empty here.
    return LangfuseEndpoint(base_url=str(base).rstrip("/"), public_key=str(public), secret_key=str(secret))


def _get_json(endpoint: LangfuseEndpoint, path: str, params: dict[str, Any]) -> dict[str, Any]:
    query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    url = f"{endpoint.base_url}{path}?{query}"
    request = urllib.request.Request(url, headers={"Authorization": endpoint.auth_header})  # noqa: S310
    with urllib.request.urlopen(request, timeout=DEFAULT_TIMEOUT_SECONDS) as response:  # noqa: S310
        payload: dict[str, Any] = json.loads(response.read().decode())
    return payload


class LangfusePaginationGuard(RuntimeError):
    """Raised when pagination hits its runaway guard, rather than truncating in silence."""


def _paginate(
    endpoint: LangfuseEndpoint, path: str, params: dict[str, Any], max_pages: int = 100
) -> Iterator[dict[str, Any]]:
    """Yield every record across pages.

    ``max_pages`` is a runaway guard against a server that never reports a last page. Hitting
    it raises, because a caller of ``fetch_traces`` is promised every trace and a short read
    would quietly change every denominator in the report.
    """
    page = 1
    while True:
        if page > max_pages:
            raise LangfusePaginationGuard(
                f"{path}: stopped after {max_pages} pages of {_PAGE_LIMIT}. "
                "Raise max_pages if the project is genuinely this large."
            )
        payload = _get_json(endpoint, path, {**params, "page": page, "limit": _PAGE_LIMIT})
        records = payload.get("data") or []
        if not records:
            return
        yield from records
        meta = payload.get("meta") or {}
        total_pages = meta.get("totalPages")
        if total_pages is not None and page >= int(total_pages):
            return
        page += 1


def fetch_traces(endpoint: LangfuseEndpoint, **filters: Any) -> list[dict[str, Any]]:
    """Every trace in the project, newest first as the API returns them."""
    return list(_paginate(endpoint, "/api/public/traces", filters))


def fetch_agent_observations(endpoint: LangfuseEndpoint) -> list[dict[str, Any]]:
    """AGENT-type observations, which carry the model that actually served the request."""
    return list(_paginate(endpoint, "/api/public/observations", {"type": "AGENT"}))


def model_by_trace(agent_observations: list[dict[str, Any]]) -> dict[str, str]:
    """Map trace id to the served model name, ignoring nulls and synthetic placeholders."""
    mapping: dict[str, str] = {}
    for observation in agent_observations:
        trace_id = observation.get("traceId")
        model = observation.get("model")
        if not trace_id or not model or model.startswith("<"):
            continue
        mapping.setdefault(trace_id, model)
    return mapping


@dataclass
class TraceBundle:
    """One trace plus the model its AGENT observation reports."""

    trace: dict[str, Any]
    served_model: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def trace_id(self) -> str:
        return str(self.trace.get("id", ""))

    @property
    def declared_model(self) -> str | None:
        metadata = self.trace.get("metadata") or {}
        value = metadata.get("model")
        return str(value) if value else None

    @property
    def model_attribution_disagrees(self) -> bool:
        """True when trace metadata names a different model than the AGENT observation.

        Worth reporting rather than silently preferring one: a third of the traces in the NMDC
        project carry a Gemini id in metadata while the AGENT observation names a Claude model.
        """
        return bool(self.served_model and self.declared_model and self.served_model != self.declared_model)


def load_bundles(endpoint: LangfuseEndpoint, **filters: Any) -> list[TraceBundle]:
    """Fetch traces and attach each one's served model."""
    traces = fetch_traces(endpoint, **filters)
    models = model_by_trace(fetch_agent_observations(endpoint))
    return [TraceBundle(trace=t, served_model=models.get(str(t.get("id")))) for t in traces]
