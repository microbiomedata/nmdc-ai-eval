"""Tests for the Langfuse reader. No network: HTTP is stubbed at the json-fetch boundary."""

import base64
import json

import pytest

from nmdc_ai_eval import langfuse_reader as reader
from nmdc_ai_eval.langfuse_reader import (
    LangfuseCredentialsMissing,
    LangfuseEndpoint,
    TraceBundle,
    endpoint_from_env,
    fetch_agent_observations,
    fetch_traces,
    load_bundles,
    model_by_trace,
)

# Obviously-fake credentials for a non-routable host. S106 flags the keyword name, not the value.
ENDPOINT = LangfuseEndpoint(
    base_url="https://example.invalid",
    public_key="pk",
    secret_key="sk",  # noqa: S106
)


def test_auth_header_is_basic_and_round_trips() -> None:
    encoded = ENDPOINT.auth_header.removeprefix("Basic ")
    assert base64.b64decode(encoded).decode() == "pk:sk"


def test_repr_does_not_leak_either_key() -> None:
    """The endpoint holds secrets, so it must not print them in a traceback."""
    text = repr(ENDPOINT)
    assert "pk" not in text.replace("public_key", "")
    assert "sk" not in text.replace("secret_key", "")
    assert "<redacted>" in text


def test_endpoint_from_env_prefers_the_bare_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "bare-pk")
    monkeypatch.setenv("NMDC_LANGFUSE_PUBLIC_KEY", "prefixed-pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "bare-sk")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://example.invalid/")
    endpoint = endpoint_from_env(load_dotenv_file=False)
    assert endpoint.public_key == "bare-pk"


def test_endpoint_from_env_accepts_the_nmdc_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NMDC_LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("NMDC_LANGFUSE_SECRET_KEY", "sk")
    monkeypatch.setenv("NMDC_LANGFUSE_BASE_URL", "https://example.invalid")
    assert endpoint_from_env(load_dotenv_file=False).public_key == "pk"


def test_endpoint_from_env_strips_a_trailing_slash(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://example.invalid/")
    assert endpoint_from_env(load_dotenv_file=False).base_url == "https://example.invalid"


def test_endpoint_from_env_names_every_missing_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_BASE_URL",
        "NMDC_LANGFUSE_PUBLIC_KEY",
        "NMDC_LANGFUSE_SECRET_KEY",
        "NMDC_LANGFUSE_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(LangfuseCredentialsMissing) as excinfo:
        endpoint_from_env(load_dotenv_file=False)
    message = str(excinfo.value)
    assert "LANGFUSE_PUBLIC_KEY" in message
    assert "LANGFUSE_SECRET_KEY" in message
    assert "LANGFUSE_BASE_URL" in message


def _stub_pages(monkeypatch: pytest.MonkeyPatch, pages: dict[str, list[dict]]) -> list[dict]:
    """Serve canned pages per path and record the params each call used."""
    seen: list[dict] = []

    def fake_get_json(endpoint, path, params):  # type: ignore[no-untyped-def]
        seen.append({"path": path, **params})
        data = pages.get(path, [])
        page = int(params.get("page", 1))
        chunk = data[(page - 1) * 2 : page * 2]
        return {"data": chunk, "meta": {"totalPages": max(1, (len(data) + 1) // 2)}}

    monkeypatch.setattr(reader, "_get_json", fake_get_json)
    return seen


def test_fetch_traces_walks_every_page(monkeypatch: pytest.MonkeyPatch) -> None:
    traces = [{"id": f"t{i}"} for i in range(5)]
    seen = _stub_pages(monkeypatch, {"/api/public/traces": traces})
    assert [t["id"] for t in fetch_traces(ENDPOINT)] == ["t0", "t1", "t2", "t3", "t4"]
    assert max(call["page"] for call in seen) >= 3


def test_fetch_agent_observations_filters_to_agent_type(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _stub_pages(monkeypatch, {"/api/public/observations": [{"traceId": "t1", "model": "m"}]})
    fetch_agent_observations(ENDPOINT)
    assert seen[0]["type"] == "AGENT"


def test_model_by_trace_skips_nulls_and_synthetic_placeholders() -> None:
    observations = [
        {"traceId": "t1", "model": "claude-opus-4-6"},
        {"traceId": "t2", "model": None},
        {"traceId": "t3", "model": "<synthetic>"},
        {"traceId": "t1", "model": "a-later-one-that-should-not-win"},
        {"model": "no-trace-id"},
    ]
    assert model_by_trace(observations) == {"t1": "claude-opus-4-6"}


def test_bundle_reports_model_attribution_disagreement() -> None:
    """A third of the NMDC traces name a Gemini id in metadata while serving Claude."""
    bundle = TraceBundle(
        trace={"id": "t1", "metadata": {"model": "gemini-2.5-flash"}},
        served_model="claude-opus-4-6",
    )
    assert bundle.declared_model == "gemini-2.5-flash"
    assert bundle.model_attribution_disagrees


def test_bundle_does_not_claim_disagreement_when_one_side_is_missing() -> None:
    assert not TraceBundle(trace={"id": "t1", "metadata": {}}, served_model="m").model_attribution_disagrees
    assert not TraceBundle(trace={"id": "t1", "metadata": {"model": "m"}}).model_attribution_disagrees


def test_bundle_trace_id_is_a_string_even_when_absent() -> None:
    assert TraceBundle(trace={}).trace_id == ""


def test_load_bundles_attaches_the_served_model(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_pages(
        monkeypatch,
        {
            "/api/public/traces": [{"id": "t1", "metadata": {"model": "gemini-2.5-flash"}}],
            "/api/public/observations": [{"traceId": "t1", "model": "claude-opus-4-6"}],
        },
    )
    (bundle,) = load_bundles(ENDPOINT)
    assert bundle.served_model == "claude-opus-4-6"
    assert bundle.model_attribution_disagrees


def test_paginate_stops_on_an_empty_page(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def fake_get_json(endpoint, path, params):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        return {"data": [], "meta": {}}

    monkeypatch.setattr(reader, "_get_json", fake_get_json)
    assert fetch_traces(ENDPOINT) == []
    assert calls["n"] == 1


def test_get_json_builds_the_url_and_sends_the_auth_header(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    class FakeResponse:
        def read(self):  # type: ignore[no-untyped-def]
            return json.dumps({"data": []}).encode()

        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *args):  # type: ignore[no-untyped-def]
            return False

    def fake_urlopen(request, timeout):  # type: ignore[no-untyped-def]
        captured["url"] = request.full_url
        captured["auth"] = request.headers.get("Authorization")
        return FakeResponse()

    monkeypatch.setattr(reader.urllib.request, "urlopen", fake_urlopen)
    reader._get_json(ENDPOINT, "/api/public/traces", {"page": 1, "limit": 100, "skip": None})
    assert captured["url"].startswith("https://example.invalid/api/public/traces?")
    assert "skip" not in captured["url"]
    assert captured["auth"] == ENDPOINT.auth_header
