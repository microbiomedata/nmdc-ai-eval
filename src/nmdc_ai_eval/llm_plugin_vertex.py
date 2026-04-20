"""llm plugin for Vertex AI models (Gemini and Claude).

Registers ``vertex/gemini-*`` and ``vertex/claude-*`` model IDs so they can be
used anywhere llm models are accepted — including llm-matrix suite evals.

Auth: reads ``GOOGLE_APPLICATION_CREDENTIALS`` and ``VERTEX_PROJECT_ID`` from
the environment (loaded from ``.env`` by dotenv). The same SA and project ID
used by ``just verify-auth`` work here.

Dispatch:
  vertex/gemini-* → google.genai.Client(vertexai=True).models.generate_content
  vertex/claude-* → anthropic.AnthropicVertex.messages.create

These correspond to the two Vertex API endpoints (:generateContent for Gemini,
:rawPredict / :streamRawPredict for Claude). They cannot be swapped — Gemini
names will fail through the Anthropic path and vice versa.

Model names: use bare Vertex model IDs (e.g. ``gemini-2.5-flash``,
``claude-haiku-4-5``, ``claude-sonnet-4-5@20250929``). The ``vertex/`` prefix
is stripped before dispatch. See ``probe_vertex_garden.py`` to discover which
names are reachable on your project.

Usage in a suite YAML matrix:
    model:
      - vertex/gemini-2.5-flash
      - vertex/claude-haiku-4-5

Usage in pilot command:
    just pilot-env-triad 5 "vertex/gemini-2.5-flash,vertex/claude-haiku-4-5"

Note: this plugin is loaded via llm's local plugin mechanism. It is registered
in pyproject.toml under [project.entry-points."llm"] — no separate install
step needed beyond ``uv sync``.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Iterator

import llm
from dotenv import load_dotenv
from pydantic import Field

if TYPE_CHECKING:
    from llm import Conversation, Prompt, Response

load_dotenv()

_DEFAULT_VERTEX_REGION = os.environ.get("CLOUD_ML_REGION", os.environ.get("GEMINI_REGION", "us-east5"))
_VERTEX_PROJECT_ID = os.environ.get("VERTEX_PROJECT_ID")
_CREDS_FILE = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")


class _VertexOptions(llm.Options):
    """Shared options for Vertex models.

    Declared so llm-matrix (and other callers) can pass ``temperature``
    without pydantic rejecting it as an extra input. Keep this minimal —
    add options only when a caller actually needs to set them.
    """

    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1)


def _get_vertex_project() -> str:
    if _VERTEX_PROJECT_ID:
        return _VERTEX_PROJECT_ID
    raise ValueError("VERTEX_PROJECT_ID is not set. Add it to your .env file. See docs/auth.md for setup instructions.")


class VertexGeminiModel(llm.Model):
    """Gemini model on Vertex AI via google.genai (generateContent endpoint)."""

    needs_key = None  # auth via service-account file, not API key
    Options = _VertexOptions  # type: ignore[assignment]

    def __init__(self, model_id: str, vertex_model_name: str) -> None:
        self.model_id = model_id
        self.vertex_model_name = vertex_model_name

    def execute(
        self,
        prompt: "Prompt",
        stream: bool,
        response: "Response",
        conversation: "Conversation | None",
    ) -> Iterator[str]:
        from google import genai
        from google.genai import types as genai_types
        from google.oauth2 import service_account

        project = _get_vertex_project()
        scopes = ["https://www.googleapis.com/auth/cloud-platform"]
        if _CREDS_FILE:
            credentials = service_account.Credentials.from_service_account_file(  # type: ignore[no-untyped-call]
                _CREDS_FILE, scopes=scopes
            )
        else:
            import google.auth

            credentials, _ = google.auth.default(scopes=scopes)

        client = genai.Client(
            vertexai=True,
            project=project,
            location=_DEFAULT_VERTEX_REGION,
            credentials=credentials,
        )
        system = prompt.system or ""
        user_text = prompt.prompt or ""
        max_tokens = getattr(getattr(prompt, "options", None), "max_tokens", None) or 4096
        temperature = getattr(getattr(prompt, "options", None), "temperature", None)
        config = genai_types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
            system_instruction=system if system else None,
        )
        api_response = client.models.generate_content(
            model=self.vertex_model_name,
            contents=user_text,
            config=config,
        )
        text = api_response.text or ""
        yield text
        usage = getattr(api_response, "usage_metadata", None)
        if usage:
            response.set_usage(
                input=getattr(usage, "prompt_token_count", None),
                output=getattr(usage, "candidates_token_count", None),
            )


class VertexClaudeModel(llm.Model):
    """Claude model on Vertex AI via AnthropicVertex (rawPredict endpoint)."""

    needs_key = None  # auth via service-account file, not API key
    Options = _VertexOptions  # type: ignore[assignment]

    def __init__(self, model_id: str, vertex_model_name: str) -> None:
        self.model_id = model_id
        self.vertex_model_name = vertex_model_name

    def execute(
        self,
        prompt: "Prompt",
        stream: bool,
        response: "Response",
        conversation: "Conversation | None",
    ) -> Iterator[str]:
        from anthropic import AnthropicVertex

        project = _get_vertex_project()
        client = AnthropicVertex(region=_DEFAULT_VERTEX_REGION, project_id=project)
        system = prompt.system or ""
        user_text = prompt.prompt or ""
        max_tokens = getattr(getattr(prompt, "options", None), "max_tokens", None) or 4096
        messages = [{"role": "user", "content": user_text}]
        kwargs: dict[str, object] = {
            "model": self.vertex_model_name,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        temperature = getattr(getattr(prompt, "options", None), "temperature", None)
        if temperature is not None:
            kwargs["temperature"] = temperature

        message = client.messages.create(**kwargs)  # type: ignore[call-overload]
        text = "".join(block.text for block in message.content if hasattr(block, "text"))
        yield text
        if message.usage:
            response.set_usage(
                input=message.usage.input_tokens,
                output=message.usage.output_tokens,
            )


# Models to register. Add more as needed — run `just probe-vertex-garden`
# to discover which names are currently reachable on your project.
_GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
]

_CLAUDE_MODELS = [
    "claude-haiku-4-5",
    "claude-sonnet-4-5@20250929",
    "claude-opus-4-1@20250514",
]


@llm.hookimpl
def register_models(register: object) -> None:
    # llm passes a callable as `register`; the type stub says `object` so
    # we use cast to satisfy mypy without a bare ignore.
    from typing import Any, Callable, cast

    _reg: Callable[..., Any] = cast(Callable[..., Any], register)
    for name in _GEMINI_MODELS:
        _reg(VertexGeminiModel(f"vertex/{name}", name))
    for name in _CLAUDE_MODELS:
        _reg(VertexClaudeModel(f"vertex/{name}", name))
