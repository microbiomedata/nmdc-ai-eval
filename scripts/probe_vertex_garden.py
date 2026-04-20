#!/usr/bin/env python3
"""Probe Vertex AI Model Garden: which model names does our SA+project allow?

Dispatches per provider to the API Vertex actually requires:

- **Google/Gemini** candidates go through the suggestor's
  ``LLMClient(access_provider="gcp")``, which uses ``google-genai`` and
  calls ``models.generate_content``. This is the Gemini-only endpoint.
- **Anthropic** candidates go through the ``anthropic`` SDK's
  ``AnthropicVertex`` client, which uses the ``rawPredict`` endpoint.
  Claude-on-Vertex cannot be reached through generateContent — a 400
  "not supported" from that path is the tell that the dispatcher, not
  the model, is wrong.
- **OpenAI/Meta** candidates also go through ``LLMClient`` today; they
  are expected to FAIL because the suggestor has no dispatch path for
  them yet. Those failures should be read as "dispatcher missing," not
  "not enabled on Model Garden."

See microbiomedata/nmdc-ai-eval#46 for context. Adding a
``gcp-anthropic`` access provider to LLMClient (pending PR on the
suggestor repo) is the long-term fix — this probe would then route
Anthropic candidates through that uniform interface instead of the
direct ``AnthropicVertex`` call here.

Live LLM calls — each success costs a few cents at most. Requires GCP
credentials (GOOGLE_APPLICATION_CREDENTIALS / VERTEX_PROJECT_ID in .env).

Usage:
    just probe-vertex-garden
    uv run python scripts/probe_vertex_garden.py
"""

import os
import sys
from collections import defaultdict
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

from anthropic import AnthropicVertex  # noqa: E402
from nmdc_metadata_suggestor_ai_tool.llm_client import LLMClient  # noqa: E402

PROMPT = "Reply with exactly one word: hello."

DEFAULT_VERTEX_REGION = os.environ.get("CLOUD_ML_REGION", "us-east5")
VERTEX_PROJECT_ID = os.environ.get("VERTEX_PROJECT_ID")

# (provider, candidate_model_name). Anthropic candidates on Vertex use
# @YYYYMMDD version stamps; the SDK resolves bare family names in some
# cases. The probe tries both forms to reveal what's actually accepted.
CANDIDATES: list[tuple[str, str]] = [
    ("google", "gemini-2.5-flash"),
    ("anthropic", "claude-haiku-4-5@20250401"),
    ("anthropic", "claude-haiku-4-5"),
    ("anthropic", "claude-sonnet-4-5@20250929"),
    ("anthropic", "claude-3-5-haiku@20241022"),
    ("openai", "gpt-4o-mini"),
    ("meta", "publishers/meta/models/llama-3.1-8b-instruct-maas"),
]


@dataclass
class Result:
    provider: str
    model: str
    status: str  # "ok" | "not_found" | "access_denied" | "quota" | "other"
    detail: str


def _classify_error(e: BaseException) -> tuple[str, str]:
    """Map an exception to (status_label, short_detail).

    String-matching on the message is crude but the probe touches
    multiple SDKs (google-genai, anthropic-vertex, openai) with
    different exception hierarchies.
    """
    text = f"{type(e).__name__}: {e}"
    lower = str(e).lower()
    if any(s in lower for s in ("not found", "404", "no such model", "unrecognized model")):
        return "not_found", text[:200]
    if any(s in lower for s in ("permission", "access denied", "403", "not allowed", "not enabled", "publisher")):
        return "access_denied", text[:200]
    if any(s in lower for s in ("quota", "rate limit", "429", "exhausted")):
        return "quota", text[:200]
    return "other", text[:200]


def _probe_via_anthropic_vertex(model: str) -> str:
    if not VERTEX_PROJECT_ID:
        raise RuntimeError("VERTEX_PROJECT_ID is not set in .env")
    client = AnthropicVertex(region=DEFAULT_VERTEX_REGION, project_id=VERTEX_PROJECT_ID)
    message = client.messages.create(
        model=model,
        max_tokens=32,
        messages=[{"role": "user", "content": PROMPT}],
    )
    return "".join(block.text for block in message.content if hasattr(block, "text")).strip()[:40]


def _probe_via_llm_client(model: str) -> str:
    client = LLMClient(access_provider="gcp", model=model)
    client.add_message(text=PROMPT)
    response = client.generate(max_tokens=32)
    return str(response).strip()[:40]


def probe_one(provider: str, model: str) -> Result:
    try:
        if provider == "anthropic":
            text = _probe_via_anthropic_vertex(model)
        else:
            text = _probe_via_llm_client(model)
        return Result(provider, model, "ok", text)
    except Exception as e:
        status, detail = _classify_error(e)
        return Result(provider, model, status, detail)


def main() -> int:
    print(f"Probing Vertex Model Garden: {len(CANDIDATES)} candidates")
    print("Dispatch: google/openai/meta via LLMClient (generateContent);")
    print("          anthropic via AnthropicVertex (rawPredict).")
    print("Each success = one live LLM call (a few cents at most)\n")

    results: list[Result] = []
    for provider, model in CANDIDATES:
        r = probe_one(provider, model)
        marker = "OK  " if r.status == "ok" else "FAIL"
        print(f"  {marker}  [{r.status:14s}] {provider:10s} {model}")
        if r.status != "ok":
            print(f"          {r.detail}")
        results.append(r)

    by_provider: dict[str, list[Result]] = defaultdict(list)
    for r in results:
        by_provider[r.provider].append(r)

    print("\nSummary")
    print(f"  {'Provider':<12} {'Tried':>6} {'OK':>4}   Working names")
    for provider, rs in sorted(by_provider.items()):
        ok_names = [r.model for r in rs if r.status == "ok"]
        print(f"  {provider:<12} {len(rs):>6} {len(ok_names):>4}   {', '.join(ok_names) or '-'}")

    n_ok = sum(1 for r in results if r.status == "ok")
    print(f"\n{n_ok}/{len(results)} candidate names accepted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
