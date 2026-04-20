#!/usr/bin/env python3
"""Probe Vertex AI Model Garden: which model names does our SA+project allow?

For each candidate model, attempts a trivial LLM call through
LLMClient(access_provider="gcp") and records the outcome. Groups results
by provider and prints a summary table. Exceptions are classified into
coarse buckets (not-found, access-denied, quota, other) so the output
tells you what to fix rather than dumping a raw traceback per model.

Live LLM calls — each success costs a few cents at most. Requires GCP
credentials (GOOGLE_APPLICATION_CREDENTIALS / VERTEX_PROJECT_ID in .env).

Usage:
    just probe-vertex-garden
    uv run python scripts/probe_vertex_garden.py
"""

import sys
from collections import defaultdict
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

from nmdc_metadata_suggestor_ai_tool.llm_client import LLMClient  # noqa: E402

PROMPT = "Reply with exactly one word: hello."

# (provider, candidate_model_name). The same provider may appear multiple
# times with different name formats — Vertex accepts bare names for some,
# publisher-prefixed paths for others.
CANDIDATES: list[tuple[str, str]] = [
    ("google", "gemini-2.5-flash"),
    ("anthropic", "claude-haiku-4-5"),
    ("anthropic", "claude-3-haiku@20240307"),
    ("anthropic", "publishers/anthropic/models/claude-3-haiku@20240307"),
    ("anthropic", "publishers/anthropic/models/claude-haiku-4-5"),
    ("openai", "gpt-4o-mini"),
    ("openai", "publishers/openai/models/gpt-4o-mini"),
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

    String-matching on the message is crude but LLMClient wraps several
    underlying SDKs (google, anthropic-vertex, openai) so there is no
    single exception hierarchy to catch by type.
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


def probe_one(provider: str, model: str) -> Result:
    try:
        client = LLMClient(access_provider="gcp", model=model)
        client.add_message(text=PROMPT)
        response = client.generate(max_tokens=32)
        return Result(provider, model, "ok", response.strip()[:40])
    except Exception as e:
        status, detail = _classify_error(e)
        return Result(provider, model, status, detail)


def main() -> int:
    print(f"Probing Vertex Model Garden: {len(CANDIDATES)} candidates")
    print("Each success = one live LLM call (a few cents at most)\n")

    results: list[Result] = []
    for provider, model in CANDIDATES:
        r = probe_one(provider, model)
        marker = "OK  " if r.status == "ok" else "FAIL"
        print(f"  {marker}  [{r.status:14s}] {model}")
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
