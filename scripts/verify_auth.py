#!/usr/bin/env python3
"""Verify API auth works for all configured providers.

Tests llm-plugin models (personal API keys), CBORG, and GCP/PNNL
pipeline credentials from .env. Each provider is checked with one
real LLM call. Missing credentials produce SKIP, not FAIL.
"""

import contextlib
import json
import os
import sys
from pathlib import Path

import llm
import yaml
from dotenv import load_dotenv

MODELS_YAML = Path(__file__).parent.parent / "datasets" / "models.yaml"
LLM_KEYS_PATH = Path.home() / ".config" / "io.datasette.llm" / "keys.json"

# Env vars the llm-* plugins typically read when the key store is empty.
_LLM_ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


def _llm_key_source(provider: str) -> str:
    """Best-effort report of which source llm would use for this provider's key.

    The llm key store takes priority over env vars, so this mirrors llm's
    resolution order. Returns 'llm-store', 'env', or 'none'.
    """
    with contextlib.suppress(OSError, json.JSONDecodeError):
        if LLM_KEYS_PATH.exists():
            with open(LLM_KEYS_PATH) as f:
                keys = json.load(f)
            if keys.get(provider):
                return "llm-store"
    env_var = _LLM_ENV_VARS.get(provider)
    if env_var and os.environ.get(env_var):
        return "env"
    return "none"


def test_llm_providers() -> list[str]:
    """Test one model per llm plugin provider. Returns list of failures."""
    with open(MODELS_YAML) as f:
        models: list[str] = yaml.safe_load(f)["models"]

    seen_providers: set[str] = set()
    failures: list[str] = []

    for name in models:
        provider = name.split("/")[0] if "/" in name else "openai"
        if provider in seen_providers:
            continue
        seen_providers.add(provider)

        source = _llm_key_source(provider)
        if source == "none":
            print(f"  SKIP  {name:45s} [key: {source:9s}] -> no credentials (llm store or env var)")
            continue
        try:
            m = llm.get_model(name)
            r = m.prompt("Reply with only: OK", temperature=0)
            text = str(r).strip()[:20]
            print(f"  OK    {name:45s} [key: {source:9s}] -> {text}")
        except Exception as e:
            err = str(e)[:200]
            print(f"  FAIL  {name:45s} [key: {source:9s}] -> {err}")
            failures.append(name)

    return failures


def test_cborg_credentials() -> list[str]:
    """Test CBORG (LBNL) credentials from .env. Returns list of failures."""
    failures: list[str] = []
    key = os.environ.get("CBORG_API_KEY")
    url = os.environ.get("CBORG_BASE_URL")

    if not key or not url:
        print("  SKIP  CBORG                                        -> no credentials in .env")
        return []

    model = os.environ.get("CBORG_TEST_MODEL", "gpt-4o-mini")
    try:
        from openai import OpenAI

        client = OpenAI(api_key=key, base_url=url)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with only: OK"}],
            temperature=0,
        )
        text = (response.choices[0].message.content or "").strip()[:20]
        print(f"  OK    CBORG ({model:37s}) -> {text}")
    except Exception as e:
        err = str(e)[:200]
        print(f"  FAIL  CBORG ({model:37s}) -> {err}")
        failures.append("cborg")

    return failures


def test_gcp_credentials() -> list[str]:
    """Test GCP Vertex AI credentials from .env. Returns list of failures."""
    failures: list[str] = []
    creds_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    project = os.environ.get("VERTEX_PROJECT_ID")

    if not creds_file and not project:
        print("  SKIP  GCP Vertex AI                                -> no credentials in .env")
        return []

    if creds_file and not Path(creds_file).exists():
        print(f"  FAIL  GCP Vertex AI  -> creds file not found: {creds_file}")
        return ["gcp"]

    try:
        from nmdc_metadata_suggestor_ai_tool.llm_client import LLMClient

        client = LLMClient(access_provider="gcp")
        client.add_message(text="Reply with only: OK")
        response = client.generate()
        text = response.strip()[:20]
        print(f"  OK    GCP Vertex ({client.model:35s}) -> {text}")
    except Exception as e:
        err = str(e)[:200]
        print(f"  FAIL  GCP Vertex AI                                -> {err}")
        failures.append("gcp")

    return failures


def test_pnnl_credentials() -> list[str]:
    """Test PNNL AI Incubator credentials from .env. Returns list of failures."""
    failures: list[str] = []
    key = os.environ.get("AI_INCUBATOR_KEY")
    url = os.environ.get("AI_INCUBATOR_BASE_URL")

    if not key or not url:
        print("  SKIP  PNNL AI Incubator                            -> no credentials in .env")
        return []

    try:
        from nmdc_metadata_suggestor_ai_tool.llm_client import LLMClient

        client = LLMClient(access_provider="pnnl")
        client.add_message(text="Reply with only: OK")
        response = client.generate()
        text = response.strip()[:20]
        print(f"  OK    PNNL ({client.model:37s}) -> {text}")
    except Exception as e:
        err = str(e)[:200]
        print(f"  FAIL  PNNL AI Incubator                            -> {err}")
        failures.append("pnnl")

    return failures


def main() -> int:
    load_dotenv()

    print("llm plugin providers (personal API keys):")
    failures = test_llm_providers()

    print("\nInstitutional providers (.env credentials):")
    failures += test_cborg_credentials()
    failures += test_gcp_credentials()
    failures += test_pnnl_credentials()

    if failures:
        print(f"\n{len(failures)} provider(s) failed: {failures}")
        print("This is informational — not all providers are required.")
        print("You only need credentials for the backends you intend to use.")
    else:
        print("\nAll configured providers authenticated.")
    return 0  # Always exit 0 — missing providers are expected


if __name__ == "__main__":
    sys.exit(main())
