#!/usr/bin/env python3
"""Verify API auth works for all configured providers.

Tests llm-plugin models (personal API keys) and GCP/PNNL pipeline
credentials from .env.
"""

import os
import sys
from pathlib import Path

import llm
import yaml
from dotenv import load_dotenv

MODELS_YAML = Path(__file__).parent.parent / "datasets" / "models.yaml"


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

        try:
            m = llm.get_model(name)
            r = m.prompt("Reply with only: OK", temperature=0)
            text = str(r).strip()[:20]
            print(f"  OK    {name:45s} -> {text}")
        except Exception as e:
            err = str(e)[:80]
            print(f"  FAIL  {name:45s} -> {err}")
            failures.append(name)

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
        err = str(e)[:80]
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
        err = str(e)[:80]
        print(f"  FAIL  PNNL AI Incubator                            -> {err}")
        failures.append("pnnl")

    return failures


def main() -> int:
    load_dotenv()

    print("llm plugin providers (personal API keys):")
    failures = test_llm_providers()

    print("\nPipeline providers (.env credentials):")
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
