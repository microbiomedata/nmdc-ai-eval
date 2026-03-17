#!/usr/bin/env python3
"""Verify API auth works for all providers by sending one cheap prompt each."""

import sys
from pathlib import Path

import llm
import yaml

MODELS_YAML = Path(__file__).parent.parent / "datasets" / "models.yaml"


def main() -> int:
    with open(MODELS_YAML) as f:
        models: list[str] = yaml.safe_load(f)["models"]

    # Test one model per provider (cheapest first)
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

    if failures:
        print(f"\n{len(failures)} provider(s) failed. Fix keys with: uv run llm keys set <provider>")
        return 1
    print("\nAll providers authenticated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
