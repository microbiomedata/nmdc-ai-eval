#!/usr/bin/env python3
"""Probe cost tiers across OpenAI / Anthropic / Google with one command.

Iterates the requested tier from ``datasets/models.yaml`` and sends one
trivial prompt to each model. Reports OK/FAIL, sample response, token
counts, elapsed time, and approximate per-call cost (from the
``pricing`` table in models.yaml).

Channels:

- ``--channel=llm`` (default): each model hits its provider's direct
  API via the ``llm`` library. Uses the llm key store or env vars.
  This is the normal eval path; works today with personal keys.
- ``--channel=cborg``: all models route through CBORG's OpenAI-
  compatible proxy. Requires ``CBORG_API_KEY`` + ``CBORG_BASE_URL``.
  Model names are passed as-is; run ``--list-cborg-models`` first
  to see what CBORG actually accepts (the llm-plugin-style names like
  ``anthropic/claude-sonnet-4-6`` are rejected).
- ``--channel=vertex``: dispatches intelligently over Vertex. Claude
  models go through ``AnthropicVertex`` (rawPredict), Gemini models
  go through ``google-genai`` (generateContent). OpenAI models fail
  with a clear message because GPT is not on Model Garden. Requires
  ``GOOGLE_APPLICATION_CREDENTIALS`` + ``VERTEX_PROJECT_ID``.

All channels cap output at 32 tokens to keep costs bounded — some
models (notably gemini-2.5-pro) ignore "reply one word" and emit 100+
tokens without a cap.

Live LLM calls — total cost for tier=full is typically well under $0.01.

Usage::

    just probe-tiers                         # tier=full, channel=llm
    just probe-tiers --channel=cborg
    just probe-tiers --channel=vertex
    just probe-tiers --tier=cheap
    just probe-tiers --list-cborg-models     # print CBORG allowlist and exit
"""

import argparse
import contextlib
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

MODELS_YAML = Path(__file__).parent.parent / "datasets" / "models.yaml"
PROMPT = "Reply with exactly one word: hello."
MAX_OUTPUT_TOKENS = 32
DEFAULT_VERTEX_REGION = os.environ.get("CLOUD_ML_REGION", "us-east5")


@dataclass
class Result:
    model: str
    provider: str
    status: str  # "ok" | "fail"
    sample: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    elapsed_s: float = 0.0
    error: str = ""


def _provider(model: str) -> str:
    """Lab/provider inferred from the llm-style model name."""
    if "/" in model:
        return model.split("/", 1)[0]
    return "openai"


def _price_for(pricing: dict[str, list[float]], model: str) -> tuple[float, float]:
    """Per-million input/output price (USD). Prefix-matching so that
    versioned names like 'claude-haiku-4-5-20251001' resolve against
    a pricing entry for 'claude-haiku-4-5'."""
    key = model.split("/", 1)[1] if "/" in model else model
    for candidate in (key, key.rsplit("-", 1)[0], key.rsplit("-", 2)[0]):
        if candidate in pricing:
            row = pricing[candidate]
            return float(row[0]), float(row[1])
    return 0.0, 0.0


def _cost(price_row: tuple[float, float], input_tokens: int, output_tokens: int) -> float:
    inp, out = price_row
    return (input_tokens * inp + output_tokens * out) / 1_000_000


def probe_via_llm(model: str, pricing: dict[str, list[float]]) -> Result:
    import llm

    start = time.time()
    try:
        m = llm.get_model(model)
        # max_tokens kwarg is accepted by the major plugins; Gemini
        # plugin maps it to max_output_tokens internally.
        response = m.prompt(PROMPT, temperature=0, max_tokens=MAX_OUTPUT_TOKENS)
        text = str(response).strip()[:40]
        elapsed = time.time() - start
        in_tokens = out_tokens = 0
        if hasattr(response, "usage"):
            with contextlib.suppress(Exception):
                usage = response.usage()
                in_tokens = getattr(usage, "input", 0) or 0
                out_tokens = getattr(usage, "output", 0) or 0
        return Result(
            model=model,
            provider=_provider(model),
            status="ok",
            sample=text,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            cost_usd=_cost(_price_for(pricing, model), in_tokens, out_tokens),
            elapsed_s=elapsed,
        )
    except Exception as e:
        return Result(
            model=model,
            provider=_provider(model),
            status="fail",
            error=str(e)[:200],
            elapsed_s=time.time() - start,
        )


def probe_via_cborg(model: str, pricing: dict[str, list[float]]) -> Result:
    from openai import OpenAI

    api_key = os.environ.get("CBORG_API_KEY")
    base_url = os.environ.get("CBORG_BASE_URL")
    if not api_key or not base_url:
        return Result(
            model=model,
            provider=_provider(model),
            status="fail",
            error="CBORG_API_KEY or CBORG_BASE_URL not set in .env",
        )
    client = OpenAI(api_key=api_key, base_url=base_url)
    start = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": PROMPT}],
            temperature=0,
            max_tokens=MAX_OUTPUT_TOKENS,
        )
        text = (response.choices[0].message.content or "").strip()[:40]
        elapsed = time.time() - start
        in_tokens = response.usage.prompt_tokens if response.usage else 0
        out_tokens = response.usage.completion_tokens if response.usage else 0
        return Result(
            model=model,
            provider=_provider(model),
            status="ok",
            sample=text,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            cost_usd=_cost(_price_for(pricing, model), in_tokens, out_tokens),
            elapsed_s=elapsed,
        )
    except Exception as e:
        return Result(
            model=model,
            provider=_provider(model),
            status="fail",
            error=str(e)[:200],
            elapsed_s=time.time() - start,
        )


def probe_via_vertex(model: str, pricing: dict[str, list[float]]) -> Result:
    """Dispatch to the right Vertex endpoint by model prefix.

    - anthropic/* -> AnthropicVertex (rawPredict)
    - gemini/* -> google-genai (generateContent)
    - gpt-* -> FAIL (not on Model Garden)
    """
    provider = _provider(model)
    project = os.environ.get("VERTEX_PROJECT_ID")
    if not project:
        return Result(
            model=model,
            provider=provider,
            status="fail",
            error="VERTEX_PROJECT_ID not set in .env",
        )

    bare_name = model.split("/", 1)[1] if "/" in model else model
    start = time.time()

    if provider == "anthropic":
        try:
            from anthropic import AnthropicVertex

            anthropic_client = AnthropicVertex(region=DEFAULT_VERTEX_REGION, project_id=project)
            message = anthropic_client.messages.create(
                model=bare_name,
                max_tokens=MAX_OUTPUT_TOKENS,
                messages=[{"role": "user", "content": PROMPT}],
            )
            text = "".join(block.text for block in message.content if hasattr(block, "text")).strip()[:40]
            elapsed = time.time() - start
            in_tokens = getattr(message.usage, "input_tokens", 0) or 0
            out_tokens = getattr(message.usage, "output_tokens", 0) or 0
            return Result(
                model=model,
                provider=provider,
                status="ok",
                sample=text,
                input_tokens=in_tokens,
                output_tokens=out_tokens,
                cost_usd=_cost(_price_for(pricing, model), in_tokens, out_tokens),
                elapsed_s=elapsed,
            )
        except Exception as e:
            return Result(
                model=model,
                provider=provider,
                status="fail",
                error=str(e)[:200],
                elapsed_s=time.time() - start,
            )

    if provider == "gemini":
        try:
            from google import genai
            from google.genai import types as genai_types

            gemini_client = genai.Client(vertexai=True, project=project, location=DEFAULT_VERTEX_REGION)
            config = genai_types.GenerateContentConfig(
                max_output_tokens=MAX_OUTPUT_TOKENS,
                temperature=0,
            )
            response = gemini_client.models.generate_content(
                model=bare_name,
                contents=PROMPT,
                config=config,
            )
            text = (response.text or "").strip()[:40]
            elapsed = time.time() - start
            usage = getattr(response, "usage_metadata", None)
            in_tokens = getattr(usage, "prompt_token_count", 0) or 0
            out_tokens = getattr(usage, "candidates_token_count", 0) or 0
            return Result(
                model=model,
                provider=provider,
                status="ok",
                sample=text,
                input_tokens=in_tokens,
                output_tokens=out_tokens,
                cost_usd=_cost(_price_for(pricing, model), in_tokens, out_tokens),
                elapsed_s=elapsed,
            )
        except Exception as e:
            return Result(
                model=model,
                provider=provider,
                status="fail",
                error=str(e)[:200],
                elapsed_s=time.time() - start,
            )

    return Result(
        model=model,
        provider=provider,
        status="fail",
        error=f"{provider} not reachable via Vertex Model Garden",
        elapsed_s=0.0,
    )


def list_cborg_models() -> int:
    """Print CBORG's allowlist (GET {CBORG_BASE_URL}/v1/models)."""
    from openai import OpenAI

    api_key = os.environ.get("CBORG_API_KEY")
    base_url = os.environ.get("CBORG_BASE_URL")
    if not api_key or not base_url:
        print("CBORG_API_KEY or CBORG_BASE_URL not set in .env", file=sys.stderr)
        return 2
    client = OpenAI(api_key=api_key, base_url=base_url)
    models = sorted(m.id for m in client.models.list().data)
    print(f"CBORG exposes {len(models)} models:\n")
    for name in models:
        print(f"  {name}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe cost tiers across providers.")
    parser.add_argument(
        "--channel",
        choices=["llm", "cborg", "vertex"],
        default="llm",
        help="llm = direct APIs; cborg = LBNL proxy; vertex = GCP Vertex (Claude+Gemini)",
    )
    parser.add_argument(
        "--tier",
        default="full",
        help="tiers.<name> key in datasets/models.yaml (default: full)",
    )
    parser.add_argument(
        "--list-cborg-models",
        action="store_true",
        help="Print CBORG's model allowlist and exit",
    )
    args = parser.parse_args()

    if args.list_cborg_models:
        return list_cborg_models()

    with open(MODELS_YAML) as f:
        cfg = yaml.safe_load(f)
    tiers = cfg.get("tiers", {})
    if args.tier not in tiers:
        print(f"Unknown tier {args.tier!r}. Available: {list(tiers)}", file=sys.stderr)
        return 2
    models: list[str] = tiers[args.tier]
    pricing: dict[str, list[float]] = cfg.get("pricing", {})

    probes = {"llm": probe_via_llm, "cborg": probe_via_cborg, "vertex": probe_via_vertex}
    probe = probes[args.channel]

    print(f"Probing {len(models)} models via {args.channel!r} channel (tier={args.tier})")
    print(f"Prompt: {PROMPT!r}  |  max_output_tokens={MAX_OUTPUT_TOKENS}\n")

    results: list[Result] = []
    for name in models:
        r = probe(name, pricing)
        marker = "OK  " if r.status == "ok" else "FAIL"
        print(
            f"  {marker}  {r.provider:10s} {name:42s} "
            f"{r.elapsed_s:5.2f}s  in={r.input_tokens:4d} out={r.output_tokens:4d}  "
            f"${r.cost_usd:.5f}"
        )
        if r.status == "ok":
            print(f"          -> {r.sample!r}")
        else:
            print(f"          {r.error}")
        results.append(r)

    by_provider: dict[str, list[Result]] = defaultdict(list)
    for r in results:
        by_provider[r.provider].append(r)

    print("\nSummary by provider")
    print(f"  {'Provider':<12} {'Tried':>6} {'OK':>4}  {'Cost (USD)':>11}")
    for provider, rs in sorted(by_provider.items()):
        ok_count = sum(1 for r in rs if r.status == "ok")
        total = sum(r.cost_usd for r in rs)
        print(f"  {provider:<12} {len(rs):>6} {ok_count:>4}  ${total:>10.5f}")

    total_cost = sum(r.cost_usd for r in results)
    total_ok = sum(1 for r in results if r.status == "ok")
    print(f"\n{total_ok}/{len(results)} OK   Total cost: ${total_cost:.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
