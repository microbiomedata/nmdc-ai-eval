#!/usr/bin/env python3
"""Run the real suggestor pipeline against ground truth and score results.

Supports two backends:
  - pipeline: uses the suggestor's LLMClient (GCP Vertex or PNNL)
  - llm: uses the llm library plugin ecosystem (OpenAI, Anthropic, Gemini, etc.)

Both backends receive the SAME prompt — the suggestor's production prompt
construction including DOI waterfall and PDF ingestion. The only difference
is which LLM API processes it. This makes results directly comparable.

Usage:
    # Single model, pipeline backend (GCP/PNNL credentials required)
    python run_pipeline_eval.py --provider gcp
    python run_pipeline_eval.py --provider gcp --model gemini-2.5-pro

    # Single model, llm backend (personal API keys)
    python run_pipeline_eval.py --backend llm --model gpt-4o
    python run_pipeline_eval.py --backend llm --model anthropic/claude-sonnet-4-5

    # Sweep all available models across both backends
    python run_pipeline_eval.py --sweep

Output: pipeline-results/{model}_{timestamp}.yaml per model, with
        elapsed_seconds, input_tokens, output_tokens, est_cost_usd.

This is a Task 1 (Field Guidance) eval ONLY. It scores which slots the
model recommends, not the values it suggests for those slots. The `value`
field from the suggestor's response is intentionally discarded — value
prediction is Task 2 (Metadata Completion), a separate evaluation.
"""

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import yaml
from pymongo import MongoClient

from nmdc_ai_eval.llm_adapter import LLMLibraryAdapter
from nmdc_ai_eval.pricing import estimate_cost
from nmdc_ai_eval.scoring import score_sets

HERE = Path(__file__).parent
GROUND_TRUTH = HERE / "ground_truth.yaml"
RESULTS_DIR = HERE / "pipeline-results"

# Fields excluded from precision scoring by default.
ENV_TRIAD = {"env_broad_scale", "env_local_scale", "env_medium"}


def load_ground_truth() -> list[dict[str, Any]]:
    with open(GROUND_TRUTH) as f:
        data = yaml.safe_load(f)
        return list(data["submissions"])  # type: ignore[index]


# ---------------------------------------------------------------------------
# Backend: suggestor pipeline (GCP / PNNL)
# ---------------------------------------------------------------------------


def instrument_for_tokens(llm_client: Any) -> dict[str, int | None]:
    """Monkey-patch an LLMClient to capture token usage from API responses."""
    usage: dict[str, int | None] = {"input_tokens": None, "output_tokens": None}

    if llm_client.access_provider == "gcp":
        original = llm_client.client.models.generate_content

        def patched_gcp(*args: Any, **kwargs: Any) -> Any:
            response = original(*args, **kwargs)
            meta = getattr(response, "usage_metadata", None)
            if meta is not None:
                in_tok = getattr(meta, "prompt_token_count", None)
                out_tok = getattr(meta, "candidates_token_count", None)
                if in_tok is not None:
                    usage["input_tokens"] = (usage["input_tokens"] or 0) + in_tok
                if out_tok is not None:
                    usage["output_tokens"] = (usage["output_tokens"] or 0) + out_tok
            return response

        llm_client.client.models.generate_content = patched_gcp

    elif llm_client.access_provider == "pnnl":
        original = llm_client.client.responses.create

        def patched_pnnl(*args: Any, **kwargs: Any) -> Any:
            response = original(*args, **kwargs)
            u = getattr(response, "usage", None)
            if u is not None:
                in_tok = getattr(u, "input_tokens", None)
                out_tok = getattr(u, "output_tokens", None)
                if in_tok is not None:
                    usage["input_tokens"] = (usage["input_tokens"] or 0) + in_tok
                if out_tok is not None:
                    usage["output_tokens"] = (usage["output_tokens"] or 0) + out_tok
            return response

        llm_client.client.responses.create = patched_pnnl

    return usage


# ---------------------------------------------------------------------------
# Context capture — intercepts add_message() to record what the LLM saw
# ---------------------------------------------------------------------------


def _wrap_for_context_capture(llm_client: Any) -> list[str]:
    """Wrap an LLMClient's add_message() to capture input context.

    Returns a mutable list that accumulates text messages as the pipeline
    adds them. Works with both LLMClient and LLMLibraryAdapter.
    """
    captured: list[str] = []
    original_add = llm_client.add_message

    def capturing_add(text: str = "", pdf_files: Any = None) -> None:
        if text:
            captured.append(text)
        original_add(text=text, pdf_files=pdf_files)

    llm_client.add_message = capturing_add
    return captured


# ---------------------------------------------------------------------------
# Verification — ask the model to cite evidence or drop recommendations
# ---------------------------------------------------------------------------

VERIFY_PROMPT = """\
You previously recommended metadata fields for a scientific submission.
Now review each recommendation against the input context below.

For each recommended field, classify it and decide whether to KEEP or DROP:

reason_category (pick one):
- "mentioned_in_text": the input text discusses this topic, measurement, or property.
  The text does NOT need to use the exact field name — if the abstract discusses
  "N fertilizer" and "N availability", that justifies recommending tot_nitro_content,
  nitrate_nitrogen, and agrochem_addition. If it mentions "crop rotation" or specific
  crops, that justifies crop_rotation. Use the relevant passage as evidence_snippet.
- "experimental_design": the study design implies this variable was controlled,
  measured, or systematically varied. E.g., "replicated field trial" implies
  experimental_factor. Use the passage describing the design as evidence_snippet.
- "domain_standard": this field is standard for the sample type but the input text
  does NOT discuss it even indirectly. E.g., recommending pH for a soil study when
  pH is never mentioned or implied. No evidence_snippet needed.

Rules:
- KEEP all "mentioned_in_text" and "experimental_design" fields.
- DROP all "domain_standard" fields — if the input doesn't discuss it, don't recommend it.
- DROP fields where the recommendation is a tautology (e.g., "sampling requires a
  collection device" is not evidence — it's true of every study).
- For evidence_snippet: quote or closely paraphrase the relevant passage from the input.
  It does NOT need to be verbatim — a faithful paraphrase is fine.

Input context:
{context}

Recommendations to verify:
{recommendations}

Output ONLY valid JSON:
{{
  "verified": [
    {{"field_name": "...", "reason_category": "...", "evidence_snippet": "...", "reason": "..."}}
  ],
  "dropped": [
    {{"field_name": "...", "reason_category": "...", "drop_reason": "..."}}
  ]
}}"""


def verify_suggestions(
    suggestions: list[dict[str, str]],
    captured_context: list[str],
    backend: str,
    provider: str | None,
    model_name: str,
) -> dict[str, Any]:
    """Ask the model to cite evidence for each recommendation, dropping unsupported ones.

    Returns {"verified": [...], "dropped": [...], "verify_tokens": {...}, "verify_elapsed": float}.
    """
    import llm as llm_lib

    context_text = "\n\n".join(captured_context)
    # Truncate context if extremely long (some schema contexts are huge)
    if len(context_text) > 50000:
        context_text = context_text[:50000] + "\n\n[... truncated ...]"

    recs_json = json.dumps(
        [{"field_name": s["field_name"], "reason": s["reason"]} for s in suggestions],
        indent=2,
    )

    prompt_text = VERIFY_PROMPT.format(context=context_text, recommendations=recs_json)

    t0 = time.time()

    # Always use llm library for verification (simple text prompt, no PDFs)
    m = llm_lib.get_model(model_name if backend == "llm" else _resolve_llm_model_for_verify(model_name))
    response = m.prompt(prompt_text, temperature=0.2)
    raw = response.text()
    elapsed = round(time.time() - t0, 2)

    # Extract token usage
    verify_tokens = {
        "input_tokens": getattr(response, "input_tokens", None),
        "output_tokens": getattr(response, "output_tokens", None),
    }

    # Parse JSON from response (strip markdown fences if present)
    cleaned = re.sub(r"^```(?:json)?\s*\n?|\n?```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return {
            "verified": suggestions,  # Fall back to unfiltered
            "dropped": [],
            "verify_error": f"Failed to parse verification response: {raw[:200]}",
            "verify_tokens": verify_tokens,
            "verify_elapsed": elapsed,
        }

    return {
        "verified": parsed.get("verified", []),
        "dropped": parsed.get("dropped", []),
        "verify_tokens": verify_tokens,
        "verify_elapsed": elapsed,
    }


def _resolve_llm_model_for_verify(pipeline_model: str) -> str:
    """Map a pipeline model name to an llm library model for verification.

    When using the pipeline backend (GCP/PNNL), the model name may not be
    recognized by llm plugins. Fall back to a cheap default.
    """
    import llm as llm_lib

    # Try the model name directly first
    try:
        llm_lib.get_model(pipeline_model)
        return pipeline_model
    except llm_lib.UnknownModelError:
        pass

    # Map known pipeline models to llm equivalents
    mapping: dict[str, str] = {
        "gemini-2.5-flash": "gemini/gemini-2.5-flash",
        "gemini-2.5-pro": "gemini/gemini-2.5-pro",
    }
    mapped = mapping.get(pipeline_model)
    if mapped:
        try:
            llm_lib.get_model(mapped)
            return mapped
        except llm_lib.UnknownModelError:
            pass

    # Last resort: use gpt-4o-mini (cheap, widely available)
    return "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Eval runner (backend-agnostic)
# ---------------------------------------------------------------------------


def _strip_dois(doc: dict[str, Any]) -> dict[str, Any]:
    """Return a deep copy of a submission doc with all DOI fields blanked.

    This prevents the suggestor pipeline from running its DOI waterfall
    and PDF download, so the LLM sees only the study description, notes,
    and schema context — no publication abstracts or full text.
    """
    import copy

    stripped = copy.deepcopy(doc)
    ms = stripped.get("metadata_submission", {})
    sf = ms.get("studyForm", {})
    sf.pop("dataDois", None)
    sf.pop("publicationDois", None)
    mf = ms.get("multiOmicsForm", {})
    mf.pop("awardDois", None)
    # Also strip protocol DOIs
    omics = mf.get("nmdc:OmicsProcessing", {})
    blocks = omics.values() if isinstance(omics, dict) else []
    for protocol_block in blocks:
        if isinstance(protocol_block, dict):
            for ep in protocol_block.get("externalProtocols", []):
                if isinstance(ep, dict):
                    ep.pop("doi", None)
    return stripped


def run_one_model(
    backend: str,
    provider: str | None,
    model: str | None,
    ground_truth: list[dict[str, Any]],
    collection: Any,
    exclude_from_precision: set[str],
    enrichment: bool = True,
    verify: bool = False,
) -> dict[str, Any]:
    """Run the pipeline eval for a single backend/model combination."""
    from nmdc_metadata_suggestor_ai_tool.recommendation_pipeline import (
        run_recommendation_pipeline,
    )

    # Resolve model name
    if backend == "llm":
        if model is None:
            raise ValueError("--model is required for --backend llm")
        model_name = model
        display_backend = f"llm ({model_name})"
    else:
        from nmdc_metadata_suggestor_ai_tool.llm_client import LLMClient

        probe = LLMClient(access_provider=provider, model=model)
        model_name = probe.model
        display_backend = f"{provider} ({model_name})"

    enrich_label = "with DOI/PDF" if enrichment else "NO enrichment"
    print(f"\n{'=' * 70}")
    print(f"Backend: {display_backend}  [{enrich_label}]")
    print(f"{'=' * 70}")

    results: list[dict[str, Any]] = []
    for entry in ground_truth:
        submission_id = entry["submission_id"]
        doc = collection.find_one({"id": submission_id})
        if doc is None:
            print(f"SKIP: {submission_id} not in MongoDB")
            results.append(
                {
                    "submission_id": submission_id,
                    "study_name": entry["study_name"],
                    "model": model_name,
                    "backend": backend,
                    "provider": provider or "llm",
                    "enrichment": enrichment,
                    "status": "skipped",
                    "reason": "submission not found in MongoDB",
                }
            )
            continue

        doc.pop("_id", None)
        if not enrichment:
            doc = _strip_dois(doc)
        expected = {s["field_name"] for s in entry["expected_slots"]}
        print(f"=== {entry['study_name'][:70]} ===")
        print(f"  Expected: {sorted(expected)}")

        # Create fresh client per submission
        usage: dict[str, int | None]
        if backend == "llm":
            from nmdc_metadata_suggestor_ai_tool.system_prompt import system_prompt

            llm_client: Any = LLMLibraryAdapter(model_name=model_name, system_prompt=system_prompt)
            usage = {"input_tokens": None, "output_tokens": None}
        else:
            from nmdc_metadata_suggestor_ai_tool.llm_client import LLMClient

            llm_client = LLMClient(access_provider=provider, model=model)
            usage = instrument_for_tokens(llm_client)

        # Capture input context for verification
        captured_context = _wrap_for_context_capture(llm_client) if verify else []

        t0 = time.time()
        try:
            output = run_recommendation_pipeline(
                submission_object=doc,
                llm_client=llm_client,
            )
            elapsed = round(time.time() - t0, 2)

            # Extract tokens (adapter has direct access; pipeline uses monkey-patch)
            if backend == "llm":
                usage = llm_client.get_token_usage()

            raw_suggestions = [{"field_name": s.field_name, "reason": s.reason} for s in output.metadata_fields]
            predicted = {s["field_name"] for s in raw_suggestions}
            scores = score_sets(predicted, expected, exclude_from_precision)
            est_cost = estimate_cost(model_name, usage["input_tokens"], usage["output_tokens"])

            print(f"  Raw ({len(predicted)}): {sorted(predicted)}")
            print(f"  Raw scores: P={scores['precision']}  R={scores['recall']}  F1={scores['f1']}")

            # Verification step: ask model to cite evidence or drop
            verification: dict[str, Any] = {}
            verified_suggestions: list[dict[str, str]] = raw_suggestions
            verified_scores: dict[str, Any] = scores
            if verify:
                print("  Verifying...")
                verification = verify_suggestions(
                    suggestions=raw_suggestions,
                    captured_context=captured_context,
                    backend=backend,
                    provider=provider,
                    model_name=model_name,
                )
                verified_suggestions = verification.get("verified", raw_suggestions)
                verified_predicted = {s["field_name"] for s in verified_suggestions}
                verified_scores = score_sets(verified_predicted, expected, exclude_from_precision)
                dropped = verification.get("dropped", [])

                print(f"  Verified ({len(verified_predicted)}): {sorted(verified_predicted)}")
                print(
                    f"  Verified scores: P={verified_scores['precision']}"
                    f"  R={verified_scores['recall']}  F1={verified_scores['f1']}"
                )
                print(f"  Dropped {len(dropped)}: {[d['field_name'] for d in dropped]}")
                if verification.get("verify_error"):
                    print(f"  Verify error: {verification['verify_error']}")

                # Add verify cost to totals
                v_in = (verification.get("verify_tokens") or {}).get("input_tokens")
                v_out = (verification.get("verify_tokens") or {}).get("output_tokens")
                v_cost = estimate_cost(model_name, v_in, v_out)
                if v_in is not None and usage["input_tokens"] is not None:
                    usage["input_tokens"] = (usage["input_tokens"] or 0) + v_in
                if v_out is not None and usage["output_tokens"] is not None:
                    usage["output_tokens"] = (usage["output_tokens"] or 0) + v_out
                if v_cost is not None and est_cost is not None:
                    est_cost = est_cost + v_cost
                elapsed = round(elapsed + verification.get("verify_elapsed", 0), 2)

            if scores.get("excluded_correct"):
                print(f"  Excluded from precision: {scores['excluded_correct']}")
            tokens_str = (
                f"in={usage['input_tokens']} out={usage['output_tokens']}"
                if usage["input_tokens"] is not None
                else "tokens=unavailable"
            )
            cost_str = f"~${est_cost:.4f}" if est_cost is not None else "cost=unavailable"
            print(f"  Time: {elapsed}s  Tokens: {tokens_str}  Cost: {cost_str}\n")

            result_entry: dict[str, Any] = {
                "submission_id": submission_id,
                "study_name": entry["study_name"],
                "package": entry["package"],
                "model": model_name,
                "backend": backend,
                "provider": provider or "llm",
                "enrichment": enrichment,
                "verified": verify,
                "elapsed_seconds": elapsed,
                "input_tokens": usage["input_tokens"],
                "output_tokens": usage["output_tokens"],
                "est_cost_usd": round(est_cost, 6) if est_cost is not None else None,
                "expected_slots": sorted(expected),
                "predicted_slots": sorted(predicted),
                "raw_scores": scores,
                "all_suggestions": raw_suggestions,
            }
            if verify:
                verified_predicted = {s["field_name"] for s in verified_suggestions}
                result_entry["verified_slots"] = sorted(verified_predicted)
                result_entry["verified_scores"] = verified_scores
                result_entry["verified_suggestions"] = verified_suggestions
                result_entry["dropped_suggestions"] = verification.get("dropped", [])
                result_entry["scores"] = verified_scores  # Use verified for aggregation
            else:
                result_entry["scores"] = scores

            results.append(result_entry)

        except Exception as e:
            elapsed = round(time.time() - t0, 2)
            print(f"  ERROR: {e} ({elapsed}s)\n")
            results.append(
                {
                    "submission_id": submission_id,
                    "study_name": entry["study_name"],
                    "model": model_name,
                    "backend": backend,
                    "provider": provider or "llm",
                    "enrichment": enrichment,
                    "status": "error",
                    "error": str(e),
                    "elapsed_seconds": elapsed,
                    "input_tokens": usage["input_tokens"],
                    "output_tokens": usage["output_tokens"],
                }
            )

    # Per-model aggregate
    scored = [r for r in results if "scores" in r]
    if scored:
        avg_p = sum(r["scores"]["precision"] for r in scored) / len(scored)
        avg_r = sum(r["scores"]["recall"] for r in scored) / len(scored)
        avg_f1 = sum(r["scores"]["f1"] for r in scored) / len(scored)
        total_time = sum(r["elapsed_seconds"] for r in scored)
        total_in = sum(r["input_tokens"] or 0 for r in scored)
        known = [r for r in scored if r["input_tokens"] is not None]
        n_known = len(known)

        print(f"--- {model_name} ({backend}) ---")
        print(f"  Accuracy:  P={avg_p:.3f}  R={avg_r:.3f}  F1={avg_f1:.3f}")
        print(f"  Time:      {total_time:.1f}s total  ({total_time / len(scored):.1f}s avg)")
        if n_known:
            total_in = sum(r["input_tokens"] or 0 for r in known)
            total_out = sum(r["output_tokens"] or 0 for r in known)
            total_cost = sum(r["est_cost_usd"] or 0.0 for r in known)
            known_note = f" ({n_known}/{len(scored)} submissions)" if n_known < len(scored) else ""
            print(f"  Tokens:    {total_in:,} input  {total_out:,} output{known_note}")
            print(f"  Cost:      ~${total_cost:.4f} total  (~${total_cost / n_known:.4f} avg{known_note})")

    return {
        "model": model_name,
        "backend": backend,
        "provider": provider or "llm",
        "enrichment": enrichment,
        "submission_count": len(results),
        "results": results,
    }


# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------


def _get_sweep_configs() -> list[tuple[str, str | None, str]]:
    """Return (backend, provider, model) triples for all available models."""
    import os

    import llm as llm_lib
    from dotenv import load_dotenv

    load_dotenv()  # .env may have GCP/PNNL creds not exported in the shell

    configs: list[tuple[str, str | None, str]] = []

    # GCP models via pipeline backend
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or os.environ.get("VERTEX_PROJECT_ID"):
        from nmdc_metadata_suggestor_ai_tool.llm_client import GEMINI_MODELS

        for m in GEMINI_MODELS:
            configs.append(("pipeline", "gcp", m))

    # PNNL models via pipeline backend
    if os.environ.get("AI_INCUBATOR_KEY") and os.environ.get("AI_INCUBATOR_BASE_URL"):
        from nmdc_metadata_suggestor_ai_tool.llm_client import PNNL_GPT_MODELS

        for m in PNNL_GPT_MODELS:
            configs.append(("pipeline", "pnnl", m))

    # llm library models (personal API keys)
    models_yaml = HERE.parent / "models.yaml"
    if models_yaml.exists():
        with open(models_yaml) as f:
            model_names = yaml.safe_load(f).get("models", [])
        for name in model_names:
            try:
                llm_lib.get_model(name)
                configs.append(("llm", None, name))
            except llm_lib.UnknownModelError:
                pass  # skip models without keys/plugins

    return configs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Run suggestor pipeline eval with DOI/PDF enrichment")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--provider", choices=["gcp", "pnnl"], help="Pipeline backend provider")
    group.add_argument("--backend", choices=["llm"], help="Use llm library (personal API keys)")
    group.add_argument("--sweep", action="store_true", help="Run all available models and backends")
    parser.add_argument("--model", default=None, help="Model name")
    parser.add_argument("--strict", action="store_true", help="Count env triad in precision")
    parser.add_argument(
        "--no-enrichment",
        action="store_true",
        help="Skip DOI waterfall and PDF download (context ablation)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Re-prompt model to cite evidence for each recommendation, dropping unsupported ones",
    )
    parser.add_argument(
        "--mongo-uri",
        default="mongodb://localhost:27017/nmdc_data_dev",
        help="MongoDB URI for data-dev submissions",
    )
    args = parser.parse_args()

    ground_truth = load_ground_truth()
    mongo_client = MongoClient(args.mongo_uri)
    db = mongo_client.get_default_database()
    collection = db["nmdc_submissions"]

    enrichment = not args.no_enrichment
    exclude = set() if args.strict else ENV_TRIAD
    if not args.strict:
        print(f"Excluding from precision: {sorted(exclude)}")
        print("(use --strict to count these as false positives)")
    if not enrichment:
        print("DOI/PDF enrichment DISABLED (--no-enrichment)")
    if args.verify:
        print("Verification ENABLED (--verify): model will cite evidence or drop recommendations")
    print()

    # Determine which models to run
    if args.sweep:
        configs = _get_sweep_configs()
        if not configs:
            print("ERROR: --sweep found no configured providers or llm models.")
            raise SystemExit(1)
        print(f"Sweep: {len(configs)} model(s)")
        for _backend, provider, model in configs:
            label = f"{provider}" if provider else "llm"
            print(f"  [{label}] {model}")
        print()
    elif args.backend == "llm":
        if not args.model:
            parser.error("--model is required with --backend llm")
        configs = [("llm", None, args.model)]
    else:
        configs = [("pipeline", args.provider, args.model or "")]

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    all_runs: list[dict[str, Any]] = []
    for backend, provider, model in configs:
        run_data = run_one_model(
            backend=backend,
            provider=provider,
            model=model or None,
            ground_truth=ground_truth,
            collection=collection,
            exclude_from_precision=exclude,
            enrichment=enrichment,
            verify=args.verify,
        )
        all_runs.append(run_data)

        # Write per-model result file
        model_slug = run_data["model"].replace("/", "-").replace("@", "-")
        backend_slug = f"{provider}" if provider else "llm"
        enrich_slug = "enriched" if enrichment else "no-enrichment"
        verify_slug = "_verified" if args.verify else ""
        result_path = RESULTS_DIR / f"{model_slug}_{backend_slug}_{enrich_slug}{verify_slug}_{timestamp}.yaml"
        per_model_data: dict[str, Any] = {
            "eval_name": "field-guidance-pipeline",
            "scoring": "strict" if args.strict else "env-triad-excluded",
            "enrichment": enrichment,
            "verified": args.verify,
            "timestamp": timestamp,
            **run_data,
        }
        with open(result_path, "w") as f:
            yaml.dump(per_model_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        print(f"  Results: {result_path}")

    # Cross-model comparison
    if len(all_runs) > 1:
        print(f"\n{'=' * 70}")
        print("CROSS-MODEL COMPARISON")
        print(f"{'=' * 70}")
        for run in all_runs:
            scored = [r for r in run["results"] if "scores" in r]
            if not scored:
                errors = len([r for r in run["results"] if "error" in r])
                print(f"  {run['model']:<28s} ({run['backend']:<8s})  {errors} errors")
                continue
            avg_p = sum(r["scores"]["precision"] for r in scored) / len(scored)
            avg_r = sum(r["scores"]["recall"] for r in scored) / len(scored)
            avg_f1 = sum(r["scores"]["f1"] for r in scored) / len(scored)
            total_cost = sum(r.get("est_cost_usd") or 0 for r in scored)
            total_time = sum(r["elapsed_seconds"] for r in scored)
            print(
                f"  {run['model']:<28s} ({run['backend']:<8s})"
                f"  P={avg_p:.3f}  R={avg_r:.3f}  F1={avg_f1:.3f}"
                f"  ${total_cost:.4f}  {total_time:.0f}s"
            )


if __name__ == "__main__":
    main()
