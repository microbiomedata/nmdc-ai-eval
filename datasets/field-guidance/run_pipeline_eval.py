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
import time
from pathlib import Path
from typing import Any

import yaml
from pymongo import MongoClient

from nmdc_ai_eval.pricing import estimate_cost

HERE = Path(__file__).parent
GROUND_TRUTH = HERE / "ground_truth.yaml"
RESULTS_DIR = HERE / "pipeline-results"

# Fields excluded from precision scoring by default.
ENV_TRIAD = {"env_broad_scale", "env_local_scale", "env_medium"}


def load_ground_truth() -> list[dict[str, Any]]:
    with open(GROUND_TRUTH) as f:
        data = yaml.safe_load(f)
        return list(data["submissions"])  # type: ignore[index]


def score_slots(
    predicted: set[str],
    expected: set[str],
    exclude_from_precision: set[str] | None = None,
) -> dict[str, Any]:
    """Precision, recall, F1 on slot name sets."""
    if not predicted and not expected:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not expected:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision_set = predicted - (exclude_from_precision or set())
    excluded_correct = predicted & (exclude_from_precision or set())

    tp = len(precision_set & expected)
    fp = len(precision_set - expected)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positives": sorted(precision_set & expected),
        "false_positives": sorted(precision_set - expected),
        "false_negatives": sorted(expected - predicted),
        "excluded_correct": sorted(excluded_correct),
    }


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
# Backend: llm library (any model via personal API keys)
# ---------------------------------------------------------------------------


class LLMLibraryAdapter:
    """Adapter that makes the llm library look like the suggestor's LLMClient.

    The suggestor's run_recommendation_pipeline() calls add_message(),
    add_schema_context(), and generate() on whatever object it receives.
    This adapter implements those methods, collecting the messages, then
    routes through the llm library's plugin ecosystem when generate() is called.

    This means the SAME prompt construction (DOI waterfall, PDF download,
    schema context) is used regardless of backend — only the LLM call differs.
    """

    def __init__(self, model_name: str) -> None:
        self.model = model_name
        self.access_provider = "llm"
        self.messages: list[str] = []
        self._pdf_paths: list[str] = []
        self._last_response: Any = None  # llm.Response for token extraction

    def add_message(self, text: str, pdf_files: list[str] | None = None) -> None:
        if text:
            self.messages.append(text)
        if pdf_files:
            self._pdf_paths.extend(pdf_files)

    def add_schema_context(self, schema: str) -> None:
        self.add_message(
            text="Utilize the following schema context to inform your metadata field recommendations:\n" + schema,
        )

    def add_schema_and_slot_examples(self) -> None:
        raise NotImplementedError

    def generate(
        self,
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        gemini_temperature: float = 0.4,
    ) -> str:
        import llm as llm_lib
        from nmdc_metadata_suggestor_ai_tool.system_prompt import system_prompt

        m = llm_lib.get_model(self.model)
        full_prompt = "\n\n".join(self.messages)

        # Build PDF attachments
        attachments: list[Any] = []
        for pdf_path in self._pdf_paths:
            try:
                attachments.append(llm_lib.Attachment(path=pdf_path, type="application/pdf"))
            except Exception:  # noqa: S110
                pass  # Skip PDFs that can't be attached (model may not support them)

        response = m.prompt(
            full_prompt,
            system=system_prompt,
            attachments=attachments if attachments else None,
            temperature=gemini_temperature,
        )
        self._last_response = response
        return response.text()

    def get_token_usage(self) -> dict[str, int | None]:
        """Extract token usage from the last llm.Response."""
        if self._last_response is None:
            return {"input_tokens": None, "output_tokens": None}
        return {
            "input_tokens": getattr(self._last_response, "input_tokens", None),
            "output_tokens": getattr(self._last_response, "output_tokens", None),
        }

    def get_duration_ms(self) -> int | None:
        """Extract duration from the last llm.Response."""
        if self._last_response is None:
            return None
        try:
            return self._last_response.duration_ms()
        except Exception:
            return None


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
            llm_client: Any = LLMLibraryAdapter(model_name=model_name)
            usage = {"input_tokens": None, "output_tokens": None}
        else:
            from nmdc_metadata_suggestor_ai_tool.llm_client import LLMClient

            llm_client = LLMClient(access_provider=provider, model=model)
            usage = instrument_for_tokens(llm_client)

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

            predicted = {s.field_name for s in output.metadata_fields}
            scores = score_slots(predicted, expected, exclude_from_precision)
            est_cost = estimate_cost(model_name, usage["input_tokens"], usage["output_tokens"])

            print(f"  Predicted ({len(predicted)}): {sorted(predicted)}")
            print(f"  P={scores['precision']}  R={scores['recall']}  F1={scores['f1']}")
            if scores.get("excluded_correct"):
                print(f"  Excluded from precision: {scores['excluded_correct']}")
            tokens_str = (
                f"in={usage['input_tokens']} out={usage['output_tokens']}"
                if usage["input_tokens"] is not None
                else "tokens=unavailable"
            )
            cost_str = f"~${est_cost:.4f}" if est_cost is not None else "cost=unavailable"
            print(f"  Time: {elapsed}s  Tokens: {tokens_str}  Cost: {cost_str}\n")

            results.append(
                {
                    "submission_id": submission_id,
                    "study_name": entry["study_name"],
                    "package": entry["package"],
                    "model": model_name,
                    "backend": backend,
                    "provider": provider or "llm",
                    "enrichment": enrichment,
                    "elapsed_seconds": elapsed,
                    "input_tokens": usage["input_tokens"],
                    "output_tokens": usage["output_tokens"],
                    "est_cost_usd": round(est_cost, 6) if est_cost is not None else None,
                    "expected_slots": sorted(expected),
                    "predicted_slots": sorted(predicted),
                    "scores": scores,
                    "all_suggestions": [
                        {"field_name": s.field_name, "reason": s.reason} for s in output.metadata_fields
                    ],
                }
            )

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
        total_out = sum(r["output_tokens"] or 0 for r in scored)
        total_cost = sum(r["est_cost_usd"] or 0.0 for r in scored)
        tokens_known = any(r["input_tokens"] is not None for r in scored)

        print(f"--- {model_name} ({backend}) ---")
        print(f"  Accuracy:  P={avg_p:.3f}  R={avg_r:.3f}  F1={avg_f1:.3f}")
        print(f"  Time:      {total_time:.1f}s total  ({total_time / len(scored):.1f}s avg)")
        if tokens_known:
            print(f"  Tokens:    {total_in:,} input  {total_out:,} output")
            print(f"  Cost:      ~${total_cost:.4f} total  (~${total_cost / len(scored):.4f} avg)")

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
        )
        all_runs.append(run_data)

        # Write per-model result file
        model_slug = run_data["model"].replace("/", "-").replace("@", "-")
        backend_slug = f"{provider}" if provider else "llm"
        enrich_slug = "enriched" if enrichment else "no-enrichment"
        result_path = RESULTS_DIR / f"{model_slug}_{backend_slug}_{enrich_slug}_{timestamp}.yaml"
        per_model_data: dict[str, Any] = {
            "eval_name": "field-guidance-pipeline",
            "scoring": "strict" if args.strict else "env-triad-excluded",
            "enrichment": enrichment,
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
