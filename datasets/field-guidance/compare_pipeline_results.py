#!/usr/bin/env python3
"""Compare pipeline eval results across models and runs.

Reads all YAML files in pipeline-results/ and prints a comparison table.

Usage:
    python compare_pipeline_results.py              # all results
    python compare_pipeline_results.py --latest     # most recent run per model
    python compare_pipeline_results.py --model gemini-2.5-flash  # filter by model
"""

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "pipeline-results"


def load_all_results() -> list[dict[str, Any]]:
    """Load all result YAML files, sorted by timestamp."""
    if not RESULTS_DIR.exists():
        return []
    results = []
    for path in sorted(RESULTS_DIR.glob("*.yaml")):
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        data["_file"] = path.name
        results.append(data)
    return results


def summarize_run(data: dict[str, Any]) -> dict[str, Any]:
    """Extract summary stats from a single run."""
    results = data.get("results", [])
    scored = [r for r in results if "scores" in r]
    errors = [r for r in results if "error" in r]

    enrich = data.get("enrichment", True)
    enrich_str = "yes" if enrich else "no"

    if not scored:
        return {
            "model": data.get("model", "?"),
            "provider": data.get("provider", "?"),
            "enrichment": enrich_str,
            "timestamp": data.get("timestamp", "?"),
            "scoring": data.get("scoring", "?"),
            "n": 0,
            "errors": len(errors),
            "file": data.get("_file", "?"),
        }

    avg_p = sum(r["scores"]["precision"] for r in scored) / len(scored)
    avg_r = sum(r["scores"]["recall"] for r in scored) / len(scored)
    avg_f1 = sum(r["scores"]["f1"] for r in scored) / len(scored)
    total_time = sum(r.get("elapsed_seconds", 0) for r in scored)
    in_vals = [r["input_tokens"] for r in scored if r.get("input_tokens") is not None]
    out_vals = [r["output_tokens"] for r in scored if r.get("output_tokens") is not None]
    cost_vals = [r["est_cost_usd"] for r in scored if r.get("est_cost_usd") is not None]
    total_in = sum(in_vals) if in_vals else None
    total_out = sum(out_vals) if out_vals else None
    total_cost = sum(cost_vals) if cost_vals else None

    return {
        "model": data.get("model", "?"),
        "provider": data.get("provider", "?"),
        "enrichment": enrich_str,
        "timestamp": data.get("timestamp", "?"),
        "scoring": data.get("scoring", "?"),
        "n": len(scored),
        "errors": len(errors),
        "precision": avg_p,
        "recall": avg_r,
        "f1": avg_f1,
        "total_time": total_time,
        "avg_time": total_time / len(scored),
        "input_tokens": total_in,
        "output_tokens": total_out,
        "total_cost": total_cost,
        "file": data.get("_file", "?"),
    }


def print_comparison(summaries: list[dict[str, Any]]) -> None:
    """Print a formatted comparison table."""
    if not summaries:
        print("No results found in pipeline-results/")
        return

    # Header
    print(
        f"{'Model':<28s} {'Prov':<5s} {'DOI':>3s}"
        f" {'P':>6s} {'R':>6s} {'F1':>6s}"
        f" {'Cost':>8s} {'Time':>6s} {'Tokens':>12s}"
        f" {'N':>3s} {'When':<16s}"
    )
    print("-" * 110)

    for s in summaries:
        doi = s.get("enrichment", "yes")
        if s["n"] == 0:
            print(
                f"{s['model']:<28s} {s['provider']:<5s} {doi:>3s}"
                f" {'—':>6s} {'—':>6s} {'—':>6s}"
                f" {'—':>8s} {'—':>6s} {'—':>12s}"
                f" {s['n']:>3d} {s['timestamp']:<16s}"
                f"  ({s['errors']} errors)"
            )
            continue

        tok = f"{s['input_tokens']}+{s['output_tokens']}" if s["input_tokens"] is not None else "—"
        cost_str = f"${s['total_cost']:>7.4f}" if s["total_cost"] is not None else f"{'—':>8s}"
        print(
            f"{s['model']:<28s} {s['provider']:<5s} {doi:>3s}"
            f" {s['precision']:>6.3f} {s['recall']:>6.3f} {s['f1']:>6.3f}"
            f" {cost_str} {s['total_time']:>5.0f}s {tok:>12s}"
            f" {s['n']:>3d} {s['timestamp']:<16s}"
        )


def print_per_submission(summaries: list[dict[str, Any]], all_data: list[dict[str, Any]]) -> None:
    """Print per-submission breakdown across models."""
    # Group by submission
    by_submission: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for data in all_data:
        model = data.get("model", "?")
        for r in data.get("results", []):
            if "scores" in r:
                by_submission[r["submission_id"]].append((model, r))

    if not by_submission:
        return

    print(f"\n{'=' * 90}")
    print("PER-SUBMISSION BREAKDOWN")
    print(f"{'=' * 90}")

    for _sub_id, entries in by_submission.items():
        study = entries[0][1]["study_name"][:60] if entries else "?"
        expected = entries[0][1].get("expected_slots", []) if entries else []
        print(f"\n{study}")
        print(f"  Expected: {expected}")
        for model, r in entries:
            s = r["scores"]
            n_pred = len(r.get("predicted_slots", []))
            print(
                f"  {model:<28s} P={s['precision']:.3f} R={s['recall']:.3f} F1={s['f1']:.3f}  "
                f"({n_pred} predicted, {len(s.get('false_positives', []))} FP, {len(s.get('false_negatives', []))} FN)"
            )


def save_summary_tsv(summaries: list[dict[str, Any]], output_path: Path) -> None:
    """Write the comparison table as a TSV file."""
    cols = [
        "model",
        "provider",
        "enrichment",
        "n",
        "errors",
        "precision",
        "recall",
        "f1",
        "total_cost",
        "total_time",
        "input_tokens",
        "output_tokens",
        "scoring",
        "timestamp",
        "file",
    ]
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for s in summaries:
            row = []
            for c in cols:
                v = s.get(c)
                if v is None:
                    row.append("")
                elif isinstance(v, float):
                    row.append(f"{v:.6f}")
                else:
                    row.append(str(v))
            f.write("\t".join(row) + "\n")
    print(f"Summary saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare pipeline eval results")
    parser.add_argument("--latest", action="store_true", help="Show only the most recent run per model")
    parser.add_argument("--model", help="Filter by model name (substring match)")
    parser.add_argument("--detail", action="store_true", help="Show per-submission breakdown")
    parser.add_argument("--save-tsv", type=Path, help="Write comparison table to a TSV file")
    args = parser.parse_args()

    all_data = load_all_results()

    if args.model:
        all_data = [d for d in all_data if args.model.lower() in d.get("model", "").lower()]

    summaries = [summarize_run(d) for d in all_data]

    if args.latest:
        # Keep only the most recent per (model, provider, enrichment) combo
        latest: dict[str, dict[str, Any]] = {}
        for s in summaries:
            key = f"{s['model']}_{s['provider']}_{s.get('enrichment', 'yes')}"
            if key not in latest or s["timestamp"] > latest[key]["timestamp"]:
                latest[key] = s
        summaries = sorted(latest.values(), key=lambda s: s.get("f1", 0), reverse=True)
    else:
        summaries.sort(key=lambda s: (s["model"], s.get("enrichment", "yes"), s["timestamp"]))

    print_comparison(summaries)

    if args.save_tsv:
        save_summary_tsv(summaries, args.save_tsv)

    if args.detail:
        # Filter all_data to match what's in summaries
        shown_files = {s["file"] for s in summaries}
        filtered = [d for d in all_data if d.get("_file") in shown_files]
        print_per_submission(summaries, filtered)


if __name__ == "__main__":
    main()
