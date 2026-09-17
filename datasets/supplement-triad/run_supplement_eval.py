#!/usr/bin/env python3
"""Run the suggestor's env triad pipeline with and without supplements, and score both.

The test case (study, snapshot, submission object, references) is imported from
the suggestor package; see README.md in this directory for the design.

Usage:
    uv run python datasets/supplement-triad/run_supplement_eval.py --provider gcp
    uv run python datasets/supplement-triad/run_supplement_eval.py --provider gcp --model gemini-2.5-pro
    uv run python datasets/supplement-triad/run_supplement_eval.py --limit 8 --chunk-size 8   # smoke run

Output: pipeline-results/{model}_{provider}_{timestamp}.yaml with both arms'
outputs (values and the model's reasons), per-slot score summaries, coverage
and arm deltas; a {model}_{provider}_{timestamp}_scores.tsv with one row per
scored suggestion; a row appended to summary.tsv; and report.md rewritten for
the latest run.
"""

import argparse
import csv
import logging
import time
from pathlib import Path
from typing import Any

import yaml
from nmdc_metadata_suggestor_ai_tool.env_triad_recommendation import get_env_triad_recommendation
from nmdc_metadata_suggestor_ai_tool.evaluation import phyllosphere
from nmdc_metadata_suggestor_ai_tool.evaluation.env_triad_scoring import (
    Reference,
    compare_outputs,
    coverage,
)
from nmdc_metadata_suggestor_ai_tool.llm_client import LLMClient
from nmdc_metadata_suggestor_ai_tool.models.llm_output import LLMOutput
from nmdc_metadata_suggestor_ai_tool.publication_ingestion.download_pdf import remove_temp_files
from nmdc_metadata_suggestor_ai_tool.publication_ingestion.supplements import (
    format_supplement_context,
    retrieve_supplements,
)

from nmdc_ai_eval.envo_scorer import get_envo_adapter
from nmdc_ai_eval.triad_output_scorer import ENV_TRIAD_SLOTS, hierarchy_scorer, score_output, summarize

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "pipeline-results"
ARMS = ("publication", "publication+supplements")
REFERENCES = ("supplement", "nmdc")

logger = logging.getLogger("supplement_eval")


class ErrorCollector(logging.Handler):
    """Keep the pipeline's per-chunk error messages; it logs them and returns nothing."""

    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def run_arm(
    name: str,
    llm_client: LLMClient,
    submission_object: dict[str, Any],
    samples: list[dict[str, Any]],
    study_context: list[str],
    chunk_size: int,
) -> tuple[LLMOutput, float, list[str]]:
    """Run one arm; return its output, wall time, and any per-chunk pipeline errors."""
    logger.info("arm %s: %d samples, %d context messages", name, len(samples), len(study_context))
    collector = ErrorCollector()
    pipeline_logger = logging.getLogger("nmdc_metadata_suggestor_ai_tool.env_triad_recommendation")
    pipeline_logger.addHandler(collector)
    started = time.monotonic()
    try:
        output = get_env_triad_recommendation(
            llm_client=llm_client,
            samples=samples,
            submission_object=submission_object,
            study_context=study_context,
            interface_names=[phyllosphere.INTERFACE],
            chunk_size=chunk_size,
        )
    finally:
        pipeline_logger.removeHandler(collector)
    elapsed = time.monotonic() - started
    logger.info("arm %s: %d suggestions in %.0fs", name, len(output.metadata_fields), elapsed)
    return output, elapsed, collector.messages


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def render_report(run: dict[str, Any]) -> str:
    """Markdown summary: coverage, per-slot ontology scores per arm and reference, arm deltas."""
    lines = [
        "# Supplement-driven env triad eval (phyllosphere)",
        "",
        f"Study `{phyllosphere.STUDY_ID}`, DOI `{phyllosphere.PUBLICATION_DOI}`, model `{run['model']}` "
        f"via `{run['provider']}`, {run['n_samples']} biosamples, chunk size {run['chunk_size']}, "
        f"run {run['timestamp']}.",
        "",
        "## Retrieval",
        "",
        f"- Supplements kept: {run['supplements']['n_kept']} (inlined as text: {run['supplements']['n_inlined']}), "
        f"skipped: {run['supplements']['n_skipped']}, {run['supplements']['elapsed_seconds']:.1f}s",
        f"- Supplement context: {run['supplements']['context_chars']:,} characters",
        f"- Samples without a supplement row: {len(run['supplements']['unmatched'])}",
        "",
        "## Coverage",
        "",
        "| Arm | samples answered | suggestions | without id | unknown id | chunk errors | wall time |",
        "|---|---|---|---|---|---|---|",
    ]
    for arm in ARMS:
        cov = run["arms"][arm]["coverage"]
        lines.append(
            f"| {arm} | {cov['samples_with_suggestions']}/{cov['n_samples']} | {cov['suggestions_total']} | "
            f"{cov['suggestions_without_id']} | {cov['suggestions_with_unknown_id']} | "
            f"{len(run['arms'][arm]['chunk_errors'])} | {run['arms'][arm]['elapsed_seconds']:.0f}s |"
        )
    for arm in ARMS:
        for message in run["arms"][arm]["chunk_errors"]:
            lines.append(f"- {arm}: `{message[:300]}`")
    lines += [
        "",
        "## Ontology score per slot (mean; exact-match rate in parentheses)",
        "",
        "Scored samples are those answered by the arm *and* holding a reference term for the slot.",
        "",
        "| Slot | Reference | " + " | ".join(ARMS) + " |",
        "|---|---|" + "---|" * len(ARMS),
    ]
    for slot in ENV_TRIAD_SLOTS:
        for ref in REFERENCES:
            cells = []
            for arm in ARMS:
                s = run["arms"][arm]["scores"][ref][slot]
                cells.append(f"{fmt(s['mean_ontology_score'])} ({fmt(s['exact_match_rate'])}, n={s['n_scored']})")
            lines.append(f"| {slot} | {ref} | " + " | ".join(cells) + " |")
    lines += ["", "## Relationship to the supplement reference", ""]
    lines.append("| Slot | Arm | exact | descendant | ancestor | unrelated | unparsed |")
    lines.append("|---|---|---|---|---|---|---|")
    for slot in ENV_TRIAD_SLOTS:
        for arm in ARMS:
            s = run["arms"][arm]["scores"]["supplement"][slot]
            r = s["relationships"]
            lines.append(
                f"| {slot} | {arm} | {r['exact']} | {r['descendant']} | {r['ancestor']} | "
                f"{r['unrelated']} | {s['unparsed']} |"
            )
    lines += ["", "## What each arm suggested", ""]
    for arm in ARMS:
        lines.append(f"### {arm}")
        lines.append("")
        for slot in ENV_TRIAD_SLOTS:
            s = run["arms"][arm]["scores"]["supplement"][slot]
            top = ", ".join(f"`{v}` ×{n}" for v, n in list(s["values"].items())[:5]) or "none scored"
            tiers = ", ".join(f"{k} {v}" for k, v in s["tiers"].items()) or "no provenance"
            outcomes = ", ".join(f"{k} {v}" for k, v in s["outcomes"].items())
            lines.append(f"- **{slot}**: {top}")
            lines.append(f"  - gate: {tiers}; {outcomes}")
        lines.append("")
    lines += [
        "## Change from adding supplements (graded by ENVO proximity)",
        "",
        "| Slot | changed | toward supplement | away from supplement | toward NMDC | away from NMDC |",
        "|---|---|---|---|---|---|",
    ]
    for slot in ENV_TRIAD_SLOTS:
        d_supp = run["deltas"]["supplement"][slot]
        d_nmdc = run["deltas"]["nmdc"][slot]
        lines.append(
            f"| {slot} | {d_supp['changed']}/{d_supp['n_compared']} | {d_supp['toward_reference']} | "
            f"{d_supp['away_from_reference']} | {d_nmdc['toward_reference']} | {d_nmdc['away_from_reference']} |"
        )
    lines.append("")
    for slot in ENV_TRIAD_SLOTS:
        transitions = run["deltas"]["supplement"][slot]["transitions"]
        if transitions:
            lines.append(f"Transitions for {slot}:")
            lines.append("")
            lines.extend(f"- {n}× `{t}`" for t, n in transitions.items())
            lines.append("")
    return "\n".join(lines)


def write_term_scores(path: Path, rows: list[dict[str, Any]]) -> None:
    """One row per scored suggestion, per arm and reference; lists joined with ' | '."""
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: " | ".join(v) if isinstance(v, list) else v for k, v in row.items()})


def append_summary(run: dict[str, Any]) -> None:
    """One row per run in summary.tsv: mean ontology score per arm, slot, and reference."""
    path = RESULTS_DIR / "summary.tsv"
    row: dict[str, Any] = {
        "timestamp": run["timestamp"],
        "model": run["model"],
        "provider": run["provider"],
        "n_samples": run["n_samples"],
        "chunk_size": run["chunk_size"],
    }
    for arm in ARMS:
        row[f"{arm}_answered"] = run["arms"][arm]["coverage"]["samples_with_suggestions"]
        row[f"{arm}_seconds"] = round(run["arms"][arm]["elapsed_seconds"])
        for ref in REFERENCES:
            for slot in ENV_TRIAD_SLOTS:
                row[f"{arm}_{ref}_{slot}"] = run["arms"][arm]["scores"][ref][slot]["mean_ontology_score"]
    new_file = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row), delimiter="\t")
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", default="gcp", choices=["gcp", "cborg", "pnnl"])
    parser.add_argument("--model", default=None, help="model name; provider default when omitted")
    # The pipeline's own default is 50. At 50, gemini-2.5-flash dropped the id
    # field for a whole chunk in one arm and truncated its JSON in the other.
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument("--limit", type=int, default=None, help="only the first N biosamples")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    study = phyllosphere.load_study()
    samples = phyllosphere.load_biosamples()
    if args.limit:
        samples = samples[: args.limit]
    sample_ids = [str(s["id"]) for s in samples]
    submission_object = phyllosphere.submission_object_for(study)
    stripped = phyllosphere.strip_triad(samples)

    started = time.monotonic()
    retrieved = retrieve_supplements(phyllosphere.PUBLICATION_DOI)
    retrieval_elapsed = time.monotonic() - started
    supplement_messages = format_supplement_context(retrieved)
    references: dict[str, Reference] = {"nmdc": phyllosphere.nmdc_reference(samples)}
    references["supplement"], unmatched = phyllosphere.supplement_reference(retrieved.files, samples)
    if unmatched:
        logger.warning("%d samples have no row in %s", len(unmatched), phyllosphere.SUPPLEMENT_REFERENCE_FILE)
    remove_temp_files([f.saved_path for f in retrieved.files if f.saved_path])

    llm_client = LLMClient(access_provider=args.provider, model=args.model)
    contexts = {"publication": [], "publication+supplements": supplement_messages}
    outputs: dict[str, LLMOutput] = {}
    arms: dict[str, dict[str, Any]] = {}
    for arm, context in contexts.items():
        output, elapsed, errors = run_arm(arm, llm_client, submission_object, stripped, context, args.chunk_size)
        outputs[arm] = output
        arms[arm] = {"elapsed_seconds": elapsed, "chunk_errors": errors, "coverage": coverage(output, sample_ids)}

    adapter = get_envo_adapter()
    term_rows: list[dict[str, Any]] = []
    for arm, output in outputs.items():
        arms[arm]["scores"] = {}
        for ref, reference in references.items():
            scores = score_output(output, reference, sample_ids, adapter=adapter)
            arms[arm]["scores"][ref] = summarize(scores)
            term_rows.extend({"arm": arm, "reference": ref, **s.as_dict()} for s in scores)
        arms[arm]["output"] = output.model_dump()
    scorer = hierarchy_scorer(adapter)
    deltas = {
        ref: {
            slot: delta.as_dict()
            for slot, delta in compare_outputs(
                outputs["publication"], outputs["publication+supplements"], reference, sample_ids, scorer=scorer
            ).items()
        }
        for ref, reference in references.items()
    }

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run: dict[str, Any] = {
        "eval_name": "supplement-triad",
        "study_id": phyllosphere.STUDY_ID,
        "publication_doi": phyllosphere.PUBLICATION_DOI,
        "model": llm_client.model,
        "provider": llm_client.access_provider,
        "n_samples": len(samples),
        "chunk_size": args.chunk_size,
        "timestamp": timestamp,
        "supplements": {
            "source": retrieved.source,
            "pmcid": retrieved.pmcid,
            "n_kept": len(retrieved.files),
            "n_inlined": sum(1 for f in retrieved.files if f.text),
            "n_skipped": len(retrieved.skipped),
            "kept": [f.filename for f in retrieved.files],
            "elapsed_seconds": retrieval_elapsed,
            "context_chars": sum(len(m) for m in supplement_messages),
            "unmatched": unmatched,
        },
        "arms": arms,
        "deltas": deltas,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    model_slug = str(llm_client.model).replace("/", "-").replace("@", "-")
    result_path = RESULTS_DIR / f"{model_slug}_{llm_client.access_provider}_{timestamp}.yaml"
    with open(result_path, "w") as f:
        yaml.dump(run, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    write_term_scores(result_path.with_name(result_path.stem + "_scores.tsv"), term_rows)
    report = render_report(run)
    (RESULTS_DIR / "report.md").write_text(report)
    append_summary(run)
    print(report)
    print(f"\nResults: {result_path}")


if __name__ == "__main__":
    main()
