#!/usr/bin/env python3
"""Run the suggestor's full env triad workflow over three NMDC studies, and score it.

The cases (studies, snapshots, submission objects, references) are defined in
studies.py beside this file; see README.md in this directory for the design.

Each study runs in two arms:

* ``samples-only``: the biosample records alone, with no submission object, so
  no study text, DOI lookup or PDF reaches the model.
* ``full-workflow``: the same records plus a submission object built from the
  study record. The pipeline adds the study name and description, fetches
  context for every DOI, downloads any PDFs it finds, and passes the result
  through its ENVO validation gate, as it would for a portal submission.

Both arms are scored against the triad stored in NMDC.

Usage:
    uv run python datasets/workflow-triad/run_workflow_eval.py --provider gcp
    uv run python datasets/workflow-triad/run_workflow_eval.py --studies geodes-lakes
    uv run python datasets/workflow-triad/run_workflow_eval.py --limit 8 --chunk-size 8   # smoke run

Output: pipeline-results/{model}_{provider}_{timestamp}.yaml with every arm's
output (values and the model's reasons), per-slot score summaries, coverage,
DOI retrieval and arm deltas per study; a {model}_{provider}_{timestamp}_scores.tsv
with one row per scored suggestion; a row appended to summary.tsv; and
report.md rewritten for the latest run.
"""

import argparse
import csv
import logging
import time
from pathlib import Path
from typing import Any

import studies  # beside this script, on sys.path when run as one
import yaml
from nmdc_metadata_suggestor_ai_tool.doi_ingestion.main import get_doi_description_or_abstract
from nmdc_metadata_suggestor_ai_tool.env_triad_recommendation import get_env_triad_recommendation
from nmdc_metadata_suggestor_ai_tool.llm_client import LLMClient
from nmdc_metadata_suggestor_ai_tool.models.llm_output import LLMOutput
from nmdc_metadata_suggestor_ai_tool.utils.submission_parser import get_submission_fields

from nmdc_ai_eval.env_triad_scoring import Reference, compare_outputs, coverage
from nmdc_ai_eval.envo_scorer import get_envo_adapter
from nmdc_ai_eval.triad_output_scorer import ENV_TRIAD_SLOTS, TermScore, hierarchy_scorer, score_output, summarize

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "pipeline-results"
ARMS = ("samples-only", "full-workflow")
# A reference env_broad_scale of plain "terrestrial biome" marks a land-cover style
# triad (NEON sites, and the 1000 Soils cores taken at them) rather than a
# submitter's specific choice; the report splits scores on it.
GENERIC_BROAD_SCALE = "ENVO:00000446"

logger = logging.getLogger("workflow_eval")


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
    samples: list[dict[str, Any]],
    submission_object: dict[str, Any] | None,
    interface_names: list[str],
    chunk_size: int,
) -> tuple[LLMOutput, float, list[str]]:
    """Run one arm; return its output, wall time, and any per-chunk pipeline errors."""
    logger.info("arm %s: %d samples", name, len(samples))
    collector = ErrorCollector()
    pipeline_logger = logging.getLogger("nmdc_metadata_suggestor_ai_tool.env_triad_recommendation")
    pipeline_logger.addHandler(collector)
    started = time.monotonic()
    try:
        output = get_env_triad_recommendation(
            llm_client=llm_client,
            samples=samples,
            submission_object=submission_object,
            interface_names=interface_names,
            chunk_size=chunk_size,
        )
    finally:
        pipeline_logger.removeHandler(collector)
    elapsed = time.monotonic() - started
    logger.info("arm %s: %d suggestions in %.0fs", name, len(output.metadata_fields), elapsed)
    return output, elapsed, collector.messages


def probe_dois(submission_object: dict[str, Any]) -> list[dict[str, Any]]:
    """What the pipeline's DOI step can fetch for this submission, DOI by DOI.

    The pipeline does not report what it retrieved, so this repeats its lookup
    (same function, same provider routing) to record it. It downloads no PDFs;
    it counts the links the pipeline would download from.
    """
    probes = []
    for entry in get_submission_fields(submission_object)["dois"]:
        provider = entry.get("provider")
        try:
            result = get_doi_description_or_abstract(
                doi=entry["value"], skip_classification=True, sources=[provider] if provider else None
            )
            probes.append(
                {
                    "doi": entry["value"],
                    "provider": provider,
                    "context_chars": len(result.context or ""),
                    "pdf_links": len(result.publication_urls or []),
                }
            )
        except Exception as exc:  # a failed lookup is a finding, not a reason to stop
            probes.append({"doi": entry["value"], "provider": provider, "error": str(exc)[:300]})
    return probes


def generic_ids(reference: Reference) -> set[str]:
    return {
        sample_id
        for sample_id, slots in reference.items()
        if [t.curie for t in slots.get("env_broad_scale", [])] == [GENERIC_BROAD_SCALE]
    }


def run_study(
    case: studies.StudyCase, llm_client: LLMClient, chunk_size: int, limit: int | None, adapter: Any
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Both arms for one study: outputs, coverage, scores, deltas and DOI probes."""
    study = studies.load_study(case)
    samples = studies.load_biosamples(case)
    if limit:
        samples = samples[:limit]
    sample_ids = [str(s["id"]) for s in samples]
    reference = studies.nmdc_reference(samples)
    generic = generic_ids(reference)
    submission_object = studies.submission_object_for(case, study)
    withheld = studies.withhold(samples)
    interface_names = list(case.packages)

    doi_probes = probe_dois(submission_object)
    contexts = {"samples-only": None, "full-workflow": submission_object}
    outputs: dict[str, LLMOutput] = {}
    arms: dict[str, dict[str, Any]] = {}
    term_rows: list[dict[str, Any]] = []
    for arm, submission in contexts.items():
        output, elapsed, errors = run_arm(
            f"{case.key}/{arm}", llm_client, withheld, submission, interface_names, chunk_size
        )
        outputs[arm] = output
        scores = score_output(output, reference, sample_ids, adapter=adapter)
        arms[arm] = {
            "elapsed_seconds": elapsed,
            "chunk_errors": errors,
            "coverage": coverage(output, sample_ids),
            "scores": summarize(scores),
            "scores_by_reference_style": split_by_style(scores, generic),
            "output": output.model_dump(),
        }
        term_rows.extend(
            {"study": case.key, "arm": arm, "generic_reference": s.sample_id in generic, **s.as_dict()} for s in scores
        )
    scorer = hierarchy_scorer(adapter)
    deltas = {
        slot: delta.as_dict()
        for slot, delta in compare_outputs(
            outputs["samples-only"], outputs["full-workflow"], reference, sample_ids, scorer=scorer
        ).items()
    }
    result = {
        "study_id": case.study_id,
        "source": case.source,
        "packages": list(case.packages),
        "n_samples": len(samples),
        "n_generic_reference": len(generic),
        "submission_object": submission_object,
        "doi_probes": doi_probes,
        "arms": arms,
        "deltas": deltas,
    }
    return result, term_rows


def split_by_style(scores: list[TermScore], generic: set[str]) -> dict[str, Any]:
    """Per-slot summaries for generic and specific reference triads, when a study has both."""
    if not generic or all(s.sample_id in generic for s in scores):
        return {}
    return {
        "generic": summarize(s for s in scores if s.sample_id in generic),
        "specific": summarize(s for s in scores if s.sample_id not in generic),
    }


def pooled(term_rows: list[dict[str, Any]], arm: str) -> dict[str, dict[str, Any]]:
    """Per-slot means over every scored suggestion of every study for one arm."""
    out: dict[str, dict[str, Any]] = {}
    for slot in ENV_TRIAD_SLOTS:
        rows = [r for r in term_rows if r["arm"] == arm and r["slot"] == slot]
        n = len(rows)
        out[slot] = {
            "n_scored": n,
            "mean_ontology_score": round(sum(r["ontology_score"] for r in rows) / n, 4) if n else None,
            "exact_match_rate": round(sum(r["exact_match"] for r in rows) / n, 4) if n else None,
        }
    return out


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def cell(summary: dict[str, Any]) -> str:
    return f"{fmt(summary['mean_ontology_score'])} ({fmt(summary['exact_match_rate'])}, n={summary['n_scored']})"


def render_report(run: dict[str, Any]) -> str:
    """Markdown summary: pooled scores, then per study coverage, DOI retrieval, scores and deltas."""
    lines = [
        "# Full-workflow env triad eval (three NMDC studies)",
        "",
        f"Model `{run['model']}` via `{run['provider']}`, chunk size {run['chunk_size']}, run {run['timestamp']}. "
        "Reference: the triad stored in NMDC. Cells: mean ontology score (exact-match rate, n scored).",
        "",
        "## All studies pooled",
        "",
        "| Slot | " + " | ".join(ARMS) + " |",
        "|---|" + "---|" * len(ARMS),
    ]
    for slot in ENV_TRIAD_SLOTS:
        lines.append(f"| {slot} | " + " | ".join(cell(run["pooled"][arm][slot]) for arm in ARMS) + " |")
    lines += ["", "## Per study", "", "| Study | Slot | " + " | ".join(ARMS) + " |", "|---|---|" + "---|" * len(ARMS)]
    for key, result in run["studies"].items():
        for slot in ENV_TRIAD_SLOTS:
            lines.append(
                f"| {key} | {slot} | " + " | ".join(cell(result["arms"][arm]["scores"][slot]) for arm in ARMS) + " |"
            )
    for key, result in run["studies"].items():
        lines += [
            "",
            f"## {key} (`{result['study_id']}`, {result['source']}, {result['n_samples']} biosamples)",
            "",
            "### Coverage",
            "",
            "| Arm | samples answered | suggestions | without id | unknown id | chunk errors | wall time |",
            "|---|---|---|---|---|---|---|",
        ]
        for arm in ARMS:
            a = result["arms"][arm]
            cov = a["coverage"]
            lines.append(
                f"| {arm} | {cov['samples_with_suggestions']}/{cov['n_samples']} | {cov['suggestions_total']} | "
                f"{cov['suggestions_without_id']} | {cov['suggestions_with_unknown_id']} | "
                f"{len(a['chunk_errors'])} | {a['elapsed_seconds']:.0f}s |"
            )
        for arm in ARMS:
            for message in result["arms"][arm]["chunk_errors"]:
                lines.append(f"- {arm}: `{message[:300]}`")
        lines += ["", "### DOI retrieval (full-workflow arm)", "", "| DOI | provider | context chars | PDF links |"]
        lines.append("|---|---|---|---|")
        for p in result["doi_probes"]:
            if "error" in p:
                lines.append(f"| `{p['doi']}` | {p['provider']} | error: {p['error'][:80]} | |")
            else:
                lines.append(f"| `{p['doi']}` | {p['provider']} | {p['context_chars']:,} | {p['pdf_links']} |")
        lines += ["", "### Relationship to the NMDC triad", ""]
        lines.append("| Slot | Arm | exact | descendant | ancestor | unrelated | unparsed |")
        lines.append("|---|---|---|---|---|---|---|")
        for slot in ENV_TRIAD_SLOTS:
            for arm in ARMS:
                s = result["arms"][arm]["scores"][slot]
                r = s["relationships"]
                lines.append(
                    f"| {slot} | {arm} | {r['exact']} | {r['descendant']} | {r['ancestor']} | "
                    f"{r['unrelated']} | {s['unparsed']} |"
                )
        styles = {arm: result["arms"][arm]["scores_by_reference_style"] for arm in ARMS}
        if any(styles.values()):
            lines += [
                "",
                f"### Generic vs specific reference triads ({result['n_generic_reference']} samples with "
                "`terrestrial biome` as the stored env_broad_scale)",
                "",
                "| Slot | Reference style | " + " | ".join(ARMS) + " |",
                "|---|---|" + "---|" * len(ARMS),
            ]
            for slot in ENV_TRIAD_SLOTS:
                for style in ("generic", "specific"):
                    lines.append(
                        f"| {slot} | {style} | " + " | ".join(cell(styles[arm][style][slot]) for arm in ARMS) + " |"
                    )
        lines += ["", "### What each arm suggested", ""]
        for arm in ARMS:
            for slot in ENV_TRIAD_SLOTS:
                s = result["arms"][arm]["scores"][slot]
                top = ", ".join(f"`{v}` ×{n}" for v, n in list(s["values"].items())[:5]) or "none scored"
                lines.append(f"- **{arm} / {slot}**: {top}")
        lines += [
            "",
            "### Change from samples-only to full-workflow (graded by ENVO proximity)",
            "",
            "| Slot | changed | toward NMDC | away from NMDC |",
            "|---|---|---|---|",
        ]
        for slot in ENV_TRIAD_SLOTS:
            d = result["deltas"][slot]
            lines.append(
                f"| {slot} | {d['changed']}/{d['n_compared']} | {d['toward_reference']} | {d['away_from_reference']} |"
            )
    lines.append("")
    return "\n".join(lines)


def write_term_scores(path: Path, rows: list[dict[str, Any]]) -> None:
    """One row per scored suggestion, per study and arm; lists joined with ' | '."""
    if not rows:
        return
    with open(path, "x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: " | ".join(v) if isinstance(v, list) else v for k, v in row.items()})


def append_summary(run: dict[str, Any]) -> None:
    """One row per run in summary.tsv: mean ontology score per arm and slot, pooled and per study."""
    path = RESULTS_DIR / "summary.tsv"
    row: dict[str, Any] = {
        "timestamp": run["timestamp"],
        "model": run["model"],
        "provider": run["provider"],
        "chunk_size": run["chunk_size"],
        "studies": ",".join(run["studies"]),
    }
    for arm in ARMS:
        for slot in ENV_TRIAD_SLOTS:
            row[f"pooled_{arm}_{slot}"] = run["pooled"][arm][slot]["mean_ontology_score"]
    for key, result in run["studies"].items():
        for arm in ARMS:
            row[f"{key}_{arm}_answered"] = result["arms"][arm]["coverage"]["samples_with_suggestions"]
            for slot in ENV_TRIAD_SLOTS:
                row[f"{key}_{arm}_{slot}"] = result["arms"][arm]["scores"][slot]["mean_ontology_score"]
    new_file = not path.exists()
    if not new_file:
        with open(path, newline="") as f:
            header = next(csv.reader(f, delimiter="\t"), [])
        if header != list(row):
            # A run over a different set of studies has different columns; keep the
            # file readable by starting a new one rather than misaligning rows.
            path = RESULTS_DIR / f"summary_{'-'.join(run['studies'])}.tsv"
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
    # The pipeline's own default is 50; at 50, gemini-2.5-flash dropped ids and
    # truncated JSON in the supplement-triad eval. 25 matches that eval.
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument("--studies", nargs="+", choices=list(studies.CASES), default=list(studies.CASES))
    parser.add_argument("--limit", type=int, default=None, help="only the first N biosamples of each study")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    llm_client = LLMClient(access_provider=args.provider, model=args.model)
    adapter = get_envo_adapter()
    results: dict[str, Any] = {}
    term_rows: list[dict[str, Any]] = []
    for key in args.studies:
        results[key], rows = run_study(studies.CASES[key], llm_client, args.chunk_size, args.limit, adapter)
        term_rows.extend(rows)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run: dict[str, Any] = {
        "eval_name": "workflow-triad",
        "model": llm_client.model,
        "provider": llm_client.access_provider,
        "chunk_size": args.chunk_size,
        "limit": args.limit,
        "timestamp": timestamp,
        "pooled": {arm: pooled(term_rows, arm) for arm in ARMS},
        "studies": results,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    model_slug = str(llm_client.model).replace("/", "-").replace("@", "-")
    result_path = RESULTS_DIR / f"{model_slug}_{llm_client.access_provider}_{timestamp}.yaml"
    # A result file is the record of one run: "x" fails on a same-second collision
    # instead of overwriting, matching the convention in .gitignore and trace_report.
    with open(result_path, "x") as f:
        yaml.dump(run, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    write_term_scores(result_path.with_name(result_path.stem + "_scores.tsv"), term_rows)
    report = render_report(run)
    (RESULTS_DIR / "report.md").write_text(report)
    append_summary(run)
    print(report)
    print(f"\nResults: {result_path}")


if __name__ == "__main__":
    main()
