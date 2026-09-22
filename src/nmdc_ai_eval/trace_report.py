"""Report on production traces without needing a reference answer.

Two questions per run, both answerable from what Langfuse already holds:

* **What did the agent do?** Turns, permission denials, terminal reason, cost. A run can report
  ``terminal_reason: completed`` having been denied every tool call it made, so a well formed
  answer is not evidence the work happened.
* **Is what it produced well formed?** Every env triad value is checked for parseability as
  ``label [CURIE]``, for the CURIE existing in ENVO, and for the label matching ENVO's own label.
  None of those checks needs a curated answer, which is what lets this run over production traffic
  where no ground truth exists.

The second half reuses ``envo_scorer.parse_label_curie`` and ``envo_scorer.validate_curie_label``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

from nmdc_ai_eval.envo_scorer import parse_label_curie

ENV_TRIAD_SLOTS = ("env_broad_scale", "env_local_scale", "env_medium")
MAPPER_BUCKETS = ("high_confidence", "needs_review", "cant_place")


@dataclass
class TermCheck:
    """Reference-free verdict on one suggested ``label [CURIE]`` value."""

    slot: str
    raw_value: str
    parsed: bool
    label: str | None = None
    curie: str | None = None
    prefix: str | None = None
    curie_resolves: bool | None = None
    label_matches: bool | None = None
    canonical_label: str | None = None

    @property
    def well_formed(self) -> bool:
        return bool(self.parsed and self.curie_resolves and self.label_matches)


def _iter_triad_values(output: dict[str, Any]) -> list[tuple[str, str]]:
    """Every (slot, raw value) pair in an ``LLMOutput``-shaped payload."""
    pairs: list[tuple[str, str]] = []
    for suggestion in output.get("metadata_fields") or []:
        if not isinstance(suggestion, dict):
            continue
        slot = suggestion.get("field_name")
        if slot not in ENV_TRIAD_SLOTS:
            continue
        value = suggestion.get("value")
        if not isinstance(value, str) or not value.strip():
            continue
        # env_medium is pipe-joined in some references and some outputs.
        for piece in value.split("|"):
            piece = piece.strip()
            if piece:
                pairs.append((str(slot), piece))
    return pairs


def check_term(slot: str, raw_value: str, label_lookup: Any = None) -> TermCheck:
    """Check one value. ``label_lookup`` takes a CURIE and returns ENVO's label or None."""
    parsed = parse_label_curie(raw_value)
    if parsed is None:
        return TermCheck(slot=slot, raw_value=raw_value, parsed=False)
    label, curie = parsed
    prefix = curie.split(":", 1)[0] if ":" in curie else None
    check = TermCheck(slot=slot, raw_value=raw_value, parsed=True, label=label, curie=curie, prefix=prefix)
    if label_lookup is None:
        return check
    canonical = label_lookup(curie)
    check.curie_resolves = canonical is not None
    check.canonical_label = canonical
    check.label_matches = bool(canonical and canonical.lower() == label.lower())
    return check


@dataclass
class TraceRow:
    """One row per trace: what the agent did, and what it produced."""

    trace_id: str
    timestamp: str
    name: str | None
    environment: str | None
    output_shape: str
    served_model: str | None
    declared_model: str | None
    model_attribution_disagrees: bool
    num_turns: int | None
    permission_denials: int | None
    terminal_reason: str | None
    is_error: Any
    total_cost_usd: float | None
    duration_ms: int | None
    triad_values: int = 0
    triad_parsed: int = 0
    triad_curie_resolves: int = 0
    triad_label_matches: int = 0
    triad_well_formed: int = 0
    non_envo_prefixes: str = ""
    mapper_high_confidence: int | None = None
    mapper_needs_review: int | None = None
    mapper_cant_place: int | None = None
    checks: list[TermCheck] = field(default_factory=list)

    def as_tsv_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row.pop("checks", None)
        return row


def _output_shape(output: Any) -> str:
    if output is None:
        return "none"
    if isinstance(output, dict):
        if "metadata_fields" in output:
            return "LLMOutput"
        if "high_confidence" in output:
            return "MetadataMapperOutput"
        return "dict:other"
    return f"not-a-dict:{type(output).__name__}"


def build_row(bundle: Any, label_lookup: Any = None) -> TraceRow:
    """Build one row from a ``langfuse_reader.TraceBundle``."""
    trace = bundle.trace
    metadata = trace.get("metadata") or {}
    output = trace.get("output")
    shape = _output_shape(output)

    row = TraceRow(
        trace_id=str(trace.get("id", "")),
        timestamp=str(trace.get("timestamp", ""))[:19],
        name=trace.get("name"),
        environment=trace.get("environment"),
        output_shape=shape,
        served_model=bundle.served_model,
        declared_model=bundle.declared_model,
        model_attribution_disagrees=bundle.model_attribution_disagrees,
        num_turns=metadata.get("num_turns"),
        permission_denials=metadata.get("permission_denials"),
        terminal_reason=metadata.get("terminal_reason"),
        is_error=metadata.get("is_error"),
        total_cost_usd=metadata.get("total_cost_usd"),
        duration_ms=metadata.get("duration_ms"),
    )

    if shape == "LLMOutput" and isinstance(output, dict):
        prefixes: Counter[str] = Counter()
        for slot, raw in _iter_triad_values(output):
            check = check_term(slot, raw, label_lookup)
            row.checks.append(check)
            row.triad_values += 1
            row.triad_parsed += int(check.parsed)
            row.triad_curie_resolves += int(bool(check.curie_resolves))
            row.triad_label_matches += int(bool(check.label_matches))
            row.triad_well_formed += int(check.well_formed)
            if check.prefix and check.prefix != "ENVO":
                prefixes[check.prefix] += 1
        row.non_envo_prefixes = ";".join(f"{p}={n}" for p, n in sorted(prefixes.items()))

    if shape == "MetadataMapperOutput" and isinstance(output, dict):
        bucket_attrs = ("mapper_high_confidence", "mapper_needs_review", "mapper_cant_place")
        for bucket, attr in zip(MAPPER_BUCKETS, bucket_attrs, strict=True):
            entries = output.get(bucket)
            setattr(row, attr, len(entries) if isinstance(entries, list) else None)

    return row


def summarize(rows: list[TraceRow]) -> dict[str, Any]:
    """Corpus-level counts. Every denominator is stated so no rate is read without one."""
    with_health = [r for r in rows if r.permission_denials is not None]
    llm_rows = [r for r in rows if r.output_shape == "LLMOutput"]
    values = sum(r.triad_values for r in llm_rows)
    return {
        "traces": len(rows),
        "date_first": min((r.timestamp for r in rows), default=""),
        "date_last": max((r.timestamp for r in rows), default=""),
        "output_shapes": dict(Counter(r.output_shape for r in rows).most_common()),
        "trace_names": dict(Counter(r.name or "<none>" for r in rows).most_common()),
        "model_attribution_disagreements": sum(1 for r in rows if r.model_attribution_disagrees),
        "traces_with_health_block": len(with_health),
        "traces_with_any_denial": sum(1 for r in with_health if (r.permission_denials or 0) > 0),
        "max_denials_in_one_run": max((r.permission_denials or 0 for r in with_health), default=0),
        "completed_despite_denials": sum(
            1 for r in with_health if (r.permission_denials or 0) > 0 and r.terminal_reason == "completed"
        ),
        "total_cost_usd": round(sum(r.total_cost_usd or 0.0 for r in rows), 4),
        "triad_values": values,
        "triad_parsed": sum(r.triad_parsed for r in llm_rows),
        "triad_curie_resolves": sum(r.triad_curie_resolves for r in llm_rows),
        "triad_label_matches": sum(r.triad_label_matches for r in llm_rows),
        "triad_well_formed": sum(r.triad_well_formed for r in llm_rows),
        "non_envo_prefixes": dict(
            Counter(p.split("=")[0] for r in llm_rows for p in r.non_envo_prefixes.split(";") if p).most_common()
        ),
    }


def _markdown(summary: dict[str, Any]) -> str:
    """A short report suitable for pasting into an issue."""
    s = summary
    lines = [
        "# Production trace report",
        "",
        f"{s['traces']} traces, {s['date_first'][:10]} to {s['date_last'][:10]}. "
        f"No model was called to produce this report.",
        "",
        "## What the agent did",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| traces carrying a run health block | {s['traces_with_health_block']} of {s['traces']} |",
        f"| of those, hit at least one permission denial | {s['traces_with_any_denial']} |",
        f"| of those, reported `completed` anyway | {s['completed_despite_denials']} |",
        f"| most denials in a single run | {s['max_denials_in_one_run']} |",
        f"| total cost recorded | ${s['total_cost_usd']} |",
        f"| traces whose metadata names a different model than the AGENT span | "
        f"{s['model_attribution_disagreements']} |",
        "",
        "## Whether the env triad values are well formed",
        "",
        "Reference free. Each check needs ENVO, not a curated answer.",
        "",
        "| Check | Count | Denominator |",
        "|---|---|---|",
        f"| values suggested | {s['triad_values']} | |",
        f"| parse as `label [CURIE]` | {s['triad_parsed']} | {s['triad_values']} |",
        f"| CURIE resolves in ENVO | {s['triad_curie_resolves']} | {s['triad_values']} |",
        f"| label matches ENVO's label | {s['triad_label_matches']} | {s['triad_values']} |",
        f"| all three | {s['triad_well_formed']} | {s['triad_values']} |",
        "",
        f"Non-ENVO prefixes seen: {s['non_envo_prefixes'] or 'none'}",
        "",
        "## Output shapes",
        "",
        "| Shape | Traces |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v} |" for k, v in s["output_shapes"].items()]
    lines += ["", "## Trace names", "", "| Name | Traces |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in s["trace_names"].items()]
    return "\n".join(lines) + "\n"


def main() -> None:  # pragma: no cover - thin CLI over tested functions
    import argparse
    import csv
    import json
    import sys
    from datetime import UTC, datetime
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("trace-reports"))
    parser.add_argument(
        "--no-envo",
        action="store_true",
        help="Skip ENVO lookups. Parse checks still run; resolution and label checks are left blank.",
    )
    parser.add_argument(
        "--envo-db",
        type=Path,
        default=None,
        help=(
            "Path to a local ENVO sqlite build, used instead of oaklib's mirror. "
            "The bbop-sqlite mirror returned HTTP 403 on 2026-09-22, which is the outage "
            "microbiomedata/nmdc-ai-eval#112 adds a fallback for."
        ),
    )
    args = parser.parse_args()

    from nmdc_ai_eval.langfuse_reader import LangfuseCredentialsMissing, endpoint_from_env, load_bundles

    try:
        endpoint = endpoint_from_env()
    except LangfuseCredentialsMissing as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"Reading traces from {endpoint.base_url}", file=sys.stderr)
    bundles = load_bundles(endpoint)
    print(f"  {len(bundles)} traces", file=sys.stderr)

    label_lookup = None
    if not args.no_envo:
        adapter = None
        try:
            if args.envo_db:
                from oaklib import get_adapter

                adapter = get_adapter(f"sqlite:{args.envo_db}")
            else:
                from nmdc_ai_eval.envo_scorer import get_envo_adapter

                adapter = get_envo_adapter()
        except Exception as exc:  # noqa: BLE001 - an unreachable ontology is a finding, not a crash
            print(
                f"ENVO unavailable ({type(exc).__name__}). Parse checks still run; "
                "CURIE resolution and label checks are left blank. "
                "Pass --envo-db PATH to use a local build.",
                file=sys.stderr,
            )
        cache: dict[str, str | None] = {}

        if adapter is not None:

            def label_lookup(curie: str) -> str | None:  # type: ignore[misc]
                if curie not in cache:
                    try:
                        cache[curie] = adapter.label(curie)
                    except Exception:  # noqa: BLE001 - an unresolvable CURIE is a finding, not a crash
                        cache[curie] = None
                return cache[curie]

    rows = [build_row(b, label_lookup) for b in bundles]
    summary = summarize(rows)

    # Timestamped and never overwritten, matching the convention stated in .gitignore
    # for datasets/*/pipeline-results/: a result file is a record of one run.
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = args.output_dir / f"traces_{stamp}.tsv"
    with open(tsv_path, "w", newline="") as handle:
        fieldnames = list(rows[0].as_tsv_dict()) if rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_tsv_dict())

    (args.output_dir / f"summary_{stamp}.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / f"report_{stamp}.md").write_text(_markdown(summary))

    print(_markdown(summary))
    print(f"Wrote {tsv_path} and its summary and report", file=sys.stderr)


if __name__ == "__main__":  # pragma: no cover
    main()
