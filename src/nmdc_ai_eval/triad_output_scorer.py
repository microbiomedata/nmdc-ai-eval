"""Ontology-aware scoring of a suggestor ``LLMOutput`` against per-sample references.

The suggestor's env triad pipeline returns one ``LLMOutput`` for many samples,
each suggestion tagged with the sample id and the validation gate's provenance.
This module scores every triad suggestion in such an output against a reference
triad for that sample, using the same ENVO relationship, hop distance, and
composite formula as :mod:`nmdc_ai_eval.envo_scorer`, and summarizes per slot.

What comes from where:

* Reference parsing, sample joining, and arm comparison come from the suggestor's
  ``evaluation.env_triad_scoring`` (it knows the dialects references arrive in).
* Relationship, hop distance, label validation, and the score formula come from
  ``envo_scorer`` (it knows ENVO).
* The enum term of the score comes from the gate's provenance: ``submission_enum``
  means the value is in the extension's curated set. The TSV enum snapshots in
  ``datasets/ebs-prediction/enum_data`` cover ``env_broad_scale`` only, and the
  gate already checked all three slots against the live schema.

A reference cell may hold several terms (``agricultural soil | plant matter``).
The prediction is scored against the one it sits closest to in ENVO.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from nmdc_metadata_suggestor_ai_tool.evaluation.env_triad_scoring import (
    Reference,
    Scorer,
    TriadTerm,
    suggested_term,
    triad_suggestions,
)
from nmdc_metadata_suggestor_ai_tool.models.llm_output import LLMOutput, TriadProvenance

from nmdc_ai_eval.envo_scorer import (
    check_relationship,
    compute_hierarchy_score,
    compute_hop_distance,
    compute_ontology_score,
    get_envo_adapter,
    validate_curie_label,
)

if TYPE_CHECKING:
    from oaklib.interfaces import OboGraphInterface  # type: ignore[import-untyped]

ENV_TRIAD_SLOTS = ("env_broad_scale", "env_local_scale", "env_medium")
RELATIONSHIPS = ("exact", "descendant", "ancestor", "unrelated")


@dataclass
class TermScore:
    """One suggestion scored against one sample's reference cell."""

    sample_id: str
    slot: str
    predicted: str | None
    reference_values: list[str]
    parse_success: bool
    curie_label_valid: bool | None
    relationship: str | None
    hop_distance: int | None
    closest_reference: str | None
    in_enum: bool | None
    tier: str | None
    outcome: str | None
    hierarchy_score: float
    ontology_score: float

    @property
    def exact_match(self) -> bool:
        return self.relationship == "exact"

    def as_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["hierarchy_score"] = round(self.hierarchy_score, 4)
        row["ontology_score"] = round(self.ontology_score, 4)
        return {**row, "exact_match": self.exact_match}


def closest_reference(
    adapter: OboGraphInterface, pred_curie: str, reference_terms: Iterable[TriadTerm]
) -> tuple[str | None, int | None, TriadTerm | None]:
    """Relationship and hops to whichever reference term scores highest on the hierarchy."""
    best: tuple[float, str | None, int | None, TriadTerm | None] = (-1.0, None, None, None)
    for ref in reference_terms:
        if not ref.curie:
            continue
        relationship = check_relationship(adapter, pred_curie, ref.curie)
        hops = compute_hop_distance(adapter, pred_curie, ref.curie)
        score = compute_hierarchy_score(relationship, hops)
        if score > best[0]:
            best = (score, relationship, hops, ref)
    return best[1], best[2], best[3]


def enum_membership(provenance: Any) -> bool | None:
    """Whether the gate found the value in the extension's curated set; None when unknown."""
    if isinstance(provenance, TriadProvenance):
        return bool(provenance.tier == "submission_enum")
    return None


def score_term(
    adapter: OboGraphInterface,
    sample_id: str,
    slot: str,
    term: TriadTerm | None,
    reference_terms: list[TriadTerm],
    provenance: Any = None,
) -> TermScore:
    """Score one suggested term against one reference cell."""
    reference_values = [t.value for t in reference_terms]
    if term is None or not term.curie:
        return TermScore(
            sample_id=sample_id,
            slot=slot,
            predicted=term.value if term else None,
            reference_values=reference_values,
            parse_success=False,
            curie_label_valid=None,
            relationship=None,
            hop_distance=None,
            closest_reference=None,
            in_enum=enum_membership(provenance),
            tier=getattr(provenance, "tier", None),
            outcome=getattr(provenance, "outcome", None),
            hierarchy_score=0.0,
            ontology_score=0.0,
        )
    label_valid = validate_curie_label(adapter, term.curie, term.label)
    relationship, hops, closest = closest_reference(adapter, term.curie, reference_terms)
    in_enum = enum_membership(provenance)
    return TermScore(
        sample_id=sample_id,
        slot=slot,
        predicted=term.value,
        reference_values=reference_values,
        parse_success=True,
        curie_label_valid=label_valid,
        relationship=relationship,
        hop_distance=hops,
        closest_reference=closest.value if closest else None,
        in_enum=in_enum,
        tier=getattr(provenance, "tier", None),
        outcome=getattr(provenance, "outcome", None),
        hierarchy_score=compute_hierarchy_score(relationship, hops),
        ontology_score=compute_ontology_score(
            parse_success=True,
            curie_label_valid=label_valid,
            relationship=relationship,
            hop_distance=hops,
            in_template_enum=in_enum,
        ),
    )


def score_output(
    output: LLMOutput,
    reference: Reference,
    sample_ids: Iterable[str],
    adapter: OboGraphInterface | None = None,
) -> list[TermScore]:
    """Score every triad suggestion in *output* for the given samples.

    Samples with no suggestion for a slot get no row; see the suggestor's
    ``coverage`` for those. Samples whose reference cell is empty for a slot are
    skipped too, since there is nothing to score against.
    """
    adapter = adapter or get_envo_adapter()
    suggestions = triad_suggestions(output)
    scores: list[TermScore] = []
    for sample_id in sample_ids:
        for slot in ENV_TRIAD_SLOTS:
            suggestion = suggestions.get((sample_id, slot))
            reference_terms = [t for t in reference.get(sample_id, {}).get(slot, []) if t.curie]
            if suggestion is None or not reference_terms:
                continue
            scores.append(
                score_term(
                    adapter,
                    sample_id,
                    slot,
                    suggested_term(suggestion),
                    reference_terms,
                    suggestion.provenance,
                )
            )
    return scores


def summarize(scores: Iterable[TermScore]) -> dict[str, dict[str, Any]]:
    """Per-slot means and relationship counts, in the shape the YAML results carry."""
    by_slot: dict[str, list[TermScore]] = {slot: [] for slot in ENV_TRIAD_SLOTS}
    for score in scores:
        by_slot.setdefault(score.slot, []).append(score)
    summary: dict[str, dict[str, Any]] = {}
    for slot, rows in by_slot.items():
        n = len(rows)
        relationships = Counter(r.relationship or "unparsed" for r in rows)
        summary[slot] = {
            "n_scored": n,
            "mean_ontology_score": round(sum(r.ontology_score for r in rows) / n, 4) if n else None,
            "mean_hierarchy_score": round(sum(r.hierarchy_score for r in rows) / n, 4) if n else None,
            "exact_match_rate": round(sum(r.exact_match for r in rows) / n, 4) if n else None,
            "relationships": {rel: relationships.get(rel, 0) for rel in RELATIONSHIPS},
            "unparsed": relationships.get("unparsed", 0),
            "curie_label_valid": sum(1 for r in rows if r.curie_label_valid),
            "in_enum": sum(1 for r in rows if r.in_enum),
            "tiers": dict(Counter(r.tier for r in rows if r.tier)),
            "outcomes": dict(Counter(r.outcome for r in rows if r.outcome)),
            "values": dict(Counter(r.predicted for r in rows if r.predicted).most_common()),
        }
    return summary


def hierarchy_scorer(adapter: OboGraphInterface | None = None) -> Scorer:
    """A ``Scorer`` for the suggestor's ``compare_outputs`` that grades by ENVO proximity.

    With it, a move from ``cropland biome`` to ``terrestrial biome`` against a
    ``terrestrial biome`` reference counts as toward the reference by 0.1, not
    as a jump from 0 to 1.
    """
    resolved = adapter or get_envo_adapter()

    def scorer(term: TriadTerm | None, reference_terms: list[TriadTerm]) -> float:
        if term is None or not term.curie:
            return 0.0
        relationship, hops, _ = closest_reference(resolved, term.curie, reference_terms)
        return compute_hierarchy_score(relationship, hops)

    return scorer
