"""Unit tests for triad_output_scorer — a toy ENVO graph, no oaklib download."""

from nmdc_metadata_suggestor_ai_tool.evaluation.env_triad_scoring import (
    TriadTerm,
    compare_outputs,
    reference_from_rows,
)
from nmdc_metadata_suggestor_ai_tool.models.llm_output import (
    LLMOutput,
    MetadataFieldSuggestion,
    TriadProvenance,
)

from nmdc_ai_eval.envo_scorer import ANCESTOR_DECAY, DESCENDANT_DECAY, W_ENUM, W_LABEL, W_PARSE
from nmdc_ai_eval.triad_output_scorer import (
    closest_reference,
    hierarchy_scorer,
    score_output,
    score_term,
    summarize,
)

# biome > terrestrial biome > anthropogenic terrestrial biome > cropland biome
PARENTS = {
    "ENVO:00000446": ["ENVO:00000428"],
    "ENVO:01000219": ["ENVO:00000446"],
    "ENVO:01000245": ["ENVO:01000219"],
    "ENVO:00002259": ["ENVO:00001998"],
    "ENVO:00001998": ["ENVO:00010483"],
    "ENVO:01001121": ["ENVO:00010483"],
}
LABELS = {
    "ENVO:00000428": "biome",
    "ENVO:00000446": "terrestrial biome",
    "ENVO:01000219": "anthropogenic terrestrial biome",
    "ENVO:01000245": "cropland biome",
    "ENVO:00001998": "soil",
    "ENVO:00002259": "agricultural soil",
    "ENVO:01001121": "plant matter",
    "ENVO:00010483": "environmental material",
}


class FakeAdapter:
    def label(self, curie: str) -> str | None:
        return LABELS.get(curie)

    def hierarchical_parents(self, curie: str) -> list[str]:
        return PARENTS.get(curie, [])

    def ancestors(self, curie: str, predicates: list[str] | None = None) -> list[str]:
        seen: list[str] = []
        frontier = list(PARENTS.get(curie, []))
        while frontier:
            node = frontier.pop()
            if node not in seen:
                seen.append(node)
                frontier.extend(PARENTS.get(node, []))
        return seen


ADAPTER = FakeAdapter()


def suggestion(sample_id: str, slot: str, value: str, tier: str = "submission_enum") -> MetadataFieldSuggestion:
    return MetadataFieldSuggestion(
        id=sample_id,
        field_name=slot,
        reason="",
        value=value,
        provenance=TriadProvenance(tier=tier, outcome="accepted"),  # type: ignore[arg-type]
    )


def test_closest_reference_picks_the_nearest_of_several_terms() -> None:
    reference = [TriadTerm("agricultural soil", "ENVO:00002259"), TriadTerm("plant matter", "ENVO:01001121")]
    relationship, hops, closest = closest_reference(ADAPTER, "ENVO:00001998", reference)
    assert (relationship, hops) == ("ancestor", 1)
    assert closest is not None and closest.curie == "ENVO:00002259"


def test_score_term_exact_descendant_ancestor_and_unrelated() -> None:
    ref = [TriadTerm("terrestrial biome", "ENVO:00000446")]
    exact = score_term(ADAPTER, "s", "env_broad_scale", TriadTerm("terrestrial biome", "ENVO:00000446"), ref)
    assert exact.relationship == "exact" and exact.exact_match
    assert exact.ontology_score == W_PARSE + W_LABEL + 0.5 + W_ENUM * 0.5  # no provenance: enum neutral

    descendant = score_term(
        ADAPTER,
        "s",
        "env_broad_scale",
        TriadTerm("cropland biome", "ENVO:01000245"),
        ref,
        TriadProvenance(tier="submission_enum", outcome="accepted"),
    )
    assert (descendant.relationship, descendant.hop_distance, descendant.in_enum) == ("descendant", 2, True)
    assert descendant.hierarchy_score == 1.0 - 2 * DESCENDANT_DECAY

    ancestor = score_term(ADAPTER, "s", "env_broad_scale", TriadTerm("biome", "ENVO:00000428"), ref)
    assert ancestor.hierarchy_score == 1.0 - ANCESTOR_DECAY

    unrelated = score_term(ADAPTER, "s", "env_broad_scale", TriadTerm("soil", "ENVO:00001998"), ref)
    assert unrelated.relationship == "unrelated" and unrelated.hierarchy_score == 0.0


def test_score_term_flags_label_mismatch_and_unparseable_values() -> None:
    ref = [TriadTerm("terrestrial biome", "ENVO:00000446")]
    wrong_label = score_term(ADAPTER, "s", "env_broad_scale", TriadTerm("agricultural biome", "ENVO:00000446"), ref)
    assert wrong_label.curie_label_valid is False and wrong_label.exact_match
    unparsed = score_term(ADAPTER, "s", "env_broad_scale", TriadTerm("just words", ""), ref)
    assert unparsed.parse_success is False and unparsed.ontology_score == 0.0


def test_score_output_skips_missing_suggestions_and_empty_references() -> None:
    reference = reference_from_rows(
        [
            {"sample_name": "a", "env_broad_scale": "Terrestrial Biome [ENVO_00000446]", "env_medium": ""},
            {"sample_name": "b", "env_broad_scale": "Terrestrial Biome [ENVO_00000446]"},
        ]
    )
    output = LLMOutput(
        metadata_fields=[
            suggestion("a", "env_broad_scale", "cropland biome [ENVO:01000245]"),
            suggestion("a", "env_medium", "soil [ENVO:00001998]"),  # reference empty: skipped
        ]
    )
    scores = score_output(output, reference, ["a", "b"], adapter=ADAPTER)
    assert [(s.sample_id, s.slot, s.relationship) for s in scores] == [("a", "env_broad_scale", "descendant")]

    summary = summarize(scores)
    broad = summary["env_broad_scale"]
    assert broad["n_scored"] == 1
    assert broad["relationships"] == {"exact": 0, "descendant": 1, "ancestor": 0, "unrelated": 0}
    assert broad["exact_match_rate"] == 0.0
    assert broad["in_enum"] == 1
    assert broad["values"] == {"cropland biome [ENVO:01000245]": 1}
    assert summary["env_medium"]["n_scored"] == 0
    assert summary["env_medium"]["mean_ontology_score"] is None


def test_hierarchy_scorer_makes_compare_outputs_grade_by_proximity() -> None:
    reference = reference_from_rows([{"sample_name": "a", "env_broad_scale": "terrestrial biome [ENVO:00000446]"}])
    baseline = LLMOutput(metadata_fields=[suggestion("a", "env_broad_scale", "biome [ENVO:00000428]")])
    treatment = LLMOutput(metadata_fields=[suggestion("a", "env_broad_scale", "cropland biome [ENVO:01000245]")])
    delta = compare_outputs(baseline, treatment, reference, ["a"], scorer=hierarchy_scorer(ADAPTER))["env_broad_scale"]
    # ancestor 1 hop (0.85) -> descendant 2 hops (0.8): away, though neither is exact
    assert (delta.changed, delta.toward_reference, delta.away_from_reference) == (1, 0, 1)
