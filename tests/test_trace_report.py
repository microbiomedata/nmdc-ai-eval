"""Tests for the reference-free production trace report. No network, no model, no ontology."""

from types import SimpleNamespace

from nmdc_ai_eval.trace_report import TraceRow, build_row, check_term, summarize

# A hand-written ENVO fragment. Two real terms, one label that does not match its CURIE.
FAKE_ENVO = {
    "ENVO:00000446": "terrestrial biome",
    "ENVO:00000218": "black smoker",
    "ENVO:00001998": "soil",
}


def fake_lookup(curie: str) -> str | None:
    return FAKE_ENVO.get(curie)


def _bundle(output, metadata=None, served="claude-opus-4-6", declared=None, name="agentic"):
    trace = {
        "id": "t1",
        "timestamp": "2026-09-01T00:00:00.000Z",
        "name": name,
        "environment": "local",
        "output": output,
        "metadata": metadata or {},
    }
    return SimpleNamespace(
        trace=trace,
        served_model=served,
        declared_model=declared,
        model_attribution_disagrees=bool(served and declared and served != declared),
    )


def test_check_term_accepts_a_matching_label() -> None:
    check = check_term("env_broad_scale", "terrestrial biome [ENVO:00000446]", fake_lookup)
    assert check.parsed and check.curie_resolves and check.label_matches
    assert check.well_formed


def test_check_term_catches_a_label_naming_a_different_term() -> None:
    """The production failure: a plausible label attached to an unrelated CURIE."""
    check = check_term("env_local_scale", "mountainous area [ENVO:00000218]", fake_lookup)
    assert check.parsed
    assert check.curie_resolves
    assert check.label_matches is False
    assert check.canonical_label == "black smoker"
    assert not check.well_formed


def test_check_term_reports_an_unresolvable_curie() -> None:
    check = check_term("env_medium", "invented thing [ENVO:99999999]", fake_lookup)
    assert check.parsed
    assert check.curie_resolves is False
    assert not check.well_formed


def test_check_term_reports_an_unparsable_value() -> None:
    check = check_term("env_medium", "just some prose", fake_lookup)
    assert check.parsed is False
    assert check.curie is None


def test_check_term_without_a_lookup_still_parses() -> None:
    """Degraded mode: ENVO unreachable, so resolution and label checks stay blank."""
    check = check_term("env_medium", "soil [ENVO:00001998]", None)
    assert check.parsed
    assert check.curie_resolves is None
    assert check.label_matches is None


def test_build_row_splits_pipe_joined_env_medium() -> None:
    output = {
        "metadata_fields": [
            {"field_name": "env_medium", "value": "soil [ENVO:00001998] | terrestrial biome [ENVO:00000446]"}
        ]
    }
    row = build_row(_bundle(output), fake_lookup)
    assert row.triad_values == 2
    assert row.triad_well_formed == 2


def test_build_row_ignores_non_triad_slots() -> None:
    output = {"metadata_fields": [{"field_name": "samp_name", "value": "soil [ENVO:00001998]"}]}
    row = build_row(_bundle(output), fake_lookup)
    assert row.triad_values == 0


def test_build_row_records_non_envo_prefixes() -> None:
    output = {"metadata_fields": [{"field_name": "env_medium", "value": "leaf [PO:0025034]"}]}
    row = build_row(_bundle(output), fake_lookup)
    assert row.non_envo_prefixes == "PO=1"


def test_build_row_carries_the_run_health_block() -> None:
    metadata = {
        "num_turns": 214,
        "permission_denials": 205,
        "terminal_reason": "completed",
        "is_error": False,
        "total_cost_usd": 2.5,
    }
    row = build_row(_bundle({"metadata_fields": []}, metadata), fake_lookup)
    assert row.permission_denials == 205
    assert row.terminal_reason == "completed"


def test_build_row_counts_mapper_confidence_buckets() -> None:
    output = {"high_confidence": [{}, {}], "needs_review": [{}], "cant_place": []}
    row = build_row(_bundle(output, name="metadata_mapper_agentic"), fake_lookup)
    assert row.output_shape == "MetadataMapperOutput"
    assert (row.mapper_high_confidence, row.mapper_needs_review, row.mapper_cant_place) == (2, 1, 0)


def test_build_row_flags_model_attribution_disagreement() -> None:
    bundle = _bundle({"metadata_fields": []}, served="claude-opus-4-6", declared="gemini-2.5-flash")
    row = build_row(bundle, fake_lookup)
    assert row.model_attribution_disagrees


def test_summarize_states_a_denominator_for_every_rate() -> None:
    rows = [
        TraceRow(
            trace_id="a",
            timestamp="2026-09-01T00:00:00",
            name="agentic",
            environment="local",
            output_shape="LLMOutput",
            served_model="m",
            declared_model="m",
            model_attribution_disagrees=False,
            num_turns=10,
            permission_denials=5,
            terminal_reason="completed",
            is_error=False,
            total_cost_usd=1.0,
            duration_ms=1,
            triad_values=3,
            triad_parsed=3,
            triad_curie_resolves=3,
            triad_label_matches=2,
            triad_well_formed=2,
        ),
        TraceRow(
            trace_id="b",
            timestamp="2026-09-02T00:00:00",
            name="agentic",
            environment="local",
            output_shape="none",
            served_model=None,
            declared_model=None,
            model_attribution_disagrees=False,
            num_turns=None,
            permission_denials=None,
            terminal_reason=None,
            is_error=None,
            total_cost_usd=None,
            duration_ms=None,
        ),
    ]
    summary = summarize(rows)
    assert summary["traces"] == 2
    assert summary["traces_with_health_block"] == 1
    assert summary["traces_with_any_denial"] == 1
    assert summary["completed_despite_denials"] == 1
    assert summary["triad_values"] == 3
    assert summary["triad_well_formed"] == 2
