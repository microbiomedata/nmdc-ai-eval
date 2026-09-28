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
    assert check.checked_against_envo is False
    assert check.well_formed is None, "unknown must not collapse to wrong"


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


def test_markdown_states_a_denominator_beside_every_count() -> None:
    """A rate without its denominator is the thing this report exists to avoid."""
    from nmdc_ai_eval.trace_report import _markdown

    summary = {
        "traces": 84,
        "environments": {"local": 76, "testing": 7, "default": 1},
        "date_first": "2026-07-14T00:00:00",
        "date_last": "2026-09-22T00:00:00",
        "output_shapes": {"LLMOutput": 32, "none": 34},
        "trace_names": {"agentic": 50},
        "model_attribution_disagreements": 21,
        "traces_with_health_block": 25,
        "traces_with_any_denial": 12,
        "max_denials_in_one_run": 205,
        "completed_despite_denials": 12,
        "total_cost_usd": 25.33,
        "triad_values": 918,
        "triad_checked_against_envo": 918,
        "triad_parsed": 918,
        "triad_with_verdict": 918,
        "triad_curie_resolves": 918,
        "triad_label_matches": 911,
        "triad_well_formed": 911,
        "non_envo_prefixes": {},
    }
    text = _markdown(summary)
    assert "84 traces, 2026-07-14 to 2026-09-22" in text
    assert "25 of 84" in text
    assert "| 911 | 918 |" in text
    assert "No model was called" in text
    assert "| LLMOutput | 32 |" in text
    assert "| agentic | 50 |" in text
    assert "Environments covered: local 76, testing 7, default 1." in text
    assert "Production trace report" not in text


def test_markdown_says_none_when_no_foreign_prefixes_were_seen() -> None:
    """Keys come from summarize() itself, so a new summary field cannot silently skip this."""
    from nmdc_ai_eval.trace_report import _markdown

    assert "Non-ENVO prefixes seen: none" in _markdown(summarize([]))


def test_markdown_explains_the_denominator_when_envo_was_unavailable() -> None:
    from nmdc_ai_eval.trace_report import _markdown

    summary = summarize([])
    summary["triad_values"] = 10
    summary["triad_checked_against_envo"] = 0
    text = _markdown(summary)
    assert "| looked up in ENVO | 0 | 10 |" in text
    assert "unknown is not the same as wrong" in text


def test_as_tsv_dict_drops_the_per_term_checks() -> None:
    """The TSV is one row per trace; the per-value detail would break that shape."""
    output = {"metadata_fields": [{"field_name": "env_medium", "value": "soil [ENVO:00001998]"}]}
    row = build_row(_bundle(output), fake_lookup)
    assert row.checks
    assert "checks" not in row.as_tsv_dict()
    assert row.as_tsv_dict()["triad_values"] == 1


def test_summarize_handles_an_empty_corpus() -> None:
    summary = summarize([])
    assert summary["traces"] == 0
    assert summary["max_denials_in_one_run"] == 0
    assert summary["date_first"] == ""


def test_build_row_tolerates_a_non_dict_output() -> None:
    for payload, expected in ((None, "none"), ([1, 2], "not-a-dict:list"), ("x", "not-a-dict:str")):
        assert build_row(_bundle(payload), fake_lookup).output_shape == expected


def test_build_row_skips_suggestions_with_no_usable_value() -> None:
    output = {
        "metadata_fields": [
            {"field_name": "env_medium", "value": ""},
            {"field_name": "env_medium", "value": 42},
            {"not_a_dict": True},
            "a bare string",
        ]
    }
    assert build_row(_bundle(output), fake_lookup).triad_values == 0


def _row(**overrides) -> TraceRow:
    base = dict(
        trace_id="r",
        timestamp="2026-09-01T00:00:00",
        name="agentic",
        environment="local",
        output_shape="LLMOutput",
        served_model="m",
        declared_model="m",
        model_attribution_disagrees=False,
        num_turns=None,
        permission_denials=None,
        terminal_reason=None,
        is_error=None,
        total_cost_usd=None,
        duration_ms=None,
    )
    base.update(overrides)
    return TraceRow(**base)


def test_summarize_adds_prefix_counts_across_rows() -> None:
    """PO=3 in one row and PO=2 in another is five values, not two rows."""
    rows = [_row(non_envo_prefixes="PO=3;UBERON=1"), _row(non_envo_prefixes="PO=2")]
    assert summarize(rows)["non_envo_prefixes"] == {"PO": 5, "UBERON": 1}


def test_a_failed_lookup_is_unknown_not_unresolved() -> None:
    """An unreadable ENVO database must not turn every value into a model failure."""
    from nmdc_ai_eval.trace_report import LookupUnavailable

    def broken_lookup(curie: str) -> str | None:
        raise LookupUnavailable("database is locked")

    check = check_term("env_medium", "soil [ENVO:00001998]", broken_lookup)
    assert check.parsed
    assert check.curie_resolves is None
    assert check.label_matches is None
    assert check.well_formed is None


def test_all_three_denominator_counts_malformed_values() -> None:
    """One valid value and one unparsable value is 1 of 2 well formed, not 1 of 1."""
    output = {
        "metadata_fields": [
            {"field_name": "env_medium", "value": "soil [ENVO:00001998]"},
            {"field_name": "env_broad_scale", "value": "just some prose"},
        ]
    }
    summary = summarize([build_row(_bundle(output), fake_lookup)])
    assert summary["triad_well_formed"] == 1
    assert summary["triad_with_verdict"] == 2


def test_all_three_denominator_leaves_out_unknown_lookups() -> None:
    """Without ENVO, a parsed value has no verdict; only the unparsable one does."""
    output = {
        "metadata_fields": [
            {"field_name": "env_medium", "value": "soil [ENVO:00001998]"},
            {"field_name": "env_broad_scale", "value": "just some prose"},
        ]
    }
    summary = summarize([build_row(_bundle(output), None)])
    assert summary["triad_with_verdict"] == 1
    assert summary["triad_well_formed"] == 0


def test_empty_corpus_tsv_still_has_a_header(tmp_path) -> None:
    from nmdc_ai_eval.trace_report import TSV_FIELDS, write_tsv

    path = tmp_path / "traces.tsv"
    write_tsv([], path)
    assert path.read_text().rstrip("\n").split("\t") == TSV_FIELDS
    assert "triad_checked_against_envo" in TSV_FIELDS
    assert "checks" not in TSV_FIELDS


def test_filter_by_environment_keeps_only_named_environments() -> None:
    from nmdc_ai_eval.trace_report import filter_by_environment

    local = _bundle(None)
    testing = _bundle(None)
    testing.trace["environment"] = "testing"
    assert filter_by_environment([local, testing], ["testing"]) == [testing]
    assert filter_by_environment([local, testing], None) == [local, testing]


def test_summary_reports_environments() -> None:
    rows = [_row(environment="local"), _row(environment="local"), _row(environment="testing")]
    assert summarize(rows)["environments"] == {"local": 2, "testing": 1}


def test_write_tsv_refuses_to_overwrite(tmp_path) -> None:
    """A result file is the record of one run; a second write to the same path must fail."""
    import pytest

    from nmdc_ai_eval.trace_report import write_tsv

    path = tmp_path / "traces.tsv"
    write_tsv([], path)
    with pytest.raises(FileExistsError):
        write_tsv([], path)


def test_pipe_splits_env_medium_only() -> None:
    """env_medium may be pipe-joined; a pipe in the single-term slots is one malformed value."""
    joined = "soil [ENVO:00001998] | terrestrial biome [ENVO:00000446]"
    output = {
        "metadata_fields": [
            {"field_name": "env_medium", "value": joined},
            {"field_name": "env_broad_scale", "value": joined},
        ]
    }
    row = build_row(_bundle(output), fake_lookup)
    by_slot = [(c.slot, c.parsed) for c in row.checks]
    assert by_slot == [("env_medium", True), ("env_medium", True), ("env_broad_scale", False)]
    assert row.triad_values == 3
    assert row.triad_well_formed == 2


def test_markdown_says_so_when_no_traces_matched() -> None:
    """An environment filter with no matches must not print an empty date range."""
    from nmdc_ai_eval.trace_report import _markdown

    text = _markdown(summarize([]))
    assert "No traces matched." in text
    assert " to ." not in text


def test_empty_env_medium_component_is_one_malformed_value() -> None:
    """A trailing or lone pipe must be counted as malformed, not dropped or passed."""
    for value in ("soil [ENVO:00001998] |", "|", "| soil [ENVO:00001998]"):
        output = {"metadata_fields": [{"field_name": "env_medium", "value": value}]}
        row = build_row(_bundle(output), fake_lookup)
        assert row.triad_values == 1, value
        assert row.triad_parsed == 0, value
        assert summarize([row])["triad_with_verdict"] == 1, value
