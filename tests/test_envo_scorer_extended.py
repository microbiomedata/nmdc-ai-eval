"""Extended tests for envo_scorer — covers orchestration helpers and summary output.

No network, no oaklib, no mocking.
"""

import pandas as pd
import pytest

from nmdc_ai_eval.envo_scorer import (
    _add_timing_cost_stubs,
    _extract_template,
    print_envo_summary,
)


class TestExtractTemplate:
    def test_standard_format(self) -> None:
        assert _extract_template("{'sampleData': 'soil_data'}") == "soil_data"

    def test_double_quotes(self) -> None:
        assert _extract_template('{"sampleData": "water_data"}') == "water_data"

    def test_equals_format(self) -> None:
        assert _extract_template("sampleData=soil_data") == "soil_data"

    def test_no_match(self) -> None:
        assert _extract_template("no template here") is None

    def test_empty_string(self) -> None:
        assert _extract_template("") is None

    def test_sediment(self) -> None:
        assert _extract_template("sampleData: sediment_data, study: foo") == "sediment_data"


class TestAddTimingCostStubs:
    def test_adds_all_keys(self) -> None:
        result: dict[str, object] = {"existing_key": 42}
        _add_timing_cost_stubs(result)
        assert "response_time_s" in result
        assert "prompt_tokens" in result
        assert "completion_tokens" in result
        assert "est_cost_usd" in result
        assert result["existing_key"] == 42

    def test_all_none(self) -> None:
        result: dict[str, object] = {}
        _add_timing_cost_stubs(result)
        assert all(result[k] is None for k in result)


class TestPrintEnvoSummary:
    @pytest.fixture
    def scored_df(self) -> pd.DataFrame:
        """Minimal DataFrame matching the schema print_envo_summary expects."""
        return pd.DataFrame(
            {
                "model": ["litellm/gpt-4o", "litellm/gpt-4o", "litellm/claude-sonnet-4-5", "litellm/claude-sonnet-4-5"],
                "ontology_score": [0.9, 0.7, 0.8, 0.85],
                "exact_match": [True, False, False, True],
                "relationship": ["exact", "ancestor", "descendant", "exact"],
                "parse_success": [True, True, True, True],
                "in_template_enum": [True, False, True, True],
                "curie_label_valid": [True, True, False, True],
            }
        )

    def test_runs_without_error(self, scored_df: pd.DataFrame, capsys: pytest.CaptureFixture[str]) -> None:
        print_envo_summary(scored_df)
        output = capsys.readouterr().out
        assert "ENVO Triad Scoring Summary" in output
        assert "gpt-4o" in output
        assert "claude-sonnet" in output

    def test_shows_model_ranking(self, scored_df: pd.DataFrame, capsys: pytest.CaptureFixture[str]) -> None:
        print_envo_summary(scored_df)
        output = capsys.readouterr().out
        assert "Model ranking" in output

    def test_shows_exact_match_rate(self, scored_df: pd.DataFrame, capsys: pytest.CaptureFixture[str]) -> None:
        print_envo_summary(scored_df)
        output = capsys.readouterr().out
        assert "Exact match rate" in output
        assert "50.0%" in output

    def test_shows_relationship_breakdown(self, scored_df: pd.DataFrame, capsys: pytest.CaptureFixture[str]) -> None:
        print_envo_summary(scored_df)
        output = capsys.readouterr().out
        assert "Relationship breakdown" in output
        assert "exact=" in output

    def test_shows_enum_compliance(self, scored_df: pd.DataFrame, capsys: pytest.CaptureFixture[str]) -> None:
        print_envo_summary(scored_df)
        output = capsys.readouterr().out
        assert "Template enum compliance" in output

    def test_shows_parse_success(self, scored_df: pd.DataFrame, capsys: pytest.CaptureFixture[str]) -> None:
        print_envo_summary(scored_df)
        output = capsys.readouterr().out
        assert "Parse success rate" in output
        assert "100.0%" in output

    def test_handles_all_parse_failures(self, capsys: pytest.CaptureFixture[str]) -> None:
        df = pd.DataFrame(
            {
                "model": ["m1", "m1"],
                "ontology_score": [0.0, 0.0],
                "exact_match": [False, False],
                "relationship": [None, None],
                "parse_success": [False, False],
                "in_template_enum": [None, None],
                "curie_label_valid": [None, None],
            }
        )
        print_envo_summary(df)
        output = capsys.readouterr().out
        assert "parse_fail=2" in output
