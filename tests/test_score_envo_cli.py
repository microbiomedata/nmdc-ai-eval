"""Test score_envo CLI and score_envo_results orchestrator.

Uses a small fixture TSV + real oaklib (ENVO sqlite). No LLM API calls.
"""

from pathlib import Path

from click.testing import CliRunner

from nmdc_ai_eval.envo_scorer import main, score_envo_results

FIXTURES_DIR = Path(__file__).parent / "fixtures"
MINI_RESULTS = FIXTURES_DIR / "mini_results.tsv"
ENUM_DIR = Path(__file__).parent.parent / "datasets" / "ebs-prediction" / "enum_data"


class TestScoreEnvoResults:
    """Test the orchestrator function directly."""

    def test_returns_dataframe(self, tmp_path: Path) -> None:
        df = score_envo_results(MINI_RESULTS, enum_dir=ENUM_DIR, output_dir=tmp_path)
        assert len(df) == 3

    def test_writes_output_tsv(self, tmp_path: Path) -> None:
        score_envo_results(MINI_RESULTS, enum_dir=ENUM_DIR, output_dir=tmp_path)
        output = tmp_path / "results_envo_scored.tsv"
        assert output.exists()
        assert output.stat().st_size > 0

    def test_adds_scoring_columns(self, tmp_path: Path) -> None:
        df = score_envo_results(MINI_RESULTS, enum_dir=ENUM_DIR, output_dir=tmp_path)
        expected_cols = [
            "parse_success",
            "pred_label",
            "pred_curie",
            "truth_label",
            "truth_curie",
            "curie_label_valid",
            "exact_match",
            "relationship",
            "hop_distance",
            "in_template_enum",
            "ontology_score",
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_exact_match_scores_highest(self, tmp_path: Path) -> None:
        df = score_envo_results(MINI_RESULTS, enum_dir=ENUM_DIR, output_dir=tmp_path)
        # Row 0 is exact match, should score higher than row 1 (ancestor) and row 2 (parse fail)
        assert df.iloc[0]["ontology_score"] > df.iloc[1]["ontology_score"]
        assert df.iloc[0]["ontology_score"] > df.iloc[2]["ontology_score"]

    def test_parse_failure_scores_zero(self, tmp_path: Path) -> None:
        df = score_envo_results(MINI_RESULTS, enum_dir=ENUM_DIR, output_dir=tmp_path)
        # Row 2 has "not a valid response" — should fail parsing
        assert df.iloc[2]["parse_success"] is False or df.iloc[2]["parse_success"] == 0
        assert df.iloc[2]["ontology_score"] == 0.0


class TestScoreEnvoCli:
    """Test the Click CLI wrapper."""

    def test_cli_runs(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(main, [str(MINI_RESULTS), "-o", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert "Scored 3 rows" in result.output

    def test_cli_creates_output(self, tmp_path: Path) -> None:
        runner = CliRunner()
        runner.invoke(main, [str(MINI_RESULTS), "-o", str(tmp_path)])
        assert (tmp_path / "results_envo_scored.tsv").exists()
