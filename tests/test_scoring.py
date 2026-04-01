"""Tests for generic set-based scoring."""

from nmdc_ai_eval.scoring import score_sets


class TestScoreSets:
    def test_exact_match(self) -> None:
        result = score_sets({"a", "b"}, {"a", "b"})
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

    def test_no_overlap(self) -> None:
        result = score_sets({"a", "b"}, {"c", "d"})
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0

    def test_partial_overlap(self) -> None:
        result = score_sets({"a", "b", "c"}, {"a", "b"})
        assert result["recall"] == 1.0
        assert result["precision"] < 1.0
        assert len(result["false_positives"]) == 1

    def test_both_empty(self) -> None:
        result = score_sets(set(), set())
        assert result["f1"] == 1.0

    def test_predicted_empty(self) -> None:
        result = score_sets(set(), {"a"})
        assert result["recall"] == 0.0

    def test_expected_empty(self) -> None:
        result = score_sets({"a"}, set())
        assert result["precision"] == 0.0

    def test_exclude_from_precision(self) -> None:
        result = score_sets({"a", "b", "x"}, {"a", "b"}, exclude_from_precision={"x"})
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["excluded_correct"] == ["x"]
        assert "x" not in result["false_positives"]

    def test_exclude_does_not_affect_recall(self) -> None:
        result = score_sets({"a", "x"}, {"a", "b"}, exclude_from_precision={"x"})
        assert result["recall"] == 0.5  # b is missing
        assert result["precision"] == 1.0  # only a in precision set, and it's a TP

    def test_excluded_item_in_expected_still_counted_for_recall(self) -> None:
        """If an excluded item is also in expected, recall should still count it."""
        result = score_sets({"a", "x"}, {"a", "x"}, exclude_from_precision={"x"})
        assert result["recall"] == 1.0  # both a and x are predicted and expected
        assert result["precision"] == 1.0  # only a in precision set, it's a TP
        assert result["true_positives"] == ["a", "x"]

    def test_early_returns_have_all_keys(self) -> None:
        for result in [score_sets(set(), set()), score_sets({"a"}, set())]:
            for key in ["true_positives", "false_positives", "false_negatives", "excluded_correct"]:
                assert key in result, f"Missing key: {key}"
