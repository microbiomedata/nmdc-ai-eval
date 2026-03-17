"""Unit tests for envo_scorer — no network, no oaklib."""

from pathlib import Path

from nmdc_ai_eval.envo_scorer import (
    _TEMPLATE_TO_ENUM_PREFIX,
    ANCESTOR_DECAY,
    DESCENDANT_DECAY,
    ENUM_NEUTRAL,
    W_ENUM,
    W_HIER,
    W_LABEL,
    W_PARSE,
    compute_enum_score,
    compute_hierarchy_score,
    compute_ontology_score,
    load_enum_for_template,
    parse_label_curie,
)

ENUM_DIR = Path(__file__).parent.parent / "datasets" / "ebs-prediction" / "enum_data"


class TestParseLabelCurie:
    def test_standard(self) -> None:
        result = parse_label_curie("terrestrial biome [ENVO:00000446]")
        assert result == ("terrestrial biome", "ENVO:00000446")

    def test_leading_underscores(self) -> None:
        result = parse_label_curie("__temperate woodland biome [ENVO:01000221]")
        assert result == ("temperate woodland biome", "ENVO:01000221")

    def test_quoted(self) -> None:
        result = parse_label_curie('"forest biome [ENVO:01000174]"')
        assert result == ("forest biome", "ENVO:01000174")

    def test_whitespace(self) -> None:
        result = parse_label_curie("  forest biome [ENVO:01000174]  ")
        assert result == ("forest biome", "ENVO:01000174")

    def test_invalid_no_curie(self) -> None:
        assert parse_label_curie("just a label") is None

    def test_invalid_empty(self) -> None:
        assert parse_label_curie("") is None

    def test_invalid_bad_curie(self) -> None:
        assert parse_label_curie("label [not:a:curie]") is None


class TestLoadEnumForTemplate:
    def test_soil(self) -> None:
        curies = load_enum_for_template("soil_data", ENUM_DIR)
        assert curies is not None
        assert len(curies) > 0
        assert all(c.startswith("ENVO:") for c in curies)

    def test_water(self) -> None:
        curies = load_enum_for_template("water_data", ENUM_DIR)
        assert curies is not None
        assert len(curies) > 10

    def test_no_enum_template(self) -> None:
        assert load_enum_for_template("air_data", ENUM_DIR) is None

    def test_unknown_template(self) -> None:
        assert load_enum_for_template("nonexistent_data", ENUM_DIR) is None

    def test_all_mapped_templates_have_files(self) -> None:
        for template in _TEMPLATE_TO_ENUM_PREFIX:
            result = load_enum_for_template(template, ENUM_DIR)
            assert result is not None, f"Missing enum file for {template}"


class TestScoreWeights:
    """Verify score formula properties."""

    def test_weights_sum_to_one(self) -> None:
        assert abs(W_PARSE + W_LABEL + W_HIER + W_ENUM - 1.0) < 1e-9

    def test_descendant_decays_slower_than_ancestor(self) -> None:
        assert DESCENDANT_DECAY < ANCESTOR_DECAY


class TestComputeHierarchyScore:
    def test_exact(self) -> None:
        assert compute_hierarchy_score("exact", 0) == 1.0

    def test_descendant_1_hop(self) -> None:
        assert compute_hierarchy_score("descendant", 1) == 1.0 - DESCENDANT_DECAY

    def test_descendant_3_hops(self) -> None:
        assert compute_hierarchy_score("descendant", 3) == 1.0 - 3 * DESCENDANT_DECAY

    def test_ancestor_1_hop(self) -> None:
        assert compute_hierarchy_score("ancestor", 1) == 1.0 - ANCESTOR_DECAY

    def test_ancestor_3_hops(self) -> None:
        assert compute_hierarchy_score("ancestor", 3) == 1.0 - 3 * ANCESTOR_DECAY

    def test_descendant_better_than_ancestor_same_hops(self) -> None:
        d = compute_hierarchy_score("descendant", 2)
        a = compute_hierarchy_score("ancestor", 2)
        assert d > a

    def test_unrelated(self) -> None:
        assert compute_hierarchy_score("unrelated", None) == 0.0

    def test_floors_at_zero(self) -> None:
        assert compute_hierarchy_score("ancestor", 100) == 0.0
        assert compute_hierarchy_score("descendant", 100) == 0.0


class TestComputeEnumScore:
    def test_in_enum(self) -> None:
        assert compute_enum_score(True) == 1.0

    def test_not_in_enum(self) -> None:
        assert compute_enum_score(False) == 0.0

    def test_no_enum(self) -> None:
        assert compute_enum_score(None) == ENUM_NEUTRAL


class TestComputeOntologyScore:
    def test_perfect_score(self) -> None:
        """Exact match, valid label, in enum → 1.0."""
        score = compute_ontology_score(True, True, "exact", 0, True)
        assert score == 1.0

    def test_parse_failure_is_zero(self) -> None:
        assert compute_ontology_score(False, False, None, None, None) == 0.0

    def test_exact_match_bad_label_no_enum(self) -> None:
        """Exact CURIE match but wrong label, no enum file."""
        score = compute_ontology_score(True, False, "exact", 0, None)
        expected = W_PARSE + 0.0 + W_HIER * 1.0 + W_ENUM * ENUM_NEUTRAL
        assert abs(score - expected) < 1e-9

    def test_descendant_1_hop_in_enum_valid_label(self) -> None:
        score = compute_ontology_score(True, True, "descendant", 1, True)
        hier = 1.0 - DESCENDANT_DECAY
        expected = W_PARSE + W_LABEL + W_HIER * hier + W_ENUM * 1.0
        assert abs(score - expected) < 1e-9

    def test_ancestor_2_hops_not_in_enum(self) -> None:
        score = compute_ontology_score(True, True, "ancestor", 2, False)
        hier = 1.0 - 2 * ANCESTOR_DECAY
        expected = W_PARSE + W_LABEL + W_HIER * hier + W_ENUM * 0.0
        assert abs(score - expected) < 1e-9

    def test_unrelated_in_enum(self) -> None:
        score = compute_ontology_score(True, True, "unrelated", None, True)
        expected = W_PARSE + W_LABEL + 0.0 + W_ENUM * 1.0
        assert abs(score - expected) < 1e-9

    def test_score_bounded_0_1(self) -> None:
        """All score combinations should be in [0, 1]."""
        for parse in (True, False):
            for label in (True, False):
                for rel in ("exact", "descendant", "ancestor", "unrelated", None):
                    for hops in (None, 0, 1, 2, 5, 100):
                        for enum in (True, False, None):
                            s = compute_ontology_score(parse, label, rel, hops, enum)
                            assert 0.0 <= s <= 1.0, f"Score {s} out of bounds: {parse=} {label=} {rel=} {hops=} {enum=}"
