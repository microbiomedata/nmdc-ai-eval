"""Integration tests for envo_scorer with oaklib (requires network on first run).

Marked slow — run with: uv run pytest -m slow
"""

import pytest

from nmdc_ai_eval.envo_scorer import (
    check_relationship,
    compute_hop_distance,
    get_envo_adapter,
    validate_curie_label,
)

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def adapter():  # type: ignore[no-untyped-def]
    return get_envo_adapter()


class TestValidateCurieLabel:
    def test_correct_label(self, adapter) -> None:  # type: ignore[no-untyped-def]
        assert validate_curie_label(adapter, "ENVO:00000446", "terrestrial biome") is True

    def test_wrong_label(self, adapter) -> None:  # type: ignore[no-untyped-def]
        assert validate_curie_label(adapter, "ENVO:00000446", "aquatic biome") is False

    def test_case_insensitive(self, adapter) -> None:  # type: ignore[no-untyped-def]
        assert validate_curie_label(adapter, "ENVO:00000446", "Terrestrial Biome") is True

    def test_nonexistent_curie(self, adapter) -> None:  # type: ignore[no-untyped-def]
        assert validate_curie_label(adapter, "ENVO:99999999", "fake") is False


class TestCheckRelationship:
    def test_exact(self, adapter) -> None:  # type: ignore[no-untyped-def]
        assert check_relationship(adapter, "ENVO:00000446", "ENVO:00000446") == "exact"

    def test_descendant(self, adapter) -> None:  # type: ignore[no-untyped-def]
        # forest biome is a subclass of terrestrial biome
        result = check_relationship(adapter, "ENVO:01000174", "ENVO:00000446")
        assert result == "descendant"

    def test_ancestor(self, adapter) -> None:  # type: ignore[no-untyped-def]
        # terrestrial biome is an ancestor of forest biome
        result = check_relationship(adapter, "ENVO:00000446", "ENVO:01000174")
        assert result == "ancestor"


class TestComputeHopDistance:
    def test_same(self, adapter) -> None:  # type: ignore[no-untyped-def]
        assert compute_hop_distance(adapter, "ENVO:00000446", "ENVO:00000446") == 0

    def test_direct_parent(self, adapter) -> None:  # type: ignore[no-untyped-def]
        # forest biome -> terrestrial biome should be 1 hop
        dist = compute_hop_distance(adapter, "ENVO:01000174", "ENVO:00000446")
        assert dist is not None
        assert dist >= 1
