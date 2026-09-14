"""Tests for the local RDFLib fallback used when Oaklib's ENVO cache is unavailable."""

from pathlib import Path

from nmdc_ai_eval.envo_scorer import _RDFLibEnvoAdapter


def test_rdflib_adapter_resolves_labels_and_hierarchy(tmp_path: Path) -> None:
    owl_path = tmp_path / "envo.ttl"
    owl_path.write_text(
        """@prefix ENVO: <http://purl.obolibrary.org/obo/ENVO_> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        ENVO:00000446 rdfs:label "terrestrial biome" .
        ENVO:01000174 rdfs:label "forest biome" ; rdfs:subClassOf ENVO:00000446 .
        """
    )

    adapter = _RDFLibEnvoAdapter(owl_path)

    assert adapter.label("ENVO:00000446") == "terrestrial biome"
    assert adapter.hierarchical_parents("ENVO:01000174") == ["ENVO:00000446"]
    assert adapter.ancestors("ENVO:01000174") == ["ENVO:00000446"]
