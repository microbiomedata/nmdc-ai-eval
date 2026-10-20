"""Tests for the exact-term check. No network and no ontology download: lookups are fakes."""

from types import SimpleNamespace

import pytest

from nmdc_ai_eval.term_check import (
    LookupUnavailable,
    OntologyTerm,
    check_output,
    check_value,
    oak_lookup,
    pass_rate,
    term_values,
)

FAKE = {
    "ENVO:00001998": OntologyTerm("soil", obsolete=False),
    "ENVO:01000048": OntologyTerm("oceanic epipelagic zone biome", obsolete=False),
    "ENVO:01001000": OntologyTerm("plant-associated environment", obsolete=False),
    "ENVO:00000020": OntologyTerm("obsolete lake", obsolete=True),
    "PO:0009005": OntologyTerm("root", obsolete=False),
}


def fake_lookup(curie: str) -> OntologyTerm | None:
    return FAKE.get(curie)


def unavailable(curie: str) -> OntologyTerm | None:
    raise LookupUnavailable("no build")


def test_exact_current_term_passes() -> None:
    verdict = check_value("env_medium", "soil [ENVO:00001998]", fake_lookup)
    assert verdict.passed is True
    assert (verdict.resolves, verdict.not_obsolete, verdict.label_exact) == (True, True, True)


@pytest.mark.parametrize(
    "value",
    [
        "Soil [ENVO:00001998]",  # case
        "soil  [ENVO:00001998]",  # trailing space in the label
        "plant associated environment [ENVO:01001000]",  # missing hyphen
    ],
)
def test_label_must_match_exactly(value: str) -> None:
    verdict = check_value("env_local_scale", value, fake_lookup)
    assert verdict.resolves is True
    assert verdict.label_exact is False
    assert verdict.passed is False


def test_obsolete_term_fails_even_with_its_label() -> None:
    verdict = check_value("env_local_scale", "obsolete lake [ENVO:00000020]", fake_lookup)
    assert verdict.label_exact is True
    assert verdict.not_obsolete is False
    assert verdict.passed is False


def test_unknown_curie_fails_every_check() -> None:
    verdict = check_value("env_broad_scale", "oceanic epipelagic zone biome [ENVO:01000033]", fake_lookup)
    assert (verdict.resolves, verdict.not_obsolete, verdict.label_exact) == (False, False, False)
    assert verdict.passed is False


def test_other_prefixes_are_checked_too() -> None:
    assert check_value("env_local_scale", "root [PO:0009005]", fake_lookup).passed is True


@pytest.mark.parametrize("value", ["soil[ENVO:00001998]", "soil [ENVO:00001998] ", "_soil [ENVO:00001998]x", "soil"])
def test_malformed_value_does_not_parse(value: str) -> None:
    assert check_value("env_medium", value, fake_lookup).passed is False


def test_unavailable_ontology_is_unknown_not_wrong() -> None:
    verdict = check_value("env_medium", "soil [ENVO:00001998]", unavailable)
    assert verdict.parsed is True
    assert verdict.passed is None


def test_term_values_takes_any_slot_and_splits_pipes() -> None:
    output = {
        "metadata_fields": [
            {"field_name": "env_medium", "value": "soil [ENVO:00001998] | root [PO:0009005]"},
            {"field_name": "samp_mat_process", "value": "sieved through 2 mm mesh"},
            {"field_name": "host_body_site", "value": "root [PO:0009005]"},
            {"field_name": "ecosystem_type", "value": ""},
            "not a dict",
        ]
    }
    assert term_values(output) == [
        ("env_medium", "soil [ENVO:00001998]"),
        ("env_medium", "root [PO:0009005]"),
        ("host_body_site", "root [PO:0009005]"),
    ]
    assert term_values({"metadata_fields": "oops"}) == []
    assert term_values(None) == []


def test_pass_rate_counts_each_failure_and_excludes_unknowns() -> None:
    output = {
        "metadata_fields": [
            {"field_name": "env_medium", "value": "soil [ENVO:00001998]"},
            {"field_name": "env_local_scale", "value": "plant associated environment [ENVO:01001000]"},
            {"field_name": "env_local_scale", "value": "obsolete lake [ENVO:00000020]"},
            {"field_name": "env_broad_scale", "value": "biome[ENVO:00000428]"},
        ]
    }
    summary = pass_rate(check_output(output, fake_lookup))
    assert summary["values"] == 4
    assert (summary["passed"], summary["failed"], summary["unknown"]) == (1, 3, 0)
    assert summary["pass_rate"] == 0.25
    assert summary["failures"] == {"label_exact": 1, "not_obsolete": 1, "unparsed": 1}

    unknown = pass_rate(check_output(output, unavailable))
    assert unknown["unknown"] == 3 and unknown["pass_rate"] == 0.0


def test_oak_lookup_reads_label_and_obsolescence_per_prefix() -> None:
    opened: list[str] = []

    def adapter_for(prefix: str) -> SimpleNamespace:
        opened.append(prefix)
        if prefix == "uberon":
            raise OSError("build missing")
        labels = {k: v.label for k, v in FAKE.items() if k.lower().startswith(prefix)}
        obsoletes = [k for k, v in FAKE.items() if v.obsolete]
        return SimpleNamespace(label=labels.get, obsoletes=lambda: iter(obsoletes))

    lookup = oak_lookup(adapter_for)
    assert lookup("ENVO:00001998") == OntologyTerm("soil", obsolete=False)
    assert lookup("ENVO:00000020") == OntologyTerm("obsolete lake", obsolete=True)
    assert lookup("ENVO:99999999") is None
    assert lookup("PO:0009005") == OntologyTerm("root", obsolete=False)
    with pytest.raises(LookupUnavailable):
        lookup("UBERON:0000955")
    assert opened.count("envo") == 1
