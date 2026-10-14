"""Tests for the workflow-triad cases: the snapshots, the submission objects, the references."""

import sys
from pathlib import Path

import pytest
from nmdc_metadata_suggestor_ai_tool.utils.submission_parser import get_submission_fields

sys.path.insert(0, str(Path(__file__).parent.parent / "datasets" / "workflow-triad"))
import studies  # noqa: E402

EXPECTED_COUNTS = {"1000-soils": 139, "geodes-lakes": 119, "crop-rhizosphere": 89}


@pytest.mark.parametrize("key", list(studies.CASES))
def test_snapshot_loads_all_biosamples_of_the_study(key: str) -> None:
    case = studies.CASES[key]
    assert studies.load_study(case)["id"] == case.study_id
    samples = studies.load_biosamples(case)
    assert len(samples) == EXPECTED_COUNTS[key]
    assert all(case.study_id in s["associated_studies"] for s in samples)


@pytest.mark.parametrize("key", list(studies.CASES))
def test_every_sample_has_a_full_reference_triad(key: str) -> None:
    # GEODES and the rhizosphere study store bare CURIEs as raw values; this fails
    # if the parser stops reading them.
    case = studies.CASES[key]
    reference = studies.nmdc_reference(studies.load_biosamples(case))
    assert len(reference) == EXPECTED_COUNTS[key]
    for slots in reference.values():
        for slot in ("env_broad_scale", "env_local_scale", "env_medium"):
            assert [t for t in slots[slot] if t.curie], slot


def test_submission_object_routes_each_doi_by_category() -> None:
    case = studies.CASES["1000-soils"]
    submission = studies.submission_object_for(case, studies.load_study(case))
    form = submission["metadata_submission"]
    assert form["packageName"] == ["soil"]
    assert {d["value"] for d in form["studyForm"]["publicationDois"]} == {
        "10.1101/2022.12.12.520098",
        "10.1016/j.apsoil.2025.106110",
        "10.1029/2024GL113091",
    }
    assert all(d["provider"] is None for d in form["studyForm"]["publicationDois"])
    assert form["studyForm"]["dataDois"] == [{"value": "10.5281/zenodo.7406532", "provider": "zenodo"}]
    assert {d["provider"] for d in form["multiOmicsForm"]["awardDois"]} == {"emsl", "jgi"}


def test_submission_object_parses_to_all_dois_and_both_packages() -> None:
    case = studies.CASES["crop-rhizosphere"]
    parsed = get_submission_fields(studies.submission_object_for(case, studies.load_study(case)))
    assert parsed["mixs_extensions"] == ["PlantAssociatedInterface", "SoilInterface"]
    assert len(parsed["dois"]) == 3
    assert parsed["study_name"].startswith("Arabidopsis, maize")


def test_withhold_removes_triad_and_leaking_fields_and_leaves_the_input_alone() -> None:
    samples = studies.load_biosamples(studies.CASES["geodes-lakes"])[:2]
    withheld = studies.withhold(samples)
    for original, copy in zip(samples, withheld, strict=True):
        assert not set(studies.WITHHELD_FIELDS) & copy.keys()
        assert copy["id"] == original["id"]
        assert copy["name"] == original["name"]
        assert "env_medium" in original
        assert "ecosystem_subtype" in original
