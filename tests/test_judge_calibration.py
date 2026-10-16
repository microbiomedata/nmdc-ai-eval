"""Structural checks on datasets/judge-calibration, the synthetic judge calibration set.

Every case folder is indexed and ranked; every input column and suggested field is a slot of the
case's package interface in nmdc-submission-schema; enum values are permissible; the hand-written
outputs parse as the suggester's LLMOutput; the TSV views match the JSON; every expected verdict
names a suggestion that exists; and the runner builds a submission object for every case without
a network call.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
from linkml_runtime import SchemaView
from nmdc_metadata_suggestor_ai_tool.models.llm_output import LLMOutput

CAL_DIR = Path(__file__).parent.parent / "datasets" / "judge-calibration"
CASES_DIR = CAL_DIR / "cases"
GRADES = ("good", "moderate", "bad")
PER_SUGGESTION = {"accuracy", "factuality", "relevancy"}
PER_OUTPUT = {"completeness", "coherence"}
INTERFACE = {
    "soil": "SoilInterface",
    "water": "WaterInterface",
    "sediment": "SedimentInterface",
    "plant-associated": "PlantAssociatedInterface",
    "air": "AirInterface",
    "host-associated": "HostAssociatedInterface",
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


@pytest.fixture(scope="module")
def sv() -> SchemaView:
    import nmdc_submission_schema

    schema = Path(nmdc_submission_schema.__file__).parent / "schema" / "nmdc_submission_schema.yaml"
    return SchemaView(str(schema))


@pytest.fixture(scope="module")
def cases() -> list[dict[str, str]]:
    return read_tsv(CAL_DIR / "cases.tsv")


@pytest.fixture(scope="module")
def interface_slots(sv: SchemaView, cases: list[dict[str, str]]) -> dict[str, set[str]]:
    return {c["case"]: set(sv.class_slots(INTERFACE[c["package"]])) for c in cases}


def output(case: str, grade: str) -> LLMOutput:
    return LLMOutput.model_validate(json.loads((CASES_DIR / case / "outputs" / f"{grade}.json").read_text()))


def sample_ids(case: str) -> set[str]:
    return {row["id"] for row in read_tsv(CASES_DIR / case / "input" / "samples.tsv")}


def test_every_case_folder_is_indexed_and_ranked(cases: list[dict[str, str]]) -> None:
    folders = {p.name for p in CASES_DIR.iterdir() if p.is_dir()}
    assert folders == {c["case"] for c in cases}
    ranks = sorted(int(c["schema_fit_rank"]) for c in cases)
    assert ranks == list(range(1, len(cases) + 1))
    for c in cases:
        assert c["case"].startswith(f"{int(c['schema_fit_rank']):02d}-")
        assert c["package"] in INTERFACE
        assert c["schema_fit_why"].strip()


def test_input_columns_are_interface_slots(cases: list[dict[str, str]], interface_slots: dict[str, set[str]]) -> None:
    for c in cases:
        path = CASES_DIR / c["case"] / "input" / "samples.tsv"
        with open(path, newline="") as fh:
            header = next(csv.reader(fh, delimiter="\t"))
        assert header[:2] == ["id", "samp_name"], path
        missing = [col for col in header[2:] if col not in interface_slots[c["case"]]]
        assert not missing, f"{path}: not slots of {INTERFACE[c['package']]}: {missing}"
        assert (CASES_DIR / c["case"] / "input" / "study.txt").read_text().startswith("Study:")


def test_outputs_parse_and_use_valid_slots_and_values(
    sv: SchemaView, cases: list[dict[str, str]], interface_slots: dict[str, set[str]]
) -> None:
    enums = sv.all_enums()
    for c in cases:
        ids = sample_ids(c["case"])
        for grade in GRADES:
            out = output(c["case"], grade)
            assert out.metadata_fields, f"{c['case']}/{grade}: no suggestions"
            for f in out.metadata_fields:
                where = f"{c['case']}/{grade}/{f.field_name}"
                assert f.field_name in interface_slots[c["case"]], where
                assert f.id is None or f.id in ids, where
                assert f.reason.strip(), where
                slot = sv.get_slot(f.field_name)
                if slot.range in enums:
                    assert str(f.value) in sv.get_enum(slot.range).permissible_values, f"{where}={f.value!r}"
                if slot.range == "float":
                    assert isinstance(f.value, (int, float)), f"{where}={f.value!r}"


def test_tsv_view_matches_json(cases: list[dict[str, str]]) -> None:
    for c in cases:
        for grade in GRADES:
            out = output(c["case"], grade)
            rendered = [[f.id or "", f.field_name, str(f.value), f.reason] for f in out.metadata_fields]
            with open(CASES_DIR / c["case"] / "outputs" / f"{grade}.tsv", newline="") as fh:
                rows = list(csv.reader(fh, delimiter="\t"))
            assert rows[0] == ["id", "field_name", "value", "reason"]
            assert rows[1:] == rendered, f"{c['case']}/{grade}.tsv differs from {grade}.json"


def test_expected_verdicts_name_real_suggestions(cases: list[dict[str, str]]) -> None:
    for c in cases:
        ids = sample_ids(c["case"])
        for grade in GRADES:
            out = output(c["case"], grade)
            present = {(f.field_name, f.id) for f in out.metadata_fields}
            fields = {f.field_name for f in out.metadata_fields}
            for row in read_tsv(CASES_DIR / c["case"] / "expected" / f"{grade}.tsv"):
                where = f"{c['case']}/{grade}: {row}"
                assert row["kind"] in {"fail", "open"}, where
                assert row["criterion"] in PER_SUGGESTION | PER_OUTPUT, where
                assert row["why"].strip(), where
                assert row["id"] == "" or row["id"] in ids, where
                if row["criterion"] in PER_SUGGESTION:
                    assert row["field_name"], where
                    key = (row["field_name"], row["id"] or None)
                    assert key in present or row["field_name"] in fields, where
            if grade == "good":
                fails = [r for r in read_tsv(CASES_DIR / c["case"] / "expected" / "good.tsv") if r["kind"] == "fail"]
                assert not fails, f"{c['case']}: a good output has expected fails"


def test_outputs_index_counts_match(cases: list[dict[str, str]]) -> None:
    index = {(r["case"], r["grade"]): r for r in read_tsv(CAL_DIR / "outputs.tsv")}
    assert set(index) == {(c["case"], g) for c in cases for g in GRADES}
    for (case, grade), row in index.items():
        out = output(case, grade)
        expected = read_tsv(CASES_DIR / case / "expected" / f"{grade}.tsv")
        assert int(row["suggestions"]) == len(out.metadata_fields), (case, grade)
        assert int(row["expected_fails"]) == sum(r["kind"] == "fail" for r in expected), (case, grade)
        assert int(row["open_questions"]) == sum(r["kind"] == "open" for r in expected), (case, grade)
        assert row["why"].strip()


def load_runner() -> Any:
    spec = importlib.util.spec_from_file_location("run_suggester", CAL_DIR / "run_suggester.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_builds_a_submission_for_every_case(cases: list[dict[str, str]]) -> None:
    runner = load_runner()
    for c in cases:
        submission = runner.build_submission(CASES_DIR / c["case"])
        ms = submission["metadata_submission"]
        assert ms["packageName"] == [c["package"]]
        assert ms["studyForm"]["studyName"] and ms["studyForm"]["description"]
        rows = ms["sampleData"][runner.SAMPLE_DATA_KEY[c["package"]]]
        assert len(rows) == len(sample_ids(c["case"]))
        assert all("samp_name" in r and "id" not in r for r in rows)
