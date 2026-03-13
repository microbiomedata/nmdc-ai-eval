"""Validate the eval TSV dataset integrity."""

import csv
from pathlib import Path

import pytest

DATASETS_DIR = Path(__file__).parent.parent / "datasets"
TSV_PATH = DATASETS_DIR / "submission-metadata-prediction" / "eval_input_target_pairs.tsv"

EXPECTED_COLUMNS = [
    "study_name",
    "description",
    "notes",
    "sampleData",
    "env_broad_scale",
    "env_local_scale",
    "env_medium",
    "geo_loc_name",
    "depth",
    "ecosystem",
    "ecosystem_type",
    "ecosystem_subtype",
    "ecosystem_category",
    "specific_ecosystem",
    "analysis_type",
]

VALID_SAMPLE_DATA = {
    "soil_data",
    "water_data",
    "plant_associated_data",
    "misc_envs_data",
    "host_associated_data",
    "sediment_data",
    "air_data",
    "metagenome_sequencing_non_interleaved_data",
}

INPUT_COLUMNS = ["study_name", "description"]


@pytest.fixture(scope="module")
def rows() -> list[dict[str, str]]:
    with open(TSV_PATH, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def test_tsv_exists() -> None:
    assert TSV_PATH.exists(), f"Expected TSV at {TSV_PATH}"


def test_column_names(rows: list[dict[str, str]]) -> None:
    assert list(rows[0].keys()) == EXPECTED_COLUMNS


def test_column_count_consistent(rows: list[dict[str, str]]) -> None:
    for i, row in enumerate(rows):
        assert len(row) == len(EXPECTED_COLUMNS), (
            f"Row {i + 2} has {len(row)} columns, expected {len(EXPECTED_COLUMNS)}"
        )


def test_no_empty_inputs(rows: list[dict[str, str]]) -> None:
    for i, row in enumerate(rows):
        for col in INPUT_COLUMNS:
            assert row[col].strip(), f"Row {i + 2} has empty {col}"


def test_sample_data_values_valid(rows: list[dict[str, str]]) -> None:
    invalid = []
    for i, row in enumerate(rows):
        if row["sampleData"] not in VALID_SAMPLE_DATA:
            invalid.append((i + 2, row["sampleData"]))
    assert not invalid, f"Unexpected sampleData values: {invalid[:10]}"


def test_unique_row_count(rows: list[dict[str, str]]) -> None:
    """Snapshot: track distinct rows.

    The TSV is a submission × biosample join, so many biosamples share identical
    metadata.  5,052 rows collapse to 322 unique rows.  This test catches
    unexpected changes to the deduplication ratio.
    """
    unique = {tuple(row.values()) for row in rows}
    assert len(unique) == 322, (
        f"Expected 322 unique rows, got {len(unique)}. If the TSV was intentionally regenerated, update this number."
    )


def test_row_count(rows: list[dict[str, str]]) -> None:
    """Snapshot test: catch unexpected changes to the dataset size."""
    assert len(rows) == 5052, (
        f"Expected 5052 rows, got {len(rows)}. If the TSV was intentionally regenerated, update this number."
    )
