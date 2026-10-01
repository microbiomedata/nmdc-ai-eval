"""Tests for env-triad suite API access without network calls."""

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import pytest

GENERATOR_PATH = Path(__file__).parent.parent / "datasets" / "env-triad-prediction" / "generate_suite.py"
SPEC = importlib.util.spec_from_file_location("env_triad_suite_generator", GENERATOR_PATH)
assert SPEC and SPEC.loader
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)


@pytest.mark.parametrize(
    ("env", "expected_url"),
    [
        ("prod", "https://api.microbiomedata.org"),
        ("dev", "https://api-dev.microbiomedata.org"),
    ],
)
def test_get_study_with_biosamples_uses_current_client_parameters(
    monkeypatch: pytest.MonkeyPatch, env: str, expected_url: str
) -> None:
    study = {"id": "nmdc:sty-test"}
    biosamples = [{"id": "nmdc:bsm-test"}]
    search = MagicMock()
    search.get_record_by_id.return_value = [study]
    search.get_linked_instances.return_value = biosamples
    constructor = MagicMock(return_value=search)
    monkeypatch.setattr(GENERATOR, "StudySearch", constructor)

    result = GENERATOR.get_study_with_biosamples("nmdc:sty-test", env=env)

    constructor.assert_called_once_with(api_base_url=expected_url)
    search.get_record_by_id.assert_called_once_with(record_id="nmdc:sty-test")
    search.get_linked_instances.assert_called_once_with(
        ids=["nmdc:sty-test"],
        types="nmdc:Biosample",
        hydrate=True,
        max_page_size=1999,
    )
    assert result == {"study": [study], "biosamples": biosamples}
