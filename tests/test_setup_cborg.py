"""`just setup-cborg` must merge aliases without clobbering, and store the key without printing it."""

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "setup_cborg.py"
spec = importlib.util.spec_from_file_location("setup_cborg", SCRIPT)
assert spec and spec.loader
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


def test_the_committed_aliases_all_use_the_cborg_key() -> None:
    entries = yaml.safe_load(setup.ALIASES.read_text())
    assert entries and all(e["model_id"].startswith("cborg/") for e in entries)
    assert {e["api_key_name"] for e in entries} == {setup.KEY_NAME}
    assert len({e["model_id"] for e in entries}) == len(entries)


def test_a_missing_target_is_created_from_the_aliases(tmp_path) -> None:
    target = tmp_path / "extra-openai-models.yaml"
    assert "created" in setup.merge_aliases(setup.ALIASES, target)
    assert target.read_text() == setup.ALIASES.read_text()


def test_other_entries_are_kept_and_a_rerun_changes_nothing(tmp_path) -> None:
    target = tmp_path / "extra-openai-models.yaml"
    other = {"model_id": "local/llama", "model_name": "llama", "api_base": "http://localhost:8000/v1"}
    target.write_text(yaml.safe_dump([other]))
    setup.merge_aliases(setup.ALIASES, target)
    merged = yaml.safe_load(target.read_text())
    assert merged[0] == other
    assert len(merged) == 1 + len(yaml.safe_load(setup.ALIASES.read_text()))
    before = target.read_text()
    assert "already has" in setup.merge_aliases(setup.ALIASES, target)
    assert target.read_text() == before


def test_a_target_that_is_not_a_list_is_left_alone(tmp_path) -> None:
    target = tmp_path / "extra-openai-models.yaml"
    target.write_text("{}\n")
    with pytest.raises(SystemExit):
        setup.merge_aliases(setup.ALIASES, target)
    assert target.read_text() == "{}\n"


def test_an_empty_target_gets_the_aliases(tmp_path) -> None:
    target = tmp_path / "extra-openai-models.yaml"
    target.write_text("")
    setup.merge_aliases(setup.ALIASES, target)
    assert len(yaml.safe_load(target.read_text())) == len(yaml.safe_load(setup.ALIASES.read_text()))


def test_a_stale_cborg_entry_is_updated_in_place(tmp_path) -> None:
    target = tmp_path / "extra-openai-models.yaml"
    stale = dict(yaml.safe_load(setup.ALIASES.read_text())[0], api_base="https://old.example/v1")
    target.write_text(yaml.safe_dump([stale]))
    assert "updated 1" in setup.merge_aliases(setup.ALIASES, target)
    assert yaml.safe_load(target.read_text())[0]["api_base"] == "https://api.cborg.lbl.gov/v1"


def test_the_key_is_stored_beside_other_keys_and_never_printed(tmp_path, capsys) -> None:
    keys = tmp_path / "keys.json"
    keys.write_text(json.dumps({"openai": "o-key"}))
    message = setup.store_key(keys, "c-secret")
    assert "c-secret" not in message and "c-secret" not in capsys.readouterr().out
    assert json.loads(keys.read_text()) == {"openai": "o-key", "cborg": "c-secret"}
    assert "already matches" in setup.store_key(keys, "c-secret")


def test_a_new_keys_file_is_private(tmp_path) -> None:
    keys = tmp_path / "keys.json"
    setup.store_key(keys, "c-secret")
    assert keys.stat().st_mode & 0o777 == 0o600


def test_without_a_key_an_existing_one_is_kept_and_none_at_all_fails(tmp_path) -> None:
    keys = tmp_path / "keys.json"
    with pytest.raises(SystemExit):
        setup.store_key(keys, None)
    keys.write_text(json.dumps({"cborg": "existing"}))
    assert "kept" in setup.store_key(keys, None)
    assert json.loads(keys.read_text())["cborg"] == "existing"


def test_an_unreadable_keys_file_is_not_overwritten(tmp_path) -> None:
    keys = tmp_path / "keys.json"
    keys.write_text("{not json")
    with pytest.raises(SystemExit):
        setup.store_key(keys, "c-secret")
    assert keys.read_text() == "{not json"
