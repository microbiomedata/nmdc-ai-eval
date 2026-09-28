"""The weekly upgrade must leave held AI packages alone and upgrade everything else."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "upgrade_except_held.py"
spec = importlib.util.spec_from_file_location("upgrade_except_held", SCRIPT)
assert spec and spec.loader
upgrade = importlib.util.module_from_spec(spec)
spec.loader.exec_module(upgrade)


def test_split_holds_exact_names_and_prefix_patterns() -> None:
    packages = ["anthropic", "duckdb", "openinference-instrumentation", "openinference-x", "pandas"]
    to_upgrade, held = upgrade.split(packages, ["anthropic", "openinference-*"])
    assert held == ["anthropic", "openinference-instrumentation", "openinference-x"]
    assert to_upgrade == ["duckdb", "pandas"]


def test_a_prefix_pattern_does_not_hold_a_similar_name() -> None:
    to_upgrade, held = upgrade.split(["llm", "llm-matrix", "llmx"], ["llm"])
    assert held == ["llm"]
    assert to_upgrade == ["llm-matrix", "llmx"]


def test_every_held_entry_matches_a_locked_package() -> None:
    """A misspelled entry would silently hold nothing."""
    import fnmatch

    packages = upgrade.locked_packages()
    unmatched = [p for p in upgrade.held_patterns() if not any(fnmatch.fnmatch(n, p) for n in packages)]
    assert unmatched == []


def test_the_project_itself_is_not_a_locked_package() -> None:
    assert "nmdc-ai-eval" not in upgrade.locked_packages()


def test_moved_reports_a_held_package_that_changed_version() -> None:
    before = {"llm": "0.35 {}", "duckdb": "1.5.5 {}"}
    after = {"llm": "0.36 {}", "duckdb": "1.5.6 {}"}
    assert upgrade.moved(before, after, ["llm"]) == ["llm: 0.35 {} -> 0.36 {}"]


def test_moved_is_empty_when_held_packages_stay_put() -> None:
    before = {"llm": "0.35 {}", "duckdb": "1.5.5 {}"}
    after = {"llm": "0.35 {}", "duckdb": "1.5.6 {}"}
    assert upgrade.moved(before, after, ["llm"]) == []


def test_moved_reports_a_held_package_that_disappeared() -> None:
    assert upgrade.moved({"llm": "0.35 {}"}, {}, ["llm"]) == ["llm: 0.35 {} -> None"]


def test_moved_catches_a_git_dependency_on_a_new_commit_with_the_same_version() -> None:
    before = {"tool": '1.2.1 {"git": "https://x/tool?branch=main#aaa"}'}
    after = {"tool": '1.2.1 {"git": "https://x/tool?branch=main#bbb"}'}
    assert len(upgrade.moved(before, after, ["tool"])) == 1


def test_moved_catches_a_new_package_matching_a_held_pattern() -> None:
    before = {"openinference-a": "1 {}"}
    after = {"openinference-a": "1 {}", "openinference-b": "1 {}"}
    assert upgrade.moved(before, after, ["openinference-*"]) == ["openinference-b: None -> 1 {}"]


def test_locked_entries_include_the_source() -> None:
    entry = upgrade.locked_entries()["nmdc-metadata-suggestor-ai-tool"]
    assert '"git"' in entry


def test_main_fails_when_resolution_moves_a_held_package(monkeypatch) -> None:
    """The whole run must stop, not just report, so no upgrade PR is opened."""
    import pytest

    entries = iter([{"llm": "0.35 {}", "duckdb": "1.5.5 {}"}, {"llm": "0.36 {}", "duckdb": "1.5.6 {}"}])
    monkeypatch.setattr(upgrade, "locked_entries", lambda: next(entries))
    monkeypatch.setattr(upgrade, "locked_packages", lambda: ["duckdb", "llm"])
    monkeypatch.setattr(upgrade, "held_patterns", lambda: ["llm"])
    monkeypatch.setattr(upgrade.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(upgrade.sys, "argv", ["upgrade_except_held.py"])
    with pytest.raises(SystemExit, match="llm: 0.35"):
        upgrade.main()
