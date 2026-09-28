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
