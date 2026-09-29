"""Upgrade every locked package except the ones listed in .github/held-packages.txt.

uv has no option to exclude packages from ``uv lock --upgrade``; it offers upgrading everything or
upgrading named packages (https://docs.astral.sh/uv/concepts/projects/sync/#upgrading-locked-package-versions).
So this names every locked package except the held ones and passes each as --upgrade-package.

Usage: python3 scripts/upgrade_except_held.py [--dry-run]
"""

from __future__ import annotations

import fnmatch
import json
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELD_FILE = ROOT / ".github" / "held-packages.txt"


def held_patterns() -> list[str]:
    lines = HELD_FILE.read_text().splitlines()
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]


def locked_entries() -> dict[str, str]:
    """Name to every locked variant of that package, except the project itself.

    A universal lock can hold several records for one name, one per Python or platform marker,
    so each name maps to the sorted set of its variants. Each variant is its version, its source
    (a git dependency can move to a new commit with the same version) and its resolution markers.
    """
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    variants: dict[str, list[str]] = {}
    for package in lock["package"]:
        if {"editable", "virtual"} & set(package.get("source", {})):
            continue
        variant = {k: package.get(k) for k in ("version", "source", "resolution-markers")}
        variants.setdefault(package["name"], []).append(json.dumps(variant, sort_keys=True))
    return {name: " | ".join(sorted(entries)) for name, entries in variants.items()}


def moved(before: dict[str, str], after: dict[str, str], patterns: list[str]) -> list[str]:
    """Held packages whose lock entry changed, appeared or disappeared.

    Patterns are matched against the packages both before and after resolution, so a newly added
    package matching a held pattern counts as a change.
    """
    names = sorted(n for n in set(before) | set(after) if any(fnmatch.fnmatch(n, p) for p in patterns))
    return [f"{n}: {before.get(n)} -> {after.get(n)}" for n in names if before.get(n) != after.get(n)]


def locked_packages() -> list[str]:
    """Every package in uv.lock except the project itself, which is not a dependency."""
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    return sorted(
        package["name"] for package in lock["package"] if not {"editable", "virtual"} & set(package.get("source", {}))
    )


def split(packages: list[str], patterns: list[str]) -> tuple[list[str], list[str]]:
    held = [p for p in packages if any(fnmatch.fnmatch(p, pattern) for pattern in patterns)]
    return [p for p in packages if p not in held], held


def main() -> None:
    to_upgrade, held = split(locked_packages(), held_patterns())
    print(f"holding {len(held)}: {', '.join(held)}")
    print(f"upgrading {len(to_upgrade)} other packages")
    command = ["uv", "lock"] + [arg for p in to_upgrade for arg in ("--upgrade-package", p)]
    if "--dry-run" in sys.argv:
        print(" ".join(command[:6]), "...")
        return
    before = locked_entries()
    subprocess.run(command, cwd=ROOT, check=True)  # noqa: S603 - "uv lock" plus names read from this repo's uv.lock
    # Leaving a package out of --upgrade-package does not pin it: uv treats the existing lock as a
    # preference, so an upgraded dependent can still pull a held package forward. Fail rather than
    # let that ride into an automatic upgrade PR; move the held package in its own PR instead.
    changed = moved(before, locked_entries(), held_patterns())
    if changed:
        sys.exit("held packages moved during resolution: " + "; ".join(changed))


if __name__ == "__main__":
    main()
