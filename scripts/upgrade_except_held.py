"""Upgrade every locked package except the ones listed in .github/held-packages.txt.

uv has no option to exclude packages from ``uv lock --upgrade``; it offers upgrading everything or
upgrading named packages (https://docs.astral.sh/uv/concepts/projects/sync/#upgrading-locked-package-versions).
So this names every locked package except the held ones and passes each as --upgrade-package.

Usage: python3 scripts/upgrade_except_held.py [--dry-run]
"""

from __future__ import annotations

import fnmatch
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELD_FILE = ROOT / ".github" / "held-packages.txt"


def held_patterns() -> list[str]:
    lines = HELD_FILE.read_text().splitlines()
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]


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
    subprocess.run(command, cwd=ROOT, check=True)  # noqa: S603 - "uv lock" plus names read from this repo's uv.lock


if __name__ == "__main__":
    main()
