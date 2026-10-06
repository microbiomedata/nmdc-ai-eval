#!/usr/bin/env python3
"""Make cborg/* model names usable in suite runs. Safe to rerun.

1. Merge config/cborg-models.yaml into extra-openai-models.yaml in llm's user directory.
   Entries with other model_ids are left alone; a cborg entry with the same model_id is
   replaced only when it differs. The file is not rewritten when nothing changes.
2. Store CBORG_API_KEY (from .env or the environment) in the llm key store as `cborg`,
   the key name the aliases use. The key is never printed. With no CBORG_API_KEY set, an
   existing `cborg` key is kept.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import yaml

ALIASES = Path(__file__).resolve().parent.parent / "config" / "cborg-models.yaml"
KEY_NAME = "cborg"


def merge_aliases(source: Path, target: Path) -> str:
    """Merge source alias entries into target; return what happened."""
    ours = yaml.safe_load(source.read_text()) or []
    if not target.exists():
        shutil.copyfile(source, target)
        return f"created {target} with {len(ours)} cborg aliases"
    existing = yaml.safe_load(target.read_text())
    if existing is None:  # an empty file
        existing = []
    if not isinstance(existing, list):
        raise SystemExit(f"{target} is not a list of model entries; not touching it")
    by_id = {entry.get("model_id"): i for i, entry in enumerate(existing) if isinstance(entry, dict)}
    added = updated = 0
    for entry in ours:
        i = by_id.get(entry["model_id"])
        if i is None:
            existing.append(entry)
            added += 1
        elif existing[i] != entry:
            existing[i] = entry
            updated += 1
    if not added and not updated:
        return f"{target} already has all {len(ours)} cborg aliases"
    target.write_text(yaml.safe_dump(existing, sort_keys=False))
    kept = len(existing) - added - updated
    return f"{target}: added {added}, updated {updated} cborg aliases, kept {kept} other entries"


def store_key(keys_path: Path, value: str | None) -> str:
    """Put value in keys.json under KEY_NAME, the way `llm keys set` does; return what happened."""
    current: dict[str, str] = {}
    if keys_path.exists():
        try:
            current = json.loads(keys_path.read_text())
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{keys_path} is not valid JSON; fix it before storing a key") from exc
    if not value:
        if current.get(KEY_NAME):
            return f"CBORG_API_KEY not set; kept the existing `{KEY_NAME}` key in the llm key store"
        raise SystemExit(
            "CBORG_API_KEY is not set in .env or the environment, and the llm key store has no `cborg` key"
        )
    if current.get(KEY_NAME) == value:
        return f"llm key store `{KEY_NAME}` already matches CBORG_API_KEY"
    if not keys_path.exists():
        current = {"// Note": "This file stores secret API credentials. Do not share!"}
        keys_path.write_text(json.dumps(current))
        keys_path.chmod(0o600)
    current[KEY_NAME] = value
    keys_path.write_text(json.dumps(current, indent=2) + "\n")
    return f"stored CBORG_API_KEY in the llm key store as `{KEY_NAME}`"


def main() -> int:
    import llm
    from dotenv import load_dotenv

    load_dotenv()
    user_dir = llm.user_dir()  # type: ignore[no-untyped-call]
    print(merge_aliases(ALIASES, user_dir / "extra-openai-models.yaml"))
    print(store_key(user_dir / "keys.json", os.environ.get("CBORG_API_KEY")))
    print("Check with: just verify-auth")
    return 0


if __name__ == "__main__":
    sys.exit(main())
