#!/usr/bin/env python3
"""Snapshot each workflow-triad study and its biosamples from the NMDC API into data/.

Usage:
    uv run python datasets/workflow-triad/fetch_snapshots.py            # all cases
    uv run python datasets/workflow-triad/fetch_snapshots.py geodes-lakes

The eval reads only these files, so a run is reproducible until they are
refreshed. Refreshing changes the gold standard: commit the new snapshot and
note the date in README.md.
"""

import json
import sys
from pathlib import Path
from typing import Any

import requests
import studies  # beside this script, on sys.path when run as one

API = "https://api.microbiomedata.org/nmdcschema"
PAGE_SIZE = 2000


def fetch_study(study_id: str) -> dict[str, Any]:
    response = requests.get(f"{API}/study_set/{study_id}", timeout=60)
    response.raise_for_status()
    study: dict[str, Any] = response.json()
    return study


def fetch_biosamples(study_id: str) -> list[dict[str, Any]]:
    """Every biosample of the study, following the API's page tokens."""
    resources: list[dict[str, Any]] = []
    token = None
    while True:
        params: dict[str, str | int] = {
            "filter": json.dumps({"associated_studies": study_id}),
            "max_page_size": PAGE_SIZE,
        }
        if token:
            params["page_token"] = token
        response = requests.get(f"{API}/biosample_set", params=params, timeout=120)
        response.raise_for_status()
        page = response.json()
        resources.extend(page["resources"])
        token = page.get("next_page_token")
        if not token:
            return resources


def write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")


def main() -> None:
    keys = sys.argv[1:] or list(studies.CASES)
    studies.DATA.mkdir(exist_ok=True)
    for key in keys:
        case = studies.CASES[key]
        study = fetch_study(case.study_id)
        samples = fetch_biosamples(case.study_id)
        write(case.study_path, study)
        write(case.biosamples_path, {"resources": samples})
        print(f"{key}: {case.study_id}, {len(samples)} biosamples")


if __name__ == "__main__":
    main()
