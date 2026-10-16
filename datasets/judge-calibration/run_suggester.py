"""Send one calibration case to the suggester and save what it returns.

Builds a submission-portal-shaped JSON object from a case's input/study.txt and
input/samples.tsv, calls run_recommendation_pipeline (the single-pass path), and writes
outputs/suggester-<provider>-<model>-<timestamp>.json (the LLMOutput verbatim) plus a .tsv
view with the same columns as the hand-written outputs (id, field_name, value, reason).

Credentials are read in-process from the .env named by --env (default: the repo's .env); nothing is printed.

    just judge-calibration-dry-run 04-soil-miscanthus-fertilization
    just judge-calibration-run 04-soil-miscanthus-fertilization gcp gemini-2.5-flash
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
SAMPLE_DATA_KEY = {
    "soil": "soil_data",
    "water": "water_data",
    "sediment": "sediment_data",
    "plant-associated": "plant_associated_data",
    "host-associated": "host_associated_data",
    "air": "air_data",
}


def package_of(case_dir: Path) -> str:
    with open(HERE / "cases.tsv", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row["case"] == case_dir.name:
                return row["package"]
    raise SystemExit(f"{case_dir.name} is not listed in cases.tsv")


def build_submission(case_dir: Path) -> dict:
    study_name = description = ""
    for line in (case_dir / "input" / "study.txt").read_text().splitlines():
        if line.startswith("Study:"):
            study_name = line[len("Study:") :].strip()
        elif line.startswith("Description:"):
            description = line[len("Description:") :].strip()
    package = package_of(case_dir)
    rows = []
    with open(case_dir / "input" / "samples.tsv", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            rows.append({k: v for k, v in row.items() if v != "" and k != "id"})
    return {
        "metadata_submission": {
            "packageName": [package],
            "templates": [package],
            "studyForm": {
                "studyName": study_name,
                "description": description,
                "notes": "",
                "dataDois": [],
                "publicationDois": [],
            },
            "multiOmicsForm": {},
            "sampleData": {SAMPLE_DATA_KEY[package]: rows},
        }
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case_dir", type=Path)
    ap.add_argument("--provider", default="gcp", choices=["gcp", "cborg", "pnnl"])
    ap.add_argument("--model", default=None, help="model id for the provider; the client's default when omitted")
    ap.add_argument(
        "--env", type=Path, default=HERE.parents[1] / ".env", help=".env read in-process (default: the repo's)"
    )
    ap.add_argument("--dry-run", action="store_true", help="print the submission object and the call, make no request")
    a = ap.parse_args()
    case_dir = a.case_dir.resolve()
    submission = build_submission(case_dir)
    package = submission["metadata_submission"]["packageName"][0]
    if a.dry_run:
        print(json.dumps(submission, indent=2))
        call = (
            f"run_recommendation_pipeline(submission_object, LLMClient(access_provider={a.provider!r}, "
            f"model={a.model!r}), interface_name={package!r})"
        )
        print(f"\nWould call: {call}", file=sys.stderr)
        return 0

    from dotenv import load_dotenv

    load_dotenv(a.env)
    from nmdc_metadata_suggestor_ai_tool.llm_client import LLMClient
    from nmdc_metadata_suggestor_ai_tool.recommendation_pipeline import run_recommendation_pipeline

    client = LLMClient(access_provider=a.provider, model=a.model) if a.model else LLMClient(access_provider=a.provider)
    result = run_recommendation_pipeline(submission, client, interface_name=package)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model = (result.model or a.model or "default").replace("/", "_")
    out = case_dir / "outputs" / f"suggester-{a.provider}-{model}-{stamp}"
    out.with_suffix(".json").write_text(result.model_dump_json(indent=2) + "\n")
    with open(out.with_suffix(".tsv"), "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["id", "field_name", "value", "reason"])
        for f in result.metadata_fields:
            w.writerow([f.id or "", f.field_name, f.value if f.value is not None else "", f.reason])
    print(f"{len(result.metadata_fields)} suggestions -> {out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
