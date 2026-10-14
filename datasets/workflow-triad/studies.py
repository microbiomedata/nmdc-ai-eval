"""Three NMDC studies as gold standards for the suggestor's full env triad workflow.

Each case is a study already in NMDC whose stored triad serves as the reference.
The three were picked for different sources and for triads that vary within the
study, so a model cannot score well by repeating one answer:

* ``1000-soils``: soil cores crowdsourced across the US; 29 distinct triads,
  submitter-entered raw values, six DOIs.
* ``geodes-lakes``: water from oligotrophic, eutrophic and humic Wisconsin
  lakes; ``env_local_scale`` varies by lake type, ``env_medium`` does not.
* ``crop-rhizosphere``: maize, miscanthus, Arabidopsis and Boechera roots,
  rhizosphere and nearby soil.

None of the three came through the submission portal; they were imported from
GOLD. :func:`submission_object_for` builds the submission-portal shaped object
the pipeline takes from the study record, so the pipeline fetches DOI context
itself, as it would for a real submission.

Snapshot provenance and the reasons other candidates were rejected: ``README.md``
in this directory.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nmdc_metadata_suggestor_ai_tool.constants import ENV_TRIAD_SLOTS

from nmdc_ai_eval.env_triad_scoring import Reference, reference_from_biosamples

DATA = Path(__file__).resolve().parent / "data"

# GOLD's ecosystem path. The portal does not collect it, and for these studies it
# names the answer: GEODES' "Epilimnion" subtype, the rhizosphere study's
# "Rhizosphere" / "Endosphere" types.
GOLD_ECOSYSTEM_FIELDS = (
    "ecosystem",
    "ecosystem_category",
    "ecosystem_type",
    "ecosystem_subtype",
    "specific_ecosystem",
)
# Identifiers and bookkeeping NMDC adds on ingest; a submitter never writes them.
NMDC_BOOKKEEPING_FIELDS = (
    "associated_studies",
    "badges",
    "biosample_categories",
    "provenance_metadata",
    "gold_biosample_identifiers",
    "img_identifiers",
    "insdc_biosample_identifiers",
    "emsl_biosample_identifiers",
    "alternative_identifiers",
    "type",
)
WITHHELD_FIELDS = (*ENV_TRIAD_SLOTS, *GOLD_ECOSYSTEM_FIELDS, *NMDC_BOOKKEEPING_FIELDS)


@dataclass(frozen=True)
class StudyCase:
    """One study: where its snapshot lives and which portal package(s) its samples use."""

    key: str
    study_id: str
    source: str
    packages: tuple[str, ...]

    @property
    def study_path(self) -> Path:
        return DATA / f"{self.key}_study.json"

    @property
    def biosamples_path(self) -> Path:
        return DATA / f"{self.key}_biosamples.json"


CASES: dict[str, StudyCase] = {
    case.key: case
    for case in (
        StudyCase("1000-soils", "nmdc:sty-11-28tm5d36", "soil", ("soil",)),
        StudyCase("geodes-lakes", "nmdc:sty-11-fkbnah04", "freshwater", ("water",)),
        # Rhizosphere and root samples go under plant-associated in the portal,
        # the bulk soil under soil; the gate is scoped to both.
        StudyCase("crop-rhizosphere", "nmdc:sty-11-prtb4s28", "plant-associated", ("plant-associated", "soil")),
    )
}


def load_study(case: StudyCase) -> dict[str, Any]:
    """The study record as the NMDC API returned it at snapshot time."""
    with case.study_path.open() as f:
        study: dict[str, Any] = json.load(f)
        return study


def load_biosamples(case: StudyCase) -> list[dict[str, Any]]:
    """The study's biosample records, in the API's order, triads still attached."""
    with case.biosamples_path.open() as f:
        resources: list[dict[str, Any]] = json.load(f)["resources"]
        return resources


def bare_doi(value: str) -> str:
    return value.removeprefix("doi:")


def submission_object_for(case: StudyCase, study: dict[str, Any]) -> dict[str, Any]:
    """A submission-portal shaped object carrying what the study record gives.

    Every associated DOI goes in the field the portal would hold it in, with the
    record's ``doi_provider`` as the provider so the pipeline's DOI routing sees
    what a submitter would have entered. Publication DOIs carry no provider, as
    in the portal.
    """
    publication, data, award = [], [], []
    for doi in study.get("associated_dois") or []:
        category = doi.get("doi_category")
        entry = {"value": bare_doi(doi["doi_value"]), "provider": doi.get("doi_provider")}
        if category == "publication_doi":
            publication.append({**entry, "provider": None})
        elif category == "award_doi":
            award.append(entry)
        else:
            data.append(entry)
    return {
        "metadata_submission": {
            "packageName": list(case.packages),
            "studyForm": {
                "studyName": study.get("name") or study.get("title"),
                "description": study.get("description"),
                "publicationDois": publication,
                "dataDois": data,
            },
            "multiOmicsForm": {"awardDois": award},
        }
    }


def withhold(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copies of *samples* without the triad and the fields a submitter would not supply."""
    return [{k: v for k, v in sample.items() if k not in WITHHELD_FIELDS} for sample in samples]


def nmdc_reference(samples: list[dict[str, Any]]) -> Reference:
    """The triad currently stored in NMDC, keyed by biosample id."""
    return reference_from_biosamples(samples)
