"""Pass/fail check that every suggested ``label [CURIE]`` names a real, current term with its exact label.

The NMDC rule (https://github.com/microbiomedata/nmdc-ai-eval/issues/138): a term is in the
ontology its prefix claims, is not obsolete, and carries that ontology's label exactly,
including case, spaces and hyphens. ``envo_scorer.validate_curie_label`` is looser: it ignores
case, looks only in ENVO and never checks obsolescence. This module applies the rule to every
slot in a suggestor output, so it needs no reference answer and runs on Langfuse traces too.

Parsing is strict on purpose. ``envo_scorer.parse_label_curie`` strips whitespace and leading
underscores, which would hide a label written with a trailing space. Here the label is the text
before `` [`` exactly as written, so ``"soil  [ENVO:00001998]"`` fails the label check.

Whether a prefix is allowed for a slot by the submission-schema patterns (point 4 of the issue)
is not checked here yet.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

# One space between label and bracket; nothing before the label or after the bracket.
_TERM_RE = re.compile(r"^(?P<label>.+?) \[(?P<curie>(?P<prefix>[A-Za-z][A-Za-z0-9_.]*):(?P<local>[A-Za-z0-9_.]+))\]$")
# A value that looks like it is trying to be a term, so a malformed one is reported, not skipped.
_LOOKS_LIKE_TERM_RE = re.compile(r"\[[A-Za-z][A-Za-z0-9_.]*:[^\]\s]+\]")


class LookupUnavailable(Exception):
    """The ontology could not be consulted, as opposed to a CURIE it does not contain."""


@dataclass(frozen=True)
class OntologyTerm:
    label: str | None
    obsolete: bool


# Takes a CURIE; returns the term, None when the ontology lacks it, or raises LookupUnavailable.
TermLookup = Callable[[str], OntologyTerm | None]


@dataclass
class TermVerdict:
    """The three checks for one value. A check is None when it could not run."""

    slot: str
    raw_value: str
    parsed: bool
    label: str | None = None
    curie: str | None = None
    prefix: str | None = None
    resolves: bool | None = None
    not_obsolete: bool | None = None
    label_exact: bool | None = None
    ontology_label: str | None = None

    @property
    def passed(self) -> bool | None:
        """True only when every check ran and passed; None when the ontology was unavailable."""
        if not self.parsed:
            return False
        checks = (self.resolves, self.not_obsolete, self.label_exact)
        if False in checks:
            return False
        if None in checks:
            return None
        return True


def check_value(slot: str, raw_value: str, lookup: TermLookup) -> TermVerdict:
    """Check one ``label [CURIE]`` value."""
    match = _TERM_RE.match(raw_value)
    if match is None:
        return TermVerdict(slot=slot, raw_value=raw_value, parsed=False)
    verdict = TermVerdict(
        slot=slot,
        raw_value=raw_value,
        parsed=True,
        label=match["label"],
        curie=match["curie"],
        prefix=match["prefix"],
    )
    try:
        term = lookup(match["curie"])
    except LookupUnavailable:
        return verdict
    verdict.resolves = term is not None
    if term is None:
        # A CURIE the ontology lacks is neither current nor correctly labelled.
        verdict.not_obsolete = False
        verdict.label_exact = False
        return verdict
    verdict.ontology_label = term.label
    verdict.not_obsolete = not term.obsolete
    verdict.label_exact = term.label is not None and term.label == match["label"]
    return verdict


def term_values(output: Any) -> list[tuple[str, str]]:
    """Every (slot, value) in an ``LLMOutput``-shaped payload that holds one or more terms.

    Any slot counts. A pipe-joined value is split into its terms, and an empty piece is kept so
    it fails to parse rather than vanishing.
    """
    pairs: list[tuple[str, str]] = []
    fields = output.get("metadata_fields") if isinstance(output, dict) else None
    if not isinstance(fields, list):
        return pairs
    for suggestion in fields:
        if not isinstance(suggestion, dict):
            continue
        slot, value = suggestion.get("field_name"), suggestion.get("value")
        if not isinstance(slot, str) or not isinstance(value, str) or not _LOOKS_LIKE_TERM_RE.search(value):
            continue
        pieces = value.split("|") if "|" in value else [value]
        pairs.extend((slot, piece.strip() if len(pieces) > 1 else piece) for piece in pieces)
    return pairs


def check_output(output: Any, lookup: TermLookup) -> list[TermVerdict]:
    return [check_value(slot, value, lookup) for slot, value in term_values(output)]


def pass_rate(verdicts: Iterable[TermVerdict]) -> dict[str, Any]:
    """Counts for one run. ``pass_rate`` is over values whose checks all ran."""
    verdicts = list(verdicts)
    outcome = Counter("unknown" if v.passed is None else "passed" if v.passed else "failed" for v in verdicts)
    failures: Counter[str] = Counter()
    for v in verdicts:
        if not v.parsed:
            failures["unparsed"] += 1
            continue
        for name in ("resolves", "not_obsolete", "label_exact"):
            if getattr(v, name) is False:
                failures[name] += 1
    decided = outcome["passed"] + outcome["failed"]
    return {
        "values": len(verdicts),
        "passed": outcome["passed"],
        "failed": outcome["failed"],
        "unknown": outcome["unknown"],
        "pass_rate": outcome["passed"] / decided if decided else None,
        "failures": dict(failures),
    }


def oak_lookup(adapter_for: Callable[[str], Any] | None = None) -> TermLookup:
    """A lookup that opens one oaklib SQLite adapter per prefix (``sqlite:obo:envo`` and so on).

    ``adapter_for`` maps a lowercase prefix to an adapter; the default uses oaklib's semantic-sql
    builds. A prefix whose build cannot be opened raises ``LookupUnavailable``, so its values are
    reported unknown rather than wrong.
    """

    @lru_cache(maxsize=None)
    def adapter(prefix: str) -> Any:
        if adapter_for is not None:
            return adapter_for(prefix)
        from oaklib import get_adapter  # type: ignore[import-untyped]

        return get_adapter(f"sqlite:obo:{prefix}")

    @lru_cache(maxsize=None)
    def obsolete_set(prefix: str) -> frozenset[str]:
        return frozenset(adapter(prefix).obsoletes())

    def lookup(curie: str) -> OntologyTerm | None:
        prefix = curie.split(":", 1)[0].lower()
        try:
            ad = adapter(prefix)
            label = ad.label(curie)
            obsolete = curie in obsolete_set(prefix)
        except Exception as exc:  # noqa: BLE001 - an unreachable ontology is unknown, not wrong
            raise LookupUnavailable(f"{prefix}: {type(exc).__name__}: {exc}") from exc
        if label is None and not obsolete:
            return None
        return OntologyTerm(label=label, obsolete=obsolete)

    return lookup
