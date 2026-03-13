"""Ontology-aware scorer for env_broad_scale predictions.

Enriches llm-matrix results with:
- Parse validation (label [CURIE] format)
- CURIE-label consistency via oaklib
- Ontology relationship (exact/ancestor/descendant/unrelated)
- Hop distance in ENVO graph (BFS = breadth-first search over subClassOf edges)
- Template enum compliance (from submission-schema value sets)
- Composite ontology score (weighted formula, bounded [0, 1])

Scoring formula
===============
    ontology_score = W_PARSE * parse_ok
                   + W_LABEL * curie_label_valid
                   + W_HIER  * hierarchy_score
                   + W_ENUM  * enum_score

Where:
    W_PARSE = 0.1  — did the LLM output valid "label [CURIE]" syntax?
    W_LABEL = 0.1  — does the CURIE resolve to the stated label in ENVO?
    W_HIER  = 0.5  — ontology proximity (exact/descendant/ancestor/unrelated)
    W_ENUM  = 0.3  — is the prediction in the template's allowed value set?

hierarchy_score:
    exact match         → 1.0
    descendant, d hops  → max(0, 1.0 − 0.1 × d)   (more specific = good)
    ancestor, d hops    → max(0, 1.0 − 0.15 × d)   (vaguer = slightly worse)
    unrelated           → 0.0

enum_score:
    in enum     → 1.0
    not in enum → 0.0
    no enum     → 0.5  (neutral — don't penalize templates without enums)

Descendants are weighted higher than ancestors: predicting a more-specific child
term shows domain knowledge, while predicting a vague parent is less useful.

Future work
===========
- LinkML schema validation: validate predictions against submission-schema enums
  directly via linkml-runtime SchemaView, rather than the bundled TSV snapshots.
  See `validate_via_linkml()` stub below.
- Response time and cost tracking: llm-matrix does not currently expose these in
  its results. When it does (or via wrapper timing), add response_time_s and
  est_cost_usd columns. See `_add_timing_cost_stubs()` below.
"""

from __future__ import annotations

import csv
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from oaklib.interfaces import OboGraphInterface  # type: ignore[import-untyped]

# Regex for "label [CURIE]" — tolerates leading underscores
_LABEL_CURIE_RE = re.compile(r"^_*(.+?)\s*\[(\w+:\d+)\]$")

# Map sampleData template names to enum file prefixes
_TEMPLATE_TO_ENUM_PREFIX: dict[str, str] = {
    "soil_data": "soil",
    "water_data": "water",
    "sediment_data": "sediment",
    "plant_associated_data": "plant_associated",
}

DEFAULT_ENUM_DIR = Path(__file__).parent.parent.parent / "datasets" / "ebs-prediction" / "enum_data"

# --- Score weights (sum to 1.0) ---
W_PARSE = 0.1
W_LABEL = 0.1
W_HIER = 0.5
W_ENUM = 0.3

# --- Hierarchy decay rates ---
DESCENDANT_DECAY = 0.1  # per hop; descendants penalized less (more specific = better)
ANCESTOR_DECAY = 0.15  # per hop; ancestors penalized more (vaguer = worse)

# --- Enum neutral value (for templates without enum files) ---
ENUM_NEUTRAL = 0.5


def parse_label_curie(text: str) -> tuple[str, str] | None:
    """Parse 'label [CURIE]' format, stripping leading underscores.

    Returns (label, curie) or None if parsing fails.
    """
    text = text.strip().strip('"')
    m = _LABEL_CURIE_RE.match(text)
    if m:
        return m.group(1).strip(), m.group(2)
    return None


def get_envo_adapter() -> "OboGraphInterface":
    """Get an oaklib adapter for ENVO. Downloads sqlite on first use (~50MB)."""
    from oaklib import get_adapter  # type: ignore[import-untyped]

    adapter: OboGraphInterface = get_adapter("sqlite:obo:envo")
    return adapter


def validate_curie_label(adapter: "OboGraphInterface", curie: str, label: str) -> bool:
    """Check if the CURIE's canonical label matches the given label (case-insensitive).

    Uses oaklib to look up the CURIE in ENVO's sqlite and compare labels.
    """
    canonical = adapter.label(curie)
    if canonical is None:
        return False
    return bool(canonical.lower() == label.lower())


def validate_via_linkml(curie: str, label: str, template: str) -> bool | None:
    """Validate a prediction against the submission-schema via LinkML (stub).

    This would use linkml-runtime's SchemaView to load the submission-schema
    and check whether the CURIE+label pair is a valid member of the
    env_broad_scale enum for the given template. Currently returns None
    (not implemented).

    To implement:
        1. uv add linkml-runtime (already a transitive dep of oaklib)
        2. Download/reference submission-schema YAML
        3. sv = SchemaView("submission_schema.yaml")
        4. Check sv.get_enum(f"{template}_env_broad_scale_enum").permissible_values
    """
    return None


def check_relationship(adapter: "OboGraphInterface", pred_curie: str, truth_curie: str) -> str:
    """Determine ontology relationship between prediction and ground truth.

    Returns: "exact", "ancestor", "descendant", or "unrelated"

    "descendant" means the prediction is MORE SPECIFIC than ground truth
    (e.g., predicted "coniferous forest biome" when truth is "forest biome").

    "ancestor" means the prediction is LESS SPECIFIC / vaguer
    (e.g., predicted "biome" when truth is "forest biome").
    """
    if pred_curie == truth_curie:
        return "exact"

    # Check if prediction is an ancestor of truth (truth descends from pred)
    truth_ancestors = set(adapter.ancestors(truth_curie, predicates=["rdfs:subClassOf"]))
    if pred_curie in truth_ancestors:
        return "ancestor"

    # Check if prediction is a descendant of truth
    pred_ancestors = set(adapter.ancestors(pred_curie, predicates=["rdfs:subClassOf"]))
    if truth_curie in pred_ancestors:
        return "descendant"

    return "unrelated"


def compute_hop_distance(adapter: "OboGraphInterface", curie_a: str, curie_b: str) -> int | None:
    """Compute shortest hop distance between two CURIEs via subClassOf.

    Uses breadth-first search (BFS): explores all nodes at distance d before
    moving to d+1, guaranteeing the shortest path.

    Returns None if no path exists (unrelated terms).
    """
    if curie_a == curie_b:
        return 0

    # Check a -> b (a is descendant of b)
    a_ancestors = list(adapter.ancestors(curie_a, predicates=["rdfs:subClassOf"]))
    if curie_b in a_ancestors:
        return _count_hops_up(adapter, curie_a, curie_b)

    # Check b -> a (b is descendant of a)
    b_ancestors = list(adapter.ancestors(curie_b, predicates=["rdfs:subClassOf"]))
    if curie_a in b_ancestors:
        return _count_hops_up(adapter, curie_b, curie_a)

    return None


def _count_hops_up(adapter: "OboGraphInterface", start: str, target: str) -> int | None:
    """BFS from start up via subClassOf to target, counting edges."""
    from collections import deque

    visited: set[str] = set()
    queue: deque[tuple[str, int]] = deque([(start, 0)])
    while queue:
        node, dist = queue.popleft()
        if node == target:
            return dist
        if node in visited:
            continue
        visited.add(node)
        for parent in adapter.hierarchical_parents(node):
            if parent not in visited:
                queue.append((parent, dist + 1))
    return None


def load_enum_for_template(template: str, enum_dir: Path = DEFAULT_ENUM_DIR) -> set[str] | None:
    """Load allowed env_broad_scale CURIEs for a sampleData template.

    Returns None if no enum file exists for this template.
    Enum files are snapshots from submission-schema's environmental_context_value_sets.
    """
    prefix = _TEMPLATE_TO_ENUM_PREFIX.get(template)
    if prefix is None:
        return None

    tsv_path = enum_dir / f"{prefix}_env_broad_scale.tsv"
    if not tsv_path.exists():
        return None

    curies: set[str] = set()
    with open(tsv_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            curies.add(row["id"])
    return curies


def compute_hierarchy_score(relationship: str | None, hop_distance: int | None) -> float:
    """Compute the hierarchy component of the ontology score.

    Formula:
        exact       → 1.0
        descendant  → max(0, 1.0 − DESCENDANT_DECAY × hops)
        ancestor    → max(0, 1.0 − ANCESTOR_DECAY × hops)
        unrelated   → 0.0

    Descendants decay slower because more-specific predictions show
    domain knowledge. Ancestors decay faster because vague predictions
    (e.g., "biome") are less informative.
    """
    if relationship == "exact":
        return 1.0
    if relationship == "descendant" and hop_distance is not None:
        return max(0.0, 1.0 - DESCENDANT_DECAY * hop_distance)
    if relationship == "ancestor" and hop_distance is not None:
        return max(0.0, 1.0 - ANCESTOR_DECAY * hop_distance)
    return 0.0


def compute_enum_score(in_template_enum: bool | None) -> float:
    """Compute the enum compliance component.

    Returns 1.0 if in enum, 0.0 if not, ENUM_NEUTRAL if no enum exists.
    """
    if in_template_enum is None:
        return ENUM_NEUTRAL
    return 1.0 if in_template_enum else 0.0


def compute_ontology_score(
    parse_success: bool,
    curie_label_valid: bool,
    relationship: str | None,
    hop_distance: int | None,
    in_template_enum: bool | None,
) -> float:
    """Compute composite ontology score, bounded [0, 1].

    ontology_score = W_PARSE * parse_ok
                   + W_LABEL * curie_label_valid
                   + W_HIER  * hierarchy_score
                   + W_ENUM  * enum_score

    Weights: W_PARSE=0.1, W_LABEL=0.1, W_HIER=0.5, W_ENUM=0.3 (sum=1.0)
    """
    if not parse_success:
        return 0.0

    hier = compute_hierarchy_score(relationship, hop_distance)
    enum = compute_enum_score(in_template_enum)

    return W_PARSE * 1.0 + W_LABEL * (1.0 if curie_label_valid else 0.0) + W_HIER * hier + W_ENUM * enum


def _add_timing_cost_stubs(result: dict[str, object]) -> None:
    """Add placeholder columns for response time and estimated cost.

    llm-matrix does not currently expose per-request timing or token counts
    in its results schema. When it does (or when we add wrapper-level timing),
    these columns should be populated:

    - response_time_s: wall-clock seconds for the LLM API call
    - prompt_tokens: input token count
    - completion_tokens: output token count
    - est_cost_usd: estimated cost based on provider pricing

    For now, these are None placeholders so the output schema is stable.
    """
    result["response_time_s"] = None
    result["prompt_tokens"] = None
    result["completion_tokens"] = None
    result["est_cost_usd"] = None


def score_envo_results(
    results_tsv: Path,
    enum_dir: Path = DEFAULT_ENUM_DIR,
    output_dir: Path | None = None,
) -> "pd.DataFrame":
    """Score env_broad_scale predictions with ontology-aware metrics.

    Reads an llm-matrix results TSV, adds scoring columns, writes enriched TSV.
    """
    import pandas as pd

    df = pd.read_csv(results_tsv, sep="\t")

    t0 = time.monotonic()
    adapter = get_envo_adapter()
    adapter_time = time.monotonic() - t0
    print(f"  oaklib adapter loaded in {adapter_time:.1f}s")

    scored_rows: list[dict[str, object]] = []

    for _, row in df.iterrows():
        result: dict[str, object] = {}

        # Parse ground truth
        truth_raw = str(row.get("case_ideal", ""))
        truth_parsed = parse_label_curie(truth_raw)

        # Parse prediction
        pred_raw = str(row.get("response_text", ""))
        pred_parsed = parse_label_curie(pred_raw)

        result["parse_success"] = pred_parsed is not None
        result["pred_label"] = pred_parsed[0] if pred_parsed else None
        result["pred_curie"] = pred_parsed[1] if pred_parsed else None
        result["truth_label"] = truth_parsed[0] if truth_parsed else None
        result["truth_curie"] = truth_parsed[1] if truth_parsed else None

        # CURIE-label validation (oaklib label lookup)
        if pred_parsed:
            result["curie_label_valid"] = validate_curie_label(adapter, pred_parsed[1], pred_parsed[0])
        else:
            result["curie_label_valid"] = None

        # Exact match (on cleaned CURIEs)
        if pred_parsed and truth_parsed:
            result["exact_match"] = pred_parsed[1] == truth_parsed[1]
        else:
            result["exact_match"] = False

        # Ontology relationship + hop distance
        if pred_parsed and truth_parsed:
            result["relationship"] = check_relationship(adapter, pred_parsed[1], truth_parsed[1])
            result["hop_distance"] = compute_hop_distance(adapter, pred_parsed[1], truth_parsed[1])
        else:
            result["relationship"] = None
            result["hop_distance"] = None

        # Template enum compliance
        template = row.get("sampleData") or _extract_template(str(row.get("case_original_input", "")))
        if template and pred_parsed:
            enum_curies = load_enum_for_template(template, enum_dir)
            if enum_curies is not None:
                result["in_template_enum"] = pred_parsed[1] in enum_curies
            else:
                result["in_template_enum"] = None
        else:
            result["in_template_enum"] = None

        # Composite score
        hop_raw = result["hop_distance"]
        hop_int: int | None = int(hop_raw) if isinstance(hop_raw, (int, float)) else None
        enum_val: bool | None = bool(result["in_template_enum"]) if result["in_template_enum"] is not None else None
        curie_valid = bool(result["curie_label_valid"]) if result["curie_label_valid"] is not None else False
        result["ontology_score"] = compute_ontology_score(
            parse_success=bool(result["parse_success"]),
            curie_label_valid=curie_valid,
            relationship=str(result["relationship"]) if result["relationship"] else None,
            hop_distance=hop_int,
            in_template_enum=enum_val,
        )

        # Timing/cost stubs
        _add_timing_cost_stubs(result)

        scored_rows.append(result)

    scored_df = pd.DataFrame(scored_rows)
    enriched = pd.concat([df, scored_df], axis=1)

    if output_dir is None:
        output_dir = results_tsv.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results_envo_scored.tsv"
    enriched.to_csv(output_path, sep="\t", index=False)

    scoring_time = time.monotonic() - t0
    print(f"  Scored {len(df)} rows in {scoring_time:.1f}s")

    print_envo_summary(enriched)

    return enriched


def _extract_template(original_input_str: str) -> str | None:
    """Extract sampleData from the original_input string repr."""
    m = re.search(r"sampleData['\"]?\s*[:=]\s*['\"]?(\w+)", original_input_str)
    return m.group(1) if m else None


def print_envo_summary(df: "pd.DataFrame") -> None:
    """Print a human-readable summary of ontology-aware scoring."""
    print("\n── ENVO Triad Scoring Summary ──")

    # Per-model ontology score
    print("\n  Model ranking (ontology_score):")
    model_scores = df.groupby("model")["ontology_score"].agg(["mean", "count"])
    model_scores = model_scores.sort_values("mean", ascending=False)
    for model, row in model_scores.iterrows():
        print(f"    {row['mean']:.3f}  {model}  (n={row['count']:.0f})")

    # Exact match rate
    print("\n  Exact match rate:")
    exact = df.groupby("model")["exact_match"].mean()
    for model, rate in exact.items():
        print(f"    {rate:.1%}  {model}")

    # Relationship breakdown
    print("\n  Relationship breakdown (descendant=more specific, ancestor=vaguer):")
    if "relationship" in df.columns:
        for model in df["model"].unique():
            subset = df[df["model"] == model]
            counts = subset["relationship"].value_counts()
            total = len(subset)
            parts = []
            for rel in ["exact", "descendant", "ancestor", "unrelated"]:
                n = counts.get(rel, 0)
                parts.append(f"{rel}={n}")
            parse_fail = subset["parse_success"].eq(False).sum()
            if parse_fail > 0:
                parts.append(f"parse_fail={parse_fail}")
            print(f"    {model}: {', '.join(parts)} (n={total})")

    # Enum compliance
    print("\n  Template enum compliance:")
    enum_rows = df[df["in_template_enum"].notna()]
    if not enum_rows.empty:
        for model in enum_rows["model"].unique():
            subset = enum_rows[enum_rows["model"] == model]
            rate = subset["in_template_enum"].mean()
            print(f"    {rate:.1%}  {model}  (n={len(subset)})")
    else:
        print("    N/A (no rows with template enums)")

    # Parse success rate
    print("\n  Parse success rate:")
    for model in df["model"].unique():
        subset = df[df["model"] == model]
        rate = subset["parse_success"].mean()
        print(f"    {rate:.1%}  {model}")

    # CURIE-label validity
    print("\n  CURIE-label validity (among parsed):")
    parsed = df[df["parse_success"].eq(True)]
    if not parsed.empty:
        for model in parsed["model"].unique():
            subset = parsed[parsed["model"] == model]
            valid_count = subset["curie_label_valid"].sum()
            print(f"    {valid_count}/{len(subset)}  {model}")

    # Score formula reminder
    print(
        f"\n  Score = {W_PARSE}×parse + {W_LABEL}×label_valid + {W_HIER}×hierarchy + {W_ENUM}×enum"
        f"  (descendant decay={DESCENDANT_DECAY}/hop, ancestor decay={ANCESTOR_DECAY}/hop)"
    )
