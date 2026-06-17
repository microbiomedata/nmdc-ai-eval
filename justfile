# List all commands
_default:
    @just --list

# --- QC (no evals, no API calls) ---

# Fix and check everything
all: fix check

# Auto-fix lint and format issues
fix:
    uv run ruff check --fix src/ tests/ datasets/
    uv run ruff format src/ tests/ datasets/

# Run all checks via pre-commit (single source of truth)
check:
    uv run pre-commit run --all-files

# Run tests (excludes API tests)
test:
    uv run pytest -v -m "not api"

# Run tests with coverage report
coverage:
    uv run pytest -m "not api" --cov=nmdc_ai_eval --cov-report=term-missing

# Audit installed deps for known vulnerabilities (advisory; also runs weekly in CI)
audit:
    uv run pip-audit --ignore-vuln CVE-2025-69872 --ignore-vuln CVE-2026-4539

# Verify API auth works for all providers (1 cheap call each)
verify-auth:
    uv run python scripts/verify_auth.py

# Probe which model names our Vertex SA+project can actually reach
probe-vertex-garden:
    uv run python scripts/probe_vertex_garden.py

# Probe cost tiers across OpenAI / Anthropic / Google (`--channel=llm|cborg`)
probe-tiers *args="":
    uv run python scripts/probe_model_tiers.py {{ args }}

# --- Setup ---

# Install dependencies and pre-commit hooks
setup:
    uv sync
    uv run pre-commit install
    @echo ""
    @echo "Set API keys as env vars (OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY)"
    @echo "or configure the llm key store: uv run llm keys set openai"

# --- Suite Generation ---

# Regenerate sampleData suite YAML
generate-sampledata per_category="10" min_pool="5":
    uv run python datasets/submission-metadata-prediction/generate_suite.py --per-category {{ per_category }} --min-pool {{ min_pool }}

# Regenerate EBS suite YAML
generate-ebs per_category="10" min_pool="10":
    uv run python datasets/ebs-prediction/generate_suite.py --per-category {{ per_category }} --min-pool {{ min_pool }}

# Regenerate field-guidance suite YAML (requires nmdc_data_dev MongoDB)
generate-field-guidance:
    uv run python datasets/field-guidance/generate_suite.py

# Regenerate env-triad suite YAML (hits NMDC public API). Pass env=dev to use api-dev if prod is down.
generate-env-triad env="prod":
    uv run python datasets/env-triad-prediction/generate_suite.py --env {{ env }}

# Regenerate TSV-based suite YAMLs (no external dependencies)
generate: generate-sampledata generate-ebs

# Regenerate all suite YAMLs (field-guidance requires nmdc_data_dev MongoDB)
generate-all: generate generate-field-guidance generate-env-triad

# --- Eval Runs (require API keys) ---

# Run a single eval suite. Optional: scorer_model="gpt-4o-mini" to pin the scoring model.
run suite_path scorer_model="":
    uv run python -m nmdc_ai_eval.run_suite {{ suite_path }} {{ if scorer_model != "" { "--scorer-model " + scorer_model } else { "" } }}

# Run sampleData eval
run-sampledata:
    just run datasets/submission-metadata-prediction/sampledata-suite.yaml

# Run EBS eval
run-ebs:
    just run datasets/ebs-prediction/ebs-suite.yaml

# Score EBS results with ontology-aware metrics
score-ebs:
    uv run python -m nmdc_ai_eval.envo_scorer datasets/ebs-prediction/ebs-suite-output/results.tsv

# Run field-guidance eval via llm-matrix (no DOI/PDF, uses models.yaml list)
run-field-guidance:
    just run datasets/field-guidance/field-guidance-suite.yaml

# Run env-triad eval via llm-matrix
run-env-triad:
    just run datasets/env-triad-prediction/env-triad-suite.yaml

# Field guidance eval: models × enrichment × verification → summary
full-eval *args="":
    uv run python datasets/field-guidance/run_full_eval.py {{ args }}

# Compare all field-guidance eval results
compare-pipeline-results *args="":
    uv run python datasets/field-guidance/compare_pipeline_results.py {{ args }}

# End-to-end value prediction evals (llm-matrix, no DOI/PDF)
eval-sampledata: clean-sampledata-outputs generate-sampledata run-sampledata
eval-ebs: clean-ebs-outputs generate-ebs run-ebs score-ebs
eval-env-triad: clean-env-triad-outputs generate-env-triad run-env-triad

# Smoke test: N cases x model(s). Defaults: 3 cases x gpt-4o-mini (~$0.001). env=dev for fallback API.
# Example: just pilot-env-triad 50 "gpt-4o-mini,cborg/claude-sonnet-4-6" dev
pilot-env-triad max_cases="3" model="gpt-4o-mini" env="prod": clean-env-triad-outputs
    uv run python datasets/env-triad-prediction/generate_suite.py --max-cases {{ max_cases }} --models {{ model }} --env {{ env }}
    just run-env-triad

# --- Cleanup ---

clean-cache:
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    rm -rf .pytest_cache .mypy_cache

clean-suites:
    rm -f datasets/submission-metadata-prediction/sampledata-suite.yaml
    rm -f datasets/ebs-prediction/ebs-suite.yaml
    rm -f datasets/field-guidance/field-guidance-suite.yaml
    rm -f datasets/env-triad-prediction/env-triad-suite.yaml

clean-sampledata-outputs:
    rm -rf datasets/submission-metadata-prediction/sampledata-suite-output/
    rm -f datasets/submission-metadata-prediction/sampledata-suite.db

clean-ebs-outputs:
    rm -rf datasets/ebs-prediction/ebs-suite-output/
    rm -f datasets/ebs-prediction/ebs-suite.db
    rm -f datasets/ebs-prediction/results_envo_scored.tsv

clean-field-guidance-outputs:
    rm -rf datasets/field-guidance/field-guidance-suite-output/
    rm -f datasets/field-guidance/field-guidance-suite.db
    rm -rf datasets/field-guidance/pipeline-results/

clean-env-triad-outputs:
    rm -rf datasets/env-triad-prediction/env-triad-suite-output/
    rm -f datasets/env-triad-prediction/env-triad-suite.db

clean-outputs: clean-sampledata-outputs clean-ebs-outputs clean-field-guidance-outputs clean-env-triad-outputs

clean-all: clean-cache clean-suites clean-outputs
