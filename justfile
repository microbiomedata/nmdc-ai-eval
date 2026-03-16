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

# Verify API auth works for all providers (1 cheap call each)
verify-auth:
    uv run python scripts/verify_auth.py

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

# Regenerate all suite YAMLs
generate: generate-sampledata generate-ebs

# --- Eval Runs (require API keys) ---

# Run a single eval suite
run suite_path:
    uv run python -m nmdc_ai_eval.run_suite {{ suite_path }}

# Run sampleData eval
run-sampledata:
    just run datasets/submission-metadata-prediction/sampledata-suite.yaml

# Run EBS eval
run-ebs:
    just run datasets/ebs-prediction/ebs-suite.yaml

# Score EBS results with ontology-aware metrics
score-ebs:
    uv run python -m nmdc_ai_eval.envo_scorer datasets/ebs-prediction/ebs-suite-output/results.tsv

# End-to-end sampleData eval: generate + run
eval-sampledata: clean-outputs generate-sampledata run-sampledata

# End-to-end EBS eval: generate + run + score
eval-ebs: clean-outputs generate-ebs run-ebs score-ebs

# Full eval: all datasets
eval-all: clean-outputs generate run-sampledata run-ebs score-ebs

# --- Cleanup ---

clean-cache:
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    rm -rf .pytest_cache .mypy_cache

clean-suites:
    rm -f datasets/submission-metadata-prediction/sampledata-suite.yaml
    rm -f datasets/ebs-prediction/ebs-suite.yaml

clean-outputs:
    rm -rf datasets/submission-metadata-prediction/sampledata-suite-output/
    rm -rf datasets/ebs-prediction/ebs-suite-output/
    rm -f datasets/submission-metadata-prediction/sampledata-suite.db
    rm -f datasets/ebs-prediction/ebs-suite.db
    rm -f datasets/ebs-prediction/results_envo_scored.tsv

clean-all: clean-cache clean-suites clean-outputs
