# List all commands
_default:
    @just --list

# Verify system prerequisites are installed
check-prereqs:
    @echo "Checking prerequisites..."
    @uv --version
    @just --version
    @git --version
    @uv run python3 --version
    @echo ""
    @echo "All prerequisites satisfied."

# Install dependencies
setup: check-prereqs
    uv sync
    @echo ""
    @echo "Now configure API keys:"
    @echo "  uv run llm keys set openai"
    @echo "  uv run llm keys set anthropic"

# Regenerate uv.lock from pyproject.toml
lock:
    uv lock

# Run ruff linter and format check
lint:
    uv run ruff check src/ tests/ datasets/
    uv run ruff format --check src/ tests/ datasets/

# Auto-fix lint and format issues
fix:
    uv run ruff check --fix src/ tests/ datasets/
    uv run ruff format src/ tests/ datasets/

# Type check
typecheck:
    uv run mypy src/

# Check for unused/missing dependencies
deptry:
    uv run deptry src/

# Scan dependencies for known CVEs (ignores acknowledged vulns with no fix)
audit:
    uv run pip-audit --ignore-vuln CVE-2025-69872

# Run tests (excludes slow/oaklib tests by default)
test:
    uv run pytest -v -m "not slow"

# Run slow tests (oaklib integration, requires network on first run)
test-slow:
    uv run pytest -v -m slow

# Run a single eval suite
run suite_path:
    uv run python -m nmdc_ai_eval.run_suite {{ suite_path }}

# Run sampleData OpenAI eval suite
run-sampledata-openai:
    just run datasets/submission-metadata-prediction/sampledata-suite-openai.yaml

# Run sampleData Anthropic eval suite
run-sampledata-anthropic:
    just run datasets/submission-metadata-prediction/sampledata-suite-anthropic.yaml

# Run all sampleData eval suites
run-sampledata-all: run-sampledata-openai run-sampledata-anthropic

# Regenerate sampleData suites then run all (cleans stale .db caches first)
eval-sampledata per_category="10" min_pool="5": clean-outputs (generate-sampledata per_category min_pool) run-sampledata-all

# Regenerate sampleData suite YAMLs (smoke test: soil_data only, see README)
generate-sampledata per_category="10" min_pool="5" provider="both":
    uv run python datasets/submission-metadata-prediction/generate_suite.py --per-category {{ per_category }} --min-pool {{ min_pool }} --provider {{ provider }}

# Full QC: lint + typecheck + deptry + audit + test
qc: lint typecheck deptry audit test

# Install pre-commit hooks (commit + push)
pre-commit-install:
    uv run pre-commit install
    uv run pre-commit install --hook-type pre-push

# Run pre-commit on all files
pre-commit:
    uv run pre-commit run --all-files

# --- env_broad_scale (EBS) Prediction ---

# Regenerate EBS suite YAMLs
generate-ebs per_category="10" min_pool="10" provider="both":
    uv run python datasets/ebs-prediction/generate_suite.py --per-category {{ per_category }} --min-pool {{ min_pool }} --provider {{ provider }}

# Run EBS OpenAI eval suite
run-ebs-openai:
    just run datasets/ebs-prediction/ebs-suite-openai.yaml

# Run EBS Anthropic eval suite
run-ebs-anthropic:
    just run datasets/ebs-prediction/ebs-suite-anthropic.yaml

# Run all EBS eval suites
run-ebs-all: run-ebs-openai run-ebs-anthropic

# Score EBS results with ontology-aware metrics
score-ebs provider:
    uv run python -m nmdc_ai_eval.score_envo datasets/ebs-prediction/ebs-suite-{{ provider }}-output/results.tsv

# Score all EBS results
score-ebs-all: (score-ebs "openai") (score-ebs "anthropic")

# End-to-end EBS eval: generate, run, score (cleans stale .db caches first)
eval-ebs per_category="10" min_pool="10": clean-outputs (generate-ebs per_category min_pool) run-ebs-all score-ebs-all

# --- Cleanup ---

# Remove Python caches (.pytest_cache, __pycache__, .mypy_cache)
clean-cache:
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    rm -rf .pytest_cache .mypy_cache

# Remove generated suite YAMLs (re-create with generate-sampledata / generate-ebs)
clean-suites:
    rm -f datasets/submission-metadata-prediction/sampledata-suite-*.yaml
    rm -f datasets/ebs-prediction/ebs-suite-*.yaml

# Remove LLM eval outputs (results TSVs, scored TSVs, llm-matrix .db files)
clean-outputs:
    rm -rf datasets/submission-metadata-prediction/sampledata-suite-*-output/
    rm -rf datasets/ebs-prediction/ebs-suite-*-output/
    rm -f datasets/submission-metadata-prediction/sampledata-suite-*.db
    rm -f datasets/ebs-prediction/ebs-suite-*.db
    rm -f datasets/ebs-prediction/results_envo_scored.tsv

# Remove all generated/cached artifacts (suites + outputs + caches)
clean-all: clean-cache clean-suites clean-outputs
