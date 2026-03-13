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

# Run tests
test:
    uv run pytest -v

# Run a single eval suite
run suite_path:
    uv run python -m nmdc_ai_eval.run_suite {{ suite_path }}

# Run OpenAI eval suite
run-openai:
    just run datasets/submission-metadata-prediction/sampledata-suite-openai.yaml

# Run Anthropic eval suite
run-anthropic:
    just run datasets/submission-metadata-prediction/sampledata-suite-anthropic.yaml

# Run all eval suites
run-all: run-openai run-anthropic

# Regenerate suites then run all (full end-to-end eval)
eval per_category="5": (generate per_category) run-all

# Regenerate suite YAMLs (default 5 per category, both providers)
generate per_category="5" provider="both":
    uv run python datasets/submission-metadata-prediction/generate_suite.py --per-category {{ per_category }} --provider {{ provider }}

# Full QC: lint + typecheck + deptry + audit + test
qc: lint typecheck deptry audit test

# Install pre-commit hooks (commit + push)
pre-commit-install:
    uv run pre-commit install
    uv run pre-commit install --hook-type pre-push

# Run pre-commit on all files
pre-commit:
    uv run pre-commit run --all-files
