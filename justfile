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
    @echo "  cp .env.example .env   # then edit with your keys"
    @echo "  uv run llm keys set openai"
    @echo "  uv run llm keys set anthropic"

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

# Scan dependencies for known CVEs
audit:
    uv run pip-audit

# Run tests
test:
    uv run pytest -v

# Run the sampleData eval suite
run suite_path="datasets/submission-metadata-prediction/sampledata-suite.yaml":
    uv run python -m nmdc_ai_eval.run_suite {{ suite_path }}

# Regenerate sampleData suite YAML (default 5 per category)
generate per_category="5":
    uv run python datasets/submission-metadata-prediction/generate_suite.py --per-category {{ per_category }}

# Full QC: lint + typecheck + deptry + test
qc: lint typecheck deptry test

# Install pre-commit hooks
pre-commit-install:
    uv run pre-commit install

# Run pre-commit on all files
pre-commit:
    uv run pre-commit run --all-files
