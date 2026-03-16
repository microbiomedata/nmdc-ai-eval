"""Test suite generation logic via imports — no subprocess, no network."""

import sys
from pathlib import Path

import pytest
import yaml

DATASETS_DIR = Path(__file__).parent.parent / "datasets"
MODELS_YAML = DATASETS_DIR / "models.yaml"
TSV_PATH = DATASETS_DIR / "submission-metadata-prediction" / "eval_input_target_pairs.tsv"

# Add dataset dirs to path so we can import generate_suite modules
sys.path.insert(0, str(DATASETS_DIR / "submission-metadata-prediction"))
sys.path.insert(0, str(DATASETS_DIR / "ebs-prediction"))


class TestModelsYaml:
    def test_loads(self) -> None:
        with open(MODELS_YAML) as f:
            data = yaml.safe_load(f)
        assert "models" in data
        assert isinstance(data["models"], list)

    def test_has_models(self) -> None:
        with open(MODELS_YAML) as f:
            models = yaml.safe_load(f)["models"]
        assert len(models) >= 1

    def test_no_duplicates(self) -> None:
        with open(MODELS_YAML) as f:
            models = yaml.safe_load(f)["models"]
        assert len(models) == len(set(models)), f"Duplicate models: {[m for m in models if models.count(m) > 1]}"

    def test_all_strings(self) -> None:
        with open(MODELS_YAML) as f:
            models = yaml.safe_load(f)["models"]
        for m in models:
            assert isinstance(m, str), f"Model entry should be a string, got {type(m)}: {m}"

    def test_all_models_registered_with_llm(self) -> None:
        """Every model in models.yaml must be recognized by an installed llm plugin."""
        import llm

        with open(MODELS_YAML) as f:
            models = yaml.safe_load(f)["models"]
        for name in models:
            try:
                llm.get_model(name)
            except llm.UnknownModelError:
                pytest.fail(
                    f"Model '{name}' not recognized by llm. "
                    f"Run `uv run llm models list` to see available models. "
                    f"You may need to install a plugin (llm-claude-3, llm-gemini, etc.)."
                )


class TestSampleDataGenerator:
    @pytest.fixture(scope="class")
    def gen(self):  # type: ignore[no-untyped-def]
        import importlib

        # The module is named generate_suite in both dirs; reload to get the right one
        sys.path.insert(0, str(DATASETS_DIR / "submission-metadata-prediction"))
        spec = importlib.util.spec_from_file_location(
            "sampledata_gen", DATASETS_DIR / "submission-metadata-prediction" / "generate_suite.py"
        )
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_load_models(self, gen) -> None:  # type: ignore[no-untyped-def]
        models = gen.load_models()
        assert isinstance(models, list)
        assert len(models) >= 1

    def test_load_rows(self, gen) -> None:  # type: ignore[no-untyped-def]
        rows = gen.load_rows(TSV_PATH)
        assert len(rows) > 0
        assert "study_name" in rows[0]
        assert "sampleData" in rows[0]

    def test_sample_by_category(self, gen) -> None:  # type: ignore[no-untyped-def]
        rows = gen.load_rows(TSV_PATH)
        sampled = gen.sample_by_category(rows, n_per_category=2, min_pool=2)
        assert len(sampled) > 0
        # Each row should have sampleData
        for row in sampled:
            assert row["sampleData"]

    def test_make_cases(self, gen) -> None:  # type: ignore[no-untyped-def]
        rows = gen.load_rows(TSV_PATH)
        sampled = gen.sample_by_category(rows, n_per_category=2, min_pool=2)
        cases = gen.make_cases(sampled)
        assert len(cases) == len(sampled)
        for case in cases:
            assert "input" in case
            assert "ideal" in case
            assert "tags" in case
            assert "original_input" in case

    def test_make_suite(self, gen) -> None:  # type: ignore[no-untyped-def]
        cases = [{"input": "test", "ideal": "soil_data", "tags": ["soil_data"], "original_input": {}}]
        models = ["litellm/gpt-4o"]
        suite = gen.make_suite(cases, models)
        assert suite["name"] == "nmdc-sampledata-prediction"
        assert suite["matrix"]["hyperparameters"]["model"] == models
        assert len(suite["cases"]) == 1

    def test_description_truncation(self, gen) -> None:  # type: ignore[no-untyped-def]
        rows = [
            {
                "study_name": "test study",
                "description": "x" * 600,
                "notes": "",
                "sampleData": "soil_data",
            }
        ]
        cases = gen.make_cases(rows)
        # Prompt should have truncated description
        assert len(cases[0]["input"]) < 600


class TestEBSGenerator:
    @pytest.fixture(scope="class")
    def gen(self):  # type: ignore[no-untyped-def]
        import importlib

        spec = importlib.util.spec_from_file_location("ebs_gen", DATASETS_DIR / "ebs-prediction" / "generate_suite.py")
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_load_models(self, gen) -> None:  # type: ignore[no-untyped-def]
        models = gen.load_models()
        assert isinstance(models, list)
        assert len(models) >= 1

    def test_load_allowed_values_soil(self, gen) -> None:  # type: ignore[no-untyped-def]
        values = gen.load_allowed_values("soil_data")
        assert values is not None
        assert len(values) > 0
        # Each value should be "label [CURIE]" format
        for v in values:
            assert "[ENVO:" in v

    def test_load_allowed_values_unknown(self, gen) -> None:  # type: ignore[no-untyped-def]
        assert gen.load_allowed_values("nonexistent_data") is None

    def test_sample_by_category(self, gen) -> None:  # type: ignore[no-untyped-def]
        rows = gen.load_rows(TSV_PATH)
        sampled = gen.sample_by_category(rows, n_per_category=2, min_pool=2)
        assert len(sampled) > 0
        for row in sampled:
            assert row["env_broad_scale"].strip()

    def test_make_cases_include_allowed_values(self, gen) -> None:  # type: ignore[no-untyped-def]
        rows = gen.load_rows(TSV_PATH)
        # Get a soil row
        soil_rows = [r for r in rows if r["sampleData"] == "soil_data" and r["env_broad_scale"].strip()][:1]
        if soil_rows:
            cases = gen.make_cases(soil_rows)
            assert len(cases) == 1
            # Soil template should have allowed values in the prompt
            assert "Allowed env_broad_scale values" in cases[0]["input"]

    def test_make_suite(self, gen) -> None:  # type: ignore[no-untyped-def]
        cases = [{"input": "test", "ideal": "biome [ENVO:00000428]", "tags": ["soil_data"], "original_input": {}}]
        models = ["litellm/gpt-4o"]
        suite = gen.make_suite(cases, models)
        assert suite["name"] == "nmdc-ebs-prediction"
        assert suite["matrix"]["hyperparameters"]["model"] == models
