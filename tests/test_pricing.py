"""Tests for model pricing and cost estimation."""

from nmdc_ai_eval.pricing import _normalize_model_name, estimate_cost, get_pricing


class TestNormalizeModelName:
    def test_strips_anthropic_prefix(self) -> None:
        assert _normalize_model_name("anthropic/claude-sonnet-4-5") == "claude-sonnet-4-5"

    def test_strips_gemini_prefix(self) -> None:
        assert _normalize_model_name("gemini/gemini-2.5-flash") == "gemini-2.5-flash"

    def test_strips_openai_prefix(self) -> None:
        assert _normalize_model_name("openai/gpt-4o") == "gpt-4o"

    def test_no_prefix_unchanged(self) -> None:
        assert _normalize_model_name("gpt-4o") == "gpt-4o"

    def test_empty_string(self) -> None:
        assert _normalize_model_name("") == ""


class TestGetPricing:
    def test_exact_match(self) -> None:
        assert get_pricing("gpt-4o") == (2.50, 10.00)

    def test_with_provider_prefix(self) -> None:
        assert get_pricing("anthropic/claude-sonnet-4-5") == (3.00, 15.00)

    def test_with_version_suffix(self) -> None:
        assert get_pricing("anthropic/claude-haiku-4-5-20251001") == (0.80, 4.00)

    def test_gemini_with_prefix(self) -> None:
        assert get_pricing("gemini/gemini-2.5-flash") == (0.15, 0.60)

    def test_pnnl_models_zero_cost(self) -> None:
        assert get_pricing("gpt-5-project") == (0.0, 0.0)
        assert get_pricing("o3-project") == (0.0, 0.0)

    def test_unknown_model_returns_none(self) -> None:
        assert get_pricing("unknown-model-xyz") is None

    def test_all_models_yaml_names(self) -> None:
        """Every model in models.yaml should have pricing."""
        models_yaml_names = [
            "gpt-4o-mini",
            "gpt-4o",
            "anthropic/claude-sonnet-4-5",
            "anthropic/claude-haiku-4-5-20251001",
            "gemini/gemini-2.5-flash",
        ]
        for name in models_yaml_names:
            assert get_pricing(name) is not None, f"No pricing for {name}"

    def test_all_suggestor_model_names(self) -> None:
        """Every model the suggestor supports should have pricing."""
        suggestor_models = [
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gpt-5-project",
            "gpt-5.1-project",
            "gpt-5.2-project",
            "gpt-4.1-project",
            "o3-project",
            "o4-mini-project",
        ]
        for name in suggestor_models:
            assert get_pricing(name) is not None, f"No pricing for {name}"


class TestEstimateCost:
    def test_basic_cost(self) -> None:
        # 1000 input tokens of gpt-4o at $2.50/1M = $0.0025
        # 500 output tokens of gpt-4o at $10.00/1M = $0.005
        cost = estimate_cost("gpt-4o", input_tokens=1000, output_tokens=500)
        assert cost is not None
        assert abs(cost - 0.0075) < 1e-10

    def test_none_input_tokens(self) -> None:
        assert estimate_cost("gpt-4o", input_tokens=None, output_tokens=500) is None

    def test_none_output_tokens(self) -> None:
        assert estimate_cost("gpt-4o", input_tokens=1000, output_tokens=None) is None

    def test_unknown_model_returns_none(self) -> None:
        assert estimate_cost("unknown", input_tokens=1000, output_tokens=500) is None

    def test_zero_cost_models(self) -> None:
        cost = estimate_cost("gpt-5-project", input_tokens=100000, output_tokens=50000)
        assert cost == 0.0

    def test_zero_tokens(self) -> None:
        cost = estimate_cost("gpt-4o", input_tokens=0, output_tokens=0)
        assert cost == 0.0

    def test_large_token_counts(self) -> None:
        cost = estimate_cost("gpt-4o", input_tokens=1_000_000, output_tokens=100_000)
        assert cost is not None
        assert cost == 2.50 + 1.00  # $2.50 input + $1.00 output
