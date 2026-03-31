"""Model pricing for cost estimation.

Approximate costs per 1M tokens (USD), sourced from provider pricing pages.
PNNL AI Incubator models show zero cost (internal allocation, not billed per-token).

Update this table when providers change pricing or new models are added.
"""

from __future__ import annotations

# (input_cost_per_1m_tokens, output_cost_per_1m_tokens)
_PRICING: dict[str, tuple[float, float]] = {
    # OpenAI — https://openai.com/pricing
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    # Anthropic — https://www.anthropic.com/pricing
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-haiku-4-5": (0.80, 4.00),
    # Google AI Studio / Vertex AI — https://ai.google.dev/pricing
    "gemini-2.5-flash": (0.15, 0.60),
    "gemini-2.5-pro": (1.25, 10.00),
    # PNNL AI Incubator — internal allocation, no per-token billing
    "gpt-5-project": (0.0, 0.0),
    "gpt-5.1-project": (0.0, 0.0),
    "gpt-5.2-project": (0.0, 0.0),
    "gpt-4.1-project": (0.0, 0.0),
    "o3-project": (0.0, 0.0),
    "o4-mini-project": (0.0, 0.0),
}


def _normalize_model_name(model: str) -> str:
    """Strip provider prefixes used by llm plugins (e.g. 'anthropic/', 'gemini/')."""
    for prefix in ("anthropic/", "gemini/", "openai/"):
        if model.startswith(prefix):
            return model[len(prefix) :]
    return model


def get_pricing(model: str) -> tuple[float, float] | None:
    """Look up (input_cost, output_cost) per 1M tokens for a model.

    Returns None if the model is not in the pricing table.
    Handles provider prefixes (e.g. 'anthropic/claude-sonnet-4-5')
    and version suffixes (e.g. 'claude-haiku-4-5-20251001').
    """
    normalized = _normalize_model_name(model)
    if normalized in _PRICING:
        return _PRICING[normalized]
    for key, price in _PRICING.items():
        if normalized.startswith(key):
            return price
    return None


def estimate_cost(
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
) -> float | None:
    """Estimate cost in USD for a single LLM call.

    Returns None if pricing is unavailable or token counts are missing.
    """
    if input_tokens is None or output_tokens is None:
        return None
    pricing = get_pricing(model)
    if pricing is None:
        return None
    input_cost, output_cost = pricing
    return (input_tokens * input_cost + output_tokens * output_cost) / 1_000_000
