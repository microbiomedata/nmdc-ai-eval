"""Register PNNL AI Incubator models for llm-matrix suite evaluations."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, cast

import llm
import yaml
from dotenv import load_dotenv
from llm.default_plugins.openai_models import Chat

if TYPE_CHECKING:
    from llm import Prompt

load_dotenv()


def _load_pnnl_models() -> list[str]:
    models_path = Path(__file__).parents[2] / "datasets" / "models.yaml"
    with open(models_path) as models_file:
        data = cast(dict[str, object], yaml.safe_load(models_file) or {})
    models = data.get("pnnl_models", [])
    return [str(model) for model in models] if isinstance(models, list) else []


class PNNLChatModel(Chat):
    """PNNL's OpenAI-compatible Chat Completions model."""

    needs_key = "pnnl"

    def build_kwargs(self, prompt: "Prompt", stream: bool) -> dict[str, Any]:
        """Exclude temperature, which PNNL's GPT-5 models do not support."""
        kwargs = cast(dict[str, Any], super().build_kwargs(prompt, stream))  # type: ignore[no-untyped-call]
        kwargs.pop("temperature", None)
        return kwargs


@llm.hookimpl
def register_models(register: object) -> None:
    """Register repository-owned aliases for the configured PNNL models."""
    base_url = os.environ.get("AI_INCUBATOR_BASE_URL")
    if not base_url:
        return
    register_model: Callable[..., Any] = cast(Callable[..., Any], register)
    for name in _load_pnnl_models():
        model = PNNLChatModel(  # type: ignore[no-untyped-call]
            f"pnnl/{name}", model_name=name, api_base=base_url, reasoning=True
        )
        register_model(model)
