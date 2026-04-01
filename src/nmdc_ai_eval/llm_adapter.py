"""Generic LLM adapter that routes prompts through Simon Willison's llm library.

Use this to evaluate any LLM pipeline that builds prompts via add_message()
and generate() — the adapter collects messages and PDF attachments, then
sends everything through whichever llm-plugin model you specify.

Example::

    adapter = LLMLibraryAdapter(model_name="gpt-4o", system_prompt="You are a helpful assistant.")
    adapter.add_message(text="What is 2+2?")
    response = adapter.generate()
    print(response)               # "4"
    print(adapter.get_token_usage())  # {"input_tokens": 12, "output_tokens": 1}
"""

from __future__ import annotations

from typing import Any

import llm as llm_lib


class LLMLibraryAdapter:
    """Adapter that routes prompts through the llm library plugin ecosystem.

    Implements add_message(), add_schema_context(), and generate() so it can
    be used as a drop-in replacement for any LLM client that follows the same
    interface (e.g. the NMDC suggestor's LLMClient).

    Constructor args:
        model_name: any model recognized by `uv run llm models list`
        system_prompt: optional system prompt sent with every generate() call
    """

    def __init__(self, model_name: str, system_prompt: str | None = None) -> None:
        self.model = model_name
        self.access_provider = "llm"
        self.system_prompt = system_prompt
        self.messages: list[str] = []
        self._pdf_paths: list[str] = []
        self._last_response: Any = None

    def add_message(self, text: str = "", pdf_files: list[str] | None = None) -> None:
        """Add a text message and/or PDF file paths to the conversation."""
        if text:
            self.messages.append(text)
        if pdf_files:
            self._pdf_paths.extend(pdf_files)

    def add_schema_context(self, schema: str) -> None:
        """Add a schema description message to the conversation."""
        self.add_message(
            text="Utilize the following schema context to inform your metadata field recommendations:\n" + schema,
        )

    def add_schema_and_slot_examples(self) -> None:
        """Placeholder for future few-shot example support."""
        raise NotImplementedError

    def generate(
        self,
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        gemini_temperature: float = 0.4,
        **_kwargs: Any,
    ) -> str:
        """Send all accumulated messages to the model and return the response text.

        Args:
            model: override the model set at construction time
            max_tokens: passed to the llm prompt as max_tokens option
            gemini_temperature: temperature for generation (default 0.4)
            **_kwargs: accept and ignore extra kwargs for interface compatibility

        PDF files are sent as llm.Attachment objects. Models that don't support
        attachments will ignore them gracefully.
        """
        effective_model = model or self.model
        m = llm_lib.get_model(effective_model)
        full_prompt = "\n\n".join(self.messages)

        attachments: list[Any] = []
        for pdf_path in self._pdf_paths:
            try:
                attachments.append(llm_lib.Attachment(path=pdf_path, type="application/pdf"))
            except Exception:  # noqa: S110
                pass  # Skip PDFs that can't be attached

        prompt_kwargs: dict[str, Any] = {
            "system": self.system_prompt,
            "temperature": gemini_temperature,
        }
        if attachments:
            prompt_kwargs["attachments"] = attachments
        if max_tokens is not None:
            prompt_kwargs["max_tokens"] = max_tokens
        response = m.prompt(full_prompt, **prompt_kwargs)
        self._last_response = response
        return response.text()

    def get_token_usage(self) -> dict[str, int | None]:
        """Extract token usage from the last llm.Response."""
        if self._last_response is None:
            return {"input_tokens": None, "output_tokens": None}
        return {
            "input_tokens": getattr(self._last_response, "input_tokens", None),
            "output_tokens": getattr(self._last_response, "output_tokens", None),
        }

    def get_duration_ms(self) -> int | None:
        """Extract duration from the last llm.Response."""
        if self._last_response is None:
            return None
        try:
            result: int = self._last_response.duration_ms()
            return result
        except Exception:
            return None
