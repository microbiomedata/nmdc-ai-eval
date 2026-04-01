"""Tests for LLMLibraryAdapter — no LLM calls, just interface verification."""

from nmdc_ai_eval.llm_adapter import LLMLibraryAdapter


class TestLLMLibraryAdapter:
    def test_init(self) -> None:
        adapter = LLMLibraryAdapter(model_name="gpt-4o", system_prompt="test")
        assert adapter.model == "gpt-4o"
        assert adapter.access_provider == "llm"
        assert adapter.system_prompt == "test"
        assert adapter.messages == []

    def test_add_message(self) -> None:
        adapter = LLMLibraryAdapter(model_name="gpt-4o")
        adapter.add_message(text="hello")
        adapter.add_message(text="world")
        assert adapter.messages == ["hello", "world"]

    def test_add_message_with_pdfs(self) -> None:
        adapter = LLMLibraryAdapter(model_name="gpt-4o")
        adapter.add_message(text="see attached", pdf_files=["a.pdf", "b.pdf"])  # noqa: S108
        assert len(adapter.messages) == 1
        assert len(adapter._pdf_paths) == 2

    def test_add_empty_message_skipped(self) -> None:
        adapter = LLMLibraryAdapter(model_name="gpt-4o")
        adapter.add_message(text="")
        assert adapter.messages == []

    def test_add_schema_context(self) -> None:
        adapter = LLMLibraryAdapter(model_name="gpt-4o")
        adapter.add_schema_context("some schema")
        assert len(adapter.messages) == 1
        assert "schema context" in adapter.messages[0]

    def test_token_usage_before_generate(self) -> None:
        adapter = LLMLibraryAdapter(model_name="gpt-4o")
        usage = adapter.get_token_usage()
        assert usage == {"input_tokens": None, "output_tokens": None}

    def test_duration_before_generate(self) -> None:
        adapter = LLMLibraryAdapter(model_name="gpt-4o")
        assert adapter.get_duration_ms() is None

    def test_system_prompt_default_none(self) -> None:
        adapter = LLMLibraryAdapter(model_name="gpt-4o")
        assert adapter.system_prompt is None
