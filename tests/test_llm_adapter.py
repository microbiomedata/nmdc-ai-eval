"""Tests for LLMLibraryAdapter — mocked LLM calls, no real API usage."""

from unittest.mock import MagicMock, patch

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

    @patch("nmdc_ai_eval.llm_adapter.llm_lib")
    def test_generate_calls_model(self, mock_llm: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.text.return_value = '{"result": "ok"}'
        mock_response.input_tokens = 100
        mock_response.output_tokens = 50
        mock_model = MagicMock()
        mock_model.prompt.return_value = mock_response
        mock_llm.get_model.return_value = mock_model

        adapter = LLMLibraryAdapter(model_name="gpt-4o", system_prompt="be helpful")
        adapter.add_message(text="hello")
        result = adapter.generate()

        assert result == '{"result": "ok"}'
        mock_llm.get_model.assert_called_once_with("gpt-4o")
        mock_model.prompt.assert_called_once()
        call_kwargs = mock_model.prompt.call_args
        assert call_kwargs.kwargs["system"] == "be helpful"
        assert call_kwargs.kwargs["temperature"] == 0.4

    @patch("nmdc_ai_eval.llm_adapter.llm_lib")
    def test_generate_model_override(self, mock_llm: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.text.return_value = "ok"
        mock_model = MagicMock()
        mock_model.prompt.return_value = mock_response
        mock_llm.get_model.return_value = mock_model

        adapter = LLMLibraryAdapter(model_name="gpt-4o")
        adapter.add_message(text="hello")
        adapter.generate(model="gpt-5.2")

        mock_llm.get_model.assert_called_once_with("gpt-5.2")

    @patch("nmdc_ai_eval.llm_adapter.llm_lib")
    def test_generate_max_tokens(self, mock_llm: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.text.return_value = "ok"
        mock_model = MagicMock()
        mock_model.prompt.return_value = mock_response
        mock_llm.get_model.return_value = mock_model

        adapter = LLMLibraryAdapter(model_name="gpt-4o")
        adapter.add_message(text="hello")
        adapter.generate(max_tokens=500)

        call_kwargs = mock_model.prompt.call_args.kwargs
        assert call_kwargs["max_tokens"] == 500

    @patch("nmdc_ai_eval.llm_adapter.llm_lib")
    def test_token_usage_after_generate(self, mock_llm: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.text.return_value = "ok"
        mock_response.input_tokens = 150
        mock_response.output_tokens = 30
        mock_model = MagicMock()
        mock_model.prompt.return_value = mock_response
        mock_llm.get_model.return_value = mock_model

        adapter = LLMLibraryAdapter(model_name="gpt-4o")
        adapter.add_message(text="hello")
        adapter.generate()

        usage = adapter.get_token_usage()
        assert usage == {"input_tokens": 150, "output_tokens": 30}

    @patch("nmdc_ai_eval.llm_adapter.llm_lib")
    def test_duration_after_generate(self, mock_llm: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.text.return_value = "ok"
        mock_response.duration_ms.return_value = 1234
        mock_model = MagicMock()
        mock_model.prompt.return_value = mock_response
        mock_llm.get_model.return_value = mock_model

        adapter = LLMLibraryAdapter(model_name="gpt-4o")
        adapter.add_message(text="hello")
        adapter.generate()

        assert adapter.get_duration_ms() == 1234
