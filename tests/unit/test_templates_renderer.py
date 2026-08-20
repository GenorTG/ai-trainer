"""Tests for inference_server/templates/renderer.py."""

from __future__ import annotations

import pytest


@pytest.mark.unit
class TestRenderChat:
    def test_chatml_fallback_no_template(self):
        from inference_server.templates.renderer import render_chat
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        result = render_chat("", messages)
        assert "<|system|>" in result
        assert "<|user|>" in result
        assert "You are helpful" in result
        assert "Hello" in result

    def test_chatml_with_bos(self):
        from inference_server.templates.renderer import render_chat
        messages = [{"role": "user", "content": "Hi"}]
        result = render_chat("", messages, bos_token="<bos>")
        assert result.startswith("<bos>")

    def test_chatml_generation_prompt(self):
        from inference_server.templates.renderer import render_chat
        messages = [{"role": "user", "content": "Hi"}]
        result = render_chat("", messages, add_generation_prompt=True)
        assert "<|assistant|>" in result

    def test_chatml_no_generation_prompt(self):
        from inference_server.templates.renderer import render_chat
        messages = [{"role": "user", "content": "Hi"}]
        result = render_chat("", messages, add_generation_prompt=False)
        assert "<|assistant|>" not in result

    def test_tool_role_in_chatml(self):
        from inference_server.templates.renderer import render_chat
        messages = [{"role": "tool", "content": "42"}]
        result = render_chat("", messages)
        assert "42" in result

    def test_assistant_role_in_chatml(self):
        from inference_server.templates.renderer import render_chat
        messages = [{"role": "assistant", "content": "Sure!"}]
        result = render_chat("", messages)
        assert "Sure!" in result

    def test_custom_template_renders(self):
        from inference_server.templates.renderer import render_chat
        template = "{% for m in messages %}ROLE={{ m.role }}|CONTENT={{ m.content }}\n{% endfor %}"
        messages = [{"role": "user", "content": "test"}]
        result = render_chat(template, messages, add_generation_prompt=False)
        assert "ROLE=user" in result
        assert "CONTENT=test" in result

    def test_template_with_tools(self):
        from inference_server.templates.renderer import render_chat
        template = "{% for t in tools %}TOOL={{ t.name }}{% endfor %}{% for m in messages %}{{ m.content }}{% endfor %}"
        messages = [{"role": "user", "content": "Hi"}]
        tools = [{"name": "search"}]
        result = render_chat(template, messages, tools=tools, add_generation_prompt=False)
        assert "TOOL=search" in result
        assert "Hi" in result

    def test_template_error_falls_back(self):
        from inference_server.templates.renderer import render_chat
        bad_template = "{{ invalid syntax }}"
        messages = [{"role": "user", "content": "Hi"}]
        result = render_chat(bad_template, messages)
        # Falls back to ChatML
        assert "<|user|>" in result

    def test_empty_messages(self):
        from inference_server.templates.renderer import render_chat
        result = render_chat("", [], add_generation_prompt=True)
        assert "<|assistant|>" in result

    def test_multiple_messages(self):
        from inference_server.templates.renderer import render_chat
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
        ]
        result = render_chat("", messages, add_generation_prompt=False)
        assert "sys" in result
        assert "q1" in result
        assert "a1" in result
        assert "q2" in result

    def test_tojson_filter(self):
        from inference_server.templates.renderer import render_chat
        template = "{{ tools | tojson }}"
        messages = [{"role": "user", "content": "Hi"}]
        tools = [{"name": "search"}]
        result = render_chat(template, messages, tools=tools, add_generation_prompt=False)
        assert "search" in result


@pytest.mark.unit
class TestChatmlWrap:
    def test_wrap_user(self):
        from inference_server.templates.renderer import _chatml_wrap
        result = _chatml_wrap("user", "\nHello")
        assert result == "<|user|>\nHello</im_end>"

    def test_wrap_system(self):
        from inference_server.templates.renderer import _chatml_wrap
        result = _chatml_wrap("system", "\nInstructions")
        assert "<|system|>" in result
        assert "</im_end>" in result


@pytest.mark.unit
class TestRenderChatmlFallback:
    def test_all_roles(self):
        from inference_server.templates.renderer import _render_chatml_fallback
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "user_msg"},
            {"role": "assistant", "content": "asst_msg"},
            {"role": "tool", "content": "tool_result"},
        ]
        result = _render_chatml_fallback(messages)
        assert "sys" in result
        assert "user_msg" in result
        assert "asst_msg" in result
        assert "tool_result" in result

    def test_unknown_role_skipped(self):
        from inference_server.templates.renderer import _render_chatml_fallback
        messages = [{"role": "custom", "content": "skip me"}]
        result = _render_chatml_fallback(messages)
        assert "skip me" not in result

    def test_message_without_content(self):
        from inference_server.templates.renderer import _render_chatml_fallback
        messages = [{"role": "user"}]
        result = _render_chatml_fallback(messages)
        assert "<|user|>" in result


@pytest.mark.unit
class TestExtractTemplateFromGguf:
    def test_returns_defaults_on_error(self, mocker):
        from inference_server.templates.renderer import extract_template_from_gguf
        mocker.patch.dict("sys.modules", {"llama_cpp": None}, clear=False)
        result = extract_template_from_gguf("/nonexistent/model.gguf")
        assert result["chat_template"] == ""
        assert result["bos_token"] == "<bos>"
        assert result["eos_token"] == "<eos>"
        assert result["supports_tools"] is False

    def test_extracts_from_gguf(self, mocker):
        from inference_server.templates.renderer import extract_template_from_gguf
        mock_llama = mocker.MagicMock()
        mock_llama.metadata = {
            "tokenizer.chat_template": "{% for m in messages %}{{ m.role }}{% endfor %}",
            "tokenizer.ggml.bos_token": "<s>",
            "tokenizer.ggml.eos_token": "</s>",
            "tokenizer.ggml.add_bos_token": True,
        }
        mock_llama_cls = mocker.MagicMock(return_value=mock_llama)
        mocker.patch.dict("sys.modules", {"llama_cpp": mocker.MagicMock(Llama=mock_llama_cls)})
        result = extract_template_from_gguf("/models/test.gguf")
        assert "{{ m.role }}" in result["chat_template"]
        assert result["bos_token"] == "<s>"
        assert result["eos_token"] == "</s>"
        assert result["add_bos"] is True

    def test_detects_tool_support(self, mocker):
        from inference_server.templates.renderer import extract_template_from_gguf
        mock_llama = mocker.MagicMock()
        mock_llama.metadata = {
            "tokenizer.chat_template": "{% for m in messages %}<tool_call>{{ m.content }}</tool_call>{% endfor %}",
            "tokenizer.ggml.bos_token": "<bos>",
            "tokenizer.ggml.eos_token": "<eos>",
            "tokenizer.ggml.add_bos_token": True,
        }
        mock_llama_cls = mocker.MagicMock(return_value=mock_llama)
        mocker.patch.dict("sys.modules", {"llama_cpp": mocker.MagicMock(Llama=mock_llama_cls)})
        result = extract_template_from_gguf("/models/tool_model.gguf")
        assert result["supports_tools"] is True


@pytest.mark.unit
class TestRenderWithModelTemplate:
    def test_one_shot_render(self, mocker):
        from inference_server.templates.renderer import render_with_model_template
        mock_llama = mocker.MagicMock()
        mock_llama.metadata = {
            "tokenizer.chat_template": "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}",
            "tokenizer.ggml.bos_token": "",
            "tokenizer.ggml.eos_token": "",
            "tokenizer.ggml.add_bos_token": True,
        }
        mock_llama_cls = mocker.MagicMock(return_value=mock_llama)
        mocker.patch.dict("sys.modules", {"llama_cpp": mocker.MagicMock(Llama=mock_llama_cls)})
        messages = [{"role": "user", "content": "Hello"}]
        result = render_with_model_template("/models/test.gguf", messages, add_generation_prompt=False)
        assert "user: Hello" in result

    def test_one_shot_fallback(self, mocker):
        from inference_server.templates.renderer import render_with_model_template
        mocker.patch.dict("sys.modules", {"llama_cpp": None}, clear=False)
        messages = [{"role": "user", "content": "Hello"}]
        result = render_with_model_template("/nonexistent.gguf", messages)
        # Falls back to ChatML
        assert "<|user|>" in result
