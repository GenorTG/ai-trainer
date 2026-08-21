"""Tests for inference_server/templates/manager.py."""

from __future__ import annotations

import pytest


@pytest.mark.unit
class TestChatTemplate:
    def test_defaults(self):
        from inference_server.templates.manager import ChatTemplate
        t = ChatTemplate()
        assert t.name == "unknown"
        assert t.template == ""
        assert t.supports_tools is False
        assert t.add_bos is True

    def test_custom_values(self):
        from inference_server.templates.manager import ChatTemplate
        t = ChatTemplate(
            name="my_model",
            template="{% for m in messages %}{{ m.role }}: {{ m.content }}{% endfor %}",
            tool_format="qwen",
            supports_tools=True,
            bos_token="<bos>",
            eos_token="<eos>",
            add_bos=False,
        )
        assert t.name == "my_model"
        assert t.tool_format == "qwen"
        assert t.supports_tools is True
        assert t.add_bos is False


@pytest.mark.unit
class TestTemplateManager:
    def test_init(self):
        from inference_server.templates.manager import TemplateManager
        mgr = TemplateManager()
        assert mgr.templates == {}

    def test_get_template_unknown(self):
        from inference_server.templates.manager import ChatTemplate, TemplateManager
        mgr = TemplateManager()
        t = mgr.get_template("nonexistent")
        assert isinstance(t, ChatTemplate)
        assert t.name == "unknown"

    def test_register_from_gguf(self, mocker):
        from inference_server.templates.manager import TemplateManager
        mgr = TemplateManager()
        mock_info = {
            "chat_template": "{% for m in messages %}<|{{m.role}}|>{{ m.content }}<|end|>{% endfor %}",
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "add_bos": True,
            "supports_tools": False,
        }
        mocker.patch(
            "inference_server.templates.manager.extract_template_from_gguf",
            return_value=mock_info,
        )
        tmpl = mgr.register_from_gguf("/models/test.gguf", model_name="test_model")
        assert tmpl.name == "test_model"
        assert tmpl.bos_token == "<bos>"
        assert tmpl.supports_tools is False

    def test_register_from_gguf_no_name(self, mocker):
        from inference_server.templates.manager import TemplateManager
        mgr = TemplateManager()
        mock_info = {
            "chat_template": "template",
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "add_bos": True,
            "supports_tools": False,
        }
        mocker.patch(
            "inference_server.templates.manager.extract_template_from_gguf",
            return_value=mock_info,
        )
        tmpl = mgr.register_from_gguf("/models/my_model.gguf")
        assert tmpl.name == "my_model"

    def test_render(self, mocker):
        from inference_server.templates.manager import TemplateManager
        mgr = TemplateManager()
        mock_info = {
            "chat_template": "{% for m in messages %}{{m.role}}: {{m.content}}{% endfor %}",
            "bos_token": "",
            "eos_token": "",
            "add_bos": True,
            "supports_tools": False,
        }
        mocker.patch(
            "inference_server.templates.manager.extract_template_from_gguf",
            return_value=mock_info,
        )
        mgr.register_from_gguf("/fake.gguf", model_name="m")
        result = mgr.render("m", [{"role": "user", "content": "hi"}])
        assert "user: hi" in result

    def test_render_unknown_model(self, mocker):
        from inference_server.templates.manager import TemplateManager
        mgr = TemplateManager()
        result = mgr.render("nonexistent", [{"role": "user", "content": "hi"}])
        # Unknown model falls back to ChatML
        assert "<|user|>" in result


@pytest.mark.unit
class TestDetectFormat:
    @pytest.mark.parametrize("template_str,expected", [
        # detection returns the FIRST matching format in dict order
        ("{{ m.role }} <|tool_call|>tool", "qwen"),
        ("{{ m.role }} <im_end>", "chatml"),
        ("<|assistant|>respond", "hermes"),
        ("<|end|>", "phi3"),
        ("<|start_header_id|>s<|end_header_id|>", "llama3"),
        ("[INST] Hello [/INST]", "mistral"),
        ("<start_of_turn>user<end_of_turn>", "gemma"),
        # gemma4 markers use single-pipe <|turn> / <|tool_response>
        ("<|turn>user<|tool_response>ok", "gemma4"),
        ("", "generic"),
        ("just random text", "generic"),
    ])
    def test_format_detection(self, template_str, expected):
        from inference_server.templates.manager import _detect_format
        assert _detect_format(template_str) == expected


@pytest.mark.unit
class TestToolPromptBuilders:
    @pytest.fixture
    def tools(self):
        return [
            {"type": "function", "function": {"name": "search", "description": "Search", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}}},
            {"type": "function", "function": {"name": "calc", "description": "Calculate", "parameters": {"type": "object", "properties": {"expr": {"type": "string"}}, "required": ["expr"]}}},
        ]

    def test_generic_prompt(self, tools):
        from inference_server.templates.manager import _build_generic_tool_prompt
        result = _build_generic_tool_prompt(tools)
        assert "search" in result
        assert "calc" in result
        assert "Available tools" in result

    def test_qwen_prompt(self, tools):
        from inference_server.templates.manager import _build_qwen_tool_prompt
        result = _build_qwen_tool_prompt(tools)
        assert "search" in result

    def test_gemma4_prompt(self, tools):
        from inference_server.templates.manager import _build_gemma4_tool_prompt
        result = _build_gemma4_tool_prompt(tools)
        assert "search" in result
        assert "params:" in result

    def test_mistral_prompt(self, tools):
        from inference_server.templates.manager import _build_mistral_tool_prompt
        result = _build_mistral_tool_prompt(tools)
        assert "[AVAILABLE_TOOLS]" in result
        assert "[/AVAILABLE_TOOLS]" in result

    def test_build_tool_system_prompt_generic(self):
        from inference_server.templates.manager import TemplateManager
        mgr = TemplateManager()
        result = mgr.build_tool_system_prompt()
        assert "Available tools" in result

    def test_build_tool_system_prompt_mistral(self):
        from inference_server.templates.manager import ChatTemplate, TemplateManager
        mgr = TemplateManager()
        mgr.templates["m"] = ChatTemplate(tool_format="mistral")
        result = mgr.build_tool_system_prompt("m")
        assert "[AVAILABLE_TOOLS]" in result

    def test_build_tool_system_prompt_gemma4(self):
        from inference_server.templates.manager import ChatTemplate, TemplateManager
        mgr = TemplateManager()
        mgr.templates["m"] = ChatTemplate(tool_format="gemma4")
        result = mgr.build_tool_system_prompt("m")
        assert "params:" in result
