# ADR-004: Native Jinja Template Rendering for Tool Calling

**Status:** Accepted  
**Date:** 2026-08-20  
**Deciders:** Chris (Master Genor), Amy  

## Context

Models like Gemma 4 have specific chat templates that handle tool-call formatting. When the model is served via llama.cpp, it can use its native Jinja template to format tool calls correctly. But our previous approach used a custom Python renderer that didn't match the model's actual template, causing tool-call parsing failures.

Chris AI v21 achieved only 40% tool calling with the custom renderer, but jumped to 90% when using the model's native template.

## Decision

Use the model's **native Jinja template** (extracted from GGUF metadata) as the primary rendering path, with fallback to the custom renderer for backward compatibility.

### Implementation
1. **Template extraction**: Read `<|template|>` metadata from GGUF file using llama-cpp-python
2. **Jinja2 rendering**: Use `jinja2.Template.render()` to format messages with tool definitions
3. **Fallback chain**: native → custom renderer → raw messages
4. **AgenticAgent integration**: Pass native template to agent loop for consistent tool-call formatting

## Rationale

- **90% tool calling accuracy** with native templates vs 40% with custom renderer
- **Model-specific optimization**: Each model family (Gemma, Llama, Qwen) has its own template quirks
- **No maintenance burden**: Templates are maintained by model creators, not us
- **Backward compatible**: Custom renderer still works for models without native templates

## Consequences

- Must parse GGUF metadata to extract templates
- Need `jinja2` as a dependency (already installed)
- Agent loop must pass template through to renderer
- Future models may use different template formats (handled by fallback chain)
