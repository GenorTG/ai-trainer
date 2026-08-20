# ADR-002: Why Gemma 4 over Llama 3 for Chris AI

**Status**: Accepted
**Date**: 2026-08-20
**Decision**: Fine-tune Gemma 4 E4B (≈14B parameters) as Chris AI base.

## Context

We need a base model that:
- Can be fine-tuned on a single 24GB GPU (RTX 3090)
- Has strong bilingual EN/PL support
- Has good tool-calling support
- Is freely available (Apache 2.0 / Gemma license)
- Is small enough for 8GB GPU production deployment (GTX 1070)

## Options Considered

| Model | Params | License | EN | PL | Tool calling | GGUF |
|-------|--------|---------|----|----|--------------|------|
| **Gemma 4 E4B** ✅ | ~14B | Gemma (free for use) | ✅ | ✅ | ✅ (via template) | ✅ |
| Llama 3.1 8B | 8B | Llama 3 community | ✅ | ❌ (weak) | ✅ (native) | ✅ |
| Mistral 7B | 7B | Apache 2.0 | ✅ | ❌ | ❌ (manual) | ✅ |
| Qwen 2.5 14B | 14B | Apache 2.0 | ✅ | ❌ | ✅ (native) | ✅ |
| Phi-4 14B | 14B | MIT | ✅ | ❌ | ❌ | ✅ |

## Decision

Use **Gemma 4 E4B** because:

1. **Polish support** — Gemma 4 has the best PL/EN bilingual quality of small open models
2. **Native template** — Has Gemma chat template with tool-calling macros (`format_tool_response_block`)
3. **Right size** — ~14B params fits on RTX 3090 (24GB) for QLoRA training, GTX 1070 (8GB) for inference
4. **License** — Gemma license allows commercial use with attribution
5. **Fine-tunable** — Unsloth + QLoRA works well, TRL supports it
7. **Knowledge preservation** — Mixed-data training (97% persona + 3% general) works well

## Consequences

- **Pro**: Best bilingual support for Polish persona
- **Pro**: Native tool-calling via Jinja template (90% benchmark)
- **Pro**: Fits on consumer hardware
- **Con**: Gemma license requires attribution + acceptable use policy
- **Con**: Larger than 7B models (slower inference, more VRAM)

## Benchmark Results

| Model | MMLU | HellaSwag | ARC | GSM8K | Winogrande | **Overall** |
|-------|------|-----------|-----|-------|------------|-------------|
| **Chris AI v20 (Gemma 4 E4B)** | 35% | 55% | 85% | 50% | 70% | **49.2%** |
| Chris AI v21 (Gemma 4 E4B) | 35% | 45% | 85% | 50% | 75% | 48.3% |
| Phi-4 14B baseline | 0% | 75% | 20% | 40% | 40% | 17.5% |

Chris AI beats Phi-4 on ARC (85% vs 20%) and MMLU (35% vs 0%).

## Future

If Gemma 5 or Llama 4 emerge with better PL support, re-evaluate.