# ADR-003: Knowledge Preservation via Data Mixing

**Status:** Accepted  
**Date:** 2026-08-20  
**Deciders:** Chris (Master Genor), Amy  

## Context

When fine-tuning a persona model, the model forgets general knowledge it learned during pre-training. Chris AI v19 had 96% personality but lost coding ability, math, and general reasoning. This is catastrophic for a model that needs to be both a personal assistant AND a useful coding/analysis tool.

## Decision

Use a **97% persona + 3% general knowledge** data mix ratio for all persona model training.

- 97% persona examples: Chris's personality, style, bilingual responses, coding patterns
- 3% general knowledge: Python coding, math reasoning, general Q&A, tool usage patterns

This is applied during data generation (gen_v21.py → gen_v22.py+), NOT during training.

## Rationale

- **97/3 preserves personality**: The 97% ratio ensures Chris's voice dominates
- **3% general knowledge prevents catastrophic forgetting**: Even small amounts of general data keep the model useful for coding, math, and analysis
- **Simple to implement**: Just concatenate persona JSONL with general JSONL before training
- **Proven in literature**: Knowledge distillation papers show 1-5% general data prevents catastrophic forgetting

## Consequences

- Model retains Chris's personality (96% quirky, 77% identity on v21)
- Model also handles coding, math, and general queries (v21: 48.3% overall benchmarks)
- Data generation scripts need a `--general-data` flag
- Future versions can tune the ratio (98/2, 95/5) based on benchmark results
