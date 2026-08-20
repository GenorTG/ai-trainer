# ADR-005: Why fan-dragon as primary development host

**Status**: Accepted
**Date**: 2026-08-20
**Decision**: All development happens on fan-dragon (RTX 3090 24GB). genorbox1 runs only the production inference server.

## Context

We have two machines in the Tailscale network:
- **fan-dragon** (genortg@100.125.137.96): RTX 3090 24GB, 32GB RAM, fish shell
- **genorbox1** (genorbox1): GTX 1070 8GB, 16GB RAM, bash

Originally development happened on genorbox1 (where the user works day-to-day).
But this created several problems:
- Training and benchmarking compete with the user's daily workflow
- Genorbox1 has only 8GB VRAM — fine for inference but slow for training
- Every test cycle was slow (genorbox1 is on a slower CPU)
- The user wanted a dedicated "AI lab" machine

## Options Considered

| Option | Pros | Cons |
|--------|------|------|
| **fan-dragon primary** ✅ | Dedicated to AI work, 24GB VRAM, fast | User has to SSH/tail logs |
| genorbox1 primary | User is always there | Slow training, can't run heavy tests |
| Both equal | Flexibility | State sync issues, confusion |

## Decision

Use **fan-dragon as primary dev host** because:

1. **RTX 3090 24GB** — Can train QLoRA 14B models in reasonable time
2. **Isolated** — Doesn't compete with daily workflow on genorbox1
3. **Test environment** — Fast CI runs (~30s for 573 tests)
4. **Models stay there** — GGUF outputs in `~/ai-trainer/models/`
5. **Production mirror** — Deploy via `tools/deploy.sh` rsync

## Consequences

- **Pro**: Fast training/benchmarking
- **Pro**: All AI work in one place
- **Pro**: Test runs are fast (30s vs 60s+ on genorbox1)
- **Con**: User has to SSH to fan-dragon to monitor
- **Con**: No local GUI access (terminal-only)
- **Con**: ComfyUI runs on fan-dragon — can pause GPU usage (respect rule)

## ComfyUI Rule

> **NEVER kill ComfyUI on fan-dragon.** If GPU busy with generations, pause benchmark/test work and notify user. Wait for user to say when to resume.

Current ComfyUI usage: ~10-17GB GPU. Leaves 6-14GB for benchmarks.

## Workflow

```
genorbox1 (user workspace)           fan-dragon (AI lab)
   │                                       │
   ├── daily tasks                        ├── training
   ├── SSH to fan-dragon ────────────────►├── benchmarking
   ├── view logs (session_status)         ├── testing
   └── git push deploys                   ├── model exports
                                           └── git commits
```

## Deployment

`tools/deploy.sh` syncs server/inference_server/, trainer/src/finetune_studio/,
and tests/ from fan-dragon to genorbox1. Production llama-server runs on genorbox1
with the v20 GGUF. Models are NOT synced (too large, downloaded separately).

## Mirror for testing

Tests run on both machines to verify cross-machine compatibility:
- fan-dragon: `cd ~/ai-trainer && python tests/run_all.py`
- genorbox1: `cd /home/genorbox1/llama-server && python3 -m pytest tests/`

Both should produce identical results (573 passing, 2 skipped).