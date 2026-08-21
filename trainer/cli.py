"""Finetune Studio CLI — manage models, training, and testing from the terminal."""
import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="finetune-studio",
        description="Finetune Studio — model training CLI",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # ── models ──
    p_models = sub.add_parser("models", help="List discovered models")
    p_models.add_argument("--dirs", nargs="*", help="Extra directories to scan")
    p_models.add_argument("--json", action="store_true", help="Output as JSON")

    # ── train ──
    p_train = sub.add_parser("train", help="Start training")
    p_train.add_argument("model", help="Path to base model (safetensors dir or GGUF)")
    p_train.add_argument("data", help="Path to training data (JSONL)")
    p_train.add_argument("-o", "--output", default="output", help="Output directory")
    p_train.add_argument("--lr", type=float, default=8e-5, help="Learning rate")
    p_train.add_argument("--epochs", type=int, default=4, help="Number of epochs")
    p_train.add_argument("--batch", type=int, default=2, help="Batch size")
    p_train.add_argument("--lora-rank", type=int, default=64, help="LoRA rank")
    p_train.add_argument("--max-seq", type=int, default=2048, help="Max sequence length")
    p_train.add_argument("--system-prompt", default="", help="System prompt for all examples")
    p_train.add_argument("--no-unsloth", action="store_true", help="Use standard transformers instead of Unsloth")

    # ── test ──
    p_test = sub.add_parser("test", help="Test a model interactively")
    p_test.add_argument("model", help="Path to model (safetensors dir or GGUF)")
    p_test.add_argument("--max-tokens", type=int, default=512, help="Max generation tokens")
    p_test.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")

    # ── suite ──
    p_suite = sub.add_parser("suite", help="Run a test suite")
    p_suite.add_argument("model", help="Path to model")
    p_suite.add_argument("suite", help="Path to test suite JSON")
    p_suite.add_argument("--max-tokens", type=int, default=512)
    p_suite.add_argument("--json", action="store_true", help="Output as JSON")

    # ── validate ──
    p_val = sub.add_parser("validate", help="Validate training data files")
    p_val.add_argument("files", nargs="+", help="Files to validate")

    # ── convert ──
    p_conv = sub.add_parser("convert", help="Convert data formats")
    p_conv.add_argument("source", help="Source file")
    p_conv.add_argument("target_format", choices=["jsonl", "json", "csv"], help="Target format")
    p_conv.add_argument("-o", "--output", help="Output path (default: source with new extension)")
    p_conv.add_argument("--system-prompt", default="", help="System prompt (for CSV→JSONL)")

    # ── webui ──
    p_web = sub.add_parser("webui", help="Start the WebUI server")
    p_web.add_argument("--host", default="0.0.0.0", help="Bind host")
    p_web.add_argument("--port", type=int, default=7860, help="Bind port")
    p_web.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "models":
        cmd_models(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "test":
        cmd_test(args)
    elif args.command == "suite":
        cmd_suite(args)
    elif args.command == "validate":
        cmd_validate(args)
    elif args.command == "convert":
        cmd_convert(args)
    elif args.command == "webui":
        cmd_webui(args)


def cmd_models(args):
    from finetune_studio.config import settings
    from finetune_studio.models.registry import scan_models
    dirs = settings.model_dirs + (args.dirs or [])
    models = scan_models(dirs)
    if args.json:
        print(json.dumps([{
            "name": m.name, "path": m.path, "format": m.format,
            "size_gb": m.size_gb, "architecture": m.architecture,
        } for m in models], indent=2))
    else:
        if not models:
            print("No models found.")
            return
        print(f"{'Name':<50} {'Format':<12} {'Size':<10} {'Arch'}")
        print("-" * 90)
        for m in models:
            print(f"{m.name:<50} {m.format:<12} {m.size_gb:<10} {m.architecture}")


def cmd_train(args):
    import time

    from finetune_studio.training.data import load_jsonl
    from finetune_studio.training.engine import TrainingConfig, TrainingEngine

    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        sys.exit(1)
    if not os.path.exists(args.data):
        print(f"Error: Data not found: {args.data}")
        sys.exit(1)

    data = load_jsonl(args.data)
    print(f"Loaded {len(data)} examples from {args.data}")

    config = TrainingConfig(
        model_path=args.model,
        output_dir=args.output,
        lora_rank=args.lora_rank,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch,
        max_seq_length=args.max_seq,
        unsloth=not args.no_unsloth,
    )

    engine = TrainingEngine()

    def on_progress(state):
        if state.status == "training":
            pct = (state.current_step / max(state.total_steps, 1)) * 100
            bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
            sys.stdout.write(f"\r\\r[{bar}] {pct:.0f}% | Step {state.current_step}/{state.total_steps} | Loss: {state.loss} | ETA: {state.eta}s")
            sys.stdout.flush()
        elif state.status == "done":
            print(f"\\n\\nTraining complete! Output: {args.output}")
        elif state.status == "error":
            print(f"\\n\\nError: {state.error}")
        elif state.status in ("loading", "saving"):
            print(f"  {state.message}")

    engine.on_update(on_progress)
    engine.start(config, data, args.system_prompt)

    # Wait for completion
    while engine.state.status in ("loading", "training", "saving"):
        time.sleep(1)

    sys.exit(0 if engine.state.status == "done" else 1)


def cmd_test(args):
    from finetune_studio.testing.inference import InferenceEngine

    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        sys.exit(1)

    engine = InferenceEngine()
    print(f"Loading {args.model}...")
    engine.load(args.model)
    fmt = "GGUF" if engine.is_gguf else "safetensors"
    print(f"Model loaded ({fmt}). Type 'quit' to exit.\\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        response = engine.generate(
            [{"role": "user", "content": user_input}],
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print(f"AI: {response}\\n")

    engine.unload()
    print("Model unloaded.")


def cmd_suite(args):
    from finetune_studio.testing.inference import InferenceEngine
    from finetune_studio.testing.suite import load_test_suite, run_suite, score_results

    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        sys.exit(1)
    if not os.path.exists(args.suite):
        print(f"Error: Suite not found: {args.suite}")
        sys.exit(1)

    engine = InferenceEngine()
    print(f"Loading {args.model}...")
    engine.load(args.model)

    cases = load_test_suite(args.suite)
    print(f"Running {len(cases)} test cases...\\n")

    results = run_suite(engine, cases, max_tokens=args.max_tokens)
    scores = score_results(results)

    if args.json:
        print(json.dumps({
            "results": [{
                "name": r.test_name, "passed": r.passed, "response": r.response,
                "keyword_hits": r.keyword_hits, "keyword_misses": r.keyword_misses,
                "forbidden_hits": r.forbidden_hits, "time_ms": r.time_ms, "error": r.error,
            } for r in results],
            "scores": scores,
        }, indent=2))
    else:
        for r in results:
            status = "PASS" if r.passed else "FAIL"
            icon = "✅" if r.passed else "❌"
            print(f"{icon} {status} {r.test_name} ({r.time_ms}ms)")
            if r.error:
                print(f"   Error: {r.error}")
            elif r.keyword_misses:
                print(f"   Missing: {', '.join(r.keyword_misses)}")
            elif r.forbidden_hits:
                print(f"   Forbidden: {', '.join(r.forbidden_hits)}")
            print(f"   {r.response[:120]}...\\n" if len(r.response) > 120 else f"   {r.response}\\n")

        print(f"\\n{'='*50}")
        print(f"Pass rate: {scores['pass_rate']}% ({scores['passed']}/{scores['total']})")
        print(f"Avg time: {scores['avg_time_ms']}ms")

    engine.unload()


def cmd_validate(args):
    from finetune_studio.data.validator import validate_file
    for f in args.files:
        report = validate_file(f)
        icon = "✅" if report["valid"] else "❌"
        print(f"{icon} {report['name']}: {report['stats']}")
        if report["errors"]:
            for e in report["errors"]:
                print(f"   ERROR: {e}")
        if report["warnings"]:
            for w in report["warnings"]:
                print(f"   WARN: {w}")


def cmd_convert(args):
    from pathlib import Path

    from finetune_studio.data.converter import csv_to_jsonl, json_to_jsonl, jsonl_to_json

    src = Path(args.source)
    if not src.exists():
        print(f"Error: File not found: {src}")
        sys.exit(1)

    target = args.output or str(src.with_suffix(f".{args.target_format}"))

    if src.suffix == ".jsonl" and args.target_format == "json":
        jsonl_to_json(str(src), target)
    elif src.suffix == ".json" and args.target_format == "jsonl":
        json_to_jsonl(str(src), target)
    elif src.suffix == ".csv" and args.target_format == "jsonl":
        csv_to_jsonl(str(src), target, system_prompt=args.system_prompt)
    else:
        print(f"Error: Cannot convert {src.suffix} → .{args.target_format}")
        sys.exit(1)

    print(f"Converted: {src} → {target}")


def cmd_webui(args):
    import uvicorn
    uvicorn.run(
        "finetune_studio.webui.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
