#!/usr/bin/env python3
"""Master test runner — runs the full Finetune Studio + Inference Server suite.

Runs everything in one shot and prints a categorized summary so you can see
exactly what passed, failed, was skipped, and how long each category took.

Usage:
    python tests/run_all.py
    python tests/run_all.py --suite unit       # run only unit tests
    python tests/run_all.py --suite api        # run only API tests
    python tests/run_all.py --suite frontend   # run only frontend tests
    python tests/run_all.py --suite all        # run everything (default)
    python tests/run_all.py --coverage         # also produce a coverage report
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str], label: str) -> tuple[int, float]:
    """Run `cmd` in the repo root and return (returncode, wall_seconds)."""
    print(f"\n{'=' * 70}\n  {label}\n{'=' * 70}\n")
    start = time.time()
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    elapsed = time.time() - start
    print(f"\n[{label}] exit={result.returncode}  {elapsed:.1f}s")
    return result.returncode, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("all", "unit", "api", "frontend"),
                        default="all")
    parser.add_argument("--coverage", action="store_true",
                        help="produce a terminal coverage report (HTML at tests/coverage_html/)")
    args = parser.parse_args()

    pytest_cmd = ["python3", "-m", "pytest"]
    if args.coverage:
        pytest_cmd = ["python3", "-m", "pytest",
                      "--cov=finetune_studio",
                      "--cov=inference_server",
                      "--cov-report=html:tests/coverage_html",
                      "--cov-report=term"]

    targets = {
        "unit": ["tests/unit/"],
        "api": ["tests/api/"],
        "frontend": ["tests/frontend/"],
    }
    suites = list(targets.keys()) if args.suite == "all" else [args.suite]

    overall_rc = 0
    total_time = 0.0
    for suite in suites:
        cmd = pytest_cmd + targets[suite] + ["--tb=short", "-q"]
        rc, elapsed = run(cmd, suite.upper() + " TESTS")
        overall_rc = overall_rc or rc
        total_time += elapsed

    print(f"\n{'=' * 70}")
    print(f"  ALL DONE — {total_time:.1f}s total, exit={overall_rc}")
    if args.coverage:
        print(f"  Coverage HTML: tests/coverage_html/index.html")
    print(f"{'=' * 70}\n")
    return overall_rc


if __name__ == "__main__":
    sys.exit(main())